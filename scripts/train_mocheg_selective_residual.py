"""Train a safe conflict-aware residual over the frozen cached verifier."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphcure.evidence_set import EvidenceSetHead
from graphcure.selective_residual import SelectiveResidualSetVerifier
from scripts.train_mocheg_cached_verifier import (
    CachedEvidenceDataset, collate, expected_calibration_error,
    validate_cache_pair,
)


@torch.inference_mode()
def evaluate(model, dataset, batch_size, device, force_anchor=False):
    model.eval(); ys=[]; preds=[]; anchors=[]; probs=[]; gates=[]; rows=[]
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    for batch in tqdm(loader, desc="evaluate", leave=False):
        ids=batch.pop("id"); y=batch.pop("labels")
        batch.pop("relevance"); batch.pop("relevance_weights")
        out=model(**{key:value.to(device) for key,value in batch.items()})
        logits=out["anchor_logits"] if force_anchor else out["verdict_logits"]
        prob=torch.softmax(logits,-1).cpu(); pred=prob.argmax(-1)
        anchor=torch.softmax(out["anchor_logits"],-1).cpu().argmax(-1)
        for i,sample_id in enumerate(ids):
            rows.append({"id":sample_id,"gold":int(y[i]),
                         "prediction":int(pred[i]),
                         "anchor_prediction":int(anchor[i]),
                         "probabilities":prob[i].tolist(),
                         "residual_gate":float(out["residual_gate"][i])})
        ys.extend(y.tolist()); preds.extend(pred.tolist()); anchors.extend(anchor.tolist())
        probs.extend(prob.tolist()); gates.extend(out["residual_gate"].cpu().tolist())
    y=np.asarray(ys); pred=np.asarray(preds); anchor=np.asarray(anchors)
    probability=np.asarray(probs)
    return {
        "samples":len(y), "accuracy":float(accuracy_score(y,pred)),
        "macro_f1":float(f1_score(y,pred,average="macro")),
        "confusion_matrix":confusion_matrix(y,pred).tolist(),
        "help_rate":float(np.mean((anchor!=y)&(pred==y))),
        "harm_rate":float(np.mean((anchor==y)&(pred!=y))),
        "gate_mean":float(np.mean(gates)), "gate_std":float(np.std(gates)),
        "ece_10":expected_calibration_error(probability,y),
    }, rows


def build_model(cache, checkpoint, args, device):
    anchor=EvidenceSetHead(
        encoder_dim=int(cache.metadata["embedding_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]), retrieval_dim=6,
        dropout=float(checkpoint["dropout"]),
    )
    anchor.load_state_dict(checkpoint["head"])
    return SelectiveResidualSetVerifier(
        anchor=anchor, encoder_dim=int(cache.metadata["embedding_dim"]),
        hidden_dim=args.hidden_dim, layers=args.layers, heads=args.heads,
        dropout=args.dropout, residual_scale=args.residual_scale,
        gate_bias=args.gate_bias,
    ).to(device)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache-root",type=Path,default=Path("data/processed/mocheg_reasoning_cache"))
    p.add_argument("--anchor-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("outputs/mocheg_selective_residual_seed42_v14"))
    p.add_argument("--hidden-dim",type=int,default=192); p.add_argument("--layers",type=int,default=2); p.add_argument("--heads",type=int,default=4); p.add_argument("--dropout",type=float,default=.1)
    p.add_argument("--residual-scale",type=float,default=2.0); p.add_argument("--gate-bias",type=float,default=-2.2)
    p.add_argument("--epochs",type=int,default=60); p.add_argument("--patience",type=int,default=10); p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--learning-rate",type=float,default=2e-4); p.add_argument("--weight-decay",type=float,default=.01)
    p.add_argument("--anchor-correct-kl",type=float,default=.5); p.add_argument("--global-kl",type=float,default=.05); p.add_argument("--gate-cost",type=float,default=.01)
    p.add_argument("--anchor-error-weight",type=float,default=1.5); p.add_argument("--minimum-delta",type=float,default=.003)
    p.add_argument("--num-workers",type=int,default=2); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device(a.device if torch.cuda.is_available() else "cpu"); a.output.mkdir(parents=True,exist_ok=True)
    train=CachedEvidenceDataset(a.cache_root/"train.pt"); val=CachedEvidenceDataset(a.cache_root/"val.pt"); validate_cache_pair(train,val)
    checkpoint=torch.load(a.anchor_checkpoint,map_location="cpu",weights_only=False)
    if checkpoint["cache_metadata"]["encoder"] != train.metadata["encoder"]:
        p.error("anchor and cache encoder mismatch")
    model=build_model(train,checkpoint,a,device)
    anchor_metrics,_=evaluate(model,val,a.batch_size,device,force_anchor=True)
    history=[{"epoch":0,**{f"val_{k}":v for k,v in anchor_metrics.items() if k!="confusion_matrix"}}]
    print(json.dumps(history[0]),flush=True)
    counts=torch.bincount(train.data["labels"],minlength=3).float()
    class_weights=(counts.sum()/(3*counts.clamp_min(1))).to(device)
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,collate_fn=collate,pin_memory=device.type=="cuda")
    parameters=[x for x in model.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=a.learning_rate,weight_decay=a.weight_decay)
    best_candidate=-1.; best_epoch=-1; stale=0
    for epoch in range(1,a.epochs+1):
        model.train(); running={key:0. for key in ("loss","ce","correct_kl","global_kl","gate")}
        for batch in tqdm(loader,desc=f"epoch {epoch}/{a.epochs}"):
            batch.pop("id"); labels=batch.pop("labels").to(device)
            batch.pop("relevance"); batch.pop("relevance_weights")
            optimizer.zero_grad(set_to_none=True)
            out=model(**{key:value.to(device) for key,value in batch.items()})
            anchor_prob=torch.softmax(out["anchor_logits"].detach(),-1)
            anchor_correct=anchor_prob.argmax(-1).eq(labels)
            ce_terms=F.cross_entropy(out["verdict_logits"],labels,weight=class_weights,reduction="none")
            sample_weights=torch.where(anchor_correct,torch.ones_like(ce_terms),torch.full_like(ce_terms,a.anchor_error_weight))
            ce=(ce_terms*sample_weights).mean()
            kl_terms=F.kl_div(F.log_softmax(out["verdict_logits"],-1),anchor_prob,reduction="none").sum(-1)
            correct_kl=kl_terms[anchor_correct].mean() if anchor_correct.any() else kl_terms.new_zeros(())
            global_kl=kl_terms.mean(); gate=out["residual_gate"].mean()
            loss=ce+a.anchor_correct_kl*correct_kl+a.global_kl*global_kl+a.gate_cost*gate
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.0); optimizer.step()
            for key,value in (("loss",loss),("ce",ce),("correct_kl",correct_kl),("global_kl",global_kl),("gate",gate)):
                running[key]+=float(value.detach())
        metrics,_=evaluate(model,val,a.batch_size,device)
        row={"epoch":epoch,**{f"train_{k}":v/len(loader) for k,v in running.items()},**{f"val_{k}":v for k,v in metrics.items() if k!="confusion_matrix"}}
        history.append(row); print(json.dumps(row),flush=True)
        if metrics["macro_f1"]>best_candidate:
            best_candidate=metrics["macro_f1"]; best_epoch=epoch; stale=0
            torch.save({"model":model.state_dict(),"settings":vars(a),"anchor_checkpoint":str(a.anchor_checkpoint),"anchor_macro_f1":anchor_metrics["macro_f1"],"best_candidate_macro_f1":best_candidate,"epoch":epoch,"cache_metadata":train.metadata},a.output/"best_candidate.pt")
        else:
            stale+=1
            if stale>=a.patience: break
    accepted=best_candidate >= anchor_metrics["macro_f1"] + a.minimum_delta
    if accepted:
        chosen=torch.load(a.output/"best_candidate.pt",map_location="cpu",weights_only=False)
        model.load_state_dict(chosen["model"]); final,rows=evaluate(model,val,a.batch_size,device)
        shutil.copy2(a.output/"best_candidate.pt",a.output/"best.pt"); mode="selective_residual"
    else:
        final,rows=evaluate(model,val,a.batch_size,device,force_anchor=True)
        final["gate_mean"] = 0.0; final["gate_std"] = 0.0
        mode="anchor_fallback"
    summary={"mode":mode,"accepted":accepted,"minimum_delta":a.minimum_delta,"anchor":anchor_metrics,"best_candidate_macro_f1":best_candidate,"best_candidate_epoch":best_epoch,"best_candidate_delta":best_candidate-anchor_metrics["macro_f1"],"final":final,"history":history,"test_split_used":False}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (a.output/"val_predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
