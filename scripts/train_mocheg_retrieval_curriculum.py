"""Grounded/natural retrieval curriculum with a frozen-anchor safety gate."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from graphcure.evidence_set import EvidenceSetHead, evidence_set_loss
from scripts.train_mocheg_cached_verifier import (
    CachedEvidenceDataset, collate, evaluate, validate_cache_pair,
)


class PairedCache(Dataset):
    def __init__(self, natural: CachedEvidenceDataset,
                 grounded: CachedEvidenceDataset) -> None:
        if natural.data["ids"] != grounded.data["ids"]:
            raise ValueError("natural and grounded train caches are not aligned")
        if not grounded.metadata.get("train_gold_injection"):
            raise ValueError("grounded cache does not declare train-only injection")
        self.natural, self.grounded = natural, grounded

    def __len__(self) -> int:
        return len(self.natural)

    def __getitem__(self, index: int) -> dict:
        return {"natural": self.natural[index], "grounded": self.grounded[index]}


def paired_collate(rows: list[dict]) -> dict:
    return {
        view: collate([row[view] for row in rows])
        for view in ("natural", "grounded")
    }


def build_head(cache: CachedEvidenceDataset, checkpoint: dict,
               device: torch.device) -> EvidenceSetHead:
    head = EvidenceSetHead(
        encoder_dim=int(cache.metadata["embedding_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]), retrieval_dim=6,
        dropout=float(checkpoint["dropout"]),
    )
    head.load_state_dict(checkpoint["head"])
    return head.to(device)


def forward_view(head, batch, device):
    batch = dict(batch); batch.pop("id")
    labels = batch.pop("labels").to(device)
    relevance = batch.pop("relevance").to(device)
    relevance_weights = batch.pop("relevance_weights").to(device)
    evidence_mask = batch["evidence_mask"].to(device)
    output = head(**{key:value.to(device) for key,value in batch.items()})
    return output, labels, relevance, relevance_weights, evidence_mask


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--natural-cache-root",type=Path,default=Path("data/processed/mocheg_reasoning_cache"))
    p.add_argument("--grounded-cache-root",type=Path,required=True)
    p.add_argument("--anchor-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("outputs/mocheg_retrieval_curriculum_seed42_v15"))
    p.add_argument("--epochs",type=int,default=40); p.add_argument("--patience",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=256); p.add_argument("--learning-rate",type=float,default=5e-5); p.add_argument("--weight-decay",type=float,default=.01)
    p.add_argument("--natural-verdict-weight",type=float,default=.35); p.add_argument("--anchor-correct-kl",type=float,default=.5)
    p.add_argument("--relevance-weight",type=float,default=.25); p.add_argument("--stance-weight",type=float,default=.15); p.add_argument("--sufficiency-weight",type=float,default=.15)
    p.add_argument("--minimum-delta",type=float,default=.003); p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device(a.device if torch.cuda.is_available() else "cpu"); a.output.mkdir(parents=True,exist_ok=True)
    natural=CachedEvidenceDataset(a.natural_cache_root/"train.pt"); grounded=CachedEvidenceDataset(a.grounded_cache_root/"train.pt"); val=CachedEvidenceDataset(a.natural_cache_root/"val.pt")
    validate_cache_pair(natural,val); validate_cache_pair(grounded,val); paired=PairedCache(natural,grounded)
    checkpoint=torch.load(a.anchor_checkpoint,map_location="cpu",weights_only=False)
    if checkpoint["cache_metadata"]["encoder"] != natural.metadata["encoder"]:
        p.error("anchor and cache encoder mismatch")
    teacher=build_head(natural,checkpoint,device).eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    student=build_head(natural,checkpoint,device)
    anchor_metrics,_=evaluate(teacher,val,a.batch_size,device)
    history=[{"epoch":0,**{f"val_{key}":value for key,value in anchor_metrics.items() if key!="confusion_matrix"}}]
    print(json.dumps(history[0]),flush=True)
    counts=torch.bincount(natural.data["labels"],minlength=3).float(); class_weights=(counts.sum()/(3*counts.clamp_min(1))).to(device)
    loader=DataLoader(paired,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,collate_fn=paired_collate,pin_memory=device.type=="cuda")
    optimizer=torch.optim.AdamW(student.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
    best=-1.; best_epoch=-1; stale=0
    for epoch in range(1,a.epochs+1):
        student.train(); sums={key:0. for key in ("loss","grounded","natural","anchor_kl")}
        for batch in tqdm(loader,desc=f"epoch {epoch}/{a.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            natural_out,labels,_,_,_=forward_view(student,batch["natural"],device)
            grounded_out,grounded_labels,relevance,relevance_weights,mask=forward_view(student,batch["grounded"],device)
            if not torch.equal(labels,grounded_labels): raise RuntimeError("paired labels differ")
            grounded_loss,_=evidence_set_loss(grounded_out,labels,relevance,mask,relevance_weights=relevance_weights,class_weights=class_weights,relevance_weight=a.relevance_weight,stance_weight=a.stance_weight,sufficiency_weight=a.sufficiency_weight)
            natural_loss=F.cross_entropy(natural_out["verdict_logits"].float(),labels,weight=class_weights)
            with torch.no_grad():
                teacher_out,_,_,_,_=forward_view(teacher,batch["natural"],device)
                teacher_prob=torch.softmax(teacher_out["verdict_logits"].float(),-1)
                teacher_correct=teacher_prob.argmax(-1).eq(labels)
            kl=F.kl_div(F.log_softmax(natural_out["verdict_logits"].float(),-1),teacher_prob,reduction="none").sum(-1)
            anchor_kl=kl[teacher_correct].mean() if teacher_correct.any() else kl.new_zeros(())
            loss=grounded_loss+a.natural_verdict_weight*natural_loss+a.anchor_correct_kl*anchor_kl
            loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),1.0); optimizer.step()
            for key,value in (("loss",loss),("grounded",grounded_loss),("natural",natural_loss),("anchor_kl",anchor_kl)): sums[key]+=float(value.detach())
        metrics,_=evaluate(student,val,a.batch_size,device)
        row={"epoch":epoch,**{f"train_{key}":value/len(loader) for key,value in sums.items()},**{f"val_{key}":value for key,value in metrics.items() if key!="confusion_matrix"}}
        history.append(row); print(json.dumps(row),flush=True)
        if metrics["macro_f1"]>best:
            best=metrics["macro_f1"]; best_epoch=epoch; stale=0
            torch.save({"head":student.state_dict(),"hidden_dim":checkpoint["hidden_dim"],"dropout":checkpoint["dropout"],"seed":a.seed,"best_val_macro_f1":best,"cache_metadata":natural.metadata,"grounded_cache_metadata":grounded.metadata,"epoch":epoch},a.output/"best_candidate.pt")
        else:
            stale+=1
            if stale>=a.patience: break
    accepted=best >= anchor_metrics["macro_f1"]+a.minimum_delta
    if accepted:
        chosen=torch.load(a.output/"best_candidate.pt",map_location="cpu",weights_only=False); student.load_state_dict(chosen["head"])
        final,rows=evaluate(student,val,a.batch_size,device); shutil.copy2(a.output/"best_candidate.pt",a.output/"best.pt"); mode="retrieval_curriculum"
    else:
        final,rows=evaluate(teacher,val,a.batch_size,device); mode="anchor_fallback"
    summary={"mode":mode,"accepted":accepted,"minimum_delta":a.minimum_delta,"anchor":anchor_metrics,"best_candidate_macro_f1":best,"best_candidate_epoch":best_epoch,"best_candidate_delta":best-anchor_metrics["macro_f1"],"natural_train_gold_coverage":float(natural.data["relevance"].bool().any(1).float().mean()),"grounded_train_gold_coverage":float(grounded.data["relevance"].bool().any(1).float().mean()),"injected_claims":grounded.metadata.get("injected_claims"),"final":final,"history":history,"test_split_used":False}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); (a.output/"val_predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
