"""Fine-tune an NLI-initialized multi-evidence verifier for MOCHEG."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_cached_verifier import expected_calibration_error


def read_documents(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("evidence_id", "").strip()
            value = row.get("Evidence", "").replace("<p>", " ").replace(
                "</p>", " ").strip()
            if key and value: result.setdefault(key, value)
    return result


def select_evidence_candidates(sample_id: str, retrieved: list[str],
                               gold: set[str], top_k: int,
                               inject_gold: bool) -> list[str]:
    retrieved = list(dict.fromkeys(retrieved))
    if not inject_gold or not gold:
        return retrieved[:top_k]
    positive = min(gold, key=lambda value: hashlib.sha256(
        f"{sample_id}\0{value}".encode()).digest())
    selected = [positive] + [value for value in retrieved if value != positive]
    return sorted(selected[:top_k], key=lambda value: hashlib.sha256(
        f"nli-set\0{sample_id}\0{value}".encode()).digest())


class EvidenceSets(Dataset):
    def __init__(self, manifest: Path, retrieval: Path, corpus: Path,
                 top_k: int, inject_gold: bool, limit: int = 0):
        claims = read_jsonl(manifest)
        if limit: claims = claims[:limit]
        retrieved = {row["id"]: row for row in read_jsonl(retrieval)}
        documents = read_documents(corpus); available = set(documents)
        self.rows = []
        for claim in claims:
            row = retrieved.get(claim["id"])
            if row is None: raise ValueError(f"retrieval missing {claim['id']}")
            ids = [str(value) for value in row.get("retrieved_evidence_ids", [])
                   if str(value) in available]
            gold = {str(value) for value in claim.get("text_evidence_ids", [])
                    if str(value) in available}
            selected = select_evidence_candidates(
                claim["id"], ids, gold, top_k, inject_gold)
            if not selected: raise ValueError(f"no evidence for {claim['id']}")
            self.rows.append({
                "id": claim["id"], "claim": claim.get("claim", ""),
                "label": int(claim["label"]),
                "evidence": [documents[value] for value in selected],
                "relevance": [value in gold for value in selected],
            })
        self.gold_coverage = float(np.mean([
            any(row["relevance"]) for row in self.rows]))

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def make_collate(tokenizer, top_k: int, max_length: int):
    def collate(rows):
        premises, hypotheses, mask, relevance = [], [], [], []
        for row in rows:
            count = len(row["evidence"])
            evidence = row["evidence"] + [row["evidence"][-1]] * (top_k-count)
            premises.extend(evidence); hypotheses.extend([row["claim"]] * top_k)
            mask.append([True] * count + [False] * (top_k-count))
            relevance.append(row["relevance"] + [False] * (top_k-count))
        encoded = tokenizer(
            premises, hypotheses, padding=True, truncation="only_first",
            max_length=max_length, return_tensors="pt")
        return {"encoded": encoded,
                "labels": torch.tensor([row["label"] for row in rows]),
                "mask": torch.tensor(mask, dtype=torch.bool),
                "relevance": torch.tensor(relevance, dtype=torch.bool)}
    return collate


def aggregate_nli_logits(pair_logits: torch.Tensor, mask: torch.Tensor,
                         temperature: float = .5) -> torch.Tensor:
    """Smooth max over evidence for each support/refute/NEI state."""
    scaled = pair_logits.float() / temperature
    scaled = scaled.masked_fill(~mask.unsqueeze(-1), -1e4)
    count = mask.sum(1, keepdim=True).clamp_min(1).float()
    return temperature * (torch.logsumexp(scaled, 1) - torch.log(count))


def reorder_nli_head(model) -> list[int]:
    """Map entailment/contradiction/neutral rows to supported/refuted/NEI."""
    labels = {int(key): str(value).lower() for key, value
              in model.config.id2label.items()}
    lookup = {}
    for index, name in labels.items():
        for target in ("entailment", "contradiction", "neutral"):
            if target in name: lookup[target] = index
    order = [lookup.get("entailment", 0), lookup.get("contradiction", 2),
             lookup.get("neutral", 1)]
    classifier = getattr(model, "classifier", None)
    if classifier is None or not hasattr(classifier, "weight"):
        raise ValueError("cannot locate NLI classification head")
    with torch.no_grad():
        classifier.weight.copy_(classifier.weight[order].clone())
        if classifier.bias is not None:
            classifier.bias.copy_(classifier.bias[order].clone())
    model.config.id2label = {0: "supported", 1: "refuted", 2: "nei"}
    model.config.label2id = {value: key for key, value
                             in model.config.id2label.items()}
    return order


@torch.inference_mode()
def evaluate(model, loader, device, top_k, temperature):
    model.eval(); labels=[]; probabilities=[]; hits=[]; stance_y=[]; stance_p=[]
    for batch in tqdm(loader, desc="validation", leave=False):
        encoded={key:value.to(device) for key,value in batch["encoded"].items()}
        mask=batch["mask"].to(device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                            enabled=device.type=="cuda"):
            pair=model(**encoded).logits.reshape(-1,top_k,3)
            claim=aggregate_nli_logits(pair,mask,temperature)
        probabilities.extend(torch.softmax(claim.float(),-1).cpu().tolist())
        labels.extend(batch["labels"].tolist())
        for i in range(len(batch["labels"])):
            positions=torch.where(batch["relevance"][i]&batch["mask"][i])[0]
            if len(positions):
                non_nei=pair[i,:,:2].logsumexp(-1)
                non_nei=non_nei.masked_fill(~mask[i],-1e4)
                hits.append(int(int(non_nei.argmax().cpu()) in positions.tolist()))
                for position in positions:
                    stance_y.append(int(batch["labels"][i]))
                    stance_p.append(int(pair[i,position].argmax().cpu()))
    y=np.asarray(labels); probability=np.asarray(probabilities); pred=probability.argmax(-1)
    return {"samples":len(y),"accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro")),
            "confusion_matrix":confusion_matrix(y,pred).tolist(),
            "evidence_selection_hit_at_1":float(np.mean(hits)) if hits else 0.0,
            "gold_pair_stance_macro_f1":float(f1_score(
                stance_y,stance_p,average="macro")) if stance_y else 0.0,
            "ece_10":expected_calibration_error(probability,y)}


def main():
    from transformers import (AutoModelForSequenceClassification,AutoTokenizer,
                              get_linear_schedule_with_warmup)
    p=argparse.ArgumentParser()
    p.add_argument("--manifest-root",type=Path,default=Path("data/processed/mocheg_manifest_strict"))
    p.add_argument("--retrieval-root",type=Path,default=Path("outputs/retrieval_mocheg_qwen3_reranked"))
    p.add_argument("--raw-root",type=Path,default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    p.add_argument("--model",default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    p.add_argument("--output",type=Path,default=Path("outputs/mocheg_nli_set_seed42_v13"))
    p.add_argument("--top-k",type=int,default=5); p.add_argument("--max-length",type=int,default=512)
    p.add_argument("--batch-size",type=int,default=2); p.add_argument("--gradient-accumulation",type=int,default=8)
    p.add_argument("--epochs",type=int,default=4); p.add_argument("--patience",type=int,default=2)
    p.add_argument("--learning-rate",type=float,default=5e-6); p.add_argument("--weight-decay",type=float,default=.01)
    p.add_argument("--warmup-ratio",type=float,default=.06); p.add_argument("--claim-loss-weight",type=float,default=1.0)
    p.add_argument("--pair-loss-weight",type=float,default=.25); p.add_argument("--weak-negative-weight",type=float,default=.25)
    p.add_argument("--temperature",type=float,default=.5); p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--limit-train",type=int,default=0); p.add_argument("--limit-val",type=int,default=0)
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    dev=torch.device(a.device if torch.cuda.is_available() else "cpu"); a.output.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForSequenceClassification.from_pretrained(a.model,torch_dtype=torch.bfloat16 if dev.type=="cuda" else torch.float32).to(dev)
    source_order=reorder_nli_head(model)
    if hasattr(model,"gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
    train=EvidenceSets(a.manifest_root/"train.jsonl",a.retrieval_root/"train.jsonl",a.raw_root/"train"/"Corpus2.csv",a.top_k,True,a.limit_train)
    val=EvidenceSets(a.manifest_root/"val.jsonl",a.retrieval_root/"val.jsonl",a.raw_root/"val"/"Corpus2.csv",a.top_k,False,a.limit_val)
    collate=make_collate(tokenizer,a.top_k,a.max_length)
    train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,collate_fn=collate,pin_memory=True)
    val_loader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,collate_fn=collate,pin_memory=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
    updates=max(1,int(np.ceil(len(train_loader)/a.gradient_accumulation)))*a.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,int(updates*a.warmup_ratio),updates)
    best=-1.; stale=0; history=[]; optimizer.zero_grad(set_to_none=True)
    for epoch in range(1,a.epochs+1):
        model.train(); running=0.; progress=tqdm(train_loader,desc=f"epoch {epoch}/{a.epochs}")
        for step,batch in enumerate(progress,1):
            encoded={key:value.to(dev,non_blocking=True) for key,value in batch["encoded"].items()}; labels=batch["labels"].to(dev); mask=batch["mask"].to(dev); relevance=batch["relevance"].to(dev)
            with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=dev.type=="cuda"):
                pair=model(**encoded).logits.reshape(-1,a.top_k,3); claim=aggregate_nli_logits(pair,mask,a.temperature)
                claim_loss=F.cross_entropy(claim.float(),labels)
                targets=torch.where(relevance,labels[:,None].expand_as(relevance),torch.full_like(relevance,2))
                terms=F.cross_entropy(pair.float().reshape(-1,3),targets.reshape(-1),reduction="none").reshape_as(relevance)
                weights=(relevance.float()+(~relevance&mask).float()*a.weak_negative_weight)*mask.float()
                pair_loss=(terms*weights).sum()/weights.sum().clamp_min(1)
                loss=a.claim_loss_weight*claim_loss+a.pair_loss_weight*pair_loss
            (loss/a.gradient_accumulation).backward(); running+=float(loss.detach())
            if step%a.gradient_accumulation==0 or step==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        metrics=evaluate(model,val_loader,dev,a.top_k,a.temperature); row={"epoch":epoch,"train_loss":running/len(train_loader),**metrics}; history.append(row); print(json.dumps(row),flush=True)
        if metrics["macro_f1"]>best:
            best=metrics["macro_f1"]; stale=0; root=a.output/"best"; temp=a.output/"best.tmp"
            if temp.exists(): shutil.rmtree(temp)
            model.save_pretrained(temp,safe_serialization=True); tokenizer.save_pretrained(temp)
            if root.exists(): shutil.rmtree(root)
            temp.replace(root); (a.output/"val_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
        else:
            stale+=1
            if stale>=a.patience: break
    summary={"model":a.model,"source_nli_order":source_order,"best_val_macro_f1":best,"train_gold_coverage_after_injection":train.gold_coverage,"validation_natural_gold_coverage":val.gold_coverage,"frozen_embedding_anchor_macro_f1":.5499479090475745,"delta_vs_anchor":best-.5499479090475745,"history":history,"test_split_used":False,"settings":{key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()}}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
