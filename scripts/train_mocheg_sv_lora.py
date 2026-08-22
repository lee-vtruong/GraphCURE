"""Train GraphCURE-SV with hierarchical and counterfactual supervision.

SV separates evidence sufficiency (enough evidence vs NEI) from polarity
(support vs refute).  Counterfactual rows remove annotated evidence from
supported/refuted claims and conservatively supervise the result as NEI.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.cache_mocheg_reasoning_features import inject_gold_candidate
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_cached_verifier import expected_calibration_error
from scripts.train_mocheg_qwen3_lora_verifier import (
    LABEL_CODES,
    as_token_id_list,
    compose_user_prompt,
    label_token_ids,
    prompt_ids,
    read_documents,
)
from scripts.prepare_mocheg_sv_folds import sha256


def deterministic_fraction(sample_id: str, seed: int) -> float:
    value = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()[:8]
    return int.from_bytes(value, "big") / float(2**64)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def balanced_class_weights(labels: list[int], mode: str) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels), minlength=3).float()
    if mode == "none":
        weights = torch.ones_like(counts)
    elif mode == "inverse":
        weights = counts.sum() / counts.clamp_min(1)
    elif mode == "sqrt":
        weights = torch.sqrt(counts.sum() / counts.clamp_min(1))
    else:
        raise ValueError(f"unknown class balance mode: {mode}")
    return weights / weights.mean()


class SufficiencyVerifierDataset(Dataset):
    def __init__(
        self, manifest: Path, retrieval: Path, corpus: Path, top_k: int,
        max_evidence_chars: int, allowed_ids: set[str] | None = None,
        inject_train_gold: bool = False, counterfactual_ratio: float = 0.0,
        counterfactual_weight: float = 0.35, seed: int = 42,
        limit: int = 0,
    ) -> None:
        claims = read_jsonl(manifest)
        if allowed_ids is not None:
            claims = [row for row in claims if row["id"] in allowed_ids]
            missing = allowed_ids - {row["id"] for row in claims}
            if missing:
                raise ValueError(f"fold IDs missing from manifest: {len(missing)}")
        if limit:
            claims = claims[:limit]
        retrieval_by_id = {row["id"]: row for row in read_jsonl(retrieval)}
        documents = read_documents(corpus)
        self.rows: list[dict] = []
        self.injected_claims = 0
        self.counterfactual_claims = 0
        for claim in claims:
            retrieved = retrieval_by_id.get(claim["id"])
            if retrieved is None:
                raise ValueError(f"retrieval missing {claim['id']}")
            natural = [
                value for value in retrieved.get("retrieved_evidence_ids", [])
                if documents.get(value)
            ][:top_k]
            candidates = natural
            if inject_train_gold:
                candidates, changed = inject_gold_candidate(
                    claim, candidates, documents, top_k
                )
                self.injected_claims += int(changed)
            self.rows.append(self._row(
                claim, candidates, documents, max_evidence_chars,
                int(claim["label"]), 1.0, "natural",
            ))

            gold = {
                str(value) for value in claim.get("text_evidence_ids", [])
                if documents.get(str(value))
            }
            eligible = int(claim["label"]) in (0, 1) and bool(gold)
            selected = deterministic_fraction(claim["id"], seed) < counterfactual_ratio
            if eligible and selected:
                counterfactual = [value for value in natural if value not in gold]
                if counterfactual:
                    self.rows.append(self._row(
                        claim, counterfactual[:top_k], documents,
                        max_evidence_chars, 2, counterfactual_weight,
                        "evidence_dropout",
                    ))
                    self.counterfactual_claims += 1

    @staticmethod
    def _row(claim, candidates, documents, max_chars, label, weight, kind):
        evidence = [documents[value] for value in candidates if documents.get(value)]
        return {
            "id": claim["id"] if kind == "natural" else f"{claim['id']}::cf",
            "parent_id": claim["id"], "label": label,
            "sample_weight": float(weight), "kind": kind,
            "user": compose_user_prompt(claim.get("claim", ""), evidence, max_chars),
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_sv_collate(tokenizer, max_length: int):
    pad = tokenizer.pad_token_id

    def collate(rows: list[dict]) -> dict:
        sequences = [prompt_ids(tokenizer, row["user"], max_length) for row in rows]
        width = max(map(len, sequences))
        input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            input_ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention[index, :len(sequence)] = 1
        return {
            "input_ids": input_ids, "attention_mask": attention,
            "class_labels": torch.tensor([row["label"] for row in rows]),
            "sample_weights": torch.tensor([row["sample_weight"] for row in rows]),
            "ids": [row["id"] for row in rows],
        }
    return collate


def hierarchical_verification_loss(
    logits: torch.Tensor, labels: torch.Tensor, sample_weights: torch.Tensor,
    class_weights: torch.Tensor | None = None, sufficiency_weight: float = 0.5,
    polarity_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Three-class loss plus explicit sufficiency and conditional polarity."""
    logits = logits.float()
    weights = sample_weights.float()
    verdict = F.cross_entropy(
        logits, labels, weight=class_weights, reduction="none"
    )
    verdict = (verdict * weights).sum() / weights.sum().clamp_min(1e-8)

    # Class 0/1 = sufficient evidence; class 2 = insufficient/NEI.
    grouped = torch.stack((torch.logsumexp(logits[:, :2], dim=-1), logits[:, 2]), -1)
    sufficient_target = (labels == 2).long()
    sufficiency = F.cross_entropy(grouped, sufficient_target, reduction="none")
    sufficiency = (sufficiency * weights).sum() / weights.sum().clamp_min(1e-8)

    polarity_mask = labels < 2
    if polarity_mask.any():
        polarity = F.cross_entropy(
            logits[polarity_mask, :2], labels[polarity_mask], reduction="none"
        )
        polarity_weights = weights[polarity_mask]
        polarity = (polarity * polarity_weights).sum() / polarity_weights.sum().clamp_min(1e-8)
    else:
        polarity = logits.sum() * 0.0
    total = verdict + sufficiency_weight * sufficiency + polarity_weight * polarity
    return total, {"verdict": verdict, "sufficiency": sufficiency, "polarity": polarity}


@torch.inference_mode()
def evaluate(model, loader, answer_ids, device):
    model.eval(); labels=[]; probabilities=[]; rows=[]
    token_index = torch.tensor(answer_ids, device=device)
    for batch in tqdm(loader, desc="validation", leave=False):
        inputs={key:batch[key].to(device,non_blocking=True) for key in ("input_ids","attention_mask")}
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            all_logits=model(**inputs).logits
        final_index=inputs["attention_mask"].sum(-1)-1
        selected=all_logits[torch.arange(len(final_index),device=device),final_index][:,token_index].float()
        probability=torch.softmax(selected,-1).cpu(); prediction=probability.argmax(-1)
        labels.extend(batch["class_labels"].tolist()); probabilities.extend(probability.tolist())
        for index,sample_id in enumerate(batch["ids"]):
            rows.append({"id":sample_id,"gold":int(batch["class_labels"][index]),"prediction":int(prediction[index]),"probabilities":probability[index].tolist()})
    y=np.asarray(labels); probability=np.asarray(probabilities); prediction=probability.argmax(-1)
    return {
        "samples":len(y),"accuracy":float(accuracy_score(y,prediction)),
        "macro_f1":float(f1_score(y,prediction,average="macro")),
        "confusion_matrix":confusion_matrix(y,prediction,labels=[0,1,2]).tolist(),
        "classification_report":classification_report(y,prediction,labels=[0,1,2],output_dict=True,zero_division=0),
        "ece_10":expected_calibration_error(probability,y),
    },rows


def load_fold(path: Path | None, index: int) -> tuple[set[str] | None,set[str] | None,dict | None]:
    if path is None:
        return None,None,None
    payload=json.loads(path.read_text(encoding="utf-8"))
    match=next((row for row in payload["folds"] if int(row["fold"])==index),None)
    if match is None: raise ValueError(f"fold {index} absent from {path}")
    return set(match["train_ids"]),set(match["val_ids"]),payload


def main() -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    p=argparse.ArgumentParser()
    p.add_argument("--manifest-root",type=Path,default=Path("data/processed/mocheg_manifest_strict")); p.add_argument("--retrieval-root",type=Path,default=Path("outputs/retrieval_mocheg_qwen3_reranked")); p.add_argument("--raw-root",type=Path,default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    p.add_argument("--fold-spec",type=Path); p.add_argument("--fold-index",type=int,default=0)
    p.add_argument("--model",default="Qwen/Qwen3-4B-Instruct-2507"); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--top-k",type=int,default=5); p.add_argument("--max-evidence-chars",type=int,default=1500); p.add_argument("--max-length",type=int,default=2048)
    p.add_argument("--lora-r",type=int,default=16); p.add_argument("--lora-alpha",type=int,default=32); p.add_argument("--lora-dropout",type=float,default=.05)
    p.add_argument("--epochs",type=int,default=3); p.add_argument("--patience",type=int,default=2); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--gradient-accumulation",type=int,default=16)
    p.add_argument("--learning-rate",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=.01); p.add_argument("--warmup-ratio",type=float,default=.05)
    p.add_argument("--sufficiency-weight",type=float,default=.5); p.add_argument("--polarity-weight",type=float,default=.25); p.add_argument("--counterfactual-ratio",type=float,default=.5); p.add_argument("--counterfactual-weight",type=float,default=.35); p.add_argument("--class-balance",choices=("none","sqrt","inverse"),default="sqrt")
    p.add_argument("--no-train-gold-injection",action="store_true"); p.add_argument("--num-workers",type=int,default=2); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42); p.add_argument("--limit-train",type=int,default=0); p.add_argument("--limit-val",type=int,default=0)
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device(a.device if torch.cuda.is_available() else "cpu"); a.output.mkdir(parents=True,exist_ok=True)
    train_ids,val_ids,fold_payload=load_fold(a.fold_spec,a.fold_index)
    cv_mode=fold_payload is not None
    train_manifest=a.manifest_root/"train.jsonl"; train_retrieval=a.retrieval_root/"train.jsonl"; train_corpus=a.raw_root/"train"/"Corpus2.csv"
    if cv_mode and fold_payload["manifest_sha256"] != sha256(train_manifest):
        raise ValueError(
            "fold specification was built from a different training manifest"
        )
    if cv_mode and train_ids & val_ids:
        raise ValueError("fold train and validation IDs overlap")
    val_manifest=train_manifest if cv_mode else a.manifest_root/"val.jsonl"; val_retrieval=train_retrieval if cv_mode else a.retrieval_root/"val.jsonl"; val_corpus=train_corpus if cv_mode else a.raw_root/"val"/"Corpus2.csv"
    tokenizer=AutoTokenizer.from_pretrained(a.model); tokenizer.padding_side="right"
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    answer_ids=label_token_ids(tokenizer)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16 if device.type=="cuda" else torch.float32,attn_implementation="sdpa").to(device); model.config.use_cache=False
    if hasattr(model,"gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
    if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=a.lora_r,lora_alpha=a.lora_alpha,lora_dropout=a.lora_dropout,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],bias="none")); model.print_trainable_parameters()
    train=SufficiencyVerifierDataset(train_manifest,train_retrieval,train_corpus,a.top_k,a.max_evidence_chars,train_ids,not a.no_train_gold_injection,a.counterfactual_ratio,a.counterfactual_weight,a.seed,a.limit_train)
    val=SufficiencyVerifierDataset(val_manifest,val_retrieval,val_corpus,a.top_k,a.max_evidence_chars,val_ids,False,0.,a.counterfactual_weight,a.seed,a.limit_val)
    collate=make_sv_collate(tokenizer,a.max_length); train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,collate_fn=collate,pin_memory=device.type=="cuda"); val_loader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,collate_fn=collate,pin_memory=device.type=="cuda")
    natural_labels=[row["label"] for row in train.rows if row["kind"]=="natural"]
    class_weights=balanced_class_weights(natural_labels,a.class_balance).to(device)
    parameters=[value for value in model.parameters() if value.requires_grad]; optimizer=torch.optim.AdamW(parameters,lr=a.learning_rate,weight_decay=a.weight_decay)
    updates=max(1,int(np.ceil(len(train_loader)/a.gradient_accumulation)))*a.epochs; scheduler=get_cosine_schedule_with_warmup(optimizer,int(updates*a.warmup_ratio),updates)
    best=-1.; best_epoch=0; stale=0; history=[]; best_state=None; answer_tensor=torch.tensor(answer_ids,device=device); optimizer.zero_grad(set_to_none=True)
    target=a.output/"best_adapter"
    for epoch in range(1,a.epochs+1):
        model.train(); sums=Counter(); progress=tqdm(train_loader,desc=f"epoch {epoch}/{a.epochs}")
        for step,batch in enumerate(progress,1):
            inputs={key:batch[key].to(device,non_blocking=True) for key in ("input_ids","attention_mask")}; labels=batch["class_labels"].to(device); sample_weights=batch["sample_weights"].to(device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                all_logits=model(**inputs).logits; final_index=inputs["attention_mask"].sum(-1)-1; logits=all_logits[torch.arange(len(final_index),device=device),final_index][:,answer_tensor]
                loss,parts=hierarchical_verification_loss(logits,labels,sample_weights,class_weights,a.sufficiency_weight,a.polarity_weight)
            (loss/a.gradient_accumulation).backward()
            for name,value in parts.items(): sums[name]+=float(value.detach())
            sums["total"]+=float(loss.detach())
            if step%a.gradient_accumulation==0 or step==len(train_loader):
                torch.nn.utils.clip_grad_norm_(parameters,1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        metrics,_=evaluate(model,val_loader,answer_ids,device); row={"epoch":epoch,**{f"train_{key}_loss":value/len(train_loader) for key,value in sums.items()},**metrics}; history.append(row); print(json.dumps(row),flush=True)
        if metrics["macro_f1"]>best:
            best=metrics["macro_f1"]; best_epoch=epoch; stale=0
            best_state={name:value.detach().cpu().clone() for name,value in model.named_parameters() if value.requires_grad}
            temporary=a.output/"best.tmp"
            if temporary.exists(): shutil.rmtree(temporary)
            model.save_pretrained(temporary,safe_serialization=True); tokenizer.save_pretrained(temporary)
            if target.exists(): shutil.rmtree(target)
            temporary.replace(target)
        else:
            stale+=1
            if stale>=a.patience: break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state,strict=False); del best_state; gc.collect()
    final,rows=evaluate(model,val_loader,answer_ids,device)
    summary={"method":"GraphCURE-SV","protocol":"train_only_cv" if cv_mode else "external_validation","fold":a.fold_index if cv_mode else None,"model":a.model,"best_epoch":best_epoch,"best_val_macro_f1":best,"final":final,"injected_train_claims":train.injected_claims,"counterfactual_train_claims":train.counterfactual_claims,"train_rows":len(train),"natural_train_label_counts":dict(Counter(map(str,natural_labels))),"class_weights":class_weights.cpu().tolist(),"history":history,"test_split_used":False,"official_validation_used_for_selection":not cv_mode,"provenance":{"git_commit":git_commit(),"train_manifest_sha256":sha256(train_manifest),"train_retrieval_sha256":sha256(train_retrieval),"fold_spec_sha256":sha256(a.fold_spec) if a.fold_spec else None},"settings":{key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()}}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); (a.output/"val_predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n",encoding="utf-8"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
