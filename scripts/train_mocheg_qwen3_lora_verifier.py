"""LoRA-tune an instruction verifier on retrieved MOCHEG evidence."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.cache_mocheg_reasoning_features import inject_gold_candidate
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_cached_verifier import expected_calibration_error
from scripts.prepare_mocheg_sv_folds import sha256


LABEL_CODES = {0: "A", 1: "B", 2: "C"}
SYSTEM_PROMPT = (
    "You are an evidence-grounded fact verifier. Use only the retrieved "
    "evidence supplied by the user. Decide whether the claim is A: supported, "
    "B: refuted, or C: not enough information. Conflicting, irrelevant, or "
    "insufficient evidence must not be treated as support. Answer with exactly "
    "one letter: A, B, or C."
)


def read_documents(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            evidence_id = row.get("evidence_id", "").strip()
            text = row.get("Evidence", "").replace("<p>", " ").replace(
                "</p>", " ").strip()
            if evidence_id and text:
                result.setdefault(evidence_id, text)
    return result


def compose_user_prompt(claim: str, evidence: list[str],
                        max_evidence_chars: int) -> str:
    sections = [f"Claim:\n{claim.strip()}", "Retrieved evidence:"]
    for index, text in enumerate(evidence, 1):
        sections.append(f"[{index}] {text[:max_evidence_chars].strip()}")
    sections.append("Return only A, B, or C.")
    return "\n\n".join(sections)


def as_token_id_list(value) -> list[int]:
    """Normalize tokenizer outputs across Transformers/tokenizers versions."""
    if hasattr(value, "ids"):
        value = value.ids
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and any(
        hasattr(item, "ids") or hasattr(item, "input_ids")
        for item in value
    ):
        flattened = []
        for item in value:
            if isinstance(item, (int, np.integer)):
                flattened.append(int(item))
            else:
                flattened.extend(as_token_id_list(item))
        return flattened
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(
        value[0], (list, tuple)
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"unsupported tokenizer output: {type(value)!r}")
    return [int(token) for token in value]


class QwenVerifierDataset(Dataset):
    def __init__(self, manifest: Path, retrieval: Path, corpus: Path,
                 top_k: int, max_evidence_chars: int,
                 inject_train_gold: bool = False, limit: int = 0,
                 allowed_ids: set[str] | None = None) -> None:
        claims = read_jsonl(manifest)
        if allowed_ids is not None:
            claims = [row for row in claims if row["id"] in allowed_ids]
            missing = allowed_ids - {row["id"] for row in claims}
            if missing:
                raise ValueError(
                    f"fold IDs missing from verifier manifest: {len(missing)}"
                )
        if limit:
            claims = claims[:limit]
        retrieval_by_id = {row["id"]: row for row in read_jsonl(retrieval)}
        documents = read_documents(corpus)
        self.rows = []
        self.injected_claims = 0
        for claim in claims:
            retrieved = retrieval_by_id.get(claim["id"])
            if retrieved is None:
                raise ValueError(f"retrieval missing {claim['id']}")
            candidates = [
                value for value in retrieved.get("retrieved_evidence_ids", [])[:top_k]
                if documents.get(value)
            ]
            if inject_train_gold:
                candidates, changed = inject_gold_candidate(
                    claim, candidates, documents, top_k
                )
                self.injected_claims += int(changed)
            evidence = [documents[value] for value in candidates[:top_k]]
            self.rows.append({
                "id": claim["id"], "label": int(claim["label"]),
                "user": compose_user_prompt(
                    claim.get("claim", ""), evidence, max_evidence_chars
                ),
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def prompt_ids(tokenizer, user: str, budget: int) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    ids = as_token_id_list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    ))
    if len(ids) <= budget:
        return ids
    prefix = min(512, budget // 3)
    return ids[:prefix] + ids[-(budget - prefix):]


def label_token_ids(tokenizer) -> list[int]:
    result = []
    for code in ("A", "B", "C"):
        ids = as_token_id_list(tokenizer.encode(code, add_special_tokens=False))
        if len(ids) != 1:
            raise ValueError(f"label {code!r} is not one token: {ids}")
        result.append(ids[0])
    if len(set(result)) != 3:
        raise ValueError("A/B/C label tokens are not distinct")
    return result


def make_collate(tokenizer, max_length: int, training: bool):
    pad = tokenizer.pad_token_id

    def collate(rows: list[dict]) -> dict:
        sequences, targets = [], []
        for row in rows:
            prompt = prompt_ids(tokenizer, row["user"], max_length - int(training))
            if training:
                answer = as_token_id_list(tokenizer.encode(
                    LABEL_CODES[row["label"]], add_special_tokens=False
                ))[0]
                sequences.append(prompt + [answer])
                targets.append([-100] * len(prompt) + [answer])
            else:
                sequences.append(prompt); targets.append([])
        width = max(len(value) for value in sequences)
        input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        for index, sequence in enumerate(sequences):
            input_ids[index, :len(sequence)] = torch.tensor(sequence)
            attention[index, :len(sequence)] = 1
            if training:
                labels[index, :len(sequence)] = torch.tensor(targets[index])
        return {
            "input_ids": input_ids, "attention_mask": attention,
            "token_labels": labels,
            "class_labels": torch.tensor([row["label"] for row in rows]),
            "ids": [row["id"] for row in rows],
        }
    return collate


@torch.inference_mode()
def evaluate(model, loader, answer_ids: list[int], device):
    model.eval(); labels=[]; probabilities=[]; rows=[]
    token_index = torch.tensor(answer_ids, device=device)
    for batch in tqdm(loader, desc="validation", leave=False):
        inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        }
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(**inputs).logits.float()
        final_index = inputs["attention_mask"].sum(-1) - 1
        selected = logits[
            torch.arange(len(final_index), device=device), final_index
        ][:, token_index]
        probability = torch.softmax(selected, -1).cpu()
        prediction = probability.argmax(-1)
        labels.extend(batch["class_labels"].tolist())
        probabilities.extend(probability.tolist())
        for index, sample_id in enumerate(batch["ids"]):
            rows.append({
                "id": sample_id, "gold": int(batch["class_labels"][index]),
                "prediction": int(prediction[index]),
                "probabilities": probability[index].tolist(),
            })
    y=np.asarray(labels); probability=np.asarray(probabilities)
    prediction=probability.argmax(-1)
    return {
        "samples":len(y), "accuracy":float(accuracy_score(y,prediction)),
        "macro_f1":float(f1_score(y,prediction,average="macro")),
        "confusion_matrix":confusion_matrix(y,prediction).tolist(),
        "ece_10":expected_calibration_error(probability,y),
    }, rows


def load_fold(path: Path | None, index: int) -> tuple[
        set[str] | None, set[str] | None, dict | None]:
    if path is None:
        return None, None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    match = next(
        (row for row in payload["folds"] if int(row["fold"]) == index), None
    )
    if match is None:
        raise ValueError(f"fold {index} absent from {path}")
    fit, held = set(match["train_ids"]), set(match["val_ids"])
    if fit & held:
        raise ValueError("fold fit and held IDs overlap")
    return fit, held, payload


def main() -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)
    p=argparse.ArgumentParser()
    p.add_argument("--manifest-root",type=Path,default=Path("data/processed/mocheg_manifest_strict")); p.add_argument("--retrieval-root",type=Path,default=Path("outputs/retrieval_mocheg_qwen3_reranked")); p.add_argument("--raw-root",type=Path,default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    p.add_argument("--fold-spec",type=Path); p.add_argument("--fold-index",type=int,default=0)
    p.add_argument("--fixed-checkpoint-epoch",type=int,default=0,help="Select exactly this epoch; required for train-only folds")
    p.add_argument("--model",default="Qwen/Qwen3-4B-Instruct-2507"); p.add_argument("--output",type=Path,default=Path("outputs/mocheg_qwen3_lora_seed42_v16"))
    p.add_argument("--top-k",type=int,default=5); p.add_argument("--max-evidence-chars",type=int,default=2200); p.add_argument("--max-length",type=int,default=3072)
    p.add_argument("--lora-r",type=int,default=16); p.add_argument("--lora-alpha",type=int,default=32); p.add_argument("--lora-dropout",type=float,default=.05)
    p.add_argument("--epochs",type=int,default=3); p.add_argument("--patience",type=int,default=2); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--gradient-accumulation",type=int,default=16)
    p.add_argument("--learning-rate",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=.01); p.add_argument("--warmup-ratio",type=float,default=.05)
    p.add_argument("--no-train-gold-injection",action="store_true"); p.add_argument("--num-workers",type=int,default=2); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--limit-train",type=int,default=0); p.add_argument("--limit-val",type=int,default=0)
    p.add_argument("--anchor-macro-f1",type=float,default=.5499479090475745)
    p.add_argument("--minimum-delta",type=float,default=.003)
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    fit_ids,held_ids,fold_payload=load_fold(a.fold_spec,a.fold_index)
    cv_mode=fold_payload is not None
    if cv_mode and not 1 <= a.fixed_checkpoint_epoch <= a.epochs:
        raise ValueError(
            "train-only folds require --fixed-checkpoint-epoch in [1, epochs]"
        )
    if a.fixed_checkpoint_epoch < 0 or a.fixed_checkpoint_epoch > a.epochs:
        raise ValueError("fixed checkpoint epoch must be in [0, epochs]")
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device(a.device if torch.cuda.is_available() else "cpu"); a.output.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(a.model); tokenizer.padding_side="right"
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    answer_ids=label_token_ids(tokenizer)
    model=AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16 if device.type=="cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache=False
    if hasattr(model,"gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
    if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()
    config=LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=a.lora_r, lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
    )
    model=get_peft_model(model,config); model.print_trainable_parameters()
    train_manifest=a.manifest_root/"train.jsonl"
    train_retrieval=a.retrieval_root/"train.jsonl"
    train_corpus=a.raw_root/"train"/"Corpus2.csv"
    if cv_mode and fold_payload["manifest_sha256"] != sha256(train_manifest):
        raise ValueError("fold specification does not match training manifest")
    val_manifest=train_manifest if cv_mode else a.manifest_root/"val.jsonl"
    val_retrieval=train_retrieval if cv_mode else a.retrieval_root/"val.jsonl"
    val_corpus=train_corpus if cv_mode else a.raw_root/"val"/"Corpus2.csv"
    train=QwenVerifierDataset(train_manifest,train_retrieval,train_corpus,a.top_k,a.max_evidence_chars,not a.no_train_gold_injection,a.limit_train,fit_ids)
    val=QwenVerifierDataset(val_manifest,val_retrieval,val_corpus,a.top_k,a.max_evidence_chars,False,a.limit_val,held_ids)
    train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,collate_fn=make_collate(tokenizer,a.max_length,True),pin_memory=device.type=="cuda")
    val_loader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,collate_fn=make_collate(tokenizer,a.max_length,False),pin_memory=device.type=="cuda")
    parameters=[value for value in model.parameters() if value.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=a.learning_rate,weight_decay=a.weight_decay)
    updates=max(1,int(np.ceil(len(train_loader)/a.gradient_accumulation)))*a.epochs
    scheduler=get_cosine_schedule_with_warmup(optimizer,int(updates*a.warmup_ratio),updates)
    fixed_epoch=a.fixed_checkpoint_epoch or None
    if fixed_epoch is None:
        initial,_=evaluate(model,val_loader,answer_ids,device)
        best=initial["macro_f1"]; best_epoch=0
        best_state={name:value.detach().cpu().clone() for name,value in model.named_parameters() if value.requires_grad}
        history=[{"epoch":0,"train_loss":None,**initial}]
        print(json.dumps(history[0]),flush=True)
    else:
        best=float("-inf"); best_epoch=0; best_state=None; history=[]
    stale=0
    target=a.output/"best_adapter"; temporary=a.output/"best.tmp"
    if fixed_epoch is None:
        if temporary.exists(): shutil.rmtree(temporary)
        model.save_pretrained(temporary,safe_serialization=True); tokenizer.save_pretrained(temporary)
        if target.exists(): shutil.rmtree(target)
        temporary.replace(target)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1,a.epochs+1):
        model.train(); running=0.; progress=tqdm(train_loader,desc=f"epoch {epoch}/{a.epochs}")
        for step,batch in enumerate(progress,1):
            inputs={key:batch[key].to(device,non_blocking=True) for key in ("input_ids","attention_mask")}
            labels=batch["token_labels"].to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                loss=model(**inputs,labels=labels).loss
            (loss/a.gradient_accumulation).backward(); running+=float(loss.detach())
            if step%a.gradient_accumulation==0 or step==len(train_loader):
                torch.nn.utils.clip_grad_norm_(parameters,1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        if fixed_epoch is not None and epoch < fixed_epoch:
            row={"epoch":epoch,"train_loss":running/len(train_loader),"held_fold_evaluated":False}; history.append(row); print(json.dumps(row),flush=True)
            continue
        metrics,_=evaluate(model,val_loader,answer_ids,device); row={"epoch":epoch,"train_loss":running/len(train_loader),"held_fold_evaluated":True,**metrics}; history.append(row); print(json.dumps(row),flush=True)
        selected=(epoch==fixed_epoch) if fixed_epoch is not None else (metrics["macro_f1"]>best)
        if selected:
            best=metrics["macro_f1"]; best_epoch=epoch; stale=0
            best_state={name:value.detach().cpu().clone() for name,value in model.named_parameters() if value.requires_grad}
            temporary=a.output/"best.tmp"; target=a.output/"best_adapter"
            if temporary.exists(): shutil.rmtree(temporary)
            model.save_pretrained(temporary,safe_serialization=True); tokenizer.save_pretrained(temporary)
            if target.exists(): shutil.rmtree(target)
            temporary.replace(target)
            if fixed_epoch is not None:
                break
        else:
            stale+=1
            if stale>=a.patience: break
    if best_state is None:
        raise RuntimeError("fixed checkpoint epoch was not reached")
    model.load_state_dict(best_state,strict=False)
    final,rows=evaluate(model,val_loader,answer_ids,device)
    anchor=a.anchor_macro_f1; accepted=best>=anchor+a.minimum_delta
    summary={"model":a.model,"mode":"train_only_fixed_epoch_anchor" if cv_mode else ("qwen3_lora" if accepted else "rejected_keep_anchor"),"accepted":None if cv_mode else accepted,"protocol":"train_only_duplicate_safe_fixed_epoch_cv" if cv_mode else "external_validation","fold":a.fold_index if cv_mode else None,"fixed_checkpoint_epoch":fixed_epoch,"official_validation_used_for_selection":False if cv_mode else True,"label_codes":LABEL_CODES,"label_token_ids":answer_ids,"train_gold_injection":not a.no_train_gold_injection,"injected_train_claims":train.injected_claims,"best_val_macro_f1":best,"best_epoch":best_epoch,"delta_vs_anchor":None if cv_mode else best-anchor,"anchor_macro_f1":None if cv_mode else anchor,"minimum_delta":a.minimum_delta,"promotion_threshold":None if cv_mode else anchor+a.minimum_delta,"final_candidate":final,"history":history,"test_split_used":False,"provenance":{"train_manifest_sha256":sha256(train_manifest),"fold_spec_sha256":sha256(a.fold_spec) if a.fold_spec else None},"settings":{key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()}}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); (a.output/"val_predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
