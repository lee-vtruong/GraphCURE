"""Phase B18-A: Grounded explanation distillation verifier training.

Supports two matched execution modes:
1. matched_control: Standard verdict-only LoRA continuation.
2. explanation_candidate: Multitask LoRA continuation optimizing:
   L = L_verdict + lambda_exp * L_explanation
   where L_explanation is only computed on samples whose teacher rationale is
   verified as grounded=true.

CRITICAL PROTOCOL REQUIREMENTS:
- Matched compute: Both modes run for the exact same number of optimizer steps.
- Pure direct verdict inference: Prompt B (explanation) is strictly forbidden at validation/test time.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from graphcure.explanation import (
    LETTER_TO_VERDICT,
    STUDENT_EXPLANATION_SYSTEM_PROMPT,
    STUDENT_VERDICT_SYSTEM_PROMPT,
    compose_student_explanation_prompt,
    compose_student_verdict_prompt,
    resolve_corpus_path,
)
from scripts.prepare_mocheg_sv_folds import sha256
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_cached_verifier import expected_calibration_error
from scripts.train_mocheg_qwen3_lora_verifier import (
    LABEL_CODES,
    as_token_id_list,
    label_token_ids,
    read_documents,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class B18VerifierDataset(Dataset):
    def __init__(
        self,
        claims: list[dict],
        retrieval_by_id: dict[str, dict],
        documents: dict[str, str],
        explanations_by_id: dict[str, dict] | None,
        top_k: int,
        max_evidence_chars: int,
        training: bool,
        inject_train_gold: bool = False,
    ) -> None:
        self.training = training
        self.rows = []
        for claim in claims:
            cid = str(claim["id"])
            claim_text = claim.get("claim", "")
            raw_label = int(claim["label"])
            retrieved = retrieval_by_id.get(cid)
            if retrieved is None:
                continue
            candidates = [
                eid for eid in retrieved.get("retrieved_evidence_ids", [])[:top_k]
                if eid in documents
            ]
            if training and inject_train_gold:
                from scripts.cache_mocheg_reasoning_features import inject_gold_candidate
                candidates, _ = inject_gold_candidate(claim, candidates, documents, top_k)
            evidence_texts = [documents[eid] for eid in candidates]
            verdict_user_prompt = compose_student_verdict_prompt(
                claim_text, evidence_texts, max_evidence_chars=max_evidence_chars
            )

            exp_data = explanations_by_id.get(cid) if explanations_by_id else None
            exp_prompt = None
            exp_target = None
            grounded = False
            if exp_data and exp_data.get("is_valid") and exp_data.get("grounded"):
                grounded = True
                exp_prompt = compose_student_explanation_prompt(
                    claim_text, evidence_texts, max_evidence_chars=max_evidence_chars
                )
                exp_target = json.dumps({
                    "verdict": exp_data.get("verdict"),
                    "key_evidence_ids": exp_data.get("key_evidence_ids", []),
                    "reason": exp_data.get("reason", ""),
                    "missing_information": exp_data.get("missing_information"),
                }, ensure_ascii=False)

            self.rows.append({
                "id": cid,
                "label": raw_label,
                "verdict_prompt": verdict_user_prompt,
                "exp_prompt": exp_prompt,
                "exp_target": exp_target,
                "grounded": grounded,
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def build_chat_prompt_ids(tokenizer: Any, system_prompt: str, user_prompt: str, budget: int) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    ids = as_token_id_list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    ))
    if len(ids) <= budget:
        return ids
    prefix = min(512, budget // 3)
    return ids[:prefix] + ids[-(budget - prefix):]


def make_b18_collate(tokenizer: Any, max_length: int, training: bool, mode: str):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id

    def collate(batch_rows: list[dict]) -> dict:
        v_sequences, v_targets = [], []
        for row in batch_rows:
            prompt = build_chat_prompt_ids(
                tokenizer, STUDENT_VERDICT_SYSTEM_PROMPT, row["verdict_prompt"], max_length - 2
            )
            if training:
                label_token = as_token_id_list(tokenizer.encode(
                    LABEL_CODES[row["label"]], add_special_tokens=False
                ))[0]
                v_sequences.append(prompt + [label_token])
                v_targets.append([-100] * len(prompt) + [label_token])
            else:
                v_sequences.append(prompt)
                v_targets.append([])

        max_v_width = max(len(s) for s in v_sequences)
        v_input_ids = torch.full((len(batch_rows), max_v_width), pad_id, dtype=torch.long)
        v_attention = torch.zeros((len(batch_rows), max_v_width), dtype=torch.long)
        v_labels = torch.full((len(batch_rows), max_v_width), -100, dtype=torch.long)

        for i, seq in enumerate(v_sequences):
            v_input_ids[i, :len(seq)] = torch.tensor(seq)
            v_attention[i, :len(seq)] = 1
            if training:
                v_labels[i, :len(seq)] = torch.tensor(v_targets[i])

        result = {
            "verdict_input_ids": v_input_ids,
            "verdict_attention_mask": v_attention,
            "verdict_labels": v_labels,
            "class_labels": torch.tensor([r["label"] for r in batch_rows], dtype=torch.long),
            "ids": [r["id"] for r in batch_rows],
            "has_explanation": False,
        }

        # Multitask explanation packing
        if training and mode == "explanation_candidate":
            valid_exp_indices = [
                i for i, r in enumerate(batch_rows)
                if r["grounded"] and r["exp_prompt"] and r["exp_target"]
            ]
            if valid_exp_indices:
                e_sequences, e_targets = [], []
                for i in valid_exp_indices:
                    row = batch_rows[i]
                    p_ids = build_chat_prompt_ids(
                        tokenizer, STUDENT_EXPLANATION_SYSTEM_PROMPT, row["exp_prompt"], max_length // 2
                    )
                    t_ids = as_token_id_list(tokenizer.encode(
                        row["exp_target"], add_special_tokens=False
                    ))[: (max_length // 2) - 1] + [eos_id]
                    e_sequences.append(p_ids + t_ids)
                    e_targets.append([-100] * len(p_ids) + t_ids)

                max_e_width = max(len(s) for s in e_sequences)
                e_input_ids = torch.full((len(e_sequences), max_e_width), pad_id, dtype=torch.long)
                e_attention = torch.zeros((len(e_sequences), max_e_width), dtype=torch.long)
                e_labels = torch.full((len(e_sequences), max_e_width), -100, dtype=torch.long)

                for i, seq in enumerate(e_sequences):
                    e_input_ids[i, :len(seq)] = torch.tensor(seq)
                    e_attention[i, :len(seq)] = 1
                    e_labels[i, :len(seq)] = torch.tensor(e_targets[i])

                result["has_explanation"] = True
                result["exp_input_ids"] = e_input_ids
                result["exp_attention_mask"] = e_attention
                result["exp_labels"] = e_labels

        return result

    return collate


@torch.inference_mode()
def evaluate_direct_verdict(model: Any, loader: DataLoader, answer_ids: list[int], device: torch.device) -> dict:
    model.eval()
    token_index = torch.tensor(answer_ids, device=device)
    all_labels, all_probs, pred_rows = [], [], []

    for batch in tqdm(loader, desc="Held-out validation", leave=False):
        input_ids = batch["verdict_input_ids"].to(device, non_blocking=True)
        attention_mask = batch["verdict_attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()

        final_index = attention_mask.sum(-1) - 1
        selected = logits[
            torch.arange(len(final_index), device=device), final_index
        ][:, token_index]
        probs = torch.softmax(selected, dim=-1).cpu()
        preds = probs.argmax(dim=-1)

        all_labels.extend(batch["class_labels"].tolist())
        all_probs.extend(probs.tolist())
        for i, cid in enumerate(batch["ids"]):
            pred_rows.append({
                "id": cid,
                "label": int(batch["class_labels"][i]),
                "prediction": int(preds[i]),
                "probabilities": probs[i].tolist(),
            })

    y_true = np.asarray(all_labels)
    y_prob = np.asarray(all_probs)
    y_pred = y_prob.argmax(axis=-1)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class_f1 = f1_score(y_true, y_pred, average=None).tolist()
    cm = confusion_matrix(y_true, y_pred).tolist()
    ece = float(expected_calibration_error(y_prob, y_true))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_supported": per_class_f1[0] if len(per_class_f1) > 0 else 0.0,
        "f1_refuted": per_class_f1[1] if len(per_class_f1) > 1 else 0.0,
        "f1_nei": per_class_f1[2] if len(per_class_f1) > 2 else 0.0,
        "ece": ece,
        "confusion_matrix": cm,
        "predictions": pred_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase B18 explanation distillation verifier")
    parser.add_argument("--mode", choices=["matched_control", "explanation_candidate"], required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Strict manifest (train.jsonl)")
    parser.add_argument("--retrieval", type=Path, required=True, help="Retrieved evidence manifest")
    parser.add_argument("--corpus", type=Path, required=True, help="Evidence corpus CSV")
    parser.add_argument("--folds", type=Path, default=None, help="B18 folds JSON (optional if --val-manifest is provided)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for evaluation (train on rest)")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Official validation manifest for full training")
    parser.add_argument("--val-retrieval", type=Path, default=None, help="Official validation retrieval for full training")
    parser.add_argument("--val-corpus", type=Path, default=None, help="Official validation corpus CSV")
    parser.add_argument("--inject-train-gold", action="store_true", help="Inject gold candidate during training like B1")
    parser.add_argument("--explanations", type=Path, default=None, help="Teacher explanations JSONL (for candidate)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lambda-exp", type=float, default=0.25, help="Weight for explanation loss")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--max-evidence-chars", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    all_claims = read_jsonl(args.manifest)
    if args.folds is not None and args.folds.is_file():
        fold_data = json.loads(args.folds.read_text(encoding="utf-8"))
        fold_entry = next(f for f in fold_data["folds"] if f["fold"] == args.fold)
        train_ids = set(fold_entry["train_ids"])
        val_ids = set(fold_entry["val_ids"])
        logging.info("Fold %d: %d train claims, %d held-out val claims", args.fold, len(train_ids), len(val_ids))
        train_claims = [c for c in all_claims if str(c["id"]) in train_ids]
        val_claims = [c for c in all_claims if str(c["id"]) in val_ids]
    elif args.val_manifest is not None and args.val_manifest.is_file():
        train_claims = all_claims
        val_claims = read_jsonl(args.val_manifest)
        logging.info("Full training mode: %d train claims, %d official val claims", len(train_claims), len(val_claims))
    else:
        raise ValueError("Either --folds or --val-manifest must be provided.")

    if args.limit > 0:
        train_claims = train_claims[:args.limit]
        val_claims = val_claims[:args.limit]

    retrieval_by_id = {str(r["id"]): r for r in read_jsonl(args.retrieval)}
    if args.val_retrieval and args.val_retrieval.is_file():
        for r in read_jsonl(args.val_retrieval):
            retrieval_by_id[str(r["id"])] = r

    corpus_path = resolve_corpus_path(args.corpus)
    documents = read_documents(corpus_path)
    if args.val_corpus and args.val_corpus.is_file():
        val_corpus_path = resolve_corpus_path(args.val_corpus)
        val_docs = read_documents(val_corpus_path)
        documents.update(val_docs)

    explanations_by_id = {}
    if args.mode == "explanation_candidate":
        if not args.explanations or not args.explanations.is_file():
            raise ValueError("--explanations path must exist when mode=explanation_candidate")
        for row in read_jsonl(args.explanations):
            explanations_by_id[str(row["id"])] = row
        logging.info("Loaded %d explanations for candidate training", len(explanations_by_id))

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    logging.info("Loading base model %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    answer_ids = label_token_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable") and device.type == "cuda":
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    train_ds = B18VerifierDataset(
        train_claims, retrieval_by_id, documents, explanations_by_id,
        args.top_k, args.max_evidence_chars, training=True,
        inject_train_gold=args.inject_train_gold,
    )
    val_ds = B18VerifierDataset(
        val_claims, retrieval_by_id, documents, None,
        args.top_k, args.max_evidence_chars, training=False,
        inject_train_gold=False,
    )

    train_collate = make_b18_collate(tokenizer, args.max_length, training=True, mode=args.mode)
    val_collate = make_b18_collate(tokenizer, args.max_length, training=False, mode="matched_control")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=val_collate)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

    best_macro_f1 = -1.0
    best_results = None
    optimizer_updates = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            v_inputs = batch["verdict_input_ids"].to(device)
            v_attention = batch["verdict_attention_mask"].to(device)
            v_labels = batch["verdict_labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(input_ids=v_inputs, attention_mask=v_attention, labels=v_labels)
                verdict_loss = outputs.loss
            (verdict_loss / args.grad_accum).backward()
            train_loss_sum += float(verdict_loss.detach())

            if batch["has_explanation"]:
                e_inputs = batch["exp_input_ids"].to(device)
                e_attention = batch["exp_attention_mask"].to(device)
                e_labels = batch["exp_labels"].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    e_outputs = model(input_ids=e_inputs, attention_mask=e_attention, labels=e_labels)
                    exp_loss = e_outputs.loss
                ((args.lambda_exp * exp_loss) / args.grad_accum).backward()
                train_loss_sum += float((args.lambda_exp * exp_loss).detach())

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_updates += 1

            pbar.set_postfix({
                "loss": f"{train_loss_sum / (step + 1):.4f}",
                "updates": optimizer_updates,
            })

        val_metrics = evaluate_direct_verdict(model, val_loader, answer_ids, device)
        logging.info(
            "Epoch %d Val: Acc=%.4f, Macro-F1=%.4f (Supp=%.4f, Ref=%.4f, NEI=%.4f)",
            epoch, val_metrics["accuracy"], val_metrics["macro_f1"],
            val_metrics["f1_supported"], val_metrics["f1_refuted"], val_metrics["f1_nei"]
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_results = val_metrics
            adapter_dir = args.output / "best_adapter"
            model.save_pretrained(adapter_dir)
            logging.info("Saved new best adapter (Macro-F1=%.4f) to %s", best_macro_f1, adapter_dir)

    # Save final artifacts
    if best_results is not None:
        preds_path = args.output / "val_predictions.jsonl"
        with preds_path.open("w", encoding="utf-8") as handle:
            for row in best_results["predictions"]:
                handle.write(json.dumps(row) + "\n")

        summary = {
            "phase": "B18-A",
            "mode": args.mode,
            "git_commit": git_commit(),
            "fold": args.fold,
            "seed": args.seed,
            "lambda_exp": args.lambda_exp if args.mode == "explanation_candidate" else 0.0,
            "epochs": args.epochs,
            "total_optimizer_updates": optimizer_updates,
            "best_accuracy": best_results["accuracy"],
            "best_macro_f1": best_results["macro_f1"],
            "f1_supported": best_results["f1_supported"],
            "f1_refuted": best_results["f1_refuted"],
            "f1_nei": best_results["f1_nei"],
            "ece": best_results["ece"],
            "confusion_matrix": best_results["confusion_matrix"],
            "output_dir": str(args.output),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logging.info("Completed %s. Summary: %s", args.mode, json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
