"""Train a B19 disagreement-aware counterfactual distillation verifier."""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from graphcure.b19 import (
    counterfactual_sufficiency_loss,
    heterogeneous_teacher_target,
    resolve_variant,
    soft_distillation_loss,
    weighted_verdict_loss,
)
from graphcure.explanation import (
    STUDENT_VERDICT_SYSTEM_PROMPT,
    compose_student_verdict_prompt,
    resolve_corpus_path,
)
from scripts.prepare_mocheg_sv_folds import sha256
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_b18_explanation_verifier import (
    build_chat_prompt_ids,
    evaluate_direct_verdict,
    git_commit,
    make_b18_collate,
    B18VerifierDataset,
)
from scripts.train_mocheg_qwen3_lora_verifier import label_token_ids, read_documents


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class B19Dataset(Dataset):
    def __init__(
        self,
        claims: list[dict],
        retrieval: dict[str, dict],
        documents: dict[str, str],
        explanations: dict[str, dict],
        direct: dict[str, dict],
        grounded: dict[str, dict],
        top_k: int,
        max_evidence_chars: int,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        self.target_sources = Counter()
        self.counterfactuals = 0
        for claim in claims:
            claim_id = str(claim["id"])
            if claim_id not in retrieval or claim_id not in direct or claim_id not in grounded:
                continue
            evidence_ids = [
                str(value) for value in retrieval[claim_id].get("retrieved_evidence_ids", [])[:top_k]
                if str(value) in documents
            ]
            evidence = [documents[value] for value in evidence_ids]
            if not evidence:
                continue
            label = int(claim["label"])
            target, target_source, disagreement = heterogeneous_teacher_target(
                label,
                direct[claim_id]["probabilities"],
                grounded[claim_id]["probabilities"],
            )
            self.target_sources[target_source] += 1
            explanation = explanations.get(claim_id, {})
            key_indices = sorted({
                int(value) - 1 for value in explanation.get("key_evidence_ids", [])
                if str(value).isdigit() and 1 <= int(value) <= len(evidence)
            })
            counterfactual_evidence = [
                text for index, text in enumerate(evidence) if index not in set(key_indices)
            ]
            cf_valid = bool(
                explanation.get("is_valid")
                and explanation.get("grounded")
                and label != 2
                and key_indices
                and counterfactual_evidence
            )
            if cf_valid:
                self.counterfactuals += 1
            self.rows.append({
                "id": claim_id,
                "label": label,
                "positive_prompt": compose_student_verdict_prompt(
                    str(claim.get("claim", "")), evidence, max_evidence_chars
                ),
                "negative_prompt": compose_student_verdict_prompt(
                    str(claim.get("claim", "")),
                    counterfactual_evidence if cf_valid else evidence,
                    max_evidence_chars,
                ),
                "cf_valid": cf_valid,
                "teacher_probabilities": target.tolist(),
                "teacher_target_source": target_source,
                "disagreement": disagreement,
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def make_collate(tokenizer: Any, max_length: int):
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def pack(prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [
            build_chat_prompt_ids(
                tokenizer, STUDENT_VERDICT_SYSTEM_PROMPT, prompt, max_length
            ) for prompt in prompts
        ]
        width = max(map(len, sequences))
        input_ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        attention = torch.zeros((len(sequences), width), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            input_ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention[index, :len(sequence)] = 1
        return input_ids, attention

    def collate(rows: list[dict]) -> dict:
        positive_ids, positive_attention = pack([row["positive_prompt"] for row in rows])
        negative_ids, negative_attention = pack([row["negative_prompt"] for row in rows])
        return {
            "positive_input_ids": positive_ids,
            "positive_attention_mask": positive_attention,
            "negative_input_ids": negative_ids,
            "negative_attention_mask": negative_attention,
            "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
            "teacher_probabilities": torch.tensor(
                [row["teacher_probabilities"] for row in rows], dtype=torch.float32
            ),
            "disagreement": torch.tensor(
                [row["disagreement"] for row in rows], dtype=torch.bool
            ),
            "cf_valid": torch.tensor([row["cf_valid"] for row in rows], dtype=torch.bool),
            "ids": [row["id"] for row in rows],
        }
    return collate


def final_class_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    output = model(input_ids=input_ids, attention_mask=attention).logits.float()
    final_index = attention.sum(-1) - 1
    return output[torch.arange(len(final_index), device=input_ids.device), final_index][:, token_ids]


def atomic_save_adapter(model: Any, target: Path) -> None:
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, safe_serialization=True)
    if target.exists():
        shutil.rmtree(target)
    temporary.rename(target)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[
        "matched_control", "ensemble_kd", "disagreement_kd",
        "counterfactual_only", "full",
    ], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--folds", type=Path, default=None,
                        help="Fold JSON for screening. Omit with --full-train.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--full-train", action="store_true",
                        help="Use ALL training data; validate on --val-manifest.")
    parser.add_argument("--val-manifest", type=Path, default=None,
                        help="Official validation manifest (required with --full-train)")
    parser.add_argument("--val-retrieval", type=Path, default=None,
                        help="Official validation retrieval (required with --full-train)")
    parser.add_argument("--val-corpus", type=Path, default=None,
                        help="Official validation corpus (required with --full-train)")
    parser.add_argument("--explanations", type=Path, required=True)
    parser.add_argument("--direct-teacher", type=Path, required=True)
    parser.add_argument("--grounded-teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda-kd", type=float, default=None)
    parser.add_argument("--lambda-cf", type=float, default=None)
    parser.add_argument("--disagreement-alpha", type=float, default=None)
    parser.add_argument("--cf-margin", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--max-evidence-chars", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.full_train:
        for name in ("val_manifest", "val_retrieval", "val_corpus"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required with --full-train")
    elif args.folds is None:
        parser.error("--folds is required unless --full-train is set")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    if summary_path.is_file() and json.loads(summary_path.read_text()).get("complete"):
        print(f"Cached complete B19 run: {summary_path}")
        return
    set_seed(args.seed)
    weights = resolve_variant(args.variant, {
        "lambda_kd": args.lambda_kd,
        "lambda_cf": args.lambda_cf,
        "disagreement_alpha": args.disagreement_alpha,
    })
    fold_data = json.loads(args.folds.read_text(encoding="utf-8")) if args.folds else None
    if args.full_train:
        claims = read_jsonl(args.manifest)
        train_claims = claims
        val_claims = read_jsonl(args.val_manifest)
        val_retrieval_path = args.val_retrieval
        val_corpus_path = args.val_corpus
        fold_used = "full_train"
        logging.info("Full-train mode: %d train claims, %d val claims",
                      len(train_claims), len(val_claims))
    else:
        entry = next(row for row in fold_data["folds"] if int(row["fold"]) == args.fold)
        train_ids, val_ids = set(map(str, entry["train_ids"])), set(map(str, entry["val_ids"]))
        claims = read_jsonl(args.manifest)
        train_claims = [row for row in claims if str(row["id"]) in train_ids]
        val_claims = [row for row in claims if str(row["id"]) in val_ids]
        val_retrieval_path = args.retrieval
        val_corpus_path = args.corpus
        fold_used = args.fold
    if args.limit:
        train_claims, val_claims = train_claims[:args.limit], val_claims[:args.limit]
    retrieval = {str(row["id"]): row for row in read_jsonl(args.retrieval)}
    documents = read_documents(resolve_corpus_path(args.corpus))
    val_retrieval = {str(row["id"]): row for row in read_jsonl(val_retrieval_path)}
    val_documents = read_documents(resolve_corpus_path(val_corpus_path))
    explanations = {str(row["id"]): row for row in read_jsonl(args.explanations)}
    direct = {str(row["id"]): row for row in read_jsonl(args.direct_teacher)}
    grounded = {str(row["id"]): row for row in read_jsonl(args.grounded_teacher)}

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    answer_ids = label_token_ids(tokenizer)
    answer_tensor = torch.tensor(answer_ids, device=device)
    resume_state_path = args.output / "resume_state.pt"
    resume_adapter = args.output / "resume_adapter"
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable") and device.type == "cuda":
        base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    if resume_state_path.is_file() and (resume_adapter / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(base, str(resume_adapter), is_trainable=True)
        resume_state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        start_epoch = int(resume_state["epoch"]) + 1
        best_f1 = float(resume_state["best_f1"])
        best_metrics = resume_state.get("best_metrics")
    else:
        model = get_peft_model(base, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
        resume_state, start_epoch, best_f1, best_metrics = None, 1, -1.0, None
    model.to(device)

    train_dataset = B19Dataset(
        train_claims, retrieval, documents, explanations, direct, grounded,
        args.top_k, args.max_evidence_chars,
    )
    val_dataset = B18VerifierDataset(
        val_claims, val_retrieval, val_documents, None,
        args.top_k, args.max_evidence_chars, training=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=make_collate(tokenizer, args.max_length),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_b18_collate(tokenizer, args.max_length, False, "matched_control"),
    )
    updates_per_epoch = max(1, int(np.ceil(len(train_loader) / args.grad_accum)))
    total_updates = updates_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_updates * 0.05), total_updates
    )
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
    history = list(resume_state.get("history", [])) if resume_state else []
    optimizer_updates = int(resume_state.get("optimizer_updates", 0)) if resume_state else 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        totals = Counter()
        progress = tqdm(train_loader, desc=f"{args.variant} epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress):
            positive_ids = batch["positive_input_ids"].to(device)
            positive_attention = batch["positive_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            disagreement = batch["disagreement"].to(device)
            teacher = batch["teacher_probabilities"].to(device)
            cf_valid = batch["cf_valid"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                positive_logits = final_class_logits(
                    model, positive_ids, positive_attention, answer_tensor
                )
                verdict_loss = weighted_verdict_loss(
                    positive_logits, labels, disagreement, weights["disagreement_alpha"]
                )
                kd_loss = soft_distillation_loss(
                    positive_logits, teacher, args.temperature
                ) if weights["lambda_kd"] else positive_logits.sum() * 0.0
                cf_loss = positive_logits.sum() * 0.0
                if weights["lambda_cf"] and bool(cf_valid.any()):
                    negative_ids = batch["negative_input_ids"].to(device)
                    negative_attention = batch["negative_attention_mask"].to(device)
                    negative_logits = final_class_logits(
                        model, negative_ids, negative_attention, answer_tensor
                    )
                    cf_loss = counterfactual_sufficiency_loss(
                        positive_logits, negative_logits, labels, cf_valid, args.cf_margin
                    )
                loss = verdict_loss + weights["lambda_kd"] * kd_loss + weights["lambda_cf"] * cf_loss
            (loss / args.grad_accum).backward()
            totals["loss"] += float(loss.detach())
            totals["verdict"] += float(verdict_loss.detach())
            totals["kd"] += float(kd_loss.detach())
            totals["cf"] += float(cf_loss.detach())
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                optimizer_updates += 1
            progress.set_postfix(loss=f"{totals['loss'] / (step + 1):.4f}")
        metrics = evaluate_direct_verdict(model, val_loader, answer_ids, device)
        predictions = metrics.pop("predictions")
        epoch_row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(1, len(train_loader)),
            "verdict_loss": totals["verdict"] / max(1, len(train_loader)),
            "kd_loss": totals["kd"] / max(1, len(train_loader)),
            "counterfactual_loss": totals["cf"] / max(1, len(train_loader)),
            **metrics,
        }
        history.append(epoch_row)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_metrics = metrics["macro_f1"], {**metrics, "predictions": predictions}
            atomic_save_adapter(model, args.output / "best_adapter")
        atomic_save_adapter(model, resume_adapter)
        torch.save({
            "epoch": epoch,
            "best_f1": best_f1,
            "best_metrics": best_metrics,
            "history": history,
            "optimizer_updates": optimizer_updates,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, resume_state_path)
        logging.info("epoch=%d variant=%s macro_f1=%.6f", epoch, args.variant, metrics["macro_f1"])

    if best_metrics is None:
        raise RuntimeError("B19 training produced no evaluation")
    predictions = best_metrics.pop("predictions")
    with (args.output / "val_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    payload = {
        "phase": "B19",
        "protocol": "full_train_official_val" if args.full_train else "train_only_duplicate_safe_fold_screen",
        "variant": args.variant,
        "fold": fold_used,
        "seed": args.seed,
        "complete": True,
        "best_macro_f1": best_f1,
        "final_candidate": best_metrics,
        "history": history,
        "loss_weights": weights,
        "temperature": args.temperature,
        "counterfactual_margin": args.cf_margin,
        "train_samples": len(train_dataset),
        "heldout_samples": len(val_dataset),
        "counterfactual_train_samples": train_dataset.counterfactuals,
        "teacher_target_sources": dict(train_dataset.target_sources),
        "optimizer_updates": optimizer_updates,
        "matched_optimizer_update_budget": True,
        "provenance": {
            "git_commit": git_commit(),
            "manifest_sha256": sha256(args.manifest),
            "retrieval_sha256": sha256(args.retrieval),
            "folds_sha256": sha256(args.folds) if args.folds else None,
            "explanations_sha256": sha256(args.explanations),
            "direct_teacher_sha256": sha256(args.direct_teacher),
            "grounded_teacher_sha256": sha256(args.grounded_teacher),
        },
        "official_validation_used": args.full_train,
        "test_split_used": False,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
