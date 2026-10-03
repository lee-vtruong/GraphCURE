"""Train direct or rationale-grounded expert verifiers on SciFact.

Supports:
1. --mode direct: Standard verdict-only supervision via Cross-Entropy on Prompt A.
2. --mode rationale: Multitask supervision on Prompt A (verdict) + Prompt B (grounded rationale)
   with weight lambda_exp (default: 0.25).
3. Cross-validation (--folds, --fold) for out-of-fold (OOF) threshold calibration.
4. Full-train mode (--full-train, --val-*) for final deployed models.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from graphcure.explanation import resolve_corpus_path
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_b18_explanation_verifier import (
    B18VerifierDataset,
    evaluate_direct_verdict,
    git_commit,
    make_b18_collate,
    sha256,
)
from scripts.train_mocheg_qwen3_lora_verifier import label_token_ids, read_documents


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["direct", "rationale"], required=True,
                        help="Training mode: direct (verdict only) or rationale (multitask)")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Train manifest (.jsonl)")
    parser.add_argument("--retrieval", type=Path, required=True,
                        help="Train retrieval (.jsonl)")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Evidence documents CSV")
    parser.add_argument("--explanations", type=Path, default=None,
                        help="Teacher explanations JSONL (required for rationale mode)")
    parser.add_argument("--folds", type=Path, default=None,
                        help="Folds JSON file for cross-validation")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index to use as validation set")
    parser.add_argument("--full-train", action="store_true",
                        help="Train on all data; evaluate on --val-manifest")
    parser.add_argument("--val-manifest", type=Path, default=None,
                        help="Validation manifest (required with --full-train)")
    parser.add_argument("--val-retrieval", type=Path, default=None,
                        help="Validation retrieval (required with --full-train)")
    parser.add_argument("--val-corpus", type=Path, default=None,
                        help="Validation corpus (required with --full-train)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory to save model adapter and predictions")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model identifier or path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-exp", type=float, default=0.25,
                        help="Loss weight for explanation task in rationale mode")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--max-evidence-chars", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()
    if args.mode == "rationale" and args.explanations is None:
        parser.error("--explanations is required when --mode rationale is selected")
    if args.full_train:
        for req in ("val_manifest", "val_retrieval", "val_corpus"):
            if getattr(args, req) is None:
                parser.error(f"--{req.replace('_', '-')} is required with --full-train")
    elif args.folds is None:
        parser.error("--folds is required unless --full-train is set")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    if summary_path.is_file():
        try:
            cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if cached_summary.get("complete"):
                logging.info("Cached complete SciFact expert run: %s", summary_path)
                return
        except Exception:
            pass

    set_seed(args.seed)

    # 1. Prepare data splits
    claims = read_jsonl(args.manifest)
    retrieval = {str(row["id"]): row for row in read_jsonl(args.retrieval)}
    documents = read_documents(resolve_corpus_path(args.corpus))

    if args.full_train:
        train_claims = claims
        val_claims = read_jsonl(args.val_manifest)
        val_retrieval = {str(row["id"]): row for row in read_jsonl(args.val_retrieval)}
        val_documents = read_documents(resolve_corpus_path(args.val_corpus))
        fold_used = "full_train"
        logging.info("SciFact full-train mode: %d train claims, %d val claims",
                     len(train_claims), len(val_claims))
    else:
        fold_data = json.loads(args.folds.read_text(encoding="utf-8"))
        entry = next(row for row in fold_data["folds"] if int(row["fold"]) == args.fold)
        train_ids = set(map(str, entry["train_ids"]))
        val_ids = set(map(str, entry["val_ids"]))
        train_claims = [row for row in claims if str(row["id"]) in train_ids]
        val_claims = [row for row in claims if str(row["id"]) in val_ids]
        val_retrieval = retrieval
        val_documents = documents
        fold_used = args.fold
        logging.info("SciFact K-fold mode: fold %d (%d train claims, %d val claims)",
                     args.fold, len(train_claims), len(val_claims))

    if args.limit:
        train_claims = train_claims[:args.limit]
        val_claims = val_claims[:args.limit]

    explanations = None
    if args.mode == "rationale":
        explanations = {str(row["id"]): row for row in read_jsonl(args.explanations)}
        logging.info("Loaded %d teacher explanations", len(explanations))

    # 2. Build Datasets
    train_dataset = B18VerifierDataset(
        claims=train_claims,
        retrieval_by_id=retrieval,
        documents=documents,
        explanations_by_id=explanations,
        top_k=args.top_k,
        max_evidence_chars=args.max_evidence_chars,
        training=True,
    )
    val_dataset = B18VerifierDataset(
        claims=val_claims,
        retrieval_by_id=val_retrieval,
        documents=val_documents,
        explanations_by_id=None,
        top_k=args.top_k,
        max_evidence_chars=args.max_evidence_chars,
        training=False,
    )

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    answer_ids = label_token_ids(tokenizer)

    collate_mode = "explanation_candidate" if args.mode == "rationale" else "matched_control"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_b18_collate(tokenizer, args.max_length, True, collate_mode),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_b18_collate(tokenizer, args.max_length, False, "matched_control"),
    )

    logging.info("Loading base model: %s", args.model)
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

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora_config)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Trainable LoRA parameters: %d", trainable_params)

    updates_per_epoch = max(1, int(np.ceil(len(train_loader) / args.grad_accum)))
    total_updates = updates_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_updates)),
        num_training_steps=total_updates,
    )

    best_adapter_dir = args.output / "best_adapter"
    best_f1 = -1.0
    best_metrics = None
    history = []
    optimizer_updates = 0

    logging.info("Starting SciFact expert training (%s mode, %d epochs)...", args.mode, args.epochs)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_v_loss = 0.0
        epoch_e_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            v_inputs = batch["verdict_input_ids"].to(device, non_blocking=True)
            v_attention = batch["verdict_attention_mask"].to(device, non_blocking=True)
            v_labels = batch["verdict_labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(input_ids=v_inputs, attention_mask=v_attention, labels=v_labels)
                verdict_loss = outputs.loss
            (verdict_loss / args.grad_accum).backward()
            epoch_v_loss += verdict_loss.item()

            if batch.get("has_explanation"):
                e_inputs = batch["exp_input_ids"].to(device, non_blocking=True)
                e_attention = batch["exp_attention_mask"].to(device, non_blocking=True)
                e_labels = batch["exp_labels"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    e_outputs = model(input_ids=e_inputs, attention_mask=e_attention, labels=e_labels)
                    exp_loss = e_outputs.loss
                ((args.lambda_exp * exp_loss) / args.grad_accum).backward()
                epoch_e_loss += exp_loss.item()

            step_count += 1
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_updates += 1

            pbar.set_postfix({
                "v_loss": f"{epoch_v_loss / max(1, step_count):.4f}",
                "e_loss": f"{epoch_e_loss / max(1, step_count):.4f}" if args.mode == "rationale" else "0.0000",
            })

        # Evaluate on held-out validation set
        val_metrics = evaluate_direct_verdict(model, val_loader, answer_ids, device)
        val_f1 = val_metrics["macro_f1"]
        logging.info("Epoch %d validation: Macro-F1=%.5f, Acc=%.5f, Supp=%.5f, Ref=%.5f, NEI=%.5f",
                     epoch, val_f1, val_metrics["accuracy"],
                     val_metrics["f1_supported"], val_metrics["f1_refuted"], val_metrics["f1_nei"])

        history.append({
            "epoch": epoch,
            "verdict_loss": epoch_v_loss / max(1, step_count),
            "explanation_loss": epoch_e_loss / max(1, step_count),
            "macro_f1": val_f1,
            "accuracy": val_metrics["accuracy"],
            "f1_supported": val_metrics["f1_supported"],
            "f1_refuted": val_metrics["f1_refuted"],
            "f1_nei": val_metrics["f1_nei"],
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = val_metrics
            best_adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_adapter_dir))
            logging.info("--> New best model saved at Epoch %d (Macro-F1=%.5f)", epoch, best_f1)

    if best_metrics is None:
        raise RuntimeError("SciFact training failed to produce an evaluation")

    # Save predictions
    val_pred_rows = best_metrics.pop("predictions")
    with (args.output / "val_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in val_pred_rows:
            handle.write(json.dumps(row) + "\n")

    summary_payload = {
        "phase": "SciFact-Adaptation-A",
        "mode": args.mode,
        "fold": fold_used,
        "seed": args.seed,
        "complete": True,
        "best_macro_f1": best_f1,
        "final_metrics": best_metrics,
        "history": history,
        "train_samples": len(train_dataset),
        "heldout_samples": len(val_dataset),
        "optimizer_updates": optimizer_updates,
        "lambda_exp": args.lambda_exp if args.mode == "rationale" else 0.0,
        "provenance": {
            "git_commit": git_commit(),
            "manifest_sha256": sha256(args.manifest),
            "retrieval_sha256": sha256(args.retrieval),
            "folds_sha256": sha256(args.folds) if args.folds else None,
            "explanations_sha256": sha256(args.explanations) if args.explanations else None,
        },
        "dev_labels_used_for_training": False,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    logging.info("Saved final summary to %s", summary_path)
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
