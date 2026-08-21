"""Fine-tune a long-context claim-level verifier on retrieved MOCHEG evidence."""
from __future__ import annotations

import argparse
import csv
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


LABEL_NAMES = {0: "supported", 1: "refuted", 2: "not enough information"}


def read_documents(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            evidence_id = row.get("evidence_id", "").strip()
            evidence = row.get("Evidence", "").replace("<p>", " ").replace(
                "</p>", " ").strip()
            if evidence_id and evidence:
                result.setdefault(evidence_id, evidence)
    return result


def compose_example(claim: str, evidence: list[str], reports: list[str] | None,
                    max_evidence_chars: int) -> str:
    """Compose model input without labels, qrel flags, or gold identifiers."""
    sections = [
        "Task: Determine whether the claim is supported, refuted, or lacks "
        "enough information using only the evidence below.",
        f"Claim:\n{claim.strip()}",
    ]
    for index, text in enumerate(evidence, 1):
        sections.append(
            f"Retrieved text evidence {index}:\n{text[:max_evidence_chars].strip()}")
    for index, report in enumerate(reports or [], 1):
        sections.append(
            f"Retrieved visual evidence report {index}:\n{report.strip()}")
    sections.append("Verdict:")
    return "\n\n".join(sections)


class LongContextDataset(Dataset):
    def __init__(self, manifest: Path, retrieval: Path, corpus: Path,
                 top_k: int, max_evidence_chars: int,
                 reports: Path | None = None, limit: int = 0):
        claims = read_jsonl(manifest)
        retrieval_by_id = {row["id"]: row for row in read_jsonl(retrieval)}
        report_by_id = ({row["id"]: row for row in read_jsonl(reports)}
                        if reports is not None else {})
        documents = read_documents(corpus)
        if limit:
            claims = claims[:limit]
        self.rows = []
        for claim in claims:
            retrieved = retrieval_by_id.get(claim["id"])
            if retrieved is None:
                raise ValueError(f"retrieval missing {claim['id']}")
            document_ids = retrieved.get("retrieved_evidence_ids", [])[:top_k]
            evidence = [documents[value] for value in document_ids
                        if value in documents]
            visual = report_by_id.get(claim["id"], {}).get("reports", [])
            self.rows.append({
                "id": claim["id"], "label": int(claim["label"]),
                "text": compose_example(claim.get("claim", ""), evidence,
                                        visual, max_evidence_chars),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_collate(tokenizer, max_length: int):
    def collate(rows):
        encoded = tokenizer(
            [row["text"] for row in rows], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt")
        return {"encoded": encoded,
                "labels": torch.tensor([row["label"] for row in rows]),
                "ids": [row["id"] for row in rows]}
    return collate


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); labels, probabilities = [], []
    for batch in tqdm(loader, desc="validation", leave=False):
        encoded = {key: value.to(device, non_blocking=True)
                   for key, value in batch["encoded"].items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(**encoded).logits
        labels.extend(batch["labels"].tolist())
        probabilities.extend(torch.softmax(logits.float(), -1).cpu().tolist())
    y = np.asarray(labels); probability = np.asarray(probabilities)
    prediction = probability.argmax(-1)
    return {
        "samples": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(y, prediction).tolist(),
        "ece_10": expected_calibration_error(probability, y),
        "label_names": LABEL_NAMES,
    }


def main() -> None:
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict"))
    parser.add_argument("--retrieval-root", type=Path, default=Path(
        "outputs/retrieval_mocheg_qwen3_reranked"))
    parser.add_argument("--raw-root", type=Path, default=Path(
        "data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--model", default="answerdotai/ModernBERT-large")
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_modernbert_text_seed42_v12"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-evidence-chars", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--warmup-ratio", type=float, default=.1)
    parser.add_argument("--label-smoothing", type=float, default=.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    args = parser.parse_args()
    if min(args.top_k, args.max_length, args.batch_size,
           args.gradient_accumulation, args.epochs) <= 0:
        parser.error("top-k, lengths, batch sizes and epochs must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3, torch_dtype=(
            torch.bfloat16 if device.type == "cuda" else torch.float32),
        id2label=LABEL_NAMES,
        label2id={value: key for key, value in LABEL_NAMES.items()},
    ).to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    train_reports = (args.report_root / "train.jsonl"
                     if args.report_root is not None else None)
    val_reports = (args.report_root / "val.jsonl"
                   if args.report_root is not None else None)
    train = LongContextDataset(
        args.manifest_root / "train.jsonl",
        args.retrieval_root / "train.jsonl",
        args.raw_root / "train" / "Corpus2.csv", args.top_k,
        args.max_evidence_chars, train_reports, args.limit_train)
    val = LongContextDataset(
        args.manifest_root / "val.jsonl",
        args.retrieval_root / "val.jsonl",
        args.raw_root / "val" / "Corpus2.csv", args.top_k,
        args.max_evidence_chars, val_reports, args.limit_val)
    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    updates_per_epoch = max(1, int(np.ceil(
        len(train_loader) / args.gradient_accumulation)))
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * args.warmup_ratio), total_updates)
    counts = torch.bincount(torch.tensor([row["label"] for row in train.rows]),
                            minlength=3).float()
    class_weights = (counts.sum() / (3 * counts.clamp_min(1))).to(device)
    best, stale, history = -1.0, 0, []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, 1):
            encoded = {key: value.to(device, non_blocking=True)
                       for key, value in batch["encoded"].items()}
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(**encoded).logits
                loss = F.cross_entropy(
                    logits.float(), labels, weight=class_weights,
                    label_smoothing=args.label_smoothing)
                scaled = loss / args.gradient_accumulation
            scaled.backward(); running += float(loss.detach())
            update = (step % args.gradient_accumulation == 0
                      or step == len(train_loader))
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        validation = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / len(train_loader),
               **validation}
        history.append(row); print(json.dumps(row), flush=True)
        if validation["macro_f1"] > best:
            best, stale = validation["macro_f1"], 0
            best_root = args.output / "best"
            temporary = args.output / "best.tmp"
            if temporary.exists(): shutil.rmtree(temporary)
            model.save_pretrained(temporary, safe_serialization=True)
            tokenizer.save_pretrained(temporary)
            if best_root.exists(): shutil.rmtree(best_root)
            temporary.replace(best_root)
            (args.output / "val_metrics.json").write_text(
                json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience: break
    summary = {
        "model": args.model, "modality": "text+visual-reports"
        if args.report_root is not None else "text-only",
        "best_val_macro_f1": best,
        "frozen_embedding_anchor_macro_f1": 0.5499479090475745,
        "delta_vs_frozen_embedding_anchor": best - 0.5499479090475745,
        "history": history, "test_split_used": False,
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
