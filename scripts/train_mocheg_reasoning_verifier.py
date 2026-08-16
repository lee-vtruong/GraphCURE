"""Train GraphCURE-R2V: a claim-level, evidence-set MOCHEG verifier."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from graphcure.evidence_set import EvidenceSetHead, evidence_set_loss, last_token_pool


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_docs(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(
                row["evidence_id"].strip(),
                row.get("Evidence", "").replace("<p>", " ").replace("</p>", " ").strip(),
            )
    return result


class MochegEvidenceSet(Dataset):
    def __init__(
        self,
        manifest: Path,
        retrieval: Path,
        raw_csv: Path,
        tokenizer,
        top_k: int,
        claim_length: int,
        evidence_length: int,
    ) -> None:
        claims = {row["id"]: row for row in read_jsonl(manifest)}
        documents = read_docs(raw_csv)
        self.rows: list[dict] = []
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.claim_length = claim_length
        self.evidence_length = evidence_length
        for retrieval_row in read_jsonl(retrieval):
            claim = claims[retrieval_row["id"]]
            gold = {str(value) for value in claim.get("text_evidence_ids", [])}
            candidate_ids = retrieval_row.get("retrieved_evidence_ids", [])[:top_k]
            candidate_scores = retrieval_row.get("retrieved_scores", [])[:top_k]
            aligned = [
                (
                    value,
                    float(candidate_scores[index])
                    if index < len(candidate_scores) else 0.0,
                )
                for index, value in enumerate(candidate_ids)
                if documents.get(value)
            ]
            ids = [value for value, _ in aligned]
            scores = [score for _, score in aligned]
            if not ids:
                ids, scores = [""], [0.0]
            top_score = float(scores[0]) if scores else 0.0
            evidence = [documents.get(value, "") for value in ids]
            features = [
                [
                    float(scores[index]) if index < len(scores) else 0.0,
                    1.0 / (index + 1),
                    top_score - (float(scores[index]) if index < len(scores) else 0.0),
                ]
                for index in range(len(ids))
            ]
            valid = [True] * len(ids)
            relevance = [value in gold for value in ids]
            while len(ids) < top_k:
                ids.append("")
                evidence.append("")
                features.append([0.0, 0.0, 0.0])
                valid.append(False)
                relevance.append(False)
            self.rows.append({
                "id": claim["id"],
                "claim": claim.get("claim", ""),
                "evidence": evidence[:top_k],
                "label": int(claim["label"]),
                "features": features[:top_k],
                "valid": valid[:top_k],
                "relevance": relevance[:top_k],
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        claim = self.tokenizer(
            "Claim: " + row["claim"],
            padding="max_length",
            truncation=True,
            max_length=self.claim_length,
            return_tensors="pt",
        )
        evidence = self.tokenizer(
            ["Evidence: " + value for value in row["evidence"]],
            padding="max_length",
            truncation=True,
            max_length=self.evidence_length,
            return_tensors="pt",
        )
        return {
            "id": row["id"],
            "claim_input_ids": claim["input_ids"].squeeze(0),
            "claim_attention_mask": claim["attention_mask"].squeeze(0),
            "evidence_input_ids": evidence["input_ids"],
            "evidence_attention_mask": evidence["attention_mask"],
            "evidence_mask": torch.tensor(row["valid"], dtype=torch.bool),
            "retrieval_features": torch.tensor(row["features"], dtype=torch.float32),
            "relevance": torch.tensor(row["relevance"], dtype=torch.float32),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


def collate(rows: list[dict]) -> dict:
    result: dict = {"id": [row["id"] for row in rows]}
    for key in rows[0]:
        if key != "id":
            result[key] = torch.stack([row[key] for row in rows])
    return result


class ReasoningVerifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        hidden_dim: int,
        dropout: float,
        gradient_checkpointing: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name.lower()
        self.encoder = AutoModel.from_pretrained(
            encoder_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        dimension = int(self.encoder.config.hidden_size)
        self.head = EvidenceSetHead(
            encoder_dim=dimension,
            hidden_dim=hidden_dim,
            retrieval_dim=3,
            dropout=dropout,
        )

    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if any(name in self.encoder_name for name in ("qwen", "llama", "gemma", "mistral")):
            return last_token_pool(hidden, mask)
        return hidden[:, 0]

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.pool(hidden, attention_mask)

    def forward(
        self,
        claim_input_ids: torch.Tensor,
        claim_attention_mask: torch.Tensor,
        evidence_input_ids: torch.Tensor,
        evidence_attention_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
        retrieval_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, top_k, length = evidence_input_ids.shape
        claim = self.encode(claim_input_ids, claim_attention_mask)
        evidence = self.encode(
            evidence_input_ids.reshape(batch * top_k, length),
            evidence_attention_mask.reshape(batch * top_k, length),
        ).reshape(batch, top_k, -1)
        return self.head(claim, evidence, evidence_mask, retrieval_features)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    result = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins, endpoint=False),
                            np.linspace(0, 1, bins + 1)[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            result += selected.mean() * abs(
                (predictions[selected] == labels[selected]).mean()
                - confidence[selected].mean()
            )
    return float(result)


@torch.inference_mode()
def evaluate(model, dataset, batch_size, workers, device) -> tuple[dict, list[dict]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    selection_hits: list[bool] = []
    coverages: list[bool] = []
    rows: list[dict] = []
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=collate, pin_memory=device.type == "cuda",
    )
    for batch in tqdm(loader, desc="evaluate", leave=False):
        ids = batch.pop("id")
        target = batch.pop("labels")
        relevance = batch.pop("relevance")
        output = model(**{key: value.to(device) for key, value in batch.items()})
        prob = torch.softmax(output["verdict_logits"].float(), -1).cpu()
        pred = prob.argmax(-1)
        attention = output["attention"].float().cpu()
        chosen = attention.argmax(-1)
        for index, sample_id in enumerate(ids):
            has_gold = bool(relevance[index].bool().any())
            hit = bool(relevance[index, chosen[index]].item()) if has_gold else False
            coverages.append(has_gold)
            if has_gold:
                selection_hits.append(hit)
            rows.append({
                "id": sample_id,
                "gold": int(target[index]),
                "prediction": int(pred[index]),
                "probabilities": prob[index].tolist(),
                "selected_rank": int(chosen[index]) + 1,
                "gold_in_candidates": has_gold,
                "selected_gold": hit,
                "sufficiency": float(torch.sigmoid(
                    output["sufficiency_logit"][index].float()
                ).cpu()),
            })
        labels.extend(target.tolist())
        predictions.extend(pred.tolist())
        probabilities.extend(prob.tolist())
    y = np.asarray(labels)
    p = np.asarray(predictions)
    prob_array = np.asarray(probabilities)
    metrics = {
        "samples": len(labels),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "confusion_matrix": confusion_matrix(y, p).tolist(),
        "retrieval_gold_coverage": float(np.mean(coverages)),
        "evidence_selection_hit_at_1": float(np.mean(selection_hits))
        if selection_hits else 0.0,
        "ece_10": expected_calibration_error(prob_array, y),
    }
    return metrics, rows


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--encoder", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/mocheg_reasoning_verifier"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--claim-length", type=int, default=128)
    parser.add_argument("--evidence-length", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--relevance-weight", type=float, default=0.25)
    parser.add_argument("--stance-weight", type=float, default=0.15)
    parser.add_argument("--sufficiency-weight", type=float, default=0.15)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--checkpoint", type=Path,
                        help="Evaluate an existing checkpoint without retraining")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Unlock the official test split after model selection is frozen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = ReasoningVerifier(
        args.encoder,
        args.hidden_dim,
        args.dropout,
        not args.no_gradient_checkpointing,
        dtype,
    ).to(device)
    if args.checkpoint is not None:
        if not args.evaluate_test:
            parser.error("--checkpoint evaluation requires --evaluate-test")
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        if checkpoint.get("encoder") != args.encoder:
            parser.error(
                f"checkpoint encoder {checkpoint.get('encoder')!r} does not match "
                f"--encoder {args.encoder!r}"
            )
        if int(checkpoint.get("top_k", args.top_k)) != args.top_k:
            parser.error("checkpoint top_k does not match --top-k")
        model.load_state_dict(checkpoint["model"])
        test_dataset = MochegEvidenceSet(
            args.manifest_root / "test.jsonl",
            args.retrieval_root / "test.jsonl",
            args.raw_root / "test" / "Corpus2.csv",
            tokenizer,
            args.top_k,
            args.claim_length,
            args.evidence_length,
        )
        metrics, prediction_rows = evaluate(
            model, test_dataset, args.batch_size, args.num_workers, device
        )
        metrics["best_val_macro_f1"] = float(
            checkpoint.get("best_val_macro_f1", -1.0)
        )
        metrics["provenance"] = {
            "git_commit": git_commit(),
            "encoder": args.encoder,
            "retrieval_root": str(args.retrieval_root),
            "checkpoint": str(args.checkpoint),
            "seed": int(checkpoint.get("seed", args.seed)),
            "top_k": args.top_k,
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda" else "cpu",
        }
        (args.output / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        (args.output / "test_predictions.jsonl").write_text(
            "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, indent=2))
        return

    datasets = {
        split: MochegEvidenceSet(
            args.manifest_root / f"{split}.jsonl",
            args.retrieval_root / f"{split}.jsonl",
            args.raw_root / split / "Corpus2.csv",
            tokenizer,
            args.top_k,
            args.claim_length,
            args.evidence_length,
        )
        for split in ("train", "val")
    }
    counts = torch.bincount(
        torch.tensor([row["label"] for row in datasets["train"].rows]), minlength=3
    ).float()
    class_weights = (counts.sum() / (3 * counts.clamp_min(1))).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    updates_per_epoch = max(1, int(np.ceil(len(loader) / args.gradient_accumulation)))
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * args.warmup_ratio)),
        num_training_steps=total_updates,
    )
    use_amp = device.type == "cuda"
    best_f1 = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch.pop("id")
            labels = batch.pop("labels").to(device)
            relevance = batch.pop("relevance").to(device)
            evidence_mask = batch["evidence_mask"].to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with amp:
                outputs = model(**inputs)
                loss, parts = evidence_set_loss(
                    outputs,
                    labels,
                    relevance,
                    evidence_mask,
                    class_weights=class_weights,
                    relevance_weight=args.relevance_weight,
                    stance_weight=args.stance_weight,
                    sufficiency_weight=args.sufficiency_weight,
                )
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            running += float(loss.detach())
            if step % args.gradient_accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{running / step:.4f}")
        validation, _ = evaluate(
            model, datasets["val"], args.batch_size, args.num_workers, device
        )
        print(json.dumps({"epoch": epoch, "train_loss": running / len(loader),
                          **{f"val_{key}": value for key, value in validation.items()
                             if key != "confusion_matrix"}}))
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "encoder": args.encoder,
                "top_k": args.top_k,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "seed": args.seed,
                "best_val_macro_f1": best_f1,
            }, args.output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after epoch {epoch}")
                break

    checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    validation, _ = evaluate(
        model, datasets["val"], args.batch_size, args.num_workers, device
    )
    validation["best_val_macro_f1"] = best_f1
    validation["provenance"] = {
        "git_commit": git_commit(),
        "encoder": args.encoder,
        "retrieval_root": str(args.retrieval_root),
        "seed": args.seed,
        "top_k": args.top_k,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    (args.output / "val_metrics.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    if args.evaluate_test:
        test_dataset = MochegEvidenceSet(
            args.manifest_root / "test.jsonl",
            args.retrieval_root / "test.jsonl",
            args.raw_root / "test" / "Corpus2.csv",
            tokenizer,
            args.top_k,
            args.claim_length,
            args.evidence_length,
        )
        metrics, prediction_rows = evaluate(
            model, test_dataset, args.batch_size, args.num_workers, device
        )
        metrics["best_val_macro_f1"] = best_f1
        metrics["provenance"] = validation["provenance"]
        (args.output / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        (args.output / "test_predictions.jsonl").write_text(
            "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
