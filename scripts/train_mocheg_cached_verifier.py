"""Train the GraphCURE-R2V evidence-set head over frozen cached embeddings."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from graphcure.evidence_set import EvidenceSetHead, evidence_set_loss


class CachedEvidenceDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "ids", "claim_embeddings", "evidence_embeddings", "evidence_mask",
            "retrieval_features", "relevance", "relevance_weights", "labels",
            "metadata",
        }
        missing = required - self.data.keys()
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        size = len(self.data["labels"])
        if any(len(self.data[key]) != size for key in required - {"metadata"}):
            raise ValueError(f"unaligned cached tensors in {path}")

    @property
    def metadata(self) -> dict:
        return self.data["metadata"]

    def __len__(self) -> int:
        return len(self.data["labels"])

    def __getitem__(self, index: int) -> dict:
        return {
            "id": self.data["ids"][index],
            "claim": self.data["claim_embeddings"][index],
            "evidence": self.data["evidence_embeddings"][index],
            "evidence_mask": self.data["evidence_mask"][index],
            "retrieval_features": self.data["retrieval_features"][index],
            "relevance": self.data["relevance"][index],
            "relevance_weights": self.data["relevance_weights"][index],
            "labels": self.data["labels"][index],
        }


def collate(rows: list[dict]) -> dict:
    result = {"id": [row["id"] for row in rows]}
    for key in rows[0]:
        if key != "id":
            result[key] = torch.stack([row[key] for row in rows])
    return result


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        chosen = (confidence > lower) & (confidence <= upper)
        if chosen.any():
            result += chosen.mean() * abs(
                (predictions[chosen] == labels[chosen]).mean()
                - confidence[chosen].mean()
            )
    return float(result)


@torch.inference_mode()
def evaluate(head, dataset, batch_size, device) -> tuple[dict, list[dict]]:
    head.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    coverage: list[bool] = []
    selection_hits: list[bool] = []
    prediction_rows: list[dict] = []
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    for batch in tqdm(loader, desc="evaluate", leave=False):
        ids = batch.pop("id")
        target = batch.pop("labels")
        relevance = batch.pop("relevance")
        batch.pop("relevance_weights")
        output = head(**{key: value.to(device) for key, value in batch.items()})
        prob = torch.softmax(output["verdict_logits"].float(), -1).cpu()
        pred = prob.argmax(-1)
        selected = output["attention"].float().cpu().argmax(-1)
        sufficiency = torch.sigmoid(output["sufficiency_logit"].float()).cpu()
        for index, sample_id in enumerate(ids):
            has_gold = bool(relevance[index].bool().any())
            selected_gold = bool(relevance[index, selected[index]]) if has_gold else False
            coverage.append(has_gold)
            if has_gold:
                selection_hits.append(selected_gold)
            prediction_rows.append({
                "id": sample_id,
                "gold": int(target[index]),
                "prediction": int(pred[index]),
                "probabilities": prob[index].tolist(),
                "selected_rank": int(selected[index]) + 1,
                "gold_in_candidates": has_gold,
                "selected_gold": selected_gold,
                "sufficiency": float(sufficiency[index]),
            })
        labels.extend(target.tolist())
        predictions.extend(pred.tolist())
        probabilities.extend(prob.tolist())
    y = np.asarray(labels)
    pred = np.asarray(predictions)
    prob = np.asarray(probabilities)
    return {
        "samples": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "retrieval_gold_coverage": float(np.mean(coverage)),
        "evidence_selection_hit_at_1": float(np.mean(selection_hits))
        if selection_hits else 0.0,
        "ece_10": expected_calibration_error(prob, y),
    }, prediction_rows


def validate_cache_pair(train: CachedEvidenceDataset, val: CachedEvidenceDataset) -> None:
    for key in ("encoder", "embedding_dim", "top_k", "max_length"):
        if train.metadata.get(key) != val.metadata.get(key):
            raise ValueError(
                f"train/val cache mismatch for {key}: "
                f"{train.metadata.get(key)!r} != {val.metadata.get(key)!r}"
            )


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/mocheg_reasoning_cache"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/mocheg_cached_verifier_seed42"))
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--relevance-weight", type=float, default=0.25)
    parser.add_argument("--stance-weight", type=float, default=0.15)
    parser.add_argument("--sufficiency-weight", type=float, default=0.15)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.checkpoint is not None:
        if not args.evaluate_test:
            parser.error("--checkpoint requires --evaluate-test")
        test = CachedEvidenceDataset(args.cache_root / "test.pt")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["cache_metadata"]["encoder"] != test.metadata["encoder"]:
            parser.error("checkpoint and test cache use different encoders")
        head = EvidenceSetHead(
            encoder_dim=int(test.metadata["embedding_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            retrieval_dim=6,
            dropout=float(checkpoint["dropout"]),
        ).to(device)
        head.load_state_dict(checkpoint["head"])
        metrics, rows = evaluate(head, test, args.batch_size, device)
        metrics["best_val_macro_f1"] = float(checkpoint["best_val_macro_f1"])
        metrics["provenance"] = {
            "git_commit": git_commit(),
            "checkpoint": str(args.checkpoint),
            "cache_metadata": test.metadata,
            "seed": int(checkpoint["seed"]),
        }
        (args.output / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        (args.output / "test_predictions.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2))
        return

    train = CachedEvidenceDataset(args.cache_root / "train.pt")
    val = CachedEvidenceDataset(args.cache_root / "val.pt")
    validate_cache_pair(train, val)
    head = EvidenceSetHead(
        encoder_dim=int(train.metadata["embedding_dim"]),
        hidden_dim=args.hidden_dim,
        retrieval_dim=6,
        dropout=args.dropout,
    ).to(device)
    counts = torch.bincount(train.data["labels"], minlength=3).float()
    class_weights = (counts.sum() / (3 * counts.clamp_min(1))).to(device)
    loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_f1 = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        running = 0.0
        parts_sum = {key: 0.0 for key in ("verdict", "relevance", "stance", "sufficiency")}
        for batch in loader:
            batch.pop("id")
            labels = batch.pop("labels").to(device)
            relevance = batch.pop("relevance").to(device)
            relevance_weights = batch.pop("relevance_weights").to(device)
            evidence_mask = batch["evidence_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = head(**{key: value.to(device) for key, value in batch.items()})
            loss, parts = evidence_set_loss(
                output,
                labels,
                relevance,
                evidence_mask,
                relevance_weights=relevance_weights,
                class_weights=class_weights,
                relevance_weight=args.relevance_weight,
                stance_weight=args.stance_weight,
                sufficiency_weight=args.sufficiency_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            for key, value in parts.items():
                parts_sum[key] += float(value)
        validation, _ = evaluate(head, val, args.batch_size, device)
        row = {
            "epoch": epoch,
            "train_loss": running / len(loader),
            **{f"train_{key}_loss": value / len(loader)
               for key, value in parts_sum.items()},
            **{f"val_{key}": value for key, value in validation.items()
               if key != "confusion_matrix"},
        }
        print(json.dumps(row))
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            stale = 0
            torch.save({
                "head": head.state_dict(),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "seed": args.seed,
                "best_val_macro_f1": best_f1,
                "cache_metadata": train.metadata,
            }, args.output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after epoch {epoch}")
                break

    checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(checkpoint["head"])
    head.to(device)
    metrics, rows = evaluate(head, val, args.batch_size, device)
    metrics["best_val_macro_f1"] = best_f1
    metrics["provenance"] = {
        "git_commit": git_commit(),
        "cache_metadata": val.metadata,
        "seed": args.seed,
        "checkpoint": str(args.output / "best.pt"),
    }
    (args.output / "val_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "val_predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
