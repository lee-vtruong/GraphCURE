"""Train validation-only GraphCURE-R2V text/visual evidence verifier."""
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

from graphcure.multimodal_evidence import (
    MultimodalEvidenceHead,
    multimodal_evidence_loss,
)
from scripts.train_mocheg_cached_verifier import expected_calibration_error


class MultimodalEvidenceDataset(Dataset):
    MODEL_FIELDS = {
        "claim": "claim_embeddings",
        "text_evidence": "text_evidence_embeddings",
        "text_mask": "text_mask",
        "text_retrieval_features": "text_retrieval_features",
        "visual_evidence": "visual_evidence_embeddings",
        "visual_mask": "visual_mask",
        "visual_retrieval_features": "visual_retrieval_features",
    }
    MODEL_KEYS = tuple(MODEL_FIELDS)
    SUPERVISION_KEYS = (
        "text_relevance",
        "text_relevance_weights",
        "visual_relevance",
        "visual_relevance_weights",
        "labels",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "ids", "metadata", *self.MODEL_FIELDS.values(),
            *self.SUPERVISION_KEYS,
        }
        missing = required - self.data.keys()
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        size = len(self.data["ids"])
        if any(len(self.data[key]) != size for key in required - {"metadata"}):
            raise ValueError(f"unaligned multimodal cache tensors in {path}")

    @property
    def metadata(self) -> dict:
        return self.data["metadata"]

    def __len__(self) -> int:
        return len(self.data["ids"])

    def __getitem__(self, index: int) -> dict:
        row = {"id": self.data["ids"][index]}
        for output_key, cache_key in self.MODEL_FIELDS.items():
            row[output_key] = self.data[cache_key][index]
        for key in self.SUPERVISION_KEYS:
            row[key] = self.data[key][index]
        return row


def collate(rows: list[dict]) -> dict:
    return {
        key: ([row[key] for row in rows] if key == "id" else torch.stack(
            [row[key] for row in rows]
        ))
        for key in rows[0]
    }


def validate_cache_pair(train: MultimodalEvidenceDataset,
                        val: MultimodalEvidenceDataset) -> None:
    for key in (
        "claim_dim", "text_dim", "visual_dim", "text_top_k",
        "visual_top_k", "visual_model",
    ):
        if train.metadata.get(key) != val.metadata.get(key):
            raise ValueError(f"train/val multimodal cache mismatch for {key}")
    if not train.metadata.get("train_gold_injection"):
        raise ValueError("training cache must inject train-only visual positives")
    if val.metadata.get("validation_gold_injection"):
        raise ValueError("validation cache must never inject gold visual evidence")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@torch.inference_mode()
def evaluate(head: MultimodalEvidenceHead, dataset: MultimodalEvidenceDataset,
             batch_size: int, device: torch.device) -> tuple[dict, list[dict]]:
    head.eval()
    labels: list[int] = []
    predictions: list[int] = []
    text_predictions: list[int] = []
    probabilities: list[list[float]] = []
    text_coverage: list[bool] = []
    visual_coverage: list[bool] = []
    text_hits: list[bool] = []
    visual_hits: list[bool] = []
    visual_mass: list[float] = []
    conflicts: list[float] = []
    gates_with_gold: list[float] = []
    gates_without_gold: list[float] = []
    rows: list[dict] = []
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    for batch in tqdm(loader, desc="evaluate", leave=False):
        ids = batch.pop("id")
        target = batch.pop("labels")
        text_relevance = batch.pop("text_relevance")
        batch.pop("text_relevance_weights")
        visual_relevance = batch.pop("visual_relevance")
        batch.pop("visual_relevance_weights")
        output = head(**{
            key: value.to(device) for key, value in batch.items()
        })
        probability = torch.softmax(
            output["verdict_logits"].float(), dim=-1
        ).cpu()
        prediction = probability.argmax(-1)
        text_prediction = output["text_verdict_logits"].float().cpu().argmax(-1)
        text_selected = output["text_attention"].cpu().argmax(-1)
        visual_selected = output["visual_attention"].cpu().argmax(-1)
        modality_mass = output["modality_mass"].float().cpu()
        conflict = output["conflict"].float().cpu().mean(-1)
        sufficiency = torch.sigmoid(
            output["sufficiency_logit"].float()
        ).cpu()
        for index, sample_id in enumerate(ids):
            has_text = bool(text_relevance[index].bool().any())
            has_visual = bool(visual_relevance[index].bool().any())
            text_coverage.append(has_text)
            visual_coverage.append(has_visual)
            if has_text:
                text_hits.append(bool(
                    text_relevance[index, text_selected[index]]
                ))
            if has_visual:
                visual_hits.append(bool(
                    visual_relevance[index, visual_selected[index]]
                ))
            visual_mass.append(float(modality_mass[index, 1]))
            if has_visual:
                gates_with_gold.append(float(modality_mass[index, 1]))
            else:
                gates_without_gold.append(float(modality_mass[index, 1]))
            conflicts.append(float(conflict[index]))
            rows.append({
                "id": sample_id,
                "gold": int(target[index]),
                "prediction": int(prediction[index]),
                "text_only_prediction": int(text_prediction[index]),
                "probabilities": probability[index].tolist(),
                "text_gold_in_candidates": has_text,
                "visual_gold_in_candidates": has_visual,
                "text_selected_gold": bool(
                    text_relevance[index, text_selected[index]]
                ) if has_text else False,
                "visual_selected_gold": bool(
                    visual_relevance[index, visual_selected[index]]
                ) if has_visual else False,
                "visual_modality_mass": float(modality_mass[index, 1]),
                "constraint_conflict": float(conflict[index]),
                "sufficiency": float(sufficiency[index]),
            })
        labels.extend(target.tolist())
        predictions.extend(prediction.tolist())
        text_predictions.extend(text_prediction.tolist())
        probabilities.extend(probability.tolist())
    y = np.asarray(labels)
    pred = np.asarray(predictions)
    text_pred = np.asarray(text_predictions)
    prob = np.asarray(probabilities)
    return {
        "samples": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "text_only_accuracy": float(accuracy_score(y, text_pred)),
        "text_only_macro_f1": float(f1_score(y, text_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "text_gold_coverage": float(np.mean(text_coverage)),
        "visual_gold_coverage": float(np.mean(visual_coverage)),
        "text_selection_hit_at_1": float(np.mean(text_hits)) if text_hits else 0.0,
        "visual_selection_hit_at_1": float(np.mean(visual_hits))
        if visual_hits else 0.0,
        "visual_modality_mass_mean": float(np.mean(visual_mass)),
        "visual_gate_gold_mean": float(np.mean(gates_with_gold))
        if gates_with_gold else 0.0,
        "visual_gate_non_gold_mean": float(np.mean(gates_without_gold))
        if gates_without_gold else 0.0,
        "visual_help_rate": float(np.mean((text_pred != y) & (pred == y))),
        "visual_harm_rate": float(np.mean((text_pred == y) & (pred != y))),
        "constraint_conflict_mean": float(np.mean(conflicts)),
        "ece_10": expected_calibration_error(prob, y),
    }, rows


def build_head(metadata: dict, args: argparse.Namespace) -> MultimodalEvidenceHead:
    return MultimodalEvidenceHead(
        claim_dim=int(metadata["claim_dim"]),
        text_dim=int(metadata["text_dim"]),
        visual_dim=int(metadata["visual_dim"]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )


TEXT_TEACHER_PREFIXES = {
    "claim_projection.": "claim_projection.",
    "evidence_projection.": "text_projection.",
    "utility.": "text_utility.",
    "stance.": "text_stance.",
    "sufficiency.": "text_sufficiency.",
    "verdict.": "text_verdict.",
}


def load_text_teacher(head: MultimodalEvidenceHead,
                      checkpoint_path: Path) -> dict:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    source = checkpoint.get("head")
    if not isinstance(source, dict):
        raise ValueError(f"{checkpoint_path} does not contain an evidence head")
    destination = head.state_dict()
    transferred = {}
    for source_key, value in source.items():
        for old_prefix, new_prefix in TEXT_TEACHER_PREFIXES.items():
            if source_key.startswith(old_prefix):
                target_key = new_prefix + source_key[len(old_prefix):]
                if target_key not in destination:
                    raise ValueError(f"missing text-teacher target: {target_key}")
                if destination[target_key].shape != value.shape:
                    raise ValueError(
                        f"text-teacher shape mismatch for {target_key}: "
                        f"{tuple(value.shape)} != {tuple(destination[target_key].shape)}"
                    )
                transferred[target_key] = value
                break
    expected = {
        key for key in destination
        if any(key.startswith(prefix) for prefix in TEXT_TEACHER_PREFIXES.values())
    }
    if set(transferred) != expected:
        missing = sorted(expected - set(transferred))
        raise ValueError(f"incomplete text-teacher transfer: {missing}")
    head.load_state_dict(transferred, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "seed": checkpoint.get("seed"),
        "best_val_macro_f1": checkpoint.get("best_val_macro_f1"),
        "transferred_tensors": len(transferred),
    }


def text_branch_modules(head: MultimodalEvidenceHead) -> tuple[torch.nn.Module, ...]:
    return (
        head.claim_projection,
        head.text_projection,
        head.text_utility,
        head.text_stance,
        head.text_sufficiency,
        head.text_verdict,
    )


def freeze_text_branch(head: MultimodalEvidenceHead) -> None:
    for module in text_branch_modules(head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False


def training_objective(
    head: MultimodalEvidenceHead,
    batch: dict,
    device: torch.device,
    class_weights: torch.Tensor,
    relevance_weight: float = 0.25,
    stance_weight: float = 0.15,
    sufficiency_weight: float = 0.15,
    conflict_weight: float = 0.05,
    gate_weight: float = 0.10,
    visual_gate_target: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one training objective without dropping model masks."""
    batch = dict(batch)
    batch.pop("id", None)
    labels = batch.pop("labels").to(device)
    supervision = {
        key: batch.pop(key).to(device)
        for key in MultimodalEvidenceDataset.SUPERVISION_KEYS
        if key != "labels"
    }
    model_batch = {
        key: value.to(device) for key, value in batch.items()
    }
    output = head(**model_batch)
    return multimodal_evidence_loss(
        output,
        labels,
        text_mask=model_batch["text_mask"],
        visual_mask=model_batch["visual_mask"],
        **supervision,
        class_weights=class_weights,
        relevance_weight=relevance_weight,
        stance_weight=stance_weight,
        sufficiency_weight=sufficiency_weight,
        conflict_weight=conflict_weight,
        gate_weight=gate_weight,
        visual_gate_target=visual_gate_target,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/mocheg_multimodal_cache"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/mocheg_multimodal_verifier_seed42"))
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--relevance-weight", type=float, default=0.25)
    parser.add_argument("--stance-weight", type=float, default=0.15)
    parser.add_argument("--sufficiency-weight", type=float, default=0.15)
    parser.add_argument("--conflict-weight", type=float, default=0.05)
    parser.add_argument("--gate-weight", type=float, default=0.10)
    parser.add_argument("--visual-gate-target", type=float, default=0.25)
    parser.add_argument("--text-checkpoint", type=Path)
    parser.add_argument("--freeze-text-branch", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    train = MultimodalEvidenceDataset(args.cache_root / "train.pt")
    val = MultimodalEvidenceDataset(args.cache_root / "val.pt")
    validate_cache_pair(train, val)
    head = build_head(train.metadata, args).to(device)
    teacher = None
    if args.text_checkpoint is not None:
        teacher = load_text_teacher(head, args.text_checkpoint)
    if args.freeze_text_branch:
        if args.text_checkpoint is None:
            parser.error("--freeze-text-branch requires --text-checkpoint")
        freeze_text_branch(head)
    counts = torch.bincount(train.data["labels"], minlength=3).float()
    class_weights = (counts.sum() / (3 * counts.clamp_min(1))).to(device)
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_f1 = -1.0
    stale = 0
    loss_keys = (
        "verdict", "relevance", "stance", "sufficiency", "conflict", "gate"
    )
    for epoch in range(1, args.epochs + 1):
        head.train()
        if args.freeze_text_branch:
            freeze_text_branch(head)
        running = 0.0
        part_sums = {key: 0.0 for key in loss_keys}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = training_objective(
                head,
                batch,
                device,
                class_weights=class_weights,
                relevance_weight=args.relevance_weight,
                stance_weight=args.stance_weight,
                sufficiency_weight=args.sufficiency_weight,
                conflict_weight=args.conflict_weight,
                gate_weight=args.gate_weight,
                visual_gate_target=args.visual_gate_target,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            for key, value in parts.items():
                part_sums[key] += float(value)
        validation, _ = evaluate(head, val, args.batch_size, device)
        row = {
            "epoch": epoch,
            "train_loss": running / len(loader),
            **{f"train_{key}_loss": value / len(loader)
               for key, value in part_sums.items()},
            **{f"val_{key}": value for key, value in validation.items()
               if key != "confusion_matrix"},
        }
        print(json.dumps(row), flush=True)
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
                "loss_weights": {
                    "relevance": args.relevance_weight,
                    "stance": args.stance_weight,
                    "sufficiency": args.sufficiency_weight,
                    "conflict": args.conflict_weight,
                    "gate": args.gate_weight,
                    "visual_gate_target": args.visual_gate_target,
                },
                "text_teacher": teacher,
                "text_branch_frozen": args.freeze_text_branch,
            }, args.output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after epoch {epoch}")
                break
    checkpoint = torch.load(
        args.output / "best.pt", map_location="cpu", weights_only=False
    )
    head.load_state_dict(checkpoint["head"])
    head.to(device)
    metrics, rows = evaluate(head, val, args.batch_size, device)
    metrics["best_val_macro_f1"] = best_f1
    metrics["provenance"] = {
        "git_commit": git_commit(),
        "cache_metadata": val.metadata,
        "seed": args.seed,
        "checkpoint": str(args.output / "best.pt"),
        "test_split_used": False,
        "text_teacher": teacher,
        "text_branch_frozen": args.freeze_text_branch,
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
