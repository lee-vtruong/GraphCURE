"""Cross-fitted set-level utility router for frozen MOCHEG experts.

The router predicts whether the frozen visual expert will help, be neutral, or
harm the frozen text anchor. Its routing threshold is selected exclusively on
out-of-fold train predictions and is then frozen for validation evaluation.
No validation labels participate in fitting or threshold selection.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphcure.multimodal_evidence import MultimodalEvidenceHead
from scripts.audit_mocheg_router import audit
from scripts.train_mocheg_multimodal_verifier import (
    MultimodalEvidenceDataset,
    collate,
)
from scripts.train_mocheg_staged_multimodal import validate_router_cache


UTILITY_HARMFUL = 0
UTILITY_NEUTRAL = 1
UTILITY_HELPFUL = 2


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def probability_features(
    text_probability: torch.Tensor,
    expert_probability: torch.Tensor,
) -> tuple[torch.Tensor, list[str]]:
    def entropy(value: torch.Tensor) -> torch.Tensor:
        return -torch.sum(value * torch.log(value.clamp_min(1e-8)), dim=-1)

    def margin(value: torch.Tensor) -> torch.Tensor:
        top = torch.topk(value, k=2, dim=-1).values
        return top[:, 0] - top[:, 1]

    values = torch.cat((
        text_probability,
        expert_probability,
        torch.abs(text_probability - expert_probability),
        entropy(text_probability).unsqueeze(-1),
        entropy(expert_probability).unsqueeze(-1),
        text_probability.max(-1).values.unsqueeze(-1),
        expert_probability.max(-1).values.unsqueeze(-1),
        margin(text_probability).unsqueeze(-1),
        margin(expert_probability).unsqueeze(-1),
        (text_probability.argmax(-1) != expert_probability.argmax(-1))
        .float().unsqueeze(-1),
    ), dim=-1)
    names = (
        [f"text_probability_{index}" for index in range(3)]
        + [f"visual_probability_{index}" for index in range(3)]
        + [f"probability_gap_{index}" for index in range(3)]
        + [
            "text_entropy", "visual_entropy", "text_confidence",
            "visual_confidence", "text_margin", "visual_margin",
            "expert_disagreement",
        ]
    )
    return values, names


def attention_features(
    attention: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
) -> tuple[torch.Tensor, list[str]]:
    count = mask.float().sum(-1).clamp_min(1.0)
    entropy = -torch.sum(
        attention * torch.log(attention.clamp_min(1e-8)), dim=-1
    ) / torch.log(count.clamp_min(2.0))
    top = torch.topk(
        attention.masked_fill(~mask.bool(), -1.0),
        k=min(2, attention.shape[1]),
        dim=-1,
    ).values
    if top.shape[1] == 1:
        concentration_margin = top[:, 0]
    else:
        concentration_margin = top[:, 0] - top[:, 1]
    values = torch.stack((
        count,
        attention.max(-1).values,
        entropy,
        concentration_margin,
    ), dim=-1)
    names = [
        f"{prefix}_candidate_count",
        f"{prefix}_attention_max",
        f"{prefix}_attention_entropy",
        f"{prefix}_attention_margin",
    ]
    return values, names


def retrieval_features(
    values: torch.Tensor,
    mask: torch.Tensor,
    attention: torch.Tensor,
    prefix: str,
) -> tuple[torch.Tensor, list[str]]:
    mask_float = mask.float()
    count = mask_float.sum(-1).clamp_min(1.0)
    masked = values.float().masked_fill(~mask.bool().unsqueeze(-1), -1e9)
    maximum = masked.max(1).values
    maximum = torch.where(maximum < -1e8, torch.zeros_like(maximum), maximum)
    mean = torch.sum(values.float() * mask_float.unsqueeze(-1), dim=1)
    mean = mean / count.unsqueeze(-1)
    variance = torch.sum(
        (values.float() - mean.unsqueeze(1)).square()
        * mask_float.unsqueeze(-1),
        dim=1,
    ) / count.unsqueeze(-1)
    weighted = torch.sum(
        values.float() * attention.unsqueeze(-1) * mask_float.unsqueeze(-1),
        dim=1,
    )
    output = torch.cat((maximum, mean, torch.sqrt(variance + 1e-8), weighted), -1)
    names = []
    for statistic in ("max", "mean", "std", "attention_weighted"):
        names.extend(
            f"{prefix}_retrieval_{statistic}_{index}"
            for index in range(values.shape[-1])
        )
    return output, names


@torch.inference_mode()
def extract_features(
    head: MultimodalEvidenceHead,
    dataset: MultimodalEvidenceDataset,
    batch_size: int,
    device: torch.device,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    head.eval()
    matrices: list[np.ndarray] = []
    labels: list[int] = []
    text_predictions: list[int] = []
    expert_predictions: list[int] = []
    ids: list[str] = []
    feature_names: list[str] | None = None
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    for batch in tqdm(loader, desc=f"{split} router features"):
        batch_ids = batch.pop("id")
        target = batch.pop("labels")
        for key in MultimodalEvidenceDataset.SUPERVISION_KEYS:
            if key != "labels":
                batch.pop(key)
        model_input = {key: value.to(device) for key, value in batch.items()}
        output = head(**model_input)
        text_probability = torch.softmax(
            output["text_verdict_logits"].float(), -1
        )
        expert_probability = torch.softmax(
            output["visual_expert_logits"].float(), -1
        )
        blocks: list[torch.Tensor] = []
        names: list[str] = []
        for block, block_names in (
            probability_features(text_probability, expert_probability),
            attention_features(
                output["text_attention"], model_input["text_mask"], "text"
            ),
            attention_features(
                output["visual_attention"], model_input["visual_mask"],
                "visual",
            ),
            retrieval_features(
                model_input["text_retrieval_features"],
                model_input["text_mask"], output["text_attention"], "text",
            ),
            retrieval_features(
                model_input["visual_retrieval_features"],
                model_input["visual_mask"], output["visual_attention"],
                "visual",
            ),
        ):
            blocks.append(block)
            names.extend(block_names)
        diagnostics = torch.cat((
            output["conflict"].float(),
            torch.sigmoid(output["sufficiency_logit"].float()).unsqueeze(-1),
            output["visual_residual_logits"].float().abs().mean(-1, keepdim=True),
        ), dim=-1)
        blocks.append(diagnostics)
        names.extend([
            "constraint_conflict_0", "constraint_conflict_1",
            "constraint_conflict_2", "multimodal_sufficiency",
            "visual_residual_magnitude",
        ])
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("router feature schema changed between batches")
        matrix = torch.cat(blocks, dim=-1).cpu().numpy()
        if not np.isfinite(matrix).all():
            raise ValueError(f"non-finite router features in {split}")
        matrices.append(matrix)
        labels.extend(target.tolist())
        text_predictions.extend(text_probability.argmax(-1).cpu().tolist())
        expert_predictions.extend(expert_probability.argmax(-1).cpu().tolist())
        ids.extend(batch_ids)
    assert feature_names is not None
    return (
        np.concatenate(matrices),
        np.asarray(labels, dtype=np.int64),
        np.asarray(text_predictions, dtype=np.int64),
        np.asarray(expert_predictions, dtype=np.int64),
        feature_names,
        ids,
    )


def utility_labels(
    gold: np.ndarray, text: np.ndarray, expert: np.ndarray
) -> np.ndarray:
    labels = np.full(len(gold), UTILITY_NEUTRAL, dtype=np.int64)
    labels[(text == gold) & (expert != gold)] = UTILITY_HARMFUL
    labels[(text != gold) & (expert == gold)] = UTILITY_HELPFUL
    return labels


def classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def sample_weights(labels: np.ndarray, neutral_weight: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    inverse = len(labels) / (3.0 * np.maximum(counts, 1.0))
    weights = inverse[labels]
    weights[labels == UTILITY_NEUTRAL] *= neutral_weight
    return weights


def utility_score(probability: np.ndarray, classes: np.ndarray,
                  harm_penalty: float) -> np.ndarray:
    by_class = {
        int(label): probability[:, index]
        for index, label in enumerate(classes)
    }
    helpful = by_class.get(UTILITY_HELPFUL, np.zeros(len(probability)))
    harmful = by_class.get(UTILITY_HARMFUL, np.zeros(len(probability)))
    return helpful - harm_penalty * harmful


def route_metrics(gold: np.ndarray, text: np.ndarray, expert: np.ndarray,
                  score: np.ndarray, threshold: float) -> dict:
    route = score >= threshold
    prediction = np.where(route, expert, text)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(gold, prediction)),
        "macro_f1": float(f1_score(gold, prediction, average="macro")),
        "route_count": int(route.sum()),
        "route_rate": float(route.mean()),
        "helpful": int(((text != gold) & (prediction == gold)).sum()),
        "harmful": int(((text == gold) & (prediction != gold)).sum()),
    }


def select_threshold(gold: np.ndarray, text: np.ndarray, expert: np.ndarray,
                     score: np.ndarray) -> dict:
    quantiles = np.linspace(0.0, 1.0, 201)
    candidates = np.unique(np.quantile(score, quantiles))
    candidates = np.append(candidates, np.nextafter(score.max(), np.inf))
    results = [
        route_metrics(gold, text, expert, score, float(threshold))
        for threshold in candidates
    ]
    return max(
        results,
        key=lambda row: (row["macro_f1"], row["accuracy"], -row["route_rate"]),
    )


def load_expert(path: Path, metadata: dict, device: torch.device
                ) -> tuple[MultimodalEvidenceHead, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["head"]
    hidden_dim = int(state["claim_projection.1.weight"].shape[0])
    head = MultimodalEvidenceHead(
        claim_dim=int(metadata["claim_dim"]),
        text_dim=int(metadata["text_dim"]),
        visual_dim=int(metadata["visual_dim"]),
        hidden_dim=hidden_dim,
        dropout=0.0,
        visual_attention_mode=checkpoint.get(
            "visual_attention_mode", "learned"
        ),
        visual_prior_temperature=float(
            checkpoint.get("visual_prior_temperature", 0.5)
        ),
        visual_residual_scale=float(
            checkpoint.get("visual_residual_scale", 0.25)
        ),
        visual_expert_mode=checkpoint.get("visual_expert_mode", "residual"),
        visual_stance_scale=float(checkpoint.get("visual_stance_scale", 1.0)),
    )
    head.load_state_dict(state, strict=True)
    head.to(device).eval()
    return head, checkpoint


def load_feature_cache(path: Path) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
]:
    payload = np.load(path, allow_pickle=False)
    return (
        payload["features"], payload["gold"], payload["text_prediction"],
        payload["expert_prediction"], payload["feature_names"].tolist(),
        payload["ids"].tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-checkpoint", type=Path)
    parser.add_argument("--expert-cache-root", type=Path)
    parser.add_argument("--router-cache-root", type=Path)
    parser.add_argument(
        "--feature-cache-root", type=Path,
        help="use cross-fitted train_oof.npz and val_full.npz instead of "
             "extracting in-sample train expert outcomes",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--neutral-weight", type=float, default=0.25)
    parser.add_argument("--harm-penalty", type=float, default=1.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.neutral_weight <= 1:
        parser.error("--neutral-weight must be in (0, 1]")
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    expert_checkpoint: dict = {}
    if args.feature_cache_root is not None:
        train_path = args.feature_cache_root / "train_oof.npz"
        val_path = args.feature_cache_root / "val_full.npz"
        train_x, train_y, train_text, train_expert, names, train_ids = (
            load_feature_cache(train_path)
        )
        val_x, val_y, val_text, val_expert, val_names, val_ids = (
            load_feature_cache(val_path)
        )
    else:
        required = {
            "--expert-checkpoint": args.expert_checkpoint,
            "--expert-cache-root": args.expert_cache_root,
            "--router-cache-root": args.router_cache_root,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            parser.error(
                "without --feature-cache-root, required: " + ", ".join(missing)
            )
        expert_train = MultimodalEvidenceDataset(
            args.expert_cache_root / "train.pt"
        )
        train = MultimodalEvidenceDataset(
            args.router_cache_root / "train.pt"
        )
        val = MultimodalEvidenceDataset(args.expert_cache_root / "val.pt")
        validate_router_cache(expert_train, train)
        head, expert_checkpoint = load_expert(
            args.expert_checkpoint, train.metadata, device
        )
        train_x, train_y, train_text, train_expert, names, train_ids = (
            extract_features(head, train, args.batch_size, device, "train")
        )
        val_x, val_y, val_text, val_expert, val_names, val_ids = (
            extract_features(head, val, args.batch_size, device, "val")
        )
    if names != val_names:
        raise RuntimeError("train/validation router feature schema mismatch")
    train_utility = utility_labels(train_y, train_text, train_expert)
    weights = sample_weights(train_utility, args.neutral_weight)
    folds = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    oof_probability = np.zeros((len(train_y), 3), dtype=np.float64)
    for fold, (fit_indices, held_indices) in enumerate(
        folds.split(train_x, train_utility), start=1
    ):
        model = classifier(args.seed + fold)
        model.fit(
            train_x[fit_indices], train_utility[fit_indices],
            sample_weight=weights[fit_indices],
        )
        predicted = model.predict_proba(train_x[held_indices])
        for column, label in enumerate(model.classes_):
            oof_probability[held_indices, int(label)] = predicted[:, column]
        print(json.dumps({
            "fold": fold,
            "fit": len(fit_indices),
            "held_out": len(held_indices),
        }), flush=True)
    oof_score = utility_score(
        oof_probability, np.arange(3), args.harm_penalty
    )
    selected = select_threshold(
        train_y, train_text, train_expert, oof_score
    )
    model = classifier(args.seed)
    model.fit(train_x, train_utility, sample_weight=weights)
    val_probability = model.predict_proba(val_x)
    val_score = utility_score(
        val_probability, model.classes_, args.harm_penalty
    )
    validation = route_metrics(
        val_y, val_text, val_expert, val_score, selected["threshold"]
    )
    validation_rows = []
    route = val_score >= selected["threshold"]
    val_prediction = np.where(route, val_expert, val_text)
    for index, sample_id in enumerate(val_ids):
        validation_rows.append({
            "id": sample_id,
            "gold": int(val_y[index]),
            "prediction": int(val_prediction[index]),
            "text_only_prediction": int(val_text[index]),
            "visual_expert_prediction": int(val_expert[index]),
            "visual_modality_mass": float(val_score[index]),
            "route_visual": bool(route[index]),
            "utility_probabilities": val_probability[index].tolist(),
        })
    router_audit = audit(
        validation_rows,
        selected["threshold"],
        args.bootstrap_iterations,
        args.seed,
    )
    counts = np.bincount(train_utility, minlength=3)
    result = {
        "protocol": "train-OOF threshold; frozen validation evaluation",
        "test_split_used": False,
        "train_samples": len(train_y),
        "val_samples": len(val_y),
        "feature_count": len(names),
        "utility_label_counts": {
            "harmful": int(counts[UTILITY_HARMFUL]),
            "neutral": int(counts[UTILITY_NEUTRAL]),
            "helpful": int(counts[UTILITY_HELPFUL]),
        },
        "oof_train_selection": selected,
        "validation": validation,
        "validation_audit": router_audit,
        "configuration": {
            "folds": args.folds,
            "neutral_weight": args.neutral_weight,
            "harm_penalty": args.harm_penalty,
            "seed": args.seed,
        },
        "provenance": {
            "git_commit": git_commit(),
            "expert_checkpoint": str(args.expert_checkpoint),
            "expert_checkpoint_stage": expert_checkpoint.get("stage"),
            "expert_cache_root": str(args.expert_cache_root),
            "router_cache_root": str(args.router_cache_root),
            "feature_cache_root": str(args.feature_cache_root)
            if args.feature_cache_root is not None else None,
            "test_split_used": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "val_predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in validation_rows) + "\n",
        encoding="utf-8",
    )
    (args.output / "feature_names.json").write_text(
        json.dumps(names, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(model, args.output / "router.joblib")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
