"""Diagnose complementarity of two train-only MOCHEG fold predictors.

This is a development diagnostic.  It may select an interpolation coefficient
or an NEI logit offset on the supplied fold, so its best score is not an
unbiased estimate and must be confirmed on untouched folds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def read_predictions(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate prediction IDs in {path}")
    return result


def metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(-1)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
        "prediction_counts": np.bincount(prediction, minlength=3).tolist(),
    }


def add_logit_bias(probability: np.ndarray, class_index: int, bias: float) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-12, 1.0))
    logits[:, class_index] += bias
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", type=Path, required=True)
    parser.add_argument("--hierarchical", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/mocheg_sv_fold0_complementarity.json"),
    )
    parser.add_argument("--minimum-delta", type=float, default=0.01)
    args = parser.parse_args()
    flat_rows = read_predictions(args.flat)
    hierarchical_rows = read_predictions(args.hierarchical)
    if set(flat_rows) != set(hierarchical_rows):
        raise ValueError("prediction ID sets do not match")
    ids = sorted(flat_rows)
    y = np.asarray([int(flat_rows[value]["gold"]) for value in ids])
    other_y = np.asarray([int(hierarchical_rows[value]["gold"]) for value in ids])
    if not np.array_equal(y, other_y):
        raise ValueError("gold labels do not match")
    flat = np.asarray([flat_rows[value]["probabilities"] for value in ids], dtype=np.float64)
    hierarchical = np.asarray(
        [hierarchical_rows[value]["probabilities"] for value in ids], dtype=np.float64
    )
    flat_prediction = flat.argmax(-1)
    hierarchical_prediction = hierarchical.argmax(-1)
    flat_correct = flat_prediction == y
    hierarchical_correct = hierarchical_prediction == y

    interpolation = []
    for alpha in np.linspace(0.0, 1.0, 101):
        probability = (1.0 - alpha) * flat + alpha * hierarchical
        row = metrics(y, probability)
        row["hierarchical_weight"] = float(alpha)
        interpolation.append(row)
    best_interpolation = max(interpolation, key=lambda row: row["macro_f1"])

    bias_rows = []
    for bias in np.linspace(-1.5, 0.5, 101):
        row = metrics(y, add_logit_bias(hierarchical, 2, float(bias)))
        row["nei_logit_bias"] = float(bias)
        bias_rows.append(row)
    best_bias = max(bias_rows, key=lambda row: row["macro_f1"])

    flat_metrics = metrics(y, flat)
    hierarchical_metrics = metrics(y, hierarchical)
    best_development_f1 = max(
        best_interpolation["macro_f1"], best_bias["macro_f1"]
    )
    payload = {
        "protocol": "train_only_fold_development_diagnostic",
        "samples": len(ids),
        "flat": flat_metrics,
        "hierarchical": hierarchical_metrics,
        "outcome_overlap": {
            "both_correct": int(np.sum(flat_correct & hierarchical_correct)),
            "flat_only_correct": int(np.sum(flat_correct & ~hierarchical_correct)),
            "hierarchical_only_correct": int(np.sum(~flat_correct & hierarchical_correct)),
            "both_wrong": int(np.sum(~flat_correct & ~hierarchical_correct)),
            "prediction_disagreements": int(np.sum(flat_prediction != hierarchical_prediction)),
        },
        "best_probability_interpolation": best_interpolation,
        "best_hierarchical_nei_bias": best_bias,
        "best_diagnostic_delta_vs_flat": float(
            best_development_f1 - flat_metrics["macro_f1"]
        ),
        "promotion_potential": bool(
            best_development_f1 - flat_metrics["macro_f1"] >= args.minimum_delta
        ),
        "warning": (
            "Hyperparameters were selected on this fold. The selected score is "
            "diagnostic only and requires frozen confirmation on untouched folds."
        ),
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
