"""Evaluate a frozen Flat/GraphCURE-SV interpolation on untouched folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scripts.analyze_mocheg_sv_complementarity import read_predictions


def paired_fold(flat_path: Path, hierarchical_path: Path, alpha: float) -> dict:
    flat_rows = read_predictions(flat_path)
    hierarchical_rows = read_predictions(hierarchical_path)
    if set(flat_rows) != set(hierarchical_rows):
        raise ValueError("prediction ID sets do not match")
    ids = sorted(flat_rows)
    y = np.asarray([int(flat_rows[value]["gold"]) for value in ids])
    other_y = np.asarray([int(hierarchical_rows[value]["gold"]) for value in ids])
    if not np.array_equal(y, other_y):
        raise ValueError("gold labels do not match")
    flat = np.asarray([flat_rows[value]["probabilities"] for value in ids])
    hierarchical = np.asarray(
        [hierarchical_rows[value]["probabilities"] for value in ids]
    )
    blended = (1.0 - alpha) * flat + alpha * hierarchical

    def score(probability):
        prediction = probability.argmax(-1)
        return {
            "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro")),
            "confusion_matrix": confusion_matrix(
                y, prediction, labels=[0, 1, 2]
            ).tolist(),
        }

    flat_metrics = score(flat)
    hierarchical_metrics = score(hierarchical)
    ensemble_metrics = score(blended)
    return {
        "samples": len(ids),
        "flat": flat_metrics,
        "hierarchical": hierarchical_metrics,
        "frozen_ensemble": ensemble_metrics,
        "ensemble_minus_flat_macro_f1": float(
            ensemble_metrics["macro_f1"] - flat_metrics["macro_f1"]
        ),
    }


def aggregate(rows: list[dict], key: str, metric: str) -> dict:
    values = np.asarray([row[key][metric] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "values": values.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flat-template",
        default="outputs/mocheg_sv_confirm/fold_{fold}_flat/val_predictions.jsonl",
    )
    parser.add_argument(
        "--hierarchical-template",
        default="outputs/mocheg_sv_confirm/fold_{fold}_hier/val_predictions.jsonl",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--hierarchical-weight", type=float, default=0.63)
    parser.add_argument("--minimum-mean-delta", type=float, default=0.015)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/mocheg_sv_confirmation.json"),
    )
    args = parser.parse_args()
    if not 0.0 <= args.hierarchical_weight <= 1.0:
        raise ValueError("hierarchical weight must be in [0, 1]")
    rows = []
    for fold in args.folds:
        result = paired_fold(
            Path(args.flat_template.format(fold=fold)),
            Path(args.hierarchical_template.format(fold=fold)),
            args.hierarchical_weight,
        )
        result["fold"] = fold
        rows.append(result)
    deltas = np.asarray(
        [row["ensemble_minus_flat_macro_f1"] for row in rows]
    )
    payload = {
        "protocol": "frozen_confirmation_on_untouched_train_only_folds",
        "development_fold_excluded": 0,
        "confirmation_folds": args.folds,
        "hierarchical_weight": args.hierarchical_weight,
        "coefficient_tuned_on_confirmation": False,
        "per_fold": rows,
        "aggregate": {
            key: {
                "accuracy": aggregate(rows, key, "accuracy"),
                "macro_f1": aggregate(rows, key, "macro_f1"),
            }
            for key in ("flat", "hierarchical", "frozen_ensemble")
        },
        "paired_macro_f1_delta": {
            "mean": float(deltas.mean()), "std": float(deltas.std()),
            "values": deltas.tolist(),
        },
        "promotion_gate": {
            "minimum_mean_delta": args.minimum_mean_delta,
            "mean_delta_passed": bool(deltas.mean() >= args.minimum_mean_delta),
            "stability_passed": bool(
                np.asarray([
                    row["frozen_ensemble"]["macro_f1"] for row in rows
                ]).std() <= 0.02
            ),
        },
        "test_split_used": False,
    }
    payload["promotion_gate"]["passed"] = bool(
        payload["promotion_gate"]["mean_delta_passed"]
        and payload["promotion_gate"]["stability_passed"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
