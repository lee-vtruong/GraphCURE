"""Validation-only complementarity diagnostic for two MOCHEG experts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p


def read_predictions(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate prediction IDs in {path}")
    return result


def prediction_metrics(gold: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(-1)
    return {
        "accuracy": float(accuracy_score(gold, prediction)),
        "macro_f1": float(f1_score(gold, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(
            gold, prediction, labels=[0, 1, 2]
        ).tolist(),
    }


def diagnose(anchor_rows: dict[str, dict], packet_rows: dict[str, dict],
             minimum_delta: float, bootstrap_iterations: int,
             seed: int) -> dict:
    if set(anchor_rows) != set(packet_rows):
        raise ValueError("prediction ID sets do not match")
    ids = sorted(anchor_rows)
    gold = np.asarray([int(anchor_rows[value]["gold"]) for value in ids])
    other_gold = np.asarray([int(packet_rows[value]["gold"]) for value in ids])
    if not np.array_equal(gold, other_gold):
        raise ValueError("gold labels do not match")
    anchor = np.asarray(
        [anchor_rows[value]["probabilities"] for value in ids], dtype=np.float64
    )
    packet = np.asarray(
        [packet_rows[value]["probabilities"] for value in ids], dtype=np.float64
    )
    anchor_prediction, packet_prediction = anchor.argmax(-1), packet.argmax(-1)
    anchor_correct, packet_correct = anchor_prediction == gold, packet_prediction == gold
    anchor_metrics = prediction_metrics(gold, anchor)
    packet_metrics = prediction_metrics(gold, packet)

    rows = []
    for packet_weight in np.linspace(0.0, 1.0, 101):
        probability = (1.0 - packet_weight) * anchor + packet_weight * packet
        row = prediction_metrics(gold, probability)
        row["packet_weight"] = float(packet_weight)
        rows.append(row)
    best = max(rows, key=lambda row: (row["macro_f1"], row["accuracy"]))
    equal = rows[50]
    best_probability = (
        (1.0 - best["packet_weight"]) * anchor
        + best["packet_weight"] * packet
    )
    best_prediction = best_probability.argmax(-1)
    best_help = int(np.sum(~anchor_correct & (best_prediction == gold)))
    best_harm = int(np.sum(anchor_correct & (best_prediction != gold)))

    oracle_prediction = anchor_prediction.copy()
    oracle_prediction[~anchor_correct & packet_correct] = packet_prediction[
        ~anchor_correct & packet_correct
    ]
    oracle_probability = np.eye(3, dtype=np.float64)[oracle_prediction]
    delta = best["macro_f1"] - anchor_metrics["macro_f1"]
    return {
        "protocol": "validation_only_expert_complementarity_diagnostic",
        "samples": len(ids),
        "anchor": anchor_metrics,
        "packet": packet_metrics,
        "outcome_overlap": {
            "both_correct": int(np.sum(anchor_correct & packet_correct)),
            "anchor_only_correct": int(np.sum(anchor_correct & ~packet_correct)),
            "packet_only_correct": int(np.sum(~anchor_correct & packet_correct)),
            "both_wrong": int(np.sum(~anchor_correct & ~packet_correct)),
            "prediction_disagreements": int(np.sum(anchor_prediction != packet_prediction)),
        },
        "equal_weight_ensemble": equal,
        "best_probability_interpolation": best,
        "best_interpolation_delta_vs_anchor": float(delta),
        "best_interpolation_helpful": best_help,
        "best_interpolation_harmful": best_harm,
        "best_interpolation_exact_mcnemar_p": exact_mcnemar_p(best_help, best_harm),
        "best_interpolation_bootstrap": bootstrap_delta(
            gold, anchor_prediction, best_prediction, bootstrap_iterations, seed
        ),
        "oracle_expert_selection": prediction_metrics(gold, oracle_probability),
        "promotion_potential": bool(delta >= minimum_delta),
        "minimum_delta": minimum_delta,
        "warning": (
            "The interpolation weight was selected on this validation split. "
            "This is diagnostic only; freeze/confirm it across multiple seeds "
            "before any test evaluation."
        ),
        "test_split_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_packet_complementarity.json"))
    parser.add_argument("--minimum-delta", type=float, default=.003)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = diagnose(
        read_predictions(args.anchor), read_predictions(args.packet),
        args.minimum_delta, args.bootstrap_iterations, args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
