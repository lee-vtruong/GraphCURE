"""Compare B6 auxiliary supervision with a matched direct-only control.

This diagnostic is validation-only. It does not tune an ensemble or touch the
official test split; it asks whether sufficiency/polarity supervision improves
the direct verdict head beyond simply continuing the article adapter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_expert_complementarity import (
    prediction_metrics,
    read_predictions,
)
from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p


def aligned_arrays(paths: dict[str, Path]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    loaded = {name: read_predictions(path) for name, path in paths.items()}
    reference = set(loaded["anchor"])
    if any(set(rows) != reference for rows in loaded.values()):
        raise ValueError("prediction ID sets do not match")
    ids = sorted(reference)
    labels = np.asarray([int(loaded["anchor"][value]["gold"]) for value in ids])
    probabilities = {}
    for name, rows in loaded.items():
        observed = np.asarray([int(rows[value]["gold"]) for value in ids])
        if not np.array_equal(labels, observed):
            raise ValueError(f"gold labels do not match for {name}")
        probabilities[name] = np.asarray(
            [rows[value]["probabilities"] for value in ids], dtype=np.float64
        )
    return labels, probabilities


def paired_comparison(labels: np.ndarray, baseline: np.ndarray,
                      candidate: np.ndarray, iterations: int,
                      seed: int) -> dict:
    baseline_prediction = baseline.argmax(-1)
    candidate_prediction = candidate.argmax(-1)
    baseline_correct = baseline_prediction == labels
    candidate_correct = candidate_prediction == labels
    helpful = int(np.sum(~baseline_correct & candidate_correct))
    harmful = int(np.sum(baseline_correct & ~candidate_correct))
    baseline_metrics = prediction_metrics(labels, baseline)
    candidate_metrics = prediction_metrics(labels, candidate)
    return {
        "macro_f1_delta": float(
            candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"]
        ),
        "accuracy_delta": float(
            candidate_metrics["accuracy"] - baseline_metrics["accuracy"]
        ),
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(
            labels, baseline_prediction, candidate_prediction,
            iterations, seed,
        ),
    }


def analyze(paths: dict[str, Path], minimum_anchor_delta: float,
            minimum_control_delta: float, minimum_bootstrap_probability: float,
            iterations: int, seed: int) -> dict:
    labels, probabilities = aligned_arrays(paths)
    metrics = {
        name: prediction_metrics(labels, values)
        for name, values in probabilities.items()
    }
    auxiliary_vs_anchor = paired_comparison(
        labels, probabilities["anchor"], probabilities["auxiliary"],
        iterations, seed,
    )
    direct_vs_anchor = paired_comparison(
        labels, probabilities["anchor"], probabilities["direct_control"],
        iterations, seed,
    )
    auxiliary_vs_direct = paired_comparison(
        labels, probabilities["direct_control"], probabilities["auxiliary"],
        iterations, seed,
    )
    gate = {
        "auxiliary_beats_anchor": (
            auxiliary_vs_anchor["macro_f1_delta"] >= minimum_anchor_delta
        ),
        "auxiliary_beats_matched_direct_control": (
            auxiliary_vs_direct["macro_f1_delta"] >= minimum_control_delta
        ),
        "bootstrap_probability_at_least_minimum": (
            auxiliary_vs_direct["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B6A_validation_only_auxiliary_causality_screen",
        "samples": int(len(labels)),
        "metrics": metrics,
        "auxiliary_vs_anchor": auxiliary_vs_anchor,
        "direct_control_vs_anchor": direct_vs_anchor,
        "auxiliary_vs_direct_control": auxiliary_vs_direct,
        "promotion_gate": gate,
        "settings": {
            "minimum_anchor_delta": minimum_anchor_delta,
            "minimum_control_delta": minimum_control_delta,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
        },
        "test_split_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, default=Path(
        "outputs/mocheg_qwen3_lora_seed42_v16/val_predictions.jsonl"))
    parser.add_argument("--direct-control", type=Path, default=Path(
        "outputs/mocheg_b6_direct_control_seed42/val_predictions.jsonl"))
    parser.add_argument("--auxiliary", type=Path, default=Path(
        "outputs/mocheg_b6_hierarchical_seed42/val_predictions.jsonl"))
    parser.add_argument("--minimum-anchor-delta", type=float, default=.005)
    parser.add_argument("--minimum-control-delta", type=float, default=.003)
    parser.add_argument("--minimum-bootstrap-probability", type=float,
                        default=.95)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b6_auxiliary_control.json"))
    args = parser.parse_args()
    paths = {
        "anchor": args.anchor,
        "direct_control": args.direct_control,
        "auxiliary": args.auxiliary,
    }
    result = analyze(
        paths, args.minimum_anchor_delta, args.minimum_control_delta,
        args.minimum_bootstrap_probability, args.bootstrap_iterations,
        args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
