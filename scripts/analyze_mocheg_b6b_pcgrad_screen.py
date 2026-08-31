"""Frozen two-seed validation screen for conflict-aware B6-B training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b6_auxiliary_control import (
    aligned_arrays,
    paired_comparison,
)
from scripts.analyze_mocheg_expert_complementarity import prediction_metrics


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "values": array.tolist(),
        "positive_seeds": int(np.sum(array > 0)),
    }


def summarize(runs: list[dict], minimum_control_mean_delta: float,
              minimum_seed87_control_delta: float,
              minimum_standard_mean_delta: float,
              minimum_bootstrap_probability: float,
              bootstrap_iterations: int, bootstrap_seed: int) -> dict:
    labels = runs[0]["labels"]
    comparisons = {}
    for baseline in ("anchor", "direct_control", "standard_auxiliary"):
        values = [
            row["metrics"]["conflict_auxiliary"]["macro_f1"]
            - row["metrics"][baseline]["macro_f1"] for row in runs
        ]
        comparisons[f"conflict_vs_{baseline}"] = aggregate(values)
    ensembles = {
        name: np.mean([row["probabilities"][name] for row in runs], axis=0)
        for name in (
            "anchor", "direct_control", "standard_auxiliary",
            "conflict_auxiliary",
        )
    }
    ensemble_metrics = {
        name: prediction_metrics(labels, probability)
        for name, probability in ensembles.items()
    }
    ensemble_vs_control = paired_comparison(
        labels, ensembles["direct_control"], ensembles["conflict_auxiliary"],
        bootstrap_iterations, bootstrap_seed,
    )
    seed87 = next(row for row in runs if row["seed"] == 87)
    seed87_delta = (
        seed87["metrics"]["conflict_auxiliary"]["macro_f1"]
        - seed87["metrics"]["direct_control"]["macro_f1"]
    )
    pcgrad_active = all(
        any(float(epoch.get("gradient_conflict_rate", 0)) > 0
            for epoch in row["history"] if int(epoch.get("epoch", 0)) > 0)
        for row in runs
    )
    gate = {
        "pcgrad_active_in_both_seeds": pcgrad_active,
        "conflict_beats_control_in_both_seeds": (
            comparisons["conflict_vs_direct_control"]["positive_seeds"]
            == len(runs)
        ),
        "mean_control_delta_at_least_minimum": (
            comparisons["conflict_vs_direct_control"]["mean"]
            >= minimum_control_mean_delta
        ),
        "seed87_control_delta_at_least_minimum": (
            seed87_delta >= minimum_seed87_control_delta
        ),
        "mean_delta_vs_standard_auxiliary_at_least_minimum": (
            comparisons["conflict_vs_standard_auxiliary"]["mean"]
            >= minimum_standard_mean_delta
        ),
        "ensemble_bootstrap_probability_at_least_minimum": (
            ensemble_vs_control["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B6B_frozen_seed42_seed87_pcgrad_screen",
        "seeds": [row["seed"] for row in runs],
        "per_seed": [{
            "seed": row["seed"],
            "metrics": row["metrics"],
            "conflict_vs_control_macro_f1": (
                row["metrics"]["conflict_auxiliary"]["macro_f1"]
                - row["metrics"]["direct_control"]["macro_f1"]
            ),
            "conflict_vs_standard_macro_f1": (
                row["metrics"]["conflict_auxiliary"]["macro_f1"]
                - row["metrics"]["standard_auxiliary"]["macro_f1"]
            ),
            "gradient_diagnostics": [{
                key: epoch.get(key) for key in (
                    "epoch", "gradient_cosine", "gradient_conflict_rate",
                    "protected_windows", "unprotected_auxiliary_windows",
                )
            } for epoch in row["history"] if int(epoch.get("epoch", 0)) > 0],
        } for row in runs],
        "paired_deltas": comparisons,
        "raw_ensembles": ensemble_metrics,
        "ensemble_conflict_vs_control": ensemble_vs_control,
        "seed87_conflict_vs_control_macro_f1": float(seed87_delta),
        "promotion_gate": gate,
        "settings": {
            "minimum_control_mean_delta": minimum_control_mean_delta,
            "minimum_seed87_control_delta": minimum_seed87_control_delta,
            "minimum_standard_mean_delta": minimum_standard_mean_delta,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "test_split_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 87])
    parser.add_argument("--article-template", default=(
        "outputs/mocheg_qwen3_lora_seed{seed}_v16"))
    parser.add_argument("--control-template", default=(
        "outputs/mocheg_b6_direct_control_seed{seed}"))
    parser.add_argument("--standard-template", default=(
        "outputs/mocheg_b6_auxiliary_seed{seed}"))
    parser.add_argument("--seed42-standard", type=Path, default=Path(
        "outputs/mocheg_b6_hierarchical_seed42"))
    parser.add_argument("--conflict-template", default=(
        "outputs/mocheg_b6b_pcgrad_seed{seed}"))
    parser.add_argument("--minimum-control-mean-delta", type=float, default=.005)
    parser.add_argument("--minimum-seed87-control-delta", type=float, default=.003)
    parser.add_argument("--minimum-standard-mean-delta", type=float, default=0)
    parser.add_argument("--minimum-bootstrap-probability", type=float,
                        default=.95)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b6b_pcgrad_screen.json"))
    args = parser.parse_args()

    runs = []
    reference_labels = None
    for seed in args.seeds:
        article = Path(args.article_template.format(seed=seed))
        control = Path(args.control_template.format(seed=seed))
        standard = (args.seed42_standard if seed == 42 else
                    Path(args.standard_template.format(seed=seed)))
        conflict = Path(args.conflict_template.format(seed=seed))
        summary = json.loads(
            (conflict / "summary.json").read_text(encoding="utf-8")
        )
        if summary.get("test_split_used") is not False:
            raise ValueError(f"seed {seed} conflict run is not validation-only")
        if summary.get("settings", {}).get("gradient_mode") != "pcgrad":
            raise ValueError(f"seed {seed} is not a PCGrad run")
        if float(summary["selected_hierarchical_weight"]) != 0:
            raise ValueError("B6-B inference weight must remain zero")
        paths = {
            "anchor": article / "val_predictions.jsonl",
            "direct_control": control / "val_predictions.jsonl",
            "standard_auxiliary": standard / "val_predictions.jsonl",
            "conflict_auxiliary": conflict / "val_predictions.jsonl",
        }
        labels, probabilities = aligned_arrays(paths)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(labels, reference_labels):
            raise ValueError("validation labels do not align across seeds")
        runs.append({
            "seed": seed,
            "labels": labels,
            "probabilities": probabilities,
            "metrics": {
                name: prediction_metrics(labels, probability)
                for name, probability in probabilities.items()
            },
            "history": summary["history"],
        })
    result = summarize(
        runs, args.minimum_control_mean_delta,
        args.minimum_seed87_control_delta,
        args.minimum_standard_mean_delta,
        args.minimum_bootstrap_probability,
        args.bootstrap_iterations, args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
