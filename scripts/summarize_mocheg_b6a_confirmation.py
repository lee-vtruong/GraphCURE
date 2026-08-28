"""Summarize frozen five-seed B6-A auxiliary-supervision confirmation."""
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


def validate_run(summary_path: Path, role: str,
                 auxiliary_examples: int | None = None) -> tuple[dict, int]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_split_used") is not False:
        raise ValueError(f"{role} run is not validation-only: {summary_path}")
    if float(summary["selected_hierarchical_weight"]) != 0:
        raise ValueError(f"{role} must use frozen hierarchical weight 0")
    counts = summary.get("training_task_counts", {})
    examples = sum(int(value) for value in counts.values())
    if role == "direct_control":
        if set(counts) != {"verdict"}:
            raise ValueError("direct control contains auxiliary training tasks")
        if auxiliary_examples is None or examples != auxiliary_examples:
            raise ValueError("direct control is not compute-matched")
    return summary, examples


def summarize(runs: list[dict], minimum_anchor_mean_delta: float,
              minimum_control_mean_delta: float,
              minimum_ensemble_delta: float,
              minimum_ensemble_f1: float,
              minimum_positive_seeds: int,
              minimum_bootstrap_probability: float,
              bootstrap_iterations: int, bootstrap_seed: int) -> dict:
    anchor_deltas = [
        row["metrics"]["auxiliary"]["macro_f1"]
        - row["metrics"]["anchor"]["macro_f1"] for row in runs
    ]
    control_deltas = [
        row["metrics"]["auxiliary"]["macro_f1"]
        - row["metrics"]["direct_control"]["macro_f1"] for row in runs
    ]
    labels = runs[0]["labels"]
    ensembles = {
        name: np.mean([row["probabilities"][name] for row in runs], axis=0)
        for name in ("anchor", "direct_control", "auxiliary")
    }
    ensemble_metrics = {
        name: prediction_metrics(labels, values)
        for name, values in ensembles.items()
    }
    ensemble_anchor = paired_comparison(
        labels, ensembles["anchor"], ensembles["auxiliary"],
        bootstrap_iterations, bootstrap_seed,
    )
    ensemble_control = paired_comparison(
        labels, ensembles["direct_control"], ensembles["auxiliary"],
        bootstrap_iterations, bootstrap_seed,
    )
    anchor_aggregate = aggregate(anchor_deltas)
    control_aggregate = aggregate(control_deltas)
    gate = {
        "mean_auxiliary_vs_anchor_at_least_minimum": (
            anchor_aggregate["mean"] >= minimum_anchor_mean_delta
        ),
        "mean_auxiliary_vs_control_at_least_minimum": (
            control_aggregate["mean"] >= minimum_control_mean_delta
        ),
        "ensemble_auxiliary_vs_anchor_at_least_minimum": (
            ensemble_anchor["macro_f1_delta"] >= minimum_ensemble_delta
        ),
        "ensemble_auxiliary_vs_control_at_least_minimum": (
            ensemble_control["macro_f1_delta"] >= minimum_ensemble_delta
        ),
        "ensemble_macro_f1_at_least_target": (
            ensemble_metrics["auxiliary"]["macro_f1"] >= minimum_ensemble_f1
        ),
        "positive_control_deltas_at_least_minimum": (
            control_aggregate["positive_seeds"] >= minimum_positive_seeds
        ),
        "ensemble_bootstrap_probability_at_least_minimum": (
            ensemble_control["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B6A_frozen_five_seed_validation_confirmation",
        "seeds": [row["seed"] for row in runs],
        "per_seed": [{
            "seed": row["seed"],
            "anchor": row["metrics"]["anchor"],
            "direct_control": row["metrics"]["direct_control"],
            "auxiliary": row["metrics"]["auxiliary"],
            "auxiliary_vs_anchor_macro_f1": anchor_delta,
            "auxiliary_vs_direct_control_macro_f1": control_delta,
        } for row, anchor_delta, control_delta in zip(
            runs, anchor_deltas, control_deltas
        )],
        "paired_auxiliary_vs_anchor": anchor_aggregate,
        "paired_auxiliary_vs_direct_control": control_aggregate,
        "raw_ensembles": ensemble_metrics,
        "ensemble_auxiliary_vs_anchor": ensemble_anchor,
        "ensemble_auxiliary_vs_direct_control": ensemble_control,
        "promotion_gate": gate,
        "settings": {
            "minimum_anchor_mean_delta": minimum_anchor_mean_delta,
            "minimum_control_mean_delta": minimum_control_mean_delta,
            "minimum_ensemble_delta": minimum_ensemble_delta,
            "minimum_ensemble_f1": minimum_ensemble_f1,
            "minimum_positive_seeds": minimum_positive_seeds,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "test_split_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[13, 21, 42, 87, 100])
    parser.add_argument("--article-template", default=(
        "outputs/mocheg_qwen3_lora_seed{seed}_v16"))
    parser.add_argument("--auxiliary-template", default=(
        "outputs/mocheg_b6_auxiliary_seed{seed}"))
    parser.add_argument("--seed42-auxiliary", type=Path, default=Path(
        "outputs/mocheg_b6_hierarchical_seed42"))
    parser.add_argument("--control-template", default=(
        "outputs/mocheg_b6_direct_control_seed{seed}"))
    parser.add_argument("--minimum-anchor-mean-delta", type=float, default=.005)
    parser.add_argument("--minimum-control-mean-delta", type=float, default=.003)
    parser.add_argument("--minimum-ensemble-delta", type=float, default=.003)
    parser.add_argument("--minimum-ensemble-f1", type=float, default=.695)
    parser.add_argument("--minimum-positive-seeds", type=int, default=4)
    parser.add_argument("--minimum-bootstrap-probability", type=float,
                        default=.95)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b6a_validation_summary.json"))
    parser.add_argument("--markdown", type=Path, default=Path(
        "outputs/mocheg_b6a_validation_summary.md"))
    args = parser.parse_args()

    runs = []
    reference_labels = None
    for seed in args.seeds:
        article = Path(args.article_template.format(seed=seed))
        auxiliary = (args.seed42_auxiliary if seed == 42 else
                     Path(args.auxiliary_template.format(seed=seed)))
        control = Path(args.control_template.format(seed=seed))
        _, auxiliary_examples = validate_run(
            auxiliary / "summary.json", "auxiliary"
        )
        validate_run(
            control / "summary.json", "direct_control", auxiliary_examples
        )
        paths = {
            "anchor": article / "val_predictions.jsonl",
            "direct_control": control / "val_predictions.jsonl",
            "auxiliary": auxiliary / "val_predictions.jsonl",
        }
        labels, probabilities = aligned_arrays(paths)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(labels, reference_labels):
            raise ValueError(f"seed {seed} validation labels are not aligned")
        runs.append({
            "seed": seed,
            "labels": labels,
            "probabilities": probabilities,
            "metrics": {
                name: prediction_metrics(labels, values)
                for name, values in probabilities.items()
            },
        })

    result = summarize(
        runs, args.minimum_anchor_mean_delta, args.minimum_control_mean_delta,
        args.minimum_ensemble_delta, args.minimum_ensemble_f1,
        args.minimum_positive_seeds, args.minimum_bootstrap_probability,
        args.bootstrap_iterations, args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MOCHEG B6-A frozen validation confirmation", "",
        "Test split used: **no**", "",
        "| Seed | Anchor F1 | Direct control F1 | Auxiliary F1 | Aux-Control |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['anchor']['macro_f1']:.4f} | "
            f"{row['direct_control']['macro_f1']:.4f} | "
            f"{row['auxiliary']['macro_f1']:.4f} | "
            f"{row['auxiliary_vs_direct_control_macro_f1']:+.4f} |"
        )
    lines += [
        "", "## Aggregate",
        "- Auxiliary vs anchor: "
        f"{result['paired_auxiliary_vs_anchor']['mean']:+.6f} +/- "
        f"{result['paired_auxiliary_vs_anchor']['std']:.6f}",
        "- Auxiliary vs direct control: "
        f"{result['paired_auxiliary_vs_direct_control']['mean']:+.6f} +/- "
        f"{result['paired_auxiliary_vs_direct_control']['std']:.6f}",
        "- Raw auxiliary ensemble F1: "
        f"{result['raw_ensembles']['auxiliary']['macro_f1']:.6f}",
        "- Promotion gate: "
        f"**{'pass' if result['promotion_gate']['passed'] else 'fail'}**",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
