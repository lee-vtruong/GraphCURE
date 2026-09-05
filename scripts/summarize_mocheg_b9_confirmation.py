"""Summarize frozen B9 confirmation on duplicate-safe train folds 1--4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b6_auxiliary_control import paired_comparison
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_b7_frozen_router import source_diagnostics
from scripts.analyze_mocheg_b9_anchor_ensemble import (
    FROZEN_SEEDS,
    REFERENCE_SEED,
    aligned_runs,
)
from scripts.analyze_mocheg_expert_complementarity import prediction_metrics


CONFIRMATION_FOLDS = (1, 2, 3, 4)


def summarize(
    runs: list[dict],
    minimum_mean_delta: float = .005,
    minimum_positive_folds: int = 3,
    minimum_aggregate_delta: float = .005,
    maximum_best_seed_drop: float = .002,
    maximum_accuracy_drop: float = .002,
    minimum_source_delta: float = -.002,
    minimum_bootstrap_probability: float = .95,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> dict:
    folds = tuple(sorted(int(row["fold"]) for row in runs))
    if folds != CONFIRMATION_FOLDS:
        raise ValueError(f"B9 confirmation requires folds {CONFIRMATION_FOLDS}")
    seen_ids: set[str] = set()
    per_fold = []
    all_labels = []
    all_sources = []
    all_probabilities = {seed: [] for seed in FROZEN_SEEDS}
    for row in sorted(runs, key=lambda value: value["fold"]):
        overlap = seen_ids & set(row["ids"])
        if overlap:
            raise ValueError(f"B9 confirmation fold overlap: {len(overlap)}")
        seen_ids.update(row["ids"])
        labels = np.asarray(row["labels"])
        probabilities = row["probabilities"]
        ensemble = np.mean(
            np.stack([probabilities[seed] for seed in FROZEN_SEEDS]), axis=0
        )
        reference = probabilities[REFERENCE_SEED]
        reference_metrics = prediction_metrics(labels, reference)
        ensemble_metrics = prediction_metrics(labels, ensemble)
        comparison = paired_comparison(
            labels, reference, ensemble,
            bootstrap_iterations, bootstrap_seed + int(row["fold"]),
        )
        per_fold.append({
            "fold": int(row["fold"]),
            "samples": int(len(labels)),
            "seed42": reference_metrics,
            "ensemble": ensemble_metrics,
            "macro_f1_delta": comparison["macro_f1_delta"],
            "accuracy_delta": comparison["accuracy_delta"],
            "helpful": comparison["helpful"],
            "harmful": comparison["harmful"],
        })
        all_labels.append(labels)
        all_sources.append(np.asarray(row["sources"]))
        for seed in FROZEN_SEEDS:
            all_probabilities[seed].append(probabilities[seed])
    labels = np.concatenate(all_labels)
    sources = np.concatenate(all_sources)
    probabilities = {
        seed: np.concatenate(parts) for seed, parts in all_probabilities.items()
    }
    reference = probabilities[REFERENCE_SEED]
    ensemble = np.mean(
        np.stack([probabilities[seed] for seed in FROZEN_SEEDS]), axis=0
    )
    per_seed = {
        str(seed): prediction_metrics(labels, probabilities[seed])
        for seed in FROZEN_SEEDS
    }
    reference_metrics = per_seed[str(REFERENCE_SEED)]
    ensemble_metrics = prediction_metrics(labels, ensemble)
    strongest_seed, strongest_metrics = max(
        per_seed.items(), key=lambda item: item[1]["macro_f1"]
    )
    aggregate_comparison = paired_comparison(
        labels, reference, ensemble, bootstrap_iterations, bootstrap_seed
    )
    by_source = source_diagnostics(labels, reference, ensemble, sources)
    deltas = np.asarray([row["macro_f1_delta"] for row in per_fold])
    best_seed_delta = (
        ensemble_metrics["macro_f1"] - strongest_metrics["macro_f1"]
    )
    gate = {
        "mean_fold_delta_at_least_minimum": (
            float(deltas.mean()) >= minimum_mean_delta
        ),
        "positive_folds_at_least_minimum": (
            int(np.sum(deltas > 0)) >= minimum_positive_folds
        ),
        "aggregate_delta_at_least_minimum": (
            aggregate_comparison["macro_f1_delta"]
            >= minimum_aggregate_delta
        ),
        "aggregate_bootstrap_probability_at_least_minimum": (
            aggregate_comparison["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
        "aggregate_accuracy_within_noninferiority_margin": (
            aggregate_comparison["accuracy_delta"] >= -maximum_accuracy_drop
        ),
        "ensemble_within_margin_of_strongest_seed": (
            best_seed_delta >= -maximum_best_seed_drop
        ),
        "aggregate_help_exceeds_harm": (
            aggregate_comparison["helpful"]
            > aggregate_comparison["harmful"]
        ),
        "all_sources_within_noninferiority_margin": all(
            row["macro_f1_delta"] >= minimum_source_delta
            for row in by_source.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B9_frozen_train_only_folds_1_to_4_confirmation",
        "folds": list(CONFIRMATION_FOLDS),
        "frozen_seeds": list(FROZEN_SEEDS),
        "reference_seed": REFERENCE_SEED,
        "samples": int(len(labels)),
        "per_fold": per_fold,
        "paired_fold_macro_f1_delta": {
            "mean": float(deltas.mean()),
            "std": float(deltas.std()),
            "values": deltas.tolist(),
            "positive_folds": int(np.sum(deltas > 0)),
        },
        "aggregate": {
            "per_seed": per_seed,
            "ensemble": ensemble_metrics,
            "strongest_constituent": {
                "seed": int(strongest_seed), **strongest_metrics,
            },
            "ensemble_minus_strongest_macro_f1": best_seed_delta,
            "comparison_vs_seed42": aggregate_comparison,
            "source_diagnostics_vs_seed42": by_source,
        },
        "promotion_gate": gate,
        "settings": {
            "combination": "unweighted_arithmetic_probability_mean",
            "minimum_mean_delta": minimum_mean_delta,
            "minimum_positive_folds": minimum_positive_folds,
            "minimum_aggregate_delta": minimum_aggregate_delta,
            "maximum_best_seed_drop": maximum_best_seed_drop,
            "maximum_accuracy_drop": maximum_accuracy_drop,
            "minimum_source_delta": minimum_source_delta,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "fold0_used_for_confirmation": False,
        "official_validation_used": False,
        "test_split_used": False,
    }


def markdown(result: dict) -> str:
    lines = [
        "# MOCHEG B9 frozen anchor-ensemble confirmation",
        "",
        "Official validation used: **no**  ",
        "Test used: **no**",
        "",
        "| Fold | Seed-42 F1 | Ensemble F1 | Delta |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["per_fold"]:
        lines.append(
            f"| {row['fold']} | {row['seed42']['macro_f1']:.4f} | "
            f"{row['ensemble']['macro_f1']:.4f} | "
            f"{row['macro_f1_delta']:+.4f} |"
        )
    delta = result["paired_fold_macro_f1_delta"]
    aggregate = result["aggregate"]
    lines.extend([
        "", "## Aggregate", "",
        f"- Fold delta: {delta['mean']:+.6f} +/- {delta['std']:.6f}",
        f"- Ensemble Macro-F1: {aggregate['ensemble']['macro_f1']:.6f}",
        "- Delta vs seed 42: "
        f"{aggregate['comparison_vs_seed42']['macro_f1_delta']:+.6f}",
        f"- Promotion gate: **{'pass' if result['promotion_gate']['passed'] else 'fail'}**",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "outputs/mocheg_b9_oof"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b9_confirmation.json"))
    parser.add_argument("--markdown", type=Path, default=Path(
        "outputs/mocheg_b9_confirmation.md"))
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    runs = []
    for fold in CONFIRMATION_FOLDS:
        paths = {
            seed: args.root / f"fold_{fold}" / f"seed_{seed}"
            for seed in FROZEN_SEEDS
        }
        for seed, path in paths.items():
            validate_run(path, fold, f"fold_{fold}_seed_{seed}")
        ids, labels, probabilities, sources = aligned_runs(
            {seed: path / "val_predictions.jsonl"
             for seed, path in paths.items()}, args.manifest
        )
        runs.append({
            "fold": fold, "ids": ids, "labels": labels,
            "probabilities": probabilities, "sources": sources,
        })
    result = summarize(
        runs, bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
