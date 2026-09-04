"""Preregistered train-fold screen for a variance-reduced anchor ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b6_auxiliary_control import paired_comparison
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_b7_frozen_router import source_diagnostics
from scripts.analyze_mocheg_expert_complementarity import (
    prediction_metrics,
    read_predictions,
)
from scripts.run_mocheg_visual_retrieval import read_jsonl


FROZEN_SEEDS = (13, 42, 87)
REFERENCE_SEED = 42


def aligned_runs(
    paths: dict[int, Path], manifest_path: Path,
) -> tuple[list[str], np.ndarray, dict[int, np.ndarray], np.ndarray]:
    if tuple(sorted(paths)) != FROZEN_SEEDS:
        raise ValueError(f"B9 requires exactly seeds {FROZEN_SEEDS}")
    loaded = {seed: read_predictions(path) for seed, path in paths.items()}
    reference_ids = set(loaded[REFERENCE_SEED])
    if any(set(rows) != reference_ids for rows in loaded.values()):
        raise ValueError("B9 prediction ID sets do not match")
    ids = sorted(reference_ids)
    labels = np.asarray([
        int(loaded[REFERENCE_SEED][sample_id]["gold"])
        for sample_id in ids
    ])
    probabilities = {}
    for seed, rows in loaded.items():
        observed = np.asarray([int(rows[value]["gold"]) for value in ids])
        if not np.array_equal(labels, observed):
            raise ValueError(f"B9 gold labels do not match for seed {seed}")
        values = np.asarray(
            [rows[value]["probabilities"] for value in ids], dtype=np.float64
        )
        if values.shape != (len(ids), 3) or not np.isfinite(values).all():
            raise ValueError(f"invalid B9 probabilities for seed {seed}")
        probabilities[seed] = values / np.clip(
            values.sum(1, keepdims=True), 1e-12, None
        )
    manifest = {row["id"]: row for row in read_jsonl(manifest_path)}
    missing = reference_ids - set(manifest)
    if missing:
        raise ValueError(f"B9 IDs missing from manifest: {len(missing)}")
    sources = np.asarray([
        str(manifest[value].get("source", "unknown") or "unknown")
        for value in ids
    ])
    return ids, labels, probabilities, sources


def screen(
    labels: np.ndarray,
    probabilities: dict[int, np.ndarray],
    sources: np.ndarray,
    minimum_reference_delta: float = .005,
    maximum_best_seed_drop: float = .002,
    maximum_accuracy_drop: float = .002,
    minimum_source_delta: float = -.002,
    minimum_bootstrap_probability: float = .95,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> dict:
    if tuple(sorted(probabilities)) != FROZEN_SEEDS:
        raise ValueError(f"B9 requires exactly seeds {FROZEN_SEEDS}")
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
    comparison = paired_comparison(
        labels, reference, ensemble, bootstrap_iterations, bootstrap_seed
    )
    by_source = source_diagnostics(
        labels, reference, ensemble, np.asarray(sources)
    )
    best_seed_delta = (
        ensemble_metrics["macro_f1"] - strongest_metrics["macro_f1"]
    )
    gate = {
        "macro_f1_delta_vs_seed42_at_least_minimum": (
            comparison["macro_f1_delta"] >= minimum_reference_delta
        ),
        "ensemble_within_margin_of_strongest_seed": (
            best_seed_delta >= -maximum_best_seed_drop
        ),
        "accuracy_within_noninferiority_margin": (
            comparison["accuracy_delta"] >= -maximum_accuracy_drop
        ),
        "bootstrap_probability_at_least_minimum": (
            comparison["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
        "help_exceeds_harm": comparison["helpful"] > comparison["harmful"],
        "all_sources_within_noninferiority_margin": all(
            row["macro_f1_delta"] >= minimum_source_delta
            for row in by_source.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B9_train_only_fold0_fixed_seed_anchor_ensemble",
        "samples": int(len(labels)),
        "frozen_seeds": list(FROZEN_SEEDS),
        "reference_seed": REFERENCE_SEED,
        "per_seed": per_seed,
        "ensemble": ensemble_metrics,
        "strongest_constituent": {
            "seed": int(strongest_seed), **strongest_metrics,
        },
        "ensemble_minus_strongest_macro_f1": best_seed_delta,
        "comparison_vs_seed42": comparison,
        "source_diagnostics_vs_seed42": by_source,
        "promotion_gate": gate,
        "settings": {
            "combination": "unweighted_arithmetic_probability_mean",
            "minimum_reference_delta": minimum_reference_delta,
            "maximum_best_seed_drop": maximum_best_seed_drop,
            "maximum_accuracy_drop": maximum_accuracy_drop,
            "minimum_source_delta": minimum_source_delta,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "warning": (
            "Fold 0 is development evidence only. If this passes, freeze the "
            "three seeds and arithmetic mean before folds 1--4."
        ),
        "official_validation_used": False,
        "test_split_used": False,
        "confirmation_required_on_folds": [1, 2, 3, 4],
    }


def parse_runs(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--run must be SEED=OUTPUT_DIR")
        seed, path = value.split("=", 1)
        result[int(seed)] = Path(path)
    if tuple(sorted(result)) != FROZEN_SEEDS:
        raise ValueError(f"--run requires exactly seeds {FROZEN_SEEDS}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=None)
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--minimum-reference-delta", type=float, default=.005)
    parser.add_argument("--maximum-best-seed-drop", type=float, default=.002)
    parser.add_argument("--maximum-accuracy-drop", type=float, default=.002)
    parser.add_argument("--minimum-source-delta", type=float, default=-.002)
    parser.add_argument("--minimum-bootstrap-probability", type=float,
                        default=.95)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b9_fold0_anchor_ensemble.json"))
    args = parser.parse_args()
    if args.fold != 0:
        raise ValueError("B9 selection is restricted to fold 0")
    paths = parse_runs(args.run) if args.run else {
        13: Path("outputs/mocheg_b9_oof/fold_0/seed_13"),
        42: Path("outputs/mocheg_b6c_oof/fold_0/anchor"),
        87: Path("outputs/mocheg_b9_oof/fold_0/seed_87"),
    }
    for seed, path in paths.items():
        validate_run(path, args.fold, f"seed_{seed}")
    _, labels, probabilities, sources = aligned_runs(
        {seed: path / "val_predictions.jsonl"
         for seed, path in paths.items()}, args.manifest
    )
    result = screen(
        labels, probabilities, sources,
        args.minimum_reference_delta, args.maximum_best_seed_drop,
        args.maximum_accuracy_drop, args.minimum_source_delta,
        args.minimum_bootstrap_probability, args.bootstrap_iterations,
        args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
