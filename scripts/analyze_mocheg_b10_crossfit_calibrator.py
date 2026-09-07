"""Cross-fitted seed-disagreement calibration on train-only OOF predictions.

For every held fold, a fixed multinomial logistic calibrator is trained only
on the other confirmation folds. There is no hyperparameter search and every
reported prediction is out-of-fold at both verifier and calibrator levels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analyze_mocheg_b6_auxiliary_control import paired_comparison
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_b7_frozen_router import source_diagnostics
from scripts.analyze_mocheg_b9_anchor_ensemble import (
    FROZEN_SEEDS,
    REFERENCE_SEED,
    aligned_runs,
)
from scripts.analyze_mocheg_expert_complementarity import prediction_metrics


CROSSFIT_FOLDS = (1, 2, 3, 4)


def disagreement_features(probabilities: dict[int, np.ndarray]) -> np.ndarray:
    """Fixed seed-probability, uncertainty, and vote representation."""
    if tuple(sorted(probabilities)) != FROZEN_SEEDS:
        raise ValueError(f"B10 requires exactly seeds {FROZEN_SEEDS}")
    stacked = np.stack(
        [np.asarray(probabilities[seed], dtype=np.float64)
         for seed in FROZEN_SEEDS], axis=1
    )
    if stacked.ndim != 3 or stacked.shape[2] != 3:
        raise ValueError("B10 expects N x 3 probabilities for every seed")
    if not np.isfinite(stacked).all() or np.any(stacked < 0):
        raise ValueError("B10 probabilities must be finite and non-negative")
    stacked /= np.clip(stacked.sum(2, keepdims=True), 1e-12, None)
    clipped = np.clip(stacked, 1e-12, 1.0)
    log_probabilities = np.log(clipped).reshape(len(stacked), -1)
    mean = stacked.mean(1)
    standard_deviation = stacked.std(1)
    entropy = -(clipped * np.log(clipped)).sum(2)
    confidence = stacked.max(2)
    predictions = stacked.argmax(2)
    votes = np.stack(
        [(predictions == class_id).mean(1) for class_id in range(3)], axis=1
    )
    return np.concatenate((
        log_probabilities, mean, standard_deviation, entropy,
        confidence, votes,
    ), axis=1)


def make_calibrator(seed: int = 2026):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs",
            max_iter=2000, random_state=seed,
        ),
    )


def crossfit(folds: list[dict], seed: int = 2026) -> list[dict]:
    observed = tuple(sorted(int(row["fold"]) for row in folds))
    if observed != CROSSFIT_FOLDS:
        raise ValueError(f"B10 requires folds {CROSSFIT_FOLDS}")
    output = []
    for held in CROSSFIT_FOLDS:
        train_rows = [row for row in folds if int(row["fold"]) != held]
        held_row = next(row for row in folds if int(row["fold"]) == held)
        train_x = np.concatenate([
            disagreement_features(row["probabilities"])
            for row in train_rows
        ])
        train_y = np.concatenate([
            np.asarray(row["labels"], dtype=np.int64) for row in train_rows
        ])
        calibrator = make_calibrator(seed + held)
        calibrator.fit(train_x, train_y)
        probability = calibrator.predict_proba(
            disagreement_features(held_row["probabilities"])
        )
        classes = calibrator[-1].classes_.astype(int)
        aligned = np.zeros((len(probability), 3), dtype=np.float64)
        aligned[:, classes] = probability
        output.append({**held_row, "calibrated_probabilities": aligned})
    return output


def screen(
    folds: list[dict],
    minimum_seed42_delta: float = .005,
    minimum_ensemble_delta: float = .003,
    minimum_positive_folds: int = 3,
    maximum_accuracy_drop: float = .002,
    minimum_source_delta: float = -.002,
    minimum_bootstrap_probability: float = .95,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> dict:
    crossfitted = crossfit(folds, bootstrap_seed)
    seen: set[str] = set()
    per_fold = []
    labels_parts = []
    sources_parts = []
    seed42_parts = []
    ensemble_parts = []
    calibrated_parts = []
    prediction_rows = []
    for row in crossfitted:
        overlap = seen & set(row["ids"])
        if overlap:
            raise ValueError(f"B10 fold overlap: {len(overlap)}")
        seen.update(row["ids"])
        labels = np.asarray(row["labels"])
        reference = row["probabilities"][REFERENCE_SEED]
        ensemble = np.mean(np.stack([
            row["probabilities"][seed] for seed in FROZEN_SEEDS
        ]), axis=0)
        calibrated = row["calibrated_probabilities"]
        calibrated_metrics = prediction_metrics(labels, calibrated)
        ensemble_metrics = prediction_metrics(labels, ensemble)
        per_fold.append({
            "fold": int(row["fold"]),
            "samples": int(len(labels)),
            "seed42": prediction_metrics(labels, reference),
            "ensemble": ensemble_metrics,
            "calibrator": calibrated_metrics,
            "calibrator_minus_ensemble_macro_f1": (
                calibrated_metrics["macro_f1"]
                - ensemble_metrics["macro_f1"]
            ),
        })
        labels_parts.append(labels)
        sources_parts.append(np.asarray(row["sources"]))
        seed42_parts.append(reference)
        ensemble_parts.append(ensemble)
        calibrated_parts.append(calibrated)
        prediction_rows.extend({
            "id": sample_id,
            "fold": int(row["fold"]),
            "gold": int(gold),
            "probabilities": probability.tolist(),
        } for sample_id, gold, probability in zip(
            row["ids"], labels, calibrated
        ))
    labels = np.concatenate(labels_parts)
    sources = np.concatenate(sources_parts)
    reference = np.concatenate(seed42_parts)
    ensemble = np.concatenate(ensemble_parts)
    calibrated = np.concatenate(calibrated_parts)
    reference_metrics = prediction_metrics(labels, reference)
    ensemble_metrics = prediction_metrics(labels, ensemble)
    calibrated_metrics = prediction_metrics(labels, calibrated)
    vs_reference = paired_comparison(
        labels, reference, calibrated,
        bootstrap_iterations, bootstrap_seed,
    )
    vs_ensemble = paired_comparison(
        labels, ensemble, calibrated,
        bootstrap_iterations, bootstrap_seed + 1,
    )
    by_source = source_diagnostics(labels, ensemble, calibrated, sources)
    fold_deltas = np.asarray([
        row["calibrator_minus_ensemble_macro_f1"] for row in per_fold
    ])
    gate = {
        "delta_vs_seed42_at_least_minimum": (
            vs_reference["macro_f1_delta"] >= minimum_seed42_delta
        ),
        "delta_vs_ensemble_at_least_minimum": (
            vs_ensemble["macro_f1_delta"] >= minimum_ensemble_delta
        ),
        "positive_folds_vs_ensemble_at_least_minimum": (
            int(np.sum(fold_deltas > 0)) >= minimum_positive_folds
        ),
        "accuracy_vs_ensemble_within_noninferiority_margin": (
            vs_ensemble["accuracy_delta"] >= -maximum_accuracy_drop
        ),
        "bootstrap_vs_ensemble_at_least_minimum": (
            vs_ensemble["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
        "help_exceeds_harm_vs_ensemble": (
            vs_ensemble["helpful"] > vs_ensemble["harmful"]
        ),
        "all_sources_within_noninferiority_margin": all(
            value["macro_f1_delta"] >= minimum_source_delta
            for value in by_source.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B10_nested_crossfit_seed_disagreement_calibration",
        "folds": list(CROSSFIT_FOLDS),
        "samples": int(len(labels)),
        "features": (
            "per-seed log probabilities, mean, standard deviation, "
            "entropy, confidence, and vote fractions"
        ),
        "calibrator": {
            "type": "standardized_multinomial_logistic_regression",
            "C": 1.0, "class_weight": "balanced", "solver": "lbfgs",
        },
        "per_fold": per_fold,
        "aggregate": {
            "seed42": reference_metrics,
            "unweighted_ensemble": ensemble_metrics,
            "crossfit_calibrator": calibrated_metrics,
            "comparison_vs_seed42": vs_reference,
            "comparison_vs_unweighted_ensemble": vs_ensemble,
            "source_diagnostics_vs_unweighted_ensemble": by_source,
            "fold_delta_vs_unweighted_ensemble": {
                "mean": float(fold_deltas.mean()),
                "std": float(fold_deltas.std()),
                "values": fold_deltas.tolist(),
                "positive_folds": int(np.sum(fold_deltas > 0)),
            },
        },
        "promotion_gate": gate,
        "settings": {
            "minimum_seed42_delta": minimum_seed42_delta,
            "minimum_ensemble_delta": minimum_ensemble_delta,
            "minimum_positive_folds": minimum_positive_folds,
            "maximum_accuracy_drop": maximum_accuracy_drop,
            "minimum_source_delta": minimum_source_delta,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "prediction_rows": prediction_rows,
        "fold0_used": False,
        "official_validation_used": False,
        "test_split_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "outputs/mocheg_b9_oof"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b10_crossfit_calibrator.json"))
    parser.add_argument("--predictions", type=Path, default=Path(
        "outputs/mocheg_b10_crossfit_predictions.jsonl"))
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    folds = []
    for fold in CROSSFIT_FOLDS:
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
        folds.append({
            "fold": fold, "ids": ids, "labels": labels,
            "probabilities": probabilities, "sources": sources,
        })
    result = screen(
        folds, bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    prediction_rows = result.pop("prediction_rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
