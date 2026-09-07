"""Nested-OOF provenance-conditioned seed-disagreement calibration."""
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
    aligned_runs,
)
from scripts.analyze_mocheg_b10_crossfit_calibrator import (
    CROSSFIT_FOLDS,
    crossfit as global_crossfit,
    disagreement_features,
    make_calibrator,
)
from scripts.analyze_mocheg_expert_complementarity import prediction_metrics


def provenance_crossfit(folds: list[dict], seed: int = 2026) -> list[dict]:
    observed = tuple(sorted(int(row["fold"]) for row in folds))
    if observed != CROSSFIT_FOLDS:
        raise ValueError(f"B11 requires folds {CROSSFIT_FOLDS}")
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
        train_sources = np.concatenate([
            np.asarray(row["sources"]).astype(str) for row in train_rows
        ])
        held_x = disagreement_features(held_row["probabilities"])
        held_sources = np.asarray(held_row["sources"]).astype(str)
        probability = np.zeros((len(held_x), 3), dtype=np.float64)
        source_models = {}
        all_sources = sorted(set(train_sources) | set(held_sources))
        for source_index, source in enumerate(all_sources):
            train_mask = train_sources == source
            held_mask = held_sources == source
            if not held_mask.any():
                continue
            classes = np.unique(train_y[train_mask])
            if classes.tolist() != [0, 1, 2]:
                raise ValueError(
                    f"B11 source {source!r} lacks all labels in training folds"
                )
            calibrator = make_calibrator(seed + held * 10 + source_index)
            calibrator.fit(train_x[train_mask], train_y[train_mask])
            predicted = calibrator.predict_proba(held_x[held_mask])
            class_ids = calibrator[-1].classes_.astype(int)
            probability[np.ix_(held_mask, class_ids)] = predicted
            source_models[source] = {
                "training_samples": int(train_mask.sum()),
                "held_samples": int(held_mask.sum()),
            }
        if np.any(probability.sum(1) == 0):
            raise RuntimeError("B11 left held rows without provenance model")
        output.append({
            **held_row,
            "calibrated_probabilities": probability,
            "source_models": source_models,
        })
    return output


def screen(
    folds: list[dict],
    minimum_ensemble_delta: float = .003,
    minimum_global_delta: float = .002,
    minimum_positive_folds: int = 3,
    maximum_accuracy_drop: float = .002,
    minimum_source_delta: float = 0.0,
    minimum_bootstrap_probability: float = .95,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> dict:
    conditioned = provenance_crossfit(folds, bootstrap_seed)
    global_rows = {
        int(row["fold"]): row for row in global_crossfit(folds, bootstrap_seed)
    }
    seen: set[str] = set()
    per_fold = []
    labels_parts = []
    source_parts = []
    ensemble_parts = []
    global_parts = []
    conditioned_parts = []
    prediction_rows = []
    for row in conditioned:
        fold = int(row["fold"])
        overlap = seen & set(row["ids"])
        if overlap:
            raise ValueError(f"B11 fold overlap: {len(overlap)}")
        seen.update(row["ids"])
        labels = np.asarray(row["labels"])
        ensemble = np.mean(np.stack([
            row["probabilities"][seed] for seed in FROZEN_SEEDS
        ]), axis=0)
        global_probability = global_rows[fold]["calibrated_probabilities"]
        provenance_probability = row["calibrated_probabilities"]
        ensemble_metrics = prediction_metrics(labels, ensemble)
        global_metrics = prediction_metrics(labels, global_probability)
        provenance_metrics = prediction_metrics(labels, provenance_probability)
        per_fold.append({
            "fold": fold,
            "samples": int(len(labels)),
            "ensemble": ensemble_metrics,
            "global_calibrator": global_metrics,
            "provenance_calibrator": provenance_metrics,
            "provenance_minus_ensemble_macro_f1": (
                provenance_metrics["macro_f1"]
                - ensemble_metrics["macro_f1"]
            ),
            "provenance_minus_global_macro_f1": (
                provenance_metrics["macro_f1"] - global_metrics["macro_f1"]
            ),
            "source_models": row["source_models"],
        })
        labels_parts.append(labels)
        source_parts.append(np.asarray(row["sources"]).astype(str))
        ensemble_parts.append(ensemble)
        global_parts.append(global_probability)
        conditioned_parts.append(provenance_probability)
        prediction_rows.extend({
            "id": sample_id, "fold": fold, "gold": int(gold),
            "source": str(source), "probabilities": probability.tolist(),
        } for sample_id, gold, source, probability in zip(
            row["ids"], labels, row["sources"], provenance_probability
        ))
    labels = np.concatenate(labels_parts)
    sources = np.concatenate(source_parts)
    ensemble = np.concatenate(ensemble_parts)
    global_probability = np.concatenate(global_parts)
    provenance_probability = np.concatenate(conditioned_parts)
    ensemble_metrics = prediction_metrics(labels, ensemble)
    global_metrics = prediction_metrics(labels, global_probability)
    provenance_metrics = prediction_metrics(labels, provenance_probability)
    vs_ensemble = paired_comparison(
        labels, ensemble, provenance_probability,
        bootstrap_iterations, bootstrap_seed,
    )
    vs_global = paired_comparison(
        labels, global_probability, provenance_probability,
        bootstrap_iterations, bootstrap_seed + 1,
    )
    by_source = source_diagnostics(
        labels, ensemble, provenance_probability, sources
    )
    fold_deltas = np.asarray([
        row["provenance_minus_ensemble_macro_f1"] for row in per_fold
    ])
    gate = {
        "delta_vs_ensemble_at_least_minimum": (
            vs_ensemble["macro_f1_delta"] >= minimum_ensemble_delta
        ),
        "delta_vs_global_calibrator_at_least_minimum": (
            vs_global["macro_f1_delta"] >= minimum_global_delta
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
        "every_source_improves_over_ensemble": all(
            value["macro_f1_delta"] >= minimum_source_delta
            for value in by_source.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B11_nested_oof_provenance_conditioned_calibration",
        "folds": list(CROSSFIT_FOLDS),
        "samples": int(len(labels)),
        "inference_feature": "claim provenance/source",
        "per_fold": per_fold,
        "aggregate": {
            "unweighted_ensemble": ensemble_metrics,
            "global_calibrator": global_metrics,
            "provenance_calibrator": provenance_metrics,
            "comparison_vs_unweighted_ensemble": vs_ensemble,
            "comparison_vs_global_calibrator": vs_global,
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
            "per_source_model": (
                "standardized multinomial logistic regression"
            ),
            "C": 1.0, "class_weight": "balanced", "solver": "lbfgs",
            "minimum_ensemble_delta": minimum_ensemble_delta,
            "minimum_global_delta": minimum_global_delta,
            "minimum_positive_folds": minimum_positive_folds,
            "maximum_accuracy_drop": maximum_accuracy_drop,
            "minimum_source_delta": minimum_source_delta,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "prediction_rows": prediction_rows,
        "exploratory_after_b10": True,
        "fresh_split_confirmation_required": True,
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
        "outputs/mocheg_b11_provenance_calibrator.json"))
    parser.add_argument("--predictions", type=Path, default=Path(
        "outputs/mocheg_b11_provenance_predictions.jsonl"))
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
    result = screen(folds)
    predictions = result.pop("prediction_rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in predictions),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
