"""Train-fold screen for prior-robust class-logit adjustment.

The Qwen verifier is frozen. Fold 0 selects two additive class-logit offsets:
supported and NEI, with refuted fixed at zero for identifiability. Source is
used only by the robustness gate and is never an inference feature. A passing
pair must be frozen before confirmation on duplicate-safe train folds 1--4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b6_auxiliary_control import paired_comparison
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_b7_frozen_router import (
    aligned_inputs,
    source_diagnostics,
)
from scripts.analyze_mocheg_expert_complementarity import prediction_metrics


SUPPORTED_BIAS_GRID = tuple(np.round(np.arange(-.30, .301, .05), 2))
NEI_BIAS_GRID = tuple(np.round(np.arange(-.30, .301, .05), 2))


def apply_logit_bias(
    probabilities: np.ndarray, supported_bias: float, nei_bias: float,
) -> np.ndarray:
    """Apply identifiable [supported, refuted, NEI] logit offsets."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("B8 expects an N x 3 probability matrix")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("B8 probabilities must be finite and non-negative")
    values = values / np.clip(values.sum(1, keepdims=True), 1e-12, None)
    logits = np.log(np.clip(values, 1e-12, 1.0))
    logits += np.asarray([supported_bias, 0.0, nei_bias])
    logits -= logits.max(1, keepdims=True)
    adjusted = np.exp(logits)
    return adjusted / adjusted.sum(1, keepdims=True)


def screen(
    labels: np.ndarray,
    anchor: np.ndarray,
    sources: np.ndarray,
    minimum_macro_f1_delta: float = .005,
    maximum_accuracy_drop: float = .002,
    minimum_source_delta: float = -.002,
    minimum_bootstrap_probability: float = .95,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    sources = np.asarray(sources)
    if len(labels) != len(anchor) or len(labels) != len(sources):
        raise ValueError("B8 labels, probabilities, and sources must align")
    anchor_metrics = prediction_metrics(labels, anchor)
    candidates = []
    for supported_bias in SUPPORTED_BIAS_GRID:
        for nei_bias in NEI_BIAS_GRID:
            adjusted = apply_logit_bias(anchor, supported_bias, nei_bias)
            metrics = prediction_metrics(labels, adjusted)
            by_source = source_diagnostics(
                labels, anchor, adjusted, sources
            )
            candidates.append({
                "supported_logit_bias": float(supported_bias),
                "refuted_logit_bias": 0.0,
                "nei_logit_bias": float(nei_bias),
                "metrics": metrics,
                "probabilities": adjusted,
                "source_diagnostics": by_source,
                "accuracy_safe": (
                    metrics["accuracy"] - anchor_metrics["accuracy"]
                    >= -maximum_accuracy_drop
                ),
                "sources_safe": all(
                    value["macro_f1_delta"] >= minimum_source_delta
                    for value in by_source.values()
                ),
            })
    eligible = [
        row for row in candidates
        if row["accuracy_safe"] and row["sources_safe"]
    ]
    if not eligible:
        raise ValueError("B8 grid has no accuracy- and source-safe candidate")
    best = max(eligible, key=lambda row: (
        row["metrics"]["macro_f1"], row["metrics"]["accuracy"],
        -abs(row["supported_logit_bias"]), -abs(row["nei_logit_bias"]),
    ))
    adjusted = best["probabilities"]
    comparison = paired_comparison(
        labels, anchor, adjusted, bootstrap_iterations, bootstrap_seed
    )
    gate = {
        "nonzero_adjustment": (
            best["supported_logit_bias"] != 0
            or best["nei_logit_bias"] != 0
        ),
        "macro_f1_delta_at_least_minimum": (
            comparison["macro_f1_delta"] >= minimum_macro_f1_delta
        ),
        "accuracy_within_noninferiority_margin": (
            comparison["accuracy_delta"] >= -maximum_accuracy_drop
        ),
        "bootstrap_probability_at_least_minimum": (
            comparison["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
        "help_exceeds_harm": (
            comparison["helpful"] > comparison["harmful"]
        ),
        "all_sources_within_noninferiority_margin": best["sources_safe"],
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B8_train_only_fold0_prior_robust_logit_adjustment",
        "samples": int(len(labels)),
        "anchor": anchor_metrics,
        "grid_size": len(candidates),
        "eligible_grid_size": len(eligible),
        "selected_adjustment": {
            "supported_logit_bias": best["supported_logit_bias"],
            "refuted_logit_bias": 0.0,
            "nei_logit_bias": best["nei_logit_bias"],
        },
        "selected_metrics": best["metrics"],
        "comparison_vs_anchor": comparison,
        "source_diagnostics": best["source_diagnostics"],
        "promotion_gate": gate,
        "settings": {
            "supported_bias_grid": SUPPORTED_BIAS_GRID,
            "nei_bias_grid": NEI_BIAS_GRID,
            "minimum_macro_f1_delta": minimum_macro_f1_delta,
            "maximum_accuracy_drop": maximum_accuracy_drop,
            "minimum_source_delta": minimum_source_delta,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "warning": (
            "Fold 0 selected the offsets and is development evidence only. "
            "Freeze both offsets before confirmation on train folds 1--4."
        ),
        "official_validation_used": False,
        "test_split_used": False,
        "confirmation_required_on_folds": [1, 2, 3, 4],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, default=Path(
        "outputs/mocheg_b6c_oof/fold_0/anchor"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--minimum-macro-f1-delta", type=float, default=.005)
    parser.add_argument("--maximum-accuracy-drop", type=float, default=.002)
    parser.add_argument("--minimum-source-delta", type=float, default=-.002)
    parser.add_argument("--minimum-bootstrap-probability", type=float,
                        default=.95)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b8_fold0_logit_adjustment.json"))
    args = parser.parse_args()
    if args.fold != 0:
        raise ValueError("B8 selection is restricted to fold 0")
    validate_run(args.anchor, args.fold, "anchor")
    _, labels, anchor, _, sources = aligned_inputs(
        args.anchor / "val_predictions.jsonl", {}, args.manifest
    )
    result = screen(
        labels, anchor, sources,
        args.minimum_macro_f1_delta, args.maximum_accuracy_drop,
        args.minimum_source_delta, args.minimum_bootstrap_probability,
        args.bootstrap_iterations, args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
