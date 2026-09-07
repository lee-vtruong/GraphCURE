"""Fresh-fold causal screen for joint-from-base constraint supervision."""
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
from scripts.prepare_mocheg_sv_folds import sha256


def validate_joint_run(path: Path, fold: int, role: str,
                       require_auxiliary: bool) -> dict:
    summary = validate_run(path, fold, role)
    if summary.get("training_from_base") is not True:
        raise ValueError(f"{role}: B12 requires training_from_base=true")
    if summary.get("fixed_checkpoint_epoch") != 3:
        raise ValueError(f"{role}: B12 requires fixed epoch 3")
    if summary.get("selected_hierarchical_weight") != 0:
        raise ValueError(f"{role}: hierarchical inference must remain disabled")
    counts = summary.get("training_task_counts", {})
    auxiliary = sum(int(counts.get(name, 0)) for name in (
        "sufficiency", "polarity", "ablation"
    ))
    if require_auxiliary and auxiliary <= 0:
        raise ValueError(f"{role}: auxiliary constraint tasks are inactive")
    if not require_auxiliary and auxiliary != 0:
        raise ValueError(f"{role}: matched control contains auxiliary tasks")
    return summary


def screen(
    labels: np.ndarray,
    anchor: np.ndarray,
    control: np.ndarray,
    joint: np.ndarray,
    sources: np.ndarray,
    minimum_anchor_delta: float = .005,
    minimum_control_delta: float = .003,
    maximum_accuracy_drop: float = .002,
    minimum_source_delta: float = -.002,
    minimum_bootstrap_probability: float = .95,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> dict:
    metrics = {
        "standard_anchor": prediction_metrics(labels, anchor),
        "compute_matched_direct_control": prediction_metrics(labels, control),
        "joint_constraints": prediction_metrics(labels, joint),
    }
    vs_anchor = paired_comparison(
        labels, anchor, joint, bootstrap_iterations, bootstrap_seed
    )
    vs_control = paired_comparison(
        labels, control, joint, bootstrap_iterations, bootstrap_seed + 1
    )
    control_vs_anchor = paired_comparison(
        labels, anchor, control, bootstrap_iterations, bootstrap_seed + 2
    )
    by_source_anchor = source_diagnostics(
        labels, anchor, joint, np.asarray(sources)
    )
    by_source_control = source_diagnostics(
        labels, control, joint, np.asarray(sources)
    )
    gate = {
        "joint_beats_standard_anchor": (
            vs_anchor["macro_f1_delta"] >= minimum_anchor_delta
        ),
        "joint_beats_compute_matched_control": (
            vs_control["macro_f1_delta"] >= minimum_control_delta
        ),
        "accuracy_vs_anchor_within_noninferiority_margin": (
            vs_anchor["accuracy_delta"] >= -maximum_accuracy_drop
        ),
        "bootstrap_vs_control_at_least_minimum": (
            vs_control["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
        "help_exceeds_harm_vs_control": (
            vs_control["helpful"] > vs_control["harmful"]
        ),
        "all_sources_safe_vs_anchor": all(
            value["macro_f1_delta"] >= minimum_source_delta
            for value in by_source_anchor.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B12_fresh_fold0_joint_from_base_constraint_screen",
        "metrics": metrics,
        "joint_vs_standard_anchor": vs_anchor,
        "joint_vs_compute_matched_control": vs_control,
        "control_vs_standard_anchor": control_vs_anchor,
        "source_diagnostics_vs_anchor": by_source_anchor,
        "source_diagnostics_vs_control": by_source_control,
        "promotion_gate": gate,
        "settings": {
            "fold_assignment_seed": 2027,
            "fixed_checkpoint_epoch": 3,
            "hierarchical_inference_weight": 0,
            "minimum_anchor_delta": minimum_anchor_delta,
            "minimum_control_delta": minimum_control_delta,
            "maximum_accuracy_drop": maximum_accuracy_drop,
            "minimum_source_delta": minimum_source_delta,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "fresh_fold_assignment": True,
        "official_validation_used": False,
        "test_split_used": False,
        "confirmation_required_on_folds": [1, 2, 3, 4],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("outputs/mocheg_b12_fresh/fold_0")
    parser.add_argument("--anchor", type=Path, default=root / "anchor")
    parser.add_argument("--control", type=Path, default=root / "direct_control")
    parser.add_argument("--joint", type=Path, default=root / "joint_constraints")
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--fold-spec", type=Path, default=Path(
        "data/processed/mocheg_b12_folds.json"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b12_fold0_screen.json"))
    args = parser.parse_args()
    if args.fold != 0:
        raise ValueError("B12 development screen is restricted to fresh fold 0")
    fold_payload = json.loads(args.fold_spec.read_text(encoding="utf-8"))
    if (
        int(fold_payload.get("seed", -1)) != 2027
        or fold_payload.get("validation_split_used") is not False
        or fold_payload.get("test_split_used") is not False
    ):
        raise ValueError("B12 requires the locked seed-2027 train-only folds")
    fold_signature = sha256(args.fold_spec)
    anchor_summary = validate_run(
        args.anchor, args.fold, "standard_anchor"
    )
    control_summary = validate_joint_run(
        args.control, args.fold, "compute_matched_direct_control", False
    )
    joint_summary = validate_joint_run(
        args.joint, args.fold, "joint_constraints", True
    )
    for role, summary in (
        ("standard_anchor", anchor_summary),
        ("compute_matched_direct_control", control_summary),
        ("joint_constraints", joint_summary),
    ):
        observed = summary.get("provenance", {}).get("fold_spec_sha256")
        if observed != fold_signature:
            raise ValueError(
                f"{role}: fold signature mismatch: {observed}"
            )
    if sum(control_summary["training_task_counts"].values()) != sum(
        joint_summary["training_task_counts"].values()
    ):
        raise ValueError("B12 control and joint task counts are not matched")
    _, labels, anchor, experts, sources = aligned_inputs(
        args.anchor / "val_predictions.jsonl",
        {
            "control": args.control / "val_predictions.jsonl",
            "joint": args.joint / "val_predictions.jsonl",
        },
        args.manifest,
    )
    result = screen(
        labels, anchor, experts["control"], experts["joint"], sources
    )
    result["training_task_counts"] = {
        "control": control_summary["training_task_counts"],
        "joint": joint_summary["training_task_counts"],
    }
    result["fold_spec_sha256"] = fold_signature
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
