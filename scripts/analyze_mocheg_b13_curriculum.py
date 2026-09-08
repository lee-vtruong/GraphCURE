"""Frozen fold-0 screen for the B13 compute-neutral curriculum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b6_auxiliary_control import paired_comparison
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_b7_frozen_router import aligned_inputs, source_diagnostics
from scripts.analyze_mocheg_expert_complementarity import prediction_metrics
from scripts.prepare_mocheg_sv_folds import sha256


METHOD = "GraphCURE-B13-compute-neutral-constraint-curriculum"


def validate_curriculum(summary: dict, fold: int) -> dict:
    if summary.get("method") != METHOD:
        raise ValueError(f"unexpected method: {summary.get('method')}")
    if summary.get("fold") != fold:
        raise ValueError("candidate fold mismatch")
    if summary.get("training_from_base") is not True:
        raise ValueError("B13 must train from the base model")
    if summary.get("fixed_checkpoint_epoch") != 3:
        raise ValueError("B13 requires fixed checkpoint epoch 3")
    if summary.get("selected_hierarchical_weight") != 0:
        raise ValueError("B13 direct verdict inference must remain frozen")
    settings = summary.get("settings", {})
    if settings.get("compute_neutral_curriculum") is not True:
        raise ValueError("compute-neutral curriculum is inactive")
    epochs = summary.get("epoch_training_task_counts", [])
    if len(epochs) != 3:
        raise ValueError("B13 requires three recorded training epochs")
    totals = {int(row.get("total", -1)) for row in epochs}
    if len(totals) != 1 or next(iter(totals)) <= 0:
        raise ValueError("B13 epoch sizes are not compute-neutral")
    first = epochs[0]
    if int(first.get("sufficiency", 0)) <= 0 or int(first.get("polarity", 0)) <= 0:
        raise ValueError("B13 first epoch has no auxiliary curriculum")
    for row in epochs[1:]:
        if set(row) - {"epoch", "verdict", "total", "target_counts"}:
            raise ValueError("B13 recovery epochs must be verdict-only")
        if int(row.get("verdict", 0)) != int(row["total"]):
            raise ValueError("B13 recovery epoch is not verdict-only")
    return {
        "epoch_size": next(iter(totals)),
        "epochs": epochs,
        "optimizer_update_budget_matches_anchor": True,
        "auxiliary_only_in_first_epoch": True,
        "verdict_recovery_epochs": 2,
    }


def screen(
    labels: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    sources: np.ndarray,
    minimum_delta: float = .005,
    maximum_accuracy_drop: float = .002,
    minimum_source_delta: float = -.002,
    minimum_bootstrap_probability: float = .95,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> dict:
    comparison = paired_comparison(
        labels, anchor, candidate, bootstrap_iterations, bootstrap_seed
    )
    source = source_diagnostics(labels, anchor, candidate, sources)
    gate = {
        "macro_f1_delta_at_least_minimum": (
            comparison["macro_f1_delta"] >= minimum_delta
        ),
        "accuracy_within_noninferiority_margin": (
            comparison["accuracy_delta"] >= -maximum_accuracy_drop
        ),
        "bootstrap_probability_at_least_minimum": (
            comparison["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
        "help_exceeds_harm": comparison["helpful"] > comparison["harmful"],
        "all_sources_safe": all(
            value["macro_f1_delta"] >= minimum_source_delta
            for value in source.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B13_fresh_fold0_compute_neutral_curriculum_screen",
        "metrics": {
            "anchor": prediction_metrics(labels, anchor),
            "candidate": prediction_metrics(labels, candidate),
        },
        "candidate_vs_anchor": comparison,
        "source_diagnostics": source,
        "promotion_gate": gate,
        "settings": {
            "minimum_delta": minimum_delta,
            "maximum_accuracy_drop": maximum_accuracy_drop,
            "minimum_source_delta": minimum_source_delta,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "official_validation_used": False,
        "test_split_used": False,
        "confirmation_required_on_folds": [1, 2, 3, 4],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, default=Path(
        "outputs/mocheg_b12_fresh/fold_0/anchor"))
    parser.add_argument("--candidate", type=Path, default=Path(
        "outputs/mocheg_b13_fresh/fold_0/curriculum"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--fold-spec", type=Path, default=Path(
        "data/processed/mocheg_b12_folds.json"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b13_fold0_screen.json"))
    args = parser.parse_args()
    if args.fold != 0:
        raise ValueError("B13 development is restricted to fresh fold 0")
    fold_payload = json.loads(args.fold_spec.read_text(encoding="utf-8"))
    if (
        int(fold_payload.get("seed", -1)) != 2027
        or fold_payload.get("validation_split_used") is not False
        or fold_payload.get("test_split_used") is not False
    ):
        raise ValueError("B13 requires the locked seed-2027 train-only folds")
    fold_signature = sha256(args.fold_spec)
    anchor_summary = validate_run(args.anchor, args.fold, "anchor")
    candidate_summary = validate_run(args.candidate, args.fold, "candidate")
    curriculum = validate_curriculum(candidate_summary, args.fold)
    for role, summary in (("anchor", anchor_summary), ("candidate", candidate_summary)):
        observed = summary.get("provenance", {}).get("fold_spec_sha256")
        if observed != fold_signature:
            raise ValueError(f"{role}: fold signature mismatch: {observed}")
    _, labels, anchor, candidates, sources = aligned_inputs(
        args.anchor / "val_predictions.jsonl",
        {"candidate": args.candidate / "val_predictions.jsonl"},
        args.manifest,
    )
    result = screen(labels, anchor, candidates["candidate"], sources)
    result["curriculum_audit"] = curriculum
    result["fold_spec_sha256"] = fold_signature
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
