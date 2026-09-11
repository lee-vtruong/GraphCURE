"""Frozen fresh-fold screen for B16 counterfactual verdict curriculum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b12_failure_atlas import classification_metrics
from scripts.analyze_mocheg_b6_auxiliary_control import paired_comparison
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_b7_frozen_router import aligned_inputs, source_diagnostics
from scripts.prepare_mocheg_sv_folds import sha256


METHOD = "GraphCURE-B16-counterfactual-verdict-curriculum"
FOLD_SEED = 2039
FRACTION = .15


def validate_counterfactual_run(summary: dict, fold: int) -> dict:
    if summary.get("method") != METHOD:
        raise ValueError(f"unexpected method: {summary.get('method')}")
    if summary.get("fold") != fold:
        raise ValueError("B16 fold mismatch")
    if summary.get("training_from_base") is not True:
        raise ValueError("B16 must train from base")
    if summary.get("fixed_checkpoint_epoch") != 3:
        raise ValueError("B16 requires fixed checkpoint epoch 3")
    if summary.get("selected_hierarchical_weight") != 0:
        raise ValueError("B16 requires direct verdict inference")
    if summary.get("counterfactual_verdict_curriculum") is not True:
        raise ValueError("counterfactual verdict curriculum is inactive")
    if abs(float(summary.get("counterfactual_verdict_fraction", -1)) - FRACTION) > 1e-12:
        raise ValueError("B16 counterfactual fraction is not frozen at 0.15")
    if summary.get("counterfactual_verdict_epochs") != 1:
        raise ValueError("B16 requires one counterfactual epoch")
    epochs = summary.get("epoch_training_task_counts", [])
    if len(epochs) != 3:
        raise ValueError("B16 requires three recorded epochs")
    totals = {int(row.get("total", -1)) for row in epochs}
    if len(totals) != 1 or next(iter(totals)) <= 0:
        raise ValueError("B16 is not compute-neutral across epochs")
    first = epochs[0]
    expected_counterfactual = int(round(next(iter(totals)) * FRACTION))
    if int(first.get("counterfactual_verdict", 0)) != expected_counterfactual:
        raise ValueError("B16 epoch 1 counterfactual count mismatch")
    if set(first.get("target_counts", {}).get(
        "counterfactual_verdict", {}
    )) != {"C"}:
        raise ValueError("B16 omissions must target the verdict token C")
    for row in epochs[1:]:
        if int(row.get("verdict", 0)) != int(row["total"]):
            raise ValueError("B16 recovery epochs must be verdict-only")
        if int(row.get("counterfactual_verdict", 0)) != 0:
            raise ValueError("counterfactual rows leaked into recovery epochs")
    return {
        "epoch_size": next(iter(totals)),
        "counterfactual_examples_epoch_1": expected_counterfactual,
        "counterfactual_fraction": FRACTION,
        "counterfactual_target": "C (NEI)",
        "verdict_recovery_epochs": 2,
        "compute_neutral": True,
    }


def screen(
    labels: np.ndarray, anchor: np.ndarray, control: np.ndarray,
    candidate: np.ndarray,
    sources: np.ndarray, minimum_delta: float = .005,
    minimum_nei_delta: float = .01, maximum_supported_drop: float = .005,
    maximum_accuracy_drop: float = .002, minimum_source_delta: float = -.002,
    minimum_bootstrap_probability: float = .95,
    bootstrap_iterations: int = 5000, bootstrap_seed: int = 2026,
) -> dict:
    anchor_metrics = classification_metrics(labels, anchor)
    candidate_metrics = classification_metrics(labels, candidate)
    paired = paired_comparison(
        labels, anchor, candidate, bootstrap_iterations, bootstrap_seed
    )
    versus_control = paired_comparison(
        labels, control, candidate, bootstrap_iterations, bootstrap_seed + 1
    )
    sources_report = source_diagnostics(labels, anchor, candidate, sources)
    class_delta = {
        name: candidate_metrics["class_f1"][name]
        - anchor_metrics["class_f1"][name]
        for name in ("supported", "refuted", "nei")
    }
    gate = {
        "macro_f1_delta_vs_anchor_at_least_minimum": (
            paired["macro_f1_delta"] >= minimum_delta
        ),
        "macro_f1_delta_vs_control_at_least_0_003": (
            versus_control["macro_f1_delta"] >= .003
        ),
        "nei_f1_delta_at_least_minimum": (
            class_delta["nei"] >= minimum_nei_delta
        ),
        "supported_f1_within_noninferiority_margin": (
            class_delta["supported"] >= -maximum_supported_drop
        ),
        "accuracy_within_noninferiority_margin": (
            paired["accuracy_delta"] >= -maximum_accuracy_drop
        ),
        "bootstrap_probability_at_least_minimum": (
            paired["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
        "bootstrap_vs_control_at_least_0_95": (
            versus_control["bootstrap"]["probability_delta_positive"] >= .95
        ),
        "help_exceeds_harm": paired["helpful"] > paired["harmful"],
        "all_sources_safe": all(
            row["macro_f1_delta"] >= minimum_source_delta
            for row in sources_report.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B16_fresh_fold0_counterfactual_verdict_screen",
        "metrics": {
            "standard_anchor": anchor_metrics,
            "matched_direct_control": classification_metrics(labels, control),
            "candidate": candidate_metrics,
        },
        "candidate_vs_anchor": paired,
        "candidate_vs_matched_control": versus_control,
        "class_f1_delta": class_delta,
        "source_diagnostics": sources_report,
        "promotion_gate": gate,
        "settings": {
            "fold_assignment_seed": FOLD_SEED,
            "counterfactual_fraction": FRACTION,
            "counterfactual_epochs": 1,
            "verdict_recovery_epochs": 2,
            "fixed_checkpoint_epoch": 3,
            "minimum_delta": minimum_delta,
            "minimum_nei_delta": minimum_nei_delta,
            "maximum_supported_drop": maximum_supported_drop,
            "maximum_accuracy_drop": maximum_accuracy_drop,
            "minimum_source_delta": minimum_source_delta,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
        },
        "fresh_fold_assignment": True,
        "official_validation_used": False,
        "test_split_used": False,
        "confirmation_required_on_folds": [1, 2, 3, 4],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("outputs/mocheg_b16_fresh/fold_0")
    parser.add_argument("--anchor", type=Path, default=root / "anchor")
    parser.add_argument(
        "--candidate", type=Path, default=root / "counterfactual"
    )
    parser.add_argument("--control", type=Path, default=root / "direct_control")
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--fold-spec", type=Path, default=Path(
        "data/processed/mocheg_b16_folds.json"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b16_fold0_screen.json"))
    args = parser.parse_args()
    if args.fold != 0:
        raise ValueError("B16 development is restricted to fresh fold 0")
    fold_payload = json.loads(args.fold_spec.read_text(encoding="utf-8"))
    if (
        int(fold_payload.get("seed", -1)) != FOLD_SEED
        or fold_payload.get("validation_split_used") is not False
        or fold_payload.get("test_split_used") is not False
        or fold_payload.get("manifest_sha256") != sha256(args.manifest)
    ):
        raise ValueError("B16 requires the locked seed-2039 train-only folds")
    fold_signature = sha256(args.fold_spec)
    anchor_summary = validate_run(args.anchor, args.fold, "anchor")
    control_summary = validate_run(args.control, args.fold, "control")
    candidate_summary = validate_run(args.candidate, args.fold, "candidate")
    curriculum = validate_counterfactual_run(candidate_summary, args.fold)
    if (
        control_summary.get("training_from_base") is not True
        or control_summary.get("fixed_checkpoint_epoch") != 3
        or control_summary.get("selected_hierarchical_weight") != 0
        or set(control_summary.get("training_task_counts", {})) != {"verdict"}
    ):
        raise ValueError("B16 matched control is not verdict-only from base")
    for role, summary in (("anchor", anchor_summary),
                          ("control", control_summary),
                          ("candidate", candidate_summary)):
        observed = summary.get("provenance", {}).get("fold_spec_sha256")
        if observed != fold_signature:
            raise ValueError(f"{role}: fold signature mismatch")
    _, labels, anchor, candidates, sources = aligned_inputs(
        args.anchor / "val_predictions.jsonl",
        {
            "control": args.control / "val_predictions.jsonl",
            "candidate": args.candidate / "val_predictions.jsonl",
        },
        args.manifest,
    )
    result = screen(
        labels, anchor, candidates["control"], candidates["candidate"], sources
    )
    result["curriculum_audit"] = curriculum
    result["fold_spec_sha256"] = fold_signature
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
