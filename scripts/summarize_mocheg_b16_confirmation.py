"""Confirm the frozen B16 counterfactual curriculum on fresh folds 1--4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b12_failure_atlas import classification_metrics
from scripts.analyze_mocheg_b16_counterfactual_curriculum import (
    FOLD_SEED,
    validate_counterfactual_run,
)
from scripts.analyze_mocheg_b6_auxiliary_control import paired_comparison
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_b7_frozen_router import aligned_inputs, source_diagnostics
from scripts.prepare_mocheg_sv_folds import sha256


CONFIRMATION_FOLDS = (1, 2, 3, 4)


def summarize(
    runs: list[dict], minimum_mean_anchor_delta: float = .005,
    minimum_mean_control_delta: float = .003,
    minimum_positive_folds: int = 3,
    minimum_aggregate_anchor_delta: float = .005,
    minimum_aggregate_control_delta: float = .003,
    minimum_nei_delta: float = .01,
    maximum_supported_drop: float = .005,
    maximum_accuracy_drop: float = .002,
    minimum_source_delta: float = -.002,
    minimum_bootstrap_probability: float = .95,
    bootstrap_iterations: int = 5000, bootstrap_seed: int = 2026,
) -> dict:
    folds = tuple(sorted(int(row["fold"]) for row in runs))
    if folds != CONFIRMATION_FOLDS:
        raise ValueError(
            f"B16 confirmation requires folds {CONFIRMATION_FOLDS}"
        )
    seen_ids: set[str] = set()
    per_fold = []
    all_labels, all_sources = [], []
    all_anchor, all_control, all_candidate = [], [], []
    for row in sorted(runs, key=lambda value: value["fold"]):
        ids = set(row["ids"])
        overlap = seen_ids & ids
        if overlap:
            raise ValueError(f"B16 confirmation fold overlap: {len(overlap)}")
        seen_ids.update(ids)
        labels = np.asarray(row["labels"])
        anchor = np.asarray(row["anchor"])
        control = np.asarray(row["control"])
        candidate = np.asarray(row["candidate"])
        versus_anchor = paired_comparison(
            labels, anchor, candidate, bootstrap_iterations,
            bootstrap_seed + int(row["fold"]),
        )
        versus_control = paired_comparison(
            labels, control, candidate, bootstrap_iterations,
            bootstrap_seed + 100 + int(row["fold"]),
        )
        per_fold.append({
            "fold": int(row["fold"]),
            "samples": int(len(labels)),
            "anchor": classification_metrics(labels, anchor),
            "matched_control": classification_metrics(labels, control),
            "candidate": classification_metrics(labels, candidate),
            "candidate_vs_anchor_macro_f1": versus_anchor["macro_f1_delta"],
            "candidate_vs_control_macro_f1": versus_control["macro_f1_delta"],
            "helpful_vs_anchor": versus_anchor["helpful"],
            "harmful_vs_anchor": versus_anchor["harmful"],
        })
        all_labels.append(labels)
        all_sources.append(np.asarray(row["sources"]))
        all_anchor.append(anchor)
        all_control.append(control)
        all_candidate.append(candidate)
    labels = np.concatenate(all_labels)
    sources = np.concatenate(all_sources)
    anchor = np.concatenate(all_anchor)
    control = np.concatenate(all_control)
    candidate = np.concatenate(all_candidate)
    anchor_metrics = classification_metrics(labels, anchor)
    control_metrics = classification_metrics(labels, control)
    candidate_metrics = classification_metrics(labels, candidate)
    versus_anchor = paired_comparison(
        labels, anchor, candidate, bootstrap_iterations, bootstrap_seed
    )
    versus_control = paired_comparison(
        labels, control, candidate, bootstrap_iterations, bootstrap_seed + 1
    )
    by_source = source_diagnostics(labels, anchor, candidate, sources)
    class_delta = {
        name: candidate_metrics["class_f1"][name]
        - anchor_metrics["class_f1"][name]
        for name in ("supported", "refuted", "nei")
    }
    anchor_deltas = np.asarray([
        row["candidate_vs_anchor_macro_f1"] for row in per_fold
    ])
    control_deltas = np.asarray([
        row["candidate_vs_control_macro_f1"] for row in per_fold
    ])
    gate = {
        "mean_anchor_delta_at_least_0_005": (
            float(anchor_deltas.mean()) >= minimum_mean_anchor_delta
        ),
        "mean_control_delta_at_least_0_003": (
            float(control_deltas.mean()) >= minimum_mean_control_delta
        ),
        "positive_anchor_folds_at_least_3": (
            int(np.sum(anchor_deltas > 0)) >= minimum_positive_folds
        ),
        "positive_control_folds_at_least_3": (
            int(np.sum(control_deltas > 0)) >= minimum_positive_folds
        ),
        "aggregate_anchor_delta_at_least_0_005": (
            versus_anchor["macro_f1_delta"] >= minimum_aggregate_anchor_delta
        ),
        "aggregate_control_delta_at_least_0_003": (
            versus_control["macro_f1_delta"] >= minimum_aggregate_control_delta
        ),
        "aggregate_nei_f1_delta_at_least_0_010": (
            class_delta["nei"] >= minimum_nei_delta
        ),
        "aggregate_supported_f1_within_margin": (
            class_delta["supported"] >= -maximum_supported_drop
        ),
        "aggregate_accuracy_within_margin": (
            versus_anchor["accuracy_delta"] >= -maximum_accuracy_drop
        ),
        "bootstrap_vs_anchor_at_least_0_95": (
            versus_anchor["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
        "bootstrap_vs_control_at_least_0_95": (
            versus_control["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
        "aggregate_help_exceeds_harm": (
            versus_anchor["helpful"] > versus_anchor["harmful"]
        ),
        "all_sources_safe": all(
            row["macro_f1_delta"] >= minimum_source_delta
            for row in by_source.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B16_frozen_fresh_folds_1_to_4_confirmation",
        "folds": list(CONFIRMATION_FOLDS),
        "samples": int(len(labels)),
        "per_fold": per_fold,
        "paired_fold_delta_vs_anchor": {
            "mean": float(anchor_deltas.mean()),
            "std": float(anchor_deltas.std()),
            "values": anchor_deltas.tolist(),
            "positive_folds": int(np.sum(anchor_deltas > 0)),
        },
        "paired_fold_delta_vs_control": {
            "mean": float(control_deltas.mean()),
            "std": float(control_deltas.std()),
            "values": control_deltas.tolist(),
            "positive_folds": int(np.sum(control_deltas > 0)),
        },
        "aggregate": {
            "anchor": anchor_metrics,
            "matched_control": control_metrics,
            "candidate": candidate_metrics,
            "candidate_vs_anchor": versus_anchor,
            "candidate_vs_matched_control": versus_control,
            "class_f1_delta_vs_anchor": class_delta,
            "source_diagnostics_vs_anchor": by_source,
        },
        "promotion_gate": gate,
        "settings": {
            "fold_assignment_seed": FOLD_SEED,
            "counterfactual_fraction": .15,
            "counterfactual_epochs": 1,
            "fixed_checkpoint_epoch": 3,
            "minimum_mean_anchor_delta": minimum_mean_anchor_delta,
            "minimum_mean_control_delta": minimum_mean_control_delta,
            "minimum_positive_folds": minimum_positive_folds,
            "minimum_aggregate_anchor_delta": minimum_aggregate_anchor_delta,
            "minimum_aggregate_control_delta": minimum_aggregate_control_delta,
            "minimum_nei_delta": minimum_nei_delta,
            "maximum_supported_drop": maximum_supported_drop,
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
        "# MOCHEG B16 frozen counterfactual-curriculum confirmation", "",
        "Fold 0 used for confirmation: **no**  ",
        "Official validation used: **no**  ", "Test used: **no**", "",
        "Frozen counterfactual fraction: `0.15` in epoch 1", "",
        "| Fold | Anchor F1 | Control F1 | B16 F1 | vs Anchor | vs Control |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_fold"]:
        lines.append(
            f"| {row['fold']} | {row['anchor']['macro_f1']:.4f} | "
            f"{row['matched_control']['macro_f1']:.4f} | "
            f"{row['candidate']['macro_f1']:.4f} | "
            f"{row['candidate_vs_anchor_macro_f1']:+.4f} | "
            f"{row['candidate_vs_control_macro_f1']:+.4f} |"
        )
    anchor_delta = result["paired_fold_delta_vs_anchor"]
    control_delta = result["paired_fold_delta_vs_control"]
    aggregate = result["aggregate"]
    lines.extend([
        "", "## Aggregate", "",
        f"- Fold delta vs anchor: {anchor_delta['mean']:+.6f} +/- "
        f"{anchor_delta['std']:.6f}",
        f"- Fold delta vs control: {control_delta['mean']:+.6f} +/- "
        f"{control_delta['std']:.6f}",
        f"- Anchor Macro-F1: {aggregate['anchor']['macro_f1']:.6f}",
        f"- Matched-control Macro-F1: "
        f"{aggregate['matched_control']['macro_f1']:.6f}",
        f"- B16 Macro-F1: {aggregate['candidate']['macro_f1']:.6f}",
        f"- B16 vs anchor: "
        f"{aggregate['candidate_vs_anchor']['macro_f1_delta']:+.6f}",
        f"- B16 vs control: "
        f"{aggregate['candidate_vs_matched_control']['macro_f1_delta']:+.6f}",
        f"- Promotion gate: "
        f"**{'pass' if result['promotion_gate']['passed'] else 'fail'}**",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "outputs/mocheg_b16_fresh"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--fold-spec", type=Path, default=Path(
        "data/processed/mocheg_b16_folds.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b16_confirmation.json"))
    parser.add_argument("--markdown", type=Path, default=Path(
        "outputs/mocheg_b16_confirmation.md"))
    args = parser.parse_args()
    fold_payload = json.loads(args.fold_spec.read_text(encoding="utf-8"))
    if (
        int(fold_payload.get("seed", -1)) != FOLD_SEED
        or fold_payload.get("validation_split_used") is not False
        or fold_payload.get("test_split_used") is not False
        or fold_payload.get("manifest_sha256") != sha256(args.manifest)
    ):
        raise ValueError("expected locked seed-2039 train-only folds")
    fold_signature = sha256(args.fold_spec)
    runs = []
    for fold in CONFIRMATION_FOLDS:
        fold_root = args.root / f"fold_{fold}"
        paths = {
            "anchor": fold_root / "anchor",
            "control": fold_root / "direct_control",
            "candidate": fold_root / "counterfactual",
        }
        summaries = {
            role: validate_run(path, fold, f"fold_{fold}_{role}")
            for role, path in paths.items()
        }
        validate_counterfactual_run(summaries["candidate"], fold)
        control = summaries["control"]
        if (
            control.get("training_from_base") is not True
            or control.get("fixed_checkpoint_epoch") != 3
            or control.get("selected_hierarchical_weight") != 0
            or set(control.get("training_task_counts", {})) != {"verdict"}
        ):
            raise ValueError(f"fold {fold}: invalid matched direct control")
        for role, summary in summaries.items():
            if summary.get("provenance", {}).get(
                "fold_spec_sha256"
            ) != fold_signature:
                raise ValueError(f"fold {fold} {role}: signature mismatch")
        ids, labels, anchor, others, sources = aligned_inputs(
            paths["anchor"] / "val_predictions.jsonl",
            {
                "control": paths["control"] / "val_predictions.jsonl",
                "candidate": paths["candidate"] / "val_predictions.jsonl",
            },
            args.manifest,
        )
        expected = set(next(
            row["val_ids"] for row in fold_payload["folds"]
            if int(row["fold"]) == fold
        ))
        if set(ids) != expected:
            raise ValueError(f"fold {fold}: held prediction IDs mismatch")
        runs.append({
            "fold": fold, "ids": ids, "labels": labels,
            "sources": sources, "anchor": anchor,
            "control": others["control"],
            "candidate": others["candidate"],
        })
    result = summarize(runs)
    result["fold_spec_sha256"] = fold_signature
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
