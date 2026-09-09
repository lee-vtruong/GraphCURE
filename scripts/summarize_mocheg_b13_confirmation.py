"""Confirm the frozen B13 anchor/hierarchical blend on train-only folds 1--4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b12_failure_atlas import normalized_probabilities
from scripts.analyze_mocheg_b13_curriculum import validate_curriculum
from scripts.analyze_mocheg_b6_auxiliary_control import paired_comparison
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_b7_frozen_router import source_diagnostics
from scripts.analyze_mocheg_expert_complementarity import (
    prediction_metrics,
    read_predictions,
)
from scripts.prepare_mocheg_sv_folds import sha256
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_qwen3_hierarchical_lora import hierarchical_probabilities


CONFIRMATION_FOLDS = (1, 2, 3, 4)
FROZEN_HIERARCHICAL_WEIGHT = .11


def summarize(
    runs: list[dict],
    hierarchical_weight: float = FROZEN_HIERARCHICAL_WEIGHT,
    minimum_mean_delta: float = .005,
    minimum_positive_folds: int = 3,
    minimum_aggregate_delta: float = .005,
    maximum_accuracy_drop: float = .002,
    minimum_source_delta: float = -.002,
    minimum_bootstrap_probability: float = .95,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> dict:
    if abs(hierarchical_weight - FROZEN_HIERARCHICAL_WEIGHT) > 1e-12:
        raise ValueError("B13 confirmation hierarchical weight is frozen at 0.11")
    folds = tuple(sorted(int(row["fold"]) for row in runs))
    if folds != CONFIRMATION_FOLDS:
        raise ValueError(
            f"B13 confirmation requires folds {CONFIRMATION_FOLDS}"
        )
    seen_ids: set[str] = set()
    per_fold = []
    all_labels, all_sources = [], []
    all_anchor, all_direct, all_hierarchical, all_fused = [], [], [], []
    for row in sorted(runs, key=lambda value: value["fold"]):
        overlap = seen_ids & set(row["ids"])
        if overlap:
            raise ValueError(f"B13 confirmation fold overlap: {len(overlap)}")
        seen_ids.update(row["ids"])
        labels = np.asarray(row["labels"])
        anchor = np.asarray(row["anchor"])
        direct = np.asarray(row["direct"])
        hierarchical = np.asarray(row["hierarchical"])
        fused = (
            (1 - hierarchical_weight) * anchor
            + hierarchical_weight * hierarchical
        )
        comparison = paired_comparison(
            labels, anchor, fused, bootstrap_iterations,
            bootstrap_seed + int(row["fold"]),
        )
        per_fold.append({
            "fold": int(row["fold"]),
            "samples": int(len(labels)),
            "anchor": prediction_metrics(labels, anchor),
            "direct": prediction_metrics(labels, direct),
            "hierarchical": prediction_metrics(labels, hierarchical),
            "frozen_blend": prediction_metrics(labels, fused),
            "macro_f1_delta": comparison["macro_f1_delta"],
            "accuracy_delta": comparison["accuracy_delta"],
            "helpful": comparison["helpful"],
            "harmful": comparison["harmful"],
        })
        all_labels.append(labels)
        all_sources.append(np.asarray(row["sources"]))
        all_anchor.append(anchor)
        all_direct.append(direct)
        all_hierarchical.append(hierarchical)
        all_fused.append(fused)
    labels = np.concatenate(all_labels)
    sources = np.concatenate(all_sources)
    anchor = np.concatenate(all_anchor)
    direct = np.concatenate(all_direct)
    hierarchical = np.concatenate(all_hierarchical)
    fused = np.concatenate(all_fused)
    aggregate_comparison = paired_comparison(
        labels, anchor, fused, bootstrap_iterations, bootstrap_seed
    )
    by_source = source_diagnostics(labels, anchor, fused, sources)
    deltas = np.asarray([row["macro_f1_delta"] for row in per_fold])
    gate = {
        "mean_fold_delta_at_least_minimum": (
            float(deltas.mean()) >= minimum_mean_delta
        ),
        "positive_folds_at_least_minimum": (
            int(np.sum(deltas > 0)) >= minimum_positive_folds
        ),
        "aggregate_delta_at_least_minimum": (
            aggregate_comparison["macro_f1_delta"] >= minimum_aggregate_delta
        ),
        "aggregate_bootstrap_probability_at_least_minimum": (
            aggregate_comparison["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
        "aggregate_accuracy_within_noninferiority_margin": (
            aggregate_comparison["accuracy_delta"] >= -maximum_accuracy_drop
        ),
        "aggregate_help_exceeds_harm": (
            aggregate_comparison["helpful"] > aggregate_comparison["harmful"]
        ),
        "all_sources_within_noninferiority_margin": all(
            value["macro_f1_delta"] >= minimum_source_delta
            for value in by_source.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "B13_frozen_train_only_folds_1_to_4_confirmation",
        "folds": list(CONFIRMATION_FOLDS),
        "frozen_hierarchical_weight": hierarchical_weight,
        "samples": int(len(labels)),
        "per_fold": per_fold,
        "paired_fold_macro_f1_delta": {
            "mean": float(deltas.mean()),
            "std": float(deltas.std()),
            "values": deltas.tolist(),
            "positive_folds": int(np.sum(deltas > 0)),
        },
        "aggregate": {
            "anchor": prediction_metrics(labels, anchor),
            "direct": prediction_metrics(labels, direct),
            "hierarchical": prediction_metrics(labels, hierarchical),
            "frozen_blend": prediction_metrics(labels, fused),
            "comparison_vs_anchor": aggregate_comparison,
            "source_diagnostics_vs_anchor": by_source,
        },
        "promotion_gate": gate,
        "settings": {
            "minimum_mean_delta": minimum_mean_delta,
            "minimum_positive_folds": minimum_positive_folds,
            "minimum_aggregate_delta": minimum_aggregate_delta,
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
        "# MOCHEG B13 frozen hierarchical-blend confirmation", "",
        "Official validation used: **no**  ", "Test used: **no**", "",
        "Frozen hierarchical weight: `0.11`", "",
        "| Fold | Anchor F1 | Direct F1 | Hierarchical F1 | Blend F1 | Delta |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_fold"]:
        lines.append(
            f"| {row['fold']} | {row['anchor']['macro_f1']:.4f} | "
            f"{row['direct']['macro_f1']:.4f} | "
            f"{row['hierarchical']['macro_f1']:.4f} | "
            f"{row['frozen_blend']['macro_f1']:.4f} | "
            f"{row['macro_f1_delta']:+.4f} |"
        )
    paired = result["paired_fold_macro_f1_delta"]
    aggregate = result["aggregate"]
    lines.extend([
        "", "## Aggregate", "",
        f"- Fold delta: {paired['mean']:+.6f} +/- {paired['std']:.6f}",
        f"- Anchor Macro-F1: {aggregate['anchor']['macro_f1']:.6f}",
        f"- Frozen blend Macro-F1: {aggregate['frozen_blend']['macro_f1']:.6f}",
        "- Aggregate delta: "
        f"{aggregate['comparison_vs_anchor']['macro_f1_delta']:+.6f}",
        f"- Promotion gate: **{'pass' if result['promotion_gate']['passed'] else 'fail'}**",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "outputs/mocheg_b13_confirm"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--fold-spec", type=Path, default=Path(
        "data/processed/mocheg_b12_folds.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b13_confirmation.json"))
    parser.add_argument("--markdown", type=Path, default=Path(
        "outputs/mocheg_b13_confirmation.md"))
    args = parser.parse_args()
    fold_payload = json.loads(args.fold_spec.read_text(encoding="utf-8"))
    if (
        int(fold_payload.get("seed", -1)) != 2027
        or fold_payload.get("validation_split_used") is not False
        or fold_payload.get("test_split_used") is not False
        or fold_payload.get("manifest_sha256") != sha256(args.manifest)
    ):
        raise ValueError("expected locked B12/B13 seed-2027 train-only folds")
    fold_signature = sha256(args.fold_spec)
    manifest = {row["id"]: row for row in read_jsonl(args.manifest)}
    runs = []
    for fold in CONFIRMATION_FOLDS:
        anchor_path = args.root / f"fold_{fold}" / "anchor"
        candidate_path = args.root / f"fold_{fold}" / "curriculum"
        anchor_summary = validate_run(anchor_path, fold, f"fold_{fold}_anchor")
        candidate_summary = validate_run(
            candidate_path, fold, f"fold_{fold}_candidate"
        )
        validate_curriculum(candidate_summary, fold)
        for role, summary in (("anchor", anchor_summary),
                              ("candidate", candidate_summary)):
            observed = summary.get("provenance", {}).get("fold_spec_sha256")
            if observed != fold_signature:
                raise ValueError(f"fold {fold} {role}: signature mismatch")
        anchor_rows = read_predictions(anchor_path / "val_predictions.jsonl")
        candidate_rows = read_predictions(
            candidate_path / "val_predictions.jsonl"
        )
        expected = set(next(
            row["val_ids"] for row in fold_payload["folds"]
            if int(row["fold"]) == fold
        ))
        if set(anchor_rows) != expected or set(candidate_rows) != expected:
            raise ValueError(f"fold {fold}: held prediction IDs mismatch")
        ids = sorted(expected)
        labels = np.asarray([int(anchor_rows[value]["gold"]) for value in ids])
        observed_labels = np.asarray([
            int(candidate_rows[value]["gold"]) for value in ids
        ])
        if not np.array_equal(labels, observed_labels):
            raise ValueError(f"fold {fold}: candidate labels are misaligned")
        anchor = normalized_probabilities(anchor_rows, ids)
        direct = normalized_probabilities(candidate_rows, ids)
        sufficiency = normalized_probabilities(
            candidate_rows, ids, "sufficiency_probabilities", 2
        )
        polarity = normalized_probabilities(
            candidate_rows, ids, "polarity_probabilities", 2
        )
        runs.append({
            "fold": fold,
            "ids": ids,
            "labels": labels,
            "sources": [
                str(manifest[value].get("source", "unknown") or "unknown")
                for value in ids
            ],
            "anchor": anchor,
            "direct": direct,
            "hierarchical": hierarchical_probabilities(
                sufficiency, polarity
            ),
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
