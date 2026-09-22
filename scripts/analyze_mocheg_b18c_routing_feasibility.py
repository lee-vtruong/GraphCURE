"""Assess whether B18-B experts have enough complementarity to justify routing.

This is a gold-aware diagnostic on train-only Fold 0. Oracle results are upper
bounds only and may never be used as an inference policy or promotion result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.summarize_mocheg_b18_seeds import compute_metrics, load_seed_predictions


def oracle_predictions(
    labels: np.ndarray,
    fallback: np.ndarray,
    experts: list[np.ndarray],
) -> np.ndarray:
    """Return a diagnostic oracle that is correct if any expert is correct."""
    result = fallback.copy()
    for index, label in enumerate(labels):
        if any(expert[index] == label for expert in experts):
            result[index] = label
    return result


def outcome_overlap(
    labels: np.ndarray, first: np.ndarray, second: np.ndarray
) -> dict[str, int]:
    first_correct = first == labels
    second_correct = second == labels
    return {
        "both_correct": int(np.sum(first_correct & second_correct)),
        "first_only_correct": int(np.sum(first_correct & ~second_correct)),
        "second_only_correct": int(np.sum(~first_correct & second_correct)),
        "both_wrong": int(np.sum(~first_correct & ~second_correct)),
        "prediction_disagreements": int(np.sum(first != second)),
    }


def group_delta(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: list[str],
) -> dict[str, Any]:
    result = {}
    for group in sorted(set(groups)):
        indices = np.asarray([i for i, value in enumerate(groups) if value == group])
        if len(indices) < 2:
            continue
        ref_metrics = compute_metrics(labels[indices], reference[indices])
        cand_metrics = compute_metrics(labels[indices], candidate[indices])
        result[group] = {
            "samples": len(indices),
            "top3_macro_f1": ref_metrics["macro_f1"],
            "distilled_macro_f1": cand_metrics["macro_f1"],
            "macro_f1_delta": cand_metrics["macro_f1"] - ref_metrics["macro_f1"],
            "top3_accuracy": ref_metrics["accuracy"],
            "distilled_accuracy": cand_metrics["accuracy"],
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--retrieval-top3", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--distilled", type=Path, required=True)
    parser.add_argument("--distilled-retrieval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--minimum-oracle-delta", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = {
        "anchor": args.anchor,
        "retrieval_top3": args.retrieval_top3,
        "generic": args.generic,
        "distilled": args.distilled,
    }
    runs = {name: load_seed_predictions(root) for name, root in roots.items()}
    ids = sorted(set.intersection(*(set(run) for run in runs.values())))
    labels = np.asarray([int(runs["anchor"][claim_id]["label"]) for claim_id in ids])
    predictions = {
        name: np.asarray([int(run[claim_id]["prediction"]) for claim_id in ids])
        for name, run in runs.items()
    }
    probabilities = {
        name: np.asarray([run[claim_id]["probabilities"] for claim_id in ids], dtype=float)
        for name, run in runs.items()
    }
    for name, run in runs.items():
        other = np.asarray([int(run[claim_id]["label"]) for claim_id in ids])
        if not np.array_equal(labels, other):
            raise ValueError(f"label mismatch in {name}")

    top3_distilled_oracle = oracle_predictions(
        labels,
        predictions["retrieval_top3"],
        [predictions["retrieval_top3"], predictions["distilled"]],
    )
    three_expert_oracle = oracle_predictions(
        labels,
        predictions["anchor"],
        [predictions["anchor"], predictions["retrieval_top3"], predictions["distilled"]],
    )
    all_expert_oracle = oracle_predictions(
        labels,
        predictions["anchor"],
        list(predictions.values()),
    )
    metrics = {name: compute_metrics(labels, pred) for name, pred in predictions.items()}
    metrics.update({
        "oracle_top3_distilled": compute_metrics(labels, top3_distilled_oracle),
        "oracle_anchor_top3_distilled": compute_metrics(labels, three_expert_oracle),
        "oracle_all": compute_metrics(labels, all_expert_oracle),
    })

    manifest = {str(row["id"]): row for row in read_jsonl(args.manifest)}
    selected = {str(row["id"]): row for row in read_jsonl(args.distilled_retrieval)}
    source_groups = [str(manifest.get(claim_id, {}).get("source", "unknown")) for claim_id in ids]
    selected_k_groups = [
        f"k={int(selected.get(claim_id, {}).get('selected_k', 0))}" for claim_id in ids
    ]
    disagreement_groups = [
        "disagree" if predictions["retrieval_top3"][i] != predictions["distilled"][i]
        else "agree"
        for i in range(len(ids))
    ]
    advantage = (
        probabilities["distilled"].max(axis=1)
        - probabilities["retrieval_top3"].max(axis=1)
    )
    advantage_groups = [
        "distilled_advantage" if value >= 0.05
        else "top3_advantage" if value <= -0.05
        else "similar_confidence"
        for value in advantage
    ]
    groups = {
        "source": group_delta(
            labels, predictions["retrieval_top3"], predictions["distilled"], source_groups
        ),
        "selected_k": group_delta(
            labels, predictions["retrieval_top3"], predictions["distilled"], selected_k_groups
        ),
        "expert_agreement": group_delta(
            labels, predictions["retrieval_top3"], predictions["distilled"], disagreement_groups
        ),
        "confidence_advantage": group_delta(
            labels, predictions["retrieval_top3"], predictions["distilled"], advantage_groups
        ),
    }
    overlap = {
        "top3_vs_distilled": outcome_overlap(
            labels, predictions["retrieval_top3"], predictions["distilled"]
        ),
        "anchor_vs_distilled": outcome_overlap(
            labels, predictions["anchor"], predictions["distilled"]
        ),
    }
    top3_f1 = metrics["retrieval_top3"]["macro_f1"]
    oracle_delta = metrics["oracle_top3_distilled"]["macro_f1"] - top3_f1
    complementary_rate = overlap["top3_vs_distilled"]["second_only_correct"] / len(ids)
    feasibility = {
        "oracle_delta_vs_top3": oracle_delta,
        "oracle_delta_at_least_minimum": oracle_delta >= args.minimum_oracle_delta,
        "distilled_only_correct_rate": complementary_rate,
        "distilled_only_correct_rate_at_least_0_02": complementary_rate >= 0.02,
        "observable_group_with_positive_delta": any(
            row["macro_f1_delta"] > 0.0
            for table in groups.values()
            for row in table.values()
        ),
    }
    feasibility["router_development_warranted"] = all([
        feasibility["oracle_delta_at_least_minimum"],
        feasibility["distilled_only_correct_rate_at_least_0_02"],
        feasibility["observable_group_with_positive_delta"],
    ])
    payload = {
        "phase": "B18-C",
        "protocol": "train_only_fold0_gold_aware_routing_feasibility_diagnostic",
        "samples": len(ids),
        "metrics": metrics,
        "outcome_overlap": overlap,
        "observable_group_atlas": groups,
        "feasibility": feasibility,
        "audit": {
            "gold_labels_used_for_oracle_diagnostic": True,
            "gold_labels_used_for_inference": False,
            "diagnostic_only": True,
            "official_validation_used": False,
            "test_split_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# MOCHEG B18-C routing feasibility diagnostic",
        "",
        "Gold-aware oracle: **diagnostic only**  ",
        "Official validation used: **no**  ",
        "Test used: **no**",
        "",
        f"- Top-3 Macro-F1: {top3_f1:.6f}",
        f"- Top-3 + distilled oracle Macro-F1: {metrics['oracle_top3_distilled']['macro_f1']:.6f}",
        f"- Oracle delta: {oracle_delta:+.6f}",
        f"- Distilled-only correct: {overlap['top3_vs_distilled']['second_only_correct']}",
        f"- Router development warranted: **{'yes' if feasibility['router_development_warranted'] else 'no'}**",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
