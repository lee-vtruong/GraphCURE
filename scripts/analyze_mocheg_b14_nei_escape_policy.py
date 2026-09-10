"""Diagnose the fixed asymmetric policy suggested by the B14 OOF atlas.

The policy preserves every determinate anchor verdict. It uses the B13 direct
curriculum expert only when the anchor predicts NEI and the expert predicts a
determinate label. Because the rule was derived from these OOF outcomes, this
script can only justify preregistration on a new fold assignment.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b12_failure_atlas import (
    LABEL_NAMES,
    classification_metrics,
)
from scripts.analyze_mocheg_b6_auxiliary_control import paired_comparison
from scripts.analyze_mocheg_b7_frozen_router import source_diagnostics
from scripts.run_mocheg_visual_retrieval import read_jsonl


LABEL_IDS = {value: key for key, value in LABEL_NAMES.items()}


def nei_escape_predictions(
    anchor: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fixed rule and return predictions plus its route mask."""
    if anchor.shape != candidate.shape:
        raise ValueError("anchor and candidate predictions are misaligned")
    route = (anchor == 2) & (candidate != 2)
    output = anchor.copy()
    output[route] = candidate[route]
    return output, route


def one_hot(prediction: np.ndarray) -> np.ndarray:
    return np.eye(3, dtype=np.float64)[prediction]


def grouped_diagnostics(
    cases: list[dict], labels: np.ndarray, anchor: np.ndarray,
    routed: np.ndarray, route: np.ndarray,
) -> dict:
    fields = ("source", "gold", "retrieval_status", "qrel_available")
    result = {}
    anchor_probabilities = one_hot(anchor)
    routed_probabilities = one_hot(routed)
    for field in fields:
        groups = {}
        for value in sorted({str(row.get(field)) for row in cases}):
            selected = np.asarray([
                str(row.get(field)) == value for row in cases
            ])
            before = classification_metrics(
                labels[selected], anchor_probabilities[selected]
            )
            after = classification_metrics(
                labels[selected], routed_probabilities[selected]
            )
            groups[value] = {
                "samples": int(selected.sum()),
                "route_count": int(np.sum(route[selected])),
                "route_rate": float(np.mean(route[selected])),
                "anchor_macro_f1": before["macro_f1"],
                "routed_macro_f1": after["macro_f1"],
                "macro_f1_delta": after["macro_f1"] - before["macro_f1"],
            }
        result[field] = groups
    return result


def evaluate(
    cases: list[dict], bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> dict:
    if not cases:
        raise ValueError("B14 NEI-escape diagnostic received no cases")
    ids = [row["id"] for row in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case IDs")
    labels = np.asarray([LABEL_IDS[row["gold"]] for row in cases])
    anchor = np.asarray([
        LABEL_IDS[row["predictions"]["anchor"]] for row in cases
    ])
    candidate = np.asarray([
        LABEL_IDS[row["predictions"]["candidate"]] for row in cases
    ])
    routed, route = nei_escape_predictions(anchor, candidate)
    anchor_probabilities = one_hot(anchor)
    candidate_probabilities = one_hot(candidate)
    routed_probabilities = one_hot(routed)
    paired = paired_comparison(
        labels, anchor_probabilities, routed_probabilities,
        bootstrap_iterations, bootstrap_seed,
    )
    source = np.asarray([str(row.get("source", "unknown")) for row in cases])
    source_results = source_diagnostics(
        labels, anchor_probabilities, routed_probabilities, source
    )
    route_gold = Counter(
        LABEL_NAMES[int(value)] for value in labels[route].tolist()
    )
    route_output = Counter(
        LABEL_NAMES[int(value)] for value in routed[route].tolist()
    )
    gate = {
        "macro_f1_delta_at_least_0_005": paired["macro_f1_delta"] >= .005,
        "accuracy_delta_positive": paired["accuracy_delta"] > 0,
        "help_exceeds_harm": paired["helpful"] > paired["harmful"],
        "bootstrap_probability_at_least_0_95": (
            paired["bootstrap"]["probability_delta_positive"] >= .95
        ),
        "both_sources_nonnegative": all(
            row["macro_f1_delta"] >= 0 for row in source_results.values()
        ),
        "route_rate_between_0_05_and_0_40": .05 <= float(route.mean()) <= .40,
    }
    gate["eligible"] = all(gate.values())
    return {
        "protocol": "B14_post_failure_OOF_asymmetric_NEI_escape_diagnostic",
        "policy": {
            "default": "anchor",
            "route_condition": (
                "anchor_prediction == nei and candidate_prediction != nei"
            ),
            "thresholds": None,
            "probability_weights": None,
        },
        "samples": len(cases),
        "metrics": {
            "anchor": classification_metrics(labels, anchor_probabilities),
            "candidate": classification_metrics(
                labels, candidate_probabilities
            ),
            "nei_escape": classification_metrics(
                labels, routed_probabilities
            ),
        },
        "comparison_vs_anchor": paired,
        "routing": {
            "route_count": int(route.sum()),
            "route_rate": float(route.mean()),
            "route_gold_counts": dict(route_gold),
            "route_output_counts": dict(route_output),
        },
        "source_diagnostics": source_results,
        "group_diagnostics": grouped_diagnostics(
            cases, labels, anchor, routed, route
        ),
        "preregistration_gate": gate,
        "interpretation": {
            "confirmatory_result": False,
            "reason": (
                "The policy was derived after inspecting these OOF outcomes. "
                "A positive diagnostic only permits preregistration on a new "
                "fold assignment."
            ),
            "fresh_fold_assignment_required": True,
        },
        "official_validation_used": False,
        "test_split_used": False,
    }


def markdown(result: dict) -> str:
    metrics = result["metrics"]
    paired = result["comparison_vs_anchor"]
    lines = [
        "# B14 asymmetric NEI-escape OOF diagnostic", "",
        "Official validation used: **no**  ", "Test used: **no**", "",
        "| System | Accuracy | Macro-F1 |", "|---|---:|---:|",
    ]
    for name in ("anchor", "candidate", "nei_escape"):
        row = metrics[name]
        lines.append(
            f"| {name} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |"
        )
    lines.extend([
        "", f"- Macro-F1 delta: {paired['macro_f1_delta']:+.6f}",
        f"- Accuracy delta: {paired['accuracy_delta']:+.6f}",
        f"- Helpful / harmful: {paired['helpful']} / {paired['harmful']}",
        f"- Route rate: {result['routing']['route_rate']:.4f}",
        "- Eligible for fresh-fold preregistration: "
        f"**{result['preregistration_gate']['eligible']}**", "",
        "This is an exploratory post-failure diagnostic, not a confirmed result.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path(
        "outputs/mocheg_b14_direct_curriculum_cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b14_nei_escape_diagnostic.json"))
    parser.add_argument("--markdown", type=Path, default=Path(
        "outputs/mocheg_b14_nei_escape_diagnostic.md"))
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    result = evaluate(
        read_jsonl(args.cases), args.bootstrap_iterations, args.bootstrap_seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
