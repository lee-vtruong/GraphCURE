"""Paired validation comparison of frozen B18B Top-K evidence policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b18b_component_ablations import load_average
from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.summarize_mocheg_b18_seeds import compute_metrics


def routed_probabilities(direct: np.ndarray, grounded: np.ndarray,
                         tau: float) -> np.ndarray:
    grounded_prediction = grounded.argmax(axis=1)
    route = (grounded_prediction == 2) & (grounded[:, 2] >= tau)
    result = direct.copy()
    result[route] = np.asarray([0.0, 0.0, 1.0])
    return result


def comparison(labels: np.ndarray, reference: np.ndarray,
               candidate: np.ndarray, iterations: int, seed: int) -> dict:
    reference_prediction = reference.argmax(axis=1)
    candidate_prediction = candidate.argmax(axis=1)
    reference_correct = reference_prediction == labels
    candidate_correct = candidate_prediction == labels
    helpful = int(np.sum(~reference_correct & candidate_correct))
    harmful = int(np.sum(reference_correct & ~candidate_correct))
    reference_metrics = compute_metrics(labels, reference_prediction)
    candidate_metrics = compute_metrics(labels, candidate_prediction)
    return {
        "macro_f1_delta": float(
            candidate_metrics["macro_f1"] - reference_metrics["macro_f1"]
        ),
        "accuracy_delta": float(
            candidate_metrics["accuracy"] - reference_metrics["accuracy"]
        ),
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(
            labels, reference_prediction, candidate_prediction,
            iterations, seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--k1-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--k3-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--k5-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--tau", type=float, default=0.49)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    ids, labels, direct = load_average(args.direct_runs)
    grounded = {}
    for k, paths in ((1, args.k1_runs), (3, args.k3_runs), (5, args.k5_runs)):
        observed_ids, observed_labels, values = load_average(paths)
        if observed_ids != ids or not np.array_equal(observed_labels, labels):
            raise ValueError(f"K={k} predictions are not aligned with direct runs")
        grounded[k] = values

    routed = {
        k: routed_probabilities(direct, values, args.tau)
        for k, values in grounded.items()
    }
    metrics = {
        "direct": compute_metrics(labels, direct.argmax(axis=1)),
        **{
            f"k{k}": compute_metrics(labels, values.argmax(axis=1))
            for k, values in routed.items()
        },
    }
    comparisons = {
        "k3_vs_k5": comparison(
            labels, routed[5], routed[3], args.iterations, args.seed
        ),
        "k3_vs_k1": comparison(
            labels, routed[1], routed[3], args.iterations, args.seed + 1
        ),
        "k3_vs_direct": comparison(
            labels, direct, routed[3], args.iterations, args.seed + 2
        ),
    }
    payload = {
        "protocol": "B18B_frozen_validation_evidence_k_comparison",
        "samples": len(labels),
        "tau": args.tau,
        "grounded_seeds_per_k": {
            "k1": len(args.k1_runs), "k3": len(args.k3_runs),
            "k5": len(args.k5_runs),
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "selection": {
            "metric": "validation_macro_f1",
            "selected_k": 3,
            "official_validation_used_for_top_k_selection": True,
            "test_used_for_top_k_selection": False,
            "statistical_comparisons_are_diagnostic": True,
        },
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# B18B frozen-validation Top-K comparison", "",
        "Official validation used for Top-K selection: **yes**  ",
        "Test used for Top-K selection: **no**", "",
        "| Policy | Accuracy | Macro-F1 | Supported F1 | Refuted F1 | NEI F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("direct", "k1", "k3", "k5"):
        row = metrics[name]
        lines.append(
            f"| {name} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['f1_supported']:.6f} | {row['f1_refuted']:.6f} | "
            f"{row['f1_nei']:.6f} |"
        )
    lines += ["", "## Paired comparisons", ""]
    for name, row in comparisons.items():
        boot = row["bootstrap"]
        lines += [
            f"### {name}", "",
            f"- Macro-F1 delta: `{row['macro_f1_delta']:+.6f}`",
            f"- Accuracy delta: `{row['accuracy_delta']:+.6f}`",
            f"- Helpful/harmful: `{row['helpful']}/{row['harmful']}`",
            f"- Exact McNemar p: `{row['exact_mcnemar_p']:.6f}`",
            f"- Bootstrap 95% CI: `[{boot['ci_95_percentile'][0]:+.6f}, "
            f"{boot['ci_95_percentile'][1]:+.6f}]`",
            f"- Bootstrap P(delta > 0): `{boot['probability_delta_positive']:.4f}`",
            "",
        ]
    lines += [
        "Selected by frozen validation Macro-F1: **K=3**.",
        "Paired confidence statistics are diagnostic because the same validation "
        "split selected K.",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
