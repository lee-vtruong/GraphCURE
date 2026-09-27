"""Evaluate a preregistered B18B router on a locked prediction split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b18b_component_ablations import load_average
from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.summarize_mocheg_b18_seeds import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--grounded-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--alignment", choices=("strict", "intersection"), default="strict",
        help="Require identical IDs, or explicitly evaluate their audited intersection",
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    direct_ids, direct_labels, direct = load_average(args.direct_runs)
    grounded_ids, grounded_labels, grounded = load_average(args.grounded_runs)
    alignment_audit = {
        "mode": args.alignment,
        "direct_samples_before_alignment": len(direct_ids),
        "grounded_samples_before_alignment": len(grounded_ids),
    }
    if direct_ids != grounded_ids:
        if args.alignment == "strict":
            raise ValueError(
                "direct and grounded prediction IDs are not aligned; use "
                "--alignment intersection only when reporting the resulting "
                "common-ID protocol explicitly"
            )
        common_ids = sorted(set(direct_ids) & set(grounded_ids))
        if not common_ids:
            raise ValueError("direct and grounded predictions have no common IDs")
        direct_index = {sample_id: index for index, sample_id in enumerate(direct_ids)}
        grounded_index = {
            sample_id: index for index, sample_id in enumerate(grounded_ids)
        }
        direct_positions = [direct_index[sample_id] for sample_id in common_ids]
        grounded_positions = [grounded_index[sample_id] for sample_id in common_ids]
        labels = direct_labels[direct_positions]
        observed_grounded_labels = grounded_labels[grounded_positions]
        direct = direct[direct_positions]
        grounded = grounded[grounded_positions]
        ids = common_ids
        if not np.array_equal(labels, observed_grounded_labels):
            raise ValueError("gold labels disagree on common prediction IDs")
    else:
        ids = direct_ids
        labels = direct_labels
        if not np.array_equal(labels, grounded_labels):
            raise ValueError("gold labels disagree across aligned predictions")
    alignment_audit.update({
        "evaluated_samples": len(ids),
        "direct_only_ids_dropped": len(set(direct_ids) - set(ids)),
        "grounded_only_ids_dropped": len(set(grounded_ids) - set(ids)),
    })

    direct_prediction = direct.argmax(axis=1)
    grounded_prediction = grounded.argmax(axis=1)
    route = (grounded_prediction == 2) & (grounded[:, 2] >= args.tau)
    routed_prediction = direct_prediction.copy()
    routed_prediction[route] = 2

    direct_correct = direct_prediction == labels
    routed_correct = routed_prediction == labels
    helpful = int(np.sum(~direct_correct & routed_correct))
    harmful = int(np.sum(direct_correct & ~routed_correct))
    direct_metrics = compute_metrics(labels, direct_prediction)
    grounded_metrics = compute_metrics(labels, grounded_prediction)
    routed_metrics = compute_metrics(labels, routed_prediction)
    comparison = {
        "macro_f1_delta": float(
            routed_metrics["macro_f1"] - direct_metrics["macro_f1"]
        ),
        "accuracy_delta": float(
            routed_metrics["accuracy"] - direct_metrics["accuracy"]
        ),
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(
            labels, direct_prediction, routed_prediction,
            args.iterations, args.seed,
        ),
    }
    payload = {
        "protocol": "B18B_frozen_router_locked_test",
        "samples": len(labels),
        "parameters": {"top_k": args.top_k, "tau": args.tau},
        "metrics": {
            "direct": direct_metrics,
            "grounded": grounded_metrics,
            "b18b": routed_metrics,
        },
        "comparison_vs_direct": comparison,
        "routing": {
            "route_count": int(route.sum()),
            "route_rate": float(route.mean()),
        },
        "alignment_audit": alignment_audit,
        "protocol_flags": {
            "top_k_selected_on_validation": True,
            "tau_selected_on_validation": True,
            "test_labels_used_for_parameter_selection": False,
            "test_labels_used_for_evaluation": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    boot = comparison["bootstrap"]
    lines = [
        "# B18B frozen-router locked-test evaluation", "",
        f"- Frozen Top-K: `{args.top_k}`",
        f"- Frozen tau: `{args.tau}`",
        f"- Alignment mode: `{args.alignment}`",
        f"- Evaluated samples: `{len(ids)}`",
        f"- Direct-only/Grounded-only IDs dropped: "
        f"`{alignment_audit['direct_only_ids_dropped']}/"
        f"{alignment_audit['grounded_only_ids_dropped']}`",
        "- Test labels used for parameter selection: **no**", "",
        "| Policy | Accuracy | Macro-F1 | Supported F1 | Refuted F1 | NEI F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("direct", "grounded", "b18b"):
        row = payload["metrics"][name]
        lines.append(
            f"| {name} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['f1_supported']:.6f} | {row['f1_refuted']:.6f} | "
            f"{row['f1_nei']:.6f} |"
        )
    lines += [
        "", "## B18B vs direct", "",
        f"- Macro-F1 delta: `{comparison['macro_f1_delta']:+.6f}`",
        f"- Accuracy delta: `{comparison['accuracy_delta']:+.6f}`",
        f"- Helpful/harmful: `{helpful}/{harmful}`",
        f"- Exact McNemar p: `{comparison['exact_mcnemar_p']:.6f}`",
        f"- Bootstrap 95% CI: `[{boot['ci_95_percentile'][0]:+.6f}, "
        f"{boot['ci_95_percentile'][1]:+.6f}]`",
        f"- Bootstrap P(delta > 0): `{boot['probability_delta_positive']:.4f}`",
        f"- Route count/rate: `{int(route.sum())}/{float(route.mean()):.6f}`",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
