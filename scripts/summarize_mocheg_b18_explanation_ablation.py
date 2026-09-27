"""Summarize five-fold matched-control vs explanation-supervision ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b18b_component_ablations import row_label
from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.summarize_mocheg_b18_seeds import (
    compute_metrics,
    load_seed_predictions,
)


def aligned_fold(control_root: Path, candidate_root: Path):
    control = load_seed_predictions(control_root)
    candidate = load_seed_predictions(candidate_root)
    control_ids = set(control)
    candidate_ids = set(candidate)
    if control_ids != candidate_ids:
        raise ValueError(
            f"prediction IDs differ: {control_root} vs {candidate_root}; "
            f"control-only={len(control_ids - candidate_ids)}, "
            f"candidate-only={len(candidate_ids - control_ids)}"
        )
    ids = sorted(control_ids)
    labels = np.asarray([row_label(control[sample_id]) for sample_id in ids])
    candidate_labels = np.asarray([
        row_label(candidate[sample_id]) for sample_id in ids
    ])
    if not np.array_equal(labels, candidate_labels):
        raise ValueError("gold labels disagree between matched runs")
    control_prediction = np.asarray([
        int(control[sample_id].get(
            "prediction",
            np.argmax(control[sample_id]["probabilities"]),
        )) for sample_id in ids
    ])
    candidate_prediction = np.asarray([
        int(candidate[sample_id].get(
            "prediction",
            np.argmax(candidate[sample_id]["probabilities"]),
        )) for sample_id in ids
    ])
    return ids, labels, control_prediction, candidate_prediction


def comparison(labels: np.ndarray, control: np.ndarray,
               candidate: np.ndarray, iterations: int, seed: int) -> dict:
    control_metrics = compute_metrics(labels, control)
    candidate_metrics = compute_metrics(labels, candidate)
    helpful = int(np.sum((control != labels) & (candidate == labels)))
    harmful = int(np.sum((control == labels) & (candidate != labels)))
    return {
        "macro_f1_delta": float(
            candidate_metrics["macro_f1"] - control_metrics["macro_f1"]
        ),
        "accuracy_delta": float(
            candidate_metrics["accuracy"] - control_metrics["accuracy"]
        ),
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(
            labels, control, candidate, iterations, seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/mocheg_b18"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    seen_ids: set[str] = set()
    per_fold = []
    labels_parts = []
    control_parts = []
    candidate_parts = []
    for fold in args.folds:
        ids, labels, control, candidate = aligned_fold(
            args.root / f"control_fold{fold}",
            args.root / f"candidate_fold{fold}",
        )
        overlap = seen_ids & set(ids)
        if overlap:
            raise ValueError(
                f"held folds overlap on {len(overlap)} prediction IDs"
            )
        seen_ids.update(ids)
        control_metrics = compute_metrics(labels, control)
        candidate_metrics = compute_metrics(labels, candidate)
        effect = comparison(
            labels, control, candidate,
            max(1000, args.iterations // len(args.folds)), args.seed + fold,
        )
        per_fold.append({
            "fold": fold,
            "samples": len(ids),
            "matched_control": control_metrics,
            "explanation_supervision": candidate_metrics,
            "comparison": effect,
        })
        labels_parts.append(labels)
        control_parts.append(control)
        candidate_parts.append(candidate)

    labels = np.concatenate(labels_parts)
    control = np.concatenate(control_parts)
    candidate = np.concatenate(candidate_parts)
    control_metrics = compute_metrics(labels, control)
    candidate_metrics = compute_metrics(labels, candidate)
    aggregate_effect = comparison(
        labels, control, candidate, args.iterations, args.seed
    )
    fold_deltas = np.asarray([
        row["comparison"]["macro_f1_delta"] for row in per_fold
    ])
    class_delta = {
        key: float(candidate_metrics[key] - control_metrics[key])
        for key in ("f1_supported", "f1_refuted", "f1_nei")
    }
    payload = {
        "protocol": "B18_five_fold_oof_explanation_supervision_ablation",
        "samples": len(labels),
        "folds": args.folds,
        "settings": {
            "control_lambda_exp": 0.0,
            "candidate_lambda_exp": 0.25,
            "top_k": 5,
        },
        "per_fold": per_fold,
        "aggregate": {
            "matched_control": control_metrics,
            "explanation_supervision": candidate_metrics,
            "comparison": aggregate_effect,
            "class_f1_delta": class_delta,
            "fold_delta": {
                "mean": float(fold_deltas.mean()),
                "std": float(fold_deltas.std()),
                "values": fold_deltas.tolist(),
                "positive_folds": int(np.sum(fold_deltas > 0)),
            },
        },
        "official_validation_used": False,
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# B18 explanation-supervision five-fold ablation", "",
        "Official validation used: **no**  ", "Test used: **no**", "",
        "| Fold | Control F1 (lambda=0) | Explanation F1 (lambda=.25) | Delta | Helpful/Harmful |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in per_fold:
        effect = row["comparison"]
        lines.append(
            f"| {row['fold']} | {row['matched_control']['macro_f1']:.6f} | "
            f"{row['explanation_supervision']['macro_f1']:.6f} | "
            f"{effect['macro_f1_delta']:+.6f} | "
            f"{effect['helpful']}/{effect['harmful']} |"
        )
    boot = aggregate_effect["bootstrap"]
    lines += [
        "", "## Aggregate", "",
        f"- Control Macro-F1: `{control_metrics['macro_f1']:.6f}`",
        f"- Explanation Macro-F1: `{candidate_metrics['macro_f1']:.6f}`",
        f"- Macro-F1 delta: `{aggregate_effect['macro_f1_delta']:+.6f}`",
        f"- Accuracy delta: `{aggregate_effect['accuracy_delta']:+.6f}`",
        f"- Mean fold delta: `{fold_deltas.mean():+.6f}`",
        f"- Positive folds: `{int(np.sum(fold_deltas > 0))}/{len(fold_deltas)}`",
        f"- Helpful/harmful: `{aggregate_effect['helpful']}/{aggregate_effect['harmful']}`",
        f"- Exact McNemar p: `{aggregate_effect['exact_mcnemar_p']:.6f}`",
        f"- Bootstrap 95% CI: `[{boot['ci_95_percentile'][0]:+.6f}, "
        f"{boot['ci_95_percentile'][1]:+.6f}]`",
        f"- Bootstrap P(delta > 0): `{boot['probability_delta_positive']:.4f}`",
        "", "## Class-F1 delta", "",
        f"- Supported: `{class_delta['f1_supported']:+.6f}`",
        f"- Refuted: `{class_delta['f1_refuted']:+.6f}`",
        f"- NEI: `{class_delta['f1_nei']:+.6f}`",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
