"""Aggregate frozen B19 matched-control vs disagreement-KD OOF predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b19_screen import comparison
from scripts.summarize_mocheg_b18_seeds import compute_metrics, load_seed_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    per_fold = []
    all_labels, all_control, all_candidate = [], [], []
    seen_ids: set[str] = set()
    for fold in args.folds:
        fold_root = args.root / f"fold_{fold}"
        control = load_seed_predictions(fold_root / "matched_control")
        candidate = load_seed_predictions(fold_root / "disagreement_kd")
        ids = sorted(set(control) & set(candidate))
        if not ids:
            raise RuntimeError(f"no paired predictions for fold {fold}")
        overlap = seen_ids.intersection(ids)
        if overlap:
            raise RuntimeError(f"claim leakage across folds; examples={sorted(overlap)[:5]}")
        seen_ids.update(ids)
        labels = np.asarray([int(control[i]["label"]) for i in ids])
        control_pred = np.asarray([int(control[i]["prediction"]) for i in ids])
        candidate_pred = np.asarray([int(candidate[i]["prediction"]) for i in ids])
        control_metrics = compute_metrics(labels, control_pred)
        candidate_metrics = compute_metrics(labels, candidate_pred)
        effect = comparison(labels, control_pred, candidate_pred, 2000, args.seed + fold)
        per_fold.append({
            "fold": fold,
            "samples": len(ids),
            "matched_control": control_metrics,
            "disagreement_kd": candidate_metrics,
            "comparison": effect,
        })
        all_labels.extend(labels.tolist())
        all_control.extend(control_pred.tolist())
        all_candidate.extend(candidate_pred.tolist())

    labels = np.asarray(all_labels)
    control_pred = np.asarray(all_control)
    candidate_pred = np.asarray(all_candidate)
    aggregate = {
        "matched_control": compute_metrics(labels, control_pred),
        "disagreement_kd": compute_metrics(labels, candidate_pred),
        "comparison": comparison(
            labels, control_pred, candidate_pred,
            args.bootstrap_iterations, args.seed,
        ),
    }
    deltas = [row["comparison"]["macro_f1_delta"] for row in per_fold]
    class_delta = {
        key: aggregate["disagreement_kd"][key] - aggregate["matched_control"][key]
        for key in ("f1_supported", "f1_refuted", "f1_nei")
    }
    gate = {
        "mean_fold_delta_at_least_0_005": float(np.mean(deltas)) >= 0.005,
        "aggregate_delta_at_least_0_005": aggregate["comparison"]["macro_f1_delta"] >= 0.005,
        "positive_folds_at_least_4": sum(x > 0 for x in deltas) >= 4,
        "bootstrap_probability_at_least_0_95": (
            aggregate["comparison"]["bootstrap"]["probability_delta_positive"] >= 0.95
        ),
        "nei_f1_nonnegative": class_delta["f1_nei"] >= 0,
        "help_exceeds_harm": (
            aggregate["comparison"]["helpful"] > aggregate["comparison"]["harmful"]
        ),
    }
    gate["passed"] = all(gate.values())
    payload = {
        "phase": "B19-B",
        "protocol": "five_fold_oof_fixed_disagreement_kd_confirmation",
        "folds": args.folds,
        "samples": len(labels),
        "per_fold": per_fold,
        "fold_delta": {
            "mean": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
            "values": deltas,
            "positive_folds": sum(x > 0 for x in deltas),
        },
        "aggregate": aggregate,
        "class_f1_delta": class_delta,
        "promotion_gate": gate,
        "official_validation_used": False,
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MOCHEG B19-B frozen five-fold OOF confirmation", "",
        "Official validation used: **no**  ", "Test used: **no**", "",
        "| Fold | Control F1 | DKD F1 | Delta | Helpful/Harmful |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in per_fold:
        effect = row["comparison"]
        lines.append(
            f"| {row['fold']} | {row['matched_control']['macro_f1']:.6f} | "
            f"{row['disagreement_kd']['macro_f1']:.6f} | "
            f"{effect['macro_f1_delta']:+.6f} | {effect['helpful']}/{effect['harmful']} |"
        )
    lines += [
        "", "## Aggregate", "",
        f"- Control Macro-F1: {aggregate['matched_control']['macro_f1']:.6f}",
        f"- DKD Macro-F1: {aggregate['disagreement_kd']['macro_f1']:.6f}",
        f"- Delta: {aggregate['comparison']['macro_f1_delta']:+.6f}",
        f"- Mean fold delta: {np.mean(deltas):+.6f}",
        f"- Positive folds: {sum(x > 0 for x in deltas)}/{len(deltas)}",
        f"- Bootstrap P(delta > 0): {aggregate['comparison']['bootstrap']['probability_delta_positive']:.4f}",
        f"- Promotion gate: **{'pass' if gate['passed'] else 'fail'}**",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
