"""Summarize all B19 Fold-0 ablations and apply a preregistered promotion gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.summarize_mocheg_b18_seeds import compute_metrics, load_seed_predictions


VARIANTS = [
    "matched_control", "ensemble_kd", "disagreement_kd",
    "counterfactual_only", "full",
]


def comparison(labels: np.ndarray, reference: np.ndarray, candidate: np.ndarray,
               iterations: int, seed: int) -> dict:
    helpful = int(np.sum((reference != labels) & (candidate == labels)))
    harmful = int(np.sum((reference == labels) & (candidate != labels)))
    ref_metrics = compute_metrics(labels, reference)
    candidate_metrics = compute_metrics(labels, candidate)
    return {
        "macro_f1_delta": candidate_metrics["macro_f1"] - ref_metrics["macro_f1"],
        "accuracy_delta": candidate_metrics["accuracy"] - ref_metrics["accuracy"],
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(labels, reference, candidate, iterations, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--minimum-control-delta", type=float, default=0.005)
    parser.add_argument("--minimum-kd-delta", type=float, default=0.003)
    parser.add_argument("--minimum-bootstrap-probability", type=float, default=0.90)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    runs = {}
    predictions = {}
    for name in VARIANTS:
        path = args.root / name
        summary_path = path / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("complete"):
            raise RuntimeError(f"incomplete B19 run: {name}")
        runs[name] = summary
        predictions[name] = load_seed_predictions(path)
    ids = sorted(set.intersection(*(set(rows) for rows in predictions.values())))
    labels = np.asarray([int(predictions["matched_control"][claim_id]["label"]) for claim_id in ids])
    pred = {
        name: np.asarray([int(rows[claim_id]["prediction"]) for claim_id in ids])
        for name, rows in predictions.items()
    }
    metrics = {name: compute_metrics(labels, values) for name, values in pred.items()}
    vs_control = comparison(
        labels, pred["matched_control"], pred["full"],
        args.bootstrap_iterations, args.seed,
    )
    vs_kd = comparison(
        labels, pred["ensemble_kd"], pred["full"],
        args.bootstrap_iterations, args.seed + 1,
    )
    class_delta = {
        key: metrics["full"][key] - metrics["matched_control"][key]
        for key in ("f1_supported", "f1_refuted", "f1_nei")
    }
    gate = {
        "full_delta_vs_control_at_least_minimum": (
            vs_control["macro_f1_delta"] >= args.minimum_control_delta
        ),
        "full_delta_vs_kd_at_least_minimum": (
            vs_kd["macro_f1_delta"] >= args.minimum_kd_delta
        ),
        "nei_f1_nonnegative": class_delta["f1_nei"] >= 0.0,
        "bootstrap_vs_control_at_least_minimum": (
            vs_control["bootstrap"]["probability_delta_positive"]
            >= args.minimum_bootstrap_probability
        ),
        "help_exceeds_harm_vs_control": vs_control["helpful"] > vs_control["harmful"],
    }
    gate["passed"] = all(gate.values())
    payload = {
        "phase": "B19-A",
        "protocol": "fold0_matched_ablation_screen",
        "samples": len(ids),
        "metrics": metrics,
        "full_vs_matched_control": vs_control,
        "full_vs_ensemble_kd": vs_kd,
        "class_f1_delta_vs_control": class_delta,
        "training_audit": {
            name: {
                "optimizer_updates": runs[name]["optimizer_updates"],
                "loss_weights": runs[name]["loss_weights"],
                "counterfactual_train_samples": runs[name]["counterfactual_train_samples"],
                "teacher_target_sources": runs[name]["teacher_target_sources"],
            } for name in VARIANTS
        },
        "promotion_gate": gate,
        "confirmation_required_on_folds": [1, 2, 3, 4] if gate["passed"] else [],
        "official_validation_used": False,
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MOCHEG B19-A disagreement-aware distillation screen",
        "", "Official validation used: **no**  ", "Test used: **no**", "",
        "| Variant | Accuracy | Macro-F1 | Supp F1 | Ref F1 | NEI F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        row = metrics[name]
        lines.append(
            f"| {name} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['f1_supported']:.4f} | {row['f1_refuted']:.4f} | {row['f1_nei']:.4f} |"
        )
    lines.extend([
        "", f"- Full vs control: {vs_control['macro_f1_delta']:+.6f}",
        f"- Full vs KD-only: {vs_kd['macro_f1_delta']:+.6f}",
        f"- Bootstrap P(delta > 0): {vs_control['bootstrap']['probability_delta_positive']:.4f}",
        f"- Promotion gate: **{'pass' if gate['passed'] else 'fail'}**",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
