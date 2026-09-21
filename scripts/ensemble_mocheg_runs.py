"""General Multi-Run & Heterogeneous Ensembling Evaluation.

Averages predicted class probabilities across arbitrary combinations of runs
(e.g., Control + B18-A + B18-B) and benchmarks against a reference baseline ensemble.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_cached_verifier import expected_calibration_error


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_run_predictions(run_path: Path, filename: str = "val_predictions.jsonl") -> dict[str, dict]:
    if run_path.is_file():
        pred_path = run_path
    else:
        pred_path = run_path / filename
        if not pred_path.is_file():
            alt_name = "test_predictions.jsonl" if filename == "val_predictions.jsonl" else "val_predictions.jsonl"
            if (run_path / alt_name).is_file():
                pred_path = run_path / alt_name
            else:
                raise FileNotFoundError(f"Predictions file missing: {pred_path}")
    rows = read_jsonl(pred_path)
    return {str(r["id"]): r for r in rows}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class = f1_score(y_true, y_pred, average=None).tolist()
    cm = confusion_matrix(y_true, y_pred).tolist()
    ece = float(expected_calibration_error(y_prob, y_true)) if y_prob is not None else None
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_supported": per_class[0] if len(per_class) > 0 else 0.0,
        "f1_refuted": per_class[1] if len(per_class) > 1 else 0.0,
        "f1_nei": per_class[2] if len(per_class) > 2 else 0.0,
        "ece": ece,
        "confusion_matrix": cm,
    }


def ensemble_predictions(
    runs_dict: list[dict[str, dict]], ordered_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    prob_sum = np.zeros((len(ordered_ids), 3), dtype=np.float64)
    for r_dict in runs_dict:
        for i, cid in enumerate(ordered_ids):
            prob_sum[i] += np.asarray(r_dict[cid]["probabilities"], dtype=np.float64)
    avg_probs = prob_sum / float(len(runs_dict))
    preds = avg_probs.argmax(axis=-1)
    return avg_probs, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Heterogeneous Ensembling for GraphCURE Verifier Runs")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="List of run directories to ensemble")
    parser.add_argument("--baseline-runs", type=Path, nargs="*", default=None, help="Optional baseline run directories to compare against")
    parser.add_argument("--pred-file", type=str, default="val_predictions.jsonl", help="Prediction filename (e.g. test_predictions.jsonl)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--markdown", type=Path, default=None, help="Output Markdown path")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_names = [p.name for p in args.runs]
    logging.info("Loading %d runs: %s (filename=%s)", len(args.runs), run_names, args.pred_file)
    loaded_runs = [load_run_predictions(p, args.pred_file) for p in args.runs]

    baseline_runs = [load_run_predictions(p, args.pred_file) for p in args.baseline_runs] if args.baseline_runs else []

    # Find common IDs
    all_dicts = loaded_runs + baseline_runs
    common_ids = set(all_dicts[0].keys())
    for d in all_dicts[1:]:
        common_ids &= set(d.keys())
    ordered_ids = sorted(common_ids)
    logging.info("Found %d common evaluation samples", len(ordered_ids))

    first_sample = all_dicts[0][ordered_ids[0]]
    label_key = "gold" if "gold" in first_sample else "label"
    y_true = np.asarray([int(all_dicts[0][cid][label_key]) for cid in ordered_ids])

    # Per-run individual metrics
    individual_reports = []
    for p, r_dict in zip(args.runs, loaded_runs):
        probs = np.asarray([r_dict[cid]["probabilities"] for cid in ordered_ids])
        if "prediction" in r_dict[ordered_ids[0]]:
            preds = np.asarray([int(r_dict[cid]["prediction"]) for cid in ordered_ids])
        else:
            preds = probs.argmax(axis=-1)
        m = compute_metrics(y_true, preds, probs)
        individual_reports.append({"name": p.name, "path": str(p), **m})

    # Ensemble of target runs
    ens_probs, ens_preds = ensemble_predictions(loaded_runs, ordered_ids)
    ens_metrics = compute_metrics(y_true, ens_preds, ens_probs)

    comparison_report = None
    if baseline_runs:
        b_probs, b_preds = ensemble_predictions(baseline_runs, ordered_ids)
        b_metrics = compute_metrics(y_true, b_preds, b_probs)

        delta_f1 = ens_metrics["macro_f1"] - b_metrics["macro_f1"]
        delta_acc = ens_metrics["accuracy"] - b_metrics["accuracy"]

        boot = bootstrap_delta(y_true, b_preds, ens_preds, iterations=args.iterations, seed=args.seed)
        p_pos = float(boot.get("probability_delta_positive", 0.0))
        ci = boot.get("ci_95_percentile", [0.0, 0.0])

        helpful = int(np.sum((b_preds != y_true) & (ens_preds == y_true)))
        harmful = int(np.sum((b_preds == y_true) & (ens_preds != y_true)))
        mcnemar_p = exact_mcnemar_p(helpful, harmful)

        comparison_report = {
            "baseline_metrics": b_metrics,
            "delta_macro_f1": delta_f1,
            "delta_accuracy": delta_acc,
            "bootstrap_p_positive": p_pos,
            "bootstrap_ci_95": ci,
            "helpful": helpful,
            "harmful": harmful,
            "exact_mcnemar_p": mcnemar_p,
        }

    output_payload = {
        "num_runs": len(args.runs),
        "run_paths": [str(p) for p in args.runs],
        "sample_count": len(ordered_ids),
        "individual_runs": individual_reports,
        "ensemble_metrics": ens_metrics,
        "comparison_against_baseline": comparison_report,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")

    # Generate Markdown Report
    lines = [
        f"# Heterogeneous Ensemble Report ({len(args.runs)} Models)",
        "",
        f"- **Samples Evaluated:** {len(ordered_ids)}",
        f"- **Ensemble Macro-F1:** **{ens_metrics['macro_f1']:.5f}**",
        f"- **Ensemble Accuracy:** {ens_metrics['accuracy']:.5f}",
        "",
        "## 1. Individual Member Models",
        "",
        "| Model Run | Macro-F1 | Accuracy | F1 Supp | F1 Ref | F1 NEI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in individual_reports:
        lines.append(
            f"| `{r['name']}` | {r['macro_f1']:.5f} | {r['accuracy']:.5f} | {r['f1_supported']:.5f} | {r['f1_refuted']:.5f} | {r['f1_nei']:.5f} |"
        )

    lines.extend([
        "",
        "## 2. Ensemble vs. Baseline Benchmark",
        "",
    ])

    if comparison_report:
        b_m = comparison_report["baseline_metrics"]
        lines.extend([
            "| Metric | Reference Baseline | Heterogeneous Ensemble | Delta |",
            "|---|---:|---:|---:|",
            f"| **Macro-F1** | {b_m['macro_f1']:.5f} | **{ens_metrics['macro_f1']:.5f}** | **{comparison_report['delta_macro_f1']:+.5f}** |",
            f"| Accuracy | {b_m['accuracy']:.5f} | {ens_metrics['accuracy']:.5f} | {comparison_report['delta_accuracy']:+.5f} |",
            f"| F1 Supported | {b_m['f1_supported']:.5f} | {ens_metrics['f1_supported']:.5f} | {ens_metrics['f1_supported'] - b_m['f1_supported']:+.5f} |",
            f"| F1 Refuted | {b_m['f1_refuted']:.5f} | {ens_metrics['f1_refuted']:.5f} | {ens_metrics['f1_refuted'] - b_m['f1_refuted']:+.5f} |",
            f"| F1 NEI | {b_m['f1_nei']:.5f} | {ens_metrics['f1_nei']:.5f} | {ens_metrics['f1_nei'] - b_m['f1_nei']:+.5f} |",
            "",
            "### Statistical Significance & Error Analysis:",
            f"- **Bootstrap P(Delta > 0):** **{comparison_report['bootstrap_p_positive']:.4f}**",
            f"- **95% Bootstrap CI:** `[{comparison_report['bootstrap_ci_95'][0]:.5f}, {comparison_report['bootstrap_ci_95'][1]:.5f}]`",
            f"- **Helpful Corrections vs Harmful Regressions:** {comparison_report['helpful']} vs {comparison_report['harmful']}",
            f"- **Exact McNemar p-value:** {comparison_report['exact_mcnemar_p']:.5f}",
        ])
    else:
        lines.extend([
            f"| Metric | Ensemble Value |",
            f"|---|---:|",
            f"| **Macro-F1** | **{ens_metrics['macro_f1']:.5f}** |",
            f"| Accuracy | {ens_metrics['accuracy']:.5f} |",
            f"| F1 Supported | {ens_metrics['f1_supported']:.5f} |",
            f"| F1 Refuted | {ens_metrics['f1_refuted']:.5f} |",
            f"| F1 NEI | {ens_metrics['f1_nei']:.5f} |",
        ])

    md_text = "\n".join(lines) + "\n"
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md_text, encoding="utf-8")

    print("\n" + md_text)


if __name__ == "__main__":
    main()
