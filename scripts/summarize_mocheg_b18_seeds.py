"""Aggregate and ensemble multi-seed B18 verifier evaluations on Fold 0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.run_mocheg_visual_retrieval import read_jsonl


def load_seed_predictions(run_dir: Path) -> dict[str, dict]:
    if run_dir.is_file():
        pred_path = run_dir
    else:
        pred_path = run_dir / "val_predictions.jsonl"
        if not pred_path.is_file():
            alt_path = run_dir / "test_predictions.jsonl"
            if alt_path.is_file():
                pred_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Predictions file missing in '{run_dir}'. Expected '{pred_path}'. "
                    f"If this is a control run, make sure it has finished training on Fold 0, "
                    f"or run without --control-roots to evaluate candidates only."
                )
    rows = read_jsonl(pred_path)
    return {str(r["id"]): r for r in rows}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class = f1_score(y_true, y_pred, average=None).tolist()
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_supported": per_class[0] if len(per_class) > 0 else 0.0,
        "f1_refuted": per_class[1] if len(per_class) > 1 else 0.0,
        "f1_nei": per_class[2] if len(per_class) > 2 else 0.0,
        "confusion_matrix": cm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed summary and ensemble for Phase B18")
    parser.add_argument("--candidate-roots", type=Path, nargs="+", required=True, help="Candidate directories")
    parser.add_argument("--control-roots", type=Path, nargs="*", default=None, help="Optional matched control directories")
    parser.add_argument("--manifest", type=Path, required=True, help="Claims manifest (train.jsonl)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--markdown", type=Path, required=True, help="Output markdown path")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    has_control = bool(args.control_roots and len(args.control_roots) > 0)
    if has_control and len(args.candidate_roots) != len(args.control_roots):
        raise ValueError("Number of candidate roots must match control roots")

    num_seeds = len(args.candidate_roots)
    cand_by_seed = [load_seed_predictions(p) for p in args.candidate_roots]
    ctrl_by_seed = [load_seed_predictions(p) for p in args.control_roots] if has_control else []

    # Find common IDs across all seeds
    all_dicts = cand_by_seed + ctrl_by_seed
    common_ids = set(cand_by_seed[0].keys())
    for c_dict in all_dicts:
        common_ids &= set(c_dict.keys())
    ordered_ids = sorted(common_ids)

    y_true = np.asarray([int(cand_by_seed[0][cid]["label"]) for cid in ordered_ids])

    # Per-seed analysis
    seed_reports = []
    deltas = []
    for s_idx in range(num_seeds):
        c_preds = np.asarray([int(cand_by_seed[s_idx][cid]["prediction"]) for cid in ordered_ids])
        m_cand = compute_metrics(y_true, c_preds)

        if has_control:
            k_preds = np.asarray([int(ctrl_by_seed[s_idx][cid]["prediction"]) for cid in ordered_ids])
            m_ctrl = compute_metrics(y_true, k_preds)
            delta_f1 = m_cand["macro_f1"] - m_ctrl["macro_f1"]
            deltas.append(delta_f1)

            boot = bootstrap_delta(y_true, k_preds, c_preds, iterations=args.iterations, seed=args.seed + s_idx)
            p_pos = float(boot.get("probability_delta_positive", 0.0))

            helpful = int(np.sum((k_preds != y_true) & (c_preds == y_true)))
            harmful = int(np.sum((k_preds == y_true) & (c_preds != y_true)))
            mcnemar_p = exact_mcnemar_p(helpful, harmful)

            seed_reports.append({
                "seed_index": s_idx,
                "candidate_dir": str(args.candidate_roots[s_idx]),
                "control_dir": str(args.control_roots[s_idx]),
                "cand_macro_f1": m_cand["macro_f1"],
                "ctrl_macro_f1": m_ctrl["macro_f1"],
                "delta_macro_f1": delta_f1,
                "cand_accuracy": m_cand["accuracy"],
                "ctrl_accuracy": m_ctrl["accuracy"],
                "delta_accuracy": m_cand["accuracy"] - m_ctrl["accuracy"],
                "cand_f1_supported": m_cand["f1_supported"],
                "ctrl_f1_supported": m_ctrl["f1_supported"],
                "cand_f1_nei": m_cand["f1_nei"],
                "ctrl_f1_nei": m_ctrl["f1_nei"],
                "helpful": helpful,
                "harmful": harmful,
                "mcnemar_p": mcnemar_p,
                "bootstrap_p_positive": p_pos,
            })
        else:
            seed_reports.append({
                "seed_index": s_idx,
                "candidate_dir": str(args.candidate_roots[s_idx]),
                "cand_macro_f1": m_cand["macro_f1"],
                "cand_accuracy": m_cand["accuracy"],
                "cand_f1_supported": m_cand["f1_supported"],
                "cand_f1_refuted": m_cand["f1_refuted"],
                "cand_f1_nei": m_cand["f1_nei"],
            })

    # Ensemble Analysis: Average probabilities across seeds
    cand_probs_sum = np.zeros((len(ordered_ids), 3), dtype=np.float64)
    ctrl_probs_sum = np.zeros((len(ordered_ids), 3), dtype=np.float64)

    for s_idx in range(num_seeds):
        for i, cid in enumerate(ordered_ids):
            cand_probs_sum[i] += np.asarray(cand_by_seed[s_idx][cid]["probabilities"])
            if has_control:
                ctrl_probs_sum[i] += np.asarray(ctrl_by_seed[s_idx][cid]["probabilities"])

    ens_cand_pred = cand_probs_sum.argmax(axis=-1)
    m_ens_cand = compute_metrics(y_true, ens_cand_pred)

    if has_control:
        ens_ctrl_pred = ctrl_probs_sum.argmax(axis=-1)
        m_ens_ctrl = compute_metrics(y_true, ens_ctrl_pred)
        ens_delta_f1 = m_ens_cand["macro_f1"] - m_ens_ctrl["macro_f1"]
        ens_delta_acc = m_ens_cand["accuracy"] - m_ens_ctrl["accuracy"]

        ens_boot = bootstrap_delta(y_true, ens_ctrl_pred, ens_cand_pred, iterations=args.iterations, seed=args.seed)
        ens_p_pos = float(ens_boot.get("probability_delta_positive", 0.0))
        ens_ci = ens_boot.get("ci_95_percentile", [0.0, 0.0])

        ens_helpful = int(np.sum((ens_ctrl_pred != y_true) & (ens_cand_pred == y_true)))
        ens_harmful = int(np.sum((ens_ctrl_pred == y_true) & (ens_cand_pred != y_true)))
        ens_mcnemar_p = exact_mcnemar_p(ens_helpful, ens_harmful)

        summary_payload = {
            "num_seeds": num_seeds,
            "sample_count": len(ordered_ids),
            "has_control": True,
            "mean_delta_macro_f1": float(np.mean(deltas)),
            "std_delta_macro_f1": float(np.std(deltas)) if num_seeds > 1 else 0.0,
            "positive_seeds": int(np.sum(np.asarray(deltas) > 0)),
            "per_seed_reports": seed_reports,
            "ensemble_results": {
                "cand_macro_f1": m_ens_cand["macro_f1"],
                "ctrl_macro_f1": m_ens_ctrl["macro_f1"],
                "delta_macro_f1": ens_delta_f1,
                "cand_accuracy": m_ens_cand["accuracy"],
                "ctrl_accuracy": m_ens_ctrl["accuracy"],
                "delta_accuracy": ens_delta_acc,
                "cand_f1_supported": m_ens_cand["f1_supported"],
                "ctrl_f1_supported": m_ens_ctrl["f1_supported"],
                "cand_f1_nei": m_ens_cand["f1_nei"],
                "ctrl_f1_nei": m_ens_ctrl["f1_nei"],
                "helpful": ens_helpful,
                "harmful": ens_harmful,
                "exact_mcnemar_p": ens_mcnemar_p,
                "bootstrap_p_positive": ens_p_pos,
                "bootstrap_ci_95": ens_ci,
            },
        }
    else:
        summary_payload = {
            "num_seeds": num_seeds,
            "sample_count": len(ordered_ids),
            "has_control": False,
            "per_seed_reports": seed_reports,
            "ensemble_results": {
                "cand_macro_f1": m_ens_cand["macro_f1"],
                "cand_accuracy": m_ens_cand["accuracy"],
                "cand_f1_supported": m_ens_cand["f1_supported"],
                "cand_f1_refuted": m_ens_cand["f1_refuted"],
                "cand_f1_nei": m_ens_cand["f1_nei"],
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    # Generate Markdown Report
    if has_control:
        lines = [
            f"# Phase B18-A: Multi-Seed Confirmation & Ensemble Summary ({num_seeds} Seeds)",
            "",
            f"- **Samples:** {len(ordered_ids)} (Fold 0 held-out validation)",
            f"- **Mean Delta Macro-F1 across seeds:** **{np.mean(deltas):+.5f} +/- {np.std(deltas):.5f}**",
            f"- **Positive Seeds:** **{int(np.sum(np.asarray(deltas) > 0))}/{num_seeds}**",
            "",
            "## 1. Per-Seed Breakdown",
            "",
            "| Seed Run | Control Macro-F1 | Candidate Macro-F1 | Delta Macro-F1 | Delta Acc | Bootstrap P(Delta > 0) | Helpful/Harmful |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for s in seed_reports:
            lines.append(
                f"| Run {s['seed_index'] + 1} | {s['ctrl_macro_f1']:.5f} | {s['cand_macro_f1']:.5f} | "
                f"**{s['delta_macro_f1']:+.5f}** | {s['delta_accuracy']:+.5f} | {s['bootstrap_p_positive']:.4f} | "
                f"{s['helpful']} / {s['harmful']} |"
            )

        lines.extend([
            "",
            f"## 2. Multi-Seed Ensemble Performance ({num_seeds}-Seed Average Probabilities)",
            "",
            "| Metric | Matched Control Ensemble | Candidate Ensemble | Delta (Cand - Ctrl) |",
            "|---|---:|---:|---:|",
            f"| **Macro-F1** | {m_ens_ctrl['macro_f1']:.5f} | **{m_ens_cand['macro_f1']:.5f}** | **{ens_delta_f1:+.5f}** |",
            f"| Accuracy | {m_ens_ctrl['accuracy']:.5f} | {m_ens_cand['accuracy']:.5f} | {ens_delta_acc:+.5f} |",
            f"| F1 Supported | {m_ens_ctrl['f1_supported']:.5f} | {m_ens_cand['f1_supported']:.5f} | {m_ens_cand['f1_supported'] - m_ens_ctrl['f1_supported']:+.5f} |",
            f"| F1 Refuted | {m_ens_ctrl['f1_refuted']:.5f} | {m_ens_cand['f1_refuted']:.5f} | {m_ens_cand['f1_refuted'] - m_ens_ctrl['f1_refuted']:+.5f} |",
            f"| F1 NEI | {m_ens_ctrl['f1_nei']:.5f} | {m_ens_cand['f1_nei']:.5f} | {m_ens_cand['f1_nei'] - m_ens_ctrl['f1_nei']:+.5f} |",
            "",
            "### Ensemble Statistical Significance:",
            f"- **Ensemble Bootstrap P(Delta > 0):** **{ens_p_pos:.4f}**",
            f"- **Ensemble 95% Bootstrap CI:** `[{ens_ci[0]:+.5f}, {ens_ci[1]:+.5f}]`",
            f"- **Ensemble Helpful vs Harmful:** {ens_helpful} vs {ens_harmful}",
            f"- **Ensemble McNemar p-value:** {ens_mcnemar_p:.5f}",
            "",
        ])
    else:
        lines = [
            f"# Phase B18-B: Multi-Seed Candidate Summary ({num_seeds} Seeds)",
            "",
            f"- **Samples:** {len(ordered_ids)} (Fold 0 held-out validation)",
            "",
            "## 1. Per-Seed Breakdown (Candidates on Filtered Evidence)",
            "",
            "| Seed Run | Candidate Macro-F1 | Accuracy | F1 Supported | F1 Refuted | F1 NEI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for s in seed_reports:
            lines.append(
                f"| Run {s['seed_index'] + 1} | **{s['cand_macro_f1']:.5f}** | {s['cand_accuracy']:.5f} | "
                f"{s['cand_f1_supported']:.5f} | {s['cand_f1_refuted']:.5f} | {s['cand_f1_nei']:.5f} |"
            )

        lines.extend([
            "",
            f"## 2. Multi-Seed Candidate Ensemble Performance ({num_seeds}-Seed Average Probabilities)",
            "",
            "| Metric | Candidate Ensemble |",
            "|---|---:|",
            f"| **Macro-F1** | **{m_ens_cand['macro_f1']:.5f}** |",
            f"| Accuracy | {m_ens_cand['accuracy']:.5f} |",
            f"| F1 Supported | {m_ens_cand['f1_supported']:.5f} |",
            f"| F1 Refuted | {m_ens_cand['f1_refuted']:.5f} |",
            f"| F1 NEI | {m_ens_cand['f1_nei']:.5f} |",
            "",
        ])

    md_content = "\n".join(lines) + "\n"
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(md_content, encoding="utf-8")
    print(md_content)


if __name__ == "__main__":
    main()
