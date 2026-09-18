"""Evaluate and compare Phase B18 explanation candidate against matched direct control.

Protocol evaluation on held-out fold:
1. Computes Macro-F1, Accuracy, and per-class F1 for candidate, control, and optional anchor.
2. Evaluates paired bootstrap delta distribution (10,000 resamples) and exact McNemar p-value.
3. Analyzes domain slices (Politifact, Snopes) and diagnostic slices.
4. Checks strictly against the preregistered promotion gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.run_mocheg_visual_retrieval import read_jsonl


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


def analyze_b18_run(
    candidate_preds: list[dict],
    control_preds: list[dict],
    anchor_preds: list[dict] | None,
    manifest_claims: list[dict],
    iterations: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    manifest_by_id = {str(r["id"]): r for r in manifest_claims}
    cand_by_id = {str(r["id"]): r for r in candidate_preds}
    ctrl_by_id = {str(r["id"]): r for r in control_preds}
    anc_by_id = {str(r["id"]): r for r in anchor_preds} if anchor_preds else None

    common_ids = sorted(set(cand_by_id.keys()) & set(ctrl_by_id.keys()))
    if not common_ids:
        raise ValueError("No common IDs between candidate and control predictions")

    y_true = np.asarray([int(cand_by_id[cid]["label"]) for cid in common_ids])
    y_cand = np.asarray([int(cand_by_id[cid]["prediction"]) for cid in common_ids])
    y_ctrl = np.asarray([int(ctrl_by_id[cid]["prediction"]) for cid in common_ids])

    cand_metrics = compute_metrics(y_true, y_cand)
    ctrl_metrics = compute_metrics(y_true, y_ctrl)

    # Paired comparisons
    ctrl_correct = y_ctrl == y_true
    cand_correct = y_cand == y_true
    helpful = int(np.sum(~ctrl_correct & cand_correct))
    harmful = int(np.sum(ctrl_correct & ~cand_correct))

    delta_macro_f1 = cand_metrics["macro_f1"] - ctrl_metrics["macro_f1"]
    delta_acc = cand_metrics["accuracy"] - ctrl_metrics["accuracy"]
    delta_nei = cand_metrics["f1_nei"] - ctrl_metrics["f1_nei"]
    delta_supp = cand_metrics["f1_supported"] - ctrl_metrics["f1_supported"]
    delta_ref = cand_metrics["f1_refuted"] - ctrl_metrics["f1_refuted"]

    boot = bootstrap_delta(y_true, y_ctrl, y_cand, iterations=iterations, seed=seed)
    mcnemar_p = exact_mcnemar_p(helpful, harmful)

    # Slice analysis
    sources = [
        manifest_by_id.get(cid, {}).get("source", "unknown").lower()
        if "source" in manifest_by_id.get(cid, {})
        else ("snopes" if "snopes" in manifest_by_id.get(cid, {}).get("snopes_url", "").lower() else "politifact")
        for cid in common_ids
    ]
    slice_analysis = {}
    for source_name in ["politifact", "snopes"]:
        idx = [i for i, s in enumerate(sources) if source_name in s]
        if idx:
            sub_true = y_true[idx]
            sub_cand = y_cand[idx]
            sub_ctrl = y_ctrl[idx]
            m_cand = compute_metrics(sub_true, sub_cand)
            m_ctrl = compute_metrics(sub_true, sub_ctrl)
            slice_analysis[source_name] = {
                "count": len(idx),
                "cand_macro_f1": m_cand["macro_f1"],
                "ctrl_macro_f1": m_ctrl["macro_f1"],
                "delta_macro_f1": m_cand["macro_f1"] - m_ctrl["macro_f1"],
            }

    # Anchor comparison (if present)
    anchor_comparison = None
    if anc_by_id:
        common_anc_ids = [cid for cid in common_ids if cid in anc_by_id]
        if common_anc_ids:
            y_anc = np.asarray([int(anc_by_id[cid]["prediction"]) for cid in common_anc_ids])
            anc_true = np.asarray([int(cand_by_id[cid]["label"]) for cid in common_anc_ids])
            anc_cand = np.asarray([int(cand_by_id[cid]["prediction"]) for cid in common_anc_ids])
            anc_metrics = compute_metrics(anc_true, y_anc)
            anc_cand_metrics = compute_metrics(anc_true, anc_cand)
            anchor_comparison = {
                "anchor_macro_f1": anc_metrics["macro_f1"],
                "cand_macro_f1": anc_cand_metrics["macro_f1"],
                "delta_vs_anchor": anc_cand_metrics["macro_f1"] - anc_metrics["macro_f1"],
            }

    # Preregistered gate check
    gate_checks = {
        "delta_macro_f1_ge_005": delta_macro_f1 >= 0.005,
        "bootstrap_positive_ge_095": float(boot.get("probability_positive", 0.0)) >= 0.95,
        "supported_f1_ge_minus_005": delta_supp >= -0.005,
        "accuracy_ge_minus_002": delta_acc >= -0.002,
        "helpful_exceeds_harmful": helpful > harmful,
    }
    all_passed = all(gate_checks.values())

    return {
        "samples": len(common_ids),
        "candidate_metrics": cand_metrics,
        "control_metrics": ctrl_metrics,
        "delta_metrics": {
            "macro_f1": delta_macro_f1,
            "accuracy": delta_acc,
            "f1_supported": delta_supp,
            "f1_refuted": delta_ref,
            "f1_nei": delta_nei,
        },
        "paired_analysis": {
            "helpful": helpful,
            "harmful": harmful,
            "exact_mcnemar_p": mcnemar_p,
            "bootstrap": boot,
        },
        "slice_analysis": slice_analysis,
        "anchor_comparison": anchor_comparison,
        "gate_checks": gate_checks,
        "passed_gate": all_passed,
    }


def generate_markdown_report(result: dict[str, Any]) -> str:
    cand = result["candidate_metrics"]
    ctrl = result["control_metrics"]
    delta = result["delta_metrics"]
    paired = result["paired_analysis"]
    boot = paired["bootstrap"]
    gate = result["gate_checks"]
    passed = result["passed_gate"]

    status_str = "**PASSED (Ready for 5-fold / 5-seed confirmation)**" if passed else "**FAILED (Does not pass promotion gate)**"

    lines = [
        "# Phase B18-A: Grounded Explanation Distillation Analysis",
        "",
        f"**Status:** {status_str}",
        f"**Sample Count (Held-out Fold):** {result['samples']}",
        "",
        "## 1. Overall Comparison: Candidate vs. Matched Control",
        "",
        "| Metric | Matched Control | Explanation Candidate | Delta (Cand - Ctrl) |",
        "|---|---:|---:|---:|",
        f"| **Macro-F1** | {ctrl['macro_f1']:.5f} | **{cand['macro_f1']:.5f}** | **{delta['macro_f1']:+.5f}** |",
        f"| Accuracy | {ctrl['accuracy']:.5f} | {cand['accuracy']:.5f} | {delta['accuracy']:+.5f} |",
        f"| F1 Supported | {ctrl['f1_supported']:.5f} | {cand['f1_supported']:.5f} | {delta['f1_supported']:+.5f} |",
        f"| F1 Refuted | {ctrl['f1_refuted']:.5f} | {cand['f1_refuted']:.5f} | {delta['f1_refuted']:+.5f} |",
        f"| F1 NEI | {ctrl['f1_nei']:.5f} | {cand['f1_nei']:.5f} | {delta['f1_nei']:+.5f} |",
        "",
        "## 2. Statistical Significance & Paired Tests",
        "",
        f"- **Helpful Corrections (Ctrl wrong -> Cand right):** {paired['helpful']}",
        f"- **Harmful Regressions (Ctrl right -> Cand wrong):** {paired['harmful']}",
        f"- **Exact McNemar p-value:** {paired['exact_mcnemar_p']:.5f}",
        f"- **Paired Bootstrap P(Delta > 0):** **{boot.get('probability_positive', 0.0):.4f}** (Required >= 0.95)",
        f"- **95% Bootstrap CI:** `[{boot.get('ci_lower', 0.0):.5f}, {boot.get('ci_upper', 0.0):.5f}]`",
        "",
        "## 3. Preregistered Promotion Gate Checks",
        "",
        "| Gate Criterion | Required Threshold | Observed | Result |",
        "|---|---|---|---|",
        f"| Delta Macro-F1 | >= +0.005 | {delta['macro_f1']:+.5f} | {'PASS' if gate['delta_macro_f1_ge_005'] else 'FAIL'} |",
        f"| Bootstrap P(Delta > 0) | >= 0.95 | {boot.get('probability_positive', 0.0):.4f} | {'PASS' if gate['bootstrap_positive_ge_095'] else 'FAIL'} |",
        f"| Supported F1 Preservation | >= -0.005 | {delta['f1_supported']:+.5f} | {'PASS' if gate['supported_f1_ge_minus_005'] else 'FAIL'} |",
        f"| Accuracy Preservation | >= -0.002 | {delta['accuracy']:+.5f} | {'PASS' if gate['accuracy_ge_minus_002'] else 'FAIL'} |",
        f"| Helpful > Harmful | Helpful > Harmful | {paired['helpful']} vs {paired['harmful']} | {'PASS' if gate['helpful_exceeds_harmful'] else 'FAIL'} |",
        "",
        "## 4. Slice Breakdown",
        "",
    ]
    for sname, sdata in result.get("slice_analysis", {}).items():
        lines.append(f"- **{sname.capitalize()}** (n={sdata['count']}): Control={sdata['ctrl_macro_f1']:.4f}, Candidate={sdata['cand_macro_f1']:.4f}, Delta={sdata['delta_macro_f1']:+.4f}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase B18 explanation verifier against matched control")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate val_predictions.jsonl")
    parser.add_argument("--control", type=Path, required=True, help="Matched control val_predictions.jsonl")
    parser.add_argument("--anchor", type=Path, default=None, help="Optional frozen anchor val_predictions.jsonl")
    parser.add_argument("--manifest", type=Path, required=True, help="Strict claims manifest (train.jsonl)")
    parser.add_argument("--output", type=Path, required=True, help="Output analysis.json")
    parser.add_argument("--markdown", type=Path, required=True, help="Output analysis.md")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cand_preds = read_jsonl(args.candidate)
    ctrl_preds = read_jsonl(args.control)
    anc_preds = read_jsonl(args.anchor) if args.anchor and args.anchor.is_file() else None
    claims = read_jsonl(args.manifest)

    report_dict = analyze_b18_run(
        cand_preds, ctrl_preds, anc_preds, claims, iterations=args.iterations, seed=args.seed
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report_dict, indent=2) + "\n", encoding="utf-8")

    md_text = generate_markdown_report(report_dict)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(md_text, encoding="utf-8")

    print(md_text)


if __name__ == "__main__":
    main()
