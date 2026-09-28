"""Run label-safe reviewer controls for the frozen B18B routing policy.

This analysis performs no training and never uses test labels for policy
selection.  It (1) tests AND against its two closest frozen validation
ablations, (2) selects a direct-only self-deferral threshold on validation and
applies it unchanged to raw and strict P1 test predictions, and (3) measures a
diagnostic evidence-coverage proxy only on decisive claims.  The proxy is not
an inference feature or a routing target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scripts.analyze_mocheg_b18b_component_ablations import load_average
from scripts.analyze_mocheg_b18b_subgroups import rank_group
from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.summarize_mocheg_b18_seeds import compute_metrics


def self_deferral(probabilities: np.ndarray, tau: float) -> np.ndarray:
    """Direct-only policy: a sufficiently large direct NEI posterior defers."""
    prediction = probabilities.argmax(axis=1)
    prediction[(prediction != 2) & (probabilities[:, 2] >= tau)] = 2
    return prediction


def and_policy(direct: np.ndarray, grounded: np.ndarray, tau: float) -> np.ndarray:
    prediction = direct.argmax(axis=1)
    grounded_prediction = grounded.argmax(axis=1)
    prediction[(grounded_prediction == 2) & (grounded[:, 2] >= tau)] = 2
    return prediction


def unconstrained_policy(direct: np.ndarray, grounded: np.ndarray, tau: float) -> np.ndarray:
    prediction = direct.argmax(axis=1)
    grounded_prediction = grounded.argmax(axis=1)
    prediction[grounded.max(axis=1) >= tau] = grounded_prediction[grounded.max(axis=1) >= tau]
    return prediction


def paired(labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
           iterations: int, seed: int) -> dict:
    base = compute_metrics(labels, baseline)
    cand = compute_metrics(labels, candidate)
    helpful = int(np.sum((baseline != labels) & (candidate == labels)))
    harmful = int(np.sum((baseline == labels) & (candidate != labels)))
    return {
        "baseline": base,
        "candidate": cand,
        "macro_f1_delta": float(cand["macro_f1"] - base["macro_f1"]),
        "accuracy_delta": float(cand["accuracy"] - base["accuracy"]),
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(labels, baseline, candidate, iterations, seed),
    }


def select_direct_tau(labels: np.ndarray, direct: np.ndarray, grid: np.ndarray) -> dict:
    rows = []
    for tau in grid:
        prediction = self_deferral(direct, float(tau))
        metric = compute_metrics(labels, prediction)
        rows.append({"tau": float(tau), **metric, "route_count": int(np.sum(prediction != direct.argmax(axis=1)))})
    # Deterministic tie break: maximize MF1, then accuracy, then prefer the
    # more conservative (larger) threshold.
    return max(rows, key=lambda row: (row["macro_f1"], row["accuracy"], row["tau"]))


def coverage_diagnostic(ids: list[str], labels: np.ndarray, direct: np.ndarray,
                        grounded: np.ndarray, manifest_path: Path,
                        retrieval_path: Path, top_k: int) -> dict:
    manifests = {str(row["id"]): row for row in read_jsonl(manifest_path)}
    retrieval = {str(row["id"]): row for row in read_jsonl(retrieval_path)}
    coverage, direct_scores, grounded_scores = [], [], []
    for index, sample_id in enumerate(ids):
        if int(labels[index]) == 2:  # Gold evidence does not define a decisive-proof target for NEI.
            continue
        gold = {str(value) for value in manifests[sample_id].get("text_evidence_ids", [])}
        if not gold:
            continue
        retrieved = [str(value) for value in retrieval[sample_id].get("retrieved_evidence_ids", [])]
        sufficient = int(rank_group(gold, retrieved[:top_k]) in {"gold_rank_1", "gold_rank_2_5"})
        coverage.append(sufficient)
        direct_scores.append(float(1.0 - direct[index, 2]))
        grounded_scores.append(float(1.0 - grounded[index, 2]))
    target = np.asarray(coverage, dtype=int)
    if len(np.unique(target)) != 2:
        raise ValueError("coverage target has only one class")
    def score(values: list[float]) -> dict:
        values_array = np.asarray(values, dtype=float)
        return {
            "auroc": float(roc_auc_score(target, values_array)),
            "average_precision": float(average_precision_score(target, values_array)),
            "brier": float(brier_score_loss(target, values_array)),
        }
    return {
        "definition": "decisive gold label with at least one gold text-evidence ID in retrieved top-K",
        "samples": int(len(target)),
        "sufficient": int(target.sum()),
        "insufficient": int((1 - target).sum()),
        "direct_one_minus_p_nei": score(direct_scores),
        "grounded_one_minus_p_nei": score(grounded_scores),
        "diagnostic_only": True,
        "gold_evidence_used_for_policy_or_inference": False,
    }


def evaluate_test_track(name: str, paths: list[Path], tau: float,
                        iterations: int, seed: int) -> dict:
    ids, labels, direct = load_average(paths)
    direct_prediction = direct.argmax(axis=1)
    self_prediction = self_deferral(direct, tau)
    return {
        "track": name,
        "samples": len(ids),
        "frozen_tau_from_validation": tau,
        "self_deferral_vs_direct": paired(labels, direct_prediction, self_prediction, iterations, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-val-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--grounded-val-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--val-retrieval", type=Path, required=True)
    parser.add_argument("--raw-direct-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--strict-direct-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--and-tau", type=float, default=0.49)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.grid_step <= 1:
        parser.error("--grid-step must be in (0, 1]")

    ids, labels, direct = load_average(args.direct_val_runs)
    grounded_ids, grounded_labels, grounded = load_average(args.grounded_val_runs)
    if ids != grounded_ids or not np.array_equal(labels, grounded_labels):
        raise ValueError("direct and grounded validation predictions are not exactly aligned")
    direct_prediction = direct.argmax(axis=1)
    grounded_prediction = grounded.argmax(axis=1)
    and_prediction = and_policy(direct, grounded, args.and_tau)
    a4_prediction = unconstrained_policy(direct, grounded, args.and_tau)
    grid = np.arange(0.0, 1.0 + args.grid_step / 2, args.grid_step)
    selected = select_direct_tau(labels, direct, grid)
    direct_self_val = self_deferral(direct, selected["tau"])

    result = {
        "protocol": "B18B_reviewer_controls_validation_selected_direct_self_deferral",
        "and_tau_frozen": args.and_tau,
        "validation": {
            "samples": len(ids),
            "and_vs_grounded_only": paired(labels, grounded_prediction, and_prediction, args.iterations, args.seed),
            "and_vs_unconstrained_confident_switch": paired(labels, a4_prediction, and_prediction, args.iterations, args.seed + 1),
            "direct_self_deferral": {
                "selection": selected,
                "vs_direct": paired(labels, direct_prediction, direct_self_val, args.iterations, args.seed + 2),
                "selection_uses_validation_labels": True,
                "test_labels_used_for_selection": False,
            },
            "coverage_diagnostic": coverage_diagnostic(ids, labels, direct, grounded, args.val_manifest, args.val_retrieval, args.top_k),
        },
        "test": {
            "raw_official": evaluate_test_track("raw_official_P1_n2442", args.raw_direct_runs, selected["tau"], args.iterations, args.seed + 3),
            "strict": evaluate_test_track("strict_P1_n2434", args.strict_direct_runs, selected["tau"], args.iterations, args.seed + 4),
            "test_labels_used_for_selection": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    v = result["validation"]
    def line(name: str, item: dict) -> str:
        boot = item["bootstrap"]
        return (f"| {name} | {item['baseline']['macro_f1']:.6f} | {item['candidate']['macro_f1']:.6f} | "
                f"{item['macro_f1_delta']:+.6f} | {item['helpful']}/{item['harmful']} | "
                f"{item['exact_mcnemar_p']:.4f} | {boot['probability_delta_positive']:.4f} |")
    lines = [
        "# MOCHEG B18B reviewer controls", "",
        "- Test labels used for policy selection: **no**", 
        "- Direct self-deferral threshold is selected only on official validation; its test results are confirmatory.",
        "- Coverage labels are retrieval diagnostics only and are never available to the router.", "",
        "## Frozen-validation mechanism comparisons", "",
        "| Comparison | Baseline MF1 | Candidate MF1 | Delta | Helpful/Harmful | McNemar p | Bootstrap P(delta > 0) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        line("AND vs grounded-only", v["and_vs_grounded_only"]),
        line("AND vs unconstrained switch (A4)", v["and_vs_unconstrained_confident_switch"]),
        line("Direct self-deferral vs direct", v["direct_self_deferral"]["vs_direct"]), "",
        f"Selected direct self-deferral tau on validation: `{selected['tau']:.2f}`.", "",
        "## Decisive-evidence coverage diagnostic", "",
        "| Score (higher = evidence coverage) | AUROC | AP | Brier |",
        "|---|---:|---:|---:|",
        f"| Direct $1-p_D(NEI)$ | {v['coverage_diagnostic']['direct_one_minus_p_nei']['auroc']:.4f} | {v['coverage_diagnostic']['direct_one_minus_p_nei']['average_precision']:.4f} | {v['coverage_diagnostic']['direct_one_minus_p_nei']['brier']:.4f} |",
        f"| Grounded $1-p_G(NEI)$ | {v['coverage_diagnostic']['grounded_one_minus_p_nei']['auroc']:.4f} | {v['coverage_diagnostic']['grounded_one_minus_p_nei']['average_precision']:.4f} | {v['coverage_diagnostic']['grounded_one_minus_p_nei']['brier']:.4f} |", "",
        "This diagnostic excludes gold NEI claims because a retrieved supporting/refuting passage is not a well-defined sufficiency target for them.", "",
        "## Frozen direct self-deferral on test", "",
    ]
    for name, item in result["test"].items():
        if name == "test_labels_used_for_selection":
            continue
        lines += [f"### {name}", line("Self-deferral vs direct", item["self_deferral_vs_direct"]), ""]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
