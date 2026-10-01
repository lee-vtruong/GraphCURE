"""Frozen-validation diagnostics for CURE complementary experts.

This reporting-only analysis consumes saved validation predictions. It computes
expert agreement/error overlap, NEI-posterior reliability curves, a gold-NEI
by retrieval-rank cross-tab, a descriptive paired-design power approximation,
and a compute--quality Pareto plot. It does not train a model or select a
router threshold. Gold labels and evidence IDs are never fed to AND.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import NormalDist

import numpy as np
from sklearn.metrics import cohen_kappa_score

from scripts.analyze_mocheg_b18b_component_ablations import load_average
from scripts.analyze_mocheg_b18b_subgroups import rank_group
from scripts.run_mocheg_visual_retrieval import read_jsonl


LABELS = ("Supported", "Refuted", "NEI")


def probability_reliability(probabilities: np.ndarray, labels: np.ndarray, bins: int) -> dict:
    """Reliability for the event that the gold label is NEI."""
    scores = probabilities[:, 2]
    target = (labels == 2).astype(float)
    rows = []
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        mask = (scores >= lo) & ((scores < hi) if index < bins - 1 else (scores <= hi))
        if not mask.any():
            rows.append({"lower": lo, "upper": hi, "count": 0, "mean_probability": None, "empirical_nei_rate": None})
            continue
        rows.append({
            "lower": lo, "upper": hi, "count": int(mask.sum()),
            "mean_probability": float(scores[mask].mean()),
            "empirical_nei_rate": float(target[mask].mean()),
        })
    ece = sum(
        row["count"] / len(scores) * abs(row["mean_probability"] - row["empirical_nei_rate"])
        for row in rows if row["count"]
    )
    brier = float(np.mean((scores - target) ** 2))
    return {"event": "gold label is NEI", "bins": bins, "ece": ece, "brier": brier, "rows": rows}


def power_approximation(n: int, helpful: int, harmful: int, alpha: float, target_power: float) -> dict:
    """Normal approximation for paired binary correctness / McNemar design.

    This is intentionally not a power claim for macro-F1; macro-F1 has no
    simple closed-form paired power calculation. It quantifies the resolution
    of the observed correctness discordance pattern.
    """
    discordant_rate = (helpful + harmful) / n
    observed_accuracy_effect = (helpful - harmful) / n
    normal = NormalDist()
    z = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(target_power)
    mde = z * math.sqrt(discordant_rate / n)
    required_n = None
    if observed_accuracy_effect:
        required_n = math.ceil((z * z * discordant_rate) / (observed_accuracy_effect * observed_accuracy_effect))
    return {
        "n": n, "helpful": helpful, "harmful": harmful,
        "discordant_rate": discordant_rate,
        "observed_accuracy_effect": observed_accuracy_effect,
        "alpha_two_sided": alpha, "target_power": target_power,
        "mde_accuracy_approx": mde,
        "required_n_at_observed_effect_approx": required_n,
        "interpretation": "normal approximation for paired binary correctness; not a Macro-F1 power calculation",
    }


def write_plots(output_dir: Path, direct_rel: dict, grounded_rel: dict, pareto: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.35), sharex=True, sharey=True)
    for axis, title, result, color in zip(
        axes, ("Direct expert", "Rationale-trained expert"), (direct_rel, grounded_rel), ("#2C7FB8", "#D95F0E"), strict=True
    ):
        points = [row for row in result["rows"] if row["count"]]
        axis.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1, label="perfect calibration")
        axis.scatter([row["mean_probability"] for row in points], [row["empirical_nei_rate"] for row in points],
                     s=[max(18, row["count"] / 5) for row in points], color=color, alpha=.85)
        axis.set_title(title)
        axis.set_xlabel("Mean predicted $p(NEI)$")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Empirical NEI frequency")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_dir / "nei_reliability.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "nei_reliability.png", dpi=250, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.2, 3.5))
    for row in pareto:
        axis.scatter(row["milliseconds_per_claim"], row["macro_f1"], s=45, color=row["color"])
        axis.annotate(row["name"], (row["milliseconds_per_claim"], row["macro_f1"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Sequential inference latency (ms/claim)")
    axis.set_ylabel("Raw-P1 Macro-F1")
    axis.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(output_dir / "compute_quality_pareto.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "compute_quality_pareto.png", dpi=250, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--grounded-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=.49)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-n", type=int, default=2442)
    parser.add_argument("--raw-helpful", type=int, default=86)
    parser.add_argument("--raw-harmful", type=int, default=78)
    parser.add_argument("--strict-n", type=int, default=2434)
    parser.add_argument("--strict-helpful", type=int, default=84)
    parser.add_argument("--strict-harmful", type=int, default=77)
    args = parser.parse_args()

    ids, labels, direct = load_average(args.direct_runs)
    grounded_ids, grounded_labels, grounded = load_average(args.grounded_runs)
    if ids != grounded_ids or not np.array_equal(labels, grounded_labels):
        raise ValueError("direct and rationale-trained prediction files are not aligned")
    manifest = {str(row["id"]): row for row in read_jsonl(args.manifest)}
    retrieval = {str(row["id"]): row for row in read_jsonl(args.retrieval)}
    if set(ids) - set(manifest) or set(ids) - set(retrieval):
        raise ValueError("manifest or retrieval metadata is missing prediction IDs")

    direct_pred, grounded_pred = direct.argmax(axis=1), grounded.argmax(axis=1)
    and_pred = direct_pred.copy()
    route = (grounded_pred == 2) & (grounded[:, 2] >= args.tau)
    and_pred[route] = 2
    agreement = np.zeros((3, 3), dtype=int)
    for d, g in zip(direct_pred, grounded_pred, strict=True):
        agreement[d, g] += 1
    d_correct, g_correct = direct_pred == labels, grounded_pred == labels
    overlap = {
        "both_correct": int((d_correct & g_correct).sum()),
        "direct_only_correct": int((d_correct & ~g_correct).sum()),
        "grounded_only_correct": int((~d_correct & g_correct).sum()),
        "both_wrong": int((~d_correct & ~g_correct).sum()),
    }
    a, b, c, d = overlap["both_wrong"], overlap["grounded_only_correct"], overlap["direct_only_correct"], overlap["both_correct"]
    denominator = a * d + b * c
    complementarity = {
        "samples": len(ids), "prediction_agreement_rate": float((direct_pred == grounded_pred).mean()),
        "verdict_cohen_kappa": float(cohen_kappa_score(direct_pred, grounded_pred)),
        "error_cohen_kappa": float(cohen_kappa_score(~d_correct, ~g_correct)),
        "yules_q_error_association": float((a * d - b * c) / denominator) if denominator else None,
        "direct_vs_grounded_prediction_matrix": agreement.tolist(),
        "matrix_rows": list(LABELS), "matrix_columns": list(LABELS), "correctness_overlap": overlap,
    }

    rank_order = ("qrel_absent", "gold_not_retrieved", "gold_rank_after_5", "gold_rank_2_5", "gold_rank_1")
    nei_rank = {name: {"samples": 0, "direct_correct": 0, "and_correct": 0, "routes": 0, "changes": 0} for name in rank_order}
    for index, sample_id in enumerate(ids):
        if labels[index] != 2:
            continue
        gold = {str(value) for value in manifest[sample_id].get("text_evidence_ids", [])}
        retrieved = [str(value) for value in retrieval[sample_id].get("retrieved_evidence_ids", [])]
        group = rank_group(gold, retrieved)
        row = nei_rank[group]
        row["samples"] += 1
        row["direct_correct"] += int(direct_pred[index] == 2)
        row["and_correct"] += int(and_pred[index] == 2)
        row["routes"] += int(route[index])
        row["changes"] += int(and_pred[index] != direct_pred[index])
    for row in nei_rank.values():
        if row["samples"]:
            row["direct_accuracy"] = row["direct_correct"] / row["samples"]
            row["and_accuracy"] = row["and_correct"] / row["samples"]
            row["accuracy_delta"] = row["and_accuracy"] - row["direct_accuracy"]

    direct_rel, grounded_rel = probability_reliability(direct, labels, args.bins), probability_reliability(grounded, labels, args.bins)
    power = {
        "raw_official": power_approximation(args.raw_n, args.raw_helpful, args.raw_harmful, .05, .80),
        "strict": power_approximation(args.strict_n, args.strict_helpful, args.strict_harmful, .05, .80),
    }
    pareto = [
        {"name": "Direct ensemble", "milliseconds_per_claim": 320.72, "macro_f1": .54531, "color": "#4C78A8"},
        {"name": "Rationale ensemble", "milliseconds_per_claim": 193.17, "macro_f1": .53986, "color": "#F58518"},
        {"name": "CURE AND", "milliseconds_per_claim": 513.88, "macro_f1": .55470, "color": "#54A24B"},
        {"name": "CURE-Ensemble", "milliseconds_per_claim": 513.88, "macro_f1": .55507, "color": "#B279A2"},
        {"name": "CURE-Student", "milliseconds_per_claim": 64.47, "macro_f1": .54940, "color": "#E45756"},
    ]
    payload = {
        "protocol": "B18B_frozen_validation_additional_diagnostics_v1", "tau": args.tau,
        "validation_labels_used_for_diagnostics": True, "test_labels_used_for_policy_selection": False,
        "expert_complementarity": complementarity,
        "nei_posterior_reliability": {"direct": direct_rel, "rationale_trained": grounded_rel},
        "gold_nei_by_retrieval_rank": {"top_k": args.top_k, "groups": nei_rank, "diagnostic_only": True},
        "paired_design_power_approximation": power,
        "pareto_inputs": pareto,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_plots(args.output_dir, direct_rel, grounded_rel, pareto)

    lines = ["# CURE additional frozen diagnostics", "", "Official validation used for diagnostics: **yes**  ", "Test used for policy selection: **no**", "",
             "## Direct expert / rationale-trained expert complementarity", "", "| Direct \\ Rationale-trained | Supported | Refuted | NEI |", "| --- | ---: | ---: | ---: |"]
    for name, row in zip(LABELS, agreement, strict=True):
        lines.append(f"| {name} | {row[0]} | {row[1]} | {row[2]} |")
    lines += ["", f"- Prediction agreement: `{complementarity['prediction_agreement_rate']:.4f}`", f"- Verdict Cohen's kappa: `{complementarity['verdict_cohen_kappa']:.4f}`", f"- Error Cohen's kappa: `{complementarity['error_cohen_kappa']:.4f}`", f"- Yule's Q (error association): `{complementarity['yules_q_error_association']:.4f}`", "",
              "| Correctness overlap | Claims |", "| --- | ---: |"]
    lines += [f"| {name.replace('_', ' ')} | {value} |" for name, value in overlap.items()]
    lines += ["", "## Gold-NEI by retrieval rank (diagnostic only)", "", "| Retrieval status | N | Direct NEI accuracy | AND NEI accuracy | Delta | Routes/changes |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in rank_order:
        row = nei_rank[name]
        if row["samples"]:
            lines.append(f"| {name} | {row['samples']} | {row['direct_accuracy']:.4f} | {row['and_accuracy']:.4f} | {row['accuracy_delta']:+.4f} | {row['routes']}/{row['changes']} |")
    lines += ["", "## Paired-design power approximation", "", "This is a normal approximation for paired binary correctness (not a power calculation for Macro-F1).", "", "| Track | Observed accuracy effect | 80% MDE | Approx. n at observed effect |", "| --- | ---: | ---: | ---: |"]
    for name, row in power.items():
        n_at_effect = row["required_n_at_observed_effect_approx"]
        lines.append(f"| {name} | {row['observed_accuracy_effect']:+.5f} | {row['mde_accuracy_approx']:.5f} | {n_at_effect if n_at_effect else '--'} |")
    lines += ["", "Reliability and compute--quality figures were written alongside this summary."]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
