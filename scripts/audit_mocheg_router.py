"""Audit whether a frozen specialist router has learned useful ranking.

This script is validation-only by default. It reports paired outcome counts,
ranking quality for visual-help cases, an exact McNemar test, and a paired
bootstrap confidence interval for the Macro-F1 delta over the text anchor.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def exact_mcnemar_p(helpful: int, harmful: int) -> float:
    """Two-sided exact binomial McNemar p-value."""
    discordant = helpful + harmful
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(helpful, harmful) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def safe_ranking(target: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(target)) < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(target, score)),
        "average_precision": float(average_precision_score(target, score)),
    }


def bootstrap_delta(
    gold: np.ndarray,
    text: np.ndarray,
    routed: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    generator = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected = generator.integers(0, len(gold), size=len(gold))
        values[iteration] = (
            f1_score(gold[selected], routed[selected], average="macro")
            - f1_score(gold[selected], text[selected], average="macro")
        )
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "iterations": iterations,
        "seed": seed,
        "mean_delta": float(values.mean()),
        "ci_95_percentile": [float(lower), float(upper)],
        "probability_delta_positive": float(np.mean(values > 0)),
    }


def audit(rows: list[dict], threshold: float, iterations: int, seed: int) -> dict:
    gold = np.asarray([row["gold"] for row in rows], dtype=np.int64)
    text = np.asarray(
        [row["text_only_prediction"] for row in rows], dtype=np.int64
    )
    expert = np.asarray(
        [row["visual_expert_prediction"] for row in rows], dtype=np.int64
    )
    gate = np.asarray(
        [row["visual_modality_mass"] for row in rows], dtype=np.float64
    )
    route = gate >= threshold
    routed = np.where(route, expert, text)
    helpful_expert = (text != gold) & (expert == gold)
    harmful_expert = (text == gold) & (expert != gold)
    decisive = helpful_expert | harmful_expert
    selected_help = (text != gold) & (routed == gold)
    selected_harm = (text == gold) & (routed != gold)

    result = {
        "samples": len(rows),
        "threshold": threshold,
        "text_anchor": {
            "accuracy": float(accuracy_score(gold, text)),
            "macro_f1": float(f1_score(gold, text, average="macro")),
        },
        "visual_expert": {
            "accuracy": float(accuracy_score(gold, expert)),
            "macro_f1": float(f1_score(gold, expert, average="macro")),
            "potential_helpful": int(helpful_expert.sum()),
            "potential_harmful": int(harmful_expert.sum()),
        },
        "hard_router": {
            "accuracy": float(accuracy_score(gold, routed)),
            "macro_f1": float(f1_score(gold, routed, average="macro")),
            "accuracy_delta": float(
                accuracy_score(gold, routed) - accuracy_score(gold, text)
            ),
            "macro_f1_delta": float(
                f1_score(gold, routed, average="macro")
                - f1_score(gold, text, average="macro")
            ),
            "route_count": int(route.sum()),
            "route_rate": float(route.mean()),
            "helpful": int(selected_help.sum()),
            "harmful": int(selected_harm.sum()),
            "exact_mcnemar_p": exact_mcnemar_p(
                int(selected_help.sum()), int(selected_harm.sum())
            ),
        },
        "gate_ranking": {
            "all_samples_helpful_target": safe_ranking(
                helpful_expert.astype(np.int64), gate
            ),
            "decisive_help_vs_harm": safe_ranking(
                helpful_expert[decisive].astype(np.int64), gate[decisive]
            ),
            "decisive_samples": int(decisive.sum()),
            "helpful_gate_mean": float(gate[helpful_expert].mean())
            if helpful_expert.any() else None,
            "harmful_gate_mean": float(gate[harmful_expert].mean())
            if harmful_expert.any() else None,
            "neutral_gate_mean": float(gate[~decisive].mean())
            if (~decisive).any() else None,
        },
        "paired_bootstrap_macro_f1": bootstrap_delta(
            gold, text, routed, iterations, seed
        ),
        "interpretation": {
            "same_validation_used_for_threshold_selection": True,
            "test_split_used": False,
            "warning": (
                "Threshold selection and this audit use the same validation "
                "split; confidence intervals are exploratory, not a final "
                "unbiased estimate."
            ),
        },
    }
    return result


def markdown(result: dict) -> str:
    hard = result["hard_router"]
    ranking = result["gate_ranking"]
    bootstrap = result["paired_bootstrap_macro_f1"]
    decisive = ranking["decisive_help_vs_harm"]
    all_samples = ranking["all_samples_helpful_target"]
    return "\n".join([
        "# MOCHEG hard-router validation audit",
        "",
        "Test split used: **no**",
        "",
        "| System | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
        f"| Text anchor | {result['text_anchor']['accuracy']:.4f} | "
        f"{result['text_anchor']['macro_f1']:.4f} |",
        f"| Visual expert | {result['visual_expert']['accuracy']:.4f} | "
        f"{result['visual_expert']['macro_f1']:.4f} |",
        f"| Hard router | {hard['accuracy']:.4f} | {hard['macro_f1']:.4f} |",
        "",
        f"- Threshold: `{result['threshold']:.4f}`; routed: "
        f"`{hard['route_count']}/{result['samples']}` ({hard['route_rate']:.4f}).",
        f"- Selected help/harm: `{hard['helpful']}/{hard['harmful']}`; exact "
        f"McNemar p: `{hard['exact_mcnemar_p']:.6f}`.",
        f"- Macro-F1 delta: `{hard['macro_f1_delta']:+.6f}`; paired bootstrap "
        f"95% CI: `[{bootstrap['ci_95_percentile'][0]:+.6f}, "
        f"{bootstrap['ci_95_percentile'][1]:+.6f}]`.",
        f"- Gate AUROC/AUPRC for helpful vs all: "
        f"`{all_samples['auroc']}` / `{all_samples['average_precision']}`.",
        f"- Gate AUROC/AUPRC among decisive help-vs-harm examples: "
        f"`{decisive['auroc']}` / `{decisive['average_precision']}`.",
        "",
        "> This is an exploratory validation audit because the same split "
        "selected the threshold. Do not interpret it as a final test result.",
        "",
    ])


def resolve_threshold(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return args.threshold
    metrics_path = args.metrics
    if metrics_path is None:
        candidate = args.predictions.with_name("val_metrics.json")
        metrics_path = candidate if candidate.exists() else None
    if metrics_path is None:
        raise ValueError("provide --threshold or --metrics")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = metrics.get("hard_routing_threshold")
    if threshold is None:
        raise ValueError(f"no hard_routing_threshold in {metrics_path}")
    return float(threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        parser.error("predictions file is empty")
    threshold = resolve_threshold(args)
    result = audit(
        rows, threshold, args.bootstrap_iterations, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {args.output} and {markdown_path}")


if __name__ == "__main__":
    main()
