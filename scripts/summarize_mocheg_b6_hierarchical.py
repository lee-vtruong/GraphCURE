"""Summarize frozen five-seed GraphCURE-B6 validation confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_mocheg_qwen3_hierarchical_lora import probability_metrics


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "values": array.tolist(),
    }


def load_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return (
        [row["id"] for row in rows],
        np.asarray([int(row["gold"]) for row in rows]),
        np.asarray([row["probabilities"] for row in rows], dtype=float),
    )


def summarize(runs: list[dict], minimum_mean_delta: float,
              minimum_ensemble_delta: float, minimum_ensemble_f1: float,
              minimum_nei_delta: float, maximum_supported_drop: float,
              minimum_positive_seeds: int) -> dict:
    paired = [
        row["candidate_metrics"]["macro_f1"]
        - row["article_metrics"]["macro_f1"]
        for row in runs
    ]
    article_ensemble = np.mean(
        [row["article_probabilities"] for row in runs], axis=0
    )
    candidate_ensemble = np.mean(
        [row["candidate_probabilities"] for row in runs], axis=0
    )
    labels = runs[0]["labels"]
    article_metrics = probability_metrics(article_ensemble, labels)
    candidate_metrics = probability_metrics(candidate_ensemble, labels)
    ensemble_delta = (
        candidate_metrics["macro_f1"] - article_metrics["macro_f1"]
    )
    nei_delta = (
        candidate_metrics["class_f1"]["nei"]
        - article_metrics["class_f1"]["nei"]
    )
    supported_delta = (
        candidate_metrics["class_f1"]["supported"]
        - article_metrics["class_f1"]["supported"]
    )
    positive = sum(value > 0 for value in paired)
    gate = {
        "mean_delta_at_least_minimum": float(np.mean(paired)) >= minimum_mean_delta,
        "ensemble_delta_at_least_minimum": ensemble_delta >= minimum_ensemble_delta,
        "ensemble_macro_f1_at_least_target": (
            candidate_metrics["macro_f1"] >= minimum_ensemble_f1
        ),
        "ensemble_nei_delta_at_least_minimum": nei_delta >= minimum_nei_delta,
        "ensemble_supported_drop_within_limit": (
            supported_delta >= -maximum_supported_drop
        ),
        "positive_seeds_at_least_minimum": positive >= minimum_positive_seeds,
    }
    gate["passed"] = all(gate.values())
    return {
        "split": "val",
        "seeds": [row["seed"] for row in runs],
        "hierarchical_weight": runs[0]["hierarchical_weight"],
        "per_seed": [{
            "seed": row["seed"],
            "article": row["article_metrics"],
            "candidate": row["candidate_metrics"],
            "macro_f1_delta": (
                row["candidate_metrics"]["macro_f1"]
                - row["article_metrics"]["macro_f1"]
            ),
        } for row in runs],
        "paired_macro_f1_delta": {
            **aggregate(paired), "positive_seeds": positive,
        },
        "raw_article_ensemble": article_metrics,
        "raw_b6_ensemble": candidate_metrics,
        "raw_ensemble_deltas": {
            "accuracy": candidate_metrics["accuracy"] - article_metrics["accuracy"],
            "macro_f1": ensemble_delta,
            "supported_f1": supported_delta,
            "refuted_f1": (
                candidate_metrics["class_f1"]["refuted"]
                - article_metrics["class_f1"]["refuted"]
            ),
            "nei_f1": nei_delta,
        },
        "promotion_gate": gate,
        "test_split_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-template",
                        default="outputs/mocheg_b6_hierarchical_seed{seed}")
    parser.add_argument("--article-template",
                        default="outputs/mocheg_qwen3_lora_seed{seed}_v16")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[13, 21, 42, 87, 100])
    parser.add_argument("--minimum-mean-delta", type=float, default=.003)
    parser.add_argument("--minimum-ensemble-delta", type=float, default=.003)
    parser.add_argument("--minimum-ensemble-f1", type=float, default=.695)
    parser.add_argument("--minimum-nei-delta", type=float, default=.02)
    parser.add_argument("--maximum-supported-drop", type=float, default=.005)
    parser.add_argument("--minimum-positive-seeds", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/mocheg_b6_validation_summary.json"))
    parser.add_argument("--markdown", type=Path,
                        default=Path("outputs/mocheg_b6_validation_summary.md"))
    args = parser.parse_args()
    runs = []
    reference_ids = reference_labels = None
    frozen_weight = None
    for seed in args.seeds:
        candidate_root = Path(args.candidate_template.format(seed=seed))
        article_root = Path(args.article_template.format(seed=seed))
        summary_path = candidate_root / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing B6 summary for seed {seed}: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("test_split_used") is not False:
            raise ValueError(f"seed {seed} is not validation-only")
        weight = float(summary["selected_hierarchical_weight"])
        if frozen_weight is None:
            frozen_weight = weight
        elif weight != frozen_weight:
            raise ValueError(
                "B6 runs use different hierarchical weights; freeze the seed-42 "
                "weight and rerun all confirmation seeds"
            )
        candidate_ids, labels, candidate_probabilities = load_predictions(
            candidate_root / "val_predictions.jsonl"
        )
        article_ids, article_labels, article_probabilities = load_predictions(
            article_root / "val_predictions.jsonl"
        )
        if candidate_ids != article_ids or not np.array_equal(
            labels, article_labels
        ):
            raise ValueError(f"seed {seed} article/B6 predictions are not aligned")
        if reference_ids is None:
            reference_ids, reference_labels = candidate_ids, labels
        elif candidate_ids != reference_ids or not np.array_equal(
            labels, reference_labels
        ):
            raise ValueError(f"seed {seed} is not aligned with earlier seeds")
        runs.append({
            "seed": seed,
            "hierarchical_weight": weight,
            "labels": labels,
            "article_probabilities": article_probabilities,
            "candidate_probabilities": candidate_probabilities,
            "article_metrics": probability_metrics(article_probabilities, labels),
            "candidate_metrics": probability_metrics(candidate_probabilities, labels),
        })
    result = summarize(
        runs, args.minimum_mean_delta, args.minimum_ensemble_delta,
        args.minimum_ensemble_f1, args.minimum_nei_delta,
        args.maximum_supported_drop, args.minimum_positive_seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# GraphCURE-B6 frozen validation confirmation", "",
        "Test split used: **no**", "",
        f"Frozen hierarchical weight: `{result['hierarchical_weight']}`", "",
        "| Seed | Article F1 | B6 F1 | Delta | B6 NEI F1 |", 
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['article']['macro_f1']:.4f} | "
            f"{row['candidate']['macro_f1']:.4f} | "
            f"{row['macro_f1_delta']:+.4f} | "
            f"{row['candidate']['class_f1']['nei']:.4f} |"
        )
    lines += [
        "", "## Aggregate",
        f"- Paired delta: {result['paired_macro_f1_delta']['mean']:+.6f} "
        f"+/- {result['paired_macro_f1_delta']['std']:.6f}",
        f"- Raw article ensemble F1: {result['raw_article_ensemble']['macro_f1']:.6f}",
        f"- Raw B6 ensemble F1: {result['raw_b6_ensemble']['macro_f1']:.6f}",
        f"- Raw ensemble delta: {result['raw_ensemble_deltas']['macro_f1']:+.6f}",
        f"- NEI F1 delta: {result['raw_ensemble_deltas']['nei_f1']:+.6f}",
        f"- Promotion gate: **{'pass' if result['promotion_gate']['passed'] else 'fail'}**",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
