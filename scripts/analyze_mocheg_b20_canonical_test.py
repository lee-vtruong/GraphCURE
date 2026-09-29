"""Paired raw/strict P1 confirmation for the B20 headline-recipe control.

This evaluator consumes frozen prediction files only.  It validates exact ID
coverage against each manifest, averages the three same-seed model posteriors,
and reports direct-only versus rationale-trained paired statistics.  It never
selects a checkpoint, seed, threshold, or policy from test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.evaluate_mocheg_b18b_canonical_router import (
    load_probability_ensemble,
    manifest,
    validate_retrieval,
)
from scripts.summarize_mocheg_b18_seeds import compute_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_track(
    name: str,
    manifest_path: Path,
    retrieval_path: Path,
    expected_samples: int,
    direct_paths: list[Path],
    rationale_paths: list[Path],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    ids, labels, _ = manifest(manifest_path, expected_samples)
    validate_retrieval(retrieval_path, ids, labels)
    direct_probs = load_probability_ensemble(direct_paths, ids, labels)
    rationale_probs = load_probability_ensemble(rationale_paths, ids, labels)
    direct_pred = direct_probs.argmax(axis=1)
    rationale_pred = rationale_probs.argmax(axis=1)
    direct_metrics = compute_metrics(labels, direct_pred)
    rationale_metrics = compute_metrics(labels, rationale_pred)
    direct_correct = direct_pred == labels
    rationale_correct = rationale_pred == labels
    helpful = int(np.sum(~direct_correct & rationale_correct))
    harmful = int(np.sum(direct_correct & ~rationale_correct))
    comparison = {
        "macro_f1_delta": float(rationale_metrics["macro_f1"] - direct_metrics["macro_f1"]),
        "accuracy_delta": float(rationale_metrics["accuracy"] - direct_metrics["accuracy"]),
        "class_f1_delta": {
            "f1_supported": float(rationale_metrics["f1_supported"] - direct_metrics["f1_supported"]),
            "f1_refuted": float(rationale_metrics["f1_refuted"] - direct_metrics["f1_refuted"]),
            "f1_nei": float(rationale_metrics["f1_nei"] - direct_metrics["f1_nei"]),
        },
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(labels, direct_pred, rationale_pred, iterations, seed),
    }
    return {
        "track": name,
        "samples": expected_samples,
        "manifest_sha256": sha256_file(manifest_path),
        "retrieval_sha256": sha256_file(retrieval_path),
        "direct_prediction_sha256": [sha256_file(path) for path in direct_paths],
        "rationale_prediction_sha256": [sha256_file(path) for path in rationale_paths],
        "headline_recipe_direct_only": direct_metrics,
        "rationale_trained": rationale_metrics,
        "rationale_minus_direct": comparison,
    }


def markdown_track(row: dict[str, Any]) -> list[str]:
    direct = row["headline_recipe_direct_only"]
    rationale = row["rationale_trained"]
    comp = row["rationale_minus_direct"]
    boot = comp["bootstrap"]
    return [
        f"## {row['track']} ($n={row['samples']:,}$)", "",
        "| Ensemble | Accuracy | Macro-F1 | F1 Supported | F1 Refuted | F1 NEI |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Headline-recipe direct-only | {direct['accuracy']:.6f} | {direct['macro_f1']:.6f} | {direct['f1_supported']:.6f} | {direct['f1_refuted']:.6f} | {direct['f1_nei']:.6f} |",
        f"| Rationale-trained | {rationale['accuracy']:.6f} | {rationale['macro_f1']:.6f} | {rationale['f1_supported']:.6f} | {rationale['f1_refuted']:.6f} | {rationale['f1_nei']:.6f} |",
        "",
        "### Rationale-trained minus headline-recipe direct-only", "",
        f"- Macro-F1 delta: `{comp['macro_f1_delta']:+.6f}`",
        f"- Accuracy delta: `{comp['accuracy_delta']:+.6f}`",
        f"- Helpful/harmful: `{comp['helpful']}/{comp['harmful']}`",
        f"- Exact McNemar p: `{comp['exact_mcnemar_p']:.4f}`",
        f"- Bootstrap 95% CI: `[{boot['ci_95_percentile'][0]:+.6f}, {boot['ci_95_percentile'][1]:+.6f}]`",
        f"- Bootstrap P(delta > 0): `{boot['probability_delta_positive']:.4f}`", "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--strict-manifest", type=Path, required=True)
    parser.add_argument("--raw-retrieval", type=Path, required=True)
    parser.add_argument("--strict-retrieval", type=Path, required=True)
    parser.add_argument("--raw-direct", type=Path, nargs=3, required=True)
    parser.add_argument("--strict-direct", type=Path, nargs=3, required=True)
    parser.add_argument("--raw-rationale", type=Path, nargs=3, required=True)
    parser.add_argument("--strict-rationale", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    raw = evaluate_track("Raw official P1", args.raw_manifest, args.raw_retrieval, 2442,
                         args.raw_direct, args.raw_rationale, args.iterations, args.seed)
    strict = evaluate_track("Strict duplicate-safe P1", args.strict_manifest, args.strict_retrieval, 2434,
                            args.strict_direct, args.strict_rationale, args.iterations, args.seed + 1)
    result = {
        "protocol": "B20_headline_recipe_direct_only_canonical_raw_and_strict_test",
        "frozen_recipe": {
            "mode": "matched_control", "lambda_exp": 0.0, "epochs": 3,
            "batch_size": 2, "gradient_accumulation": 4, "learning_rate": 2e-4,
            "top_k": 5, "inject_train_gold": False,
        },
        "test_labels_used_for_selection": False,
        "raw": raw,
        "strict": strict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# B20 headline-recipe direct-only canonical test confirmation", "",
             "Test labels used for policy/checkpoint/seed selection: **no**.",
             "The direct-only and rationale-trained ensembles use the same frozen seeds (42, 87, 100).", ""]
    lines.extend(markdown_track(raw))
    lines.extend(markdown_track(strict))
    lines.append("Raw and strict predictions are independently inferred under their respective retrieval manifests.")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
