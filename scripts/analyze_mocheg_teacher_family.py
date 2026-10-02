"""Frozen raw/strict comparison of two rationale-teacher families."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.evaluate_mocheg_b18b_canonical_router import (
    load_probability_ensemble, manifest, validate_retrieval,
)
from scripts.summarize_mocheg_b18_seeds import compute_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    name: str, manifest_path: Path, retrieval_path: Path, expected: int,
    candidate: list[Path], reference: list[Path], iterations: int, seed: int,
) -> dict[str, Any]:
    ids, labels, _ = manifest(manifest_path, expected)
    validate_retrieval(retrieval_path, ids, labels)
    p_candidate = load_probability_ensemble(candidate, ids, labels)
    p_reference = load_probability_ensemble(reference, ids, labels)
    y_candidate, y_reference = p_candidate.argmax(axis=1), p_reference.argmax(axis=1)
    candidate_metrics = compute_metrics(labels, y_candidate)
    reference_metrics = compute_metrics(labels, y_reference)
    helpful = int(np.sum((y_candidate == labels) & (y_reference != labels)))
    harmful = int(np.sum((y_candidate != labels) & (y_reference == labels)))
    return {
        "track": name, "samples": expected,
        "manifest_sha256": sha256(manifest_path), "retrieval_sha256": sha256(retrieval_path),
        "candidate_prediction_sha256": [sha256(path) for path in candidate],
        "reference_prediction_sha256": [sha256(path) for path in reference],
        "candidate": candidate_metrics, "reference": reference_metrics,
        "candidate_minus_reference": {
            "macro_f1_delta": float(candidate_metrics["macro_f1"] - reference_metrics["macro_f1"]),
            "accuracy_delta": float(candidate_metrics["accuracy"] - reference_metrics["accuracy"]),
            "helpful": helpful, "harmful": harmful,
            "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
            "bootstrap": bootstrap_delta(labels, y_reference, y_candidate, iterations, seed),
        },
    }


def markdown(track: dict[str, Any], candidate_name: str, reference_name: str) -> list[str]:
    candidate, reference, comparison = track["candidate"], track["reference"], track["candidate_minus_reference"]
    bootstrap = comparison["bootstrap"]
    return [
        f"## {track['track']} ($n={track['samples']:,}$)", "",
        "| Rationale teacher | Accuracy | Macro-F1 | F1 Supported | F1 Refuted | F1 NEI |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| {candidate_name} | {candidate['accuracy']:.6f} | {candidate['macro_f1']:.6f} | {candidate['f1_supported']:.6f} | {candidate['f1_refuted']:.6f} | {candidate['f1_nei']:.6f} |",
        f"| {reference_name} | {reference['accuracy']:.6f} | {reference['macro_f1']:.6f} | {reference['f1_supported']:.6f} | {reference['f1_refuted']:.6f} | {reference['f1_nei']:.6f} |",
        "", f"### {candidate_name} minus {reference_name}", "",
        f"- Macro-F1 delta: `{comparison['macro_f1_delta']:+.6f}`",
        f"- Accuracy delta: `{comparison['accuracy_delta']:+.6f}`",
        f"- Helpful/harmful: `{comparison['helpful']}/{comparison['harmful']}`",
        f"- Exact McNemar p: `{comparison['exact_mcnemar_p']:.4f}`",
        f"- Bootstrap 95% CI: `[{bootstrap['ci_95_percentile'][0]:+.6f}, {bootstrap['ci_95_percentile'][1]:+.6f}]`",
        f"- Bootstrap P(delta > 0): `{bootstrap['probability_delta_positive']:.4f}`", "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--strict-manifest", type=Path, required=True)
    parser.add_argument("--raw-retrieval", type=Path, required=True)
    parser.add_argument("--strict-retrieval", type=Path, required=True)
    parser.add_argument("--raw-candidate", type=Path, nargs=3, required=True)
    parser.add_argument("--strict-candidate", type=Path, nargs=3, required=True)
    parser.add_argument("--raw-reference", type=Path, nargs=3, required=True)
    parser.add_argument("--strict-reference", type=Path, nargs=3, required=True)
    parser.add_argument("--candidate-teacher", required=True)
    parser.add_argument("--reference-teacher", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    raw = evaluate("Raw official P1", args.raw_manifest, args.raw_retrieval, 2442,
                   args.raw_candidate, args.raw_reference, args.iterations, args.seed)
    strict = evaluate("Strict duplicate-safe P1", args.strict_manifest, args.strict_retrieval, 2434,
                      args.strict_candidate, args.strict_reference, args.iterations, args.seed + 1)
    result = {
        "protocol": "B23_frozen_teacher_family_ablation",
        "candidate_teacher": args.candidate_teacher,
        "reference_teacher": args.reference_teacher,
        "student_recipe_fixed": {"model": "Qwen/Qwen3-4B-Instruct-2507", "lambda_exp": .25,
                                 "seeds": [42, 87, 100], "top_k": 5},
        "test_labels_used_for_selection": False,
        "raw": raw, "strict": strict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# B23 frozen teacher-family ablation", "",
             "Only the rationale-teacher family changes. Student recipe, seeds, retrieval, prompt, and test policy are fixed.",
             "Test labels used for teacher/model/seed/policy selection: **no**.", ""]
    lines.extend(markdown(raw, args.candidate_teacher, args.reference_teacher))
    lines.extend(markdown(strict, args.candidate_teacher, args.reference_teacher))
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
