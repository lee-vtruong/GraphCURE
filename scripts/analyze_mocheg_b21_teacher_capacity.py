"""Frozen raw/strict test analysis for the Qwen2.5 teacher-capacity ablation."""
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    name: str, manifest_path: Path, retrieval_path: Path, expected: int,
    teacher_3b: list[Path], teacher_7b: list[Path], iterations: int, seed: int,
) -> dict[str, Any]:
    ids, labels, _ = manifest(manifest_path, expected)
    validate_retrieval(retrieval_path, ids, labels)
    p3 = load_probability_ensemble(teacher_3b, ids, labels)
    p7 = load_probability_ensemble(teacher_7b, ids, labels)
    y3, y7 = p3.argmax(axis=1), p7.argmax(axis=1)
    m3, m7 = compute_metrics(labels, y3), compute_metrics(labels, y7)
    helpful = int(np.sum((y3 == labels) & (y7 != labels)))
    harmful = int(np.sum((y3 != labels) & (y7 == labels)))
    return {
        "track": name, "samples": expected,
        "manifest_sha256": sha256(manifest_path), "retrieval_sha256": sha256(retrieval_path),
        "teacher_3b_prediction_sha256": [sha256(p) for p in teacher_3b],
        "teacher_7b_prediction_sha256": [sha256(p) for p in teacher_7b],
        "teacher_3b": m3, "teacher_7b": m7,
        "teacher_7b_minus_3b": {
            "macro_f1_delta": float(m7["macro_f1"] - m3["macro_f1"]),
            "accuracy_delta": float(m7["accuracy"] - m3["accuracy"]),
            "class_f1_delta": {
                "f1_supported": float(m7["f1_supported"] - m3["f1_supported"]),
                "f1_refuted": float(m7["f1_refuted"] - m3["f1_refuted"]),
                "f1_nei": float(m7["f1_nei"] - m3["f1_nei"]),
            },
            "helpful": helpful, "harmful": harmful,
            "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
            "bootstrap": bootstrap_delta(labels, y3, y7, iterations, seed),
        },
    }


def report(track: dict[str, Any]) -> list[str]:
    low, high, comp = track["teacher_3b"], track["teacher_7b"], track["teacher_7b_minus_3b"]
    boot = comp["bootstrap"]
    return [
        f"## {track['track']} ($n={track['samples']:,}$)", "",
        "| Rationale teacher | Accuracy | Macro-F1 | F1 Supported | F1 Refuted | F1 NEI |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Qwen2.5-3B-Instruct | {low['accuracy']:.6f} | {low['macro_f1']:.6f} | {low['f1_supported']:.6f} | {low['f1_refuted']:.6f} | {low['f1_nei']:.6f} |",
        f"| Qwen2.5-7B-Instruct | {high['accuracy']:.6f} | {high['macro_f1']:.6f} | {high['f1_supported']:.6f} | {high['f1_refuted']:.6f} | {high['f1_nei']:.6f} |",
        "", "### 7B teacher minus 3B teacher", "",
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
    parser.add_argument("--raw-3b", type=Path, nargs=3, required=True)
    parser.add_argument("--strict-3b", type=Path, nargs=3, required=True)
    parser.add_argument("--raw-7b", type=Path, nargs=3, required=True)
    parser.add_argument("--strict-7b", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    raw = evaluate("Raw official P1", args.raw_manifest, args.raw_retrieval, 2442,
                   args.raw_3b, args.raw_7b, args.iterations, args.seed)
    strict = evaluate("Strict duplicate-safe P1", args.strict_manifest, args.strict_retrieval, 2434,
                      args.strict_3b, args.strict_7b, args.iterations, args.seed + 1)
    result = {
        "protocol": "B21_frozen_Qwen25_teacher_capacity_ablation",
        "teacher_variable": {"lower_capacity": "Qwen/Qwen2.5-3B-Instruct", "reference": "Qwen/Qwen2.5-7B-Instruct"},
        "student_recipe_fixed": {"model": "Qwen/Qwen3-4B-Instruct-2507", "lambda_exp": 0.25, "seeds": [42, 87, 100], "top_k": 5},
        "test_labels_used_for_selection": False, "raw": raw, "strict": strict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# B21 frozen Qwen2.5 teacher-capacity ablation", "",
             "Only teacher capacity changes: Qwen2.5-3B versus Qwen2.5-7B. Student recipe, seeds, retrieval, prompt, and test policy are fixed.",
             "Test labels used for teacher/model/seed/policy selection: **no**.", ""]
    lines.extend(report(raw)); lines.extend(report(strict))
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
