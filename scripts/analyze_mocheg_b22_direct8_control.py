"""Evaluate the fixed-size direct-only control against heterogeneous CURE.

Every member path is explicit.  This prevents a test-time decision to omit a
weak additional direct seed and compares the two eight-member systems under
the same raw/strict P1 manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.evaluate_mocheg_b18b_canonical_router import (
    load_probability_ensemble, manifest, validate_retrieval,
)
from scripts.summarize_mocheg_b18_seeds import compute_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
            iterations: int, seed: int) -> dict:
    base_correct, candidate_correct = baseline == labels, candidate == labels
    helpful = int(np.sum(~base_correct & candidate_correct))
    harmful = int(np.sum(base_correct & ~candidate_correct))
    return {
        "macro_f1_delta": float(compute_metrics(labels, candidate)["macro_f1"] - compute_metrics(labels, baseline)["macro_f1"]),
        "accuracy_delta": float(np.mean(candidate_correct) - np.mean(base_correct)),
        "helpful": helpful, "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(labels, baseline, candidate, iterations, seed),
    }


def track(name: str, manifest_path: Path, retrieval_path: Path, expected: int,
          legacy: list[Path], added: list[Path], rationale: list[Path],
          iterations: int, seed: int) -> dict:
    ids, labels, _ = manifest(manifest_path, expected)
    validate_retrieval(retrieval_path, ids, labels)
    direct5 = load_probability_ensemble(legacy, ids, labels)
    direct8 = load_probability_ensemble(legacy + added, ids, labels)
    heterogeneous8 = load_probability_ensemble(legacy + rationale, ids, labels)
    predictions = {
        "direct5": direct5.argmax(axis=1), "direct8": direct8.argmax(axis=1),
        "heterogeneous8": heterogeneous8.argmax(axis=1),
    }
    return {
        "name": name, "samples": expected,
        "manifest_sha256": sha256_file(manifest_path),
        "retrieval_sha256": sha256_file(retrieval_path),
        "metrics": {key: compute_metrics(labels, value) for key, value in predictions.items()},
        "direct8_minus_direct5": compare(labels, predictions["direct5"], predictions["direct8"], iterations, seed),
        "heterogeneous8_minus_direct8": compare(labels, predictions["direct8"], predictions["heterogeneous8"], iterations, seed + 1),
        "input_prediction_sha256": {str(path): sha256_file(path) for path in legacy + added + rationale},
    }


def row(name: str, metric: dict) -> str:
    return (f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | "
            f"{metric['f1_supported']:.6f} | {metric['f1_refuted']:.6f} | {metric['f1_nei']:.6f} |")


def comparison(label: str, values: dict) -> list[str]:
    bootstrap = values["bootstrap"]
    return [f"### {label}", "", f"- Macro-F1 delta: `{values['macro_f1_delta']:+.6f}`",
            f"- Accuracy delta: `{values['accuracy_delta']:+.6f}`",
            f"- Helpful/harmful: `{values['helpful']}/{values['harmful']}`",
            f"- Exact McNemar p: `{values['exact_mcnemar_p']:.4f}`",
            f"- Bootstrap 95% CI: `[{bootstrap['ci_95_percentile'][0]:+.6f}, {bootstrap['ci_95_percentile'][1]:+.6f}]`",
            f"- Bootstrap P(delta > 0): `{bootstrap['probability_delta_positive']:.4f}`", ""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--strict-manifest", type=Path, required=True)
    parser.add_argument("--raw-retrieval", type=Path, required=True)
    parser.add_argument("--strict-retrieval", type=Path, required=True)
    parser.add_argument("--raw-legacy-direct", type=Path, nargs=5, required=True)
    parser.add_argument("--strict-legacy-direct", type=Path, nargs=5, required=True)
    parser.add_argument("--raw-added-direct", type=Path, nargs=3, required=True)
    parser.add_argument("--strict-added-direct", type=Path, nargs=3, required=True)
    parser.add_argument("--raw-rationale", type=Path, nargs=3, required=True)
    parser.add_argument("--strict-rationale", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    raw = track("Raw official P1", args.raw_manifest, args.raw_retrieval, 2442,
                args.raw_legacy_direct, args.raw_added_direct, args.raw_rationale,
                args.iterations, args.seed)
    strict = track("Strict duplicate-safe P1", args.strict_manifest, args.strict_retrieval, 2434,
                   args.strict_legacy_direct, args.strict_added_direct, args.strict_rationale,
                   args.iterations, args.seed + 10)
    result = {
        "protocol": "B22_predeclared_eight_member_direct_only_size_control",
        "predeclared_added_seeds": [314, 2718, 2026],
        "member_policy": "all five legacy and all three added direct members are included; no test-time pruning or weighting",
        "test_labels_used_for_seed_checkpoint_or_weight_selection": False,
        "raw": raw, "strict": strict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# B22 predeclared eight-member direct-only size control", "",
             "- Added seeds are fixed before test inference: `314, 2718, 2026`.",
             "- Every predeclared member receives equal probability weight; no test-time member pruning, seed selection, or weighting is performed.",
             "- The heterogeneous comparator has the same member count: five legacy direct experts plus three rationale-trained experts.",
             ""]
    for report in (raw, strict):
        metrics = report["metrics"]
        lines.extend([f"## {report['name']} ($n={report['samples']:,}$)", "",
                      "| Ensemble | Accuracy | Macro-F1 | F1 Supported | F1 Refuted | F1 NEI |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |",
                      row("Legacy direct (5)", metrics["direct5"]),
                      row("Direct-only size control (8)", metrics["direct8"]),
                      row("Heterogeneous CURE-Ensemble (5+3)", metrics["heterogeneous8"]), ""])
        lines.extend(comparison("Direct-only 8 minus legacy direct 5", report["direct8_minus_direct5"]))
        lines.extend(comparison("Heterogeneous 5+3 minus direct-only 8", report["heterogeneous8_minus_direct8"]))
    lines.append("Raw and strict members are independently inferred against their respective retrieval manifests.")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
