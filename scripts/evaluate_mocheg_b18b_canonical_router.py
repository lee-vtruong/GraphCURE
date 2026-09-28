"""Materialize a checksum-audited canonical B18B AND evaluation.

This evaluator deliberately consumes *saved, frozen* raw-official prediction
files.  It neither loads a model nor tunes any policy on the test split.  Its
job is to make the raw P1 and strict P1 reports reproducible under their
respective frozen retrieval inputs:

* raw official P1: the 2,442 IDs in the official manifest;
* strict P1: the 2,434-ID duplicate-safe subset of the raw ID universe;
* direct and grounded experts must each cover every raw ID exactly once;
* if shared raw/strict ranked evidence IDs differ, strict predictions must be
  supplied from independent frozen inference (rather than an invalid slice of
  raw predictions); and
* the already frozen AND policy is evaluated at an explicit ``(K, tau)``.

The output records hashes for every input and materializes direct, grounded,
and AND predictions in both manifest orders.  It is intentionally stricter
than an intersection evaluation: any missing, duplicated, or mismatched ID is
an error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.summarize_mocheg_b18_seeds import compute_metrics


LABEL_KEYS = ("label", "gold", "gold_label", "target")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON row in {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no JSONL rows in {path}")
    return rows


def id_map(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "id" not in row:
            raise KeyError(f"row without id in {path}")
        sample_id = str(row["id"])
        if sample_id in result:
            raise ValueError(f"duplicate id {sample_id!r} in {path}")
        result[sample_id] = row
    return result


def row_label(row: dict[str, Any], path: Path, sample_id: str) -> int:
    for key in LABEL_KEYS:
        if key in row:
            value = int(row[key])
            if value not in (0, 1, 2):
                raise ValueError(f"invalid label {value} for {sample_id!r} in {path}")
            return value
    raise KeyError(f"no label field for {sample_id!r} in {path}; expected {LABEL_KEYS}")


def manifest(path: Path, expected: int) -> tuple[list[str], np.ndarray, dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    by_id = id_map(rows, path)
    if len(rows) != expected:
        raise ValueError(f"{path} has {len(rows)} IDs, expected {expected}")
    ordered_ids = [str(row["id"]) for row in rows]
    labels = np.asarray([row_label(by_id[sample_id], path, sample_id) for sample_id in ordered_ids])
    return ordered_ids, labels, by_id


def validate_retrieval(
    path: Path,
    ordered_ids: list[str],
    labels: np.ndarray,
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id = id_map(rows, path)
    expected_ids = set(ordered_ids)
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        extra = sorted(set(by_id) - expected_ids)
        raise ValueError(
            f"retrieval IDs do not match manifest for {path}: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    for index, sample_id in enumerate(ordered_ids):
        row = by_id[sample_id]
        if any(key in row for key in LABEL_KEYS):
            retrieved_label = row_label(row, path, sample_id)
            if retrieved_label != int(labels[index]):
                raise ValueError(f"retrieval label mismatch for {sample_id!r} in {path}")
    return by_id


def retrieval_fingerprint(row: dict[str, Any]) -> dict[str, Any]:
    """Return the evidence identity consumed by the verifier.

    The raw and duplicate-safe manifests are generated in separate retrieval
    passes.  Their continuous dense/reranker scores and run signatures can
    consequently differ at floating-point precision even when the ranked
    evidence IDs (the verifier input) are identical.  For deriving the strict
    track from a raw prediction, evidence identity and order must match; score
    metadata must not be mistaken for a changed verifier input.
    """
    return {"retrieved_evidence_ids": row.get("retrieved_evidence_ids")}


def validate_strict_subset(
    raw_ids: list[str],
    raw_labels: np.ndarray,
    raw_manifest: dict[str, dict[str, Any]],
    strict_ids: list[str],
    strict_labels: np.ndarray,
    strict_manifest: dict[str, dict[str, Any]],
    raw_retrieval: dict[str, dict[str, Any]],
    strict_retrieval: dict[str, dict[str, Any]],
) -> bool:
    raw_index = {sample_id: index for index, sample_id in enumerate(raw_ids)}
    if not set(strict_ids) < set(raw_ids):
        raise ValueError("strict manifest IDs are not a proper subset of raw official IDs")
    evidence_ids_match = True
    for index, sample_id in enumerate(strict_ids):
        if int(strict_labels[index]) != int(raw_labels[raw_index[sample_id]]):
            raise ValueError(f"raw/strict manifest label mismatch for {sample_id!r}")
        for key in ("claim", "claim_id"):
            if key in raw_manifest[sample_id] and key in strict_manifest[sample_id]:
                if raw_manifest[sample_id][key] != strict_manifest[sample_id][key]:
                    raise ValueError(f"raw/strict manifest {key} mismatch for {sample_id!r}")
        if retrieval_fingerprint(raw_retrieval[sample_id]) != retrieval_fingerprint(
            strict_retrieval[sample_id]
        ):
            evidence_ids_match = False
    return evidence_ids_match


def load_probability_ensemble(
    paths: list[Path],
    ordered_ids: list[str],
    labels: np.ndarray,
) -> np.ndarray:
    expected_ids = set(ordered_ids)
    all_probabilities: list[np.ndarray] = []
    for path in paths:
        rows = read_jsonl(path)
        by_id = id_map(rows, path)
        if set(by_id) != expected_ids:
            missing = sorted(expected_ids - set(by_id))
            extra = sorted(set(by_id) - expected_ids)
            raise ValueError(
                f"prediction IDs do not exactly match raw manifest for {path}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        probabilities: list[list[float]] = []
        for index, sample_id in enumerate(ordered_ids):
            row = by_id[sample_id]
            if row_label(row, path, sample_id) != int(labels[index]):
                raise ValueError(f"prediction label mismatch for {sample_id!r} in {path}")
            values = np.asarray(row.get("probabilities"), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid probability vector for {sample_id!r} in {path}")
            if np.any(values < -1e-8) or not np.isclose(values.sum(), 1.0, atol=1e-4):
                raise ValueError(f"non-probability vector for {sample_id!r} in {path}")
            probabilities.append(values.tolist())
        all_probabilities.append(np.asarray(probabilities, dtype=float))
    return np.mean(all_probabilities, axis=0)


def evaluate_policy(
    labels: np.ndarray,
    direct: np.ndarray,
    grounded: np.ndarray,
    tau: float,
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    direct_prediction = direct.argmax(axis=1)
    grounded_prediction = grounded.argmax(axis=1)
    route = (grounded_prediction == 2) & (grounded[:, 2] >= tau)
    and_prediction = direct_prediction.copy()
    and_prediction[route] = 2
    direct_metrics = compute_metrics(labels, direct_prediction)
    grounded_metrics = compute_metrics(labels, grounded_prediction)
    and_metrics = compute_metrics(labels, and_prediction)
    direct_correct = direct_prediction == labels
    and_correct = and_prediction == labels
    helpful = int(np.sum(~direct_correct & and_correct))
    harmful = int(np.sum(direct_correct & ~and_correct))
    comparison = {
        "macro_f1_delta": float(and_metrics["macro_f1"] - direct_metrics["macro_f1"]),
        "accuracy_delta": float(and_metrics["accuracy"] - direct_metrics["accuracy"]),
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(labels, direct_prediction, and_prediction, iterations, seed),
    }
    return ({
        "direct": direct_metrics,
        "grounded": grounded_metrics,
        "and": and_metrics,
        "comparison_vs_direct": comparison,
        "routing": {"route_count": int(route.sum()), "route_rate": float(route.mean())},
    }, direct_prediction, grounded_prediction, and_prediction)


def write_predictions(
    path: Path,
    ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    prediction: np.ndarray,
) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(ids):
            handle.write(json.dumps({
                "id": sample_id,
                "label": int(labels[index]),
                "prediction": int(prediction[index]),
                "probabilities": probabilities[index].tolist(),
            }) + "\n")
    return sha256_file(path)


def compact_row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} | "
        f"{metrics['f1_supported']:.6f} | {metrics['f1_refuted']:.6f} | "
        f"{metrics['f1_nei']:.6f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--strict-manifest", type=Path, required=True)
    parser.add_argument("--raw-retrieval", type=Path, required=True)
    parser.add_argument("--strict-retrieval", type=Path, required=True)
    parser.add_argument("--direct-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--grounded-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--strict-direct-runs", type=Path, nargs="+")
    parser.add_argument("--strict-grounded-runs", type=Path, nargs="+")
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--expected-raw", type=int, default=2442)
    parser.add_argument("--expected-strict", type=int, default=2434)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--prediction-provenance",
        choices=("existing_frozen_predictions", "canonical_raw_inference"),
        default="existing_frozen_predictions",
        help="Whether canonical-tag raw predictions were freshly inferred before this audit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.tau <= 1.0:
        parser.error("--tau must be in [0, 1]")
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    raw_ids, raw_labels, raw_manifest = manifest(args.raw_manifest, args.expected_raw)
    strict_ids, strict_labels, strict_manifest = manifest(
        args.strict_manifest, args.expected_strict
    )
    raw_retrieval = validate_retrieval(args.raw_retrieval, raw_ids, raw_labels)
    strict_retrieval = validate_retrieval(args.strict_retrieval, strict_ids, strict_labels)
    shared_evidence_ids_match = validate_strict_subset(
        raw_ids, raw_labels, raw_manifest, strict_ids, strict_labels, strict_manifest,
        raw_retrieval, strict_retrieval,
    )

    direct_raw = load_probability_ensemble(args.direct_runs, raw_ids, raw_labels)
    grounded_raw = load_probability_ensemble(args.grounded_runs, raw_ids, raw_labels)
    raw_result, raw_direct_pred, raw_grounded_pred, raw_and_pred = evaluate_policy(
        raw_labels, direct_raw, grounded_raw, args.tau, args.iterations, args.seed
    )
    strict_prediction_source: str
    if shared_evidence_ids_match:
        if bool(args.strict_direct_runs) != bool(args.strict_grounded_runs):
            parser.error("provide both --strict-direct-runs and --strict-grounded-runs, or neither")
        # The verified input is identical, so slicing the raw prediction is exact.
        raw_index = {sample_id: index for index, sample_id in enumerate(raw_ids)}
        strict_positions = np.asarray([raw_index[sample_id] for sample_id in strict_ids], dtype=int)
        strict_direct = direct_raw[strict_positions]
        strict_grounded = grounded_raw[strict_positions]
        strict_prediction_source = "verified_raw_prediction_subset"
    else:
        if not args.strict_direct_runs or not args.strict_grounded_runs:
            parser.error(
                "raw and strict ranked evidence IDs differ; independent strict "
                "predictions are required via --strict-direct-runs and "
                "--strict-grounded-runs"
            )
        if len(args.strict_direct_runs) != len(args.direct_runs):
            parser.error("strict direct member count must equal raw direct member count")
        if len(args.strict_grounded_runs) != len(args.grounded_runs):
            parser.error("strict grounded member count must equal raw grounded member count")
        strict_direct = load_probability_ensemble(
            args.strict_direct_runs, strict_ids, strict_labels
        )
        strict_grounded = load_probability_ensemble(
            args.strict_grounded_runs, strict_ids, strict_labels
        )
        strict_prediction_source = "independent_strict_inference"
    strict_result, strict_direct_pred, strict_grounded_pred, strict_and_pred = evaluate_policy(
        strict_labels, strict_direct, strict_grounded, args.tau, args.iterations, args.seed + 1
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_hashes = {
        "raw_direct_ensemble": write_predictions(
            args.output_dir / "raw_direct_ensemble_predictions.jsonl",
            raw_ids, raw_labels, direct_raw, raw_direct_pred,
        ),
        "raw_grounded_ensemble": write_predictions(
            args.output_dir / "raw_grounded_ensemble_predictions.jsonl",
            raw_ids, raw_labels, grounded_raw, raw_grounded_pred,
        ),
        "raw_and": write_predictions(
            args.output_dir / "raw_and_predictions.jsonl",
            raw_ids, raw_labels, direct_raw, raw_and_pred,
        ),
        "strict_direct_ensemble": write_predictions(
            args.output_dir / "strict_direct_ensemble_predictions.jsonl",
            strict_ids, strict_labels, strict_direct, strict_direct_pred,
        ),
        "strict_grounded_ensemble": write_predictions(
            args.output_dir / "strict_grounded_ensemble_predictions.jsonl",
            strict_ids, strict_labels, strict_grounded, strict_grounded_pred,
        ),
        "strict_and": write_predictions(
            args.output_dir / "strict_and_predictions.jsonl",
            strict_ids, strict_labels, strict_direct, strict_and_pred,
        ),
    }
    audit = {
        "protocol": "P1_B18B_canonical_AND_raw_and_strict",
        "parameters": {
            "top_k": args.top_k,
            "tau": args.tau,
            "parameter_source": "frozen validation policy; no test-time tuning",
        },
        "input_hashes": {
            "raw_manifest_sha256": sha256_file(args.raw_manifest),
            "strict_manifest_sha256": sha256_file(args.strict_manifest),
            "raw_retrieval_sha256": sha256_file(args.raw_retrieval),
            "strict_retrieval_sha256": sha256_file(args.strict_retrieval),
            "direct_prediction_sha256": {
                str(path): sha256_file(path) for path in args.direct_runs
            },
            "grounded_prediction_sha256": {
                str(path): sha256_file(path) for path in args.grounded_runs
            },
            "strict_direct_prediction_sha256": {
                str(path): sha256_file(path) for path in (args.strict_direct_runs or [])
            },
            "strict_grounded_prediction_sha256": {
                str(path): sha256_file(path) for path in (args.strict_grounded_runs or [])
            },
        },
        "alignment_audit": {
            "raw_samples": len(raw_ids),
            "strict_samples": len(strict_ids),
            "strict_is_proper_subset_of_raw": True,
            "raw_prediction_ids_match_raw_manifest": True,
            "strict_prediction_ids_match_strict_manifest": True,
            "raw_strict_shared_ranked_evidence_ids_match": shared_evidence_ids_match,
            "strict_prediction_source": strict_prediction_source,
            "direct_member_count": len(args.direct_runs),
            "grounded_member_count": len(args.grounded_runs),
        },
        "raw_official": raw_result,
        "strict": strict_result,
        "output_prediction_sha256": prediction_hashes,
        "protocol_flags": {
            "test_labels_used_for_parameter_selection": False,
            "test_labels_used_for_evaluation": True,
            "gold_evidence_used": False,
            "test_predictions_recomputed": (
                args.prediction_provenance == "canonical_raw_inference"
            ),
            "evaluation_from_frozen_saved_predictions": True,
            "prediction_provenance": args.prediction_provenance,
        },
    }
    json_path = args.output_dir / "canonical_router.json"
    json_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    def section(track: str, result: dict[str, Any]) -> list[str]:
        comparison = result["comparison_vs_direct"]
        bootstrap = comparison["bootstrap"]
        lines = [f"## {track}", "", "| Policy | Accuracy | Macro-F1 | F1 Supported | F1 Refuted | F1 NEI |",
                 "|---|---:|---:|---:|---:|---:|"]
        lines.extend(compact_row(name, result[name]) for name in ("direct", "grounded", "and"))
        lines.extend([
            "", "### AND vs. direct", "",
            f"- Macro-F1 delta: `{comparison['macro_f1_delta']:+.6f}`",
            f"- Accuracy delta: `{comparison['accuracy_delta']:+.6f}`",
            f"- Helpful/harmful: `{comparison['helpful']}/{comparison['harmful']}`",
            f"- Exact McNemar p: `{comparison['exact_mcnemar_p']:.6f}`",
            f"- Bootstrap 95% CI: `[{bootstrap['ci_95_percentile'][0]:+.6f}, "
            f"{bootstrap['ci_95_percentile'][1]:+.6f}]`",
            f"- Bootstrap P(delta > 0): `{bootstrap['probability_delta_positive']:.4f}`",
            f"- Route count/rate: `{result['routing']['route_count']}/"
            f"{result['routing']['route_rate']:.6f}`",
        ])
        return lines

    markdown = [
        "# Canonical B18B AND raw/strict test audit", "",
        "- Policy: **Asymmetric NEI Deferral (AND)**",
        f"- Frozen parameters: `K={args.top_k}`, `tau={args.tau:.2f}`",
        "- Parameter source: **frozen validation policy; no test-time tuning**",
        "- Evaluation uses frozen saved predictions; no parameter is tuned here.",
        f"- Prediction provenance: `{args.prediction_provenance}`.",
        "- Gold evidence used: **no**", "",
    ]
    markdown.extend(section("Raw official P1 (n=2,442)", raw_result))
    markdown.extend([""])
    markdown.extend(section("Strict duplicate-safe P1 (n=2,434)", strict_result))
    markdown.extend([
        "", "## Alignment and provenance audit", "",
        f"- Raw/strict manifest hashes: `{audit['input_hashes']['raw_manifest_sha256']}` / "
        f"`{audit['input_hashes']['strict_manifest_sha256']}`",
        f"- Raw/strict retrieval hashes: `{audit['input_hashes']['raw_retrieval_sha256']}` / "
        f"`{audit['input_hashes']['strict_retrieval_sha256']}`",
        "- Strict IDs are a proper subset of raw official IDs: **yes**",
        "- Every direct and grounded member matches all 2,442 raw IDs exactly: **yes**",
        f"- Shared raw/strict ranked evidence IDs match exactly: "
        f"**{'yes' if shared_evidence_ids_match else 'no'}**",
        f"- Strict prediction source: `{strict_prediction_source}`",
        "- Test labels used for policy selection: **no**",
        "- Materialized AND prediction hashes are stored in `canonical_router.json`.",
    ])
    markdown_path = args.output_dir / "canonical_router.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
