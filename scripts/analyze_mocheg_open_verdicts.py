"""Evaluate the preregistered C2c direct/constraint matched verifier screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from graphcure.open_web import load_jsonl, sha256_file
from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = probabilities.argmax(-1)
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(
            labels, prediction, labels=[0, 1, 2], average="macro",
            zero_division=0,
        )),
        "class_f1": {
            name: float(value) for name, value in zip(
                ("supported", "refuted", "nei"),
                f1_score(
                    labels, prediction, labels=[0, 1, 2], average=None,
                    zero_division=0,
                ),
                strict=True,
            )
        },
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=[0, 1, 2]
        ).tolist(),
    }


def comparison(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    before = baseline.argmax(-1)
    after = candidate.argmax(-1)
    before_correct = before == labels
    after_correct = after == labels
    helpful = int(np.sum(~before_correct & after_correct))
    harmful = int(np.sum(before_correct & ~after_correct))
    before_metrics = metrics(labels, baseline)
    after_metrics = metrics(labels, candidate)
    return {
        "macro_f1_delta": (
            after_metrics["macro_f1"] - before_metrics["macro_f1"]
        ),
        "accuracy_delta": (
            after_metrics["accuracy"] - before_metrics["accuracy"]
        ),
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(
            labels, before, after, iterations, seed
        ),
    }


def read_anchor(path: Path, ids: list[str], labels: np.ndarray) -> np.ndarray:
    rows = {str(row["id"]): row for row in load_jsonl(path)}
    if set(rows) != set(ids):
        raise ValueError("anchor prediction IDs do not match C2c IDs")
    observed = np.asarray([int(rows[item]["gold"]) for item in ids])
    if not np.array_equal(observed, labels):
        raise ValueError("anchor gold labels do not match validation manifest")
    return np.asarray([rows[item]["probabilities"] for item in ids], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verdict-root", type=Path, required=True)
    parser.add_argument("--anchor-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--minimum-delta", type=float, default=.003)
    parser.add_argument("--minimum-bootstrap-probability", type=float, default=.95)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    if "test" in args.manifest.stem.casefold():
        parser.error("C2c development analysis must not use the test split")

    summary_path = args.verdict_root / "summary.json"
    score_path = args.verdict_root / "verdict_scores.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("complete") or summary.get("invalid_probability_rows"):
        parser.error("C2c verdict scores are incomplete or invalid")
    if any(summary.get(key) for key in (
        "label_used", "gold_evidence_used", "test_split_used"
    )):
        parser.error("C2c scorer provenance is contaminated")

    manifest_rows = load_jsonl(args.manifest)
    ids = [str(row["id"]) for row in manifest_rows]
    if len(ids) != len(set(ids)):
        parser.error("duplicate IDs in validation manifest")
    labels = np.asarray([int(row["label"]) for row in manifest_rows])
    score_rows = load_jsonl(score_path)
    indexed = {
        (str(row["id"]), str(row["mode"])): row for row in score_rows
    }
    if len(indexed) != len(score_rows):
        parser.error("duplicate C2c score rows")
    expected = {(item, mode) for item in ids for mode in ("direct", "constraint")}
    if set(indexed) != expected:
        parser.error("C2c score IDs/modes do not exactly match the manifest")
    probabilities = {
        mode: np.asarray([
            indexed[(item, mode)]["probabilities"] for item in ids
        ], dtype=float)
        for mode in ("direct", "constraint")
    }
    probabilities["fixed_equal_ensemble"] = (
        probabilities["direct"] + probabilities["constraint"]
    ) / 2
    all_metrics = {
        name: metrics(labels, values) for name, values in probabilities.items()
    }
    constraint_vs_direct = comparison(
        labels, probabilities["direct"], probabilities["constraint"],
        args.bootstrap_iterations, args.bootstrap_seed,
    )
    ensemble_vs_direct = comparison(
        labels, probabilities["direct"], probabilities["fixed_equal_ensemble"],
        args.bootstrap_iterations, args.bootstrap_seed + 1,
    )

    source_diagnostics = {}
    sources = np.asarray([
        str(row.get("source") or row.get("metadata", {}).get("source") or "unknown")
        for row in manifest_rows
    ])
    for source in sorted(set(sources)):
        selected = sources == source
        if selected.sum() < 2:
            continue
        before = metrics(labels[selected], probabilities["direct"][selected])
        after = metrics(labels[selected], probabilities["constraint"][selected])
        source_diagnostics[source] = {
            "samples": int(selected.sum()),
            "direct_macro_f1": before["macro_f1"],
            "constraint_macro_f1": after["macro_f1"],
            "macro_f1_delta": after["macro_f1"] - before["macro_f1"],
        }

    gate = {
        "constraint_delta_at_least_minimum": (
            constraint_vs_direct["macro_f1_delta"] >= args.minimum_delta
        ),
        "bootstrap_probability_at_least_minimum": (
            constraint_vs_direct["bootstrap"]["probability_delta_positive"]
            >= args.minimum_bootstrap_probability
        ),
        "help_exceeds_harm": (
            constraint_vs_direct["helpful"] > constraint_vs_direct["harmful"]
        ),
        "all_sources_nonnegative": all(
            row["macro_f1_delta"] >= 0 for row in source_diagnostics.values()
        ),
    }
    anchor_result = None
    if args.anchor_predictions:
        anchor = read_anchor(args.anchor_predictions, ids, labels)
        anchor_result = {
            "metrics": metrics(labels, anchor),
            "constraint_vs_anchor": comparison(
                labels, anchor, probabilities["constraint"],
                args.bootstrap_iterations, args.bootstrap_seed + 2,
            ),
            "fixed_equal_ensemble_vs_anchor": comparison(
                labels, anchor, probabilities["fixed_equal_ensemble"],
                args.bootstrap_iterations, args.bootstrap_seed + 3,
            ),
            "predictions_sha256": sha256_file(args.anchor_predictions),
        }
        gate["constraint_noninferior_to_closed_anchor"] = (
            anchor_result["constraint_vs_anchor"]["macro_f1_delta"] >= 0
        )
    gate["passed"] = all(gate.values())

    result = {
        "protocol": "P2_open_web_C2c_preregistered_matched_verifier_screen",
        "metrics": all_metrics,
        "constraint_vs_direct": constraint_vs_direct,
        "fixed_equal_ensemble_vs_direct": ensemble_vs_direct,
        "closed_anchor": anchor_result,
        "source_diagnostics": source_diagnostics,
        "promotion_gate": gate,
        "settings": {
            "minimum_delta": args.minimum_delta,
            "minimum_bootstrap_probability": args.minimum_bootstrap_probability,
            "fixed_ensemble_weight": .5,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "verdict_summary_sha256": sha256_file(summary_path),
            "verdict_scores_sha256": sha256_file(score_path),
        },
        "validation_label_used_for_evaluation": True,
        "validation_label_used_for_scoring": False,
        "gold_evidence_used": False,
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.predictions_output.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(ids):
            handle.write(json.dumps({
                "id": item, "gold": int(labels[index]),
                "direct_probabilities": probabilities["direct"][index].tolist(),
                "constraint_probabilities": probabilities["constraint"][index].tolist(),
                "fixed_equal_ensemble_probabilities": probabilities[
                    "fixed_equal_ensemble"
                ][index].tolist(),
            }) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
