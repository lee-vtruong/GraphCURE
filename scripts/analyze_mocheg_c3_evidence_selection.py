"""Evaluate the frozen C3 constraint-safe evidence-selection experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graphcure.open_web import load_jsonl, sha256_file
from scripts.analyze_mocheg_open_verdicts import comparison, metrics, read_anchor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--anchor-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--minimum-delta", type=float, default=.003)
    parser.add_argument("--minimum-bootstrap-probability", type=float, default=.95)
    parser.add_argument("--minimum-source-delta", type=float, default=-.002)
    parser.add_argument("--minimum-macro-f1", type=float, default=.61)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    if "test" in args.manifest.stem.casefold():
        parser.error("C3 development analysis must not use test")

    summary_path = args.score_root / "summary.json"
    score_path = args.score_root / "verdict_scores.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != "P2_open_web_C3_matched_evidence_selection_screen":
        parser.error("unexpected C3 scorer protocol")
    if not summary.get("complete") or summary.get("invalid_probability_rows"):
        parser.error("C3 scorer output is incomplete or invalid")
    if any(summary.get(key) for key in (
        "label_used", "gold_evidence_used", "test_split_used"
    )):
        parser.error("C3 scorer output contains forbidden supervision")

    manifests = load_jsonl(args.manifest)
    ids = [str(row["id"]) for row in manifests]
    labels = np.asarray([int(row["label"]) for row in manifests])
    if len(ids) != len(set(ids)):
        parser.error("duplicate validation manifest IDs")
    rows = load_jsonl(score_path)
    indexed = {(str(row["id"]), str(row["mode"])): row for row in rows}
    if len(indexed) != len(rows):
        parser.error("duplicate C3 score rows")
    modes = ("rank_control", "constraint_selector")
    expected = {(item, mode) for item in ids for mode in modes}
    if set(indexed) != expected:
        parser.error("C3 score IDs or modes do not match validation manifest")
    probability = {
        mode: np.asarray([
            indexed[(item, mode)]["probabilities"] for item in ids
        ], dtype=float)
        for mode in modes
    }
    anchor = read_anchor(args.anchor_predictions, ids, labels)
    all_metrics = {
        "closed_anchor": metrics(labels, anchor),
        **{mode: metrics(labels, values) for mode, values in probability.items()},
    }
    selector_vs_control = comparison(
        labels, probability["rank_control"], probability["constraint_selector"],
        args.bootstrap_iterations, args.bootstrap_seed,
    )
    selector_vs_anchor = comparison(
        labels, anchor, probability["constraint_selector"],
        args.bootstrap_iterations, args.bootstrap_seed + 1,
    )
    sources = np.asarray([
        str(row.get("source", "unknown") or "unknown") for row in manifests
    ])
    source_diagnostics = {}
    for source in sorted(set(sources.tolist())):
        selected = sources == source
        before = metrics(labels[selected], probability["rank_control"][selected])
        after = metrics(labels[selected], probability["constraint_selector"][selected])
        source_diagnostics[source] = {
            "samples": int(selected.sum()),
            "control_macro_f1": before["macro_f1"],
            "selector_macro_f1": after["macro_f1"],
            "macro_f1_delta": after["macro_f1"] - before["macro_f1"],
        }
    class_delta = {
        label: (
            all_metrics["constraint_selector"]["class_f1"][label]
            - all_metrics["rank_control"]["class_f1"][label]
        )
        for label in ("supported", "refuted", "nei")
    }
    gate = {
        "selector_delta_at_least_minimum": (
            selector_vs_control["macro_f1_delta"] >= args.minimum_delta
        ),
        "selector_macro_f1_at_least_minimum": (
            all_metrics["constraint_selector"]["macro_f1"] >= args.minimum_macro_f1
        ),
        "bootstrap_probability_at_least_minimum": (
            selector_vs_control["bootstrap"]["probability_delta_positive"]
            >= args.minimum_bootstrap_probability
        ),
        "help_exceeds_harm": (
            selector_vs_control["helpful"] > selector_vs_control["harmful"]
        ),
        "all_sources_within_noninferiority_margin": all(
            row["macro_f1_delta"] >= args.minimum_source_delta
            for row in source_diagnostics.values()
        ),
    }
    gate["passed"] = all(gate.values())
    result = {
        "protocol": "P2_open_web_C3_preregistered_evidence_selection_evaluation",
        "metrics": all_metrics,
        "selector_vs_rank_control": selector_vs_control,
        "selector_vs_closed_anchor": selector_vs_anchor,
        "class_f1_delta_vs_control": class_delta,
        "source_diagnostics": source_diagnostics,
        "promotion_gate": gate,
        "settings": {
            "minimum_delta": args.minimum_delta,
            "minimum_macro_f1": args.minimum_macro_f1,
            "minimum_bootstrap_probability": args.minimum_bootstrap_probability,
            "minimum_source_delta": args.minimum_source_delta,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "score_summary_sha256": sha256_file(summary_path),
            "scores_sha256": sha256_file(score_path),
            "anchor_predictions_sha256": sha256_file(args.anchor_predictions),
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
        for index, sample_id in enumerate(ids):
            handle.write(json.dumps({
                "id": sample_id, "gold": int(labels[index]),
                "rank_control_probabilities": probability[
                    "rank_control"
                ][index].tolist(),
                "constraint_selector_probabilities": probability[
                    "constraint_selector"
                ][index].tolist(),
            }) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
