"""Summarize five-seed frozen C3 evidence-selection confirmation."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from graphcure.open_web import load_jsonl, sha256_file
from scripts.analyze_mocheg_open_verdicts import comparison, metrics, read_anchor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42, 87, 100])
    parser.add_argument(
        "--anchor-template",
        default="outputs/mocheg_qwen3_lora_seed{seed}_v16/val_predictions.jsonl",
    )
    parser.add_argument("--minimum-mean-delta", type=float, default=.005)
    parser.add_argument("--minimum-ensemble-delta", type=float, default=.005)
    parser.add_argument("--minimum-positive-seeds", type=int, default=4)
    parser.add_argument("--minimum-bootstrap-probability", type=float, default=.95)
    parser.add_argument("--minimum-source-delta", type=float, default=-.002)
    args = parser.parse_args()
    if "test" in args.manifest.stem.casefold():
        parser.error("C3 confirmation must not consume test")

    manifests = load_jsonl(args.manifest)
    ids = [str(row["id"]) for row in manifests]
    labels = np.asarray([int(row["label"]) for row in manifests])
    sources = np.asarray([
        str(row.get("source", "unknown") or "unknown") for row in manifests
    ])
    if len(ids) != len(set(ids)):
        parser.error("duplicate IDs in validation manifest")

    per_seed = []
    control_probabilities = []
    selector_probabilities = []
    anchor_probabilities = []
    selection_hash = None
    for seed in args.seeds:
        directory = args.root / f"seed_{seed}"
        analysis_path = directory / "analysis.json"
        predictions_path = directory / "predictions.jsonl"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        if analysis.get("protocol") != (
            "P2_open_web_C3_preregistered_evidence_selection_evaluation"
        ):
            parser.error(f"seed {seed}: unexpected analysis protocol")
        if analysis.get("test_split_used"):
            parser.error(f"seed {seed}: test split was used")
        rows = {str(row["id"]): row for row in load_jsonl(predictions_path)}
        if set(rows) != set(ids):
            parser.error(f"seed {seed}: prediction IDs do not match manifest")
        observed = np.asarray([int(rows[item]["gold"]) for item in ids])
        if not np.array_equal(observed, labels):
            parser.error(f"seed {seed}: gold labels disagree")
        control = np.asarray([
            rows[item]["rank_control_probabilities"] for item in ids
        ], dtype=float)
        selector = np.asarray([
            rows[item]["constraint_selector_probabilities"] for item in ids
        ], dtype=float)
        anchor_path = Path(args.anchor_template.format(seed=seed))
        anchor = read_anchor(anchor_path, ids, labels)
        control_probabilities.append(control)
        selector_probabilities.append(selector)
        anchor_probabilities.append(anchor)
        # Scores differ across seeds, so use their scorer selection SHA from summary.
        scorer_summary = json.loads(
            (directory / "scores" / "summary.json").read_text(encoding="utf-8")
        )
        current_selection_hash = scorer_summary.get("selection_sha256")
        if selection_hash is None:
            selection_hash = current_selection_hash
        elif selection_hash != current_selection_hash:
            parser.error("C3 seed runs used different evidence selections")
        per_seed.append({
            "seed": seed,
            "rank_control": metrics(labels, control),
            "constraint_selector": metrics(labels, selector),
            "closed_anchor": metrics(labels, anchor),
            "selector_minus_control_macro_f1": (
                metrics(labels, selector)["macro_f1"]
                - metrics(labels, control)["macro_f1"]
            ),
            "selector_minus_anchor_macro_f1": (
                metrics(labels, selector)["macro_f1"]
                - metrics(labels, anchor)["macro_f1"]
            ),
            "analysis_sha256": sha256_file(analysis_path),
            "predictions_sha256": sha256_file(predictions_path),
            "anchor_predictions_sha256": sha256_file(anchor_path),
        })

    control_ensemble = np.mean(control_probabilities, axis=0)
    selector_ensemble = np.mean(selector_probabilities, axis=0)
    anchor_ensemble = np.mean(anchor_probabilities, axis=0)
    selector_vs_control = comparison(
        labels, control_ensemble, selector_ensemble, 5000, 2026
    )
    selector_vs_anchor = comparison(
        labels, anchor_ensemble, selector_ensemble, 5000, 2027
    )
    source_diagnostics = {}
    for source in sorted(set(sources.tolist())):
        selected = sources == source
        before = metrics(labels[selected], control_ensemble[selected])
        after = metrics(labels[selected], selector_ensemble[selected])
        source_diagnostics[source] = {
            "samples": int(selected.sum()),
            "control_macro_f1": before["macro_f1"],
            "selector_macro_f1": after["macro_f1"],
            "macro_f1_delta": after["macro_f1"] - before["macro_f1"],
        }
    control_deltas = [
        row["selector_minus_control_macro_f1"] for row in per_seed
    ]
    anchor_deltas = [
        row["selector_minus_anchor_macro_f1"] for row in per_seed
    ]
    paired = {
        "selector_vs_control": {
            "mean": statistics.fmean(control_deltas),
            "std": statistics.pstdev(control_deltas),
            "values": control_deltas,
            "positive_seeds": sum(value > 0 for value in control_deltas),
        },
        "selector_vs_closed_anchor": {
            "mean": statistics.fmean(anchor_deltas),
            "std": statistics.pstdev(anchor_deltas),
            "values": anchor_deltas,
            "positive_seeds": sum(value > 0 for value in anchor_deltas),
        },
    }
    ensemble_metrics = {
        "rank_control": metrics(labels, control_ensemble),
        "constraint_selector": metrics(labels, selector_ensemble),
        "closed_anchor": metrics(labels, anchor_ensemble),
    }
    gate = {
        "mean_selector_delta_at_least_minimum": (
            paired["selector_vs_control"]["mean"] >= args.minimum_mean_delta
        ),
        "positive_seeds_at_least_minimum": (
            paired["selector_vs_control"]["positive_seeds"]
            >= args.minimum_positive_seeds
        ),
        "ensemble_delta_at_least_minimum": (
            selector_vs_control["macro_f1_delta"] >= args.minimum_ensemble_delta
        ),
        "ensemble_bootstrap_probability_at_least_minimum": (
            selector_vs_control["bootstrap"]["probability_delta_positive"]
            >= args.minimum_bootstrap_probability
        ),
        "ensemble_help_exceeds_harm": (
            selector_vs_control["helpful"] > selector_vs_control["harmful"]
        ),
        "all_sources_within_noninferiority_margin": all(
            row["macro_f1_delta"] >= args.minimum_source_delta
            for row in source_diagnostics.values()
        ),
        "open_selector_ensemble_noninferior_to_closed_anchor": (
            selector_vs_anchor["macro_f1_delta"] >= 0
        ),
    }
    gate["passed"] = all(gate.values())
    result = {
        "protocol": "P2_open_web_C3_frozen_five_seed_confirmation",
        "per_seed": per_seed,
        "paired_deltas": paired,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_selector_vs_control": selector_vs_control,
        "ensemble_selector_vs_closed_anchor": selector_vs_anchor,
        "source_diagnostics_vs_control": source_diagnostics,
        "promotion_gate": gate,
        "selection_sha256": selection_hash,
        "validation_label_used_for_evaluation": True,
        "validation_label_used_for_training": False,
        "gold_evidence_used": False,
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MOCHEG Phase C3 frozen validation confirmation", "",
        "Test split used: **no**", "",
        "| Seed | Rank control F1 | Constraint selector F1 | Closed anchor F1 | Selector-Control |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        lines.append(
            f"| {row['seed']} | {row['rank_control']['macro_f1']:.4f} | "
            f"{row['constraint_selector']['macro_f1']:.4f} | "
            f"{row['closed_anchor']['macro_f1']:.4f} | "
            f"{row['selector_minus_control_macro_f1']:+.4f} |"
        )
    lines.extend([
        "", "## Aggregate", "",
        f"- Mean selector-control delta: {paired['selector_vs_control']['mean']:+.6f} +/- {paired['selector_vs_control']['std']:.6f}",
        f"- Selector ensemble Macro-F1: {ensemble_metrics['constraint_selector']['macro_f1']:.6f}",
        f"- Closed anchor ensemble Macro-F1: {ensemble_metrics['closed_anchor']['macro_f1']:.6f}",
        f"- Promotion gate: **{'pass' if gate['passed'] else 'fail'}**",
    ])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
