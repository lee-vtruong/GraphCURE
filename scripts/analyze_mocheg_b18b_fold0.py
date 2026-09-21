"""Paired fold-0 analysis for B18-B evidence-selection ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.summarize_mocheg_b18_seeds import compute_metrics, load_seed_predictions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_path(root: Path) -> Path:
    return root if root.is_file() else root / "val_predictions.jsonl"


def compare(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    reference_metrics = compute_metrics(labels, reference)
    candidate_metrics = compute_metrics(labels, candidate)
    helpful = int(np.sum((reference != labels) & (candidate == labels)))
    harmful = int(np.sum((reference == labels) & (candidate != labels)))
    return {
        "macro_f1_delta": candidate_metrics["macro_f1"] - reference_metrics["macro_f1"],
        "accuracy_delta": candidate_metrics["accuracy"] - reference_metrics["accuracy"],
        "class_f1_delta": {
            "supported": candidate_metrics["f1_supported"] - reference_metrics["f1_supported"],
            "refuted": candidate_metrics["f1_refuted"] - reference_metrics["f1_refuted"],
            "nei": candidate_metrics["f1_nei"] - reference_metrics["f1_nei"],
        },
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(
            labels, reference, candidate, iterations=iterations, seed=seed
        ),
    }


def source_diagnostics(
    ids: list[str],
    labels: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    manifest_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = {}
    for index, claim_id in enumerate(ids):
        source = str(manifest_rows.get(claim_id, {}).get("source", "unknown"))
        grouped.setdefault(source, []).append(index)
    result = {}
    for source, indices in sorted(grouped.items()):
        positions = np.asarray(indices, dtype=np.int64)
        anchor_metrics = compute_metrics(labels[positions], anchor[positions])
        candidate_metrics = compute_metrics(labels[positions], candidate[positions])
        result[source] = {
            "samples": len(indices),
            "anchor_macro_f1": anchor_metrics["macro_f1"],
            "distilled_macro_f1": candidate_metrics["macro_f1"],
            "macro_f1_delta": candidate_metrics["macro_f1"] - anchor_metrics["macro_f1"],
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--retrieval-top3", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--distilled", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--minimum-anchor-delta", type=float, default=0.003)
    parser.add_argument("--minimum-ablation-delta", type=float, default=0.0)
    parser.add_argument("--minimum-bootstrap-probability", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = {
        "anchor": args.anchor,
        "retrieval_top3": args.retrieval_top3,
        "generic": args.generic,
        "distilled": args.distilled,
    }
    runs = {name: load_seed_predictions(root) for name, root in roots.items()}
    common_ids = set.intersection(*(set(run) for run in runs.values()))
    ids = sorted(common_ids)
    if not ids:
        raise ValueError("no common prediction IDs")
    labels = np.asarray([int(runs["anchor"][claim_id]["label"]) for claim_id in ids])
    predictions = {
        name: np.asarray([int(run[claim_id]["prediction"]) for claim_id in ids])
        for name, run in runs.items()
    }
    for name, run in runs.items():
        other_labels = np.asarray([int(run[claim_id]["label"]) for claim_id in ids])
        if not np.array_equal(labels, other_labels):
            raise ValueError(f"label mismatch in run {name}")

    metrics = {name: compute_metrics(labels, pred) for name, pred in predictions.items()}
    comparisons = {
        "distilled_vs_anchor": compare(
            labels, predictions["anchor"], predictions["distilled"],
            iterations=args.iterations, seed=args.bootstrap_seed,
        ),
        "distilled_vs_retrieval_top3": compare(
            labels, predictions["retrieval_top3"], predictions["distilled"],
            iterations=args.iterations, seed=args.bootstrap_seed + 1,
        ),
        "distilled_vs_generic": compare(
            labels, predictions["generic"], predictions["distilled"],
            iterations=args.iterations, seed=args.bootstrap_seed + 2,
        ),
        "generic_vs_retrieval_top3": compare(
            labels, predictions["retrieval_top3"], predictions["generic"],
            iterations=args.iterations, seed=args.bootstrap_seed + 3,
        ),
    }
    primary = comparisons["distilled_vs_anchor"]
    manifest_rows = {str(row["id"]): row for row in read_jsonl(args.manifest)}
    sources = source_diagnostics(
        ids, labels, predictions["anchor"], predictions["distilled"], manifest_rows
    )
    gate = {
        "distilled_delta_vs_anchor_at_least_minimum": (
            primary["macro_f1_delta"] >= args.minimum_anchor_delta
        ),
        "distilled_noninferior_to_top3": (
            comparisons["distilled_vs_retrieval_top3"]["macro_f1_delta"]
            >= args.minimum_ablation_delta
        ),
        "distilled_noninferior_to_generic": (
            comparisons["distilled_vs_generic"]["macro_f1_delta"]
            >= args.minimum_ablation_delta
        ),
        "bootstrap_probability_at_least_minimum": (
            primary["bootstrap"]["probability_delta_positive"]
            >= args.minimum_bootstrap_probability
        ),
        "help_exceeds_harm": primary["helpful"] > primary["harmful"],
        "all_sources_nonnegative": all(
            row["macro_f1_delta"] >= 0.0 for row in sources.values()
        ),
    }
    gate["passed"] = all(gate.values())
    payload = {
        "phase": "B18-B",
        "protocol": "fold0_matched_compute_evidence_selection_ablation_v1",
        "samples": len(ids),
        "metrics": metrics,
        "comparisons": comparisons,
        "source_diagnostics": sources,
        "promotion_gate": gate,
        "audit": {
            "prediction_sha256": {
                name: sha256_file(prediction_path(root)) for name, root in roots.items()
            },
            "manifest_sha256": sha256_file(args.manifest),
            "official_validation_used": False,
            "test_split_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# MOCHEG B18-B fold-0 evidence-selection ablation",
        "",
        "Official validation used: **no**  ",
        "Test used: **no**",
        "",
        "| Run | Accuracy | Macro-F1 | Supported F1 | Refuted F1 | NEI F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        lines.append(
            f"| {name} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['f1_supported']:.4f} | {row['f1_refuted']:.4f} | {row['f1_nei']:.4f} |"
        )
    lines.extend([
        "",
        "## Primary comparison",
        "",
        f"- Distilled minus anchor Macro-F1: {primary['macro_f1_delta']:+.6f}",
        f"- Bootstrap P(delta > 0): {primary['bootstrap']['probability_delta_positive']:.4f}",
        f"- Helpful/harmful: {primary['helpful']}/{primary['harmful']}",
        f"- Promotion gate: **{'pass' if gate['passed'] else 'fail'}**",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
