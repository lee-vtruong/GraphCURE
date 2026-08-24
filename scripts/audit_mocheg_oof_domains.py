"""Audit train-only OOF verifier errors across retrieval/domain strata."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scripts.analyze_mocheg_sv_complementarity import read_predictions
from scripts.run_mocheg_visual_retrieval import read_jsonl


def bin_edges(values: list[float]) -> list[float]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not len(finite):
        return []
    return np.unique(np.quantile(finite, [0.25, 0.5, 0.75])).tolist()


def bin_name(value: float, edges: list[float]) -> str:
    if not np.isfinite(value):
        return "missing"
    return f"q{int(np.searchsorted(edges, value, side='right')) + 1}"


def group_metrics(rows: list[dict]) -> dict:
    y = np.asarray([row["gold"] for row in rows])
    prediction = np.asarray([row["prediction"] for row in rows])
    return {
        "samples": len(rows),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(
            y, prediction, labels=[0, 1, 2], average="macro", zero_division=0
        )),
        "label_counts": dict(Counter(map(str, y.tolist()))),
        "confusion_matrix": confusion_matrix(
            y, prediction, labels=[0, 1, 2]
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/processed/mocheg_manifest_strict/train.jsonl"),
    )
    parser.add_argument(
        "--retrieval", type=Path,
        default=Path("outputs/retrieval_mocheg_qwen3_reranked/train.jsonl"),
    )
    parser.add_argument(
        "--fold-spec", type=Path,
        default=Path("data/processed/mocheg_sv_folds.json"),
    )
    parser.add_argument(
        "--fold0-predictions", type=Path,
        default=Path("outputs/mocheg_sv_screen_flat_fold0/val_predictions.jsonl"),
    )
    parser.add_argument(
        "--confirmation-template",
        default="outputs/mocheg_sv_confirm/fold_{fold}_flat/val_predictions.jsonl",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/mocheg_flat_oof_domain_audit.json"),
    )
    args = parser.parse_args()

    claims = {row["id"]: row for row in read_jsonl(args.manifest)}
    retrieval = {row["id"]: row for row in read_jsonl(args.retrieval)}
    fold_spec = json.loads(args.fold_spec.read_text(encoding="utf-8"))
    predictions = {}
    prediction_fold = {}
    expected_oof_ids = set()
    fold_by_number = {int(row["fold"]): row for row in fold_spec["folds"]}
    for fold in args.folds:
        path = args.fold0_predictions if fold == 0 else Path(
            args.confirmation_template.format(fold=fold)
        )
        fold_rows = read_predictions(path)
        expected = set(fold_by_number[fold]["val_ids"])
        expected_oof_ids.update(expected)
        if set(fold_rows) != expected:
            raise ValueError(
                f"fold {fold} prediction IDs differ from held-out fold IDs"
            )
        overlap = set(predictions) & set(fold_rows)
        if overlap:
            raise ValueError(f"duplicate OOF predictions: {len(overlap)}")
        predictions.update(fold_rows)
        prediction_fold.update({value: fold for value in fold_rows})
    if set(predictions) != expected_oof_ids:
        raise ValueError("OOF predictions do not cover the requested held-out folds")

    claim_lengths = [len(row.get("claim", "").split()) for row in claims.values()]
    confidences = [
        float(retrieval[value].get("retrieval_confidence", np.nan))
        for value in claims
    ]
    margins = []
    for value in claims:
        scores = retrieval[value].get("retrieved_scores", [])
        margins.append(
            float(scores[0]) - float(scores[1]) if len(scores) > 1 else np.nan
        )
    length_edges = bin_edges(claim_lengths)
    confidence_edges = bin_edges(confidences)
    margin_edges = bin_edges(margins)

    audit_rows = []
    for index, sample_id in enumerate(predictions):
        claim = claims[sample_id]
        retrieved = retrieval[sample_id]
        candidate_ids = {
            str(value) for value in
            retrieved.get("retrieved_evidence_ids", [])[:args.top_k]
        }
        gold_ids = {str(value) for value in claim.get("text_evidence_ids", [])}
        scores = retrieved.get("retrieved_scores", [])
        margin = (
            float(scores[0]) - float(scores[1]) if len(scores) > 1 else np.nan
        )
        prediction = predictions[sample_id]
        audit_rows.append({
            "id": sample_id, "fold": prediction_fold[sample_id],
            "gold": int(prediction["gold"]),
            "prediction": int(prediction["prediction"]),
            "source": claim.get("source", "unknown") or "unknown",
            "qrel": "available" if gold_ids else "absent",
            "gold_coverage": "hit" if candidate_ids & gold_ids else "miss",
            "claim_length": bin_name(len(claim.get("claim", "").split()), length_edges),
            "retrieval_confidence": bin_name(
                float(retrieved.get("retrieval_confidence", np.nan)), confidence_edges
            ),
            "retrieval_margin": bin_name(margin, margin_edges),
        })

    strata = {}
    for field in (
        "fold", "source", "qrel", "gold_coverage", "claim_length",
        "retrieval_confidence", "retrieval_margin",
    ):
        strata[field] = {}
        for value in sorted({str(row[field]) for row in audit_rows}):
            subset = [row for row in audit_rows if str(row[field]) == value]
            strata[field][value] = group_metrics(subset)
    worst = []
    for field, values in strata.items():
        for value, metrics in values.items():
            if metrics["samples"] >= 100:
                worst.append({"field": field, "value": value, **metrics})
    worst.sort(key=lambda row: row["macro_f1"])
    payload = {
        "protocol": "strict_train_only_out_of_fold_domain_audit",
        "folds": args.folds,
        "samples": len(audit_rows),
        "overall": group_metrics(audit_rows),
        "quantile_edges": {
            "claim_length": length_edges,
            "retrieval_confidence": confidence_edges,
            "retrieval_margin": margin_edges,
        },
        "strata": strata,
        "worst_groups_minimum_100_samples": worst[:12],
        "test_split_used": False,
        "validation_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "saved": str(args.output), "samples": len(audit_rows),
        "overall": payload["overall"],
        "worst_groups": payload["worst_groups_minimum_100_samples"],
        "test_split_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
