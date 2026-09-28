"""Frozen-validation robustness audit for B18B versus the direct ensemble."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scripts.analyze_mocheg_b18b_component_ablations import load_average
from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.run_mocheg_visual_retrieval import read_jsonl


LABEL_NAMES = {0: "supported", 1: "refuted", 2: "nei"}


def fixed_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    per_class = f1_score(
        labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(
            labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0
        )),
        "f1_supported": float(per_class[0]),
        "f1_refuted": float(per_class[1]),
        "f1_nei": float(per_class[2]),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1, 2]
        ).tolist(),
    }


def edges(values: list[float]) -> list[float]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    return np.unique(np.quantile(finite, [0.25, 0.5, 0.75])).tolist() if len(finite) else []


def quantile_name(value: float, boundaries: list[float]) -> str:
    if not np.isfinite(value):
        return "missing"
    return f"q{int(np.searchsorted(boundaries, value, side='right')) + 1}"


def rank_group(gold: set[str], retrieved: list[str]) -> str:
    if not gold:
        return "qrel_absent"
    ranks = [index + 1 for index, value in enumerate(retrieved) if value in gold]
    if not ranks:
        return "gold_not_retrieved"
    rank = min(ranks)
    if rank == 1:
        return "gold_rank_1"
    if rank <= 5:
        return "gold_rank_2_5"
    return "gold_rank_after_5"


def comparison(labels: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> dict:
    anchor_metrics = fixed_metrics(labels, anchor)
    candidate_metrics = fixed_metrics(labels, candidate)
    helpful = int(np.sum((anchor != labels) & (candidate == labels)))
    harmful = int(np.sum((anchor == labels) & (candidate != labels)))
    return {
        "samples": len(labels),
        "anchor_macro_f1": anchor_metrics["macro_f1"],
        "candidate_macro_f1": candidate_metrics["macro_f1"],
        "macro_f1_delta": candidate_metrics["macro_f1"] - anchor_metrics["macro_f1"],
        "accuracy_delta": candidate_metrics["accuracy"] - anchor_metrics["accuracy"],
        "helpful": helpful,
        "harmful": harmful,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--grounded-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.49)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    ids, labels, direct = load_average(args.direct_runs)
    grounded_ids, grounded_labels, grounded = load_average(args.grounded_runs)
    if ids != grounded_ids or not np.array_equal(labels, grounded_labels):
        raise ValueError("direct and grounded predictions are not aligned")
    manifests = {str(row["id"]): row for row in read_jsonl(args.manifest)}
    retrieval = {str(row["id"]): row for row in read_jsonl(args.retrieval)}
    missing = set(ids) - set(manifests) | (set(ids) - set(retrieval))
    if missing:
        raise ValueError(f"metadata missing for {len(missing)} prediction IDs")

    direct_pred = direct.argmax(axis=1)
    grounded_pred = grounded.argmax(axis=1)
    route_mask = (grounded_pred == 2) & (grounded[:, 2] >= args.tau)
    candidate_pred = direct_pred.copy()
    candidate_pred[route_mask] = 2
    change_mask = candidate_pred != direct_pred

    lengths = [len(manifests[cid].get("claim", "").split()) for cid in ids]
    confidences = [float(retrieval[cid].get("retrieval_confidence", np.nan)) for cid in ids]
    margins = []
    for cid in ids:
        scores = retrieval[cid].get("retrieved_scores", [])
        margins.append(float(scores[0]) - float(scores[1]) if len(scores) > 1 else np.nan)
    length_edges, confidence_edges, margin_edges = edges(lengths), edges(confidences), edges(margins)

    metadata: list[dict] = []
    for index, cid in enumerate(ids):
        claim, retrieved = manifests[cid], retrieval[cid]
        gold_ids = {str(value) for value in claim.get("text_evidence_ids", [])}
        retrieved_ids = [str(value) for value in retrieved.get("retrieved_evidence_ids", [])]
        top_ids = set(retrieved_ids[: args.top_k])
        source = str(claim.get("source", "unknown") or "unknown").strip().lower()
        metadata.append({
            "source": source,
            "gold_label": LABEL_NAMES[int(labels[index])],
            "qrel": "available" if gold_ids else "absent",
            "gold_coverage": "hit" if gold_ids and gold_ids & top_ids else
                             ("miss" if gold_ids else "qrel_absent"),
            "retrieval_status": rank_group(gold_ids, retrieved_ids),
            "claim_length": quantile_name(lengths[index], length_edges),
            "retrieval_confidence": quantile_name(confidences[index], confidence_edges),
            "retrieval_margin": quantile_name(margins[index], margin_edges),
        })

    fields = ("source", "gold_label", "qrel", "gold_coverage", "retrieval_status",
              "claim_length", "retrieval_confidence", "retrieval_margin")
    groups = {}
    for field in fields:
        groups[field] = {}
        for value in sorted({row[field] for row in metadata}):
            mask = np.asarray([row[field] == value for row in metadata])
            result = comparison(labels[mask], direct_pred[mask], candidate_pred[mask])
            result["activation_count"] = int(route_mask[mask].sum())
            result["activation_rate"] = float(route_mask[mask].mean())
            result["prediction_change_count"] = int(change_mask[mask].sum())
            result["prediction_change_rate"] = float(change_mask[mask].mean())
            groups[field][value] = result

    overall = comparison(labels, direct_pred, candidate_pred)
    overall["activation_count"] = int(route_mask.sum())
    overall["activation_rate"] = float(route_mask.mean())
    overall["prediction_change_count"] = int(change_mask.sum())
    overall["prediction_change_rate"] = float(change_mask.mean())
    overall["exact_mcnemar_p"] = exact_mcnemar_p(overall["helpful"], overall["harmful"])
    overall["bootstrap"] = bootstrap_delta(
        labels, direct_pred, candidate_pred,
        iterations=args.bootstrap_iterations, seed=args.seed,
    )
    transitions = Counter(
        (LABEL_NAMES[int(gold)], LABEL_NAMES[int(before)], LABEL_NAMES[int(after)])
        for gold, before, after in zip(labels, direct_pred, candidate_pred, strict=True)
        if before != after
    )
    payload = {
        "protocol": "B18B_frozen_validation_subgroup_audit_v1",
        "tau": args.tau,
        "top_k": args.top_k,
        "overall": overall,
        "quantile_edges": {"claim_length": length_edges, "retrieval_confidence": confidence_edges,
                           "retrieval_margin": margin_edges},
        "groups": groups,
        "prediction_transitions": [
            {"gold": key[0], "direct": key[1], "b18b": key[2], "count": count}
            for key, count in transitions.most_common()
        ],
        "official_validation_used_for_evaluation": True,
        "official_validation_used_for_policy_selection": False,
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# MOCHEG B18B frozen-validation subgroup audit", "",
        "Official validation used for evaluation: **yes**  ",
        "Official validation used for policy selection: **no**  ",
        "Test used: **no**", "",
        f"- Overall Macro-F1 delta: `{overall['macro_f1_delta']:+.6f}`",
        f"- Helpful/harmful: `{overall['helpful']}/{overall['harmful']}`",
        f"- Router activation count/rate: `{overall['activation_count']}/{overall['activation_rate']:.4f}`",
        f"- Actual prediction changes: `{overall['prediction_change_count']}/{overall['prediction_change_rate']:.4f}`",
        f"- Bootstrap P(delta > 0): `{overall['bootstrap']['probability_delta_positive']:.4f}`", "",
    ]
    for field in fields:
        lines += [f"## {field}", "",
                  "| Group | N | Direct F1 | B18B F1 | Delta | Active/Changed | Helpful/Harmful |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for value, row in groups[field].items():
            lines.append(
                f"| {value} | {row['samples']} | {row['anchor_macro_f1']:.6f} | "
                f"{row['candidate_macro_f1']:.6f} | {row['macro_f1_delta']:+.6f} | "
                f"{row['activation_count']}/{row['prediction_change_count']} | "
                f"{row['helpful']}/{row['harmful']} |"
            )
        lines.append("")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
