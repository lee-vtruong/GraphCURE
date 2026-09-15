"""Diagnose C2c open-web help/harm before preregistering Phase C3."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from graphcure.open_web import load_jsonl, sha256_file
from scripts.analyze_mocheg_open_verdicts import comparison, metrics


LABEL_NAMES = {0: "supported", 1: "refuted", 2: "nei"}


def normalized_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(-1) / np.log(probability.shape[1])


def quartile_groups(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=object)
    for quartile, indices in enumerate(np.array_split(order, 4), 1):
        result[indices] = f"q{quartile}"
    return result


def group_report(
    labels: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
) -> dict:
    result = {}
    for value in sorted(set(groups.tolist())):
        selected = groups == value
        if not selected.any():
            continue
        anchor_metrics = metrics(labels[selected], anchor[selected])
        candidate_metrics = metrics(labels[selected], candidate[selected])
        before = anchor[selected].argmax(-1) == labels[selected]
        after = candidate[selected].argmax(-1) == labels[selected]
        result[str(value)] = {
            "samples": int(selected.sum()),
            "anchor_macro_f1": anchor_metrics["macro_f1"],
            "candidate_macro_f1": candidate_metrics["macro_f1"],
            "macro_f1_delta": (
                candidate_metrics["macro_f1"] - anchor_metrics["macro_f1"]
            ),
            "helpful": int(np.sum(~before & after)),
            "harmful": int(np.sum(before & ~after)),
        }
    return result


def claim_features(
    ids: list[str], shortlist_path: Path, constraint_path: Path
) -> dict[str, np.ndarray]:
    shortlist = {str(row["id"]): row for row in load_jsonl(shortlist_path)}
    constraints: dict[str, list[dict]] = {item: [] for item in ids}
    for row in load_jsonl(constraint_path):
        sample_id = str(row["id"])
        if sample_id in constraints:
            constraints[sample_id].append(row)
    result: dict[str, list[float]] = {
        "evidence_count": [], "domain_count": [], "social_share": [],
        "stance_support_max": [], "stance_refute_max": [],
        "stance_neither_mean": [], "sufficiency_sufficient_max": [],
        "sufficiency_irrelevant_mean": [], "entity_conflict_mean": [],
        "temporal_conflict_mean": [], "stance_conflict": [],
    }
    for sample_id in ids:
        evidence = shortlist[sample_id].get("evidence_shortlist", [])
        result["evidence_count"].append(float(len(evidence)))
        result["domain_count"].append(float(len({
            row.get("domain") for row in evidence if row.get("domain")
        })))
        result["social_share"].append(
            sum(row.get("source_family") == "social" for row in evidence)
            / max(1, len(evidence))
        )
        by_task: dict[str, list[dict]] = {}
        for row in constraints[sample_id]:
            by_task.setdefault(str(row["task"]), []).append(row["probabilities"])

        def values(task: str, code: str) -> np.ndarray:
            return np.asarray([
                float(row[code]) for row in by_task.get(task, [])
            ], dtype=float)

        support = values("stance", "A")
        refute = values("stance", "B")
        neither = values("stance", "C")
        sufficient = values("sufficiency", "A")
        irrelevant = values("sufficiency", "C")
        entity_conflict = values("entity", "B")
        temporal_conflict = values("temporal", "B")
        arrays = (
            support, refute, neither, sufficient, irrelevant,
            entity_conflict, temporal_conflict,
        )
        if any(len(value) != len(evidence) for value in arrays):
            raise ValueError(f"constraint/evidence mismatch for {sample_id}")
        result["stance_support_max"].append(float(support.max()))
        result["stance_refute_max"].append(float(refute.max()))
        result["stance_neither_mean"].append(float(neither.mean()))
        result["sufficiency_sufficient_max"].append(float(sufficient.max()))
        result["sufficiency_irrelevant_mean"].append(float(irrelevant.mean()))
        result["entity_conflict_mean"].append(float(entity_conflict.mean()))
        result["temporal_conflict_mean"].append(float(temporal_conflict.mean()))
        result["stance_conflict"].append(float(min(
            support.max(), refute.max()
        )))
    return {key: np.asarray(value) for key, value in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-predictions", type=Path, required=True)
    parser.add_argument("--c2c-predictions", type=Path, required=True)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--constraint-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()
    if "test" in args.manifest.stem.casefold():
        parser.error("failure atlas must not consume test")

    manifest = {str(row["id"]): row for row in load_jsonl(args.manifest)}
    anchor_rows = {
        str(row["id"]): row for row in load_jsonl(args.anchor_predictions)
    }
    c2c_rows = {str(row["id"]): row for row in load_jsonl(args.c2c_predictions)}
    if not manifest or set(manifest) != set(anchor_rows) or set(manifest) != set(c2c_rows):
        parser.error("manifest, anchor, and C2c prediction IDs must match")
    ids = sorted(manifest)
    labels = np.asarray([int(manifest[item]["label"]) for item in ids])
    observed = np.asarray([int(anchor_rows[item]["gold"]) for item in ids])
    if not np.array_equal(labels, observed):
        parser.error("anchor labels disagree with manifest")
    anchor = np.asarray([
        anchor_rows[item]["probabilities"] for item in ids
    ], dtype=float)
    candidates = {
        "direct": np.asarray([
            c2c_rows[item]["direct_probabilities"] for item in ids
        ], dtype=float),
        "constraint": np.asarray([
            c2c_rows[item]["constraint_probabilities"] for item in ids
        ], dtype=float),
        "fixed_equal_ensemble": np.asarray([
            c2c_rows[item]["fixed_equal_ensemble_probabilities"] for item in ids
        ], dtype=float),
    }
    feature = claim_features(ids, args.shortlist, args.constraint_scores)
    feature.update({
        "anchor_confidence": anchor.max(-1),
        "anchor_entropy": normalized_entropy(anchor),
        "open_confidence": candidates["fixed_equal_ensemble"].max(-1),
        "open_entropy": normalized_entropy(candidates["fixed_equal_ensemble"]),
        "open_confidence_advantage": (
            candidates["fixed_equal_ensemble"].max(-1) - anchor.max(-1)
        ),
    })
    groups = {
        "source": np.asarray([
            str(manifest[item].get("source", "unknown") or "unknown") for item in ids
        ]),
        "gold_label": np.asarray([LABEL_NAMES[int(value)] for value in labels]),
        "anchor_prediction": np.asarray([
            LABEL_NAMES[int(value)] for value in anchor.argmax(-1)
        ]),
        "anchor_open_agreement": np.asarray([
            "agree" if left == right else "disagree"
            for left, right in zip(
                anchor.argmax(-1),
                candidates["fixed_equal_ensemble"].argmax(-1),
                strict=True,
            )
        ]),
    }
    groups.update({key: quartile_groups(value) for key, value in feature.items()})

    candidate_results = {}
    for name, probability in candidates.items():
        before = anchor.argmax(-1)
        after = probability.argmax(-1)
        helpful_mask = (before != labels) & (after == labels)
        harmful_mask = (before == labels) & (after != labels)
        oracle = anchor.copy()
        oracle[helpful_mask] = probability[helpful_mask]
        transitions = Counter(
            (
                LABEL_NAMES[int(gold)], LABEL_NAMES[int(old)],
                LABEL_NAMES[int(new)],
                "helpful" if help else "harmful" if harm else "other",
            )
            for gold, old, new, help, harm in zip(
                labels, before, after, helpful_mask, harmful_mask, strict=True
            )
            if old != new
        )
        examples = []
        for index in np.flatnonzero(harmful_mask)[:args.examples]:
            examples.append({
                "id": ids[index],
                "source": str(manifest[ids[index]].get("source", "unknown")),
                "gold": LABEL_NAMES[int(labels[index])],
                "anchor": LABEL_NAMES[int(before[index])],
                "candidate": LABEL_NAMES[int(after[index])],
                "claim": manifest[ids[index]].get("claim", ""),
                "features": {key: float(value[index]) for key, value in feature.items()},
            })
        candidate_results[name] = {
            "metrics": metrics(labels, probability),
            "comparison_vs_anchor": comparison(
                labels, anchor, probability, 5000, 2026
            ),
            "oracle_anchor_or_candidate": metrics(labels, oracle),
            "transition_counts": [
                {
                    "gold": key[0], "anchor": key[1], "candidate": key[2],
                    "effect": key[3], "count": count,
                }
                for key, count in transitions.most_common()
            ],
            "group_diagnostics": {
                key: group_report(labels, anchor, probability, values)
                for key, values in groups.items()
            },
            "harmful_examples": examples,
        }

    primary = candidate_results["fixed_equal_ensemble"]
    result = {
        "protocol": "P2_open_web_C2c_posthoc_failure_atlas",
        "anchor": metrics(labels, anchor),
        "candidates": candidate_results,
        "primary_diagnosis": {
            "open_has_complementary_signal": (
                primary["oracle_anchor_or_candidate"]["macro_f1"]
                > metrics(labels, anchor)["macro_f1"] + .02
            ),
            "open_is_not_safe_as_replacement": (
                primary["comparison_vs_anchor"]["macro_f1_delta"] < 0
            ),
            "next_action": (
                "Use the atlas to preregister evidence selection or robustness "
                "training; do not tune a validation router."
            ),
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "anchor_predictions_sha256": sha256_file(args.anchor_predictions),
            "c2c_predictions_sha256": sha256_file(args.c2c_predictions),
            "shortlist_sha256": sha256_file(args.shortlist),
            "constraint_scores_sha256": sha256_file(args.constraint_scores),
        },
        "validation_label_used_for_diagnosis": True,
        "validation_label_used_for_training": False,
        "exploratory_only": True,
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
