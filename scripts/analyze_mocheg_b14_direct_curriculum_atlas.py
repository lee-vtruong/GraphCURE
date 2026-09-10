"""Cross-fold failure atlas for the unstable B13 direct curriculum signal.

This is a post-failure diagnostic.  It combines the five disjoint held-out
partitions of the seed-2027 train-only fold assignment.  It never reads the
official validation or test split and it performs no model selection.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b12_failure_atlas import (
    LABEL_NAMES,
    auxiliary_metrics,
    build_atlas,
    classification_metrics,
    comparison,
    normalized_probabilities,
    subset_report,
)
from scripts.analyze_mocheg_b13_curriculum import validate_curriculum
from scripts.analyze_mocheg_b13_failure_atlas import (
    prediction_shift,
    transition_summary,
)
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_expert_complementarity import read_predictions
from scripts.prepare_mocheg_sv_folds import sha256
from scripts.run_mocheg_visual_retrieval import read_jsonl


FOLDS = (0, 1, 2, 3, 4)


def fold_diagnostics(runs: list[dict]) -> dict:
    """Summarize the direction and dispersion of paired fold effects."""
    rows = []
    for run in sorted(runs, key=lambda value: int(value["fold"])):
        labels = np.asarray(run["labels"])
        anchor = np.asarray(run["anchor"])
        candidate = np.asarray(run["candidate"])
        effect = comparison(labels, anchor, candidate)
        rows.append({
            "fold": int(run["fold"]),
            "samples": int(len(labels)),
            "anchor": classification_metrics(labels, anchor),
            "candidate": classification_metrics(labels, candidate),
            **effect,
        })
    deltas = np.asarray([row["macro_f1_delta"] for row in rows])
    return {
        "per_fold": rows,
        "macro_f1_delta": {
            "mean": float(deltas.mean()),
            "std": float(deltas.std()),
            "minimum": float(deltas.min()),
            "maximum": float(deltas.max()),
            "values": deltas.tolist(),
            "positive_folds": int(np.sum(deltas > 0)),
        },
        "stable_positive_in_all_folds": bool(np.all(deltas > 0)),
    }


def interaction_atlas(
    ids: list[str],
    cases: list[dict],
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    minimum_group_size: int,
) -> dict:
    """Measure interpretable two-way slice interactions without tuning."""
    by_id = {row["id"]: row for row in cases}
    if set(by_id) != set(ids):
        raise ValueError("interaction atlas case IDs are misaligned")
    fields = (
        ("source", "qrel"),
        ("source", "gold_label"),
        ("gold_label", "qrel"),
        ("gold_label", "retrieval_status"),
    )
    result = {}
    for left, right in fields:
        name = f"{left}_x_{right}"
        groups = {}
        values = sorted({
            (row["groups"][left], row["groups"][right]) for row in cases
        })
        for left_value, right_value in values:
            selected = np.asarray([
                by_id[sample_id]["groups"][left] == left_value
                and by_id[sample_id]["groups"][right] == right_value
                for sample_id in ids
            ])
            if int(selected.sum()) < minimum_group_size:
                continue
            report = subset_report(selected, labels, probabilities)
            groups[f"{left_value}|{right_value}"] = {
                "samples": report["samples"],
                "label_counts": report["label_counts"],
                "anchor": report["models"]["anchor"],
                "candidate": report["models"]["joint"],
                "candidate_vs_anchor": report["joint_vs_anchor"],
            }
        result[name] = groups
    return result


def markdown_summary(result: dict) -> str:
    overall = result["overall"]
    effect = result["candidate_vs_anchor"]
    folds = result["fold_diagnostics"]
    lines = [
        "# B14 direct-curriculum OOF failure atlas", "",
        "This is a post-failure diagnostic over five disjoint train-only folds.",
        "Official validation used: **no**  ", "Test used: **no**", "",
        "## Aggregate OOF result", "",
        "| Model | Accuracy | Macro-F1 |", "|---|---:|---:|",
        f"| anchor | {overall['anchor']['accuracy']:.4f} | "
        f"{overall['anchor']['macro_f1']:.4f} |",
        f"| direct curriculum | {overall['candidate']['accuracy']:.4f} | "
        f"{overall['candidate']['macro_f1']:.4f} |",
        "",
        f"- Macro-F1 delta: {effect['macro_f1_delta']:+.6f}",
        f"- Helpful / harmful: {effect['helpful']} / {effect['harmful']}",
        "", "## Fold stability", "",
        "| Fold | Anchor F1 | Candidate F1 | Delta | Help | Harm |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in folds["per_fold"]:
        lines.append(
            f"| {row['fold']} | {row['anchor']['macro_f1']:.4f} | "
            f"{row['candidate']['macro_f1']:.4f} | "
            f"{row['macro_f1_delta']:+.4f} | {row['helpful']} | "
            f"{row['harmful']} |"
        )
    lines.extend([
        "", "## Worst single-factor slices", "",
        "| Field | Value | N | MF1 delta | Help | Harm |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in result["worst_candidate_vs_anchor_groups"][:15]:
        lines.append(
            f"| {row['field']} | {row['value']} | {row['samples']} | "
            f"{row['macro_f1_delta']:+.4f} | {row['helpful']} | "
            f"{row['harmful']} |"
        )
    lines.extend([
        "", "## Decision", "",
        "- B13 hierarchical blend is closed and must not be retuned.",
        "- The direct-curriculum result is diagnostic-only; it is not a "
        "confirmatory improvement.",
        "- A B14 intervention must be preregistered on a new fold assignment "
        "after this atlas identifies a repeatable failure slice.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold0-root", type=Path, default=Path(
        "outputs/mocheg_b12_fresh/fold_0"))
    parser.add_argument("--fold0-candidate", type=Path, default=Path(
        "outputs/mocheg_b13_fresh/fold_0/curriculum"))
    parser.add_argument("--confirmation-root", type=Path, default=Path(
        "outputs/mocheg_b13_confirm"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--retrieval", type=Path, default=Path(
        "outputs/retrieval_mocheg_qwen3_reranked/train.jsonl"))
    parser.add_argument("--targets", type=Path, default=Path(
        "data/processed/mocheg_b6_targets_natural/train.jsonl"))
    parser.add_argument("--fold-spec", type=Path, default=Path(
        "data/processed/mocheg_b12_folds.json"))
    parser.add_argument("--minimum-group-size", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b14_direct_curriculum_atlas.json"))
    parser.add_argument("--markdown", type=Path, default=Path(
        "outputs/mocheg_b14_direct_curriculum_atlas.md"))
    parser.add_argument("--cases", type=Path, default=Path(
        "outputs/mocheg_b14_direct_curriculum_cases.jsonl"))
    args = parser.parse_args()

    fold_payload = json.loads(args.fold_spec.read_text(encoding="utf-8"))
    if (
        int(fold_payload.get("seed", -1)) != 2027
        or fold_payload.get("validation_split_used") is not False
        or fold_payload.get("test_split_used") is not False
        or fold_payload.get("manifest_sha256") != sha256(args.manifest)
    ):
        raise ValueError("expected locked seed-2027 train-only folds")
    fold_signature = sha256(args.fold_spec)
    fold_ids = {
        int(row["fold"]): set(row["val_ids"])
        for row in fold_payload["folds"]
    }
    if set(fold_ids) != set(FOLDS):
        raise ValueError("fold specification must contain exactly folds 0--4")

    seen: set[str] = set()
    runs, all_ids = [], []
    anchor_rows_all, candidate_rows_all = {}, {}
    for fold in FOLDS:
        if fold == 0:
            anchor_path = args.fold0_root / "anchor"
            candidate_path = args.fold0_candidate
        else:
            root = args.confirmation_root / f"fold_{fold}"
            anchor_path, candidate_path = root / "anchor", root / "curriculum"
        anchor_summary = validate_run(anchor_path, fold, f"fold_{fold}_anchor")
        candidate_summary = validate_run(
            candidate_path, fold, f"fold_{fold}_candidate"
        )
        validate_curriculum(candidate_summary, fold)
        for role, summary in (("anchor", anchor_summary),
                              ("candidate", candidate_summary)):
            observed = summary.get("provenance", {}).get("fold_spec_sha256")
            if observed != fold_signature:
                raise ValueError(f"fold {fold} {role}: signature mismatch")
        anchor_rows = read_predictions(anchor_path / "val_predictions.jsonl")
        candidate_rows = read_predictions(
            candidate_path / "val_predictions.jsonl"
        )
        expected = fold_ids[fold]
        if set(anchor_rows) != expected or set(candidate_rows) != expected:
            raise ValueError(f"fold {fold}: held prediction IDs mismatch")
        overlap = seen & expected
        if overlap:
            raise ValueError(f"fold {fold}: {len(overlap)} duplicate OOF IDs")
        seen.update(expected)
        ids = sorted(expected)
        labels = np.asarray([int(anchor_rows[value]["gold"]) for value in ids])
        candidate_labels = np.asarray([
            int(candidate_rows[value]["gold"]) for value in ids
        ])
        if not np.array_equal(labels, candidate_labels):
            raise ValueError(f"fold {fold}: gold labels are misaligned")
        anchor = normalized_probabilities(anchor_rows, ids)
        candidate = normalized_probabilities(candidate_rows, ids)
        runs.append({
            "fold": fold, "labels": labels,
            "anchor": anchor, "candidate": candidate,
        })
        all_ids.extend(ids)
        anchor_rows_all.update(anchor_rows)
        candidate_rows_all.update(candidate_rows)

    manifests = {row["id"]: row for row in read_jsonl(args.manifest)}
    if seen != set(manifests):
        raise ValueError(
            f"OOF union mismatch: predictions={len(seen)} manifest={len(manifests)}"
        )
    retrieval = {row["id"]: row for row in read_jsonl(args.retrieval)}
    targets = {row["id"]: row for row in read_jsonl(args.targets)}
    for name, rows in (("retrieval", retrieval), ("targets", targets)):
        missing = seen - set(rows)
        if missing:
            raise ValueError(f"{name}: missing {len(missing)} OOF IDs")
    if any(
        targets[value].get("train_gold_injected") is not False
        or targets[value].get("candidate_ids")
        != targets[value].get("natural_candidate_ids")
        for value in seen
    ):
        raise ValueError("OOF targets contain gold injection")

    ids = sorted(all_ids)
    labels = np.asarray([int(anchor_rows_all[value]["gold"]) for value in ids])
    anchor = normalized_probabilities(anchor_rows_all, ids)
    candidate = normalized_probabilities(candidate_rows_all, ids)
    generic_probabilities = {
        "anchor": anchor, "control": anchor, "joint": candidate,
    }
    generic, cases = build_atlas(
        ids, labels, generic_probabilities, manifests, retrieval, targets,
        candidate_rows_all, args.minimum_group_size,
    )
    result = {
        "protocol": "B14_post_failure_five_fold_train_only_OOF_atlas",
        "samples": len(ids),
        "folds": list(FOLDS),
        "overall": {
            "anchor": generic["overall"]["anchor"],
            "candidate": generic["overall"]["joint"],
        },
        "candidate_vs_anchor": generic["comparisons"]["joint_vs_anchor"],
        "fold_diagnostics": fold_diagnostics(runs),
        "prediction_shift": prediction_shift(labels, anchor, candidate),
        "transition_summary": transition_summary(cases),
        "auxiliary_heads": auxiliary_metrics(ids, targets, candidate_rows_all),
        "oracle_anchor_or_candidate": generic["oracle_anchor_or_joint"],
        "slices": generic["slices"],
        "interactions": interaction_atlas(
            ids, cases, labels, generic_probabilities, args.minimum_group_size
        ),
        "worst_candidate_vs_anchor_groups": (
            generic["worst_joint_vs_anchor_groups_minimum_size"]
        ),
        "decision": {
            "b13_hierarchical_blend_closed": True,
            "direct_curriculum_confirmed": False,
            "reason": (
                "B13 primary blend failed independent confirmation; direct "
                "curriculum is a post-hoc unstable signal requiring a new "
                "preregistered B14 hypothesis and fresh folds."
            ),
            "new_fold_assignment_required_before_b14_confirmation": True,
        },
        "fold_spec_sha256": fold_signature,
        "post_failure_exploratory_diagnostic": True,
        "official_validation_used": False,
        "test_split_used": False,
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "retrieval_sha256": sha256(args.retrieval),
            "targets_sha256": sha256(args.targets),
        },
    }

    cleaned_cases = []
    for row in cases:
        row = dict(row)
        row["predictions"] = {
            "anchor": row["predictions"]["anchor"],
            "candidate": row["predictions"]["joint"],
        }
        row["confidence"] = {
            "anchor": row["confidence"]["anchor"],
            "candidate": row["confidence"]["joint"],
        }
        row["entropy"] = {
            "anchor": row["entropy"]["anchor"],
            "candidate": row["entropy"]["joint"],
        }
        row["candidate_vs_anchor"] = row.pop("joint_vs_anchor")
        row.pop("joint_vs_control", None)
        cleaned_cases.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_summary(result), encoding="utf-8")
    args.cases.parent.mkdir(parents=True, exist_ok=True)
    args.cases.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False)
                  for row in cleaned_cases) + "\n",
        encoding="utf-8",
    )
    print(markdown_summary(result))


if __name__ == "__main__":
    main()
