"""Leakage-safe failure atlas for the failed B13 curriculum candidate."""
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
    normalized_probabilities,
)
from scripts.analyze_mocheg_b13_curriculum import validate_curriculum
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_expert_complementarity import read_predictions
from scripts.prepare_mocheg_sv_folds import sha256
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_qwen3_hierarchical_lora import hierarchical_probabilities


def interpolation_diagnostic(
    labels: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
) -> dict:
    rows = []
    for index in range(101):
        weight = index / 100
        probabilities = (1 - weight) * anchor + weight * candidate
        metrics = classification_metrics(labels, probabilities)
        rows.append({
            "candidate_weight": weight,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        })
    best = max(rows, key=lambda row: (row["macro_f1"], row["accuracy"],
                                      -row["candidate_weight"]))
    anchor_f1 = rows[0]["macro_f1"]
    return {
        "best": best,
        "macro_f1_delta_vs_anchor": best["macro_f1"] - anchor_f1,
        "candidate_signal_used": best["candidate_weight"] > 0,
        "exploratory_only": True,
        "weights_evaluated": len(rows),
    }


def head_verdict_diagnostic(
    labels: np.ndarray,
    anchor: np.ndarray,
    direct: np.ndarray,
    hierarchical: np.ndarray,
) -> dict:
    direct_prediction = direct.argmax(axis=1)
    hierarchical_prediction = hierarchical.argmax(axis=1)
    direct_correct = direct_prediction == labels
    hierarchical_correct = hierarchical_prediction == labels
    return {
        "direct": classification_metrics(labels, direct),
        "hierarchical": classification_metrics(labels, hierarchical),
        "direct_hierarchical_agreement": float(np.mean(
            direct_prediction == hierarchical_prediction
        )),
        "hierarchical_only_correct": int(np.sum(
            ~direct_correct & hierarchical_correct
        )),
        "direct_only_correct": int(np.sum(
            direct_correct & ~hierarchical_correct
        )),
        "anchor_hierarchical_interpolation": interpolation_diagnostic(
            labels, anchor, hierarchical
        ),
        "direct_hierarchical_interpolation": interpolation_diagnostic(
            labels, direct, hierarchical
        ),
    }


def transition_summary(cases: list[dict]) -> dict:
    transitions = Counter()
    effects = Counter()
    for row in cases:
        anchor = row["predictions"]["anchor"]
        candidate = row["predictions"]["joint"]
        if anchor == candidate:
            continue
        effect = row["joint_vs_anchor"]
        transitions[(row["gold"], anchor, candidate, effect)] += 1
        effects[effect] += 1
    return {
        "effect_counts": dict(effects),
        "transitions": [{
            "gold": gold,
            "anchor": anchor,
            "candidate": candidate,
            "effect": effect,
            "count": count,
        } for (gold, anchor, candidate, effect), count
            in transitions.most_common()],
    }


def prediction_shift(labels: np.ndarray, anchor: np.ndarray,
                     candidate: np.ndarray) -> dict:
    anchor_prediction = anchor.argmax(axis=1)
    candidate_prediction = candidate.argmax(axis=1)
    return {
        "anchor_prediction_counts": dict(Counter(
            LABEL_NAMES[int(value)] for value in anchor_prediction.tolist()
        )),
        "candidate_prediction_counts": dict(Counter(
            LABEL_NAMES[int(value)] for value in candidate_prediction.tolist()
        )),
        "candidate_minus_anchor": {
            LABEL_NAMES[index]: int(
                np.sum(candidate_prediction == index)
                - np.sum(anchor_prediction == index)
            ) for index in range(3)
        },
        "correct_supported_lost_to_nei": int(np.sum(
            (labels == 0) & (anchor_prediction == 0)
            & (candidate_prediction == 2)
        )),
        "correct_refuted_lost_to_nei": int(np.sum(
            (labels == 1) & (anchor_prediction == 1)
            & (candidate_prediction == 2)
        )),
    }


def markdown_summary(result: dict) -> str:
    overall = result["overall"]
    interpolation = result["probability_interpolation"]
    head_verdict = result["head_verdict_diagnostic"]
    shift = result["prediction_shift"]
    lines = [
        "# B13 curriculum failure atlas (fresh train-only fold 0)", "",
        "Official validation used: **no**  ", "Test used: **no**", "",
        "## Overall", "",
        "| Model | Accuracy | Macro-F1 |", "|---|---:|---:|",
        f"| anchor | {overall['anchor']['accuracy']:.4f} | "
        f"{overall['anchor']['macro_f1']:.4f} |",
        f"| curriculum | {overall['candidate']['accuracy']:.4f} | "
        f"{overall['candidate']['macro_f1']:.4f} |",
        "", "## Prediction shift", "",
        f"- Candidate minus anchor: `{shift['candidate_minus_anchor']}`",
        "- Correct supported/refuted lost to NEI: "
        f"`{shift['correct_supported_lost_to_nei']}` / "
        f"`{shift['correct_refuted_lost_to_nei']}`",
        "", "## Exploratory probability interpolation", "",
        f"- Best candidate weight: {interpolation['best']['candidate_weight']:.2f}",
        f"- Best Macro-F1: {interpolation['best']['macro_f1']:.6f}",
        f"- Delta over anchor: "
        f"{interpolation['macro_f1_delta_vs_anchor']:+.6f}",
        "", "## Head-derived hierarchical verdict", "",
        f"- Hierarchical Macro-F1: "
        f"{head_verdict['hierarchical']['macro_f1']:.6f}",
        f"- Direct/hierarchical agreement: "
        f"{head_verdict['direct_hierarchical_agreement']:.4f}",
        f"- Hierarchical-only/direct-only correct: "
        f"{head_verdict['hierarchical_only_correct']} / "
        f"{head_verdict['direct_only_correct']}",
        f"- Best anchor + hierarchical Macro-F1: "
        f"{head_verdict['anchor_hierarchical_interpolation']['best']['macro_f1']:.6f}",
        f"- Delta over anchor: "
        f"{head_verdict['anchor_hierarchical_interpolation']['macro_f1_delta_vs_anchor']:+.6f}",
        "", "## Auxiliary heads", "",
        "| Head | Accuracy | Macro-F1 |", "|---|---:|---:|",
    ]
    for name, row in result["auxiliary_heads"].items():
        lines.append(
            f"| {name} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |"
        )
    lines.extend([
        "", "## Worst slices", "",
        "| Field | Value | N | MF1 delta | Help | Harm |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in result["worst_candidate_vs_anchor_groups"][:12]:
        lines.append(
            f"| {row['field']} | {row['value']} | {row['samples']} | "
            f"{row['macro_f1_delta']:+.4f} | {row['helpful']} | "
            f"{row['harmful']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, default=Path(
        "outputs/mocheg_b12_fresh/fold_0/anchor"))
    parser.add_argument("--candidate", type=Path, default=Path(
        "outputs/mocheg_b13_fresh/fold_0/curriculum"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--retrieval", type=Path, default=Path(
        "outputs/retrieval_mocheg_qwen3_reranked/train.jsonl"))
    parser.add_argument("--targets", type=Path, default=Path(
        "data/processed/mocheg_b6_targets_natural/train.jsonl"))
    parser.add_argument("--fold-spec", type=Path, default=Path(
        "data/processed/mocheg_b12_folds.json"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--minimum-group-size", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b13_failure_atlas.json"))
    parser.add_argument("--markdown", type=Path, default=Path(
        "outputs/mocheg_b13_failure_atlas.md"))
    parser.add_argument("--cases", type=Path, default=Path(
        "outputs/mocheg_b13_failure_cases.jsonl"))
    args = parser.parse_args()
    if args.fold != 0:
        raise ValueError("B13 diagnosis is restricted to fresh fold 0")
    fold_payload = json.loads(args.fold_spec.read_text(encoding="utf-8"))
    if (
        int(fold_payload.get("seed", -1)) != 2027
        or fold_payload.get("validation_split_used") is not False
        or fold_payload.get("test_split_used") is not False
        or fold_payload.get("manifest_sha256") != sha256(args.manifest)
    ):
        raise ValueError("expected locked B12/B13 seed-2027 train-only folds")
    fold_signature = sha256(args.fold_spec)
    anchor_summary = validate_run(args.anchor, args.fold, "anchor")
    candidate_summary = validate_run(args.candidate, args.fold, "candidate")
    curriculum = validate_curriculum(candidate_summary, args.fold)
    for name, summary in (("anchor", anchor_summary),
                          ("candidate", candidate_summary)):
        if summary.get("provenance", {}).get("fold_spec_sha256") != fold_signature:
            raise ValueError(f"{name}: fold signature mismatch")

    prediction_rows = {
        "anchor": read_predictions(args.anchor / "val_predictions.jsonl"),
        "candidate": read_predictions(args.candidate / "val_predictions.jsonl"),
    }
    expected_ids = set(next(
        row["val_ids"] for row in fold_payload["folds"]
        if int(row["fold"]) == args.fold
    ))
    if any(set(rows) != expected_ids for rows in prediction_rows.values()):
        raise ValueError("prediction IDs do not equal the locked held fold")
    ids = sorted(expected_ids)
    labels = np.asarray([
        int(prediction_rows["anchor"][value]["gold"]) for value in ids
    ])
    anchor = normalized_probabilities(prediction_rows["anchor"], ids)
    candidate = normalized_probabilities(prediction_rows["candidate"], ids)
    sufficiency = normalized_probabilities(
        prediction_rows["candidate"], ids, "sufficiency_probabilities", 2
    )
    polarity = normalized_probabilities(
        prediction_rows["candidate"], ids, "polarity_probabilities", 2
    )
    hierarchical = hierarchical_probabilities(sufficiency, polarity)
    observed = np.asarray([
        int(prediction_rows["candidate"][value]["gold"]) for value in ids
    ])
    if not np.array_equal(labels, observed):
        raise ValueError("candidate gold labels are not aligned")

    manifests = {row["id"]: row for row in read_jsonl(args.manifest)}
    retrieval = {row["id"]: row for row in read_jsonl(args.retrieval)}
    targets = {row["id"]: row for row in read_jsonl(args.targets)}
    for name, rows in (("manifest", manifests), ("retrieval", retrieval),
                       ("targets", targets)):
        missing = expected_ids - set(rows)
        if missing:
            raise ValueError(f"{name}: missing {len(missing)} held-fold IDs")
    if any(
        targets[value].get("train_gold_injected") is not False
        or targets[value].get("candidate_ids")
        != targets[value].get("natural_candidate_ids")
        for value in expected_ids
    ):
        raise ValueError("held-fold targets contain gold injection")

    generic_probabilities = {
        "anchor": anchor, "control": anchor, "joint": candidate,
    }
    generic, cases = build_atlas(
        ids, labels, generic_probabilities, manifests, retrieval, targets,
        prediction_rows["candidate"], args.minimum_group_size,
    )
    result = {
        "protocol": "B13_diagnostic_only_curriculum_failure_atlas",
        "samples": len(ids),
        "overall": {
            "anchor": generic["overall"]["anchor"],
            "candidate": generic["overall"]["joint"],
        },
        "candidate_vs_anchor": generic["comparisons"]["joint_vs_anchor"],
        "prediction_shift": prediction_shift(
            labels, anchor, candidate
        ),
        "transition_summary": transition_summary(cases),
        "auxiliary_heads": auxiliary_metrics(
            ids, targets, prediction_rows["candidate"]
        ),
        "oracle_anchor_or_candidate": generic["oracle_anchor_or_joint"],
        "probability_interpolation": interpolation_diagnostic(
            labels, anchor, candidate
        ),
        "head_verdict_diagnostic": head_verdict_diagnostic(
            labels, anchor, candidate, hierarchical
        ),
        "slices": generic["slices"],
        "worst_candidate_vs_anchor_groups": (
            generic["worst_joint_vs_anchor_groups_minimum_size"]
        ),
        "curriculum_audit": curriculum,
        "fold_spec_sha256": fold_signature,
        "exploratory_diagnostic": True,
        "fresh_confirmation_required": True,
        "official_validation_used": False,
        "test_split_used": False,
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "retrieval_sha256": sha256(args.retrieval),
            "targets_sha256": sha256(args.targets),
            "anchor_predictions_sha256": sha256(
                args.anchor / "val_predictions.jsonl"
            ),
            "candidate_predictions_sha256": sha256(
                args.candidate / "val_predictions.jsonl"
            ),
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
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_summary(result), encoding="utf-8")
    args.cases.parent.mkdir(parents=True, exist_ok=True)
    args.cases.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False)
                  for row in cleaned_cases) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "saved": str(args.output),
        "markdown": str(args.markdown),
        "cases": str(args.cases),
        "overall": result["overall"],
        "prediction_shift": result["prediction_shift"],
        "transition_summary": result["transition_summary"],
        "auxiliary_heads": result["auxiliary_heads"],
        "oracle": result["oracle_anchor_or_candidate"],
        "probability_interpolation": result["probability_interpolation"],
        "head_verdict_diagnostic": result["head_verdict_diagnostic"],
        "worst_groups": result["worst_candidate_vs_anchor_groups"][:12],
        "official_validation_used": False,
        "test_split_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
