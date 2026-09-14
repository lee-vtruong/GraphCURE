"""Diagnose the failed B16 confirmation without fitting or selecting a model.

B17 treats the compute-matched direct control as the primary baseline.  It
uses only held predictions from B16 confirmation folds 1--4 and train-split
metadata.  Official validation and test are never read.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b12_failure_atlas import (
    LABEL_NAMES,
    classification_metrics,
    comparison,
    confidence_bin,
    normalized_probabilities,
    quantile_bin,
    quantile_edges,
    rank_bin,
)
from scripts.analyze_mocheg_b16_counterfactual_curriculum import (
    FOLD_SEED,
    validate_counterfactual_run,
)
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_expert_complementarity import read_predictions
from scripts.prepare_mocheg_sv_folds import sha256
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.summarize_mocheg_b16_confirmation import CONFIRMATION_FOLDS


def validate_matched_control_run(summary: dict, fold: int) -> None:
    """Require the exact compute-matched verdict-only B16 control."""
    if summary.get("fold") != fold:
        raise ValueError("matched control fold mismatch")
    if summary.get("training_from_base") is not True:
        raise ValueError("matched control must train from base")
    if summary.get("fixed_checkpoint_epoch") != 3:
        raise ValueError("matched control requires fixed checkpoint epoch 3")
    if summary.get("selected_hierarchical_weight") != 0:
        raise ValueError("matched control requires direct verdict inference")
    if set(summary.get("training_task_counts", {})) != {"verdict"}:
        raise ValueError("matched control must contain verdict training only")
    if summary.get("counterfactual_verdict_curriculum") is True:
        raise ValueError("counterfactual curriculum leaked into matched control")


def transition_rows(labels: np.ndarray, baseline: np.ndarray,
                    candidate: np.ndarray) -> list[dict]:
    """Count all changed predictions and attach their causal outcome."""
    baseline_prediction = baseline.argmax(1)
    candidate_prediction = candidate.argmax(1)
    counts = Counter()
    for gold, old, new in zip(
        labels.tolist(), baseline_prediction.tolist(),
        candidate_prediction.tolist(),
    ):
        if old == new:
            continue
        effect = (
            "helpful" if old != gold and new == gold
            else "harmful" if old == gold and new != gold
            else "wrong_to_wrong"
        )
        counts[(LABEL_NAMES[gold], LABEL_NAMES[old], LABEL_NAMES[new], effect)] += 1
    return [
        {
            "gold": gold, "baseline": old, "candidate": new,
            "effect": effect, "count": int(count),
        }
        for (gold, old, new, effect), count in counts.most_common()
    ]


def prediction_shift(probabilities: dict[str, np.ndarray]) -> dict:
    names = tuple(probabilities)
    counts = {
        name: Counter(
            LABEL_NAMES[int(value)] for value in matrix.argmax(1).tolist()
        )
        for name, matrix in probabilities.items()
    }
    return {
        "prediction_counts": {
            name: dict(value) for name, value in counts.items()
        },
        "candidate_minus_control": {
            label: int(counts["candidate"][label] - counts["control"][label])
            for label in LABEL_NAMES.values()
        },
        "models": list(names),
    }


def subset(mask: np.ndarray, labels: np.ndarray,
           probabilities: dict[str, np.ndarray]) -> dict:
    selected_labels = labels[mask]
    selected = {name: value[mask] for name, value in probabilities.items()}
    return {
        "samples": int(mask.sum()),
        "label_counts": dict(Counter(
            LABEL_NAMES[int(value)] for value in selected_labels.tolist()
        )),
        "models": {
            name: classification_metrics(selected_labels, value)
            for name, value in selected.items()
        },
        "candidate_vs_control": comparison(
            selected_labels, selected["control"], selected["candidate"]
        ),
        "candidate_vs_anchor": comparison(
            selected_labels, selected["anchor"], selected["candidate"]
        ),
        "control_vs_anchor": comparison(
            selected_labels, selected["anchor"], selected["control"]
        ),
    }


def build_atlas(ids: list[str], folds: np.ndarray, labels: np.ndarray,
                probabilities: dict[str, np.ndarray],
                manifests: dict[str, dict], retrieval: dict[str, dict],
                targets: dict[str, dict], training_audit: list[dict],
                minimum_group_size: int = 50) -> tuple[dict, list[dict]]:
    """Build the B17 atlas from already frozen held predictions."""
    if not ids or set(probabilities) != {"anchor", "control", "candidate"}:
        raise ValueError("B17 requires aligned anchor/control/candidate inputs")
    predictions = {name: value.argmax(1) for name, value in probabilities.items()}
    confidence = {name: value.max(1) for name, value in probabilities.items()}
    claim_lengths = [len(manifests[value].get("claim", "").split()) for value in ids]
    retrieval_confidences = [
        float(retrieval[value].get("retrieval_confidence", np.nan))
        for value in ids
    ]
    retrieval_margins = []
    for value in ids:
        scores = retrieval[value].get("retrieved_scores", [])
        retrieval_margins.append(
            float(scores[0]) - float(scores[1]) if len(scores) > 1 else math.nan
        )
    edges = {
        "claim_length": quantile_edges(claim_lengths),
        "retrieval_confidence": quantile_edges(retrieval_confidences),
        "retrieval_margin": quantile_edges(retrieval_margins),
    }
    cases = []
    for index, sample_id in enumerate(ids):
        target = targets[sample_id]
        retrieved = retrieval[sample_id]
        rank = retrieved.get("first_gold_rank")
        rank = int(rank) if rank is not None else None
        qrel = bool(target.get("qrel_available"))
        natural_hit = bool(target.get("natural_gold_hit"))
        control_correct = predictions["control"][index] == labels[index]
        candidate_correct = predictions["candidate"][index] == labels[index]
        effect = (
            "helpful" if not control_correct and candidate_correct
            else "harmful" if control_correct and not candidate_correct
            else "both_correct" if control_correct else "both_wrong"
        )
        row = {
            "id": sample_id,
            "fold": int(folds[index]),
            "claim": manifests[sample_id].get("claim", ""),
            "source": str(manifests[sample_id].get("source", "unknown") or "unknown"),
            "gold": LABEL_NAMES[int(labels[index])],
            "predictions": {
                name: LABEL_NAMES[int(value[index])]
                for name, value in predictions.items()
            },
            "confidence": {
                name: float(value[index]) for name, value in confidence.items()
            },
            "candidate_vs_control": effect,
            "qrel_available": qrel,
            "counterfactual_eligible": target.get("polarity_target") is not None,
            "retrieval_status": rank_bin(qrel, rank, natural_hit),
            "first_gold_rank": rank,
            "retrieval_confidence": retrieval_confidences[index],
            "retrieval_margin": (
                retrieval_margins[index]
                if np.isfinite(retrieval_margins[index]) else None
            ),
            "claim_words": claim_lengths[index],
        }
        row["groups"] = {
            "fold": str(row["fold"]),
            "gold_label": row["gold"],
            "source": row["source"],
            "qrel": "available" if qrel else "absent",
            "counterfactual_eligible": str(row["counterfactual_eligible"]).lower(),
            "retrieval_status": row["retrieval_status"],
            "claim_length": quantile_bin(claim_lengths[index], edges["claim_length"]),
            "retrieval_confidence": quantile_bin(
                retrieval_confidences[index], edges["retrieval_confidence"]
            ),
            "retrieval_margin": quantile_bin(
                retrieval_margins[index], edges["retrieval_margin"]
            ),
            "control_prediction": row["predictions"]["control"],
            "control_confidence": confidence_bin(confidence["control"][index]),
            "confidence_advantage": (
                "candidate_higher" if confidence["candidate"][index]
                > confidence["control"][index] else "control_higher_or_equal"
            ),
        }
        cases.append(row)

    slices = {}
    eligible_groups = []
    for field in cases[0]["groups"]:
        groups = {}
        for value in sorted({row["groups"][field] for row in cases}):
            mask = np.asarray([row["groups"][field] == value for row in cases])
            report = subset(mask, labels, probabilities)
            groups[value] = report
            if report["samples"] >= minimum_group_size:
                effect = report["candidate_vs_control"]
                eligible_groups.append({
                    "field": field, "value": value,
                    "samples": report["samples"], **effect,
                })
        slices[field] = groups

    interactions = {}
    for left, right in (
        ("fold", "gold_label"), ("source", "qrel"),
        ("qrel", "gold_label"),
        ("counterfactual_eligible", "retrieval_status"),
    ):
        groups = {}
        values = sorted({
            (row["groups"][left], row["groups"][right]) for row in cases
        })
        for left_value, right_value in values:
            mask = np.asarray([
                row["groups"][left] == left_value
                and row["groups"][right] == right_value for row in cases
            ])
            if int(mask.sum()) >= minimum_group_size:
                groups[f"{left_value}|{right_value}"] = subset(
                    mask, labels, probabilities
                )
        interactions[f"{left}_x_{right}"] = groups

    metrics = {
        name: classification_metrics(labels, value)
        for name, value in probabilities.items()
    }
    candidate_vs_control = comparison(
        labels, probabilities["control"], probabilities["candidate"]
    )
    control_vs_anchor = comparison(
        labels, probabilities["anchor"], probabilities["control"]
    )
    class_delta_vs_control = {
        name: metrics["candidate"]["class_f1"][name]
        - metrics["control"]["class_f1"][name]
        for name in LABEL_NAMES.values()
    }
    diagnosis = {
        "matched_control_is_stronger_than_anchor": (
            control_vs_anchor["macro_f1_delta"] > 0
        ),
        "counterfactual_beats_matched_control": (
            candidate_vs_control["macro_f1_delta"] > 0
        ),
        "counterfactual_improves_nei_vs_control": (
            class_delta_vs_control["nei"] > 0
        ),
        "b16_mechanism_confirmed": (
            candidate_vs_control["macro_f1_delta"] >= .003
            and class_delta_vs_control["nei"] >= .01
        ),
    }
    diagnosis["conclusion"] = (
        "B16 did not identify a counterfactual-treatment benefit beyond the "
        "matched training trajectory; do not tune the omission ratio on these "
        "folds. Use the matched control as the reference for any new hypothesis."
        if not diagnosis["b16_mechanism_confirmed"] else
        "Counterfactual benefit remains plausible and requires a fresh preregistration."
    )
    ranked_harm = sorted(
        (
            {
                **row,
                "harm_minus_help": int(row["harmful"] - row["helpful"]),
                "harm_share_among_disagreements": (
                    float(row["harmful"] / row["prediction_disagreements"])
                    if row["prediction_disagreements"] else 0.0
                ),
            }
            for row in eligible_groups
        ),
        key=lambda row: (
            row["harm_minus_help"], row["harm_share_among_disagreements"],
            row["samples"],
        ),
        reverse=True,
    )
    priority = {"harmful": 0, "helpful": 1, "both_wrong": 2, "both_correct": 3}
    cases.sort(key=lambda row: (
        priority[row["candidate_vs_control"]],
        -abs(row["confidence"]["candidate"] - row["confidence"]["control"]),
        row["id"],
    ))
    return {
        "protocol": "B17_post_failure_B16_confirmation_atlas",
        "samples": len(ids),
        "folds": list(CONFIRMATION_FOLDS),
        "metrics": metrics,
        "comparisons": {
            "candidate_vs_matched_control": candidate_vs_control,
            "matched_control_vs_anchor": control_vs_anchor,
            "candidate_vs_anchor": comparison(
                labels, probabilities["anchor"], probabilities["candidate"]
            ),
        },
        "class_f1_delta_candidate_vs_control": class_delta_vs_control,
        "prediction_shift": prediction_shift(probabilities),
        "transitions_vs_control": transition_rows(
            labels, probabilities["control"], probabilities["candidate"]
        ),
        "transitions_vs_anchor": transition_rows(
            labels, probabilities["anchor"], probabilities["candidate"]
        ),
        "training_exposure_audit": training_audit,
        "quantile_edges": edges,
        "slices": slices,
        "interactions": interactions,
        "worst_candidate_vs_control_groups": sorted(
            eligible_groups,
            key=lambda row: (row["macro_f1_delta"], row["accuracy_delta"]),
        )[:25],
        "best_candidate_vs_control_groups": sorted(
            eligible_groups,
            key=lambda row: (row["macro_f1_delta"], row["accuracy_delta"]),
            reverse=True,
        )[:25],
        "highest_harm_excess_groups": ranked_harm[:25],
        "diagnosis": diagnosis,
        "b17_decision": {
            "phase_type": "diagnostic_only",
            "b16_closed": True,
            "new_training_or_threshold_selection": False,
            "next_intervention_requires_fresh_fold_assignment": True,
        },
        "minimum_group_size": minimum_group_size,
        "fold0_used": False,
        "official_validation_used": False,
        "test_split_used": False,
    }, cases


def markdown(result: dict) -> str:
    metrics, comparisons = result["metrics"], result["comparisons"]
    lines = [
        "# MOCHEG B17: B16 confirmation failure atlas", "",
        "Fold 0 used: **no**  ", "Official validation used: **no**  ",
        "Test used: **no**", "", "## Overall", "",
        "| Model | Accuracy | Macro-F1 |", "|---|---:|---:|",
    ]
    for name in ("anchor", "control", "candidate"):
        row = metrics[name]
        lines.append(f"| {name} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")
    effect = comparisons["candidate_vs_matched_control"]
    lines.extend([
        "", "## Primary causal comparison: B16 vs matched control", "",
        f"- Macro-F1 delta: {effect['macro_f1_delta']:+.6f}",
        f"- Accuracy delta: {effect['accuracy_delta']:+.6f}",
        f"- Helpful / harmful: {effect['helpful']} / {effect['harmful']}",
        f"- Class-F1 delta: `{json.dumps(result['class_f1_delta_candidate_vs_control'])}`",
        "", "## Worst slices versus matched control", "",
        "| Field | Value | N | MF1 delta | Help | Harm |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in result["worst_candidate_vs_control_groups"][:15]:
        lines.append(
            f"| {row['field']} | {row['value']} | {row['samples']} | "
            f"{row['macro_f1_delta']:+.4f} | {row['helpful']} | {row['harmful']} |"
        )
    lines.extend([
        "", "## Diagnosis", "",
        f"- Matched control stronger than anchor: "
        f"**{result['diagnosis']['matched_control_is_stronger_than_anchor']}**",
        f"- B16 beats matched control: "
        f"**{result['diagnosis']['counterfactual_beats_matched_control']}**",
        f"- B16 improves NEI over control: "
        f"**{result['diagnosis']['counterfactual_improves_nei_vs_control']}**",
        f"- Mechanism confirmed: **{result['diagnosis']['b16_mechanism_confirmed']}**",
        "", result["diagnosis"]["conclusion"], "",
        "B17 is diagnostic-only. It does not authorize validation/test use or "
        "hyperparameter tuning on these folds.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/mocheg_b16_fresh"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--retrieval", type=Path, default=Path(
        "outputs/retrieval_mocheg_qwen3_reranked/train.jsonl"))
    parser.add_argument("--targets", type=Path, default=Path(
        "data/processed/mocheg_b6_targets_natural/train.jsonl"))
    parser.add_argument("--fold-spec", type=Path, default=Path(
        "data/processed/mocheg_b16_folds.json"))
    parser.add_argument("--minimum-group-size", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b17_b16_failure_atlas.json"))
    parser.add_argument("--markdown", type=Path, default=Path(
        "outputs/mocheg_b17_b16_failure_atlas.md"))
    parser.add_argument("--cases", type=Path, default=Path(
        "outputs/mocheg_b17_b16_failure_cases.jsonl"))
    args = parser.parse_args()

    fold_payload = json.loads(args.fold_spec.read_text(encoding="utf-8"))
    if (
        int(fold_payload.get("seed", -1)) != FOLD_SEED
        or fold_payload.get("validation_split_used") is not False
        or fold_payload.get("test_split_used") is not False
        or fold_payload.get("manifest_sha256") != sha256(args.manifest)
    ):
        raise ValueError("expected locked seed-2039 train-only folds")
    fold_signature = sha256(args.fold_spec)
    expected_by_fold = {
        int(row["fold"]): set(row["val_ids"]) for row in fold_payload["folds"]
    }
    all_rows = {name: {} for name in ("anchor", "control", "candidate")}
    fold_by_id, seen, training_audit = {}, set(), []
    for fold in CONFIRMATION_FOLDS:
        root = args.root / f"fold_{fold}"
        paths = {
            "anchor": root / "anchor",
            "control": root / "direct_control",
            "candidate": root / "counterfactual",
        }
        summaries = {
            name: validate_run(path, fold, f"fold_{fold}_{name}")
            for name, path in paths.items()
        }
        validate_matched_control_run(summaries["control"], fold)
        validate_counterfactual_run(summaries["candidate"], fold)
        for name, summary in summaries.items():
            if summary.get("provenance", {}).get("fold_spec_sha256") != fold_signature:
                raise ValueError(f"fold {fold} {name}: fold signature mismatch")
            rows = read_predictions(paths[name] / "val_predictions.jsonl")
            if set(rows) != expected_by_fold[fold]:
                raise ValueError(f"fold {fold} {name}: held IDs mismatch")
            all_rows[name].update(rows)
        overlap = seen & expected_by_fold[fold]
        if overlap:
            raise ValueError(f"fold {fold}: duplicate held IDs")
        seen.update(expected_by_fold[fold])
        fold_by_id.update({value: fold for value in expected_by_fold[fold]})
        candidate_summary = summaries["candidate"]
        training_audit.append({
            "fold": fold,
            "matched_control_training_task_counts": summaries["control"].get(
                "training_task_counts"
            ),
            "candidate_training_task_counts": candidate_summary.get(
                "training_task_counts"
            ),
            "candidate_epoch_training_task_counts": candidate_summary.get(
                "epoch_training_task_counts"
            ),
            "counterfactual_fraction": candidate_summary.get(
                "counterfactual_verdict_fraction"
            ),
            "counterfactual_epochs": candidate_summary.get(
                "counterfactual_verdict_epochs"
            ),
            "fixed_checkpoint_epoch": candidate_summary.get(
                "fixed_checkpoint_epoch"
            ),
        })

    ids = sorted(seen)
    manifests = {row["id"]: row for row in read_jsonl(args.manifest)}
    retrieval = {row["id"]: row for row in read_jsonl(args.retrieval)}
    targets = {row["id"]: row for row in read_jsonl(args.targets)}
    for name, rows in (("manifest", manifests), ("retrieval", retrieval),
                       ("targets", targets)):
        missing = seen - set(rows)
        if missing:
            raise ValueError(f"{name}: missing {len(missing)} held IDs")
    labels = np.asarray([int(all_rows["anchor"][value]["gold"]) for value in ids])
    for name in ("control", "candidate"):
        other = np.asarray([int(all_rows[name][value]["gold"]) for value in ids])
        if not np.array_equal(labels, other):
            raise ValueError(f"{name}: gold labels are misaligned")
    probabilities = {
        name: normalized_probabilities(rows, ids)
        for name, rows in all_rows.items()
    }
    result, cases = build_atlas(
        ids, np.asarray([fold_by_id[value] for value in ids]), labels,
        probabilities, manifests, retrieval, targets, training_audit,
        args.minimum_group_size,
    )
    result["fold_spec_sha256"] = fold_signature
    result["provenance"] = {
        "manifest_sha256": sha256(args.manifest),
        "retrieval_sha256": sha256(args.retrieval),
        "targets_sha256": sha256(args.targets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(result), encoding="utf-8")
    args.cases.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in cases) + "\n",
        encoding="utf-8",
    )
    print(markdown(result))


if __name__ == "__main__":
    main()
