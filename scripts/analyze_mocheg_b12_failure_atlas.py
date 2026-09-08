"""Build a leakage-safe failure atlas for the failed B12 fold-0 screen.

This is a diagnostic, not a model-selection script. It explains the effects of
additional optimization and joint constraint supervision using only the held
partition of the fresh train-only B12 fold assignment. Official validation and
test data are never read.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scripts.analyze_mocheg_b12_joint_constraints import (
    validate_joint_run,
)
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_expert_complementarity import read_predictions
from scripts.prepare_mocheg_sv_folds import sha256
from scripts.run_mocheg_visual_retrieval import read_jsonl


LABEL_NAMES = {0: "supported", 1: "refuted", 2: "nei"}


def normalized_probabilities(rows: dict[str, dict], ids: list[str],
                             field: str = "probabilities",
                             classes: int = 3) -> np.ndarray:
    values = np.asarray([rows[value][field] for value in ids], dtype=np.float64)
    if values.shape != (len(ids), classes) or not np.isfinite(values).all():
        raise ValueError(f"invalid {field} matrix: {values.shape}")
    return values / np.clip(values.sum(axis=1, keepdims=True), 1e-12, None)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray,
                           classes: int = 3,
                           class_names: dict[int, str] | None = None) -> dict:
    prediction = probabilities.argmax(axis=1)
    class_ids = list(range(classes))
    names = class_names or {
        index: LABEL_NAMES.get(index, str(index)) for index in class_ids
    }
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(
            labels, prediction, labels=class_ids, average="macro",
            zero_division=0,
        )),
        "class_f1": {
            names[index]: float(value)
            for index, value in zip(class_ids, f1_score(
                labels, prediction, labels=class_ids, average=None,
                zero_division=0,
            ))
        },
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=class_ids
        ).tolist(),
    }


def comparison(labels: np.ndarray, baseline: np.ndarray,
               candidate: np.ndarray) -> dict:
    baseline_prediction = baseline.argmax(axis=1)
    candidate_prediction = candidate.argmax(axis=1)
    baseline_correct = baseline_prediction == labels
    candidate_correct = candidate_prediction == labels
    baseline_metrics = classification_metrics(labels, baseline)
    candidate_metrics = classification_metrics(labels, candidate)
    return {
        "macro_f1_delta": float(
            candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"]
        ),
        "accuracy_delta": float(
            candidate_metrics["accuracy"] - baseline_metrics["accuracy"]
        ),
        "helpful": int(np.sum(~baseline_correct & candidate_correct)),
        "harmful": int(np.sum(baseline_correct & ~candidate_correct)),
        "net_corrections": int(np.sum(candidate_correct) - np.sum(baseline_correct)),
        "prediction_disagreements": int(np.sum(
            baseline_prediction != candidate_prediction
        )),
    }


def entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def quantile_edges(values: list[float]) -> list[float]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not len(finite):
        return []
    return np.unique(np.quantile(finite, [.25, .5, .75])).tolist()


def quantile_bin(value: float, edges: list[float]) -> str:
    if not np.isfinite(value):
        return "missing"
    return f"q{int(np.searchsorted(edges, value, side='right')) + 1}"


def confidence_bin(value: float) -> str:
    if value < .5:
        return "lt_0.50"
    if value < .7:
        return "0.50_0.70"
    if value < .9:
        return "0.70_0.90"
    return "ge_0.90"


def rank_bin(qrel_available: bool, first_gold_rank: int | None,
             natural_gold_hit: bool) -> str:
    if not qrel_available:
        return "qrel_absent"
    if natural_gold_hit:
        if first_gold_rank == 1:
            return "gold_rank_1"
        return "gold_rank_2_5"
    if first_gold_rank is not None:
        return "gold_rank_after_5"
    return "gold_not_retrieved"


def subset_report(indices: np.ndarray, labels: np.ndarray,
                  probabilities: dict[str, np.ndarray]) -> dict:
    selected_labels = labels[indices]
    selected = {
        name: values[indices] for name, values in probabilities.items()
    }
    return {
        "samples": int(indices.sum()),
        "label_counts": dict(Counter(
            LABEL_NAMES[int(value)] for value in selected_labels.tolist()
        )),
        "models": {
            name: classification_metrics(selected_labels, values)
            for name, values in selected.items()
        },
        "joint_vs_anchor": comparison(
            selected_labels, selected["anchor"], selected["joint"]
        ),
        "joint_vs_control": comparison(
            selected_labels, selected["control"], selected["joint"]
        ),
        "control_vs_anchor": comparison(
            selected_labels, selected["anchor"], selected["control"]
        ),
    }


def auxiliary_metrics(ids: list[str], targets: dict[str, dict],
                      joint_rows: dict[str, dict]) -> dict:
    sufficient_ids = [
        value for value in ids
        if targets[value].get("sufficiency_target") is not None
    ]
    sufficiency_probabilities = normalized_probabilities(
        joint_rows, sufficient_ids, "sufficiency_probabilities", 2
    )
    # Model order is [Y, N], target convention is 1=sufficient, 0=insufficient.
    sufficiency_labels = np.asarray([
        int(targets[value]["sufficiency_target"]) for value in sufficient_ids
    ])
    sufficiency_model_order = sufficiency_probabilities[:, [1, 0]]

    polarity_ids = [
        value for value in ids
        if targets[value].get("polarity_target") is not None
    ]
    polarity_probabilities = normalized_probabilities(
        joint_rows, polarity_ids, "polarity_probabilities", 2
    )
    polarity_labels = np.asarray([
        int(targets[value]["polarity_target"]) for value in polarity_ids
    ])
    return {
        "sufficiency": {
            **classification_metrics(
                sufficiency_labels, sufficiency_model_order, classes=2,
                class_names={0: "insufficient", 1: "sufficient"},
            ),
            "target_counts": dict(Counter(
                "sufficient" if value == 1 else "insufficient"
                for value in sufficiency_labels.tolist()
            )),
            "probability_order": ["insufficient", "sufficient"],
        },
        "polarity": {
            **classification_metrics(
                polarity_labels, polarity_probabilities, classes=2,
                class_names={0: "supported", 1: "refuted"},
            ),
            "target_counts": dict(Counter(
                LABEL_NAMES[value] for value in polarity_labels.tolist()
            )),
            "probability_order": ["supported", "refuted"],
        },
    }


def build_atlas(ids: list[str], labels: np.ndarray,
                probabilities: dict[str, np.ndarray],
                manifests: dict[str, dict], retrieval: dict[str, dict],
                targets: dict[str, dict], joint_rows: dict[str, dict],
                minimum_group_size: int = 50) -> tuple[dict, list[dict]]:
    if not ids:
        raise ValueError("B12 failure atlas received no samples")
    prediction = {
        name: values.argmax(axis=1)
        for name, values in probabilities.items()
    }
    correct = {
        name: values == labels for name, values in prediction.items()
    }
    confidence = {
        name: values.max(axis=1) for name, values in probabilities.items()
    }
    entropies = {
        name: entropy(values) for name, values in probabilities.items()
    }

    claim_lengths = [len(manifests[value].get("claim", "").split()) for value in ids]
    retrieval_confidences = [
        float(retrieval[value].get("retrieval_confidence", np.nan))
        for value in ids
    ]
    retrieval_margins = []
    for value in ids:
        scores = retrieval[value].get("retrieved_scores", [])
        retrieval_margins.append(
            float(scores[0]) - float(scores[1])
            if len(scores) > 1 else math.nan
        )
    edges = {
        "claim_length": quantile_edges(claim_lengths),
        "retrieval_confidence": quantile_edges(retrieval_confidences),
        "retrieval_margin": quantile_edges(retrieval_margins),
    }

    case_rows = []
    for index, sample_id in enumerate(ids):
        target = targets[sample_id]
        retrieved = retrieval[sample_id]
        rank = retrieved.get("first_gold_rank")
        rank = int(rank) if rank is not None else None
        qrel = bool(target.get("qrel_available"))
        natural_hit = bool(target.get("natural_gold_hit"))
        sufficiency_probability = np.asarray(
            joint_rows[sample_id]["sufficiency_probabilities"], dtype=float
        )
        sufficiency_prediction = int(sufficiency_probability.argmax() == 0)
        polarity_probability = np.asarray(
            joint_rows[sample_id]["polarity_probabilities"], dtype=float
        )
        polarity_prediction = int(polarity_probability.argmax())
        anchor_transition = (
            "helpful" if not correct["anchor"][index] and correct["joint"][index]
            else "harmful" if correct["anchor"][index] and not correct["joint"][index]
            else "both_correct" if correct["anchor"][index]
            else "both_wrong"
        )
        control_transition = (
            "helpful" if not correct["control"][index] and correct["joint"][index]
            else "harmful" if correct["control"][index] and not correct["joint"][index]
            else "both_correct" if correct["control"][index]
            else "both_wrong"
        )
        row = {
            "id": sample_id,
            "claim": manifests[sample_id].get("claim", ""),
            "source": str(manifests[sample_id].get("source", "unknown") or "unknown"),
            "gold": LABEL_NAMES[int(labels[index])],
            "predictions": {
                name: LABEL_NAMES[int(values[index])]
                for name, values in prediction.items()
            },
            "confidence": {
                name: float(values[index]) for name, values in confidence.items()
            },
            "entropy": {
                name: float(values[index]) for name, values in entropies.items()
            },
            "joint_vs_anchor": anchor_transition,
            "joint_vs_control": control_transition,
            "qrel_available": qrel,
            "natural_gold_hit_at_5": natural_hit,
            "first_gold_rank": rank,
            "retrieval_status": rank_bin(qrel, rank, natural_hit),
            "retrieval_confidence": retrieval_confidences[index],
            "retrieval_margin": (
                retrieval_margins[index]
                if np.isfinite(retrieval_margins[index]) else None
            ),
            "claim_words": claim_lengths[index],
            "sufficiency_target": target.get("sufficiency_target"),
            "sufficiency_prediction": sufficiency_prediction,
            "sufficiency_confidence": float(sufficiency_probability.max()),
            "polarity_target": target.get("polarity_target"),
            "polarity_prediction": polarity_prediction,
            "polarity_confidence": float(polarity_probability.max()),
        }
        row["groups"] = {
            "gold_label": row["gold"],
            "source": row["source"],
            "qrel": "available" if qrel else "absent",
            "retrieval_status": row["retrieval_status"],
            "claim_length": quantile_bin(claim_lengths[index], edges["claim_length"]),
            "retrieval_confidence": quantile_bin(
                retrieval_confidences[index], edges["retrieval_confidence"]
            ),
            "retrieval_margin": quantile_bin(
                retrieval_margins[index], edges["retrieval_margin"]
            ),
            "anchor_confidence": confidence_bin(confidence["anchor"][index]),
            "anchor_prediction": LABEL_NAMES[int(prediction["anchor"][index])],
            "sufficiency_target": (
                "unlabelled" if target.get("sufficiency_target") is None
                else "sufficient" if int(target["sufficiency_target"]) == 1
                else "insufficient"
            ),
        }
        case_rows.append(row)

    slices = {}
    for field in next(iter(case_rows))["groups"]:
        slices[field] = {}
        values = sorted({row["groups"][field] for row in case_rows})
        for value in values:
            selected = np.asarray([
                row["groups"][field] == value for row in case_rows
            ])
            slices[field][value] = subset_report(
                selected, labels, probabilities
            )

    eligible = []
    for field, groups in slices.items():
        for value, report in groups.items():
            if report["samples"] >= minimum_group_size:
                eligible.append({
                    "field": field,
                    "value": value,
                    "samples": report["samples"],
                    **report["joint_vs_anchor"],
                })
    worst = sorted(eligible, key=lambda row: (
        row["macro_f1_delta"], row["accuracy_delta"]
    ))

    transitions = Counter(
        (
            LABEL_NAMES[int(labels[index])],
            LABEL_NAMES[int(prediction["anchor"][index])],
            LABEL_NAMES[int(prediction["joint"][index])],
        )
        for index in range(len(ids))
        if prediction["anchor"][index] != prediction["joint"][index]
    )
    transition_rows = [{
        "gold": gold, "anchor": anchor, "joint": joint, "count": int(count),
        "effect": (
            "helpful" if anchor != gold and joint == gold
            else "harmful" if anchor == gold and joint != gold
            else "wrong_to_wrong"
        ),
    } for (gold, anchor, joint), count in transitions.most_common()]

    overall = {
        name: classification_metrics(labels, values)
        for name, values in probabilities.items()
    }
    joint_vs_anchor = comparison(
        labels, probabilities["anchor"], probabilities["joint"]
    )
    joint_vs_control = comparison(
        labels, probabilities["control"], probabilities["joint"]
    )
    control_vs_anchor = comparison(
        labels, probabilities["anchor"], probabilities["control"]
    )
    oracle_prediction = prediction["anchor"].copy()
    use_joint = ~correct["anchor"] & correct["joint"]
    oracle_prediction[use_joint] = prediction["joint"][use_joint]
    oracle = classification_metrics(labels, np.eye(3)[oracle_prediction])
    source_losses = {
        value: report["joint_vs_anchor"]["macro_f1_delta"]
        for value, report in slices["source"].items()
    }
    auxiliary_gain = joint_vs_control["macro_f1_delta"]
    compute_damage = control_vs_anchor["macro_f1_delta"]
    recovery = (
        auxiliary_gain / abs(compute_damage)
        if compute_damage < 0 else None
    )
    atlas = {
        "protocol": "B13_diagnostic_only_B12_fresh_fold0_failure_atlas",
        "samples": len(ids),
        "overall": overall,
        "comparisons": {
            "joint_vs_anchor": joint_vs_anchor,
            "joint_vs_compute_matched_control": joint_vs_control,
            "compute_matched_control_vs_anchor": control_vs_anchor,
        },
        "effect_decomposition": {
            "compute_matched_control_minus_anchor_macro_f1": compute_damage,
            "joint_constraints_minus_control_macro_f1": auxiliary_gain,
            "joint_constraints_minus_anchor_macro_f1": joint_vs_anchor["macro_f1_delta"],
            "fraction_of_compute_damage_recovered_by_auxiliary": recovery,
        },
        "oracle_anchor_or_joint": {
            **oracle,
            "macro_f1_delta_vs_anchor": (
                oracle["macro_f1"] - overall["anchor"]["macro_f1"]
            ),
            "joint_only_correct": int(use_joint.sum()),
            "anchor_only_correct": int(np.sum(
                correct["anchor"] & ~correct["joint"]
            )),
        },
        "auxiliary_heads": auxiliary_metrics(ids, targets, joint_rows),
        "quantile_edges": edges,
        "slices": slices,
        "worst_joint_vs_anchor_groups_minimum_size": worst[:20],
        "prediction_transitions": transition_rows,
        "evidence_based_signals": {
            "additional_optimization_hurts_anchor": compute_damage < -.005,
            "auxiliary_constraints_help_vs_matched_control": auxiliary_gain > 0,
            "auxiliary_fully_recovers_compute_damage": (
                joint_vs_anchor["macro_f1_delta"] >= 0
            ),
            "largest_source_harm": min(source_losses, key=source_losses.get),
            "largest_source_harm_macro_f1_delta": min(source_losses.values()),
            "anchor_joint_complementarity_macro_f1_ceiling": oracle["macro_f1"],
        },
        "minimum_slice_size": minimum_group_size,
        "fresh_fold_assignment": True,
        "exploratory_diagnostic": True,
        "official_validation_used": False,
        "test_split_used": False,
    }
    priority = {"harmful": 0, "helpful": 1, "both_wrong": 2, "both_correct": 3}
    case_rows.sort(key=lambda row: (
        priority[row["joint_vs_anchor"]],
        -abs(row["confidence"]["joint"] - row["confidence"]["anchor"]),
        row["id"],
    ))
    return atlas, case_rows


def markdown_summary(atlas: dict) -> str:
    overall = atlas["overall"]
    lines = [
        "# B12 failure atlas (fresh train-only fold 0)", "",
        "Official validation used: **no**  ",
        "Test used: **no**", "",
        "## Overall", "",
        "| Model | Accuracy | Macro-F1 |", "|---|---:|---:|",
    ]
    for name in ("anchor", "control", "joint"):
        row = overall[name]
        lines.append(
            f"| {name} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |"
        )
    effect = atlas["effect_decomposition"]
    lines.extend([
        "", "## Effect decomposition", "",
        f"- Extra-compute control minus anchor: "
        f"{effect['compute_matched_control_minus_anchor_macro_f1']:+.6f}",
        f"- Joint constraints minus matched control: "
        f"{effect['joint_constraints_minus_control_macro_f1']:+.6f}",
        f"- Joint constraints minus anchor: "
        f"{effect['joint_constraints_minus_anchor_macro_f1']:+.6f}",
        f"- Fraction of compute damage recovered: "
        f"{effect['fraction_of_compute_damage_recovered_by_auxiliary']}",
        "", "## Auxiliary heads", "",
        "| Head | Samples | Accuracy | Macro-F1 |", "|---|---:|---:|---:|",
    ])
    for name, row in atlas["auxiliary_heads"].items():
        lines.append(
            f"| {name} | {row['samples']} | {row['accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} |"
        )
    oracle = atlas["oracle_anchor_or_joint"]
    lines.extend([
        "", "## Complementarity ceiling", "",
        f"- Oracle anchor-or-joint Macro-F1: {oracle['macro_f1']:.6f}",
        f"- Delta over anchor: {oracle['macro_f1_delta_vs_anchor']:+.6f}",
        f"- Joint-only correct / anchor-only correct: "
        f"{oracle['joint_only_correct']} / {oracle['anchor_only_correct']}",
        "", "## Worst slices (minimum group size enforced)", "",
        "| Field | Value | N | MF1 delta | Acc delta | Help | Harm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in atlas["worst_joint_vs_anchor_groups_minimum_size"][:12]:
        lines.append(
            f"| {row['field']} | {row['value']} | {row['samples']} | "
            f"{row['macro_f1_delta']:+.4f} | {row['accuracy_delta']:+.4f} | "
            f"{row['helpful']} | {row['harmful']} |"
        )
    lines.extend(["", "## Evidence-based signals", ""])
    for name, value in atlas["evidence_based_signals"].items():
        lines.append(f"- {name}: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("outputs/mocheg_b12_fresh/fold_0")
    parser.add_argument("--anchor", type=Path, default=root / "anchor")
    parser.add_argument("--control", type=Path, default=root / "direct_control")
    parser.add_argument("--joint", type=Path, default=root / "joint_constraints")
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
        "outputs/mocheg_b12_failure_atlas.json"))
    parser.add_argument("--markdown", type=Path, default=Path(
        "outputs/mocheg_b12_failure_atlas.md"))
    parser.add_argument("--cases", type=Path, default=Path(
        "outputs/mocheg_b12_failure_cases.jsonl"))
    args = parser.parse_args()
    if args.fold != 0:
        raise ValueError("B12 failure diagnosis is restricted to fresh fold 0")
    fold_payload = json.loads(args.fold_spec.read_text(encoding="utf-8"))
    if (
        int(fold_payload.get("seed", -1)) != 2027
        or fold_payload.get("validation_split_used") is not False
        or fold_payload.get("test_split_used") is not False
    ):
        raise ValueError("expected locked B12 seed-2027 train-only folds")
    if fold_payload.get("manifest_sha256") != sha256(args.manifest):
        raise ValueError("B12 fold specification does not match the manifest")
    fold_signature = sha256(args.fold_spec)
    summaries = {
        "anchor": validate_run(args.anchor, args.fold, "anchor"),
        "control": validate_joint_run(args.control, args.fold, "control", False),
        "joint": validate_joint_run(args.joint, args.fold, "joint", True),
    }
    for name, summary in summaries.items():
        if summary.get("provenance", {}).get("fold_spec_sha256") != fold_signature:
            raise ValueError(f"{name}: B12 fold signature mismatch")
    if sum(summaries["control"]["training_task_counts"].values()) != sum(
        summaries["joint"]["training_task_counts"].values()
    ):
        raise ValueError("joint and matched-control update counts differ")

    prediction_rows = {
        "anchor": read_predictions(args.anchor / "val_predictions.jsonl"),
        "control": read_predictions(args.control / "val_predictions.jsonl"),
        "joint": read_predictions(args.joint / "val_predictions.jsonl"),
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
    probabilities = {
        name: normalized_probabilities(rows, ids)
        for name, rows in prediction_rows.items()
    }
    for name, rows in prediction_rows.items():
        observed = np.asarray([int(rows[value]["gold"]) for value in ids])
        if not np.array_equal(labels, observed):
            raise ValueError(f"{name}: gold labels are not aligned")
    manifests = {row["id"]: row for row in read_jsonl(args.manifest)}
    retrieval = {row["id"]: row for row in read_jsonl(args.retrieval)}
    targets = {row["id"]: row for row in read_jsonl(args.targets)}
    for name, rows in (
        ("manifest", manifests), ("retrieval", retrieval), ("targets", targets)
    ):
        missing = expected_ids - set(rows)
        if missing:
            raise ValueError(f"{name}: missing {len(missing)} held-fold IDs")
    if any(
        targets[value].get("train_gold_injected") is not False
        or targets[value].get("candidate_ids")
        != targets[value].get("natural_candidate_ids")
        for value in expected_ids
    ):
        raise ValueError("held-fold diagnostic targets contain gold injection")

    atlas, cases = build_atlas(
        ids, labels, probabilities, manifests, retrieval, targets,
        prediction_rows["joint"], args.minimum_group_size,
    )
    atlas["fold_spec_sha256"] = fold_signature
    atlas["provenance"] = {
        "manifest_sha256": sha256(args.manifest),
        "retrieval_sha256": sha256(args.retrieval),
        "targets_sha256": sha256(args.targets),
        "anchor_predictions_sha256": sha256(
            args.anchor / "val_predictions.jsonl"
        ),
        "control_predictions_sha256": sha256(
            args.control / "val_predictions.jsonl"
        ),
        "joint_predictions_sha256": sha256(
            args.joint / "val_predictions.jsonl"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(atlas, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_summary(atlas), encoding="utf-8")
    args.cases.parent.mkdir(parents=True, exist_ok=True)
    args.cases.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in cases) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "saved": str(args.output),
        "markdown": str(args.markdown),
        "cases": str(args.cases),
        "effect_decomposition": atlas["effect_decomposition"],
        "auxiliary_heads": atlas["auxiliary_heads"],
        "oracle_anchor_or_joint": atlas["oracle_anchor_or_joint"],
        "worst_groups": atlas["worst_joint_vs_anchor_groups_minimum_size"][:12],
        "evidence_based_signals": atlas["evidence_based_signals"],
        "official_validation_used": False,
        "test_split_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
