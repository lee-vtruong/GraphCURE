"""Cross-fitted value-of-information gate for B14 NEI escapes.

This diagnostic asks whether observable inference-time signals can distinguish
helpful from harmful B13 expert calls. It uses five-way cross-fitting and a
fixed 0.5 threshold. Source, qrel annotations, gold ranks and labels are never
features. The inspected seed-2027 folds remain development-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analyze_mocheg_b12_failure_atlas import (
    LABEL_NAMES,
    classification_metrics,
)
from scripts.analyze_mocheg_b14_nei_escape_policy import LABEL_IDS, one_hot
from scripts.analyze_mocheg_b6_auxiliary_control import paired_comparison
from scripts.analyze_mocheg_b7_frozen_router import source_diagnostics
from scripts.prepare_mocheg_sv_folds import sha256
from scripts.run_mocheg_visual_retrieval import read_jsonl


FEATURE_NAMES = (
    "retrieval_confidence",
    "retrieval_margin",
    "claim_words",
    "anchor_confidence",
    "candidate_confidence",
    "candidate_confidence_advantage",
    "anchor_entropy",
    "candidate_entropy",
    "candidate_entropy_minus_anchor",
    "sufficiency_predicts_sufficient",
    "sufficiency_confidence",
    "sufficiency_probability",
    "polarity_predicts_refuted",
    "polarity_confidence",
    "candidate_predicts_supported",
    "candidate_predicts_refuted",
)
PROHIBITED_FEATURES = {
    "source", "qrel_available", "gold", "first_gold_rank",
    "natural_gold_hit_at_5", "sufficiency_target", "polarity_target",
}


def observable_features(row: dict) -> np.ndarray:
    """Create source-free, annotation-free inference features."""
    anchor_confidence = float(row["confidence"]["anchor"])
    candidate_confidence = float(row["confidence"]["candidate"])
    anchor_entropy = float(row["entropy"]["anchor"])
    candidate_entropy = float(row["entropy"]["candidate"])
    sufficiency_prediction = int(row["sufficiency_prediction"])
    sufficiency_confidence = float(row["sufficiency_confidence"])
    sufficiency_probability = (
        sufficiency_confidence if sufficiency_prediction == 1
        else 1.0 - sufficiency_confidence
    )
    candidate_prediction = LABEL_IDS[row["predictions"]["candidate"]]
    return np.asarray([
        float(row.get("retrieval_confidence", np.nan)),
        float(row.get("retrieval_margin")
              if row.get("retrieval_margin") is not None else np.nan),
        float(row["claim_words"]),
        anchor_confidence,
        candidate_confidence,
        candidate_confidence - anchor_confidence,
        anchor_entropy,
        candidate_entropy,
        candidate_entropy - anchor_entropy,
        float(sufficiency_prediction == 1),
        sufficiency_confidence,
        sufficiency_probability,
        float(int(row["polarity_prediction"]) == 1),
        float(row["polarity_confidence"]),
        float(candidate_prediction == 0),
        float(candidate_prediction == 1),
    ], dtype=np.float64)


def make_gate(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000,
            random_state=seed,
        )),
    ])


def metric_or_none(labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def crossfit_scores(
    features: np.ndarray, utility: np.ndarray, decisive: np.ndarray,
    route_candidate: np.ndarray, folds: np.ndarray, seed: int,
) -> tuple[np.ndarray, list[dict], list[np.ndarray]]:
    scores = np.zeros(len(features), dtype=np.float64)
    audits, coefficients = [], []
    for fold in sorted(np.unique(folds).tolist()):
        train = (folds != fold) & route_candidate & decisive
        held = (folds == fold) & route_candidate
        if len(np.unique(utility[train])) != 2:
            raise ValueError(f"fold {fold}: gate training has fewer than 2 classes")
        model = make_gate(seed + int(fold))
        model.fit(features[train], utility[train])
        scores[held] = model.predict_proba(features[held])[:, 1]
        transformed = model.named_steps["imputer"].transform(features[train])
        transformed = model.named_steps["scaler"].transform(transformed)
        classifier = model.named_steps["classifier"]
        coefficients.append(classifier.coef_[0].copy())
        audits.append({
            "fold": int(fold),
            "train_decisive": int(train.sum()),
            "train_helpful": int(utility[train].sum()),
            "train_harmful": int(train.sum() - utility[train].sum()),
            "held_route_candidates": int(held.sum()),
            "held_ranking": metric_or_none(
                utility[held & decisive], scores[held & decisive]
            ),
            "mean_transformed_feature": transformed.mean(axis=0).tolist(),
        })
    return scores, audits, coefficients


def evaluate(
    cases: list[dict], fold_ids: dict[int, set[str]],
    threshold: float = .5, bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> tuple[dict, list[dict]]:
    if threshold != .5:
        raise ValueError("B15 crossfit value threshold is frozen at 0.5")
    id_to_fold = {
        sample_id: fold for fold, ids in fold_ids.items() for sample_id in ids
    }
    case_ids = [row["id"] for row in cases]
    if set(case_ids) != set(id_to_fold) or len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs do not equal the disjoint fold union")
    labels = np.asarray([LABEL_IDS[row["gold"]] for row in cases])
    anchor = np.asarray([
        LABEL_IDS[row["predictions"]["anchor"]] for row in cases
    ])
    candidate = np.asarray([
        LABEL_IDS[row["predictions"]["candidate"]] for row in cases
    ])
    folds = np.asarray([id_to_fold[value] for value in case_ids])
    features = np.stack([observable_features(row) for row in cases])
    route_candidate = (anchor == 2) & (candidate != 2)
    anchor_correct = anchor == labels
    candidate_correct = candidate == labels
    helpful = ~anchor_correct & candidate_correct
    harmful = anchor_correct & ~candidate_correct
    decisive = helpful | harmful
    utility = helpful.astype(np.int64)
    scores, fold_audits, coefficients = crossfit_scores(
        features, utility, decisive, route_candidate, folds, bootstrap_seed
    )
    route = route_candidate & (scores >= threshold)
    prediction = anchor.copy()
    prediction[route] = candidate[route]
    anchor_probabilities = one_hot(anchor)
    routed_probabilities = one_hot(prediction)
    paired = paired_comparison(
        labels, anchor_probabilities, routed_probabilities,
        bootstrap_iterations, bootstrap_seed,
    )
    sources = np.asarray([str(row.get("source", "unknown")) for row in cases])
    source = source_diagnostics(
        labels, anchor_probabilities, routed_probabilities, sources
    )
    per_fold = []
    for fold in sorted(fold_ids):
        held = folds == fold
        fold_effect = paired_comparison(
            labels[held], anchor_probabilities[held],
            routed_probabilities[held], max(200, bootstrap_iterations // 5),
            bootstrap_seed + fold,
        )
        per_fold.append({
            "fold": fold,
            "samples": int(held.sum()),
            "route_count": int(route[held].sum()),
            "route_rate": float(route[held].mean()),
            **fold_effect,
        })
    decisive_routes = route_candidate & decisive
    ranking = metric_or_none(utility[decisive_routes], scores[decisive_routes])
    coefficient_matrix = np.stack(coefficients)
    coefficient_summary = sorted([
        {
            "feature": name,
            "mean": float(coefficient_matrix[:, index].mean()),
            "std": float(coefficient_matrix[:, index].std()),
        }
        for index, name in enumerate(FEATURE_NAMES)
    ], key=lambda row: abs(row["mean"]), reverse=True)
    deltas = np.asarray([row["macro_f1_delta"] for row in per_fold])
    gate = {
        "ranking_auroc_at_least_0_60": (
            ranking["auroc"] is not None and ranking["auroc"] >= .60
        ),
        "macro_f1_delta_at_least_0_005": paired["macro_f1_delta"] >= .005,
        "bootstrap_probability_at_least_0_95": (
            paired["bootstrap"]["probability_delta_positive"] >= .95
        ),
        "positive_folds_at_least_4": int(np.sum(deltas > 0)) >= 4,
        "both_sources_nonnegative": all(
            row["macro_f1_delta"] >= 0 for row in source.values()
        ),
        "help_exceeds_harm": paired["helpful"] > paired["harmful"],
    }
    gate["eligible"] = all(gate.values())
    result = {
        "protocol": "B15_post_failure_crossfit_value_of_information_gate",
        "samples": len(cases),
        "feature_names": list(FEATURE_NAMES),
        "prohibited_features": sorted(PROHIBITED_FEATURES),
        "threshold": threshold,
        "route_candidate_condition": (
            "anchor_prediction == nei and candidate_prediction != nei"
        ),
        "utility_target": "candidate_helpful_vs_candidate_harmful",
        "metrics": {
            "anchor": classification_metrics(labels, anchor_probabilities),
            "crossfit_value_gate": classification_metrics(
                labels, routed_probabilities
            ),
        },
        "comparison_vs_anchor": paired,
        "ranking_on_decisive_route_candidates": ranking,
        "routing": {
            "route_candidates": int(route_candidate.sum()),
            "selected_routes": int(route.sum()),
            "route_rate": float(route.mean()),
        },
        "per_fold": per_fold,
        "source_diagnostics": source,
        "fold_training_audit": fold_audits,
        "coefficient_summary": coefficient_summary,
        "preregistration_gate": gate,
        "interpretation": {
            "confirmatory_result": False,
            "fresh_fold_assignment_required": True,
        },
        "official_validation_used": False,
        "test_split_used": False,
    }
    scored_cases = []
    for index, row in enumerate(cases):
        scored_cases.append({
            "id": row["id"],
            "fold": int(folds[index]),
            "route_candidate": bool(route_candidate[index]),
            "decisive": bool(decisive[index]),
            "utility": (
                "helpful" if helpful[index] else
                "harmful" if harmful[index] else "neutral"
            ),
            "crossfit_value_score": float(scores[index]),
            "routed": bool(route[index]),
        })
    return result, scored_cases


def markdown(result: dict) -> str:
    anchor = result["metrics"]["anchor"]
    routed = result["metrics"]["crossfit_value_gate"]
    paired = result["comparison_vs_anchor"]
    ranking = result["ranking_on_decisive_route_candidates"]
    lines = [
        "# B15 cross-fitted value gate diagnostic", "",
        "Official validation used: **no**  ", "Test used: **no**", "",
        "| System | Accuracy | Macro-F1 |", "|---|---:|---:|",
        f"| anchor | {anchor['accuracy']:.4f} | {anchor['macro_f1']:.4f} |",
        f"| value gate | {routed['accuracy']:.4f} | {routed['macro_f1']:.4f} |",
        "", f"- Macro-F1 delta: {paired['macro_f1_delta']:+.6f}",
        f"- Helpful / harmful: {paired['helpful']} / {paired['harmful']}",
        f"- Decisive AUROC: {ranking['auroc']}",
        f"- Selected route rate: {result['routing']['route_rate']:.4f}",
        "- Eligible for fresh-fold preregistration: "
        f"**{result['preregistration_gate']['eligible']}**", "",
        "This is exploratory cross-fitted development, not confirmation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path(
        "outputs/mocheg_b14_direct_curriculum_cases.jsonl"))
    parser.add_argument("--fold-spec", type=Path, default=Path(
        "data/processed/mocheg_b12_folds.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b15_crossfit_value_gate.json"))
    parser.add_argument("--markdown", type=Path, default=Path(
        "outputs/mocheg_b15_crossfit_value_gate.md"))
    parser.add_argument("--scored-cases", type=Path, default=Path(
        "outputs/mocheg_b15_crossfit_value_cases.jsonl"))
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    fold_payload = json.loads(args.fold_spec.read_text(encoding="utf-8"))
    if (
        int(fold_payload.get("seed", -1)) != 2027
        or fold_payload.get("validation_split_used") is not False
        or fold_payload.get("test_split_used") is not False
    ):
        raise ValueError("expected locked seed-2027 train-only folds")
    fold_ids = {
        int(row["fold"]): set(row["val_ids"])
        for row in fold_payload["folds"]
    }
    if set(fold_ids) != {0, 1, 2, 3, 4}:
        raise ValueError("fold specification must contain folds 0--4")
    result, scored = evaluate(
        read_jsonl(args.cases), fold_ids, args.threshold,
        args.bootstrap_iterations, args.bootstrap_seed,
    )
    result["fold_spec_sha256"] = sha256(args.fold_spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(result), encoding="utf-8")
    args.scored_cases.parent.mkdir(parents=True, exist_ok=True)
    args.scored_cases.write_text(
        "\n".join(json.dumps(row) for row in scored) + "\n",
        encoding="utf-8",
    )
    print(markdown(result))


if __name__ == "__main__":
    main()
