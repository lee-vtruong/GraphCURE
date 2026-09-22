"""Cross-fitted expected-utility router trained only where B18-B experts disagree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.summarize_mocheg_b18_seeds import compute_metrics, load_seed_predictions
from scripts.train_mocheg_b18c_crossfit_router import build_features, stable_fold


HARMFUL = 0
NEUTRAL = 1
HELPFUL = 2


def utility_scores(model, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.named_steps["logisticregression"].classes_)
    helpful = probabilities[:, classes.index(HELPFUL)]
    harmful = probabilities[:, classes.index(HARMFUL)]
    return helpful - harmful


def choose_utility_threshold(
    labels: np.ndarray,
    top3: np.ndarray,
    distilled: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    minimum_route_rate: float,
    maximum_route_rate: float,
) -> dict[str, float]:
    disagreement = top3 != distilled
    candidates: list[dict[str, float]] = []
    for threshold in thresholds:
        route = disagreement & (scores >= threshold)
        route_rate = float(np.mean(route))
        if not minimum_route_rate <= route_rate <= maximum_route_rate:
            continue
        prediction = np.where(route, distilled, top3)
        candidates.append({
            "threshold": float(threshold),
            "macro_f1": float(f1_score(labels, prediction, average="macro")),
            "accuracy": float(np.mean(prediction == labels)),
            "route_rate": route_rate,
        })
    if not candidates:
        return {
            "threshold": 1.0,
            "macro_f1": float(f1_score(labels, top3, average="macro")),
            "accuracy": float(np.mean(top3 == labels)),
            "route_rate": 0.0,
        }
    return max(
        candidates,
        key=lambda row: (row["macro_f1"], row["accuracy"], -row["route_rate"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-top3", type=Path, required=True)
    parser.add_argument("--distilled", type=Path, required=True)
    parser.add_argument("--distilled-retrieval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-route-rate", type=float, default=0.005)
    parser.add_argument("--maximum-route-rate", type=float, default=0.15)
    parser.add_argument("--minimum-delta", type=float, default=0.005)
    parser.add_argument("--minimum-bootstrap-probability", type=float, default=0.95)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top3_rows = load_seed_predictions(args.retrieval_top3)
    distilled_rows = load_seed_predictions(args.distilled)
    selected_rows = {str(row["id"]): row for row in read_jsonl(args.distilled_retrieval)}
    manifest_rows = {str(row["id"]): row for row in read_jsonl(args.manifest)}
    ids = sorted(set(top3_rows) & set(distilled_rows) & set(selected_rows))
    labels = np.asarray([int(top3_rows[claim_id]["label"]) for claim_id in ids])
    if labels.size == 0:
        raise ValueError("no common claims")
    if not np.array_equal(
        labels, np.asarray([int(distilled_rows[claim_id]["label"]) for claim_id in ids])
    ):
        raise ValueError("label mismatch between experts")
    top3 = np.asarray([int(top3_rows[claim_id]["prediction"]) for claim_id in ids])
    distilled = np.asarray([int(distilled_rows[claim_id]["prediction"]) for claim_id in ids])
    features, feature_names = build_features(
        ids, top3_rows, distilled_rows, selected_rows, manifest_rows
    )
    disagreement_column = feature_names.index("experts_disagree")
    features = np.delete(features, disagreement_column, axis=1)
    feature_names.pop(disagreement_column)

    disagreement = top3 != distilled
    outcomes = np.full(len(ids), NEUTRAL, dtype=np.int64)
    outcomes[(top3 == labels) & (distilled != labels)] = HARMFUL
    outcomes[(top3 != labels) & (distilled == labels)] = HELPFUL
    fold_ids = np.asarray([stable_fold(claim_id, args.folds) for claim_id in ids])
    oof_scores = np.full(len(ids), -1.0, dtype=np.float64)
    oof_predictions = top3.copy()
    per_fold: list[dict] = []
    coefficient_rows: list[np.ndarray] = []

    for fold in range(args.folds):
        train = (fold_ids != fold) & disagreement
        heldout = fold_ids == fold
        if set(np.unique(outcomes[train])) != {HARMFUL, NEUTRAL, HELPFUL}:
            raise ValueError(f"fold {fold} training partition lacks a utility outcome")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=args.seed + fold,
            ),
        )
        model.fit(features[train], outcomes[train])
        train_scores = utility_scores(model, features[fold_ids != fold])
        train_disagreement_scores = train_scores[disagreement[fold_ids != fold]]
        thresholds = np.unique(np.quantile(train_disagreement_scores, np.linspace(0, 1, 101)))
        selected = choose_utility_threshold(
            labels[fold_ids != fold],
            top3[fold_ids != fold],
            distilled[fold_ids != fold],
            train_scores,
            thresholds,
            args.minimum_route_rate,
            args.maximum_route_rate,
        )
        heldout_scores = utility_scores(model, features[heldout])
        route = disagreement[heldout] & (heldout_scores >= selected["threshold"])
        oof_scores[heldout] = heldout_scores
        oof_predictions[heldout] = np.where(route, distilled[heldout], top3[heldout])
        reference = compute_metrics(labels[heldout], top3[heldout])
        candidate = compute_metrics(labels[heldout], oof_predictions[heldout])
        helpful = int(np.sum((top3[heldout] != labels[heldout]) &
                             (oof_predictions[heldout] == labels[heldout])))
        harmful = int(np.sum((top3[heldout] == labels[heldout]) &
                             (oof_predictions[heldout] != labels[heldout])))
        per_fold.append({
            "fold": fold,
            "samples": int(np.sum(heldout)),
            "disagreements": int(np.sum(disagreement[heldout])),
            "threshold": selected["threshold"],
            "train_selected_policy": selected,
            "route_count": int(np.sum(route)),
            "route_rate": float(np.mean(route)),
            "top3_macro_f1": reference["macro_f1"],
            "router_macro_f1": candidate["macro_f1"],
            "macro_f1_delta": candidate["macro_f1"] - reference["macro_f1"],
            "helpful": helpful,
            "harmful": harmful,
        })
        logistic = model.named_steps["logisticregression"]
        scaler = model.named_steps["standardscaler"]
        class_rows = {int(label): row for label, row in zip(logistic.classes_, logistic.coef_)}
        coefficient_rows.append(
            (class_rows[HELPFUL] - class_rows[HARMFUL]) / scaler.scale_
        )

    top3_metrics = compute_metrics(labels, top3)
    distilled_metrics = compute_metrics(labels, distilled)
    router_metrics = compute_metrics(labels, oof_predictions)
    helpful = int(np.sum((top3 != labels) & (oof_predictions == labels)))
    harmful = int(np.sum((top3 == labels) & (oof_predictions != labels)))
    bootstrap = bootstrap_delta(
        labels, top3, oof_predictions, args.bootstrap_iterations, args.seed
    )

    decisive = disagreement & (outcomes != NEUTRAL)
    decisive_target = (outcomes[decisive] == HELPFUL).astype(int)
    disagreement_target = (outcomes[disagreement] == HELPFUL).astype(int)
    ranking = {
        "disagreement_samples": int(np.sum(disagreement)),
        "decisive_samples": int(np.sum(decisive)),
        "helpful_vs_all_disagreements_auroc": float(
            roc_auc_score(disagreement_target, oof_scores[disagreement])
        ),
        "helpful_vs_all_disagreements_average_precision": float(
            average_precision_score(disagreement_target, oof_scores[disagreement])
        ),
        "helpful_vs_harmful_auroc": float(
            roc_auc_score(decisive_target, oof_scores[decisive])
        ),
        "helpful_vs_harmful_average_precision": float(
            average_precision_score(decisive_target, oof_scores[decisive])
        ),
    }
    sources = np.asarray([
        str(manifest_rows.get(claim_id, {}).get("source", "unknown")) for claim_id in ids
    ])
    source_diagnostics = {}
    for source in sorted(set(sources.tolist())):
        selected = sources == source
        reference = compute_metrics(labels[selected], top3[selected])
        candidate = compute_metrics(labels[selected], oof_predictions[selected])
        source_diagnostics[source] = {
            "samples": int(np.sum(selected)),
            "top3_macro_f1": reference["macro_f1"],
            "router_macro_f1": candidate["macro_f1"],
            "macro_f1_delta": candidate["macro_f1"] - reference["macro_f1"],
        }
    fold_deltas = [row["macro_f1_delta"] for row in per_fold]
    gate = {
        "aggregate_delta_at_least_minimum": (
            router_metrics["macro_f1"] - top3_metrics["macro_f1"] >= args.minimum_delta
        ),
        "positive_folds_at_least_4": sum(value > 0 for value in fold_deltas) >= 4,
        "bootstrap_probability_at_least_minimum": (
            bootstrap["probability_delta_positive"] >= args.minimum_bootstrap_probability
        ),
        "help_exceeds_harm": helpful > harmful,
        "all_sources_nonnegative": all(
            row["macro_f1_delta"] >= 0 for row in source_diagnostics.values()
        ),
    }
    gate["passed"] = all(gate.values())
    coefficients = np.mean(np.asarray(coefficient_rows), axis=0)
    top_coefficients = sorted(
        ({"feature": name, "utility_coefficient": float(value)}
         for name, value in zip(feature_names, coefficients)),
        key=lambda row: abs(row["utility_coefficient"]),
        reverse=True,
    )[:15]
    payload = {
        "phase": "B18-C2",
        "protocol": "fold0_internal_5fold_disagreement_expected_utility_router",
        "samples": len(ids),
        "metrics": {
            "retrieval_top3": top3_metrics,
            "distilled": distilled_metrics,
            "crossfit_utility_router": router_metrics,
        },
        "comparison_vs_top3": {
            "macro_f1_delta": router_metrics["macro_f1"] - top3_metrics["macro_f1"],
            "accuracy_delta": router_metrics["accuracy"] - top3_metrics["accuracy"],
            "helpful": helpful,
            "harmful": harmful,
            "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
            "bootstrap": bootstrap,
        },
        "conditional_value_ranking": ranking,
        "per_fold": per_fold,
        "source_diagnostics": source_diagnostics,
        "top_utility_coefficients": top_coefficients,
        "promotion_gate": gate,
        "audit": {
            "training_population": "expert_disagreements_only",
            "router_outcomes": ["harmful", "both_wrong", "helpful"],
            "router_labels_are_fold_disjoint": True,
            "gold_used_for_router_training": True,
            "gold_used_for_expert_inference": False,
            "diagnostic_development_screen": True,
            "official_validation_used": False,
            "test_split_used": False,
            "fresh_fold_confirmation_required_if_passed": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MOCHEG B18-C2 disagreement utility-router screen",
        "",
        "Official validation used: **no**  ",
        "Test used: **no**",
        "",
        f"- Top-3 Macro-F1: {top3_metrics['macro_f1']:.6f}",
        f"- Utility-router Macro-F1: {router_metrics['macro_f1']:.6f}",
        f"- Delta: {router_metrics['macro_f1'] - top3_metrics['macro_f1']:+.6f}",
        f"- Helpful/harmful: {helpful}/{harmful}",
        f"- Bootstrap P(delta > 0): {bootstrap['probability_delta_positive']:.4f}",
        f"- Promotion gate: **{'pass' if gate['passed'] else 'fail'}**",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
