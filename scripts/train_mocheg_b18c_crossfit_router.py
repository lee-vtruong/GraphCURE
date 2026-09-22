"""Train-only cross-fitted value router for Top-3 and distilled B18-B experts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.summarize_mocheg_b18_seeds import compute_metrics, load_seed_predictions


def entropy(probabilities: np.ndarray) -> float:
    values = np.clip(probabilities.astype(float), 1e-9, 1.0)
    return float(-np.sum(values * np.log(values)))


def probability_margin(probabilities: np.ndarray) -> float:
    ordered = np.sort(probabilities.astype(float))
    return float(ordered[-1] - ordered[-2])


def stable_fold(claim_id: str, folds: int) -> int:
    digest = hashlib.sha256(claim_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def build_features(
    ids: list[str],
    top3_rows: dict[str, dict[str, Any]],
    distilled_rows: dict[str, dict[str, Any]],
    selected_rows: dict[str, dict[str, Any]],
    manifest_rows: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    names = [
        "top3_p_supported", "top3_p_refuted", "top3_p_nei",
        "distilled_p_supported", "distilled_p_refuted", "distilled_p_nei",
        "top3_confidence", "distilled_confidence", "confidence_advantage",
        "top3_margin", "distilled_margin", "top3_entropy", "distilled_entropy",
        "experts_disagree", "top3_predicts_supported", "top3_predicts_refuted",
        "top3_predicts_nei", "distilled_predicts_supported",
        "distilled_predicts_refuted", "distilled_predicts_nei",
        "source_politifact", "source_snopes", "selected_k_fraction",
        "selector_top_score", "selector_score_gap",
    ]
    rows: list[list[float]] = []
    for claim_id in ids:
        top3_probs = np.asarray(top3_rows[claim_id]["probabilities"], dtype=float)
        distilled_probs = np.asarray(distilled_rows[claim_id]["probabilities"], dtype=float)
        top3_prediction = int(top3_rows[claim_id]["prediction"])
        distilled_prediction = int(distilled_rows[claim_id]["prediction"])
        selected = selected_rows.get(claim_id, {})
        scores = [float(value) for value in selected.get("retrieved_scores", [])]
        selector_top = scores[0] if scores else 0.0
        selector_gap = scores[0] - scores[1] if len(scores) > 1 else 0.0
        source = str(manifest_rows.get(claim_id, {}).get("source", "")).lower()
        top3_confidence = float(top3_probs.max())
        distilled_confidence = float(distilled_probs.max())
        rows.append([
            *top3_probs.tolist(),
            *distilled_probs.tolist(),
            top3_confidence,
            distilled_confidence,
            distilled_confidence - top3_confidence,
            probability_margin(top3_probs),
            probability_margin(distilled_probs),
            entropy(top3_probs),
            entropy(distilled_probs),
            float(top3_prediction != distilled_prediction),
            *[float(top3_prediction == label) for label in range(3)],
            *[float(distilled_prediction == label) for label in range(3)],
            float(source == "politifact"),
            float(source == "snopes"),
            float(selected.get("selected_k", 0)) / 3.0,
            selector_top,
            selector_gap,
        ])
    return np.asarray(rows, dtype=np.float64), names


def choose_threshold(
    labels: np.ndarray,
    top3: np.ndarray,
    distilled: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    minimum_route_rate: float,
    maximum_route_rate: float,
) -> dict[str, float]:
    candidates = []
    disagreement = top3 != distilled
    for threshold in thresholds:
        route = (scores >= threshold) & disagreement
        route_rate = float(np.mean(route))
        if route_rate < minimum_route_rate or route_rate > maximum_route_rate:
            continue
        routed = np.where(route, distilled, top3)
        candidates.append({
            "threshold": float(threshold),
            "macro_f1": float(f1_score(labels, routed, average="macro")),
            "accuracy": float(np.mean(routed == labels)),
            "route_rate": route_rate,
        })
    if not candidates:
        return {"threshold": 1.0, "macro_f1": float(f1_score(labels, top3, average="macro")),
                "accuracy": float(np.mean(top3 == labels)), "route_rate": 0.0}
    return max(candidates, key=lambda row: (row["macro_f1"], row["accuracy"], -row["route_rate"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-top3", type=Path, required=True)
    parser.add_argument("--distilled", type=Path, required=True)
    parser.add_argument("--distilled-retrieval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-route-rate", type=float, default=0.01)
    parser.add_argument("--maximum-route-rate", type=float, default=0.25)
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
    distilled_labels = np.asarray([int(distilled_rows[claim_id]["label"]) for claim_id in ids])
    if not np.array_equal(labels, distilled_labels):
        raise ValueError("label mismatch between experts")
    top3 = np.asarray([int(top3_rows[claim_id]["prediction"]) for claim_id in ids])
    distilled = np.asarray([int(distilled_rows[claim_id]["prediction"]) for claim_id in ids])
    features, feature_names = build_features(
        ids, top3_rows, distilled_rows, selected_rows, manifest_rows
    )
    fold_ids = np.asarray([stable_fold(claim_id, args.folds) for claim_id in ids])
    helpful_target = ((top3 != labels) & (distilled == labels)).astype(int)
    harmful = (top3 == labels) & (distilled != labels)
    neutral = ~(helpful_target.astype(bool) | harmful)
    sample_weights = np.where(neutral, 0.1, 1.0)
    thresholds = np.linspace(0.05, 0.95, 91)

    oof_scores = np.zeros(len(ids), dtype=np.float64)
    oof_predictions = top3.copy()
    per_fold = []
    coefficients = []
    for fold in range(args.folds):
        train = fold_ids != fold
        heldout = fold_ids == fold
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=args.seed + fold),
        )
        model.fit(features[train], helpful_target[train],
                  logisticregression__sample_weight=sample_weights[train])
        train_scores = model.predict_proba(features[train])[:, 1]
        selected_threshold = choose_threshold(
            labels[train], top3[train], distilled[train], train_scores,
            thresholds, args.minimum_route_rate, args.maximum_route_rate,
        )
        heldout_scores = model.predict_proba(features[heldout])[:, 1]
        route = (
            (heldout_scores >= selected_threshold["threshold"])
            & (top3[heldout] != distilled[heldout])
        )
        oof_scores[heldout] = heldout_scores
        oof_predictions[heldout] = np.where(route, distilled[heldout], top3[heldout])
        heldout_metrics = compute_metrics(labels[heldout], oof_predictions[heldout])
        top3_metrics = compute_metrics(labels[heldout], top3[heldout])
        helpful = int(np.sum((top3[heldout] != labels[heldout]) &
                             (oof_predictions[heldout] == labels[heldout])))
        harm = int(np.sum((top3[heldout] == labels[heldout]) &
                          (oof_predictions[heldout] != labels[heldout])))
        per_fold.append({
            "fold": fold,
            "samples": int(np.sum(heldout)),
            "threshold": selected_threshold["threshold"],
            "train_selected_policy": selected_threshold,
            "route_count": int(np.sum(route)),
            "route_rate": float(np.mean(route)),
            "top3_macro_f1": top3_metrics["macro_f1"],
            "router_macro_f1": heldout_metrics["macro_f1"],
            "macro_f1_delta": heldout_metrics["macro_f1"] - top3_metrics["macro_f1"],
            "helpful": helpful,
            "harmful": harm,
        })
        logistic = model.named_steps["logisticregression"]
        scaler = model.named_steps["standardscaler"]
        coefficients.append((logistic.coef_[0] / scaler.scale_).tolist())

    top3_metrics = compute_metrics(labels, top3)
    distilled_metrics = compute_metrics(labels, distilled)
    router_metrics = compute_metrics(labels, oof_predictions)
    helpful = int(np.sum((top3 != labels) & (oof_predictions == labels)))
    harm = int(np.sum((top3 == labels) & (oof_predictions != labels)))
    bootstrap = bootstrap_delta(
        labels, top3, oof_predictions,
        iterations=args.bootstrap_iterations, seed=args.seed,
    )
    source_diagnostics = {}
    sources = [str(manifest_rows.get(claim_id, {}).get("source", "unknown")) for claim_id in ids]
    for source in sorted(set(sources)):
        indices = np.asarray([i for i, value in enumerate(sources) if value == source])
        ref = compute_metrics(labels[indices], top3[indices])
        routed = compute_metrics(labels[indices], oof_predictions[indices])
        source_diagnostics[source] = {
            "samples": len(indices),
            "top3_macro_f1": ref["macro_f1"],
            "router_macro_f1": routed["macro_f1"],
            "macro_f1_delta": routed["macro_f1"] - ref["macro_f1"],
        }
    fold_deltas = [row["macro_f1_delta"] for row in per_fold]
    gate = {
        "aggregate_delta_at_least_minimum": (
            router_metrics["macro_f1"] - top3_metrics["macro_f1"] >= args.minimum_delta
        ),
        "positive_folds_at_least_4": sum(delta > 0 for delta in fold_deltas) >= 4,
        "bootstrap_probability_at_least_minimum": (
            bootstrap["probability_delta_positive"] >= args.minimum_bootstrap_probability
        ),
        "help_exceeds_harm": helpful > harm,
        "all_sources_nonnegative": all(
            row["macro_f1_delta"] >= 0.0 for row in source_diagnostics.values()
        ),
    }
    gate["passed"] = all(gate.values())
    ranking = {
        "auroc": float(roc_auc_score(helpful_target, oof_scores)),
        "average_precision": float(average_precision_score(helpful_target, oof_scores)),
        "helpful_rate": float(np.mean(helpful_target)),
    }
    mean_coefficients = np.mean(np.asarray(coefficients), axis=0)
    coefficient_rows = sorted(
        ({"feature": name, "mean_coefficient": float(value)}
         for name, value in zip(feature_names, mean_coefficients)),
        key=lambda row: abs(row["mean_coefficient"]), reverse=True,
    )
    payload = {
        "phase": "B18-C",
        "protocol": "fold0_internal_5fold_crossfit_value_router_screen",
        "samples": len(ids),
        "metrics": {
            "retrieval_top3": top3_metrics,
            "distilled": distilled_metrics,
            "crossfit_router": router_metrics,
        },
        "comparison_vs_top3": {
            "macro_f1_delta": router_metrics["macro_f1"] - top3_metrics["macro_f1"],
            "accuracy_delta": router_metrics["accuracy"] - top3_metrics["accuracy"],
            "helpful": helpful,
            "harmful": harm,
            "exact_mcnemar_p": exact_mcnemar_p(helpful, harm),
            "bootstrap": bootstrap,
        },
        "value_ranking": ranking,
        "per_fold": per_fold,
        "source_diagnostics": source_diagnostics,
        "top_coefficients": coefficient_rows[:15],
        "promotion_gate": gate,
        "audit": {
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
        "# MOCHEG B18-C cross-fitted value-router screen",
        "",
        "Official validation used: **no**  ",
        "Test used: **no**",
        "",
        f"- Top-3 Macro-F1: {top3_metrics['macro_f1']:.6f}",
        f"- Crossfit router Macro-F1: {router_metrics['macro_f1']:.6f}",
        f"- Delta: {router_metrics['macro_f1'] - top3_metrics['macro_f1']:+.6f}",
        f"- Helpful/harmful: {helpful}/{harm}",
        f"- Bootstrap P(delta > 0): {bootstrap['probability_delta_positive']:.4f}",
        f"- Promotion gate: **{'pass' if gate['passed'] else 'fail'}**",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
