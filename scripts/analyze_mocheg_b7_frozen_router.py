"""Train-fold development screen for a frozen-anchor selective router.

No model is trained and the anchor is never modified.  A small, preregistered
grid selects when to replace the anchor probability with one already-computed
B6-C expert probability.  Fold 0 is development-only; a passing policy must be
frozen before confirmation on train folds 1--4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_b6_auxiliary_control import paired_comparison
from scripts.analyze_mocheg_b6c_oof_screen import validate_run
from scripts.analyze_mocheg_expert_complementarity import (
    prediction_metrics,
    read_predictions,
)
from scripts.run_mocheg_visual_retrieval import read_jsonl


ANCHOR_CONFIDENCE_MAX = (0.45, 0.55, 0.65, 0.75, 0.85, 1.01)
EXPERT_CONFIDENCE_MIN = (0.0, 0.50, 0.60, 0.70, 0.80)
CONFIDENCE_ADVANTAGE_MIN = (-0.20, -0.10, 0.0, 0.10, 0.20)
CLASS_MODES = (
    "any", "expert_supported", "expert_refuted", "expert_nei",
    "anchor_supported", "anchor_refuted", "anchor_nei",
)


def aligned_inputs(
    anchor_path: Path, expert_paths: dict[str, Path], manifest_path: Path
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    loaded = {
        "anchor": read_predictions(anchor_path),
        **{name: read_predictions(path) for name, path in expert_paths.items()},
    }
    reference = set(loaded["anchor"])
    if any(set(rows) != reference for rows in loaded.values()):
        raise ValueError("B7 prediction ID sets do not match")
    ids = sorted(reference)
    labels = np.asarray(
        [int(loaded["anchor"][sample_id]["gold"]) for sample_id in ids]
    )
    probabilities = {}
    for name, rows in loaded.items():
        observed = np.asarray([int(rows[value]["gold"]) for value in ids])
        if not np.array_equal(labels, observed):
            raise ValueError(f"B7 gold labels do not match for {name}")
        values = np.asarray(
            [rows[value]["probabilities"] for value in ids], dtype=np.float64
        )
        if values.shape != (len(ids), 3) or not np.isfinite(values).all():
            raise ValueError(f"invalid probability matrix for {name}")
        probabilities[name] = values / np.clip(
            values.sum(1, keepdims=True), 1e-12, None
        )
    manifest = {row["id"]: row for row in read_jsonl(manifest_path)}
    missing = reference - set(manifest)
    if missing:
        raise ValueError(f"B7 IDs missing from manifest: {len(missing)}")
    sources = np.asarray([
        str(manifest[value].get("source", "unknown") or "unknown")
        for value in ids
    ])
    return (
        ids, labels, probabilities.pop("anchor"), probabilities, sources
    )


def class_route_mask(
    mode: str, anchor_prediction: np.ndarray, expert_prediction: np.ndarray
) -> np.ndarray:
    if mode == "any":
        return np.ones(len(anchor_prediction), dtype=bool)
    owner, class_name = mode.split("_", 1)
    class_id = {"supported": 0, "refuted": 1, "nei": 2}[class_name]
    prediction = anchor_prediction if owner == "anchor" else expert_prediction
    return prediction == class_id


def route_mask(
    anchor: np.ndarray, expert: np.ndarray, anchor_confidence_max: float,
    expert_confidence_min: float, confidence_advantage_min: float,
    class_mode: str,
) -> np.ndarray:
    anchor_prediction = anchor.argmax(-1)
    expert_prediction = expert.argmax(-1)
    anchor_confidence = anchor.max(-1)
    expert_confidence = expert.max(-1)
    return (
        (anchor_prediction != expert_prediction)
        & (anchor_confidence <= anchor_confidence_max)
        & (expert_confidence >= expert_confidence_min)
        & ((expert_confidence - anchor_confidence) >= confidence_advantage_min)
        & class_route_mask(class_mode, anchor_prediction, expert_prediction)
    )


def policy_metrics(
    labels: np.ndarray, anchor: np.ndarray, expert: np.ndarray,
    route: np.ndarray,
) -> tuple[dict, np.ndarray]:
    probability = np.where(route[:, None], expert, anchor)
    metrics = prediction_metrics(labels, probability)
    anchor_correct = anchor.argmax(-1) == labels
    routed_correct = probability.argmax(-1) == labels
    metrics.update({
        "route_count": int(route.sum()),
        "route_rate": float(route.mean()),
        "helpful": int(np.sum(~anchor_correct & routed_correct)),
        "harmful": int(np.sum(anchor_correct & ~routed_correct)),
    })
    return metrics, probability


def source_diagnostics(
    labels: np.ndarray, anchor: np.ndarray, routed: np.ndarray,
    sources: np.ndarray,
) -> dict:
    result = {}
    for source in sorted(set(sources.tolist())):
        selected = sources == source
        anchor_metrics = prediction_metrics(labels[selected], anchor[selected])
        routed_metrics = prediction_metrics(labels[selected], routed[selected])
        result[source] = {
            "samples": int(selected.sum()),
            "anchor_macro_f1": anchor_metrics["macro_f1"],
            "routed_macro_f1": routed_metrics["macro_f1"],
            "macro_f1_delta": (
                routed_metrics["macro_f1"] - anchor_metrics["macro_f1"]
            ),
        }
    return result


def screen(
    labels: np.ndarray,
    anchor: np.ndarray,
    experts: dict[str, np.ndarray],
    sources: np.ndarray,
    minimum_route_rate: float = .01,
    maximum_route_rate: float = .25,
    minimum_macro_f1_delta: float = .005,
    minimum_source_delta: float = -.002,
    minimum_bootstrap_probability: float = .95,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> dict:
    if not 0 <= minimum_route_rate <= maximum_route_rate <= 1:
        raise ValueError("invalid B7 route-rate bounds")
    anchor_metrics = prediction_metrics(labels, anchor)
    candidates = []
    raw_experts = {
        name: prediction_metrics(labels, probability)
        for name, probability in experts.items()
    }
    for expert_name, expert in experts.items():
        for anchor_max in ANCHOR_CONFIDENCE_MAX:
            for expert_min in EXPERT_CONFIDENCE_MIN:
                for advantage in CONFIDENCE_ADVANTAGE_MIN:
                    for class_mode in CLASS_MODES:
                        route = route_mask(
                            anchor, expert, anchor_max, expert_min,
                            advantage, class_mode,
                        )
                        rate = float(route.mean())
                        if not minimum_route_rate <= rate <= maximum_route_rate:
                            continue
                        metrics, probability = policy_metrics(
                            labels, anchor, expert, route
                        )
                        candidates.append({
                            "expert": expert_name,
                            "anchor_confidence_max": anchor_max,
                            "expert_confidence_min": expert_min,
                            "confidence_advantage_min": advantage,
                            "class_mode": class_mode,
                            "metrics": metrics,
                            "probabilities": probability,
                            "route": route,
                        })
    if not candidates:
        raise ValueError("B7 policy grid produced no eligible route")
    best = max(candidates, key=lambda row: (
        row["metrics"]["macro_f1"], row["metrics"]["accuracy"],
        -row["metrics"]["route_rate"], row["expert"], row["class_mode"],
    ))
    routed = best["probabilities"]
    comparison = paired_comparison(
        labels, anchor, routed, bootstrap_iterations, bootstrap_seed
    )
    by_source = source_diagnostics(labels, anchor, routed, sources)
    all_sources_safe = all(
        row["macro_f1_delta"] >= minimum_source_delta
        for row in by_source.values()
    )
    gate = {
        "macro_f1_delta_at_least_minimum": (
            comparison["macro_f1_delta"] >= minimum_macro_f1_delta
        ),
        "bootstrap_probability_at_least_minimum": (
            comparison["bootstrap"]["probability_delta_positive"]
            >= minimum_bootstrap_probability
        ),
        "selected_help_exceeds_harm": (
            comparison["helpful"] > comparison["harmful"]
        ),
        "all_sources_within_noninferiority_margin": all_sources_safe,
        "route_rate_within_bounds": (
            minimum_route_rate <= best["metrics"]["route_rate"]
            <= maximum_route_rate
        ),
    }
    gate["passed"] = all(gate.values())
    anchor_prediction = anchor.argmax(-1)
    oracle = {}
    for name, expert in experts.items():
        expert_prediction = expert.argmax(-1)
        prediction = anchor_prediction.copy()
        helpful = ((anchor_prediction != labels)
                   & (expert_prediction == labels))
        prediction[helpful] = expert_prediction[helpful]
        oracle[name] = prediction_metrics(labels, np.eye(3)[prediction])
    return {
        "protocol": "B7_train_only_fold0_frozen_anchor_router_screen",
        "samples": int(len(labels)),
        "anchor": anchor_metrics,
        "raw_experts": raw_experts,
        "eligible_policy_count": len(candidates),
        "selected_policy": {
            key: best[key] for key in (
                "expert", "anchor_confidence_max", "expert_confidence_min",
                "confidence_advantage_min", "class_mode",
            )
        },
        "selected_metrics": best["metrics"],
        "comparison_vs_anchor": comparison,
        "source_diagnostics": by_source,
        "oracle_expert_selection": oracle,
        "promotion_gate": gate,
        "settings": {
            "minimum_route_rate": minimum_route_rate,
            "maximum_route_rate": maximum_route_rate,
            "minimum_macro_f1_delta": minimum_macro_f1_delta,
            "minimum_source_delta": minimum_source_delta,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
            "anchor_confidence_max_grid": ANCHOR_CONFIDENCE_MAX,
            "expert_confidence_min_grid": EXPERT_CONFIDENCE_MIN,
            "confidence_advantage_min_grid": CONFIDENCE_ADVANTAGE_MIN,
            "class_modes": CLASS_MODES,
        },
        "warning": (
            "Fold 0 selected this policy and is development evidence only. "
            "Freeze the exact rule before train-fold confirmation."
        ),
        "official_validation_used": False,
        "test_split_used": False,
        "confirmation_required_on_folds": [1, 2, 3, 4],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("outputs/mocheg_b6c_oof/fold_0")
    parser.add_argument("--anchor", type=Path, default=root / "anchor")
    parser.add_argument(
        "--expert", action="append", default=None,
        help="Repeat NAME=OUTPUT_DIR; defaults to all B6-C continuations",
    )
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--minimum-route-rate", type=float, default=.01)
    parser.add_argument("--maximum-route-rate", type=float, default=.25)
    parser.add_argument("--minimum-macro-f1-delta", type=float, default=.005)
    parser.add_argument("--minimum-source-delta", type=float, default=-.002)
    parser.add_argument("--minimum-bootstrap-probability", type=float,
                        default=.95)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b7_fold0_router_screen.json"))
    args = parser.parse_args()
    if args.fold != 0:
        raise ValueError("B7 policy selection is restricted to fold 0")
    default_experts = {
        "direct_control": root / "direct_control",
        "standard_auxiliary": root / "standard_auxiliary",
        "soft_025": root / "soft_025",
        "soft_050": root / "soft_050",
        "severity_010": root / "severity_010",
    }
    if args.expert:
        default_experts = {}
        for value in args.expert:
            if "=" not in value:
                raise ValueError("--expert must be NAME=OUTPUT_DIR")
            name, path = value.split("=", 1)
            default_experts[name] = Path(path)
    validate_run(args.anchor, args.fold, "anchor")
    for name, path in default_experts.items():
        validate_run(path, args.fold, name)
    _, labels, anchor, experts, sources = aligned_inputs(
        args.anchor / "val_predictions.jsonl",
        {name: path / "val_predictions.jsonl"
         for name, path in default_experts.items()},
        args.manifest,
    )
    result = screen(
        labels, anchor, experts, sources,
        args.minimum_route_rate, args.maximum_route_rate,
        args.minimum_macro_f1_delta, args.minimum_source_delta,
        args.minimum_bootstrap_probability, args.bootstrap_iterations,
        args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
