"""Select one soft-conflict configuration on train-only development fold 0.

This is a development screen, not confirmatory evidence.  It refuses runs that
used official validation, selected an epoch from held-fold performance, or
read test.  The winning configuration must then be frozen and evaluated on
folds 1--4 by a separate confirmation step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_mocheg_b6_auxiliary_control import (
    aligned_arrays,
    paired_comparison,
)
from scripts.analyze_mocheg_expert_complementarity import prediction_metrics


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=OUTPUT_DIR")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("candidate must be NAME=OUTPUT_DIR")
    return name.strip(), Path(path.strip())


def validate_run(path: Path, fold: int, role: str) -> dict:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_split_used") is not False:
        raise ValueError(f"{role}: test split safety marker failed")
    if summary.get("official_validation_used_for_selection") is not False:
        raise ValueError(f"{role}: official validation was used")
    if summary.get("protocol") != "train_only_duplicate_safe_fixed_epoch_cv":
        raise ValueError(f"{role}: not a fixed-epoch train-only CV run")
    if int(summary.get("fold", -1)) != fold:
        raise ValueError(f"{role}: expected fold {fold}")
    fixed = summary.get("fixed_checkpoint_epoch")
    if not fixed or int(summary.get("best_epoch", -1)) != int(fixed):
        raise ValueError(f"{role}: checkpoint was not selected at fixed epoch")
    if not (path / "val_predictions.jsonl").exists():
        raise FileNotFoundError(path / "val_predictions.jsonl")
    return summary


def screen(
    anchor_dir: Path,
    control_dir: Path,
    standard_dir: Path,
    candidates: list[tuple[str, Path]],
    fold: int,
    minimum_control_delta: float,
    minimum_standard_delta: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict:
    if fold != 0:
        raise ValueError("B6-C configuration selection is restricted to fold 0")
    if len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    summaries = {
        "anchor": validate_run(anchor_dir, fold, "anchor"),
        "direct_control": validate_run(control_dir, fold, "direct_control"),
        "standard_auxiliary": validate_run(
            standard_dir, fold, "standard_auxiliary"
        ),
    }
    for name, path in candidates:
        summary = validate_run(path, fold, name)
        settings = summary.get("settings", {})
        if settings.get("gradient_mode") != "pcgrad":
            raise ValueError(f"{name}: candidate is not a PCGrad-family run")
        if float(summary.get("selected_hierarchical_weight", -1)) != 0:
            raise ValueError(f"{name}: hierarchical inference must be disabled")
        summaries[name] = summary

    paths = {
        "anchor": anchor_dir / "val_predictions.jsonl",
        "direct_control": control_dir / "val_predictions.jsonl",
        "standard_auxiliary": standard_dir / "val_predictions.jsonl",
        **{
            name: path / "val_predictions.jsonl" for name, path in candidates
        },
    }
    labels, probabilities = aligned_arrays(paths)
    metrics = {
        name: prediction_metrics(labels, values)
        for name, values in probabilities.items()
    }
    winner = max(
        (name for name, _ in candidates),
        key=lambda name: (metrics[name]["macro_f1"], name),
    )
    comparisons = {
        baseline: paired_comparison(
            labels, probabilities[baseline], probabilities[winner],
            bootstrap_iterations, bootstrap_seed,
        )
        for baseline in ("anchor", "direct_control", "standard_auxiliary")
    }
    history = summaries[winner].get("history", [])
    conflict_active = any(
        float(row.get("gradient_conflict_rate", 0)) > 0
        and float(row.get("applied_projection_strength_mean", 0)) > 0
        for row in history if int(row.get("epoch", 0)) > 0
    )
    gate = {
        "conflict_projection_active": conflict_active,
        "winner_beats_anchor": (
            comparisons["anchor"]["macro_f1_delta"] > 0
        ),
        "winner_delta_vs_control_at_least_minimum": (
            comparisons["direct_control"]["macro_f1_delta"]
            >= minimum_control_delta
        ),
        "winner_delta_vs_standard_at_least_minimum": (
            comparisons["standard_auxiliary"]["macro_f1_delta"]
            >= minimum_standard_delta
        ),
    }
    gate["passed"] = all(gate.values())
    winner_settings = summaries[winner]["settings"]
    return {
        "protocol": "B6C_train_only_fold0_soft_conflict_development_screen",
        "fold": fold,
        "samples": int(len(labels)),
        "candidate_names": [name for name, _ in candidates],
        "metrics": metrics,
        "selected_candidate": winner,
        "selected_configuration": {
            key: winner_settings.get(key) for key in (
                "gradient_mode", "projection_strength",
                "conflict_temperature", "auxiliary_gradient_scale",
                "fixed_checkpoint_epoch", "learning_rate",
            )
        },
        "winner_comparisons": comparisons,
        "promotion_gate": gate,
        "settings": {
            "minimum_control_delta": minimum_control_delta,
            "minimum_standard_delta": minimum_standard_delta,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "official_validation_used": False,
        "test_split_used": False,
        "confirmation_required_on_folds": [1, 2, 3, 4],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("outputs/mocheg_b6c_oof/fold_0")
    parser.add_argument("--anchor", type=Path, default=root / "anchor")
    parser.add_argument("--direct-control", type=Path,
                        default=root / "direct_control")
    parser.add_argument("--standard", type=Path,
                        default=root / "standard_auxiliary")
    parser.add_argument(
        "--candidate", action="append", type=parse_candidate,
        default=None, help="Repeat NAME=OUTPUT_DIR",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--minimum-control-delta", type=float, default=.003)
    parser.add_argument("--minimum-standard-delta", type=float, default=.002)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_b6c_fold0_screen.json"))
    args = parser.parse_args()
    candidates = args.candidate or [
        ("soft_025", root / "soft_025"),
        ("soft_050", root / "soft_050"),
        ("severity_010", root / "severity_010"),
    ]
    result = screen(
        args.anchor, args.direct_control, args.standard, candidates,
        args.fold, args.minimum_control_delta, args.minimum_standard_delta,
        args.bootstrap_iterations, args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
