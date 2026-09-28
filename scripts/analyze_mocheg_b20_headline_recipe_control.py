"""Analyze the B20 headline-recipe causal control on official validation.

The comparison is deliberately narrow: a verdict-only control is trained with
the exact optimizer/data recipe of the B18A rationale-trained runs.  It is not
an AND evaluation and it must not be used to choose a test-time policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.summarize_mocheg_b18_seeds import compute_metrics, load_seed_predictions


EXPECTED = {
    "mode": "matched_control",
    "lambda_exp": 0.0,
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation": 4,
    "learning_rate": 2e-4,
    "top_k": 5,
    "inject_train_gold": False,
    "explanations_used": False,
}


def load_summary(run: Path) -> dict:
    path = run / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_control(run: Path) -> dict:
    summary = load_summary(run)
    failures = {}
    for field, expected in EXPECTED.items():
        actual = summary.get(field)
        if isinstance(expected, float):
            if actual is None or not np.isclose(float(actual), expected):
                failures[field] = {"expected": expected, "actual": actual}
        elif actual != expected:
            failures[field] = {"expected": expected, "actual": actual}
    if failures:
        raise ValueError(f"{run}: control recipe audit failed: {failures}")
    return summary


def average(runs: list[Path]) -> tuple[list[str], np.ndarray, np.ndarray]:
    loaded = [load_seed_predictions(run) for run in runs]
    common = set(loaded[0])
    for rows in loaded[1:]:
        common &= set(rows)
    ids = sorted(common)
    labels = np.asarray([int(loaded[0][cid]["label"]) for cid in ids])
    for rows in loaded[1:]:
        other = np.asarray([int(rows[cid]["label"]) for cid in ids])
        if not np.array_equal(labels, other):
            raise ValueError("Seed labels are not aligned")
    probabilities = np.mean(
        [np.asarray([rows[cid]["probabilities"] for cid in ids], dtype=np.float64) for rows in loaded],
        axis=0,
    )
    return ids, labels, probabilities


def comparison(labels: np.ndarray, base: np.ndarray, candidate: np.ndarray, iterations: int, seed: int) -> dict:
    base_pred, candidate_pred = base.argmax(axis=1), candidate.argmax(axis=1)
    base_metrics, candidate_metrics = compute_metrics(labels, base_pred), compute_metrics(labels, candidate_pred)
    helpful = int(np.sum((base_pred != labels) & (candidate_pred == labels)))
    harmful = int(np.sum((base_pred == labels) & (candidate_pred != labels)))
    bootstrap = bootstrap_delta(labels, base_pred, candidate_pred, iterations=iterations, seed=seed)
    return {
        "baseline": base_metrics,
        "candidate": candidate_metrics,
        "macro_f1_delta": candidate_metrics["macro_f1"] - base_metrics["macro_f1"],
        "accuracy_delta": candidate_metrics["accuracy"] - base_metrics["accuracy"],
        "class_f1_delta": {
            name: candidate_metrics[name] - base_metrics[name]
            for name in ("f1_supported", "f1_refuted", "f1_nei")
        },
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--rationale-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if len(args.control_runs) != len(args.rationale_runs):
        raise ValueError("Control and rationale runs must have the same seed count")

    audited = [verify_control(run) for run in args.control_runs]
    control_ids, labels, control = average(args.control_runs)
    rationale_ids, rationale_labels, rationale = average(args.rationale_runs)
    if control_ids != rationale_ids or not np.array_equal(labels, rationale_labels):
        raise ValueError("Control and rationale ensembles are not exactly aligned")

    result = {
        "protocol": "B20_headline_recipe_direct_only_causal_control",
        "split": "official_validation",
        "official_validation_used_for_evaluation": True,
        "official_validation_used_for_checkpoint_selection": True,
        "official_validation_used_for_router_policy_selection": False,
        "test_split_used": False,
        "samples": len(control_ids),
        "seeds": [summary["seed"] for summary in audited],
        "control_recipe_audit": EXPECTED,
        "control_summaries": [str(run / "summary.json") for run in args.control_runs],
        "rationale_runs": [str(run) for run in args.rationale_runs],
        "rationale_vs_headline_recipe_direct": comparison(labels, control, rationale, args.iterations, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    effect = result["rationale_vs_headline_recipe_direct"]
    boot = effect["bootstrap"]
    ci = boot["ci_95_percentile"]
    lines = [
        "# MOCHEG B20 headline-recipe direct-only causal control",
        "",
        "Official validation used for evaluation: **yes**  ",
        "Official validation used for checkpoint selection: **yes** (matched for both arms)  ",
        "Official validation used for router-policy selection: **no**  ",
        "Test used: **no**",
        "",
        "This compares a fresh verdict-only control against the rationale-trained B18A expert under the same headline optimizer/data recipe. It does not select or evaluate a test policy.",
        "",
        "| Ensemble | Accuracy | Macro-F1 | F1 Supported | F1 Refuted | F1 NEI |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Headline-recipe direct-only | {effect['baseline']['accuracy']:.6f} | {effect['baseline']['macro_f1']:.6f} | {effect['baseline']['f1_supported']:.6f} | {effect['baseline']['f1_refuted']:.6f} | {effect['baseline']['f1_nei']:.6f} |",
        f"| Rationale-trained B18A | {effect['candidate']['accuracy']:.6f} | {effect['candidate']['macro_f1']:.6f} | {effect['candidate']['f1_supported']:.6f} | {effect['candidate']['f1_refuted']:.6f} | {effect['candidate']['f1_nei']:.6f} |",
        "",
        "## Rationale-trained minus headline-recipe direct-only",
        "",
        f"- Macro-F1 delta: `{effect['macro_f1_delta']:+.6f}`",
        f"- Accuracy delta: `{effect['accuracy_delta']:+.6f}`",
        f"- Helpful/harmful: `{effect['helpful']}/{effect['harmful']}`",
        f"- Exact McNemar p: `{effect['exact_mcnemar_p']:.6f}`",
        f"- Bootstrap 95% CI: `[{ci[0]:+.6f}, {ci[1]:+.6f}]`",
        f"- Bootstrap P(delta > 0): `{boot['probability_delta_positive']:.4f}`",
        "",
        "The causal claim is supported only if the fresh direct-only control is worse under this exact matched recipe. Otherwise the headline B18A contrast remains attributable to a mixture of recipe and rationale differences.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
