"""Evaluate frozen CURE AND on one external three-way verification protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_mocheg_router import bootstrap_delta, exact_mcnemar_p
from scripts.evaluate_mocheg_b18b_canonical_router import (
    load_probability_ensemble, manifest, validate_retrieval, write_predictions,
)
from scripts.summarize_mocheg_b18_seeds import compute_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--direct-runs", type=Path, nargs=5, required=True)
    parser.add_argument("--rationale-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--tau", type=float, default=.49)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.tau <= 1:
        parser.error("tau must be in [0, 1]")
    ids, labels, _ = manifest(args.manifest, args.expected_samples)
    validate_retrieval(args.retrieval, ids, labels)
    direct = load_probability_ensemble(args.direct_runs, ids, labels)
    rationale = load_probability_ensemble(args.rationale_runs, ids, labels)
    direct_prediction = direct.argmax(axis=1)
    rationale_prediction = rationale.argmax(axis=1)
    route = (rationale_prediction == 2) & (rationale[:, 2] >= args.tau)
    and_prediction = direct_prediction.copy()
    and_prediction[route] = 2
    direct_metrics = compute_metrics(labels, direct_prediction)
    rationale_metrics = compute_metrics(labels, rationale_prediction)
    and_metrics = compute_metrics(labels, and_prediction)
    helpful = int(np.sum((direct_prediction != labels) & (and_prediction == labels)))
    harmful = int(np.sum((direct_prediction == labels) & (and_prediction != labels)))
    comparison = {
        "macro_f1_delta": float(and_metrics["macro_f1"] - direct_metrics["macro_f1"]),
        "accuracy_delta": float(and_metrics["accuracy"] - direct_metrics["accuracy"]),
        "helpful": helpful, "harmful": harmful,
        "exact_mcnemar_p": exact_mcnemar_p(helpful, harmful),
        "bootstrap": bootstrap_delta(labels, direct_prediction, and_prediction,
                                     args.iterations, args.seed),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {
        "direct": write_predictions(args.output_dir / "direct_predictions.jsonl", ids, labels, direct, direct_prediction),
        "rationale": write_predictions(args.output_dir / "rationale_predictions.jsonl", ids, labels, rationale, rationale_prediction),
        "and": write_predictions(args.output_dir / "and_predictions.jsonl", ids, labels, direct, and_prediction),
    }
    report = {
        "protocol": "external_zero_shot_frozen_CURE_AND",
        "parameters": {"top_k": args.top_k, "tau": args.tau,
                       "source": "frozen MOCHEG validation policy; no external-data tuning"},
        "samples": len(ids), "manifest_sha256": sha256(args.manifest),
        "retrieval_sha256": sha256(args.retrieval),
        "direct": direct_metrics, "rationale": rationale_metrics, "and": and_metrics,
        "and_minus_direct": comparison, "route_count": int(route.sum()),
        "route_rate": float(route.mean()),
        "input_prediction_sha256": {str(path): sha256(path) for path in args.direct_runs + args.rationale_runs},
        "output_prediction_sha256": output_hashes,
        "external_labels_used_for_policy_selection": False,
        "gold_evidence_used": False,
        "test_split_used": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    boot = comparison["bootstrap"]
    lines = ["# Frozen CURE AND external zero-shot audit", "",
             "- External labels used for model, seed, K, or threshold selection: **no**",
             "- Gold/cited evidence used for retrieval or inference: **no**",
             f"- Frozen policy transferred from MOCHEG: `K={args.top_k}`, `tau={args.tau:.2f}`", "",
             "| System | Accuracy | Macro-F1 | F1 Supported | F1 Refuted | F1 NEI |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, metrics in (("Direct (5)", direct_metrics), ("Rationale-trained (3)", rationale_metrics), ("CURE AND", and_metrics)):
        lines.append(f"| {name} | {metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} | {metrics['f1_supported']:.6f} | {metrics['f1_refuted']:.6f} | {metrics['f1_nei']:.6f} |")
    lines.extend(["", "## AND minus direct", "", f"- Macro-F1 delta: `{comparison['macro_f1_delta']:+.6f}`",
                  f"- Accuracy delta: `{comparison['accuracy_delta']:+.6f}`",
                  f"- Helpful/harmful: `{helpful}/{harmful}`", f"- Exact McNemar p: `{comparison['exact_mcnemar_p']:.4f}`",
                  f"- Bootstrap 95% CI: `[{boot['ci_95_percentile'][0]:+.6f}, {boot['ci_95_percentile'][1]:+.6f}]`",
                  f"- Bootstrap P(delta > 0): `{boot['probability_delta_positive']:.4f}`",
                  f"- Route count/rate: `{int(route.sum())}/{route.mean():.6f}`", "",
                  "This is a zero-shot external development-split audit, not an official SciFact test or a SciFact leaderboard comparison."])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
