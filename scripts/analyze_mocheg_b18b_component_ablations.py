"""Train/validation-only component ablations for the B18B dual-expert router.

The script consumes saved probability predictions and never reads the official
test split. All policies are deterministic and are intended for ablation, not
for selecting a new test threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_mocheg_b18_seeds import compute_metrics, load_seed_predictions


def load_average(paths: list[Path]):
    runs = [load_seed_predictions(path) for path in paths]
    ids = sorted(set.intersection(*(set(run) for run in runs)))
    if not ids:
        raise ValueError("no common prediction IDs")
    labels = np.asarray([int(runs[0][i]["label"]) for i in ids])
    probs = np.mean(
        [np.asarray([runs[j][i]["probabilities"] for i in ids], dtype=float)
         for j in range(len(runs))], axis=0,
    )
    for run in runs[1:]:
        other = np.asarray([int(run[i]["label"]) for i in ids])
        if not np.array_equal(labels, other):
            raise ValueError("label mismatch across prediction files")
    return ids, labels, probs


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-9, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def evaluate(labels: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    return compute_metrics(labels, pred, probs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direct-runs", type=Path, nargs="+", required=True)
    p.add_argument("--grounded-runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    p.add_argument("--tau", type=float, default=0.49)
    args = p.parse_args()

    ids_d, labels, direct = load_average(args.direct_runs)
    ids_g, labels_g, grounded = load_average(args.grounded_runs)
    if ids_d != ids_g or not np.array_equal(labels, labels_g):
        raise ValueError("direct and grounded predictions are not aligned")

    direct_pred = direct.argmax(axis=1)
    grounded_pred = grounded.argmax(axis=1)
    direct_conf = direct.max(axis=1)
    grounded_conf = grounded.max(axis=1)

    # A0/A1: single experts.
    policies = {
        "A0_direct_only": direct,
        "A1_grounded_only": grounded,
        # A2: symmetric posterior averaging.
        "A2_equal_ensemble": 0.5 * (direct + grounded),
        # A3: symmetric confidence routing; route only on disagreement and
        # choose the more confident expert. Agreement keeps the shared class.
        "A3_symmetric_confidence": np.where(
            ((direct_pred != grounded_pred) & (grounded_conf > direct_conf))[:, None],
            grounded, direct,
        ),
        # A4: no asymmetric NEI rule; route to grounded whenever it is
        # sufficiently confident, regardless of its predicted class.
        "A4_no_asymmetric_nei": np.where(
            (grounded_conf >= args.tau)[:, None], grounded, direct
        ),
        # A5: no confidence; route only on prediction disagreement.
        "A5_disagreement_only": np.where(
            (direct_pred != grounded_pred)[:, None], grounded, direct
        ),
        # A6: confidence-only, without disagreement restriction: always use
        # the more confident expert. This intentionally differs from A3.
        "A6_max_confidence_only": np.where(
            (grounded_conf > direct_conf)[:, None], grounded, direct
        ),
        # A8: registered asymmetric B18B deferral.
        "A8_b18b_asymmetric": np.where(
            ((grounded_pred == 2) & (grounded[:, 2] >= args.tau))[:, None],
            np.eye(3, dtype=float)[np.full(len(labels), 2)], direct,
        ),
    }

    metrics = {name: evaluate(labels, value) for name, value in policies.items()}
    policy_rows = {}
    for name, value in policies.items():
        policy_rows[name] = {
            "metrics": metrics[name],
            "route_count": int(np.sum(np.argmax(value, axis=1) != direct_pred)),
        }
    payload = {
        "protocol": "B18B_train_only_component_ablation",
        "samples": len(labels),
        "tau": args.tau,
        "policies": policy_rows,
        "feature_audit": {
            "direct_entropy_mean": float(entropy(direct).mean()),
            "grounded_entropy_mean": float(entropy(grounded).mean()),
            "prediction_disagreement_count": int(np.sum(direct_pred != grounded_pred)),
        },
        "official_validation_used": False,
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# B18B train-only component ablations", "",
        "Official validation used: **no**  ", "Test used: **no**", "",
        "| Variant | Accuracy | Macro-F1 | Supported F1 | Refuted F1 | NEI F1 | Routes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        lines.append(
            f"| {name} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['f1_supported']:.6f} | {row['f1_refuted']:.6f} | "
            f"{row['f1_nei']:.6f} | {payload['policies'][name]['route_count']} |"
        )
    lines += [
        "", f"Threshold used for A4/A8: `{args.tau}`", 
        "All results are train-only diagnostics; no policy was selected on test.",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
