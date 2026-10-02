"""Post-hoc, label-only diagnostic for a frozen external CURE AND transfer.

This script never changes a checkpoint, seed, retrieval configuration, K, or
threshold.  It explains already-materialized direct/AND predictions, making
the costs of routing explicit after an external evaluation has completed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from scripts.summarize_mocheg_b18_seeds import compute_metrics


NAMES = ("Supported", "Refuted", "NEI")


def read_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = str(row["id"])
            if sample_id in rows:
                raise ValueError(f"duplicate ID {sample_id!r} in {path}")
            rows[sample_id] = row
    return rows


def pct(value: int, total: int) -> str:
    return f"{value / total:.4f}" if total else "--"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--rationale", type=Path, required=True)
    parser.add_argument("--and-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=.49,
                        help="Frozen rationale NEI threshold used by AND")
    args = parser.parse_args()

    direct, rationale, routed = (read_jsonl(path) for path in
                                 (args.direct, args.rationale, args.and_predictions))
    ids = sorted(direct)
    if set(ids) != set(rationale) or set(ids) != set(routed):
        raise ValueError("direct, rationale, and AND IDs must match exactly")

    labels = np.asarray([int(direct[i]["label"]) for i in ids])
    direct_pred = np.asarray([int(direct[i]["prediction"]) for i in ids])
    rationale_pred = np.asarray([int(rationale[i]["prediction"]) for i in ids])
    rationale_prob = np.asarray([rationale[i]["probabilities"] for i in ids], dtype=float)
    and_pred = np.asarray([int(routed[i]["prediction"]) for i in ids])
    if any(int(rationale[i]["label"]) != int(direct[i]["label"]) or
           int(routed[i]["label"]) != int(direct[i]["label"]) for i in ids):
        raise ValueError("label mismatch across materialized prediction files")

    activation_mask = (rationale_pred == 2) & (rationale_prob[:, 2] >= args.tau)
    changed_mask = and_pred != direct_pred
    expected_and = direct_pred.copy()
    expected_and[activation_mask] = 2
    if not np.array_equal(and_pred, expected_and):
        raise ValueError("materialized AND predictions do not match frozen tau policy")
    helpful = (~(direct_pred == labels) & (and_pred == labels))
    harmful = ((direct_pred == labels) & ~(and_pred == labels))

    by_gold = []
    for gold, name in enumerate(NAMES):
        mask = labels == gold
        by_gold.append({
            "gold": name, "samples": int(mask.sum()),
            "direct_accuracy": float(accuracy_score(labels[mask], direct_pred[mask])),
            "and_accuracy": float(accuracy_score(labels[mask], and_pred[mask])),
            "accuracy_delta": float(accuracy_score(labels[mask], and_pred[mask]) -
                                    accuracy_score(labels[mask], direct_pred[mask])),
            "activations": int((activation_mask & mask).sum()),
            "changes": int((changed_mask & mask).sum()),
            "helpful": int((helpful & mask).sum()),
            "harmful": int((harmful & mask).sum()),
        })

    activation_outcomes = Counter()
    for index in np.flatnonzero(activation_mask):
        before = "correct" if direct_pred[index] == labels[index] else "wrong"
        after = "correct" if and_pred[index] == labels[index] else "wrong"
        activation_outcomes[f"direct_{before}_and_{after}"] += 1
    transition_counts = Counter(
        (NAMES[labels[i]], NAMES[direct_pred[i]], NAMES[and_pred[i]])
        for i in np.flatnonzero(changed_mask)
    )

    report = {
        "protocol": "post_hoc_external_zero_shot_transition_diagnostic",
        "policy_changed": False,
        "claims": len(ids),
        "direct_metrics": compute_metrics(labels, direct_pred),
        "rationale_metrics": compute_metrics(labels, rationale_pred),
        "and_metrics": compute_metrics(labels, and_pred),
        "activation_count": int(activation_mask.sum()),
        "activation_rate": float(activation_mask.mean()),
        "changed_prediction_count": int(changed_mask.sum()),
        "changed_prediction_rate": float(changed_mask.mean()),
        "helpful": int(helpful.sum()), "harmful": int(harmful.sum()),
        "by_gold": by_gold,
        "activation_outcomes": dict(activation_outcomes),
        "changed_prediction_transitions": [
            {"gold": key[0], "direct": key[1], "and": key[2], "count": count}
            for key, count in sorted(transition_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_transition_diagnostic.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = ["# CURE frozen external-transfer transition diagnostic", "",
             "- Post-hoc only: external labels are used for this explanation, never for routing policy selection.",
             "- Checkpoint, seed, retrieval, `K=5`, and `tau=.49` are unchanged.", "",
             f"- AND activations: `{activation_mask.sum()}/{len(ids)}`; actual prediction changes: `{changed_mask.sum()}/{len(ids)}`.", "",
             "## AND activation and changes by gold class", "",
             "| Gold class | N | Direct accuracy | AND accuracy | Delta | Activations / changes | Helpful/Harmful |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in by_gold:
        lines.append(
            f"| {row['gold']} | {row['samples']} | {row['direct_accuracy']:.4f} | "
            f"{row['and_accuracy']:.4f} | {row['accuracy_delta']:+.4f} | {row['activations']}/{row['changes']} | "
            f"{row['helpful']}/{row['harmful']} |")
    lines.extend(["", "## Outcomes among activated claims", "",
                  "| Direct outcome $\\rightarrow$ AND outcome | Claims |", "| --- | ---: |"])
    for key in ("direct_wrong_and_correct", "direct_correct_and_wrong",
                "direct_wrong_and_wrong", "direct_correct_and_correct"):
        lines.append(f"| {key.replace('_', ' ')} | {activation_outcomes[key]} |")
    lines.extend(["", "## Changed-prediction transitions", "",
                  "| Gold | Direct | AND | Claims |", "| --- | --- | --- | ---: |"])
    for row in report["changed_prediction_transitions"]:
        lines.append(f"| {row['gold']} | {row['direct']} | {row['and']} | {row['count']} |")
    lines.extend(["", "The diagnostic is descriptive. It does not establish a new policy and does not make SciFact comparable to its standard pipeline leaderboard."])
    (args.output_dir / "external_transition_diagnostic.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
