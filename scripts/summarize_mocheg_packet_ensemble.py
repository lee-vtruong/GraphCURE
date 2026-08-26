"""Confirm a frozen article/packet interpolation across MOCHEG seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_mocheg_expert_complementarity import (
    prediction_metrics,
    read_predictions,
)
from scripts.audit_mocheg_router import bootstrap_delta


def align_pair(anchor_rows: dict[str, dict], packet_rows: dict[str, dict]):
    if set(anchor_rows) != set(packet_rows):
        raise ValueError("prediction ID sets do not match")
    ids = sorted(anchor_rows)
    gold = np.asarray([int(anchor_rows[value]["gold"]) for value in ids])
    packet_gold = np.asarray([int(packet_rows[value]["gold"]) for value in ids])
    if not np.array_equal(gold, packet_gold):
        raise ValueError("gold labels do not match")
    anchor = np.asarray(
        [anchor_rows[value]["probabilities"] for value in ids], dtype=np.float64
    )
    packet = np.asarray(
        [packet_rows[value]["probabilities"] for value in ids], dtype=np.float64
    )
    return ids, gold, anchor, packet


def summarize(seed_pairs: list[tuple[int, dict[str, dict], dict[str, dict]]],
              packet_weight: float, minimum_delta: float,
              maximum_delta_std: float, minimum_positive_seeds: int,
              bootstrap_iterations: int, bootstrap_seed: int) -> dict:
    if not 0 <= packet_weight <= 1:
        raise ValueError("packet weight must be in [0, 1]")
    per_seed, anchor_probabilities, mixed_probabilities = [], [], []
    reference_ids = None; reference_gold = None
    for seed, anchor_rows, packet_rows in seed_pairs:
        ids, gold, anchor, packet = align_pair(anchor_rows, packet_rows)
        if reference_ids is None:
            reference_ids, reference_gold = ids, gold
        elif ids != reference_ids or not np.array_equal(gold, reference_gold):
            raise ValueError("all seeds must use the same validation examples")
        mixed = (1.0 - packet_weight) * anchor + packet_weight * packet
        anchor_metrics = prediction_metrics(gold, anchor)
        packet_metrics = prediction_metrics(gold, packet)
        mixed_metrics = prediction_metrics(gold, mixed)
        per_seed.append({
            "seed": seed, "anchor": anchor_metrics, "packet": packet_metrics,
            "frozen_ensemble": mixed_metrics,
            "ensemble_minus_anchor_macro_f1": float(
                mixed_metrics["macro_f1"] - anchor_metrics["macro_f1"]
            ),
        })
        anchor_probabilities.append(anchor); mixed_probabilities.append(mixed)
    if reference_gold is None:
        raise ValueError("at least one seed is required")
    deltas = np.asarray([
        row["ensemble_minus_anchor_macro_f1"] for row in per_seed
    ], dtype=np.float64)
    anchor_ensemble = np.mean(anchor_probabilities, axis=0)
    mixed_ensemble = np.mean(mixed_probabilities, axis=0)
    anchor_metrics = prediction_metrics(reference_gold, anchor_ensemble)
    mixed_metrics = prediction_metrics(reference_gold, mixed_ensemble)
    raw_delta = mixed_metrics["macro_f1"] - anchor_metrics["macro_f1"]
    positive_seeds = int(np.sum(deltas > 0))
    bootstrap = bootstrap_delta(
        reference_gold, anchor_ensemble.argmax(-1), mixed_ensemble.argmax(-1),
        bootstrap_iterations, bootstrap_seed,
    )
    gate = {
        "minimum_mean_delta": minimum_delta,
        "maximum_delta_std": maximum_delta_std,
        "minimum_positive_seeds": minimum_positive_seeds,
        "mean_delta_passed": bool(deltas.mean() >= minimum_delta),
        "delta_stability_passed": bool(deltas.std() <= maximum_delta_std),
        "positive_seed_count_passed": bool(positive_seeds >= minimum_positive_seeds),
        "raw_ensemble_delta_passed": bool(raw_delta >= minimum_delta),
    }
    gate["passed"] = bool(all(
        gate[key] for key in (
            "mean_delta_passed", "delta_stability_passed",
            "positive_seed_count_passed", "raw_ensemble_delta_passed",
        )
    ))
    return {
        "protocol": "frozen_validation_multiseed_confirmation",
        "packet_weight": packet_weight,
        "seeds": [row["seed"] for row in per_seed],
        "per_seed": per_seed,
        "paired_macro_f1_delta": {
            "mean": float(deltas.mean()), "std": float(deltas.std()),
            "values": deltas.tolist(), "positive_seeds": positive_seeds,
        },
        "raw_anchor_ensemble": anchor_metrics,
        "raw_frozen_packet_ensemble": mixed_metrics,
        "raw_ensemble_macro_f1_delta": float(raw_delta),
        "paired_bootstrap_raw_ensemble": bootstrap,
        "promotion_gate": gate,
        "test_split_used": False,
    }


def markdown(payload: dict) -> str:
    lines = [
        "# MOCHEG frozen article/packet validation confirmation", "",
        "Test split used: **no**", "",
        f"Frozen packet weight: `{payload['packet_weight']:.2f}`", "",
        "| Seed | Article F1 | Packet F1 | Ensemble F1 | Delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['anchor']['macro_f1']:.4f} | "
            f"{row['packet']['macro_f1']:.4f} | "
            f"{row['frozen_ensemble']['macro_f1']:.4f} | "
            f"{row['ensemble_minus_anchor_macro_f1']:+.4f} |"
        )
    lines += [
        "", "## Aggregate", "",
        f"- Paired delta: {payload['paired_macro_f1_delta']['mean']:+.6f} "
        f"+/- {payload['paired_macro_f1_delta']['std']:.6f}",
        f"- Raw article ensemble F1: "
        f"{payload['raw_anchor_ensemble']['macro_f1']:.6f}",
        f"- Raw frozen packet ensemble F1: "
        f"{payload['raw_frozen_packet_ensemble']['macro_f1']:.6f}",
        f"- Raw ensemble delta: {payload['raw_ensemble_macro_f1_delta']:+.6f}",
        f"- Promotion gate: **{'pass' if payload['promotion_gate']['passed'] else 'fail'}**",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-template", default=(
        "outputs/mocheg_qwen3_lora_seed{seed}_v16/val_predictions.jsonl"))
    parser.add_argument("--packet-template", default=(
        "outputs/mocheg_packet_qwen3_seed{seed}/val_predictions.jsonl"))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[13, 21, 42, 87, 100])
    parser.add_argument("--packet-weight", type=float, default=.44)
    parser.add_argument("--minimum-delta", type=float, default=.003)
    parser.add_argument("--maximum-delta-std", type=float, default=.01)
    parser.add_argument("--minimum-positive-seeds", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_packet_multiseed_confirmation.json"))
    args = parser.parse_args()
    pairs = []
    for seed in args.seeds:
        anchor_path = Path(args.anchor_template.format(seed=seed))
        packet_path = Path(args.packet_template.format(seed=seed))
        if not anchor_path.exists() or not packet_path.exists():
            parser.error(f"missing seed {seed}: {anchor_path} or {packet_path}")
        pairs.append((seed, read_predictions(anchor_path), read_predictions(packet_path)))
    result = summarize(
        pairs, args.packet_weight, args.minimum_delta,
        args.maximum_delta_std, args.minimum_positive_seeds,
        args.bootstrap_iterations, args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
