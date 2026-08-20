"""Audit frozen visual-expert utility by evidence-sufficiency strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from scripts.audit_mocheg_router import exact_mcnemar_p


def group_metrics(rows: list[dict], chosen: np.ndarray) -> dict:
    indices = np.flatnonzero(chosen)
    if len(indices) == 0:
        return {"samples": 0}
    gold = np.asarray([rows[index]["gold"] for index in indices])
    text = np.asarray([
        rows[index]["text_only_prediction"] for index in indices
    ])
    expert = np.asarray([
        rows[index]["visual_expert_prediction"] for index in indices
    ])
    helpful = (text != gold) & (expert == gold)
    harmful = (text == gold) & (expert != gold)
    return {
        "samples": len(indices),
        "text_accuracy": float(accuracy_score(gold, text)),
        "text_macro_f1": float(f1_score(
            gold, text, labels=[0, 1, 2], average="macro", zero_division=0
        )),
        "expert_accuracy": float(accuracy_score(gold, expert)),
        "expert_macro_f1": float(f1_score(
            gold, expert, labels=[0, 1, 2], average="macro", zero_division=0
        )),
        "expert_minus_text_accuracy": float(
            accuracy_score(gold, expert) - accuracy_score(gold, text)
        ),
        "expert_minus_text_macro_f1": float(
            f1_score(
                gold, expert, labels=[0, 1, 2], average="macro",
                zero_division=0,
            )
            - f1_score(
                gold, text, labels=[0, 1, 2], average="macro",
                zero_division=0,
            )
        ),
        "helpful": int(helpful.sum()),
        "harmful": int(harmful.sum()),
        "net_corrections": int(helpful.sum() - harmful.sum()),
        "decisive": int((helpful | harmful).sum()),
        "exact_mcnemar_p": exact_mcnemar_p(
            int(helpful.sum()), int(harmful.sum())
        ),
    }


def audit(rows: list[dict]) -> dict:
    gold_available = np.asarray([
        bool(row.get("visual_gold_in_candidates")) for row in rows
    ])
    selected_gold = np.asarray([
        bool(row.get("visual_selected_gold")) for row in rows
    ])
    result = {
        "all": group_metrics(rows, np.ones(len(rows), dtype=bool)),
        "gold_in_candidates": group_metrics(rows, gold_available),
        "no_gold_in_candidates": group_metrics(rows, ~gold_available),
        "selected_gold": group_metrics(rows, selected_gold),
        "gold_available_not_selected": group_metrics(
            rows, gold_available & ~selected_gold
        ),
        "test_split_used": False,
        "warning": (
            "Qrel strata are diagnostic supervision and are not available to "
            "the inference router."
        ),
    }
    gold_policy_prediction = np.asarray([
        row["visual_expert_prediction"]
        if row.get("visual_gold_in_candidates")
        else row["text_only_prediction"]
        for row in rows
    ])
    gold = np.asarray([row["gold"] for row in rows])
    result["qrel_oracle_sufficiency_policy"] = {
        "accuracy": float(accuracy_score(gold, gold_policy_prediction)),
        "macro_f1": float(f1_score(
            gold, gold_policy_prediction, labels=[0, 1, 2],
            average="macro", zero_division=0,
        )),
        "diagnostic_only": True,
    }
    return result


def markdown(result: dict) -> str:
    lines = [
        "# MOCHEG visual-expert utility audit",
        "",
        "Test split used: **no**",
        "",
        "| Stratum | N | Text F1 | Expert F1 | Delta | Help | Harm | Net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "all", "gold_in_candidates", "no_gold_in_candidates",
        "selected_gold", "gold_available_not_selected",
    ):
        row = result[name]
        lines.append(
            f"| {name} | {row['samples']} | {row.get('text_macro_f1', 0):.4f} "
            f"| {row.get('expert_macro_f1', 0):.4f} | "
            f"{row.get('expert_minus_text_macro_f1', 0):+.4f} | "
            f"{row.get('helpful', 0)} | {row.get('harmful', 0)} | "
            f"{row.get('net_corrections', 0):+d} |"
        )
    policy = result["qrel_oracle_sufficiency_policy"]
    lines.extend([
        "",
        f"- Qrel-oracle sufficiency policy: Accuracy `{policy['accuracy']:.4f}`, "
        f"Macro-F1 `{policy['macro_f1']:.4f}` (diagnostic only).",
        "- Qrels are never an inference feature.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "test" in args.predictions.name.lower():
        parser.error("expert development audit must not use test predictions")
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        parser.error("predictions file is empty")
    result = audit(rows)
    result["provenance"] = {"predictions": str(args.predictions)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(
        markdown(result), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
