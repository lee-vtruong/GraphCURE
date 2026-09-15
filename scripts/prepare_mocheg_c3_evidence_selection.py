"""Build label-free rank and constraint-safe top-k evidence sets for C3."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from graphcure.open_web import load_jsonl, normalize_space, sha256_file, sha256_text


TASKS = ("stance", "sufficiency", "entity", "temporal")


def load_scores(path: Path) -> dict[str, dict[str, dict[str, dict]]]:
    result: dict[str, dict[str, dict[str, dict]]] = {}
    for row in load_jsonl(path):
        claim = result.setdefault(str(row["id"]), {})
        evidence = claim.setdefault(str(row["evidence_id"]), {})
        task = str(row["task"])
        if task in evidence:
            raise ValueError(
                f"duplicate score for {row['id']} {row['evidence_id']} {task}"
            )
        evidence[task] = row
    return result


def constraint_stage(tasks: dict[str, dict], stage: str) -> bool:
    prediction = {task: tasks[task]["prediction"] for task in TASKS}
    stance_decisive = prediction["stance"] != "C"
    evidence_relevant = prediction["sufficiency"] != "C"
    entity_safe = prediction["entity"] != "B"
    temporal_safe = prediction["temporal"] != "B"
    predicates = {
        "safe_decisive": (
            stance_decisive and evidence_relevant
            and entity_safe and temporal_safe
        ),
        "decisive_relevant": stance_decisive and evidence_relevant,
        "decisive": stance_decisive,
        "nonirrelevant": evidence_relevant,
        "rank_fallback": True,
    }
    return predicates[stage]


def select_constraint_safe(
    evidence: list[dict], scores: dict[str, dict], top_k: int
) -> list[dict]:
    ranked = sorted(evidence, key=lambda row: int(row.get("shortlist_rank", 10**9)))
    selected: list[dict] = []
    selected_ids: set[str] = set()
    selected_text: set[str] = set()

    def add(row: dict, stage: str, enforce_text_deduplication: bool) -> bool:
        evidence_id = str(row["evidence_id"])
        text_key = sha256_text(normalize_space(row.get("text", "")).casefold())
        if evidence_id in selected_ids:
            return False
        if enforce_text_deduplication and text_key in selected_text:
            return False
        item = dict(row)
        item["c3_selection_stage"] = stage
        item["c3_rank"] = len(selected) + 1
        selected.append(item)
        selected_ids.add(evidence_id)
        selected_text.add(text_key)
        return True

    for stage in (
        "safe_decisive", "decisive_relevant", "decisive",
        "nonirrelevant", "rank_fallback",
    ):
        for row in ranked:
            evidence_id = str(row["evidence_id"])
            tasks = scores.get(evidence_id, {})
            if set(tasks) != set(TASKS):
                raise ValueError(f"incomplete constraints for evidence {evidence_id}")
            if constraint_stage(tasks, stage):
                add(row, stage, enforce_text_deduplication=True)
            if len(selected) == top_k:
                return selected
    # Only claims with fewer than top-k unique texts reach this auditable fallback.
    for row in ranked:
        add(row, "duplicate_relaxation", enforce_text_deduplication=False)
        if len(selected) == top_k:
            return selected
    return selected


def prediction_profile(
    rows: list[dict], scores: dict[str, dict[str, dict]]
) -> dict[str, dict[str, int]]:
    result = {task: Counter() for task in TASKS}
    for row in rows:
        tasks = scores[str(row["evidence_id"])]
        for task in TASKS:
            result[task][tasks[task]["prediction"]] += 1
    return {
        task: {code: result[task][code] for code in "ABC"} for task in TASKS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--constraint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("top-k must be positive")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    constraint_summary_path = args.constraint_root / "summary.json"
    constraint_path = args.constraint_root / "constraint_scores.jsonl"
    constraint_summary = json.loads(
        constraint_summary_path.read_text(encoding="utf-8")
    )
    forbidden = ("label_used", "gold_evidence_used", "test_split_used")
    if any(audit.get(key) or constraint_summary.get(key) for key in forbidden):
        parser.error("C3 preparation inputs contain forbidden supervision")
    shortlist_hash = sha256_file(args.shortlist)
    if shortlist_hash != audit.get("shortlist_sha256"):
        parser.error("C2a shortlist hash mismatch")
    if shortlist_hash != constraint_summary.get("shortlist_sha256"):
        parser.error("C2b constraint scores do not match C2a shortlist")
    if not constraint_summary.get("complete"):
        parser.error("C2b constraint scores are incomplete")

    rows = load_jsonl(args.shortlist)
    scores = load_scores(constraint_path)
    output_rows = []
    changed_claims = 0
    overlap = []
    original_ranks = []
    stage_counts: Counter[str] = Counter()
    control_profile = {task: Counter() for task in TASKS}
    selected_profile = {task: Counter() for task in TASKS}
    selected_social = 0
    selected_domains = []
    for row in rows:
        sample_id = str(row["id"])
        evidence = row.get("evidence_shortlist", [])
        claim_scores = scores.get(sample_id, {})
        control = evidence[:args.top_k]
        selected = select_constraint_safe(evidence, claim_scores, args.top_k)
        control_ids = [str(item["evidence_id"]) for item in control]
        selected_ids = [str(item["evidence_id"]) for item in selected]
        changed_claims += int(control_ids != selected_ids)
        overlap.append(len(set(control_ids) & set(selected_ids)) / max(
            1, len(set(control_ids) | set(selected_ids))
        ))
        original_ranks.extend(int(item["shortlist_rank"]) for item in selected)
        stage_counts.update(item["c3_selection_stage"] for item in selected)
        selected_social += sum(
            item.get("source_family") == "social" for item in selected
        )
        selected_domains.append(len({
            item.get("domain") for item in selected if item.get("domain")
        }))
        for task, counts in prediction_profile(control, claim_scores).items():
            control_profile[task].update(counts)
        for task, counts in prediction_profile(selected, claim_scores).items():
            selected_profile[task].update(counts)
        output_rows.append({
            "id": sample_id,
            "claim_id": row.get("claim_id"),
            "claim": row.get("claim", ""),
            "rank_control_evidence": control,
            "constraint_selected_evidence": selected,
            "label_included": False,
            "gold_evidence_used": False,
        })

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "val.jsonl"
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    selected_total = sum(
        len(row["constraint_selected_evidence"]) for row in output_rows
    )
    summary = {
        "protocol": "P2_open_web_C3_training_aligned_constraint_selection",
        "claims": len(rows),
        "top_k": args.top_k,
        "changed_claims": changed_claims,
        "changed_claim_rate": changed_claims / max(1, len(rows)),
        "mean_control_selected_jaccard": statistics.fmean(overlap),
        "selection_stage_counts": dict(stage_counts),
        "selected_original_rank_mean": statistics.fmean(original_ranks),
        "selected_original_rank_median": statistics.median(original_ranks),
        "selected_domain_count_minimum": min(selected_domains),
        "selected_domain_count_median": statistics.median(selected_domains),
        "selected_social_share": selected_social / max(1, selected_total),
        "control_constraint_profile": {
            task: {code: control_profile[task][code] for code in "ABC"}
            for task in TASKS
        },
        "selected_constraint_profile": {
            task: {code: selected_profile[task][code] for code in "ABC"}
            for task in TASKS
        },
        "shortlist_sha256": shortlist_hash,
        "constraint_scores_sha256": sha256_file(constraint_path),
        "output_sha256": sha256_file(output_path),
        "selection_policy": [
            "safe_decisive", "decisive_relevant", "decisive",
            "nonirrelevant", "rank_fallback", "duplicate_relaxation",
        ],
        "verifier_alignment": {
            "top_k": 5,
            "max_evidence_chars": 2200,
            "constraint_annotations_in_prompt": False,
        },
        "label_used": False,
        "gold_evidence_used": False,
        "test_split_used": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
