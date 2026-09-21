"""Calibrate the B18-B adaptive selector on claim-disjoint train-side dev data.

The calibration target is teacher key-evidence attribution only. Verdict labels,
gold evidence, official validation, and test data are never consulted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from graphcure.explanation import resolve_corpus_path
from graphcure.selector import AdaptiveSelectorPolicy, extract_selector_pairs
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_b18b_sentence_selector import split_pairs_by_claim
from scripts.train_mocheg_qwen3_lora_verifier import read_documents


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_grid(raw: str) -> list[float]:
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values:
        raise ValueError("calibration grid cannot be empty")
    return values


def attribution_metrics(
    records: list[dict[str, Any]],
    *,
    score_threshold: float,
    adaptive_margin: float,
    min_k: int,
    max_k: int,
) -> dict[str, float | int]:
    policy = AdaptiveSelectorPolicy(
        min_k=min_k,
        max_k=max_k,
        score_threshold=score_threshold,
        adaptive_margin=adaptive_margin,
    )
    hits = 0
    selected_total = 0
    key_total = 0
    claim_coverages: list[float] = []
    claim_precisions: list[float] = []
    selected_counts: list[int] = []
    for record in records:
        selected_ids, _ = policy.select(record["candidate_ids"], record["scores"])
        selected = set(selected_ids)
        keys = record["key_evidence_ids"]
        overlap = len(selected & keys)
        hits += overlap
        selected_total += len(selected)
        key_total += len(keys)
        claim_coverages.append(overlap / len(keys))
        claim_precisions.append(overlap / len(selected) if selected else 0.0)
        selected_counts.append(len(selected))

    recall = hits / key_total if key_total else 0.0
    precision = hits / selected_total if selected_total else 0.0
    beta2 = 4.0
    f2 = (
        (1.0 + beta2) * precision * recall / (beta2 * precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "score_threshold": score_threshold,
        "adaptive_margin": adaptive_margin,
        "claims": len(records),
        "micro_teacher_key_recall": recall,
        "micro_teacher_key_precision": precision,
        "teacher_key_f2": f2,
        "mean_teacher_key_coverage": float(np.mean(claim_coverages)),
        "mean_teacher_key_precision": float(np.mean(claim_precisions)),
        "mean_selected_k": float(np.mean(selected_counts)),
    }


def choose_policy(
    rows: list[dict[str, float | int]], minimum_teacher_recall: float
) -> tuple[dict[str, float | int], bool]:
    eligible = [row for row in rows if row["micro_teacher_key_recall"] >= minimum_teacher_recall]
    candidates = eligible or rows
    chosen = max(
        candidates,
        key=lambda row: (
            row["teacher_key_f2"],
            row["micro_teacher_key_recall"],
            -row["mean_selected_k"],
            row["micro_teacher_key_precision"],
        ),
    )
    return chosen, bool(eligible)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--selector-training-summary", type=Path, required=True)
    parser.add_argument("--explanations", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-name", default="distilled_ce_adaptive")
    parser.add_argument("--top-k-candidates", type=int, default=5)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--threshold-grid", default="-3,-2,-1,-0.5,0,0.5,1")
    parser.add_argument("--margin-grid", default="1,1.5,2,2.5,3,4")
    parser.add_argument("--minimum-teacher-recall", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_summary = json.loads(args.selector_training_summary.read_text(encoding="utf-8"))
    corpus_path = resolve_corpus_path(args.corpus)
    audit = training_summary.get("audit", {})
    if audit.get("explanations_sha256") != sha256_file(args.explanations):
        raise ValueError("explanations hash does not match selector training summary")
    if audit.get("corpus_sha256") != sha256_file(corpus_path):
        raise ValueError("corpus hash does not match selector training summary")

    explanations = read_jsonl(args.explanations)
    documents = read_documents(corpus_path)
    pairs = extract_selector_pairs(
        explanations=explanations,
        documents=documents,
        max_neg_per_grounded=4,
        sample_ungrounded_negatives=True,
        max_neg_per_ungrounded=2,
        seed=int(training_summary["seed"]),
    )
    _, dev_pairs = split_pairs_by_claim(
        pairs,
        float(training_summary["dev_fraction"]),
        int(training_summary["seed"]),
    )
    dev_claim_ids = {str(pair["claim_id"]) for pair in dev_pairs}

    claims = {str(row["id"]): str(row.get("claim", "")) for row in read_jsonl(args.manifest)}
    retrieval = {str(row["id"]): row for row in read_jsonl(args.retrieval)}
    teacher = {str(row["id"]): row for row in explanations}

    metadata: list[tuple[str, list[str], set[str]]] = []
    model_inputs: list[tuple[str, str]] = []
    for claim_id in sorted(dev_claim_ids):
        row = retrieval.get(claim_id)
        teacher_row = teacher.get(claim_id)
        claim = claims.get(claim_id, "")
        if not row or not claim or not teacher_row:
            continue
        if not teacher_row.get("is_valid") or not teacher_row.get("grounded"):
            continue
        candidate_ids = [
            evidence_id
            for evidence_id in row.get("retrieved_evidence_ids", [])[: args.top_k_candidates]
            if evidence_id in documents
        ]
        teacher_retrieved = teacher_row.get("retrieved_evidence_ids", [])
        key_ids = {
            teacher_retrieved[index - 1]
            for index in teacher_row.get("key_evidence_ids", [])
            if 1 <= index <= len(teacher_retrieved)
        }
        key_ids &= set(candidate_ids)
        if not candidate_ids or not key_ids:
            continue
        metadata.append((claim_id, candidate_ids, key_ids))
        model_inputs.extend((claim, documents[evidence_id]) for evidence_id in candidate_ids)

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(args.selector, device=args.device)
    scores = [
        float(score)
        for score in model.predict(
            model_inputs,
            batch_size=args.batch_size,
            show_progress_bar=True,
        )
    ]
    records: list[dict[str, Any]] = []
    offset = 0
    for claim_id, candidate_ids, key_ids in metadata:
        count = len(candidate_ids)
        records.append({
            "claim_id": claim_id,
            "candidate_ids": candidate_ids,
            "key_evidence_ids": key_ids,
            "scores": scores[offset : offset + count],
        })
        offset += count

    grid = [
        attribution_metrics(
            records,
            score_threshold=threshold,
            adaptive_margin=margin,
            min_k=args.min_k,
            max_k=args.max_k,
        )
        for threshold in parse_grid(args.threshold_grid)
        for margin in parse_grid(args.margin_grid)
    ]
    chosen, recall_constraint_satisfied = choose_policy(grid, args.minimum_teacher_recall)
    payload = {
        "phase": "B18-B",
        "protocol": "train_side_claim_disjoint_teacher_attribution_calibration_v1",
        "policy_name": args.policy_name,
        "selector": args.selector,
        "dev_claims_available": len(dev_claim_ids),
        "grounded_dev_claims_scored": len(records),
        "selection_objective": "maximize teacher-key F2 subject to minimum micro recall",
        "minimum_teacher_recall": args.minimum_teacher_recall,
        "recall_constraint_satisfied": recall_constraint_satisfied,
        "selected_policy": chosen,
        "grid": grid,
        "audit": {
            "selector_training_summary_sha256": sha256_file(args.selector_training_summary),
            "explanations_sha256": sha256_file(args.explanations),
            "retrieval_sha256": sha256_file(args.retrieval),
            "manifest_sha256": sha256_file(args.manifest),
            "corpus_sha256": sha256_file(corpus_path),
            "teacher_pseudo_labels_used_for_calibration": True,
            "verdict_labels_used": False,
            "gold_evidence_used": False,
            "official_validation_used": False,
            "test_split_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
