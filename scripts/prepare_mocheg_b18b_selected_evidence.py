"""Phase B18-B: Adaptive Evidence Selector & Manifest Generator.

Reranks candidate evidence passages using the trained Cross-Encoder selector
and adaptively prunes them down to 1-3 dense, relevant passages.
Outputs standard retrieval manifests compatible with the downstream verifier.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from graphcure.explanation import resolve_corpus_path
from graphcure.selector import AdaptiveSelectorPolicy, mock_score_claim_evidence_pairs
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_qwen3_lora_verifier import read_documents


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare adaptively filtered evidence manifest for B18-B")
    parser.add_argument("--selector", type=Path, required=True, help="Trained CrossEncoder directory or model name")
    parser.add_argument("--retrieval", type=Path, required=True, help="Input retrieval manifest (.jsonl)")
    parser.add_argument("--manifest", type=Path, required=True, help="Claims manifest (.jsonl)")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus2.csv path")
    parser.add_argument("--output", type=Path, required=True, help="Output filtered retrieval (.jsonl)")
    parser.add_argument("--summary", type=Path, required=True, help="Output diagnostic summary (.json)")
    parser.add_argument("--top-k-candidates", type=int, default=5, help="Number of initial retrieval candidates to rerank")
    parser.add_argument("--min-k", type=int, default=1, help="Minimum passages to keep per claim")
    parser.add_argument("--max-k", type=int, default=3, help="Maximum passages to keep per claim")
    parser.add_argument("--adaptive-margin", type=float, default=1.5, help="Relative score margin from top rank")
    parser.add_argument("--score-threshold", type=float, default=-1.0, help="Absolute score threshold for rank >= 2")
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=["adaptive", "fixed_top_1", "fixed_top_3", "original_top_5", "oracle_teacher"],
        default="adaptive",
        help="Evidence selection policy mode for ablations",
    )
    parser.add_argument(
        "--teacher-explanations",
        type=Path,
        default=None,
        help="Optional teacher explanations JSONL to evaluate teacher-key coverage & attribution metrics",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock", action="store_true", help="Deterministic mock scoring for testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.info("Reading claims manifest: %s", args.manifest)
    claims = {str(c["id"]): c.get("claim", "") for c in read_jsonl(args.manifest)}
    logging.info("Loaded %d claim texts", len(claims))

    corpus_path = resolve_corpus_path(args.corpus)
    logging.info("Reading corpus: %s", corpus_path)
    documents = read_documents(corpus_path)
    logging.info("Loaded %d documents from corpus", len(documents))

    logging.info("Reading retrieval candidates: %s", args.retrieval)
    retrieval_rows = read_jsonl(args.retrieval)
    if args.limit > 0:
        retrieval_rows = retrieval_rows[:args.limit]
    logging.info("Loaded %d retrieval rows to filter", len(retrieval_rows))

    teacher_explanations: dict[str, dict[str, Any]] = {}
    if args.teacher_explanations:
        logging.info("Reading teacher explanations from: %s", args.teacher_explanations)
        teacher_explanations = {str(r["id"]): r for r in read_jsonl(args.teacher_explanations)}
        logging.info("Loaded %d teacher explanation records for attribution evaluation", len(teacher_explanations))

    # Configure policy parameters based on policy_mode
    min_k = args.min_k
    max_k = args.max_k
    score_threshold = args.score_threshold
    adaptive_margin = args.adaptive_margin

    if args.policy_mode == "fixed_top_1":
        min_k, max_k = 1, 1
    elif args.policy_mode == "fixed_top_3":
        min_k, max_k = 3, 3
    elif args.policy_mode == "original_top_5":
        min_k, max_k = args.top_k_candidates, args.top_k_candidates

    policy = AdaptiveSelectorPolicy(
        min_k=min_k,
        max_k=max_k,
        score_threshold=score_threshold,
        adaptive_margin=adaptive_margin,
    )

    is_mock = args.mock
    if not is_mock and (args.selector / "mock_selector_config.json").is_file():
        logging.warning("Detected mock selector config in %s, falling back to mock scoring", args.selector)
        is_mock = True

    model = None
    if not is_mock:
        from sentence_transformers import CrossEncoder
        logging.info("Loading CrossEncoder selector: %s (device=%s)", args.selector, args.device)
        model = CrossEncoder(str(args.selector), device=args.device)

    filtered_rows: list[dict[str, Any]] = []
    k_distribution: Counter = Counter()
    gold_hits: list[float] = []

    # Attribution tracking
    teacher_coverages: list[float] = []
    teacher_precisions: list[float] = []
    recalls_at_1: list[float] = []
    recalls_at_2: list[float] = []
    recalls_at_3: list[float] = []

    # Batch prediction logic
    for row in tqdm(retrieval_rows, desc="Filtering evidence candidates"):
        cid = str(row["id"])
        claim_text = claims.get(cid, "")
        raw_candidates = row.get("retrieved_evidence_ids", [])[:args.top_k_candidates]
        valid_candidates = [eid for eid in raw_candidates if eid in documents]

        if not valid_candidates or not claim_text:
            filtered_rows.append({
                **row,
                "retrieved_evidence_ids": [],
                "retrieved_scores": [],
                "retrieval_confidence": 0.0,
                "original_k": len(valid_candidates),
                "selected_k": 0,
            })
            k_distribution[0] += 1
            continue

        pairs = [(claim_text, documents[eid]) for eid in valid_candidates]
        if is_mock:
            raw_scores = mock_score_claim_evidence_pairs(pairs)
        else:
            preds = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
            raw_scores = [float(s) for s in preds]

        # Check for Oracle Teacher selection mode
        t_row = teacher_explanations.get(cid)
        key_eids: set[str] = set()
        if t_row and t_row.get("is_valid") and t_row.get("grounded"):
            raw_indices = t_row.get("key_evidence_ids", [])
            retrieved_eids_in_exp = t_row.get("retrieved_evidence_ids", [])
            for idx in raw_indices:
                if 1 <= idx <= len(retrieved_eids_in_exp):
                    key_eids.add(retrieved_eids_in_exp[idx - 1])

        if args.policy_mode == "oracle_teacher" and key_eids:
            oracle_ids = [eid for eid in valid_candidates if eid in key_eids]
            if not oracle_ids:
                oracle_ids = [valid_candidates[0]]
            selected_ids = oracle_ids[:args.max_k]
            selected_scores = [raw_scores[valid_candidates.index(eid)] for eid in selected_ids]
        else:
            selected_ids, selected_scores = policy.select(valid_candidates, raw_scores)

        selected_k = len(selected_ids)
        k_distribution[selected_k] += 1

        # Evaluate Teacher Attribution metrics if ground truth teacher key exists
        if key_eids:
            sel_set = set(selected_ids)
            cov = len(sel_set & key_eids) / float(len(key_eids))
            prec = len(sel_set & key_eids) / float(len(sel_set)) if sel_set else 0.0
            teacher_coverages.append(cov)
            teacher_precisions.append(prec)

            # Measure ranking recall from raw_scores
            order = np.argsort(-np.asarray(raw_scores, dtype=np.float32)).tolist()
            rank_1_eid = valid_candidates[order[0]]
            recalls_at_1.append(float(rank_1_eid in key_eids))

            rank_2_eids = {valid_candidates[idx] for idx in order[:2]}
            recalls_at_2.append(float(bool(rank_2_eids & key_eids)))

            rank_3_eids = {valid_candidates[idx] for idx in order[:3]}
            recalls_at_3.append(float(bool(rank_3_eids & key_eids)))

        # Check gold evidence retention if gold is present
        gold_set = set(row.get("gold_evidence_ids", []))
        if gold_set:
            hit = any(eid in gold_set for eid in selected_ids)
            gold_hits.append(float(hit))

        filtered_rows.append({
            **row,
            "retrieved_evidence_ids": selected_ids,
            "retrieved_scores": selected_scores,
            "retrieval_confidence": selected_scores[0] if selected_scores else 0.0,
            "original_k": len(valid_candidates),
            "selected_k": selected_k,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in filtered_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logging.info("Filtered retrieval manifest written to: %s", args.output)

    avg_k = float(np.mean([r["selected_k"] for r in filtered_rows])) if filtered_rows else 0.0
    gold_recall = float(np.mean(gold_hits)) if gold_hits else None

    total_count = max(1, len(filtered_rows))
    k_dist_pct = {
        f"K={k}": round(count / total_count * 100, 2)
        for k, count in sorted(k_distribution.items())
    }

    attribution_metrics = None
    if teacher_coverages:
        attribution_metrics = {
            "grounded_claims_evaluated": len(teacher_coverages),
            "teacher_key_coverage_mean": float(np.mean(teacher_coverages)),
            "teacher_key_precision_mean": float(np.mean(teacher_precisions)),
            "pseudo_recall_at_1": float(np.mean(recalls_at_1)),
            "pseudo_recall_at_2": float(np.mean(recalls_at_2)),
            "pseudo_recall_at_3": float(np.mean(recalls_at_3)),
        }

    summary_payload = {
        "phase": "B18-B",
        "task": "adaptive_evidence_filtering",
        "selector": str(args.selector),
        "policy_mode": args.policy_mode,
        "total_claims": len(filtered_rows),
        "policy": {
            "min_k": min_k,
            "max_k": max_k,
            "score_threshold": score_threshold,
            "adaptive_margin": adaptive_margin,
            "top_k_candidates_evaluated": args.top_k_candidates,
        },
        "avg_selected_passages": avg_k,
        "k_distribution": dict(sorted(k_distribution.items())),
        "k_distribution_percent": k_dist_pct,
        "gold_recall_at_selected": gold_recall,
        "teacher_attribution": attribution_metrics,
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    logging.info("Diagnostic summary written to: %s", args.summary)
    logging.info("Filtering stats: avg_k=%.2f, distribution=%s", avg_k, dict(k_distribution))
    if attribution_metrics:
        logging.info(
            "Teacher attribution stats: Coverage=%.4f, Precision=%.4f, Recall@1=%.4f, Recall@2=%.4f, Recall@3=%.4f",
            attribution_metrics["teacher_key_coverage_mean"],
            attribution_metrics["teacher_key_precision_mean"],
            attribution_metrics["pseudo_recall_at_1"],
            attribution_metrics["pseudo_recall_at_2"],
            attribution_metrics["pseudo_recall_at_3"],
        )


if __name__ == "__main__":
    main()
