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

    policy = AdaptiveSelectorPolicy(
        min_k=args.min_k,
        max_k=args.max_k,
        score_threshold=args.score_threshold,
        adaptive_margin=args.adaptive_margin,
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

        selected_ids, selected_scores = policy.select(valid_candidates, raw_scores)
        selected_k = len(selected_ids)
        k_distribution[selected_k] += 1

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

    summary_payload = {
        "phase": "B18-B",
        "task": "adaptive_evidence_filtering",
        "selector": str(args.selector),
        "total_claims": len(filtered_rows),
        "policy": {
            "min_k": args.min_k,
            "max_k": args.max_k,
            "score_threshold": args.score_threshold,
            "adaptive_margin": args.adaptive_margin,
            "top_k_candidates_evaluated": args.top_k_candidates,
        },
        "avg_selected_passages": avg_k,
        "k_distribution": dict(sorted(k_distribution.items())),
        "gold_recall_at_selected": gold_recall,
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    logging.info("Diagnostic summary written to: %s", args.summary)
    logging.info("Filtering stats: avg_k=%.2f, distribution=%s", avg_k, dict(k_distribution))


if __name__ == "__main__":
    main()
