"""Rank claim-conditioned MOCHEG atomic evidence with dense/lexical/parent views."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from graphcure.retrieval import (FACT_CHECK_RETRIEVAL_INSTRUCTION,
                                 first_relevant_rank, reciprocal_rank_fusion,
                                 retrieval_confidence)
from scripts.prepare_mocheg_atomic_evidence import normalized_text, read_jsonl


def read_documents(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        return {row["evidence_id"].strip(): row.get("Evidence", "").strip()
                for row in csv.DictReader(handle) if row.get("evidence_id", "").strip()}


def token_overlap(query: str, document: str) -> float:
    q, d = set(normalized_text(query).split()), set(normalized_text(document).split())
    return len(q & d) / max(1, len(q | d))


def diverse_order(order: list[int], parents: list[str], limit: int,
                  max_per_parent: int) -> list[int]:
    counts: dict[str, int] = {}
    selected = []
    for index in order:
        parent = parents[index]
        if counts.get(parent, 0) >= max_per_parent:
            continue
        selected.append(index); counts[parent] = counts.get(parent, 0) + 1
        if len(selected) == limit:
            break
    return selected


def load_model(name: str, device: str, max_length: int) -> SentenceTransformer:
    kwargs = {"trust_remote_code": True}
    if device.startswith("cuda") and torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    try:
        model = SentenceTransformer(name, device=device, model_kwargs=kwargs)
    except TypeError:
        model = SentenceTransformer(name, device=device, trust_remote_code=True)
    model.max_seq_length = max_length
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=Path("data/processed/mocheg_atomic_manifest"))
    parser.add_argument("--candidate-root", type=Path, default=Path("outputs/retrieval_mocheg_atomic_candidates"))
    parser.add_argument("--corpus-root", type=Path, default=Path("data/processed/mocheg_atomic_corpus"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/retrieval_mocheg_atomic_dense"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/processed/retrieval_cache"))
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--instruction", default=FACT_CHECK_RETRIEVAL_INSTRUCTION)
    parser.add_argument("--output-k", type=int, default=50)
    parser.add_argument("--max-per-parent", type=int, default=3)
    parser.add_argument("--dense-weight", type=float, default=.65)
    parser.add_argument("--lexical-weight", type=float, default=.20)
    parser.add_argument("--parent-weight", type=float, default=.15)
    parser.add_argument("--rrf-k", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    if args.output_k <= 0 or args.max_per_parent <= 0:
        parser.error("--output-k and --max-per-parent must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model, args.device, args.max_length)
    signature = hashlib.sha256(json.dumps({
        "model": args.model, "instruction": args.instruction,
        "output_k": args.output_k, "max_per_parent": args.max_per_parent,
        "weights": [args.dense_weight, args.lexical_weight, args.parent_weight],
        "rrf_k": args.rrf_k,
    }, sort_keys=True).encode()).hexdigest()
    summary_path = args.output_root / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists() else {}
    )
    for split in args.splits:
        claims = {row["id"]: row for row in read_jsonl(args.manifest_root / f"{split}.jsonl")}
        candidates = read_jsonl(args.candidate_root / f"{split}.jsonl")
        documents = read_documents(args.corpus_root / split / "Corpus2.csv")
        used_ids = list(dict.fromkeys(value for row in candidates
                                      for value in row["retrieved_evidence_ids"]
                                      if value in documents))
        cache_key = hashlib.sha256((args.model + "\0" + "\0".join(used_ids)).encode()).hexdigest()[:16]
        cache = args.cache_root / f"atomic-{split}-{cache_key}.npy"
        if cache.exists():
            embeddings = np.load(cache)
        else:
            embeddings = model.encode([documents[value] for value in used_ids],
                batch_size=args.batch_size, normalize_embeddings=True,
                show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
            np.save(cache, embeddings)
        doc_index = {value: index for index, value in enumerate(used_ids)}
        ordered_claims = [claims[row["id"]] for row in candidates]
        queries = [f"Instruct: {args.instruction}\nQuery: {row.get('claim', '').strip()}"
                   for row in ordered_claims]
        query_embeddings = model.encode(queries, batch_size=args.batch_size,
            normalize_embeddings=True, show_progress_bar=True,
            convert_to_numpy=True).astype(np.float32)
        output, ranks = [], []
        for claim, candidate, query_embedding in tqdm(
            zip(ordered_claims, candidates, query_embeddings, strict=True),
            total=len(candidates), desc=f"{split} atomic ranking"
        ):
            valid_positions = [index for index, value in enumerate(candidate["retrieved_evidence_ids"])
                               if value in doc_index]
            ids = [candidate["retrieved_evidence_ids"][index] for index in valid_positions]
            parents = [candidate["parent_evidence_ids"][index] for index in valid_positions]
            parent_ranks = [candidate["parent_ranks"][index] for index in valid_positions]
            dense = np.asarray([float(query_embedding @ embeddings[doc_index[value]]) for value in ids])
            lexical = np.asarray([token_overlap(claim.get("claim", ""), documents[value]) for value in ids])
            dense_order = np.argsort(-dense, kind="stable").tolist()
            lexical_order = np.argsort(-lexical, kind="stable").tolist()
            parent_order = sorted(range(len(ids)), key=lambda index: (parent_ranks[index], index))
            fused = reciprocal_rank_fusion(
                [dense_order, lexical_order, parent_order],
                [args.dense_weight, args.lexical_weight, args.parent_weight],
                args.rrf_k,
            )
            selected = diverse_order([index for index, _ in fused], parents,
                                     args.output_k, args.max_per_parent)
            score_by_index = dict(fused)
            ranked_ids = [ids[index] for index in selected]
            values = [float(score_by_index[index]) for index in selected]
            gold = set(map(str, claim.get("text_evidence_ids", [])))
            rank = first_relevant_rank(ranked_ids, gold); ranks.append(rank)
            output.append({
                "id": claim["id"], "claim_id": claim["claim_id"],
                "label": int(claim["label"]), "retrieved_evidence_ids": ranked_ids,
                "retrieved_scores": values,
                "dense_scores": [float(dense[index]) for index in selected],
                "lexical_scores": [float(lexical[index]) for index in selected],
                "parent_evidence_ids": [parents[index] for index in selected],
                "retrieval_confidence": retrieval_confidence(values),
                "gold_evidence_ids": sorted(gold), "first_gold_rank": rank,
                "retrieval_signature": signature,
                "train_gold_injection": False, "validation_gold_injection": False,
            })
        (args.output_root / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
            encoding="utf-8")
        result = {"claims": len(output), "candidate_sentences": len(used_ids),
                  "model": args.model, "retrieval_signature": signature}
        for cutoff in sorted({min(value, args.output_k)
                              for value in (1, 5, 10, args.output_k)}):
            result[f"recall@{cutoff}"] = float(np.mean([
                rank is not None and rank <= cutoff for rank in ranks]))
        result["mrr"] = float(np.mean([1 / rank if rank else 0 for rank in ranks]))
        result["validation_gold_injection"] = False
        summary[split] = result; print(json.dumps({split: result}, indent=2))
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
