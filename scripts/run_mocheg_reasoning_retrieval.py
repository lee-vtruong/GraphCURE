"""Reasoning-aware hybrid candidate retrieval for the MOCHEG fixed corpus.

The script uses a modern instruction-tuned dense retriever and a lexical view.
It fuses *ranks* rather than incomparable raw scores, and caches document
embeddings so query/prompt ablations do not repeatedly encode the corpus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from graphcure.retrieval import (
    first_relevant_rank,
    reciprocal_rank_fusion,
    retrieval_confidence,
    top_indices,
)


DEFAULT_INSTRUCTION = (
    "Given a fact-checking claim, retrieve documents that provide direct, "
    "independent evidence to support, refute, or resolve the claim. Preserve "
    "entities, dates, locations, quantities, and negation."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_corpus(path: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            evidence_id = row["evidence_id"].strip()
            text = row.get("Evidence", "").replace("<p>", " ").replace("</p>", " ").strip()
            if evidence_id and text and evidence_id not in seen:
                seen.add(evidence_id)
                ids.append(evidence_id)
                texts.append(text)
    return ids, texts


def cache_key(model_name: str, ids: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode())
    for value in ids:
        digest.update(b"\0")
        digest.update(value.encode())
    return digest.hexdigest()[:16]


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


def encode_documents(
    model: SentenceTransformer,
    texts: list[str],
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached) == len(texts):
            print(f"reusing {cache_path} {cached.shape}")
            return cached
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/retrieval_mocheg_qwen3_hybrid"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/retrieval_cache"))
    parser.add_argument("--dense-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--output-k", type=int, default=50)
    parser.add_argument("--dense-weight", type=float, default=0.75)
    parser.add_argument("--lexical-weight", type=float, default=0.25)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    if args.output_k > args.candidate_k:
        parser.error("--output-k cannot exceed --candidate-k")
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = load_model(args.dense_model, args.device, args.max_length)
    summary_path = args.output_root / "summary.json"
    summary: dict[str, dict] = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists() else {}
    )

    for split in args.splits:
        claims = read_jsonl(args.manifest_root / f"{split}.jsonl")
        doc_ids, documents = read_corpus(args.raw_root / split / "Corpus2.csv")
        key = cache_key(args.dense_model, doc_ids)
        doc_embeddings = encode_documents(
            model,
            documents,
            args.cache_root / f"{split}-{key}.npy",
            args.batch_size,
        )
        queries = [
            f"Instruct: {args.instruction}\nQuery: {row.get('claim', '').strip()}"
            for row in claims
        ]
        query_embeddings = model.encode(
            queries,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        lexical = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.995,
            sublinear_tf=True,
            max_features=300_000,
            strip_accents="unicode",
        )
        doc_lexical = lexical.fit_transform(documents)
        query_lexical = lexical.transform([row.get("claim", "") for row in claims])
        output: list[dict] = []
        ranks: list[int | None] = []

        for start in tqdm(
            range(0, len(claims), args.score_batch_size),
            desc=f"{split} hybrid retrieval",
        ):
            end = min(start + args.score_batch_size, len(claims))
            dense_block = query_embeddings[start:end] @ doc_embeddings.T
            lexical_block = (query_lexical[start:end] @ doc_lexical.T).toarray()
            for offset, row in enumerate(claims[start:end]):
                dense_scores = dense_block[offset]
                lexical_scores = lexical_block[offset]
                dense_order = top_indices(dense_scores, args.candidate_k)
                lexical_order = top_indices(lexical_scores, args.candidate_k)
                fused = reciprocal_rank_fusion(
                    [dense_order, lexical_order],
                    weights=[args.dense_weight, args.lexical_weight],
                    rank_constant=args.rrf_k,
                    limit=args.output_k,
                )
                indices = [index for index, _ in fused]
                ids = [doc_ids[index] for index in indices]
                fused_scores = [float(score) for _, score in fused]
                gold = {str(value) for value in row.get("text_evidence_ids", [])}
                rank = first_relevant_rank(ids, gold)
                ranks.append(rank)
                output.append({
                    "id": row["id"],
                    "claim_id": row["claim_id"],
                    "label": int(row["label"]),
                    "retrieved_evidence_ids": ids,
                    "retrieved_scores": fused_scores,
                    "dense_scores": [float(dense_scores[index]) for index in indices],
                    "lexical_scores": [float(lexical_scores[index]) for index in indices],
                    "retrieval_confidence": retrieval_confidence(fused_scores),
                    "gold_evidence_ids": sorted(gold),
                    "first_gold_rank": rank,
                })

        target = args.output_root / f"{split}.jsonl"
        target.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
            encoding="utf-8",
        )
        split_summary = {
            "claims": len(claims),
            "corpus_documents": len(documents),
            "model": args.dense_model,
        }
        for cutoff in (1, 5, 10, args.output_k):
            cutoff = min(cutoff, args.output_k)
            split_summary[f"recall@{cutoff}"] = float(np.mean([
                rank is not None and rank <= cutoff for rank in ranks
            ]))
        split_summary["mrr"] = float(np.mean([
            1.0 / rank if rank is not None else 0.0 for rank in ranks
        ]))
        summary[split] = split_summary
        print(json.dumps({split: split_summary}, indent=2))

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
