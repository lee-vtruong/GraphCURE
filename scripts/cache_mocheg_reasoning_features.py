"""Cache frozen claim/evidence embeddings for fast GraphCURE-R2V screening."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from graphcure.retrieval import (
    FACT_CHECK_RETRIEVAL_INSTRUCTION,
    evidence_candidate_features,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_docs(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            evidence_id = row["evidence_id"].strip()
            text = row.get("Evidence", "").replace("<p>", " ").replace("</p>", " ").strip()
            if evidence_id and text:
                result.setdefault(evidence_id, text)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_encoder(name: str, device: str, max_length: int) -> SentenceTransformer:
    model_kwargs = {"trust_remote_code": True}
    if device.startswith("cuda") and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    try:
        model = SentenceTransformer(
            name, device=device, model_kwargs=model_kwargs
        )
    except TypeError:
        model = SentenceTransformer(name, device=device, trust_remote_code=True)
    model.max_seq_length = max_length
    return model


def inject_gold_candidate(
    claim: dict, candidates: list[str], documents: dict[str, str], top_k: int
) -> tuple[list[str], bool]:
    """Inject one train-only gold document at a deterministic non-label rank."""
    result = list(dict.fromkeys(candidates))[:top_k]
    gold = sorted({
        str(value) for value in claim.get("text_evidence_ids", [])
        if documents.get(str(value))
    })
    if not gold or any(value in gold for value in result):
        return result, False
    digest = int(hashlib.sha256(str(claim["id"]).encode()).hexdigest(), 16)
    selected = gold[digest % len(gold)]
    position = (digest // max(1, len(gold))) % top_k
    result.insert(position, selected)
    return list(dict.fromkeys(result))[:top_k], True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/processed/mocheg_reasoning_cache"))
    parser.add_argument("--encoder", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--inject-train-gold", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    model = load_encoder(args.encoder, args.device, args.max_length)
    for split in args.splits:
        manifest_path = args.manifest_root / f"{split}.jsonl"
        retrieval_path = args.retrieval_root / f"{split}.jsonl"
        claims = {row["id"]: row for row in read_jsonl(manifest_path)}
        retrieval_rows = read_jsonl(retrieval_path)
        documents = read_docs(args.raw_root / split / "Corpus2.csv")
        ordered_claims = [claims[row["id"]] for row in retrieval_rows]
        query_texts = [
            f"Instruct: {FACT_CHECK_RETRIEVAL_INSTRUCTION}\nQuery: "
            f"{row.get('claim', '').strip()}"
            for row in ordered_claims
        ]
        natural_candidate_ids = [
            [
                evidence_id
                for evidence_id in row.get("retrieved_evidence_ids", [])[:args.top_k]
                if documents.get(evidence_id)
            ]
            for row in retrieval_rows
        ]
        injected = 0
        candidate_ids = []
        for claim, candidates in zip(
            ordered_claims, natural_candidate_ids, strict=True
        ):
            if args.inject_train_gold and split == "train":
                candidates, changed = inject_gold_candidate(
                    claim, candidates, documents, args.top_k
                )
                injected += int(changed)
            candidate_ids.append(candidates)
        unique_ids = list(dict.fromkeys(
            evidence_id for values in candidate_ids for evidence_id in values
        ))
        print(
            f"{split}: encoding {len(query_texts)} claims and "
            f"{len(unique_ids)} unique evidence documents"
        )
        claim_embeddings = model.encode(
            query_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        document_embeddings = model.encode(
            [documents[evidence_id] for evidence_id in unique_ids],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        document_index = {
            evidence_id: index for index, evidence_id in enumerate(unique_ids)
        }
        dimension = int(claim_embeddings.shape[1])
        evidence_embeddings = torch.zeros(
            len(retrieval_rows), args.top_k, dimension, dtype=torch.float16
        )
        evidence_mask = torch.zeros(
            len(retrieval_rows), args.top_k, dtype=torch.bool
        )
        retrieval_features = torch.zeros(
            len(retrieval_rows), args.top_k, 6, dtype=torch.float32
        )
        relevance = torch.zeros(
            len(retrieval_rows), args.top_k, dtype=torch.float32
        )
        relevance_weights = torch.zeros_like(relevance)

        for row_index, (claim, retrieval, ids) in enumerate(tqdm(
            zip(ordered_claims, retrieval_rows, candidate_ids, strict=True),
            total=len(retrieval_rows),
            desc=f"{split} assemble cache",
        )):
            gold = {str(value) for value in claim.get("text_evidence_ids", [])}
            raw_ids = retrieval.get("retrieved_evidence_ids", [])[:args.top_k]
            raw_scores = retrieval.get("retrieved_scores", [])[:args.top_k]
            score_by_id = {
                evidence_id: float(raw_scores[index]) if index < len(raw_scores) else 0.0
                for index, evidence_id in enumerate(raw_ids)
            }
            scores = [score_by_id.get(evidence_id, 0.0) for evidence_id in ids]
            top_score = scores[0] if scores else 0.0
            for rank_index, evidence_id in enumerate(ids[:args.top_k]):
                is_gold = evidence_id in gold
                features, weight = evidence_candidate_features(
                    claim.get("claim", ""),
                    documents[evidence_id],
                    scores[rank_index],
                    top_score,
                    rank_index + 1,
                    is_gold,
                )
                evidence_embeddings[row_index, rank_index] = torch.from_numpy(
                    document_embeddings[document_index[evidence_id]]
                ).half()
                evidence_mask[row_index, rank_index] = True
                retrieval_features[row_index, rank_index] = torch.tensor(features)
                relevance[row_index, rank_index] = float(is_gold)
                relevance_weights[row_index, rank_index] = weight

        payload = {
            "version": 1,
            "ids": [row["id"] for row in ordered_claims],
            "claim_embeddings": torch.from_numpy(claim_embeddings).half(),
            "evidence_embeddings": evidence_embeddings,
            "evidence_mask": evidence_mask,
            "retrieval_features": retrieval_features,
            "relevance": relevance,
            "relevance_weights": relevance_weights,
            "labels": torch.tensor(
                [int(row["label"]) for row in ordered_claims], dtype=torch.long
            ),
            "metadata": {
                "split": split,
                "samples": len(retrieval_rows),
                "encoder": args.encoder,
                "embedding_dim": dimension,
                "top_k": args.top_k,
                "max_length": args.max_length,
                "retrieval_file": str(retrieval_path),
                "retrieval_sha256": sha256(retrieval_path),
                "manifest_file": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "git_commit": git_commit(),
                "train_gold_injection": bool(
                    args.inject_train_gold and split == "train"
                ),
                "injected_claims": injected,
            },
        }
        target = args.output_root / f"{split}.pt"
        torch.save(payload, target)
        print(
            json.dumps({
                "saved": str(target),
                **payload["metadata"],
                "gold_coverage": float(relevance.bool().any(dim=1).float().mean()),
            }, indent=2)
        )


if __name__ == "__main__":
    main()
