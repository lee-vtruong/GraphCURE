"""Encode claim-conditioned visual reports into multimodal verifier caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scripts.cache_mocheg_reasoning_features import load_encoder, read_jsonl


REPORT_INSTRUCTION = (
    "Encode a claim-conditioned visual evidence report for multimodal fact "
    "verification. Preserve visible entities, OCR, temporal/provenance clues, "
    "evidence relation, uncertainty, and sufficiency."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report_features(ranks: list[int], scores: list[float]) -> list[list[float]]:
    """Inference-available rank/score features; injected rank zero stays zero."""
    observed = [float(score) for rank, score in zip(ranks, scores, strict=True)
                if rank > 0]
    lower, upper = (min(observed), max(observed)) if observed else (0.0, 0.0)
    maximum_rank = max([rank for rank in ranks if rank > 0], default=1)
    result = []
    for rank, score in zip(ranks, scores, strict=True):
        if rank <= 0:
            result.append([0.0, 0.0, 0.0])
            continue
        normalized = ((float(score) - lower) / (upper - lower)
                      if upper > lower else 1.0)
        percentile = 1.0 - (rank - 1) / maximum_rank
        result.append([1.0 / rank, normalized, percentile])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-cache-root", type=Path, default=Path(
        "data/processed/mocheg_reasoning_cache"))
    parser.add_argument("--report-root", type=Path, default=Path(
        "data/processed/mocheg_claim_visual_reports_v10"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "data/processed/mocheg_visual_report_cache_v10"))
    parser.add_argument("--encoder", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"),
                        default=("train", "val"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = load_encoder(args.encoder, args.device, args.max_length)

    signatures = set()
    for split in args.splits:
        text_path = args.text_cache_root / f"{split}.pt"
        report_path = args.report_root / f"{split}.jsonl"
        report_metadata_path = args.report_root / f"{split}.metadata.json"
        text = torch.load(text_path, map_location="cpu", weights_only=False)
        rows = read_jsonl(report_path)
        metadata = json.loads(report_metadata_path.read_text(encoding="utf-8"))
        signature = metadata.get("report_signature")
        signatures.add(signature)
        if text["ids"] != [row["id"] for row in rows]:
            raise ValueError(f"{split}: text cache and visual reports do not align")
        if text["metadata"].get("encoder") != args.encoder:
            raise ValueError(
                f"{split}: use the text-cache encoder "
                f"{text['metadata'].get('encoder')}, not {args.encoder}")
        top_k = int(metadata["top_k"])
        if any(len(row.get("reports", [])) != top_k for row in rows):
            raise ValueError(f"{split}: incomplete report candidate set")
        flat = [
            f"Instruct: {REPORT_INSTRUCTION}\nDocument: {report}"
            for row in rows for report in row["reports"]
        ]
        print(f"{split}: encoding {len(flat)} claim-conditioned reports")
        encoded = model.encode(
            flat, batch_size=args.batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        dimension = int(encoded.shape[-1])
        if dimension != int(text["metadata"]["embedding_dim"]):
            raise ValueError("report and text embedding dimensions differ")
        visual = torch.from_numpy(encoded).reshape(
            len(rows), top_k, dimension).half()
        visual_mask = torch.ones(len(rows), top_k, dtype=torch.bool)
        retrieval_features = torch.zeros(len(rows), top_k, 3)
        relevance = torch.zeros(len(rows), top_k)
        relevance_weights = torch.ones(len(rows), top_k)
        for index, row in enumerate(tqdm(rows, desc=f"{split} assemble reports")):
            ranks = [int(value) for value in row["retrieval_ranks"]]
            scores = [float(value) for value in row["retrieval_scores"]]
            flags = [bool(value) for value in row["gold_flags"]]
            retrieval_features[index] = torch.tensor(
                report_features(ranks, scores), dtype=torch.float32)
            relevance[index] = torch.tensor(flags, dtype=torch.float32)
            # Non-qrel images are weak negatives because annotations may be
            # incomplete. Downweight them instead of asserting irrelevance.
            relevance_weights[index] = torch.tensor(
                [1.0 if flag else 0.25 for flag in flags])
        payload = {
            "version": 1,
            "ids": text["ids"], "labels": text["labels"],
            "claim_embeddings": text["claim_embeddings"],
            "text_evidence_embeddings": text["evidence_embeddings"],
            "text_mask": text["evidence_mask"],
            "text_retrieval_features": text["retrieval_features"],
            "text_relevance": text["relevance"],
            "text_relevance_weights": text["relevance_weights"],
            "visual_evidence_embeddings": visual,
            "visual_mask": visual_mask,
            "visual_retrieval_features": retrieval_features,
            "visual_relevance": relevance,
            "visual_relevance_weights": relevance_weights,
            "metadata": {
                "split": split, "samples": len(rows),
                "claim_dim": int(text["claim_embeddings"].shape[-1]),
                "text_dim": int(text["evidence_embeddings"].shape[-1]),
                "visual_dim": dimension,
                "text_top_k": int(text["evidence_embeddings"].shape[1]),
                "visual_top_k": top_k,
                "visual_model": f"{metadata['model']} reports -> {args.encoder}",
                "report_signature": signature,
                "report_encoder": args.encoder,
                "train_gold_injection": bool(
                    metadata.get("train_gold_injection", False)),
                "validation_gold_injection": False,
                "report_file": str(report_path),
                "report_sha256": sha256(report_path),
                "text_cache_sha256": sha256(text_path),
            },
        }
        target = args.output_root / f"{split}.pt"
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        result = {
            "saved": str(target), **payload["metadata"],
            "visual_gold_coverage": float(
                relevance.bool().any(1).float().mean()),
        }
        target.with_suffix(".metadata.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    if len(signatures) != 1:
        raise ValueError(f"train/val report signature mismatch: {signatures}")


if __name__ == "__main__":
    main()
