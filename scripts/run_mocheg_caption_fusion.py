"""Retrieve MOCHEG images through pixel-derived descriptors and rank fusion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from graphcure.retrieval import first_relevant_rank, top_indices
from scripts.run_mocheg_reasoning_retrieval import load_model
from scripts.run_mocheg_visual_ensemble import (
    aligned_gold_images,
    fuse_visual_orders,
)
from scripts.run_mocheg_visual_retrieval import (
    image_corpus,
    read_jsonl,
    retrieval_summary,
)
from scripts.caption_mocheg_images import (
    DESCRIPTOR_TEXT_NORMALIZATION_VERSION,
    clean_descriptor,
)


CAPTION_RETRIEVAL_INSTRUCTION = (
    "Retrieve pixel-derived image descriptions that provide visual evidence "
    "for the fact-checking claim, including entities, events, scenes, visible "
    "text, quantities, dates, locations, and source-image reuse clues."
)


def read_descriptors(path: Path, image_names: list[str]) -> tuple[list[str], str]:
    rows = read_jsonl(path)
    by_id: dict[str, dict] = {}
    signatures: set[str] = set()
    for row in rows:
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in by_id:
            raise ValueError(f"missing or duplicate image_id in {path}: {image_id}")
        by_id[image_id] = row
        signatures.add(str(row.get("descriptor_signature", "")))
    missing = [name for name in image_names if name not in by_id]
    extra = sorted(set(by_id) - set(image_names))
    if missing or extra:
        raise ValueError(
            f"descriptor corpus mismatch: missing={len(missing)}, extra={len(extra)}; "
            "finish descriptor generation before retrieval"
        )
    if len(signatures) != 1 or "" in signatures:
        raise ValueError("descriptor file must contain one non-empty signature")
    return [
        clean_descriptor(str(by_id[name].get("descriptor", "")))
        for name in image_names
    ], signatures.pop()


def embedding_cache_path(cache_root: Path, split: str, model: str,
                         descriptor_signature: str) -> Path:
    key = hashlib.sha256(
        f"{model}:{descriptor_signature}".encode()
    ).hexdigest()[:16]
    return cache_root / f"mocheg-{split}-descriptor-text-{key}.npy"


def encode_documents(model, texts: list[str], target: Path,
                     batch_size: int) -> np.ndarray:
    if target.exists():
        cached = np.load(target, mmap_mode="r")
        if cached.ndim == 2 and len(cached) == len(texts):
            print(f"loading cached descriptor embeddings: {target}")
            return cached
        raise ValueError(f"invalid descriptor embedding cache: {target}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, embeddings)
    temporary.replace(target)
    return embeddings


def candidate_union_diagnostics(direct_hits: list[bool],
                                caption_hits: list[bool],
                                union_hits: list[bool],
                                annotated: list[bool]) -> dict:
    if not (
        len(direct_hits) == len(caption_hits) == len(union_hits) == len(annotated)
    ):
        raise ValueError("candidate diagnostic vectors must align")
    indices = [index for index, value in enumerate(annotated) if value]
    conditional = lambda values: (
        float(np.mean([values[index] for index in indices])) if indices else 0.0
    )
    caption_only = sum(
        annotated[index] and caption_hits[index] and not direct_hits[index]
        for index in range(len(annotated))
    )
    return {
        "raw_direct_candidate_recall": float(np.mean(direct_hits)),
        "raw_caption_candidate_recall": float(np.mean(caption_hits)),
        "raw_union_candidate_recall": float(np.mean(union_hits)),
        "conditional_direct_candidate_recall": conditional(direct_hits),
        "conditional_caption_candidate_recall": conditional(caption_hits),
        "conditional_union_candidate_recall": conditional(union_hits),
        "caption_only_gold_recoveries": int(caption_only),
        "claims_with_gold_images": len(indices),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--descriptor-root", type=Path,
                        default=Path("data/processed/mocheg_visual_descriptors"))
    parser.add_argument("--direct-root", type=Path,
                        default=Path("outputs/retrieval_mocheg_qwen3vl_images_top200"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/retrieval_mocheg_caption_fusion"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/retrieval_cache"))
    parser.add_argument("--text-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--instruction", default=CAPTION_RETRIEVAL_INSTRUCTION)
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--output-k", type=int, default=200)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--fusion-weights", type=float, nargs=3,
                        default=[1.0, 1.0, 0.5],
                        metavar=("DIRECT", "DENSE", "LEXICAL"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["val"])
    args = parser.parse_args()
    if args.candidate_k <= 0 or args.output_k <= 0:
        parser.error("candidate and output depths must be positive")
    if any(weight <= 0 for weight in args.fusion_weights):
        parser.error("all fusion weights must be positive")
    if "test" in args.splits:
        parser.error("test is locked until the multimodal configuration freezes")

    args.output_root.mkdir(parents=True, exist_ok=True)
    model = load_model(args.text_model, args.device, args.max_length)
    summary_path = args.output_root / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists() else {}
    )

    for split in args.splits:
        claims = read_jsonl(args.manifest_root / f"{split}.jsonl")
        image_names, _ = image_corpus(args.raw_root / split / "images")
        if args.candidate_k > len(image_names):
            parser.error(f"--candidate-k exceeds the {split} image corpus size")
        descriptors, descriptor_sig = read_descriptors(
            args.descriptor_root / f"{split}.jsonl", image_names
        )
        direct_rows = read_jsonl(args.direct_root / f"{split}.jsonl")
        direct_by_id = {row["id"]: row for row in direct_rows}
        if set(direct_by_id) != {row["id"] for row in claims}:
            raise ValueError("direct visual retrieval does not align with manifest")

        document_embeddings = encode_documents(
            model,
            descriptors,
            embedding_cache_path(
                args.cache_root, split, args.text_model, descriptor_sig
            ),
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
        descriptor_lexical = lexical.fit_transform(descriptors)
        query_lexical = lexical.transform(
            [str(row.get("claim", "")) for row in claims]
        )
        score_device = torch.device(
            args.device if args.device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        document_tensor = torch.tensor(
            np.asarray(document_embeddings), dtype=torch.float32,
            device=score_device,
        )
        query_tensor = torch.tensor(
            query_embeddings, dtype=torch.float32, device=score_device
        )
        image_index = {name: index for index, name in enumerate(image_names)}
        corpus_names = set(image_names)
        output: list[dict] = []
        annotated: list[bool] = []
        ranks: dict[str, list[int | None]] = {
            "direct": [], "caption_dense": [],
            "caption_lexical": [], "fused": [],
        }
        direct_hits: list[bool] = []
        caption_hits: list[bool] = []
        union_hits: list[bool] = []

        for start in tqdm(
            range(0, len(claims), args.score_batch_size),
            desc=f"{split} pixel-descriptor fusion",
        ):
            end = min(start + args.score_batch_size, len(claims))
            dense_block = (
                query_tensor[start:end] @ document_tensor.T
            ).cpu().numpy()
            lexical_block = (
                query_lexical[start:end] @ descriptor_lexical.T
            ).toarray()
            for offset, claim in enumerate(claims[start:end]):
                gold = aligned_gold_images(claim, corpus_names)
                dense_order = top_indices(dense_block[offset], args.candidate_k)
                lexical_order = top_indices(
                    lexical_block[offset], args.candidate_k
                )
                direct_ids = direct_by_id[claim["id"]]["retrieved_image_ids"]
                direct_order = np.asarray([
                    image_index[value] for value in direct_ids
                    if value in image_index
                ], dtype=np.int64)
                orders = [direct_order, dense_order, lexical_order]
                fused = fuse_visual_orders(
                    orders, args.fusion_weights, args.rrf_k, args.output_k
                )
                fused_indices = [index for index, _ in fused]
                fused_ids = [image_names[index] for index in fused_indices]
                named_orders = {
                    "direct": direct_order,
                    "caption_dense": dense_order,
                    "caption_lexical": lexical_order,
                }
                direct_set = {
                    image_names[index] for index in direct_order
                }
                caption_set = {
                    image_names[index]
                    for order in (dense_order, lexical_order) for index in order
                }
                direct_hits.append(bool(gold & direct_set))
                caption_hits.append(bool(gold & caption_set))
                union_hits.append(bool(gold & (direct_set | caption_set)))
                per_view_rank = {}
                for name, order in named_orders.items():
                    ids = [image_names[index] for index in order]
                    rank = first_relevant_rank(ids, gold)
                    ranks[name].append(rank)
                    per_view_rank[name] = rank
                fused_rank = first_relevant_rank(fused_ids, gold)
                ranks["fused"].append(fused_rank)
                annotated.append(bool(gold))
                output.append({
                    "id": claim["id"],
                    "claim_id": claim["claim_id"],
                    "label": int(claim["label"]),
                    "retrieved_image_ids": fused_ids,
                    "retrieved_image_scores": [
                        float(score) for _, score in fused
                    ],
                    "gold_image_ids": sorted(gold),
                    "first_gold_image_rank": fused_rank,
                    "per_view_first_gold_rank": per_view_rank,
                })

        # Preserve the standard depth curve when materializing a larger pool
        # for reranking. Previously an ``--output-k 400`` run omitted the
        # intermediate Recall@200/300 values needed to select the smallest
        # candidate pool that satisfies the validation-only ceiling gate.
        cutoffs = sorted(
            cutoff
            for cutoff in {1, 5, 10, 50, 100, 200, 300, 400, args.output_k}
            if cutoff <= args.output_k
        )
        signature_payload = {
            "text_model": args.text_model,
            "instruction": args.instruction,
            "descriptor_signature": descriptor_sig,
            "candidate_k": args.candidate_k,
            "output_k": args.output_k,
            "rrf_k": args.rrf_k,
            "fusion_weights": args.fusion_weights,
            "descriptor_text_normalization": (
                DESCRIPTOR_TEXT_NORMALIZATION_VERSION
            ),
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode()
        ).hexdigest()
        split_summary = retrieval_summary(ranks["fused"], annotated, cutoffs)
        split_summary.update({
            "image_corpus": len(image_names),
            "text_model": args.text_model,
            "descriptor_signature": descriptor_sig,
            "fusion_signature": signature,
            "fusion_weights": args.fusion_weights,
            "descriptor_text_normalization": (
                DESCRIPTOR_TEXT_NORMALIZATION_VERSION
            ),
            "candidate_k": args.candidate_k,
            "output_k": args.output_k,
            "per_view": {
                name: retrieval_summary(
                    values,
                    annotated,
                    [10, 50, 100, 200],
                )
                for name, values in ranks.items() if name != "fused"
            },
            "candidate_union": candidate_union_diagnostics(
                direct_hits, caption_hits, union_hits, annotated
            ),
        })
        (args.output_root / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
            encoding="utf-8",
        )
        summary[split] = split_summary
        print(json.dumps({split: split_summary}, indent=2))

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
