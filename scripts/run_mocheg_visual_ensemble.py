"""Validation-first constraint-aligned visual retrieval for MOCHEG.

The ensemble reuses one frozen image embedding corpus and embeds each claim
under four GraphCURE constraint-specific retrieval prompts. Candidate ranks are
combined with reciprocal-rank fusion, avoiding dataset filename/topic leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from graphcure.retrieval import (
    first_relevant_rank,
    reciprocal_rank_fusion,
    top_indices,
)
from scripts.run_mocheg_visual_retrieval import (
    encode_images,
    encode_queries,
    image_corpus,
    read_jsonl,
    retrieval_summary,
)


CONSTRAINT_VISUAL_VIEWS = {
    "semantic": (
        "Retrieve images that semantically depict the objects, actions, scene, "
        "or situation described by the fact-checking claim."
    ),
    "entity": (
        "Retrieve images of the same named people, organizations, landmarks, "
        "products, animals, or other entities mentioned in the claim."
    ),
    "temporal_provenance": (
        "Retrieve the source image or visually related images of the same "
        "real-world event, including images reused with a different time, "
        "place, or event description."
    ),
    "contextual_ocr": (
        "Retrieve screenshots, memes, documents, charts, maps, and social-media "
        "images whose visible text or contextual details relate to the claim."
    ),
}


def aligned_gold_images(claim: dict, corpus_names: set[str]) -> set[str]:
    return {
        str(value)
        for value in (
            claim.get("image_candidate_names", [])
            + claim.get("image_evidence_ids", [])
        )
        if str(value) in corpus_names
    }


def fuse_visual_orders(orders: list[np.ndarray], weights: list[float],
                       rank_constant: float, limit: int) -> list[tuple[int, float]]:
    return reciprocal_rank_fusion(
        orders, weights=weights, rank_constant=rank_constant, limit=limit
    )


def main() -> None:
    from sentence_transformers import SentenceTransformer
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/retrieval_mocheg_qwen3vl_constraint_ensemble"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/retrieval_cache"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--output-k", type=int, default=200)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--view-weights", type=float, nargs=4,
                        default=[1.0, 1.0, 1.0, 1.0],
                        metavar=("SEM", "ENT", "TEMP", "CTX"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-shard-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["val"])
    args = parser.parse_args()

    if args.candidate_k <= 0 or args.output_k <= 0:
        parser.error("candidate and output depths must be positive")
    if args.output_k > args.candidate_k * len(CONSTRAINT_VISUAL_VIEWS):
        parser.error("--output-k exceeds the maximum candidate union")
    if any(weight <= 0 for weight in args.view_weights):
        parser.error("all --view-weights must be positive")
    if "test" in args.splits:
        parser.error("test is locked until the visual retrieval configuration freezes")

    args.output_root.mkdir(parents=True, exist_ok=True)
    # Match the direct visual-retrieval model construction exactly so its
    # frozen image embedding cache remains numerically compatible.
    model = SentenceTransformer(args.model, device=args.device)
    summary_path = args.output_root / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists() else {}
    )
    view_names = list(CONSTRAINT_VISUAL_VIEWS)
    instructions = list(CONSTRAINT_VISUAL_VIEWS.values())
    signature_payload = {
        "model": args.model,
        "views": CONSTRAINT_VISUAL_VIEWS,
        "view_weights": args.view_weights,
        "candidate_k": args.candidate_k,
        "output_k": args.output_k,
        "rrf_k": args.rrf_k,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()

    for split in args.splits:
        claims = read_jsonl(args.manifest_root / f"{split}.jsonl")
        image_names, image_paths = image_corpus(args.raw_root / split / "images")
        if args.candidate_k > len(image_names):
            parser.error(f"--candidate-k exceeds the {split} image corpus size")
        image_embeddings = encode_images(
            model, image_names, image_paths, args.cache_root, split,
            args.model, args.batch_size, args.image_shard_size,
        )
        query_views = [
            encode_queries(
                model, claims, args.cache_root, split, args.model,
                instruction, args.query_batch_size,
            )
            for instruction in instructions
        ]
        score_device = torch.device(
            args.device if args.device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        # Copy the read-only NumPy caches once, then perform the expensive
        # multi-view matrix products on the selected accelerator.
        image_score_embeddings = torch.tensor(
            np.asarray(image_embeddings), dtype=torch.float32, device=score_device
        )
        query_score_views = [
            torch.tensor(
                np.asarray(queries), dtype=torch.float32, device=score_device
            )
            for queries in query_views
        ]
        print(
            f"{split}: similarity scoring on {score_device} with "
            f"{len(query_score_views)} constraint views"
        )

        corpus_names = set(image_names)
        output: list[dict] = []
        annotated: list[bool] = []
        fused_ranks: list[int | None] = []
        view_ranks: dict[str, list[int | None]] = {
            name: [] for name in view_names
        }

        for start in tqdm(
            range(0, len(claims), args.score_batch_size),
            desc=f"{split} constraint-aligned visual fusion",
        ):
            end = min(start + args.score_batch_size, len(claims))
            score_blocks = [
                (queries[start:end] @ image_score_embeddings.T).cpu().numpy()
                for queries in query_score_views
            ]
            for offset, claim in enumerate(claims[start:end]):
                gold = aligned_gold_images(claim, corpus_names)
                orders = [
                    top_indices(scores[offset], args.candidate_k)
                    for scores in score_blocks
                ]
                fused = fuse_visual_orders(
                    orders, args.view_weights, args.rrf_k, args.output_k
                )
                indices = [index for index, _ in fused]
                ids = [image_names[index] for index in indices]
                rank = first_relevant_rank(ids, gold)
                fused_ranks.append(rank)
                annotated.append(bool(gold))

                per_view_rank: dict[str, int | None] = {}
                for name, order in zip(view_names, orders, strict=True):
                    view_ids = [image_names[index] for index in order]
                    view_rank = first_relevant_rank(view_ids, gold)
                    view_ranks[name].append(view_rank)
                    per_view_rank[name] = view_rank

                output.append({
                    "id": claim["id"],
                    "claim_id": claim["claim_id"],
                    "label": int(claim["label"]),
                    "retrieved_image_ids": ids,
                    "retrieved_image_scores": [
                        float(score) for _, score in fused
                    ],
                    "gold_image_ids": sorted(gold),
                    "first_gold_image_rank": rank,
                    "per_view_first_gold_rank": per_view_rank,
                    "visual_retrieval_model": args.model,
                    "visual_retrieval_signature": signature,
                })

        cutoffs = sorted(
            cutoff for cutoff in {1, 5, 10, 50, 100, args.output_k}
            if cutoff <= args.output_k
        )
        split_summary = retrieval_summary(
            fused_ranks, annotated, cutoffs
        )
        split_summary.update({
            "image_corpus": len(image_names),
            "model": args.model,
            "candidate_k_per_view": args.candidate_k,
            "output_k": args.output_k,
            "rrf_k": args.rrf_k,
            "views": CONSTRAINT_VISUAL_VIEWS,
            "view_weights": args.view_weights,
            "per_view": {
                name: retrieval_summary(
                    ranks,
                    annotated,
                    [
                        cutoff for cutoff in (10, 50, 100, 200, 300)
                        if cutoff <= args.candidate_k
                    ],
                )
                for name, ranks in view_ranks.items()
            },
            "visual_retrieval_signature": signature,
        })
        target = args.output_root / f"{split}.jsonl"
        target.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
            encoding="utf-8",
        )
        summary[split] = split_summary
        print(json.dumps({split: split_summary}, indent=2))

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
