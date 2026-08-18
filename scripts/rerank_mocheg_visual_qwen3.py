"""Validation-first multimodal reranking of MOCHEG image candidates.

Each candidate document contains the original pixels and a claim-independent
pixel descriptor. Qwen3-VL-Reranker scores claim--document relevance with
cross-attention. Gold image identifiers are used only after scoring to report
retrieval metrics; they never enter model inputs.

The output is appended one claim at a time and can be resumed safely. Test is
intentionally locked until the visual configuration is frozen on validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from graphcure.retrieval import first_relevant_rank
from scripts.run_mocheg_caption_fusion import read_descriptors
from scripts.run_mocheg_visual_ensemble import aligned_gold_images
from scripts.run_mocheg_visual_retrieval import (
    image_corpus,
    normalize_image_paths,
    retrieval_summary,
)


VISUAL_RERANK_INSTRUCTION = (
    "Rank the candidate image by its utility as evidence for verifying the "
    "claim. Check depicted people, objects, actions, event, location, time, "
    "visible text, quantities, document or meme context, and possible image "
    "reuse. Visual or lexical similarity alone is insufficient."
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_resumable_jsonl(path: Path) -> list[dict]:
    """Read JSONL while tolerating only an interrupted final line."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            print(f"warning: ignoring interrupted final JSONL line in {path}")
    return rows


def stable_rerank(candidate_ids: list[str], scores: np.ndarray,
                  output_k: int) -> tuple[list[str], list[float]]:
    if len(candidate_ids) != len(scores):
        raise ValueError("candidate IDs and reranker scores must align")
    order = np.argsort(-np.asarray(scores), kind="stable")[:output_k]
    return (
        [candidate_ids[int(index)] for index in order],
        [float(scores[int(index)]) for index in order],
    )


def score_in_chunks(model: Any, pairs: list[tuple[Any, Any]], batch_size: int,
                    chunk_size: int, progress: tqdm) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start:start + chunk_size]
        predicted = model.predict(
            chunk,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        values = np.asarray(predicted, dtype=np.float32).reshape(-1)
        if len(values) != len(chunk):
            raise ValueError(
                "reranker returned a non-scalar score for each visual pair"
            )
        scores.append(values)
        progress.update(len(chunk))
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def main() -> None:
    import torch
    from sentence_transformers import CrossEncoder

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--descriptor-root", type=Path,
                        default=Path("data/processed/mocheg_visual_descriptors_v3"))
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/retrieval_mocheg_visual_reranked"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/retrieval_cache"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Reranker-2B")
    parser.add_argument("--instruction", default=VISUAL_RERANK_INSTRUCTION)
    parser.add_argument("--candidate-k", type=int, default=400)
    parser.add_argument("--output-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-chunk-size", type=int, default=16)
    # Qwen's official multimodal reranker uses 10,240 tokens by default.
    # A 1,024-token cap can truncate the expanded visual-token sequence and
    # make Transformers reject the input because image placeholder counts no
    # longer match. Dynamic padding means this is a cap, not forced padding of
    # every pair to 10,240 tokens.
    parser.add_argument("--max-length", type=int, default=10240)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--limit-claims", type=int, default=0,
                        help="Smoke-test only; zero processes the full split")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    args = parser.parse_args()

    if args.candidate_k <= 0 or args.output_k <= 0:
        parser.error("--candidate-k and --output-k must be positive")
    if args.output_k > args.candidate_k:
        parser.error("--output-k cannot exceed --candidate-k")
    if args.batch_size <= 0 or args.score_chunk_size <= 0:
        parser.error("batch and chunk sizes must be positive")
    if args.flush_every <= 0 or args.limit_claims < 0:
        parser.error("--flush-every must be positive and --limit-claims non-negative")
    if "test" in args.splits:
        parser.error(
            "test is locked until the validation visual reranker is frozen"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists() else {}
    )
    reranker = None

    for split in args.splits:
        manifest_path = args.manifest_root / f"{split}.jsonl"
        retrieval_path = args.retrieval_root / f"{split}.jsonl"
        descriptor_path = args.descriptor_root / f"{split}.jsonl"
        claims = read_resumable_jsonl(manifest_path)
        if args.limit_claims:
            claims = claims[:args.limit_claims]
        claim_ids = {row["id"] for row in claims}
        retrieved = [
            row for row in read_resumable_jsonl(retrieval_path)
            if row.get("id") in claim_ids
        ]
        if [row["id"] for row in retrieved] != [row["id"] for row in claims]:
            raise ValueError(
                f"{split}: retrieval rows do not align with the selected manifest"
            )

        image_names, image_paths = image_corpus(args.raw_root / split / "images")
        normalized_paths, converted = normalize_image_paths(
            image_paths, args.cache_root
        )
        if converted:
            print(f"{split}: using {converted} normalized image file(s)")
        descriptors, descriptor_signature = read_descriptors(
            descriptor_path, image_names
        )
        descriptor_by_id = dict(zip(image_names, descriptors, strict=True))
        path_by_id = dict(zip(image_names, normalized_paths, strict=True))
        corpus_names = set(image_names)

        signature_payload = {
            "model": args.model,
            "instruction": args.instruction,
            "candidate_k": args.candidate_k,
            "output_k": args.output_k,
            "max_length": args.max_length,
            "retrieval_sha256": file_sha256(retrieval_path),
            "descriptor_signature": descriptor_signature,
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode()
        ).hexdigest()
        target = args.output_root / f"{split}.jsonl"
        output = (
            read_resumable_jsonl(target)
            if args.resume and target.exists() else []
        )
        if output and any(
            row.get("visual_reranker_signature") != signature for row in output
        ):
            parser.error(
                f"cannot resume {target}: settings or inputs changed; use a "
                "new output root or rerun without --resume"
            )
        if len({row["id"] for row in output}) != len(output):
            parser.error(f"cannot resume {target}: duplicate claim IDs")
        completed = {row["id"] for row in output}
        pending = [
            (claim, retrieval)
            for claim, retrieval in zip(claims, retrieved, strict=True)
            if claim["id"] not in completed
        ]
        if completed:
            print(f"{split}: resuming after {len(completed)} completed claims")
        # Canonicalize before append. If the previous process died halfway
        # through its last JSON object, this removes that partial line so a
        # second interruption cannot turn it into an invalid middle line.
        target.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output)
            + ("\n" if output else ""),
            encoding="utf-8",
        )

        if pending and reranker is None:
            dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
            model_kwargs = {"torch_dtype": dtype}
            try:
                reranker = CrossEncoder(
                    args.model,
                    device=args.device,
                    max_length=args.max_length,
                    trust_remote_code=True,
                    model_kwargs=model_kwargs,
                )
            except TypeError:
                # Compatibility with older sentence-transformers releases.
                reranker = CrossEncoder(
                    args.model,
                    device=args.device,
                    max_length=args.max_length,
                    trust_remote_code=True,
                )

        total_pairs = sum(min(
            args.candidate_k,
            len(row.get("retrieved_image_ids", [])),
        ) for _, row in pending)
        with target.open("a", encoding="utf-8", buffering=1) as handle, tqdm(
            total=total_pairs,
            desc=f"{split} Qwen3-VL visual pairs",
            unit="pair",
        ) as pair_progress:
            for item_index, (claim, row) in enumerate(pending, start=1):
                candidates = [
                    str(image_id)
                    for image_id in row.get("retrieved_image_ids", [])[
                        :args.candidate_k
                    ]
                    if str(image_id) in path_by_id
                ]
                query = (
                    f"Instruction: {args.instruction}\n"
                    f"Fact-checking claim: {claim.get('claim', '').strip()}"
                )
                pairs = [
                    (
                        query,
                        {
                            "image": path_by_id[image_id],
                            "text": descriptor_by_id[image_id],
                        },
                    )
                    for image_id in candidates
                ]
                scores = score_in_chunks(
                    reranker, pairs, args.batch_size,
                    args.score_chunk_size, pair_progress,
                ) if pairs else np.empty(0, dtype=np.float32)
                ids, values = stable_rerank(candidates, scores, args.output_k)
                gold = aligned_gold_images(claim, corpus_names)
                rank = first_relevant_rank(ids, gold)
                result_row = {
                    "id": claim["id"],
                    "claim_id": claim["claim_id"],
                    "label": int(claim["label"]),
                    "retrieved_image_ids": ids,
                    "retrieved_image_scores": values,
                    "gold_image_ids": sorted(gold),
                    "first_gold_image_rank": rank,
                    "input_candidate_count": len(candidates),
                    "visual_reranker_model": args.model,
                    "visual_reranker_signature": signature,
                }
                output.append(result_row)
                handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                if item_index % args.flush_every == 0:
                    handle.flush()

        order_by_id = {row["id"]: index for index, row in enumerate(claims)}
        output.sort(key=lambda row: order_by_id[row["id"]])
        target.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
            encoding="utf-8",
        )
        ranks = [row.get("first_gold_image_rank") for row in output]
        annotated = [bool(row.get("gold_image_ids")) for row in output]
        cutoffs = [
            cutoff for cutoff in (1, 5, 10, 50, 100)
            if cutoff <= args.output_k
        ]
        result = retrieval_summary(ranks, annotated, cutoffs)
        result.update({
            "processed_claims": len(output),
            "expected_claims": len(claims),
            "complete": len(output) == len(claims),
            "claims_with_gold_images": int(sum(annotated)),
            "model": args.model,
            "candidate_k": args.candidate_k,
            "output_k": args.output_k,
            "descriptor_signature": descriptor_signature,
            "retrieval_sha256": signature_payload["retrieval_sha256"],
            "visual_reranker_signature": signature,
            "gold_used_for_scoring": False,
        })
        summary[split] = result
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({split: result}, indent=2))


if __name__ == "__main__":
    main()
