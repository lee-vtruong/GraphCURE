"""Assemble frozen text and visual evidence caches for GraphCURE-R2V-v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scripts.run_mocheg_visual_ensemble import aligned_gold_images
from scripts.run_mocheg_visual_retrieval import read_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def find_image_cache(cache_root: Path, split: str,
                     model_name: str) -> tuple[np.ndarray, list[str], Path]:
    matches = []
    for metadata_path in cache_root.glob(f"mocheg-{split}-images-*.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        array_path = metadata_path.with_suffix(".npy")
        if metadata.get("model") == model_name and array_path.exists():
            matches.append((metadata_path, array_path, metadata))
    if len(matches) != 1:
        raise ValueError(
            f"expected one complete {split} image cache for {model_name}, "
            f"found {len(matches)}"
        )
    metadata_path, array_path, metadata = matches[0]
    names = [str(value) for value in metadata.get("names", [])]
    embeddings = np.load(array_path, mmap_mode="r")
    if embeddings.ndim != 2 or len(embeddings) != len(names):
        raise ValueError(f"unaligned image cache: {array_path}")
    return embeddings, names, metadata_path


def select_visual_candidates(sample_id: str, retrieved_ids: list[str],
                             gold_ids: set[str], top_k: int,
                             inject_gold: bool) -> list[str]:
    retrieved = list(dict.fromkeys(str(value) for value in retrieved_ids))
    if not inject_gold:
        return retrieved[:top_k]
    positives = sorted(gold_ids)[:top_k]
    selected = positives + [
        value for value in retrieved if value not in gold_ids
    ][:max(0, top_k - len(positives))]
    # Avoid a positional shortcut in which injected positives are always the
    # first candidates. The stable hash makes caches exactly reproducible.
    return sorted(
        selected,
        key=lambda value: hashlib.sha256(
            f"{sample_id}\0{value}".encode()
        ).digest(),
    )


def visual_candidate_features(candidate_id: str, retrieved_ids: list[str],
                              retrieved_scores: list[float]) -> list[float]:
    unique_ids = list(dict.fromkeys(str(value) for value in retrieved_ids))
    rank_by_id = {value: index + 1 for index, value in enumerate(unique_ids)}
    score_by_id = {
        str(value): float(retrieved_scores[index])
        for index, value in enumerate(retrieved_ids)
        if index < len(retrieved_scores)
    }
    if candidate_id not in rank_by_id:
        return [0.0, 0.0, 0.0]
    rank = rank_by_id[candidate_id]
    rank_percentile = 1.0 - (rank - 1) / max(1, len(unique_ids))
    values = list(score_by_id.values())
    if values:
        lower, upper = min(values), max(values)
        normalized_score = (
            (score_by_id.get(candidate_id, lower) - lower) / (upper - lower)
            if upper > lower else 1.0
        )
    else:
        normalized_score = 0.0
    return [1.0 / rank, float(normalized_score), float(rank_percentile)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--text-cache-root", type=Path,
                        default=Path("data/processed/mocheg_reasoning_cache"))
    parser.add_argument("--train-visual-retrieval", type=Path)
    parser.add_argument("--val-visual-retrieval", type=Path)
    parser.add_argument("--image-cache-root", type=Path,
                        default=Path("data/processed/retrieval_cache"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/processed/mocheg_multimodal_cache"))
    parser.add_argument("--visual-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--visual-top-k", type=int, default=32)
    parser.add_argument(
        "--train-candidate-policy",
        choices=("inject", "retrieved"),
        default="inject",
        help="inject train qrel positives or preserve natural retrieval",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val"),
        default=("train", "val"),
    )
    args = parser.parse_args()
    if args.visual_top_k <= 0:
        parser.error("--visual-top-k must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    retrieval_paths = {
        "train": args.train_visual_retrieval,
        "val": args.val_visual_retrieval,
    }
    for split in args.splits:
        if retrieval_paths[split] is None:
            parser.error(f"--{split}-visual-retrieval is required for {split}")

    for split in args.splits:
        manifest_path = args.manifest_root / f"{split}.jsonl"
        text_cache_path = args.text_cache_root / f"{split}.pt"
        retrieval_path = retrieval_paths[split]
        claims = read_jsonl(manifest_path)
        text = torch.load(text_cache_path, map_location="cpu", weights_only=False)
        if text["ids"] != [row["id"] for row in claims]:
            raise ValueError(f"{split}: text cache and manifest IDs do not align")
        retrieval_rows = read_jsonl(retrieval_path)
        if [row["id"] for row in retrieval_rows] != text["ids"]:
            raise ValueError(f"{split}: visual retrieval IDs do not align")
        image_embeddings, image_names, image_metadata_path = find_image_cache(
            args.image_cache_root, split, args.visual_model
        )
        image_index = {value: index for index, value in enumerate(image_names)}
        corpus_names = set(image_names)
        samples = len(claims)
        visual_dim = int(image_embeddings.shape[1])
        visual_evidence = torch.zeros(
            samples, args.visual_top_k, visual_dim, dtype=torch.float16
        )
        visual_mask = torch.zeros(
            samples, args.visual_top_k, dtype=torch.bool
        )
        visual_features = torch.zeros(
            samples, args.visual_top_k, 3, dtype=torch.float32
        )
        visual_relevance = torch.zeros(
            samples, args.visual_top_k, dtype=torch.float32
        )
        visual_weights = torch.zeros_like(visual_relevance)
        injected = 0
        claims_with_gold = 0

        for row_index, (claim, retrieval) in enumerate(tqdm(
            zip(claims, retrieval_rows, strict=True),
            total=samples,
            desc=f"{split} assemble multimodal cache",
        )):
            gold = aligned_gold_images(claim, corpus_names)
            claims_with_gold += bool(gold)
            all_ids = [
                str(value)
                for value in retrieval.get("retrieved_image_ids", [])
            ]
            all_scores = [
                float(value)
                for value in retrieval.get("retrieved_image_scores", [])
            ]
            # Filter identifiers and scores together. Filtering only the IDs
            # silently assigns the score of a missing image to the next valid
            # image and corrupts the learned retrieval features.
            valid_pairs = [
                (image_id, all_scores[index])
                for index, image_id in enumerate(all_ids)
                if image_id in image_index and index < len(all_scores)
            ]
            raw_ids = [image_id for image_id, _ in valid_pairs]
            raw_scores = [score for _, score in valid_pairs]
            inject_train_gold = (
                split == "train" and args.train_candidate_policy == "inject"
            )
            selected = select_visual_candidates(
                claim["id"], raw_ids, gold, args.visual_top_k,
                inject_gold=inject_train_gold,
            )
            if inject_train_gold:
                injected += len([value for value in selected if value in gold
                                 and value not in raw_ids[:args.visual_top_k]])
            for candidate_index, image_id in enumerate(selected):
                features = visual_candidate_features(
                    image_id, raw_ids, raw_scores
                )
                visual_evidence[row_index, candidate_index] = torch.from_numpy(
                    np.asarray(image_embeddings[image_index[image_id]])
                ).half()
                visual_mask[row_index, candidate_index] = True
                visual_features[row_index, candidate_index] = torch.tensor(features)
                is_gold = image_id in gold
                visual_relevance[row_index, candidate_index] = float(is_gold)
                visual_weights[row_index, candidate_index] = (
                    1.0 if is_gold else 1.0 + 0.25 * features[2]
                )

        payload = {
            "version": 1,
            "ids": text["ids"],
            "labels": text["labels"],
            "claim_embeddings": text["claim_embeddings"],
            "text_evidence_embeddings": text["evidence_embeddings"],
            "text_mask": text["evidence_mask"],
            "text_retrieval_features": text["retrieval_features"],
            "text_relevance": text["relevance"],
            "text_relevance_weights": text["relevance_weights"],
            "visual_evidence_embeddings": visual_evidence,
            "visual_mask": visual_mask,
            "visual_retrieval_features": visual_features,
            "visual_relevance": visual_relevance,
            "visual_relevance_weights": visual_weights,
            "metadata": {
                "split": split,
                "samples": samples,
                "claim_dim": int(text["claim_embeddings"].shape[-1]),
                "text_dim": int(text["evidence_embeddings"].shape[-1]),
                "visual_dim": visual_dim,
                "text_top_k": int(text["evidence_embeddings"].shape[1]),
                "visual_top_k": args.visual_top_k,
                "visual_model": args.visual_model,
                "train_gold_injection": (
                    split == "train"
                    and args.train_candidate_policy == "inject"
                ),
                "train_candidate_policy": args.train_candidate_policy
                if split == "train" else "not_applicable",
                "validation_gold_injection": False,
                "claims_with_gold_images": claims_with_gold,
                "injected_gold_candidates": injected,
                "manifest_sha256": sha256(manifest_path),
                "text_cache_sha256": sha256(text_cache_path),
                "visual_retrieval_sha256": sha256(retrieval_path),
                "image_cache_metadata_sha256": sha256(image_metadata_path),
                "git_commit": git_commit(),
            },
        }
        target = args.output_root / f"{split}.pt"
        coverage = float(
            visual_relevance.bool().any(1).float().mean()
        )
        result = {
            "saved": str(target),
            **payload["metadata"],
            "visual_gold_coverage": coverage,
        }
        # A killed multi-gigabyte save must not leave a path that looks like a
        # valid cache. Replace the target only after serialization completes.
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        target.with_suffix(".metadata.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
