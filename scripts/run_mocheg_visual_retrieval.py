"""Validation-first Qwen3-VL image retrieval for MOCHEG.

The default split is validation. Test retrieval must only be requested after a
multimodal verifier configuration is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from graphcure.retrieval import first_relevant_rank, top_indices


VISUAL_RETRIEVAL_INSTRUCTION = (
    "Retrieve visual evidence useful for verifying the fact-checking claim. "
    "Match people, objects, events, locations, time cues, quantities, and "
    "possible image reuse; visual similarity alone is insufficient."
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def image_corpus(directory: Path) -> tuple[list[str], list[str]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing image corpus: {directory}")
    paths = sorted((path for path in directory.iterdir() if path.is_file()),
                   key=lambda path: path.name)
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate image basenames in {directory}")
    return names, [str(path.resolve()) for path in paths]


def corpus_fingerprint(names: list[str], paths: list[str]) -> str:
    digest = hashlib.sha256()
    for name, raw_path in zip(names, paths, strict=True):
        path = Path(raw_path)
        digest.update(name.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def is_torchvision_supported_image(path: Path) -> bool:
    """Return whether torchvision's generic decoder accepts the file magic."""
    with path.open("rb") as handle:
        header = handle.read(12)
    return (
        header.startswith(b"\xff\xd8\xff")
        or header.startswith(b"\x89PNG\r\n\x1a\n")
        or header.startswith((b"GIF87a", b"GIF89a"))
        or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
    )


def normalize_image_paths(paths: list[str], cache_root: Path) -> tuple[list[str], int]:
    """Convert decoder-incompatible images to cached RGB PNG files.

    MOCHEG contains a small number of BMP, TIFF, and PSD payloads, including
    files with misleading extensions. Qwen's current torchvision-backed image
    loader rejects these formats, so format detection must use file magic.
    """
    from PIL import Image, ImageOps

    normalized: list[str] = []
    converted = 0
    target_root = cache_root / "normalized_images"
    for raw_path in paths:
        source = Path(raw_path)
        if is_torchvision_supported_image(source):
            normalized.append(str(source))
            continue

        stat = source.stat()
        identity = hashlib.sha256(
            f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        ).hexdigest()[:24]
        target = target_root / f"{identity}.png"
        if not target.exists():
            target_root.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(source) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    temporary = target.with_suffix(".tmp")
                    image.save(temporary, format="PNG")
                    temporary.replace(target)
            except Exception as error:
                raise RuntimeError(
                    f"cannot normalize unsupported image {source}: {error}"
                ) from error
        normalized.append(str(target.resolve()))
        converted += 1
    return normalized, converted


def encode_images(model: Any, names: list[str], paths: list[str],
                  cache_root: Path, split: str, model_name: str,
                  batch_size: int, shard_size: int = 256) -> np.ndarray:
    fingerprint = corpus_fingerprint(names, paths)
    key = hashlib.sha256(f"{model_name}:{fingerprint}".encode()).hexdigest()[:16]
    target = cache_root / f"mocheg-{split}-images-{key}.npy"
    metadata_path = target.with_suffix(".json")
    if target.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("names") == names and metadata.get("fingerprint") == fingerprint:
            print(f"loading cached image embeddings: {target}")
            return np.load(target, mmap_mode="r")

    cache_root.mkdir(parents=True, exist_ok=True)
    normalized_paths, converted = normalize_image_paths(paths, cache_root)
    if converted:
        print(
            f"{split}: normalized {converted} decoder-incompatible image(s) "
            f"to {cache_root / 'normalized_images'}"
        )

    # Persist each shard immediately. A corrupt late image, interruption, or
    # transient CUDA failure can then resume without re-encoding prior shards.
    part_root = cache_root / "image_embedding_parts" / target.stem
    part_root.mkdir(parents=True, exist_ok=True)
    parts: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(normalized_paths), shard_size),
        desc=f"{split} image embedding shards",
    ):
        end = min(start + shard_size, len(normalized_paths))
        part_path = part_root / f"{start:08d}-{end:08d}.npy"
        if part_path.exists():
            part = np.load(part_path, mmap_mode="r")
            if part.ndim != 2 or part.shape[0] != end - start:
                raise ValueError(
                    f"invalid cached image shard {part_path}; remove only this "
                    "file and rerun"
                )
        else:
            inputs = [
                {"image": path} for path in normalized_paths[start:end]
            ]
            part = model.encode(
                inputs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            temporary = part_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, part)
            temporary.replace(part_path)
        parts.append(part)

    embeddings = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, embeddings)
    temporary.replace(target)
    metadata_path.write_text(json.dumps({
        "model": model_name,
        "split": split,
        "fingerprint": fingerprint,
        "images": len(names),
        "names": names,
        "shard_size": shard_size,
        "normalized_images": converted,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return embeddings


def encode_queries(model: Any, claims: list[dict], cache_root: Path,
                   split: str, model_name: str, instruction: str,
                   batch_size: int) -> np.ndarray:
    """Encode text queries with the SentenceTransformers prompt API."""
    texts = [str(row.get("claim", "")) for row in claims]
    digest = hashlib.sha256()
    digest.update(model_name.encode())
    digest.update(b"\0")
    digest.update(instruction.encode())
    digest.update(b"\0")
    for row, claim in zip(claims, texts, strict=True):
        digest.update(str(row.get("id", "")).encode())
        digest.update(b"\0")
        digest.update(claim.encode())
        digest.update(b"\n")
    fingerprint = digest.hexdigest()
    target = cache_root / f"mocheg-{split}-queries-{fingerprint[:16]}.npy"
    metadata_path = target.with_suffix(".json")
    if target.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("fingerprint") == fingerprint
            and metadata.get("claims") == len(claims)
        ):
            print(f"loading cached query embeddings: {target}")
            return np.load(target, mmap_mode="r")

    # The SentenceTransformers integration accepts modality dictionaries with
    # text/image/video keys only. Its documented instruction interface is the
    # prompt argument, not an `instruction` key inside each input dictionary.
    embeddings = model.encode(
        texts,
        prompt=instruction,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, embeddings)
    temporary.replace(target)
    metadata_path.write_text(json.dumps({
        "model": model_name,
        "split": split,
        "instruction": instruction,
        "fingerprint": fingerprint,
        "claims": len(claims),
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return embeddings


def retrieval_summary(ranks: list[int | None], annotated: list[bool],
                      cutoffs: list[int]) -> dict:
    result = {
        "claims": len(ranks),
        "claims_with_gold_images": int(sum(annotated)),
    }
    for cutoff in cutoffs:
        raw = [rank is not None and rank <= cutoff for rank in ranks]
        conditional = [
            rank is not None and rank <= cutoff
            for rank, has_gold in zip(ranks, annotated, strict=True) if has_gold
        ]
        result[f"recall@{cutoff}"] = float(np.mean(raw))
        result[f"conditional_recall@{cutoff}"] = (
            float(np.mean(conditional)) if conditional else 0.0
        )
    reciprocal = [1.0 / rank if rank is not None else 0.0 for rank in ranks]
    conditional_reciprocal = [
        1.0 / rank if rank is not None else 0.0
        for rank, has_gold in zip(ranks, annotated, strict=True) if has_gold
    ]
    result["mrr"] = float(np.mean(reciprocal))
    result["conditional_mrr"] = (
        float(np.mean(conditional_reciprocal)) if conditional_reciprocal else 0.0
    )
    return result


def main() -> None:
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/retrieval_mocheg_qwen3vl_images"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/retrieval_cache"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--instruction", default=VISUAL_RETRIEVAL_INSTRUCTION)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-shard-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["val"])
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.image_shard_size <= 0:
        parser.error("--image-shard-size must be positive")
    if "test" in args.splits:
        print("WARNING: test visual retrieval requested; only do this after v2 freeze")

    args.output_root.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model, device=args.device)
    summary_path = args.output_root / "summary.json"
    summary = (json.loads(summary_path.read_text(encoding="utf-8"))
               if summary_path.exists() else {})

    signature = hashlib.sha256(json.dumps({
        "model": args.model,
        "instruction": args.instruction,
        "top_k": args.top_k,
    }, sort_keys=True).encode()).hexdigest()

    for split in args.splits:
        claims = read_jsonl(args.manifest_root / f"{split}.jsonl")
        image_names, image_paths = image_corpus(args.raw_root / split / "images")
        if args.top_k > len(image_names):
            parser.error(f"--top-k exceeds the {split} image corpus size")
        image_embeddings = encode_images(
            model, image_names, image_paths, args.cache_root, split,
            args.model, args.batch_size, args.image_shard_size,
        )
        query_embeddings = encode_queries(
            model, claims, args.cache_root, split, args.model,
            args.instruction, args.query_batch_size,
        )

        output = []
        ranks: list[int | None] = []
        annotated: list[bool] = []
        corpus_names = set(image_names)
        for start in tqdm(
            range(0, len(claims), args.score_batch_size),
            desc=f"{split} visual retrieval",
        ):
            end = min(start + args.score_batch_size, len(claims))
            scores = query_embeddings[start:end] @ image_embeddings.T
            for offset, claim in enumerate(claims[start:end]):
                indices = top_indices(scores[offset], args.top_k)
                ids = [image_names[index] for index in indices]
                values = [float(scores[offset, index]) for index in indices]
                # MOCHEG image qrels expose both DOCUMENT# and evidence_id.
                gold = {
                    str(value) for value in (
                        claim.get("image_candidate_names", [])
                        + claim.get("image_evidence_ids", [])
                    ) if str(value) in corpus_names
                }
                rank = first_relevant_rank(ids, gold)
                ranks.append(rank)
                annotated.append(bool(gold))
                output.append({
                    "id": claim["id"],
                    "claim_id": claim["claim_id"],
                    "label": int(claim["label"]),
                    "retrieved_image_ids": ids,
                    "retrieved_image_scores": values,
                    "gold_image_ids": sorted(gold),
                    "first_gold_image_rank": rank,
                    "visual_retrieval_model": args.model,
                    "visual_retrieval_signature": signature,
                })

        if not any(annotated):
            raise ValueError(
                f"no {split} gold image IDs align with the split image corpus"
            )

        target = args.output_root / f"{split}.jsonl"
        target.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
            encoding="utf-8",
        )
        split_summary = retrieval_summary(
            ranks, annotated, sorted({1, 5, 10, args.top_k})
        )
        split_summary.update({
            "image_corpus": len(image_names),
            "model": args.model,
            "visual_retrieval_signature": signature,
        })
        summary[split] = split_summary
        print(json.dumps({split: split_summary}, indent=2))

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
