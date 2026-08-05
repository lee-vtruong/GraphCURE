from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack official NewsCLIPpings CLIP embeddings into mmap arrays."
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw/news_clippings/news_clippings",
        help="Directory containing data/ and embeddings/ from the official release.",
    )
    parser.add_argument("--output", default="data/processed/newsclippings_clip")
    parser.add_argument("--subset", default="merged_balanced")
    parser.add_argument(
        "--splits", nargs="+", default=("train", "val", "test")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    # Only load pickle files obtained from the official NewsCLIPpings release.
    with path.open("rb") as handle:
        return pickle.load(handle)


def embedding_path(root: Path, kind: str, split: str) -> Path:
    return root / "embeddings" / kind / f"{kind}_{split}.pkl"


def prepare_split(
    raw_root: Path, output_root: Path, subset: str, split: str, overwrite: bool
) -> None:
    destination = output_root / split
    expected = (
        destination / "text_embeddings.npy",
        destination / "image_embeddings.npy",
        destination / "labels.npy",
        destination / "records.jsonl",
        destination / "meta.json",
    )
    if all(path.exists() for path in expected) and not overwrite:
        print(f"[{split}] already prepared; use --overwrite to replace")
        return
    destination.mkdir(parents=True, exist_ok=True)

    annotation_path = raw_root / "data" / subset / f"{split}.json"
    text_path = embedding_path(raw_root, "clip_text_embeddings", split)
    image_path = embedding_path(raw_root, "clip_image_embeddings", split)
    for path in (annotation_path, text_path, image_path):
        if not path.exists():
            raise FileNotFoundError(path)

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations = payload["annotations"]
    print(f"[{split}] loading official embedding dictionaries")
    text_store = load_pickle(text_path)
    image_store = load_pickle(image_path)

    valid = [
        ann
        for ann in annotations
        if ann["id"] in text_store and ann["image_id"] in image_store
    ]
    missing = len(annotations) - len(valid)
    if not valid:
        raise RuntimeError(f"No valid samples found for {split}")

    text_dim = int(np.asarray(text_store[valid[0]["id"]]).size)
    image_dim = int(np.asarray(image_store[valid[0]["image_id"]]).size)
    text_array = np.lib.format.open_memmap(
        destination / "text_embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(valid), text_dim),
    )
    image_array = np.lib.format.open_memmap(
        destination / "image_embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(valid), image_dim),
    )
    labels = np.lib.format.open_memmap(
        destination / "labels.npy", mode="w+", dtype=np.int64, shape=(len(valid),)
    )

    with (destination / "records.jsonl").open("w", encoding="utf-8") as records:
        for index, ann in enumerate(tqdm(valid, desc=f"packing {split}")):
            text_array[index] = np.asarray(text_store[ann["id"]], dtype=np.float32).reshape(-1)
            image_array[index] = np.asarray(
                image_store[ann["image_id"]], dtype=np.float32
            ).reshape(-1)
            labels[index] = int(bool(ann["falsified"]))
            records.write(
                json.dumps(
                    {
                        "sample_id": f"{split}-{index}",
                        "caption_id": ann["id"],
                        "image_id": ann["image_id"],
                        "falsified": bool(ann["falsified"]),
                        "similarity_score": ann.get("similarity_score"),
                        "source_dataset": ann.get("source_dataset"),
                    }
                )
                + "\n"
            )
    text_array.flush()
    image_array.flush()
    labels.flush()
    meta = {
        "source": "NewsCLIPpings official release",
        "subset": subset,
        "split": split,
        "samples": len(valid),
        "missing_embeddings": missing,
        "text_dim": text_dim,
        "image_dim": image_dim,
        "label_mapping": {"pristine": 0, "falsified": 1},
    }
    (destination / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[{split}] wrote {len(valid)} samples to {destination}; missing={missing}")


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output)
    for split in args.splits:
        prepare_split(raw_root, output_root, args.subset, split, args.overwrite)


if __name__ == "__main__":
    main()

