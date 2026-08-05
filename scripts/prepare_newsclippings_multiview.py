from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


KINDS = {
    "sbert_embeddings": ("sbert_embeddings", "caption_id"),
    "facenet_embeddings": ("facenet_embeddings", "image_id"),
    "places_embeddings": ("places_resnet50", "image_id"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add SBERT, FaceNet and Places views to packed NewsCLIPpings")
    parser.add_argument("--raw-root", default="data/raw/news_clippings/news_clippings")
    parser.add_argument("--processed-root", default="data/processed/newsclippings_clip")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    # Pickle is unsafe; only use files from the official dataset release.
    with path.open("rb") as handle:
        return pickle.load(handle)


def lookup(store: dict, key: object) -> object | None:
    for candidate in (key, str(key)):
        if candidate in store:
            return store[candidate]
    try:
        numeric = int(key)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return store.get(numeric)


def vector(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return None
    if array.ndim == 1:
        return array
    # FaceNet may return one vector per detected face. Mean pooling is fixed,
    # permutation invariant, and avoids leaking the number of detections.
    return array.reshape(-1, array.shape[-1]).mean(axis=0)


def infer_dim(store: dict) -> int:
    for value in store.values():
        item = vector(value)
        if item is not None:
            return int(item.size)
    raise RuntimeError("Embedding dictionary contains no usable vectors")


def prepare_split(raw: Path, processed: Path, split: str, overwrite: bool) -> None:
    directory = processed / split
    records = [json.loads(line) for line in
               (directory / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    stores, dims = {}, {}
    for output_name, (official_name, _) in KINDS.items():
        path = raw / "embeddings" / official_name / f"{official_name}_{split}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        stores[output_name] = load_pickle(path)
        dims[output_name] = infer_dim(stores[output_name])

    arrays = {}
    for name, dim in dims.items():
        path = directory / f"{name}.npy"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite to regenerate all views")
        arrays[name] = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(records), dim)
        )
        arrays[name][:] = 0
    mask = np.lib.format.open_memmap(
        directory / "view_mask.npy", mode="w+", dtype=np.float32,
        shape=(len(records), 5)
    )
    mask[:] = 0
    mask[:, 0] = 1  # packed CLIP text/image are always present
    availability = {name: 0 for name in KINDS}
    mask_columns = {"sbert_embeddings": 1, "facenet_embeddings": 2,
                    "places_embeddings": 3}
    for row_index, record in enumerate(tqdm(records, desc=f"multi-view {split}")):
        for name, (_, id_field) in KINDS.items():
            item = vector(lookup(stores[name], record[id_field]))
            if item is None:
                continue
            if item.size != dims[name]:
                raise ValueError(f"Inconsistent {name} dimension at row {row_index}")
            arrays[name][row_index] = item
            mask[row_index, mask_columns[name]] = 1
            availability[name] += 1
    for array in arrays.values():
        array.flush()
    mask.flush()
    meta = {
        "samples": len(records), "dimensions": dims, "available": availability,
        "view_mask": ["clip", "sbert", "facenet", "places", "temporal"],
        "facenet_pooling": "mean over detected faces",
        "temporal": "unavailable in this adapter",
    }
    (directory / "multiview_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{split}] {json.dumps(meta)}")


def main() -> None:
    args = parse_args()
    for split in args.splits:
        prepare_split(Path(args.raw_root), Path(args.processed_root), split, args.overwrite)


if __name__ == "__main__":
    main()
