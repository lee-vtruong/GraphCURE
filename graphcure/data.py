from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
import numpy as np


def resolve_newsclippings_source(
    value: object, names: list[str] | dict[str, str]
) -> str:
    """Resolve list- or JSON-dict source mappings from official releases."""
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        index = int(value)
        if isinstance(names, dict):
            if str(index) in names:
                return names[str(index)]
            if index in names:
                return names[index]  # type: ignore[index]
            raise KeyError(f"source_dataset key {index} absent from {sorted(names)}")
        if 0 <= index < len(names):
            return names[index]
        raise IndexError(f"source_dataset index {index} outside source_datasets")
    return str(value)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class EmbeddingManifestDataset(Dataset):
    """Manifest dataset for cached-backbone experiments.

    Required fields: id, label, text_embedding, image_embedding.
    Optional: metadata, constraint_labels, conflict_labels, counterfactual.
    Embeddings may be inline arrays or paths to torch tensors.
    """

    def __init__(self, manifest: str | Path) -> None:
        self.rows = read_jsonl(manifest)
        self.base = Path(manifest).parent

    def __len__(self) -> int:
        return len(self.rows)

    def _tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                path = self.base / path
            return torch.load(path, map_location="cpu", weights_only=True).float()
        return torch.tensor(value, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        item = {
            "id": row["id"],
            "text_embedding": self._tensor(row["text_embedding"]),
            "image_embedding": self._tensor(row["image_embedding"]),
            "metadata": self._tensor(row.get("metadata", [0.0] * 16)),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "constraint_labels": torch.tensor(
                row.get("constraint_labels", [-100] * 4), dtype=torch.long
            ),
            "conflict_labels": torch.tensor(
                row.get("conflict_labels", [-1.0] * 5), dtype=torch.float32
            ),
        }
        cf = row.get("counterfactual")
        if cf:
            item["cf_text_embedding"] = self._tensor(
                cf.get("text_embedding", row["text_embedding"])
            )
            item["cf_image_embedding"] = self._tensor(
                cf.get("image_embedding", row["image_embedding"])
            )
            item["cf_metadata"] = self._tensor(cf.get("metadata", row.get("metadata", [0.0] * 16)))
            item["changed_mask"] = torch.tensor(cf["changed_mask"], dtype=torch.bool)
        return item


class PackedEmbeddingDataset(Dataset):
    """Memory-mapped embedding dataset produced by prepare_newsclippings.py."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.text = np.load(self.directory / "text_embeddings.npy", mmap_mode="r")
        self.image = np.load(self.directory / "image_embeddings.npy", mmap_mode="r")
        self.labels = np.load(self.directory / "labels.npy", mmap_mode="r")
        constraint_path = self.directory / "constraint_labels.npy"
        self.constraint_labels = (
            np.load(constraint_path, mmap_mode="r") if constraint_path.exists() else None
        )
        self.extra = {}
        for name in ("sbert_embeddings", "facenet_embeddings", "places_embeddings",
                     "view_mask"):
            path = self.directory / f"{name}.npy"
            if path.exists():
                self.extra[name] = np.load(path, mmap_mode="r")
        self.ids = read_jsonl(self.directory / "records.jsonl")
        if not (len(self.text) == len(self.image) == len(self.labels) == len(self.ids)):
            raise ValueError(f"Inconsistent packed dataset lengths in {self.directory}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # np.array(copy=True) avoids returning non-writable mmap views to torch.
        item = {
            "id": self.ids[index]["sample_id"],
            "text_embedding": torch.from_numpy(np.array(self.text[index], copy=True)),
            "image_embedding": torch.from_numpy(np.array(self.image[index], copy=True)),
            "metadata": torch.zeros(16, dtype=torch.float32),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "constraint_labels": (
                torch.from_numpy(np.array(self.constraint_labels[index], copy=True)).long()
                if self.constraint_labels is not None
                else torch.full((4,), -100, dtype=torch.long)
            ),
            "conflict_labels": torch.full((5,), -1.0, dtype=torch.float32),
        }
        for name, array in self.extra.items():
            item[name] = torch.from_numpy(np.array(array[index], copy=True)).float()
        return item


def build_dataset(config: dict[str, Any], split: str) -> Dataset:
    data_format = config.get("format", "manifest")
    if data_format == "manifest":
        return EmbeddingManifestDataset(config[f"{split}_manifest"])
    if data_format == "packed_npy":
        return PackedEmbeddingDataset(config[f"{split}_dir"])
    raise ValueError(f"Unsupported data format: {data_format}")


def collate_manifest(items: list[dict[str, Any]]) -> dict[str, Any]:
    # Counterfactual batches require all rows to carry a pair. Generate manifests
    # accordingly to keep the loss semantics explicit.
    keys = set.intersection(*(set(item) for item in items))
    batch: dict[str, Any] = {"id": [item["id"] for item in items]}
    for key in keys - {"id"}:
        batch[key] = torch.stack([item[key] for item in items])
    return batch
