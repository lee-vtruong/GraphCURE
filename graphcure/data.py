from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


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


def collate_manifest(items: list[dict[str, Any]]) -> dict[str, Any]:
    # Counterfactual batches require all rows to carry a pair. Generate manifests
    # accordingly to keep the loss semantics explicit.
    keys = set.intersection(*(set(item) for item in items))
    batch: dict[str, Any] = {"id": [item["id"] for item in items]}
    for key in keys - {"id"}:
        batch[key] = torch.stack([item[key] for item in items])
    return batch

