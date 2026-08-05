from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graphcure.data import newsclippings_constraint, resolve_newsclippings_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Index official pristine/falsified NewsCLIPpings pairs")
    parser.add_argument("--raw-root", default="data/raw/news_clippings/news_clippings")
    parser.add_argument("--processed-root", default="data/processed/newsclippings_clip")
    parser.add_argument("--subset", default="merged_balanced")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for split in args.splits:
        directory = Path(args.processed_root) / split
        index_path = directory / "counterfactual_indices.npy"
        mask_path = directory / "changed_masks.npy"
        if index_path.exists() and not args.overwrite:
            print(f"[{split}] already indexed; use --overwrite to replace")
            continue
        records = [json.loads(x) for x in
                   (directory / "records.jsonl").read_text(encoding="utf-8").splitlines() if x]
        payload = json.loads((Path(args.raw_root) / "data" / args.subset /
                              f"{split}.json").read_text(encoding="utf-8"))
        sources = payload.get("source_datasets", [])
        indices = np.empty(len(records), dtype=np.int64)
        masks = np.zeros((len(records), 4), dtype=np.bool_)
        if len(records) % 2:
            raise ValueError(f"{split} has odd sample count")
        for i in range(0, len(records), 2):
            a, b = records[i], records[i + 1]
            if a["falsified"] or not b["falsified"] or a["caption_id"] != b["caption_id"]:
                raise ValueError(f"Invalid official pair at {split}:{i}")
            source = resolve_newsclippings_source(a["source_dataset"], sources)
            changed = newsclippings_constraint(source, a.get("similarity_score"))
            indices[i], indices[i + 1] = i + 1, i
            masks[i, changed] = masks[i + 1, changed] = True
        np.save(index_path, indices)
        np.save(mask_path, masks)
        meta = {"pairs": len(records) // 2, "bidirectional_rows": len(records),
                "changed_counts": masks.sum(0).tolist(),
                "constraint_order": ["semantic", "entity", "temporal", "contextual"]}
        (directory / "counterfactual_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[{split}] {meta}")


if __name__ == "__main__":
    main()
