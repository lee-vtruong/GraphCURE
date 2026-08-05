from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


UNKNOWN, SATISFIED, VIOLATED = -100, 0, 1
CONSTRAINTS = ("semantic", "entity", "temporal", "contextual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create weak constraint targets from NewsCLIPpings generation subsets."
    )
    parser.add_argument("--raw-root", default="data/raw/news_clippings/news_clippings")
    parser.add_argument("--processed-root", default="data/processed/newsclippings_clip")
    parser.add_argument("--subset", default="merged_balanced")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def constraint_for(source: str, similarity: str | None) -> int:
    value = f"{source} {similarity or ''}".lower()
    if "person" in value or "sbert" in value:
        return 1
    if "scene" in value or "place" in value or "resnet" in value:
        return 3
    if "semantic" in value or "clip_text" in value:
        return 0
    raise ValueError(f"Cannot map generation source to constraint: {value!r}")


def resolve_source(value: object, names: list[str]) -> str:
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        index = int(value)
        if not 0 <= index < len(names):
            raise IndexError(f"source_dataset index {index} outside source_datasets")
        return names[index]
    return str(value)


def annotate(raw_root: Path, processed_root: Path, subset: str, split: str,
             overwrite: bool) -> None:
    directory = processed_root / split
    output = directory / "constraint_labels.npy"
    report_path = directory / "constraint_labels_meta.json"
    if output.exists() and not overwrite:
        print(f"[{split}] {output} exists; use --overwrite to replace")
        return
    payload = json.loads(
        (raw_root / "data" / subset / f"{split}.json").read_text(encoding="utf-8")
    )
    names = payload.get("source_datasets", [])
    records = [json.loads(line) for line in
               (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()
               if line]
    targets = np.full((len(records), 4), UNKNOWN, dtype=np.int64)
    counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        source = resolve_source(record.get("source_dataset", ""), names)
        constraint = constraint_for(source, record.get("similarity_score"))
        state = VIOLATED if record["falsified"] else SATISFIED
        targets[index, constraint] = state
        state_name = "violated" if state == VIOLATED else "satisfied"
        counts[f"{CONSTRAINTS[constraint]}:{state_name}"] += 1
    np.save(output, targets)
    report = {
        "supervision": "weak labels derived from generation subset; not human ground truth",
        "state_mapping": {"satisfied": SATISFIED, "violated": VIOLATED,
                          "unknown": UNKNOWN},
        "constraint_order": CONSTRAINTS,
        "source_datasets": names,
        "counts": dict(sorted(counts.items())),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{split}] wrote {output}: {dict(sorted(counts.items()))}")


def main() -> None:
    args = parse_args()
    for split in args.splits:
        annotate(Path(args.raw_root), Path(args.processed_root), args.subset,
                 split, args.overwrite)


if __name__ == "__main__":
    main()
