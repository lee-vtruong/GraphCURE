"""Check claim/text/evidence leakage across MOCHEG manifest splits."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def values(rows: list[dict[str, Any]], key: str) -> set[str]:
    return {str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/mocheg_manifest"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_audit/mocheg_split_leakage.json"))
    args = parser.parse_args()
    data = {split: load(args.root / f"{split}.jsonl") for split in ("train", "val", "test")}
    report: dict[str, Any] = {"splits": {k: len(v) for k, v in data.items()}, "pairwise": {}}
    failures = []
    fields = ("claim_id", "claim", "snopes_url")
    for left, right in combinations(data, 2):
        pair = {}
        for field in fields:
            overlap = values(data[left], field) & values(data[right], field)
            pair[field] = {"count": len(overlap), "examples": sorted(overlap)[:10]}
            if field in {"claim_id", "claim"} and overlap:
                failures.append(f"{left}-{right}:{field}:{len(overlap)}")
        for field in ("text_evidence_ids", "text_sentence_ids", "image_evidence_ids"):
            a = {x for row in data[left] for x in row.get(field, [])}
            b = {x for row in data[right] for x in row.get(field, [])}
            overlap = a & b
            pair[field] = {"count": len(overlap), "examples": sorted(overlap)[:10]}
        report["pairwise"][f"{left}__{right}"] = pair
    report["status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
