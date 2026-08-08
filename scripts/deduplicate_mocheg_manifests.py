"""Create a strict no-duplicate-claim-text MOCHEG protocol.

Official manifests remain untouched. Split priority is train > val > test.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("data/processed/mocheg_manifest"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--report", type=Path, default=Path("outputs/data_audit/mocheg_strict_dedup.json"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    report = {"priority": ["train", "val", "test"], "splits": {}, "removed": []}

    for split in ("train", "val", "test"):
        source = args.input_root / f"{split}.jsonl"
        kept, removed = [], []
        with source.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = norm(row.get("claim", ""))
                if key and key in seen:
                    removed.append({"id": row["id"], "claim_id": row["claim_id"],
                                    "duplicate_of_split": seen[key], "claim": row.get("claim", "")})
                    continue
                if key:
                    seen[key] = split
                kept.append(row)
        target = args.output_root / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["splits"][split] = {"input": len(kept) + len(removed), "kept": len(kept), "removed": len(removed)}
        report["removed"].extend([{**x, "from_split": split} for x in removed])

    report["status"] = "pass"
    report["removed_by_split"] = dict(Counter(x["from_split"] for x in report["removed"]))
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "removed"}, indent=2))


if __name__ == "__main__":
    main()
