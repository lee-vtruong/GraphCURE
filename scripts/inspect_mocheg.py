"""Audit an extracted MOCHEG release before building training manifests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")
CSV_NAMES = ("Corpus2.csv", "img_evidence_qrels.csv",
             "text_evidence_qrels_article_level.csv",
             "text_evidence_qrels_sentence_level.csv")
MAGIC = {b"\xff\xd8\xff": "jpeg", b"\x89PNG\r\n\x1a\n": "png", b"BM": "bmp",
         b"GIF87a": "gif", b"GIF89a": "gif", b"II*\x00": "tiff", b"MM\x00*": "tiff"}
MAGIC[b"8BPS"] = "psd"


def image_kind(path: Path) -> str | None:
    with path.open("rb") as handle:
        head = handle.read(16)
    for signature, kind in MAGIC.items():
        if head.startswith(signature):
            return kind
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    return None


def audit_csv(path: Path, sample_rows: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        samples, rows, malformed = [], 0, 0
        for row in reader:
            rows += 1
            malformed += int(None in row)
            if len(samples) < sample_rows:
                samples.append({str(k): str(v)[:300] for k, v in row.items() if k is not None})
    return {"path": str(path), "columns": columns, "rows": rows,
            "malformed_rows": malformed, "samples": samples}


def audit_images(directory: Path, verify_all: bool) -> dict[str, Any]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    checked = files if verify_all else files[:1000]
    kinds, invalid = Counter(), []
    for path in checked:
        kind = image_kind(path)
        kinds[kind] += 1
        if kind is None:
            invalid.append(str(path))
    return {
        "directory": str(directory), "files": len(files),
        "suffixes": dict(sorted(Counter(p.suffix.lower() or "<none>" for p in files).items())),
        "zero_byte_files": [str(p) for p in files if p.stat().st_size == 0],
        "magic_checked": len(checked),
        "magic_types": {str(k): v for k, v in sorted(kinds.items(), key=lambda x: str(x[0]))},
        "invalid_magic": invalid,
        "verification": "all" if verify_all else "first_1000_sorted",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/data_audit/mocheg_schema.json"))
    parser.add_argument("--sample-rows", type=int, default=2)
    parser.add_argument("--verify-all-images", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"MOCHEG root does not exist: {root}")

    report: dict[str, Any] = {"root": str(root), "splits": {}}
    failures: list[str] = []
    for split in SPLITS:
        split_dir, csv_report = root / split, {}
        for name in CSV_NAMES:
            path = split_dir / name
            if path.is_file():
                csv_report[name] = audit_csv(path, args.sample_rows)
            else:
                failures.append(f"missing: {path}")
        image_dir = split_dir / "images"
        images = audit_images(image_dir, args.verify_all_images) if image_dir.is_dir() else None
        if images is None:
            failures.append(f"missing: {image_dir}")
        elif images["zero_byte_files"] or images["invalid_magic"]:
            failures.append(f"invalid images in {image_dir}")
        report["splits"][split] = {"csv": csv_report, "images": images}

    report.update(status="pass" if not failures else "fail", failures=failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nAudit report: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
