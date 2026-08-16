"""Materialize leakage-safe MOCHEG close/open/gold protocols.

The strict claim manifest contains evidence-derived fields for auditing.  Those
fields must not silently enter a closed-book experiment.  This script creates
three explicit, ID-aligned manifests and fails if labels or retrieval rows do
not match the strict split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def gold_text_qrels(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for row in read_csv(path):
        if row.get("RELEVANCY", "0").strip() != "1":
            continue
        claim_id = row.get("TOPIC", "").strip()
        evidence_id = row.get("evidence_id", "").strip()
        if claim_id and evidence_id and evidence_id not in result[claim_id]:
            result[claim_id].append(evidence_id)
    return result


def close_row(source: dict[str, Any]) -> dict[str, Any]:
    """Retain claim content only; all qrel/fact-check article fields are hidden."""
    return {
        "id": source["id"],
        "claim_id": source["claim_id"],
        "label": source["label"],
        "label_name": source.get("label_name", ""),
        "claim": source.get("claim", ""),
        "claim_image_paths": source.get("claim_image_paths", []),
        "evidence_texts": [],
        "text_evidence_ids": [],
        "image_paths": [],
        "image_evidence_ids": [],
        "protocol": "close",
        "model_inputs": ["claim", "claim_image_paths"],
        "evidence_provenance": "none",
    }


def build_split(
    split: str,
    manifest_root: Path,
    retrieval_root: Path,
    raw_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    source_rows = read_jsonl(manifest_root / f"{split}.jsonl")
    retrieved_rows = read_jsonl(retrieval_root / f"{split}.jsonl")
    source = {row["id"]: row for row in source_rows}
    retrieved = {row["id"]: row for row in retrieved_rows}
    failures: list[str] = []

    if len(source) != len(source_rows):
        failures.append(f"{split}: duplicate IDs in strict manifest")
    if len(retrieved) != len(retrieved_rows):
        failures.append(f"{split}: duplicate IDs in retrieval output")
    missing = sorted(set(source) - set(retrieved))
    extra = sorted(set(retrieved) - set(source))
    if missing:
        failures.append(f"{split}: retrieval missing {len(missing)} claim IDs")
    if extra:
        failures.append(f"{split}: retrieval has {len(extra)} unexpected claim IDs")

    corpus = read_csv(raw_root / split / "Corpus2.csv")
    documents = {
        row.get("evidence_id", "").strip(): row.get("Evidence", "")
        for row in corpus
        if row.get("evidence_id", "").strip()
    }
    gold_ids = gold_text_qrels(
        raw_root / split / "text_evidence_qrels_article_level.csv"
    )

    close_rows: list[dict[str, Any]] = []
    open_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for row in source_rows:
        sample_id = row["id"]
        close = close_row(row)
        close_rows.append(close)

        retrieved_row = retrieved.get(sample_id, {})
        if retrieved_row and int(retrieved_row.get("label", row["label"])) != int(row["label"]):
            failures.append(f"{split}:{sample_id}: retrieval label mismatch")
        retrieved_ids = [
            evidence_id
            for evidence_id in retrieved_row.get("retrieved_evidence_ids", [])
            if evidence_id in documents
        ]
        opened = dict(close)
        opened.update(
            {
                "evidence_texts": [documents[evidence_id] for evidence_id in retrieved_ids],
                "text_evidence_ids": retrieved_ids,
                "retrieved_scores": retrieved_row.get("retrieved_scores", [])[: len(retrieved_ids)],
                "retrieval_confidence": retrieved_row.get("retrieval_confidence", 0.0),
                "protocol": "open_retrieved",
                "model_inputs": ["claim", "claim_image_paths", "evidence_texts"],
                "evidence_provenance": "system_retrieved",
            }
        )
        open_rows.append(opened)

        ids = gold_ids.get(str(row.get("claim_id", "")), [])
        gold = dict(close)
        gold.update(
            {
                "evidence_texts": [documents[evidence_id] for evidence_id in ids if evidence_id in documents],
                "text_evidence_ids": [evidence_id for evidence_id in ids if evidence_id in documents],
                "image_paths": row.get("image_paths", []),
                "image_evidence_ids": row.get("image_evidence_ids", []),
                "protocol": "open_gold_oracle",
                "model_inputs": ["claim", "claim_image_paths", "evidence_texts", "image_paths"],
                "evidence_provenance": "gold_qrels_oracle",
            }
        )
        gold_rows.append(gold)

        audit_rows.append(
            {
                "id": sample_id,
                "label": row["label"],
                "retrieved_evidence_ids": retrieved_ids,
                "gold_evidence_ids": ids,
                "first_gold_rank": retrieved_row.get("first_gold_rank"),
            }
        )

    for name, protocol_rows in (
        ("close", close_rows),
        ("open_retrieved", open_rows),
        ("open_gold_oracle", gold_rows),
    ):
        ids = [row["id"] for row in protocol_rows]
        labels = [row["label"] for row in protocol_rows]
        if ids != [row["id"] for row in source_rows]:
            failures.append(f"{split}:{name}: ID order mismatch")
        if labels != [row["label"] for row in source_rows]:
            failures.append(f"{split}:{name}: label mismatch")

    leaked_close = [
        row["id"]
        for row in close_rows
        if row["evidence_texts"]
        or row["text_evidence_ids"]
        or row["image_paths"]
        or row["image_evidence_ids"]
    ]
    if leaked_close:
        failures.append(f"{split}: close protocol leaks evidence in {len(leaked_close)} rows")

    write_jsonl(output_root / "close" / f"{split}.jsonl", close_rows)
    write_jsonl(output_root / "open_retrieved" / f"{split}.jsonl", open_rows)
    write_jsonl(output_root / "open_gold_oracle" / f"{split}.jsonl", gold_rows)
    write_jsonl(output_root / "audit" / f"{split}.jsonl", audit_rows)

    summary = {
        "split": split,
        "samples": len(source_rows),
        "label_counts": dict(Counter(str(row["label"]) for row in source_rows)),
        "close_claim_images": sum(bool(row["claim_image_paths"]) for row in close_rows),
        "close_evidence_rows": sum(bool(row["evidence_texts"] or row["image_paths"]) for row in close_rows),
        "retrieved_text_evidence_rows": sum(bool(row["evidence_texts"]) for row in open_rows),
        "gold_text_evidence_rows": sum(bool(row["evidence_texts"]) for row in gold_rows),
        "gold_image_evidence_rows": sum(bool(row["image_paths"]) for row in gold_rows),
    }
    return summary, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/processed/mocheg_manifest_strict"),
    )
    parser.add_argument(
        "--retrieval-root",
        type=Path,
        default=Path("outputs/retrieval_mocheg_dense_top50"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/mocheg_dataset/extracted/mocheg"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/mocheg_protocols"),
    )
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for split in SPLITS:
        summary, split_failures = build_split(
            split,
            args.manifest_root,
            args.retrieval_root,
            args.raw_root,
            args.output_root,
        )
        summaries.append(summary)
        failures.extend(split_failures)

    report = {
        "status": "pass" if not failures else "fail",
        "definitions": {
            "close": "claim and claim-owned image only; no qrel/retrieved evidence",
            "open_retrieved": "claim plus system-retrieved text evidence",
            "open_gold_oracle": "claim plus gold text/image evidence; diagnostic only",
        },
        "splits": summaries,
        "failures": failures,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "protocol_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
