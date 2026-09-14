"""Build C2a observable constraint tables from a frozen C1 snapshot."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from graphcure.open_web import (
    evidence_quality,
    extract_entities,
    extract_temporal_mentions,
    load_jsonl,
    sha256_file,
    sha256_text,
    source_family,
)


def normalized_set(values: list[str]) -> set[str]:
    return {value.casefold() for value in values}


def overlap(left: list[str], right: list[str]) -> dict:
    a, b = normalized_set(left), normalized_set(right)
    shared = sorted(a & b)
    return {
        "shared": shared,
        "count": len(shared),
        "jaccard": len(a & b) / len(a | b) if a or b else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-analysis-chars", type=int, default=5000)
    args = parser.parse_args()
    freeze_path = args.freeze_manifest or args.snapshot_root / "c1_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "c1_retrieval_snapshot_frozen":
        parser.error("C2a requires a frozen C1 retrieval snapshot")
    if freeze.get("test_split_used") or freeze.get("gold_evidence_used"):
        parser.error("C2a development input cannot use test or gold evidence")

    rows_path = args.snapshot_root / f"{args.split}.jsonl"
    if sha256_file(rows_path) != freeze.get("rows_sha256"):
        parser.error("snapshot rows changed after C1 freeze")
    rows = load_jsonl(rows_path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / f"{args.split}.jsonl"

    family_counts: Counter[str] = Counter()
    fetch_counts: Counter[str] = Counter()
    entity_overlaps, temporal_overlaps, usable_counts = [], [], []
    claim_entity_overlap = []
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            claim = row.get("claim", "")
            claim_entities = extract_entities(claim)[:20]
            claim_temporal = extract_temporal_mentions(claim)[:20]
            table = []
            for rank, item in enumerate(row.get("evidence", []), start=1):
                text = item.get("text", "")
                analysis_text = (
                    f"{item.get('title', '')} {item.get('snippet', '')} "
                    f"{text[:args.max_analysis_chars]}"
                )
                entities = extract_entities(analysis_text)[:50]
                temporal = extract_temporal_mentions(
                    f"{item.get('published_at') or ''} {analysis_text}"
                )[:50]
                entity_relation = overlap(claim_entities, entities)
                temporal_relation = overlap(claim_temporal, temporal)
                family = source_family(item.get("domain"))
                quality = evidence_quality(claim, [item])
                family_counts[family] += 1
                fetch_counts[item.get("fetch_status", "missing")] += 1
                entity_overlaps.append(entity_relation["count"])
                temporal_overlaps.append(temporal_relation["count"])
                table.append({
                    "evidence_id": sha256_text(item["canonical_url"])[:20],
                    "rank": rank,
                    "canonical_url": item["canonical_url"],
                    "domain": item.get("domain"),
                    "source_family": family,
                    "https": item["canonical_url"].startswith("https://"),
                    "title": item.get("title", ""),
                    "published_at": item.get("published_at"),
                    "query_constraints": sorted(item.get("query_ranks", {})),
                    "query_ranks": item.get("query_ranks", {}),
                    "fetch_status": item.get("fetch_status"),
                    "text": text,
                    "text_chars": len(text),
                    "observable_quality": quality,
                    "entities": entities,
                    "entity_overlap": entity_relation,
                    "temporal_mentions": temporal,
                    "temporal_overlap": temporal_relation,
                    "stance": "unscored",
                    "sufficiency": "unscored",
                })
            usable_counts.append(sum(
                len(item.get("text", "").strip()) >= 80 for item in table
            ))
            claim_entity_overlap.append(any(
                item["entity_overlap"]["count"] > 0 for item in table
            ))
            output = {
                "id": row["id"],
                "claim_id": row.get("claim_id"),
                "claim": claim,
                "claim_constraints": {
                    "entities": claim_entities,
                    "temporal_mentions": claim_temporal,
                },
                "retrieval": {
                    "executed_queries": row.get("executed_queries", row.get("queries", [])),
                    "query_diagnostics": row.get("query_diagnostics", []),
                    "search_calls": row.get("search_calls"),
                },
                "evidence_table": table,
                "label_included": False,
                "gold_evidence_used": False,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    summary = {
        "protocol": "P2_open_web_C2a_observable_constraint_table",
        "split": args.split,
        "claims": len(rows),
        "evidence_rows": sum(family_counts.values()),
        "source_family_counts": dict(family_counts),
        "fetch_status_counts": dict(fetch_counts),
        "claims_with_entity_overlap": sum(claim_entity_overlap),
        "claim_entity_overlap_rate": (
            sum(claim_entity_overlap) / len(claim_entity_overlap)
            if claim_entity_overlap else 0.0
        ),
        "evidence_with_entity_overlap_rate": (
            sum(value > 0 for value in entity_overlaps) / len(entity_overlaps)
            if entity_overlaps else 0.0
        ),
        "evidence_with_temporal_overlap_rate": (
            sum(value > 0 for value in temporal_overlaps) / len(temporal_overlaps)
            if temporal_overlaps else 0.0
        ),
        "minimum_usable_evidence_per_claim": min(usable_counts) if usable_counts else 0,
        "median_usable_evidence_per_claim": (
            statistics.median(usable_counts) if usable_counts else 0.0
        ),
        "input_rows_sha256": freeze["rows_sha256"],
        "output_sha256": sha256_file(output_path),
        "stance_scored": False,
        "sufficiency_scored": False,
        "label_used": False,
        "gold_evidence_used": False,
        "test_split_used": args.split == "test",
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
