"""Simulate a label-free adaptive query policy on an existing C1 snapshot."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from graphcure.open_web import evidence_quality, load_jsonl


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def policy_requires_expansion(quality: dict, args: argparse.Namespace) -> bool:
    return (
        quality["results"] < args.min_results
        or quality["usable_snippets"] < args.min_usable_snippets
        or quality["domains"] < args.min_domains
        or quality["claim_keyword_coverage"] < args.min_keyword_coverage
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--min-results", type=int, default=8)
    parser.add_argument("--min-usable-snippets", type=int, default=5)
    parser.add_argument("--min-domains", type=int, default=5)
    parser.add_argument("--min-keyword-coverage", type=float, default=0.35)
    args = parser.parse_args()

    rows = load_jsonl(args.snapshot_root / f"{args.split}.jsonl")
    if not rows:
        parser.error("snapshot has no rows")

    audit_rows = []
    original_calls = 0
    adaptive_calls = 0
    expansions = 0
    selected_usable = []
    selected_domains = []
    selected_coverage = []

    for row in rows:
        if not row.get("queries"):
            raise ValueError(f"claim {row.get('id')} has no query plan")
        first_constraint = row["queries"][0]["constraint"]
        first_evidence = [
            item for item in row.get("evidence", [])
            if first_constraint in item.get("query_ranks", {})
        ]
        first_quality = evidence_quality(row.get("claim", ""), first_evidence)
        expand = policy_requires_expansion(first_quality, args)
        selected = row.get("evidence", []) if expand else first_evidence
        selected_quality = evidence_quality(row.get("claim", ""), selected)
        calls = int(row.get("search_calls", len(row["queries"])))
        original_calls += calls
        adaptive_calls += calls if expand else 1
        expansions += int(expand)
        selected_usable.append(selected_quality["usable_snippets"])
        selected_domains.append(selected_quality["domains"])
        selected_coverage.append(selected_quality["claim_keyword_coverage"])
        audit_rows.append({
            "id": row.get("id"),
            "claim": row.get("claim", ""),
            "expand_query": expand,
            "first_constraint": first_constraint,
            "first_quality": first_quality,
            "selected_quality": selected_quality,
            "selected_evidence": selected[:5],
            "labels_or_gold_used": False,
        })

    result = {
        "protocol": "C1.2_label_free_counterfactual_query_budget",
        "snapshot_root": str(args.snapshot_root),
        "split": args.split,
        "claims": len(rows),
        "policy": {
            "first_query": "first deterministic constraint query (semantic)",
            "min_results": args.min_results,
            "min_usable_snippets": args.min_usable_snippets,
            "min_domains": args.min_domains,
            "min_keyword_coverage": args.min_keyword_coverage,
        },
        "original_search_calls": original_calls,
        "adaptive_search_calls": adaptive_calls,
        "saved_search_calls": original_calls - adaptive_calls,
        "mean_queries_per_claim": adaptive_calls / len(rows),
        "expanded_claims": expansions,
        "expansion_rate": expansions / len(rows),
        "selected_evidence_quality": {
            "minimum_usable_snippets": min(selected_usable),
            "median_usable_snippets": statistics.median(selected_usable),
            "minimum_domains": min(selected_domains),
            "median_domains": statistics.median(selected_domains),
            "minimum_keyword_coverage": min(selected_coverage),
            "median_keyword_coverage": statistics.median(selected_coverage),
        },
        "label_used": False,
        "gold_evidence_used": False,
        "test_split_used": args.split == "test",
    }
    atomic_json(args.output, result)
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in audit_rows),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
