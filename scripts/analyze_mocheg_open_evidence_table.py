"""Audit C2a constraints and build a label-free, source-diverse C2b shortlist."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from graphcure.open_web import load_jsonl, normalize_space, sha256_file, sha256_text


def is_usable(row: dict, minimum_chars: int) -> bool:
    return len(normalize_space(row.get("text", ""))) >= minimum_chars


def select_diverse(
    evidence: list[dict],
    top_k: int,
    minimum_chars: int,
    max_per_domain: int,
    max_social: int,
) -> list[dict]:
    """Preserve search rank while constraining domain/social concentration."""
    ranked = sorted(evidence, key=lambda row: int(row.get("rank", 10**9)))
    selected: list[dict] = []
    selected_ids: set[str] = set()
    domain_counts: Counter[str] = Counter()
    social_count = 0

    def add(row: dict, stage: str, enforce_caps: bool) -> bool:
        nonlocal social_count
        evidence_id = str(row.get("evidence_id", ""))
        domain = str(row.get("domain") or "unknown")
        social = row.get("source_family") == "social"
        if evidence_id in selected_ids:
            return False
        if enforce_caps and domain_counts[domain] >= max_per_domain:
            return False
        if enforce_caps and social and social_count >= max_social:
            return False
        item = dict(row)
        item["shortlist_stage"] = stage
        item["original_rank"] = int(row.get("rank", 10**9))
        item["shortlist_rank"] = len(selected) + 1
        selected.append(item)
        selected_ids.add(evidence_id)
        domain_counts[domain] += 1
        social_count += int(social)
        return True

    # Stage 1 is the intended policy: usable, distinct-domain, social-capped.
    for row in ranked:
        if is_usable(row, minimum_chars):
            add(row, "usable_diverse", enforce_caps=True)
        if len(selected) == top_k:
            return selected
    # Stage 2 fills from usable evidence if strict diversity left too few rows.
    for row in ranked:
        if is_usable(row, minimum_chars):
            add(row, "usable_cap_relaxation", enforce_caps=False)
        if len(selected) == top_k:
            return selected
    # Stage 3 retains a short snippet only when a claim lacks enough usable rows.
    for row in ranked:
        add(row, "short_text_fallback", enforce_caps=False)
        if len(selected) == top_k:
            return selected
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shortlist-output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--minimum-text-chars", type=int, default=80)
    parser.add_argument("--max-per-domain", type=int, default=1)
    parser.add_argument("--max-social", type=int, default=2)
    args = parser.parse_args()
    if min(args.top_k, args.minimum_text_chars, args.max_per_domain) <= 0:
        parser.error("top-k, minimum characters and domain cap must be positive")
    if args.max_social < 0:
        parser.error("social cap cannot be negative")

    summary_path = args.table_root / "summary.json"
    table_path = args.table_root / f"{args.split}.jsonl"
    table_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if table_summary.get("label_used") or table_summary.get("gold_evidence_used"):
        parser.error("C2a table contains forbidden supervision")
    if sha256_file(table_path) != table_summary.get("output_sha256"):
        parser.error("C2a table hash mismatch")
    rows = load_jsonl(table_path)

    weak_claims = []
    social_dominated = []
    temporal_claims = 0
    temporal_claims_with_overlap = 0
    shortlist_usable, shortlist_domains = [], []
    shortlist_families: Counter[str] = Counter()
    fallback_stages: Counter[str] = Counter()
    duplicate_text_keys: Counter[str] = Counter()
    args.shortlist_output.parent.mkdir(parents=True, exist_ok=True)
    with args.shortlist_output.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc=f"{args.split} C2a audit"):
            evidence = row.get("evidence_table", [])
            usable = [item for item in evidence if is_usable(
                item, args.minimum_text_chars
            )]
            if len(usable) < 5:
                weak_claims.append({
                    "id": row["id"], "claim": row.get("claim", ""),
                    "usable_evidence": len(usable), "evidence": len(evidence),
                })
            social = sum(item.get("source_family") == "social" for item in evidence)
            if evidence and social / len(evidence) > 0.5:
                social_dominated.append({
                    "id": row["id"], "claim": row.get("claim", ""),
                    "social_share": social / len(evidence),
                })
            claim_temporal = row.get("claim_constraints", {}).get(
                "temporal_mentions", []
            )
            if claim_temporal:
                temporal_claims += 1
                temporal_claims_with_overlap += int(any(
                    item.get("temporal_overlap", {}).get("count", 0) > 0
                    for item in evidence
                ))
            for item in evidence:
                visible = normalize_space(item.get("text", "")).casefold()
                if len(visible) >= args.minimum_text_chars:
                    duplicate_text_keys[sha256_text(visible)] += 1

            shortlist = select_diverse(
                evidence, args.top_k, args.minimum_text_chars,
                args.max_per_domain, args.max_social,
            )
            shortlist_usable.append(sum(
                is_usable(item, args.minimum_text_chars) for item in shortlist
            ))
            shortlist_domains.append(len({
                item.get("domain") for item in shortlist if item.get("domain")
            }))
            shortlist_families.update(
                item.get("source_family", "unknown") for item in shortlist
            )
            fallback_stages.update(
                item["shortlist_stage"] for item in shortlist
            )
            output = {
                "id": row["id"], "claim_id": row.get("claim_id"),
                "claim": row.get("claim", ""),
                "claim_constraints": row.get("claim_constraints", {}),
                "evidence_shortlist": shortlist,
                "label_included": False, "gold_evidence_used": False,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    shortlist_rows = sum(shortlist_families.values())
    report = {
        "protocol": "P2_open_web_C2a_audit_and_diverse_shortlist",
        "split": args.split,
        "claims": len(rows),
        "weak_claims_below_5_usable": len(weak_claims),
        "weak_claim_rate": len(weak_claims) / len(rows),
        "weak_claim_examples": weak_claims[:20],
        "social_dominated_claims": len(social_dominated),
        "social_dominated_claim_rate": len(social_dominated) / len(rows),
        "social_dominated_examples": social_dominated[:20],
        "temporal_claims": temporal_claims,
        "temporal_claims_with_any_overlap": temporal_claims_with_overlap,
        "conditional_temporal_claim_coverage": (
            temporal_claims_with_overlap / temporal_claims
            if temporal_claims else None
        ),
        "duplicate_usable_text_groups": sum(
            count > 1 for count in duplicate_text_keys.values()
        ),
        "shortlist": {
            "top_k": args.top_k,
            "minimum_usable": min(shortlist_usable) if shortlist_usable else 0,
            "median_usable": statistics.median(shortlist_usable),
            "minimum_domains": min(shortlist_domains) if shortlist_domains else 0,
            "median_domains": statistics.median(shortlist_domains),
            "source_family_counts": dict(shortlist_families),
            "social_share": (
                shortlist_families["social"] / shortlist_rows
                if shortlist_rows else 0.0
            ),
            "selection_stage_counts": dict(fallback_stages),
        },
        "input_table_sha256": table_summary["output_sha256"],
        "shortlist_sha256": sha256_file(args.shortlist_output),
        "label_used": False,
        "gold_evidence_used": False,
        "test_split_used": args.split == "test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
