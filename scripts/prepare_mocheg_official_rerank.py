"""Seed an official-split reranker run from the frozen strict-split output.

The official MOCHEG test manifest contains eight claim-text duplicates removed
by the strict protocol.  Retrieval/reranking is claim independent, so rows for
the 2,434 shared IDs can be reused exactly.  ``rerank_mocheg_qwen3 --resume``
then scores only the eight official-only claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unique_by_id(rows: list[dict], source: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in result:
            raise ValueError(f"duplicate id {sample_id!r} in {source}")
        result[sample_id] = row
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-manifest",
        type=Path,
        default=Path("data/processed/mocheg_manifest/test.jsonl"),
    )
    parser.add_argument(
        "--strict-manifest",
        type=Path,
        default=Path("data/processed/mocheg_manifest_strict/test.jsonl"),
    )
    parser.add_argument(
        "--official-candidates",
        type=Path,
        default=Path("outputs/retrieval_mocheg_qwen3_hybrid_official/test.jsonl"),
    )
    parser.add_argument(
        "--strict-candidates",
        type=Path,
        default=Path("outputs/retrieval_mocheg_qwen3_hybrid/test.jsonl"),
    )
    parser.add_argument(
        "--strict-reranked",
        type=Path,
        default=Path("outputs/retrieval_mocheg_qwen3_reranked/test.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/retrieval_mocheg_qwen3_reranked_official/test.jsonl"),
    )
    parser.add_argument("--expected-official", type=int, default=2442)
    parser.add_argument("--expected-strict", type=int, default=2434)
    args = parser.parse_args()

    official_rows = read_jsonl(args.official_manifest)
    strict_rows = read_jsonl(args.strict_manifest)
    candidate_rows = read_jsonl(args.official_candidates)
    strict_candidate_rows = read_jsonl(args.strict_candidates)
    reranked_rows = read_jsonl(args.strict_reranked)
    official = unique_by_id(official_rows, args.official_manifest)
    strict = unique_by_id(strict_rows, args.strict_manifest)
    candidates = unique_by_id(candidate_rows, args.official_candidates)
    strict_candidates = unique_by_id(strict_candidate_rows, args.strict_candidates)
    reranked = unique_by_id(reranked_rows, args.strict_reranked)

    if len(official) != args.expected_official:
        parser.error(f"official manifest has {len(official)}, expected {args.expected_official}")
    if len(strict) != args.expected_strict:
        parser.error(f"strict manifest has {len(strict)}, expected {args.expected_strict}")
    if set(candidates) != set(official):
        parser.error("official candidate IDs do not exactly match official manifest IDs")
    if set(reranked) != set(strict):
        parser.error("strict reranked IDs do not exactly match strict manifest IDs")
    if set(strict_candidates) != set(strict):
        parser.error("strict candidate IDs do not exactly match strict manifest IDs")
    if not set(strict) < set(official):
        parser.error("strict IDs must be a proper subset of official IDs")

    for sample_id, strict_row in strict.items():
        official_row = official[sample_id]
        for key in ("claim_id", "claim", "label"):
            if strict_row.get(key) != official_row.get(key):
                parser.error(f"shared sample {sample_id} differs in field {key}")
        if strict_candidates[sample_id].get("retrieved_evidence_ids") != candidates[
            sample_id
        ].get("retrieved_evidence_ids"):
            parser.error(
                f"official and strict candidate rankings differ for {sample_id}; "
                "run a full official rerank instead of seeding"
            )
        if reranked[sample_id].get("retrieval_signature") != candidates[sample_id].get(
            "retrieval_signature"
        ):
            parser.error(f"retrieval signature mismatch for shared sample {sample_id}")

    existing: dict[str, dict] = {}
    if args.output.exists():
        existing = unique_by_id(read_jsonl(args.output), args.output)
        if not set(existing) <= set(official):
            parser.error("existing output contains IDs outside the official manifest")
    seeded = dict(existing)
    for sample_id, row in reranked.items():
        previous = seeded.get(sample_id)
        if previous is not None and previous.get("reranker_signature") != row.get(
            "reranker_signature"
        ):
            parser.error(f"existing reranker signature mismatch for {sample_id}")
        seeded.setdefault(sample_id, row)

    ordered = [seeded[row["id"]] for row in official_rows if row["id"] in seeded]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ordered) + "\n",
        encoding="utf-8",
    )
    pending = [row["id"] for row in official_rows if row["id"] not in seeded]
    report = {
        "official_samples": len(official),
        "strict_samples_reused": len(reranked),
        "already_present_official_only": len(existing) - len(set(existing) & set(strict)),
        "pending_official_only": len(pending),
        "pending_ids": pending,
        "official_manifest_sha256": sha256(args.official_manifest),
        "official_candidates_sha256": sha256(args.official_candidates),
        "strict_candidates_sha256": sha256(args.strict_candidates),
        "strict_reranked_sha256": sha256(args.strict_reranked),
        "gold_used_for_ranking": False,
    }
    report_path = args.output.parent / "seed_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
