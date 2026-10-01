"""Export reproducible qualitative AND-routing case cards from frozen artifacts.

This is a reporting utility.  It neither loads a model nor changes a routing
policy.  It joins a frozen manifest, frozen retrieval output, and the
materialized direct/AND prediction files from the canonical router evaluation.
Cases are selected deterministically (lexicographic ID) within predeclared
direct-to-AND transition strata, so no test labels are used for system or
parameter selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


LABEL_NAMES = {0: "Supported", 1: "Refuted", 2: "NEI"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON row in {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no JSONL rows in {path}")
    return rows


def index_by_id(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "id" not in row:
            raise KeyError(f"row without id in {path}")
        key = str(row["id"])
        if key in result:
            raise ValueError(f"duplicate id {key!r} in {path}")
        result[key] = row
    return result


def read_corpus(path: Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            evidence_id = str(row.get("evidence_id", "")).strip()
            text = str(row.get("Evidence", "")).replace("<p>", " ").replace("</p>", " ").strip()
            if evidence_id and text:
                documents.setdefault(evidence_id, text)
    return documents


def label(row: dict[str, Any], path: Path) -> int:
    for key in ("label", "gold", "gold_label", "target"):
        if key in row:
            value = int(row[key])
            if value in LABEL_NAMES:
                return value
    raise KeyError(f"no valid label in {path} row {row.get('id')!r}")


def prediction(row: dict[str, Any], path: Path) -> int:
    """Read a model output, never the gold label retained beside it."""
    if "prediction" not in row:
        raise KeyError(f"no prediction field in {path} row {row.get('id')!r}")
    value = int(row["prediction"])
    if value not in LABEL_NAMES:
        raise ValueError(f"invalid prediction {value} in {path} row {row.get('id')!r}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--direct-predictions", type=Path, required=True)
    parser.add_argument("--and-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--per-pattern", type=int, default=2)
    parser.add_argument("--max-claim-chars", type=int, default=500)
    parser.add_argument("--max-evidence-chars", type=int, default=800)
    args = parser.parse_args()
    if args.top_k < 1 or args.per_pattern < 1:
        parser.error("--top-k and --per-pattern must be positive")

    manifest = index_by_id(read_jsonl(args.manifest), args.manifest)
    retrieval = index_by_id(read_jsonl(args.retrieval), args.retrieval)
    direct = index_by_id(read_jsonl(args.direct_predictions), args.direct_predictions)
    routed = index_by_id(read_jsonl(args.and_predictions), args.and_predictions)
    ids = set(manifest)
    for name, mapping in (("retrieval", retrieval), ("direct", direct), ("and", routed)):
        if set(mapping) != ids:
            raise ValueError(f"{name} IDs do not exactly match manifest IDs")
    documents = read_corpus(args.corpus)

    # These are the six transition types summarized in Appendix D.  Gold is
    # used only to stratify retrospective case reporting, never for routing.
    patterns = [
        (2, 1, 2, "helpful_nei_from_refuted"),
        (2, 0, 2, "helpful_nei_from_supported"),
        (0, 0, 2, "harmful_supported_to_nei"),
        (0, 1, 2, "both_wrong_supported_refuted_to_nei"),
        (1, 1, 2, "harmful_refuted_to_nei"),
        (1, 0, 2, "both_wrong_refuted_supported_to_nei"),
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prediction_changes = 0
    for sample_id in sorted(ids):
        gold = label(manifest[sample_id], args.manifest)
        direct_pred = prediction(direct[sample_id], args.direct_predictions)
        and_pred = prediction(routed[sample_id], args.and_predictions)
        prediction_changes += int(direct_pred != and_pred)
        for expected_gold, expected_direct, expected_and, pattern in patterns:
            if (gold, direct_pred, and_pred) == (expected_gold, expected_direct, expected_and):
                buckets[pattern].append({
                    "id": sample_id,
                    "gold": gold,
                    "direct": direct_pred,
                    "and": and_pred,
                })
                break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    pattern_counts = {name: len(buckets[name]) for _, _, _, name in patterns}
    for _, _, _, pattern in patterns:
        for case in buckets[pattern][:args.per_pattern]:
            sample_id = case["id"]
            claim = str(manifest[sample_id].get("claim", ""))
            evidence_ids = [str(value) for value in retrieval[sample_id].get("retrieved_evidence_ids", [])[:args.top_k]]
            evidence = [
                {"evidence_id": evidence_id, "text": clean(documents[evidence_id], args.max_evidence_chars)}
                for evidence_id in evidence_ids if evidence_id in documents
            ]
            selected.append({
                "pattern": pattern,
                **case,
                "claim": clean(claim, args.max_claim_chars),
                "retrieved_evidence": evidence,
                "direct_probabilities": direct[sample_id].get("probabilities"),
                "and_probabilities": routed[sample_id].get("probabilities"),
            })

    json_path = args.output_dir / "qualitative_cases.jsonl"
    with json_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# CURE AND qualitative routing case cards", "",
        "- Selection: deterministic lexicographic ID within six predeclared transition strata.",
        "- Gold labels are used only for retrospective reporting; no model, seed, K, or threshold is changed.",
        f"- Cases per stratum: {args.per_pattern}; retrieved passages shown: up to {args.top_k}.",
        f"- Direct-to-AND prediction changes in the supplied artifact: **{prediction_changes}**.",
        f"- Input hashes (manifest/retrieval/direct/AND): `{sha256_file(args.manifest)}` / `{sha256_file(args.retrieval)}` / `{sha256_file(args.direct_predictions)}` / `{sha256_file(args.and_predictions)}`.",
        "", "## Full transition-stratum counts", "",
        "| Pattern | Matching claims |", "| --- | ---: |",
    ]
    for _, _, _, pattern in patterns:
        lines.append(f"| {pattern} | {pattern_counts[pattern]} |")
    for index, row in enumerate(selected, start=1):
        lines.extend([
            "", f"## Case {index}: {row['pattern']}", "",
            f"- ID: `{row['id']}`",
            f"- Gold / direct / AND: **{LABEL_NAMES[row['gold']]} / {LABEL_NAMES[row['direct']]} / {LABEL_NAMES[row['and']]}**",
            f"- Claim: {row['claim']}",
            f"- Direct probabilities [S,R,N]: `{row['direct_probabilities']}`",
            f"- AND base probabilities [S,R,N]: `{row['and_probabilities']}`",
            "- Retrieved evidence:",
        ])
        if row["retrieved_evidence"]:
            for evidence in row["retrieved_evidence"]:
                lines.append(f"  - `{evidence['evidence_id']}`: {evidence['text']}")
        else:
            lines.append("  - _No retrieved text found in the supplied corpus._")
    markdown_path = args.output_dir / "qualitative_cases.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path} and {markdown_path} ({len(selected)} case cards).")


if __name__ == "__main__":
    main()
