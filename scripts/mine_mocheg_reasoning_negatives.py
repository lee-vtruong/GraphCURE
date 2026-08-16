"""Mine structured hard negatives for evidence utility learning.

Unlike random or dense-only negatives, these candidates are stratified by
fact-checking failure modes: negation, conflicting quantities, and high lexical
overlap. Only the train split should be used to optimize model parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from graphcure.retrieval import contradiction_features


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_docs(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(
                row["evidence_id"].strip(),
                row.get("Evidence", "").replace("<p>", " ").replace("</p>", " ").strip(),
            )
    return result


def negative_type(features: dict[str, float | bool]) -> str:
    if features["number_mismatch"]:
        return "quantity_trap"
    if features["negation_mismatch"]:
        return "negation_trap"
    if float(features["lexical_jaccard"]) >= 0.18:
        return "lexical_trap"
    return "semantic_hard"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict/train.jsonl"))
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/mocheg_reasoning_negatives/train.jsonl"))
    parser.add_argument("--positives-per-claim", type=int, default=4)
    parser.add_argument("--negatives-per-claim", type=int, default=12)
    args = parser.parse_args()

    claims = {row["id"]: row for row in read_jsonl(args.manifest)}
    documents = read_docs(args.raw_csv)
    output: list[dict] = []
    type_counts: Counter[str] = Counter()
    claims_without_positive = 0
    for retrieval in read_jsonl(args.retrieval):
        claim = claims[retrieval["id"]]
        gold = [
            str(value) for value in claim.get("text_evidence_ids", [])
            if documents.get(str(value))
        ]
        if not gold:
            claims_without_positive += 1
        scores = retrieval.get("retrieved_scores", [])
        negatives: list[dict] = []
        for rank, evidence_id in enumerate(
            retrieval.get("retrieved_evidence_ids", []), start=1
        ):
            if evidence_id in gold or not documents.get(evidence_id):
                continue
            features = contradiction_features(claim.get("claim", ""), documents[evidence_id])
            kind = negative_type(features)
            type_counts[kind] += 1
            negatives.append({
                "evidence_id": evidence_id,
                "rank": rank,
                "score": float(scores[rank - 1]) if rank <= len(scores) else 0.0,
                "type": kind,
                **features,
            })
            if len(negatives) >= args.negatives_per_claim:
                break
        output.append({
            "id": claim["id"],
            "claim_id": claim["claim_id"],
            "claim": claim.get("claim", ""),
            "label": int(claim["label"]),
            "positive_evidence_ids": gold[:args.positives_per_claim],
            "hard_negatives": negatives,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
        encoding="utf-8",
    )
    report = {
        "claims": len(output),
        "claims_without_text_positive": claims_without_positive,
        "negative_type_counts": dict(type_counts),
        "warning": (
            "Non-qrel candidates are weak negatives and may contain unlabeled relevant "
            "evidence; use relevance loss weights conservatively."
        ),
    }
    report_path = args.output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
