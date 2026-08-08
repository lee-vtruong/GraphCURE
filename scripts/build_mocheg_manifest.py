"""Build a claim-level, evidence-aware MOCHEG JSONL manifest.

The output is an audit/interchange format; it deliberately does not create
embeddings or silently collapse MOCHEG's three-way labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = {
    "true": 0,
    "supported": 0,
    "refuted": 1,
    "false": 1,
    "nei": 2,
    "mixture": 2,
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def relevant(path: Path, key: str = "evidence_id") -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for row in rows(path):
        if row.get("RELEVANCY", "0").strip() != "1":
            continue
        topic, value = row.get("TOPIC", "").strip(), row.get(key, "").strip()
        if topic and value and value not in result[topic]:
            result[topic].append(value)
    return result


def image_index(root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for directory in (root / "images", root / "train" / "images",
                      root / "val" / "images", root / "test" / "images"):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.name not in index:
                index[path.name] = str(path)
    return index


def build(root: Path, split: str, output: Path) -> dict[str, Any]:
    split_dir = root / split
    corpus = rows(split_dir / "Corpus2.csv")
    by_claim: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in corpus:
        by_claim[row["claim_id"].strip()].append(row)

    text_article = relevant(split_dir / "text_evidence_qrels_article_level.csv")
    text_sentence = relevant(split_dir / "text_evidence_qrels_sentence_level.csv",
                              key="DOCUMENT#")
    image_qrels = relevant(split_dir / "img_evidence_qrels.csv", key="DOCUMENT#")
    image_evidence = relevant(split_dir / "img_evidence_qrels.csv", key="evidence_id")
    images = image_index(root)

    manifest, orphan = [], Counter()
    for claim_id, evidence_rows in sorted(by_claim.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        first = evidence_rows[0]
        raw_label = first.get("cleaned_truthfulness", "").strip().lower()
        label = LABELS.get(raw_label, -1)
        if label < 0:
            orphan[f"unknown_label:{raw_label}"] += 1
        image_names = image_qrels.get(claim_id, [])
        proof_names = image_evidence.get(claim_id, [])
        missing = [name for name in set(image_names + proof_names) if name not in images]
        orphan["missing_image_references"] += len(missing)
        manifest.append({
            "id": f"mocheg-{split}-claim-{claim_id}",
            "claim_id": claim_id,
            "label": label,
            "label_name": raw_label,
            "claim": first.get("Claim", ""),
            "evidence_texts": [row.get("Evidence", "") for row in evidence_rows if row.get("Evidence", "")],
            "ruling_outline": first.get("ruling_outline", ""),
            "origin": first.get("Origin", ""),
            "headline": first.get("Headline", ""),
            "text_evidence_ids": text_article.get(claim_id, []),
            "text_sentence_ids": text_sentence.get(claim_id, []),
            "image_candidate_names": image_names,
            "image_evidence_ids": proof_names,
            "image_paths": [images[name] for name in image_names + proof_names if name in images],
            "missing_image_names": missing,
            "source": first.get("fact_checkor_website", ""),
            "snopes_url": first.get("Snopes URL", ""),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "split": split, "claims": len(manifest), "corpus_rows": len(corpus),
        "label_counts": dict(Counter(item["label_name"] for item in manifest)),
        "mapped_label_counts": dict(Counter(item["label"] for item in manifest)),
        "claims_with_text_evidence": sum(bool(x["text_evidence_ids"]) for x in manifest),
        "claims_with_image_evidence": sum(bool(x["image_evidence_ids"]) for x in manifest),
        "claims_with_image_candidates": sum(bool(x["image_candidate_names"]) for x in manifest),
        "claims_with_missing_images": sum(bool(x["missing_image_names"]) for x in manifest),
        "orphan_counters": dict(orphan), "image_index_size": len(images),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/mocheg_manifest"))
    args = parser.parse_args()
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        output = args.output_root / f"{split}.jsonl"
        print(json.dumps(build(args.root.resolve(), split, output), indent=2))


if __name__ == "__main__":
    main()
