"""Create an independently retrieved, zero-shot SciFact evaluation protocol.

SciFact development labels are derived only for evaluation.  Retrieval uses a
deterministic corpus-wide TF--IDF ranker and never reads cited document IDs or
gold rationale sentence IDs.  The generated manifest/corpus/retrieval files
are compatible with the existing Qwen direct-verdict inference dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


LABELS = {"SUPPORT": 0, "CONTRADICT": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_for_claim(row: dict) -> int:
    evidence = row.get("evidence", {})
    if not evidence:
        return 2  # NOT ENOUGH INFORMATION
    values = {
        annotation["label"]
        for annotations in evidence.values()
        for annotation in annotations
    }
    if len(values) != 1 or not values <= set(LABELS):
        raise ValueError(f"unexpected or conflicting SciFact evidence labels for {row['id']}: {values}")
    return LABELS[values.pop()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-claims", type=int, default=0,
                        help="Diagnostic-only cap; zero evaluates every labelled input claim.")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    corpus_rows = read_jsonl(args.corpus)
    claims = read_jsonl(args.claims)
    if args.max_claims:
        claims = claims[:args.max_claims]
    if not corpus_rows or not claims:
        raise ValueError("SciFact corpus and claim split must be non-empty")
    doc_ids = [str(row["doc_id"]) for row in corpus_rows]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("duplicate SciFact corpus doc IDs")
    text = [
        " ".join(part for part in [row.get("title", ""), *row.get("abstract", [])] if part).strip()
        for row in corpus_rows
    ]
    if any(not value for value in text):
        raise ValueError("empty SciFact corpus text")
    labels = [label_for_claim(row) for row in claims]
    if len({str(row["id"]) for row in claims}) != len(claims):
        raise ValueError("duplicate SciFact claim IDs")

    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        sublinear_tf=True, norm="l2", dtype=np.float32,
    )
    document_matrix = vectorizer.fit_transform(text)
    query_matrix = vectorizer.transform([row["claim"] for row in claims])
    scores = (query_matrix @ document_matrix.T).toarray()
    if args.top_k > len(doc_ids):
        raise ValueError("top-k exceeds corpus size")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "dev_manifest.jsonl"
    retrieval_path = args.output_root / "dev_retrieval_tfidf.jsonl"
    corpus_path = args.output_root / "Corpus2.csv"
    manifest_rows = [
        {"id": f"scifact-dev-{row['id']}", "claim": row["claim"], "label": label}
        for row, label in zip(claims, labels)
    ]
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with corpus_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["evidence_id", "Evidence"])
        writer.writeheader()
        for doc_id, value in zip(doc_ids, text):
            writer.writerow({"evidence_id": doc_id, "Evidence": value})
    with retrieval_path.open("w", encoding="utf-8") as handle:
        for row, label, row_scores in zip(manifest_rows, labels, scores):
            candidate = np.argpartition(-row_scores, args.top_k - 1)[:args.top_k]
            candidate = candidate[np.argsort(-row_scores[candidate], kind="stable")]
            handle.write(json.dumps({
                "id": row["id"], "label": label,
                "retrieved_evidence_ids": [doc_ids[int(index)] for index in candidate],
                "tfidf_scores": [float(row_scores[int(index)]) for index in candidate],
            }) + "\n")
    report = {
        "protocol": "SciFact_dev_zero_shot_independent_tfidf_retrieval",
        "split": "dev", "claims": len(manifest_rows), "corpus_documents": len(doc_ids),
        "top_k": args.top_k,
        "label_counts": {str(label): labels.count(label) for label in (0, 1, 2)},
        "retriever": {
            "name": "TF-IDF", "ngram_range": [1, 2], "sublinear_tf": True,
            "uses_claim_labels": False, "uses_cited_doc_ids": False,
            "uses_gold_evidence": False,
        },
        "source_hashes": {"corpus": sha256(args.corpus), "claims": sha256(args.claims)},
        "output_hashes": {
            "manifest": sha256(manifest_path), "retrieval": sha256(retrieval_path),
            "corpus": sha256(corpus_path),
        },
        "gold_labels_used_for_evaluation_only": True,
        "gold_evidence_used": False,
        "test_split_used": False,
    }
    report_path = args.output_root / "protocol.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
