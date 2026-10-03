"""Prepare SciFact training protocol and 5-fold splits for domain adaptation.

Extracts SciFact claims_train.jsonl and corpus.jsonl, performs deterministic TF-IDF
retrieval (exact same vectorizer config as the zero-shot dev audit), and generates
stratified 5-fold cross-validation splits for OOF threshold calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold


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
        return 2  # NOT ENOUGH INFORMATION (NEI)
    values = {
        annotation["label"]
        for annotations in evidence.values()
        for annotation in annotations
    }
    if not values:
        return 2  # NOT ENOUGH INFORMATION (empty annotations)
    if len(values) != 1 or not values <= set(LABELS):
        raise ValueError(f"unexpected or conflicting SciFact evidence labels for {row['id']}: {values}")
    return LABELS[values.pop()]


def build_corpus_documents(corpus_rows: list[dict]) -> tuple[list[str], list[str]]:
    doc_ids = [str(row["doc_id"]) for row in corpus_rows]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("duplicate SciFact corpus doc IDs")
    text = [
        " ".join(part for part in [row.get("title", ""), *row.get("abstract", [])] if part).strip()
        for row in corpus_rows
    ]
    if any(not value for value in text):
        raise ValueError("empty SciFact corpus text")
    return doc_ids, text


def compute_tfidf_retrieval(
    corpus_texts: list[str],
    doc_ids: list[str],
    claims: list[dict],
    top_k: int = 5,
) -> tuple[list[list[str]], list[list[float]]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    document_matrix = vectorizer.fit_transform(corpus_texts)
    query_matrix = vectorizer.transform([row["claim"] for row in claims])
    scores = (query_matrix @ document_matrix.T).toarray()
    if top_k > len(doc_ids):
        raise ValueError("top-k exceeds corpus size")

    retrieved_ids = []
    retrieved_scores = []
    for row_scores in scores:
        candidate = np.argpartition(-row_scores, top_k - 1)[:top_k]
        candidate = candidate[np.argsort(-row_scores[candidate], kind="stable")]
        retrieved_ids.append([doc_ids[int(idx)] for idx in candidate])
        retrieved_scores.append([float(row_scores[int(idx)]) for idx in candidate])
    return retrieved_ids, retrieved_scores


def build_stratified_folds(
    claims: list[dict],
    labels: list[int],
    num_folds: int = 5,
    seed: int = 2040,
) -> list[dict]:
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    all_ids = np.array([f"scifact-train-{r['id']}" for r in claims])
    labels_arr = np.array(labels)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_ids, labels_arr)):
        folds.append({
            "fold": fold_idx,
            "train_ids": all_ids[train_idx].tolist(),
            "val_ids": all_ids[val_idx].tolist(),
            "train_label_counts": dict(Counter(map(str, labels_arr[train_idx].tolist()))),
            "val_label_counts": dict(Counter(map(str, labels_arr[val_idx].tolist()))),
        })
    return folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Path to SciFact corpus.jsonl")
    parser.add_argument("--claims-train", type=Path, required=True,
                        help="Path to SciFact claims_train.jsonl")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Output directory for protocol artifacts")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of retrieved documents per claim (default: 5)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of stratified folds for train OOF (default: 5)")
    parser.add_argument("--seed", type=int, default=2040,
                        help="Random seed for fold splitting (default: 2040)")
    parser.add_argument("--max-claims", type=int, default=0,
                        help="Cap claims for diagnostic testing (0 for full)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    corpus_rows = read_jsonl(args.corpus)
    claims = read_jsonl(args.claims_train)
    if args.max_claims:
        claims = claims[:args.max_claims]

    if not corpus_rows or not claims:
        raise ValueError("SciFact corpus and train claims must be non-empty")

    doc_ids, corpus_texts = build_corpus_documents(corpus_rows)
    labels = [label_for_claim(row) for row in claims]
    if len({str(row["id"]) for row in claims}) != len(claims):
        raise ValueError("duplicate SciFact train claim IDs")

    retrieved_ids, retrieved_scores = compute_tfidf_retrieval(
        corpus_texts, doc_ids, claims, top_k=args.top_k
    )

    manifest_path = args.output_root / "train_manifest.jsonl"
    retrieval_path = args.output_root / "train_retrieval_tfidf.jsonl"
    corpus_path = args.output_root / "Corpus2.csv"
    folds_path = args.output_root / "scifact_adaptation_folds.json"

    manifest_rows = [
        {"id": f"scifact-train-{row['id']}", "claim": row["claim"], "label": label}
        for row, label in zip(claims, labels)
    ]
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with corpus_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["evidence_id", "Evidence"])
        writer.writeheader()
        for doc_id, value in zip(doc_ids, corpus_texts):
            writer.writerow({"evidence_id": doc_id, "Evidence": value})

    with retrieval_path.open("w", encoding="utf-8") as handle:
        for row, label, r_ids, r_scores in zip(manifest_rows, labels, retrieved_ids, retrieved_scores):
            handle.write(json.dumps({
                "id": row["id"],
                "label": label,
                "retrieved_evidence_ids": r_ids,
                "tfidf_scores": r_scores,
            }) + "\n")

    fold_entries = build_stratified_folds(claims, labels, num_folds=args.folds, seed=args.seed)
    folds_data = {
        "protocol": "scifact_adaptation_stratified_cv",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "samples": len(manifest_rows),
        "fold_count": args.folds,
        "seed": args.seed,
        "folds": fold_entries,
    }
    folds_path.write_text(json.dumps(folds_data, indent=2) + "\n", encoding="utf-8")

    report = {
        "protocol": "SciFact_train_adaptation_protocol",
        "split": "train",
        "claims": len(manifest_rows),
        "corpus_documents": len(doc_ids),
        "top_k": args.top_k,
        "num_folds": args.folds,
        "label_counts": {str(label): labels.count(label) for label in (0, 1, 2)},
        "retriever": {
            "name": "TF-IDF",
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "uses_claim_labels": False,
            "uses_cited_doc_ids": False,
            "uses_gold_evidence": False,
        },
        "source_hashes": {
            "corpus": sha256(args.corpus),
            "claims_train": sha256(args.claims_train),
        },
        "output_hashes": {
            "manifest": sha256(manifest_path),
            "retrieval": sha256(retrieval_path),
            "corpus": sha256(corpus_path),
            "folds": sha256(folds_path),
        },
        "gold_evidence_used": False,
        "dev_split_used": False,
        "test_split_used": False,
    }
    report_path = args.output_root / "protocol.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
