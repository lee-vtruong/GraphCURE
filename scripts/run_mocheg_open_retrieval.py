"""Open-book MOCHEG retrieval baseline using TF-IDF cosine search."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def read_csv(path):
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-root", type=Path, default=Path("data/processed/mocheg_manifest"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/mocheg_open_tfidf"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-features", type=int, default=200000)
    a = p.parse_args(); a.output_root.mkdir(parents=True, exist_ok=True); summary = {}
    for split in ("train", "val", "test"):
        claims = [json.loads(x) for x in (a.manifest_root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        corpus = read_csv(a.raw_root / split / "Corpus2.csv")
        docs, ids = [], []
        for row in corpus:
            text = row.get("Evidence", "").replace("<p>", " ").replace("</p>", " ").strip()
            if text and row.get("evidence_id") not in ids:
                ids.append(row["evidence_id"]); docs.append(text)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=a.max_features)
        matrix = vec.fit_transform(docs); scores = linear_kernel(vec.transform([x.get("claim", "") for x in claims]), matrix)
        out_rows, hits, rr = [], [], []
        for i, claim in enumerate(claims):
            order = np.argsort(-scores[i])[:a.top_k]; retrieved = [ids[j] for j in order]
            retrieved_scores = [float(scores[i, j]) for j in order]
            gold = set(claim.get("text_evidence_ids", [])); rank = next((r + 1 for r, x in enumerate(retrieved) if x in gold), None)
            hits.append(float(rank is not None)); rr.append(1 / rank if rank else 0.0)
            out_rows.append({"id": claim["id"], "claim_id": claim["claim_id"], "label": claim["label"], "retrieved_evidence_ids": retrieved, "retrieved_scores": retrieved_scores, "retrieval_confidence": retrieved_scores[0] if retrieved_scores else 0.0, "gold_evidence_ids": sorted(gold), "first_gold_rank": rank})
        (a.output_root / f"{split}.jsonl").write_text("\n".join(json.dumps(x) for x in out_rows) + "\n", encoding="utf-8")
        summary[split] = {"claims": len(claims), "corpus_documents": len(docs), f"recall@{a.top_k}": float(np.mean(hits)), "mrr": float(np.mean(rr))}
        print(split, summary[split])
    (a.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if __name__ == "__main__": main()
