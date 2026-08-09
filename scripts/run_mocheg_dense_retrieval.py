"""Dense MPNet open-book retrieval baseline for MOCHEG."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def csv_rows(path):
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f: return list(csv.DictReader(f))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest-root", type=Path, default=Path("data/processed/mocheg_manifest_strict")); p.add_argument("--raw-root", type=Path, default=Path("data/raw/mocheg_dataset/extracted/mocheg")); p.add_argument("--output-root", type=Path, default=Path("outputs/mocheg_open_dense")); p.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2"); p.add_argument("--top-k", type=int, default=10); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--device", default="cuda"); a = p.parse_args(); a.output_root.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(a.model, device=a.device)
    summary = {}
    for split in ("train", "val", "test"):
        claims = [json.loads(x) for x in (a.manifest_root / f"{split}.jsonl").read_text().splitlines() if x.strip()]
        docs, ids = [], []
        for row in csv_rows(a.raw_root / split / "Corpus2.csv"):
            text = row.get("Evidence", "").replace("<p>", " ").replace("</p>", " ").strip()
            if text and row["evidence_id"] not in ids: ids.append(row["evidence_id"]); docs.append(text)
        de = model.encode(docs, batch_size=a.batch_size, normalize_embeddings=True, show_progress_bar=True)
        qe = model.encode([x.get("claim", "") for x in claims], batch_size=a.batch_size, normalize_embeddings=True, show_progress_bar=True)
        scores = qe @ de.T; out, hit, rr = [], [], []
        for i, claim in enumerate(claims):
            order = np.argsort(-scores[i])[:a.top_k]; retrieved = [ids[j] for j in order]; vals = [float(scores[i, j]) for j in order]; gold = set(claim.get("text_evidence_ids", [])); rank = next((r + 1 for r, x in enumerate(retrieved) if x in gold), None)
            hit.append(float(rank is not None)); rr.append(1 / rank if rank else 0.0)
            out.append({"id": claim["id"], "claim_id": claim["claim_id"], "label": claim["label"], "retrieved_evidence_ids": retrieved, "retrieved_scores": vals, "retrieval_confidence": vals[0] if vals else 0.0, "gold_evidence_ids": sorted(gold), "first_gold_rank": rank})
        (a.output_root / f"{split}.jsonl").write_text("\n".join(json.dumps(x) for x in out) + "\n")
        summary[split] = {"claims": len(claims), "corpus_documents": len(docs), f"recall@{a.top_k}": float(np.mean(hit)), "mrr": float(np.mean(rr))}; print(split, summary[split])
    (a.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if __name__ == "__main__": main()
