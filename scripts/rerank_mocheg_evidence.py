"""Cross-encoder reranking for dense MOCHEG retrieval candidates."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from sentence_transformers import CrossEncoder

def read_csv(path):
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f: return list(csv.DictReader(f))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--retrieval-root", type=Path, required=True); p.add_argument("--manifest-root", type=Path, required=True); p.add_argument("--raw-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"); p.add_argument("--top-k", type=int, default=5); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--device", default="cuda"); a = p.parse_args(); a.output_root.mkdir(parents=True, exist_ok=True); model = CrossEncoder(a.model, device=a.device)
    summary = {}
    for split in ("train", "val", "test"):
        docs = {r["evidence_id"]: r.get("Evidence", "").replace("<p>", " ").replace("</p>", " ").strip() for r in read_csv(a.raw_root / split / "Corpus2.csv")}
        claims = {r["id"]: r for r in [json.loads(x) for x in (a.manifest_root / f"{split}.jsonl").read_text().splitlines() if x.strip()]}
        out, hit, rr = [], [], []
        with (a.retrieval_root / f"{split}.jsonl").open() as f:
            for line in f:
                r = json.loads(line); claim = claims[r["id"]]["claim"]; candidates = [x for x in r["retrieved_evidence_ids"] if docs.get(x)]; pairs = [(claim, docs[x]) for x in candidates]; scores = model.predict(pairs, batch_size=a.batch_size, show_progress_bar=False) if pairs else np.array([]); order = np.argsort(-scores)[:a.top_k]; ids = [candidates[i] for i in order]; vals = [float(scores[i]) for i in order]; gold = set(r.get("gold_evidence_ids", [])); rank = next((j + 1 for j, x in enumerate(ids) if x in gold), None); hit.append(float(rank is not None)); rr.append(1 / rank if rank else 0.0); out.append({**r, "retrieved_evidence_ids": ids, "retrieved_scores": vals, "retrieval_confidence": vals[0] if vals else 0.0, "first_gold_rank": rank})
        (a.output_root / f"{split}.jsonl").write_text("\n".join(json.dumps(x) for x in out) + "\n"); summary[split] = {"claims": len(out), "recall@" + str(a.top_k): float(np.mean(hit)), "mrr": float(np.mean(rr))}; print(split, summary[split])
    (a.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if __name__ == "__main__": main()
