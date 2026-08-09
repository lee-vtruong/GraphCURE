"""Attach retrieved (not gold) evidence text to MOCHEG claims."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-root", type=Path, default=Path("outputs/mocheg_open_tfidf_strict"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    p.add_argument("--output-root", type=Path, default=Path("data/processed/mocheg_open_manifest"))
    a = p.parse_args(); a.output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        docs = {}
        with (a.raw_root / split / "Corpus2.csv").open(encoding="utf-8-sig", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                docs.setdefault(row["evidence_id"], row.get("Evidence", ""))
        out = []
        with (a.retrieval_root / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                retrieved = row.get("retrieved_evidence_ids", [])
                row["evidence_texts"] = [docs[x] for x in retrieved if docs.get(x)]
                row["text_evidence_ids"] = retrieved
                out.append(row)
        (a.output_root / f"{split}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
        print(split, len(out))

if __name__ == "__main__": main()
