"""Build an oracle-evidence MOCHEG manifest for a diagnostic upper bound.

Only article-level qrels marked relevant are exposed as evidence.  This is
*not* a deployable setting: it measures the verifier when retrieval is perfect.
"""
from __future__ import annotations

import argparse, csv, json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def build(split: str, manifest_root: Path, raw_root: Path, output_root: Path) -> dict:
    source = [json.loads(x) for x in (manifest_root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if x]
    corpus = read_csv(raw_root / split / "Corpus2.csv")
    docs = {r.get("evidence_id", "").strip(): r.get("Evidence", "") for r in corpus}
    qrels = defaultdict(list)
    for r in read_csv(raw_root / split / "text_evidence_qrels_article_level.csv"):
        if r.get("RELEVANCY", "0").strip() == "1":
            cid, eid = r.get("TOPIC", "").strip(), r.get("evidence_id", "").strip()
            if cid and eid and eid not in qrels[cid]: qrels[cid].append(eid)
    out = []
    for row in source:
        ids = qrels.get(str(row.get("claim_id", "")), [])
        item = dict(row)
        item["text_evidence_ids"] = ids
        item["evidence_texts"] = [docs[e] for e in ids if docs.get(e)]
        item["gold_evidence_oracle"] = True
        out.append(item)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{split}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
    return {"split": split, "claims": len(out), "claims_with_gold_evidence": sum(bool(x["evidence_texts"]) for x in out), "evidence_pairs": sum(len(x["text_evidence_ids"]) for x in out)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-root", type=Path, default=Path("data/processed/mocheg_manifest_strict"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    p.add_argument("--output-root", type=Path, default=Path("data/processed/mocheg_gold_manifest"))
    a = p.parse_args()
    summary = [build(s, a.manifest_root, a.raw_root, a.output_root) for s in ("train", "val", "test")]
    (a.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
