"""Heuristic subclaim-aware dense retrieval for MOCHEG."""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def split_claim(x):
    parts = [p.strip() for p in re.split(r"\s+(?:and|but|while|because|although)\s+|[;:]", x, flags=re.I) if len(p.strip().split()) >= 5]
    return parts[:4] or [x]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest-root",type=Path,required=True); p.add_argument("--raw-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--model",default="sentence-transformers/all-mpnet-base-v2"); p.add_argument("--top-k",type=int,default=5); p.add_argument("--device",default="cuda"); a=p.parse_args(); a.output_root.mkdir(parents=True,exist_ok=True); model=SentenceTransformer(a.model,device=a.device); summary={}
    for split in ("train","val","test"):
        claims=[json.loads(x) for x in (a.manifest_root/f"{split}.jsonl").read_text().splitlines() if x.strip()]; docs=[]; ids=[]
        with (a.raw_root/split/"Corpus2.csv").open(encoding="utf-8-sig",errors="replace",newline="") as f:
            for r in csv.DictReader(f):
                t=r.get("Evidence","").replace("<p>"," ").replace("</p>"," ").strip()
                if t and r["evidence_id"] not in ids: ids.append(r["evidence_id"]); docs.append(t)
        de=model.encode(docs,normalize_embeddings=True,show_progress_bar=True); out=[]; hit=[]; rr=[]
        for c in claims:
            subs=split_claim(c.get("claim","")); qe=model.encode(subs,normalize_embeddings=True); scores=qe@de.T; chosen={}
            for row in scores:
                for j in np.argsort(-row)[:a.top_k]: chosen[ids[j]]=max(chosen.get(ids[j],-1.0),float(row[j]))
            ranked=sorted(chosen,key=chosen.get,reverse=True)[:a.top_k]; vals=[chosen[x] for x in ranked]; gold=set(c.get("text_evidence_ids",[])); rank=next((i+1 for i,x in enumerate(ranked) if x in gold),None); hit.append(float(rank is not None)); rr.append(1/rank if rank else 0.0); out.append({"id":c["id"],"claim_id":c["claim_id"],"label":c["label"],"retrieved_evidence_ids":ranked,"retrieved_scores":vals,"retrieval_confidence":vals[0] if vals else 0.0,"gold_evidence_ids":sorted(gold),"first_gold_rank":rank,"subclaims":subs})
        (a.output_root/f"{split}.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in out)+"\n"); summary[split]={"claims":len(out),f"recall@{a.top_k}":float(np.mean(hit)),"mrr":float(np.mean(rr))}; print(split,summary[split])
    (a.output_root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__": main()
