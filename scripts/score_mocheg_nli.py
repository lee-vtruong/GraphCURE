"""Score retrieved MOCHEG evidence with a three-way NLI cross-encoder."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from sentence_transformers import CrossEncoder

def main():
    p=argparse.ArgumentParser(); p.add_argument("--retrieval-root",type=Path,required=True); p.add_argument("--manifest-root",type=Path,required=True); p.add_argument("--raw-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--model",default="cross-encoder/nli-deberta-v3-base"); p.add_argument("--device",default="cuda"); a=p.parse_args(); a.output_root.mkdir(parents=True,exist_ok=True); model=CrossEncoder(a.model,device=a.device)
    labels = {str(v).lower(): int(k) for k, v in model.model.config.id2label.items()}
    support_idx = next((i for name, i in labels.items() if "entail" in name), 0)
    contra_idx = next((i for name, i in labels.items() if "contrad" in name), 2)
    neutral_idx = next((i for name, i in labels.items() if "neutral" in name), 1)
    for split in ("train","val","test"):
        docs={}
        with (a.raw_root/split/"Corpus2.csv").open(encoding="utf-8-sig",errors="replace",newline="") as f:
            for r in csv.DictReader(f): docs.setdefault(r["evidence_id"],r.get("Evidence","").replace("<p>"," ").replace("</p>"," ").strip())
        claims={json.loads(x)["id"]:json.loads(x) for x in (a.manifest_root/f"{split}.jsonl").read_text().splitlines() if x.strip()}; out=[]
        for line in (a.retrieval_root/f"{split}.jsonl").read_text().splitlines():
            r=json.loads(line); claim=claims[r["id"]]["claim"]; ids=[x for x in r.get("retrieved_evidence_ids",[]) if docs.get(x)]; scores=model.predict([(claim,docs[x]) for x in ids],apply_softmax=True,show_progress_bar=False) if ids else np.zeros((0,3));
            mean=scores.mean(0) if len(scores) else np.zeros(3); r["nli_support"]=float(mean[support_idx]); r["nli_neutral"]=float(mean[neutral_idx]); r["nli_contradiction"]=float(mean[contra_idx]); r["nli_margin"]=float(mean[support_idx]-mean[contra_idx]); out.append(r)
        (a.output_root/f"{split}.jsonl").write_text("\n".join(json.dumps(x) for x in out)+"\n")
        print(split,len(out))
if __name__=="__main__": main()
