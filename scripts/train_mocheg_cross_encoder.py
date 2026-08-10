"""Fine-tune a claim-evidence cross-encoder with MOCHEG qrels."""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--raw-csv",type=Path,required=True); p.add_argument("--qrels",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--base-model",default="cross-encoder/ms-marco-MiniLM-L-6-v2"); p.add_argument("--device",default="cuda"); p.add_argument("--epochs",type=int,default=2); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--seed",type=int,default=42); a=p.parse_args(); random.seed(a.seed)
    claims={json.loads(x)["claim_id"]:json.loads(x).get("claim","") for x in a.manifest.read_text().splitlines() if x.strip()}; docs={}
    with a.raw_csv.open(encoding="utf-8-sig",errors="replace",newline="") as f:
        for r in csv.DictReader(f): docs.setdefault(r["evidence_id"],r.get("Evidence","").replace("<p>"," ").replace("</p>"," ").strip())
    positives={}
    with a.qrels.open(encoding="utf-8-sig",errors="replace",newline="") as f:
        for r in csv.DictReader(f):
            if r.get("RELEVANCY","0").strip()=="1": positives.setdefault(r["TOPIC"].strip(),set()).add(r["evidence_id"].strip())
    examples=[]; all_ids=list(docs)
    for cid, claim in claims.items():
        pos=positives.get(cid,set())
        for eid in list(pos)[:4]:
            if claim and docs.get(eid): examples.append(InputExample(texts=[claim,docs[eid]],label=1.0))
        neg=[x for x in all_ids if x not in pos]
        for eid in random.sample(neg,min(4,len(neg))):
            if claim and docs.get(eid): examples.append(InputExample(texts=[claim,docs[eid]],label=0.0))
    model=CrossEncoder(a.base_model,num_labels=1,device=a.device); loader=DataLoader(examples,shuffle=True,batch_size=a.batch_size); warmup=max(1,int(len(loader)*a.epochs*0.1)); model.fit(train_dataloader=loader,epochs=a.epochs,warmup_steps=warmup,show_progress_bar=True); a.output.mkdir(parents=True,exist_ok=True); model.save(str(a.output))
    print("examples",len(examples),"saved",a.output)
if __name__=="__main__": main()
