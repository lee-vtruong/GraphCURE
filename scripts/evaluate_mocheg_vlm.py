"""Small Qwen2.5-VL smoke evaluator for MOCHEG retrieved evidence."""
from __future__ import annotations
import argparse, json, re, csv
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROMPT = """You are a fact-checking verifier. Given a claim and retrieved evidence, output ONLY JSON with keys label and rationale. label must be exactly supported, refuted, or nei.\nClaim: {claim}\nEvidence: {evidence}"""

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--retrieval",type=Path,required=True); p.add_argument("--raw-csv",type=Path,required=True); p.add_argument("--model",default="Qwen/Qwen2.5-VL-7B-Instruct"); p.add_argument("--device",default="cuda"); p.add_argument("--limit",type=int,default=100); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); dev=a.device if a.device=="cpu" or torch.cuda.is_available() else "cpu"
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()][:a.limit]; retr={json.loads(x)["id"]:json.loads(x) for x in a.retrieval.read_text().splitlines() if x.strip()}; docs={}
    with a.raw_csv.open(encoding="utf-8-sig",errors="replace",newline="") as f:
        for r in csv.DictReader(f): docs.setdefault(r["evidence_id"],r.get("Evidence","").replace("<p>"," ").replace("</p>"," ").strip())
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,torch_dtype="auto",device_map="auto"); proc=AutoProcessor.from_pretrained(a.model); out=[]
    for row in rows:
        rr=retr[row["id"]]; evidence=" ".join(docs.get(x,"") for x in rr.get("retrieved_evidence_ids",[])[:1]); text=PROMPT.format(claim=row.get("claim",""),evidence=evidence); inputs=proc(text=text,return_tensors="pt").to(model.device); generated=model.generate(**inputs,max_new_tokens=128); decoded=proc.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]; match=re.search(r"\{.*\}",decoded,re.S)
        try: parsed=json.loads(match.group(0)) if match else {"label":"nei","rationale":decoded}
        except json.JSONDecodeError: parsed={"label":"nei","rationale":decoded}
        out.append({"id":row["id"],"gold":row.get("label_name"),"prediction":parsed,"raw":decoded})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in out)+"\n"); print("saved",a.output,len(out))
if __name__=="__main__": main()
