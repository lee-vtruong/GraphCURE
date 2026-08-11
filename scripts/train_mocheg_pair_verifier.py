"""Supervised claim-evidence three-way verifier."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

class Pairs(Dataset):
    def __init__(self,path,tokenizer,max_len=256):
        self.rows=[]; self.tok=tokenizer; self.max_len=max_len
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            r=json.loads(line); ev=r.get('evidence_texts',[])
            if not ev: ev=['']
            for e in ev[:5]: self.rows.append((r.get('claim',''),e,int(r['label'])))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        c,e,y=self.rows[i]; x=self.tok(c,e,truncation=True,padding='max_length',max_length=self.max_len,return_tensors='pt'); return {k:v.squeeze(0) for k,v in x.items()}|{'labels':torch.tensor(y)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest-root',type=Path,default=Path('data/processed/mocheg_gold_manifest')); p.add_argument('--model',default='microsoft/deberta-v3-base'); p.add_argument('--out',type=Path,default=Path('outputs/mocheg_pair_verifier')); p.add_argument('--epochs',type=int,default=3); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--device',default='cuda'); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(a.model); model=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=3).to(a.device); tr=Pairs(a.manifest_root/'train.jsonl',tok); va=Pairs(a.manifest_root/'val.jsonl',tok); te=Pairs(a.manifest_root/'test.jsonl',tok); counts=torch.bincount(torch.tensor([x[2] for x in tr.rows]),minlength=3).float(); class_w=(counts.sum()/(3*counts)).to(a.device); loader=DataLoader(tr,batch_size=a.batch_size,shuffle=True); opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01); sch=get_linear_schedule_with_warmup(opt,int(len(loader)*.1),len(loader)*a.epochs); best=-1
    for ep in range(1,a.epochs+1):
        model.train()
        for b in loader:
            b={k:v.to(a.device) for k,v in b.items()}; opt.zero_grad(); labels=b.pop('labels'); logits=model(**b).logits; loss=torch.nn.functional.cross_entropy(logits.float(),labels,weight=class_w.float()); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
        model.eval(); ys=[];ps=[]
        with torch.no_grad():
            for b in DataLoader(va,batch_size=a.batch_size):
                y=b.pop('labels').tolist(); z=model(**{k:v.to(a.device) for k,v in b.items()}).logits.argmax(-1).cpu().tolist(); ys+=y;ps+=z
        f=f1_score(ys,ps,average='macro'); print(json.dumps({'epoch':ep,'val_macro_f1':f,'val_accuracy':accuracy_score(ys,ps)}))
        if f>best: best=f; model.save_pretrained(a.out); tok.save_pretrained(a.out)
    model=AutoModelForSequenceClassification.from_pretrained(a.out).to(a.device).eval(); ys=[];ps=[]
    with torch.no_grad():
        for b in DataLoader(te,batch_size=a.batch_size):
            y=b.pop('labels').tolist(); z=model(**{k:v.to(a.device) for k,v in b.items()}).logits.argmax(-1).cpu().tolist(); ys+=y;ps+=z
    out={'samples':len(ys),'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro'),'confusion_matrix':confusion_matrix(ys,ps).tolist()}; (a.out/'test_metrics.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
