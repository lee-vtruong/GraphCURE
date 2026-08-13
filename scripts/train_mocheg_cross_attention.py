"""Claim-level cross-attention verifier over retrieved top-k evidence.

Each claim is one training example. Evidence paragraphs are jointly encoded
with the claim, then an attention pool selects useful evidence before one
three-way verdict is produced. This avoids pair-level label duplication.
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class Claims(Dataset):
    def __init__(self, manifest, retrieval, raw_root, split, tok, k=5, max_len=192):
        self.tok, self.k, self.max_len = tok, k, max_len
        rows = {json.loads(x)["id"]: json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()}
        docs = {}
        with (raw_root / split / "Corpus2.csv").open(encoding="utf-8-sig", errors="replace", newline="") as f:
            for r in csv.DictReader(f): docs.setdefault(r["evidence_id"], r.get("Evidence", ""))
        self.items=[]
        for line in retrieval.read_text(encoding="utf-8").splitlines():
            r=json.loads(line); base=rows.get(r["id"])
            if not base: continue
            ev=[docs[x] for x in r.get("retrieved_evidence_ids",[])[:k] if docs.get(x)]
            self.items.append((base["claim"], ev or [""], int(base["label"])))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def collate(batch):
    claims=[]; ev=[]; ys=[]; counts=[]
    for c,e,y in batch: claims += [c]*len(e); ev += e; counts.append(len(e)); ys.append(y)
    return claims, ev, torch.tensor(ys), counts

class CrossAttentionVerifier(nn.Module):
    def __init__(self, name, hidden=256, dropout=.2):
        super().__init__(); self.encoder=AutoModel.from_pretrained(name); d=self.encoder.config.hidden_size
        self.proj=nn.Linear(d,hidden); self.score=nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1)); self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Dropout(dropout),nn.Linear(hidden,3))
    def forward(self, input_ids, attention_mask, counts):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0]; h=self.proj(h); outs=[]; pos=0
        for n in counts:
            z=h[pos:pos+n]; a=torch.softmax(self.score(z).squeeze(-1),0); outs.append((z*a[:,None]).sum(0)); pos+=n
        return self.head(torch.stack(outs)), None

def evaluate(model, loader, tok, device, max_len):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for claims,ev,y,counts in loader:
            x=tok(claims,ev,truncation=True,padding=True,max_length=max_len,return_tensors='pt').to(device); z,_=model(x.input_ids,x.attention_mask,counts); ys+=y.tolist(); ps+=z.argmax(-1).cpu().tolist()
    return accuracy_score(ys,ps), f1_score(ys,ps,average='macro'), ys, ps

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest-root',type=Path,default=Path('data/processed/mocheg_manifest_strict')); p.add_argument('--retrieval-root',type=Path,default=Path('outputs/retrieval_mocheg_dense_top50')); p.add_argument('--raw-root',type=Path,default=Path('data/raw/mocheg_dataset/extracted/mocheg')); p.add_argument('--model',default='microsoft/deberta-v3-base'); p.add_argument('--out',type=Path,default=Path('outputs/mocheg_cross_attention')); p.add_argument('--top-k',type=int,default=5); p.add_argument('--epochs',type=int,default=3); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--max-len',type=int,default=192); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); a.out.mkdir(parents=True,exist_ok=True); tok=AutoTokenizer.from_pretrained(a.model)
    ds={s:Claims(a.manifest_root/f'{s}.jsonl',a.retrieval_root/f'{s}.jsonl',a.raw_root,s,tok,a.top_k,a.max_len) for s in ('train','val','test')}; loaders={s:DataLoader(ds[s],batch_size=a.batch_size,shuffle=s=='train',collate_fn=collate) for s in ds}; counts=torch.bincount(torch.tensor([x[2] for x in ds['train'].items]),minlength=3).float(); model=CrossAttentionVerifier(a.model).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01); loss_fn=nn.CrossEntropyLoss(weight=(counts.sum()/(3*counts)).to(dev)); sch=get_linear_schedule_with_warmup(opt,int(len(loaders['train'])*.1),len(loaders['train'])*a.epochs); best=-1
    for ep in range(1,a.epochs+1):
        model.train()
        for claims,ev,y,counts2 in loaders['train']:
            x=tok(claims,ev,truncation=True,padding=True,max_length=a.max_len,return_tensors='pt').to(dev); y=y.to(dev); opt.zero_grad(); z,_=model(x.input_ids,x.attention_mask,counts2); loss=loss_fn(z.float(),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
        acc,f,_,_=evaluate(model,loaders['val'],tok,dev,a.max_len); print(json.dumps({'epoch':ep,'val_accuracy':acc,'val_macro_f1':f}))
        if f>best: best=f; torch.save(model.state_dict(),a.out/'best.pt')
    model.load_state_dict(torch.load(a.out/'best.pt',map_location=dev,weights_only=True)); acc,f,ys,ps=evaluate(model,loaders['test'],tok,dev,a.max_len); out={'samples':len(ys),'top_k':a.top_k,'accuracy':acc,'macro_f1':f,'confusion_matrix':confusion_matrix(ys,ps).tolist()}; (a.out/'test_metrics.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
