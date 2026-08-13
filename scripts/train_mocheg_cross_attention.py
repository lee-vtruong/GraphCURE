"""Claim-level cross-attention verifier over top-k retrieved evidence."""
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
        docs={}
        with (raw_root/split/'Corpus2.csv').open(encoding='utf-8-sig',errors='replace',newline='') as f:
            for r in csv.DictReader(f): docs.setdefault(r['evidence_id'], r.get('Evidence',''))
        mr={json.loads(x)['id']:json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()}
        self.rows=[]; self.tok=tok; self.k=k; self.max_len=max_len
        for line in retrieval.read_text(encoding='utf-8').splitlines():
            r=json.loads(line); base=mr[r['id']]; ids=r.get('retrieved_evidence_ids',[])[:k]
            ev=[docs[x] for x in ids if docs.get(x)]
            self.rows.append((base['claim'], ev or [''], int(base['label'])))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        c,ev,y=self.rows[i]
        ct=self.tok(c,truncation=True,padding='max_length',max_length=self.max_len,return_tensors='pt')
        et=[self.tok(x,truncation=True,padding='max_length',max_length=self.max_len,return_tensors='pt') for x in ev]
        ids=torch.cat([x['input_ids'] for x in et],0); mask=torch.cat([x['attention_mask'] for x in et],0)
        return {'claim_input_ids':ct['input_ids'].squeeze(0),'claim_attention_mask':ct['attention_mask'].squeeze(0),'evidence_input_ids':ids,'evidence_attention_mask':mask,'labels':torch.tensor(y)}

def collate(xs):
    out={}
    for k in xs[0]: out[k]=torch.stack([x[k] for x in xs])
    return out

class CrossAttnVerifier(nn.Module):
    def __init__(self,name,labels=3,dropout=.2):
        super().__init__(); self.enc=AutoModel.from_pretrained(name); d=self.enc.config.hidden_size
        self.attn=nn.MultiheadAttention(d,8,dropout=dropout,batch_first=True); self.norm=nn.LayerNorm(d); self.head=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,labels))
    def forward(self,claim_input_ids,claim_attention_mask,evidence_input_ids,evidence_attention_mask):
        b,k,l=evidence_input_ids.shape; c=self.enc(input_ids=claim_input_ids,attention_mask=claim_attention_mask).last_hidden_state[:,0:1].float()
        e=self.enc(input_ids=evidence_input_ids.reshape(b*k,l),attention_mask=evidence_attention_mask.reshape(b*k,l)).last_hidden_state[:,0].reshape(b,k,-1).float()
        q,_=self.attn(c,e,e); h=self.norm(c.squeeze(1)+q.squeeze(1)); return self.head(h)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest-root',type=Path,default=Path('data/processed/mocheg_manifest_strict')); p.add_argument('--retrieval-root',type=Path,default=Path('outputs/retrieval_mocheg_dense_top50')); p.add_argument('--raw-root',type=Path,default=Path('data/raw/mocheg_dataset/extracted/mocheg')); p.add_argument('--model',default='microsoft/deberta-v3-base'); p.add_argument('--out',type=Path,default=Path('outputs/mocheg_cross_attention')); p.add_argument('--top-k',type=int,default=5); p.add_argument('--epochs',type=int,default=3); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--max-len',type=int,default=192); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); a.out.mkdir(parents=True,exist_ok=True); tok=AutoTokenizer.from_pretrained(a.model)
    ds={s:Claims(a.manifest_root/f'{s}.jsonl',a.retrieval_root/f'{s}.jsonl',a.raw_root,s,tok,a.top_k,a.max_len) for s in ('train','val','test')}; model=CrossAttnVerifier(a.model).to(dev); counts=torch.bincount(torch.tensor([r[2] for r in ds['train'].rows]),minlength=3).float(); loss_fn=nn.CrossEntropyLoss(weight=(counts.sum()/(3*counts)).to(dev)); opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01); loader=DataLoader(ds['train'],batch_size=a.batch_size,shuffle=True,collate_fn=collate); best=-1
    for ep in range(1,a.epochs+1):
        model.train(); total=0
        for b in loader:
            b={k:v.to(dev) for k,v in b.items()}; y=b.pop('labels'); opt.zero_grad(); z=model(**b); loss=loss_fn(z.float(),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()
        model.eval(); ys=[];ps=[]
        with torch.no_grad():
            for b in DataLoader(ds['val'],batch_size=a.batch_size,collate_fn=collate):
                y=b.pop('labels'); z=model(**{k:v.to(dev) for k,v in b.items()}); ys+=y.tolist(); ps+=z.argmax(-1).cpu().tolist()
        f=f1_score(ys,ps,average='macro'); print(json.dumps({'epoch':ep,'loss':total/max(1,len(loader)),'val_macro_f1':f,'val_accuracy':accuracy_score(ys,ps)}))
        if f>best: best=f; torch.save(model.state_dict(),a.out/'best.pt')
    model.load_state_dict(torch.load(a.out/'best.pt',map_location=dev,weights_only=True)); model.eval(); ys=[];ps=[]
    with torch.no_grad():
        for b in DataLoader(ds['test'],batch_size=a.batch_size,collate_fn=collate):
            y=b.pop('labels'); z=model(**{k:v.to(dev) for k,v in b.items()}); ys+=y.tolist(); ps+=z.argmax(-1).cpu().tolist()
    out={'samples':len(ys),'top_k':a.top_k,'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro'),'confusion_matrix':confusion_matrix(ys,ps).tolist()}; (a.out/'test_metrics.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
