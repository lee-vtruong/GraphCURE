"""Train a claim-level learned aggregator over retrieved NLI evidence.

Features combine retrieval scores, NLI stance, rank-weighted pooling, and
evidence sufficiency statistics. This avoids assigning the claim label to
every individual evidence paragraph.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def rows(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def feat(r, k):
    ns=r.get('nli_scores',[])[:k]; rs=np.asarray(r.get('retrieved_scores',[])[:len(ns)],dtype=np.float32)
    if not ns: return np.zeros(34,dtype=np.float32)
    a=np.asarray([[x['support'],x['contradiction'],x['neutral']] for x in ns],dtype=np.float32)
    if len(rs)!=len(a): rs=np.zeros(len(a),dtype=np.float32)
    w=np.exp(rs-rs.max()); w=w/(w.sum()+1e-8)
    pooled=(a*w[:,None]).sum(0); mean=a.mean(0); mx=a.max(0); sd=a.std(0)
    top=a[0]; suff=np.array([r.get('retrieval_confidence',0.0),len(ns)/max(k,1),float(rs[0] if len(rs) else 0),float(rs.mean() if len(rs) else 0)],dtype=np.float32)
    return np.concatenate([pooled,mean,mx,sd,top,suff]).astype(np.float32)

def make(path,k):
    rr=rows(path); x=np.stack([feat(r,k) for r in rr]); y=np.asarray([int(r['label']) for r in rr],dtype=np.int64); return torch.tensor(x),torch.tensor(y)

class Net(nn.Module):
    def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,3))
    def forward(self,x): return self.net(x)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--top-k',type=int,default=20); p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); a.out.mkdir(parents=True,exist_ok=True)
    tr=make(a.root/'train.jsonl',a.top_k); va=make(a.root/'val.jsonl',a.top_k); te=make(a.root/'test.jsonl',a.top_k); model=Net(tr[0].shape[1]).to(dev); c=torch.bincount(tr[1],minlength=3).float(); loss_fn=nn.CrossEntropyLoss(weight=(c.sum()/(3*c)).to(dev)); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-2); best=-1
    for ep in range(1,a.epochs+1):
        model.train(); idx=torch.randperm(len(tr[1]))
        for start in range(0,len(idx),a.batch_size):
            ix=idx[start:start+a.batch_size]; opt.zero_grad(); z=model(tr[0][ix].to(dev)); loss=loss_fn(z,tr[1][ix].to(dev)); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): pv=model(va[0].to(dev)).argmax(-1).cpu().numpy()
        f=f1_score(va[1].numpy(),pv,average='macro')
        if ep%10==0 or f>best: print(json.dumps({'epoch':ep,'val_macro_f1':float(f),'val_accuracy':float(accuracy_score(va[1],pv))}))
        if f>best: best=f; torch.save(model.state_dict(),a.out/'best.pt')
    model.load_state_dict(torch.load(a.out/'best.pt',map_location=dev,weights_only=True)); model.eval()
    with torch.no_grad(): pred=model(te[0].to(dev)).argmax(-1).cpu().numpy()
    out={'samples':len(pred),'top_k':a.top_k,'accuracy':float(accuracy_score(te[1],pred)),'macro_f1':float(f1_score(te[1],pred,average='macro')),'confusion_matrix':confusion_matrix(te[1],pred).tolist()}; (a.out/'test_metrics.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
