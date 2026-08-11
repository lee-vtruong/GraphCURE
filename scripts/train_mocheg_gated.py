"""Train claim-anchored gated evidence verifier on cached MOCHEG embeddings."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class GatedVerifier(nn.Module):
    def __init__(self, d=768, meta=4, hidden=512, dropout=.2):
        super().__init__()
        self.claim = nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.evidence = nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden*2+meta, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.head = nn.Sequential(nn.Linear(hidden+meta, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
    def forward(self, c, e, m):
        hc, he = self.claim(c), self.evidence(e)
        g = torch.sigmoid(self.gate(torch.cat([hc, he, m], -1)))
        return self.head(torch.cat([hc + g * he, m], -1)), g

def load(path):
    p=torch.load(path, map_location="cpu", weights_only=True)
    m=p.get("nli_metadata", torch.zeros(len(p["labels"]),4)).float()
    return p["claim_embeddings"].float(), p["evidence_embeddings"].float(), m[:, :4], p["labels"].long()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=Path("data/processed/mocheg_gold_embeddings")); ap.add_argument("--out", type=Path, default=Path("outputs/mocheg_gated_regularized")); ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--device",default="cuda"); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--gate-penalty",type=float,default=.05); ap.add_argument("--evidence-dropout",type=float,default=.30); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device(a.device if torch.cuda.is_available() else "cpu"); a.out.mkdir(parents=True,exist_ok=True)
    tr=load(a.root/"train.pt"); va=load(a.root/"val.pt"); te=load(a.root/"test.pt")
    model=GatedVerifier(d=tr[0].shape[1]).to(dev); counts=torch.bincount(tr[3],minlength=3).float(); weights=(counts.sum()/(3*counts)).to(dev); loss_fn=nn.CrossEntropyLoss(weight=weights); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-2); best=-1
    def loader(x, shuffle): return DataLoader(TensorDataset(*x),batch_size=a.batch_size,shuffle=shuffle)
    for ep in range(1,a.epochs+1):
        model.train(); total=0
        for c,e,m,y in loader(tr,True):
            c,e,m,y=c.to(dev),e.to(dev),m.to(dev),y.to(dev); opt.zero_grad()
            if a.evidence_dropout > 0:
                e = e * (torch.rand(e.size(0), 1, device=dev) >= a.evidence_dropout)
            logits,g=model(c,e,m); loss=loss_fn(logits,y) + a.gate_penalty*g.mean(); loss.backward(); opt.step(); total+=loss.item()*len(y)
        model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for c,e,m,y in loader(va,False): ys += y.tolist(); ps += model(c.to(dev),e.to(dev),m.to(dev))[0].argmax(-1).cpu().tolist()
        f=f1_score(ys,ps,average="macro"); print(json.dumps({"epoch":ep,"loss":total/len(tr[3]),"val_macro_f1":f,"val_accuracy":accuracy_score(ys,ps)}))
        if f>best: best=f; torch.save(model.state_dict(),a.out/"best.pt")
    model.load_state_dict(torch.load(a.out/"best.pt",map_location=dev,weights_only=True)); model.eval(); ys=[];ps=[]; gates=[]
    with torch.no_grad():
        for c,e,m,y in loader(te,False): q,g=model(c.to(dev),e.to(dev),m.to(dev)); ys+=y.tolist(); ps+=q.argmax(-1).cpu().tolist(); gates+=g.squeeze(-1).cpu().tolist()
    out={"samples":len(ys),"accuracy":accuracy_score(ys,ps),"macro_f1":f1_score(ys,ps,average="macro"),"confusion_matrix":confusion_matrix(ys,ps).tolist(),"gate_mean":float(np.mean(gates)),"gate_std":float(np.std(gates))}; (a.out/"test_metrics.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
