"""Train a claim-anchored verifier with gold/retrieved evidence consistency."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scripts.train_mocheg_gated import GatedVerifier, load

def main():
    p=argparse.ArgumentParser(); p.add_argument('--gold-root',type=Path,default=Path('data/processed/mocheg_gold_embeddings')); p.add_argument('--retrieved-root',type=Path,default=Path('data/processed/mocheg_relation_embeddings')); p.add_argument('--out',type=Path,default=Path('outputs/mocheg_consistency')); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--device',default='cuda'); p.add_argument('--consistency-weight',type=float,default=.5); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); a.out.mkdir(parents=True,exist_ok=True)
    gold={s:load(a.gold_root/f'{s}.pt') for s in ('train','val','test')}; ret={s:load(a.retrieved_root/f'{s}.pt') for s in ('train','val','test')}
    model=GatedVerifier(d=gold['train'][0].shape[1]).to(dev); counts=torch.bincount(gold['train'][3],minlength=3).float(); ce=nn.CrossEntropyLoss(weight=(counts.sum()/(3*counts)).to(dev)); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-2)
    def run(split,train=False):
        ds=TensorDataset(*(gold[split]+ret[split])); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=train); ys=[];ps=[]
        for batch in loader:
            gc,ge,gm,y,rc,re,rm,_=batch; gc,ge,gm,y,rc,re,rm=[x.to(dev) for x in (gc,ge,gm,y,rc,re,rm)]
            if train:
                opt.zero_grad(); lg,_=model(gc,ge,gm); lr,_=model(rc,re,rm); loss=ce(lr,y)+a.consistency_weight*nn.functional.kl_div(lr.log_softmax(-1),lg.softmax(-1),reduction='batchmean'); loss.backward(); opt.step()
            else:
                with torch.no_grad(): lg,_=model(rc,re,rm); ys+=y.cpu().tolist(); ps+=lg.argmax(-1).cpu().tolist()
        return f1_score(ys,ps,average='macro') if not train else 0, (ys,ps)
    best=-1
    for ep in range(1,a.epochs+1):
        model.train(); run('train',True); model.eval(); f,(ys,ps)=run('val'); print(json.dumps({'epoch':ep,'val_macro_f1':f,'val_accuracy':accuracy_score(ys,ps)}))
        if f>best: best=f; torch.save(model.state_dict(),a.out/'best.pt')
    model.load_state_dict(torch.load(a.out/'best.pt',map_location=dev,weights_only=True)); model.eval(); _,(ys,ps)=run('test'); out={'samples':len(ys),'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro'),'confusion_matrix':confusion_matrix(ys,ps).tolist(),'consistency_weight':a.consistency_weight}; (a.out/'test_metrics.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
