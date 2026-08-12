"""Evaluate open-only NLI aggregation and a validation-calibrated NEI rule."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

def load(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def scores(rows, tau, margin):
    y=[]; p=[]
    for r in rows:
        ns=r.get('nli_scores',[])
        if not ns: pred=2
        else:
            sup=max(x['support'] for x in ns); con=max(x['contradiction'] for x in ns); neu=max(x['neutral'] for x in ns)
            if max(sup,con)<tau or abs(sup-con)<margin: pred=2
            else: pred=0 if sup>con else 1
        y.append(int(r['label'])); p.append(pred)
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),'confusion_matrix':confusion_matrix(y,p).tolist()}
def main():
    a=argparse.ArgumentParser(); a.add_argument('--root',type=Path,required=True); a.add_argument('--output',type=Path,required=True); a=a.parse_args()
    tr=load(a.root/'val.jsonl'); best=None
    for tau in np.arange(.30,.86,.02):
        for margin in np.arange(0,.31,.02):
            m=scores(tr,float(tau),float(margin)); key=m['macro_f1']
            if best is None or key>best[0]: best=(key,{'tau':float(tau),'margin':float(margin),**m})
    test=scores(load(a.root/'test.jsonl'),best[1]['tau'],best[1]['margin']); out={'selection':best[1],'test':test}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
