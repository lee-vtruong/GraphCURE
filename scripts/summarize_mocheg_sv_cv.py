"""Summarize GraphCURE-SV train-only cross-validation runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-template",default="outputs/mocheg_sv_cv/fold_{fold}")
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--baseline-macro-f1",type=float)
    parser.add_argument("--minimum-delta",type=float,default=.015)
    parser.add_argument("--output",type=Path,default=Path("outputs/mocheg_sv_cv_summary.json"))
    args=parser.parse_args(); rows=[]
    for fold in range(args.folds):
        path=Path(args.run_template.format(fold=fold))/"summary.json"
        result=json.loads(path.read_text(encoding="utf-8"))
        if result.get("test_split_used") or result.get("protocol")!="train_only_cv":
            raise ValueError(f"non-CV result rejected: {path}")
        rows.append({"fold":fold,"accuracy":result["final"]["accuracy"],"macro_f1":result["final"]["macro_f1"],"path":str(path)})
    f1=np.asarray([row["macro_f1"] for row in rows]); accuracy=np.asarray([row["accuracy"] for row in rows])
    delta=None if args.baseline_macro_f1 is None else float(f1.mean()-args.baseline_macro_f1)
    payload={"protocol":"train_only_cv","folds":args.folds,"per_fold":rows,"aggregate":{"accuracy_mean":float(accuracy.mean()),"accuracy_std":float(accuracy.std()),"macro_f1_mean":float(f1.mean()),"macro_f1_std":float(f1.std())},"baseline_macro_f1":args.baseline_macro_f1,"delta_vs_baseline":delta,"promotion_gate":{"minimum_delta":args.minimum_delta,"mean_delta_passed":delta is not None and delta>=args.minimum_delta,"stability_passed":float(f1.std())<=.02,"passed":delta is not None and delta>=args.minimum_delta and float(f1.std())<=.02},"test_split_used":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
