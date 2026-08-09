"""Evaluate open/closed routing at fixed validation-selected budgets."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scripts.evaluate_mocheg_router import logits
import torch

def load_conf(path, split):
    return np.array([json.loads(x)["retrieval_confidence"] for x in (path / f"{split}.jsonl").read_text().splitlines() if x])

def main():
    p = argparse.ArgumentParser(); p.add_argument("--claim-config", required=True); p.add_argument("--claim-checkpoint", required=True); p.add_argument("--open-config", required=True); p.add_argument("--open-checkpoint", required=True); p.add_argument("--retrieval-root", type=Path, required=True); p.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.25, 0.50]); p.add_argument("--device", default="cuda"); p.add_argument("--output", type=Path, default=Path("outputs/mocheg_budget_router.json")); a = p.parse_args(); dev = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    cv, y = logits(a.claim_config, a.claim_checkpoint, "val", dev); ov, y2 = logits(a.open_config, a.open_checkpoint, "val", dev); ct = load_conf(a.retrieval_root, "val"); assert np.array_equal(y, y2)
    ctest, yt = logits(a.claim_config, a.claim_checkpoint, "test", dev); otest, yt2 = logits(a.open_config, a.open_checkpoint, "test", dev); tt = load_conf(a.retrieval_root, "test"); assert np.array_equal(yt, yt2)
    rows = []
    for budget in a.budgets:
        threshold = float(np.quantile(ct, 1.0 - budget)); val_open = ct >= threshold; test_open = tt >= threshold
        vp = np.where(val_open, np.argmax(ov, 1), np.argmax(cv, 1)); tp = np.where(test_open, np.argmax(otest, 1), np.argmax(ctest, 1))
        rows.append({"budget": budget, "threshold": threshold, "val_open_coverage": float(val_open.mean()), "val_macro_f1": float(f1_score(y, vp, average="macro")), "test_open_coverage": float(test_open.mean()), "test_accuracy": float(accuracy_score(yt, tp)), "test_macro_f1": float(f1_score(yt, tp, average="macro"))})
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(rows, indent=2) + "\n"); print(json.dumps(rows, indent=2))

if __name__ == "__main__": main()
