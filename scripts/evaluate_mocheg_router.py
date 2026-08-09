"""Select an open/closed routing threshold on validation, then lock on test."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from graphcure.data import build_dataset, collate_manifest
from graphcure.model import GraphCURE, GraphCUREConfig
from scripts.train import model_forward

def logits(config, checkpoint, split, device):
    cfg = yaml.safe_load(Path(config).read_text()); ds = build_dataset(cfg["data"], split)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], collate_fn=collate_manifest)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False); model = GraphCURE(GraphCUREConfig(**ck["config"])); model.load_state_dict(ck["model"]); model.to(device).eval()
    out, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            moved = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out.append(model_forward(model, moved)["verdict_logits"].cpu().numpy()); labels.extend(batch["label"].tolist())
    return np.concatenate(out), np.array(labels)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--claim-config", required=True); p.add_argument("--claim-checkpoint", required=True); p.add_argument("--open-config", required=True); p.add_argument("--open-checkpoint", required=True); p.add_argument("--retrieval-root", type=Path, required=True); p.add_argument("--device", default="cuda"); p.add_argument("--output", type=Path, default=Path("outputs/mocheg_router.json")); a = p.parse_args()
    dev = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    result = {}
    for split in ("val", "test"):
        claim, y = logits(a.claim_config, a.claim_checkpoint, split, dev); open_, y2 = logits(a.open_config, a.open_checkpoint, split, dev)
        assert np.array_equal(y, y2)
        conf = np.array([json.loads(x)["retrieval_confidence"] for x in (a.retrieval_root / f"{split}.jsonl").read_text().splitlines() if x])
        result[split] = {"y": y.tolist(), "claim": claim.tolist(), "open": open_.tolist(), "confidence": conf.tolist()}
    candidates = np.arange(0.0, 1.001, 0.01); best = None
    for t in candidates:
        r = result["val"]; pred = np.where(r["confidence"] >= t, np.argmax(r["open"], 1), np.argmax(r["claim"], 1)); f = f1_score(r["y"], pred, average="macro"); cov = float(np.mean(r["confidence"] >= t))
        item = (f, -cov, float(t))
        if best is None or item > best[0]: best = (item, {"threshold": float(t), "val_macro_f1": float(f), "val_accuracy": float(accuracy_score(r["y"], pred)), "val_open_coverage": cov})
    threshold = best[1]["threshold"]; r = result["test"]; confidence = np.asarray(r["confidence"]); pred = np.where(confidence >= threshold, np.argmax(r["open"], 1), np.argmax(r["claim"], 1))
    result["selection"] = best[1]; result["test"] = {"accuracy": float(accuracy_score(r["y"], pred)), "macro_f1": float(f1_score(r["y"], pred, average="macro")), "open_coverage": float(np.mean(confidence >= threshold)), "threshold": threshold}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"selection": result["selection"], "test": result["test"]}, indent=2))

if __name__ == "__main__": main()
