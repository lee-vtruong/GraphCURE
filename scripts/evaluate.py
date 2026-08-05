from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from graphcure.data import build_dataset, collate_manifest
from graphcure.model import GraphCURE, GraphCUREConfig
from scripts.train import model_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(cfg["data"], "test")
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_manifest,
        pin_memory=device.type == "cuda",
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GraphCURE(GraphCUREConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    labels: list[int] = []
    predictions: list[int] = []
    mix_gates: list[np.ndarray] = []
    cf_changed, cf_unchanged, cf_verdict_flip = [], [], []
    verdict_pair_order, constraint_pair_order = [], []
    with torch.inference_mode():
        for batch in loader:
            text = batch["text_embedding"].to(device, non_blocking=True)
            image = batch["image_embedding"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            moved = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            output_batch = model_forward(model, moved)
            logits = output_batch["verdict_logits"]
            if model.cfg.architecture == "multi_adaptive_graph":
                mix_gates.append(output_batch["node_mix_gates"].cpu().numpy())
            if "cf_text_embedding" in moved:
                cf_output = model_forward(model, moved, prefix="cf_")
                factual_state = output_batch["constraint_prob"].argmax(-1)
                paired_state = cf_output["constraint_prob"].argmax(-1)
                changed = moved["changed_mask"].bool()
                cf_changed.extend((factual_state[changed] != paired_state[changed]).cpu().tolist())
                cf_unchanged.extend((factual_state[~changed] == paired_state[~changed]).cpu().tolist())
                cf_verdict_flip.extend(
                    (logits.argmax(-1) != cf_output["verdict_logits"].argmax(-1)).cpu().tolist()
                )
                direction = (moved["cf_label"].float() - moved["label"].float()).sign()
                factual_verdict_score = logits[:, 1] - logits[:, 0]
                paired_verdict_score = (
                    cf_output["verdict_logits"][:, 1] - cf_output["verdict_logits"][:, 0]
                )
                verdict_pair_order.extend(
                    (direction * (paired_verdict_score - factual_verdict_score) > 0).cpu().tolist()
                )
                factual_violated = output_batch["constraint_prob"][..., 1]
                paired_violated = cf_output["constraint_prob"][..., 1]
                selected_delta = (paired_violated - factual_violated)[changed]
                selected_direction = direction[:, None].expand_as(changed)[changed]
                constraint_pair_order.extend(
                    (selected_direction * selected_delta > 0).cpu().tolist()
                )
            labels.extend(batch["label"].tolist())
            predictions.extend(logits.argmax(-1).cpu().tolist())

    metrics = {
        "samples": len(labels),
        "accuracy": float(np.mean(np.asarray(labels) == np.asarray(predictions))),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels, predictions, output_dict=True, zero_division=0
        ),
        "provenance": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip() or "unknown",
            "config_file": args.config,
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "checkpoint": args.checkpoint,
            "architecture": checkpoint.get("architecture", model.cfg.architecture),
            "seed": checkpoint.get("seed"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }
    if mix_gates:
        gate_array = np.concatenate(mix_gates, axis=0)
        metrics["node_mix_gate_mean"] = gate_array.mean(axis=0).tolist()
        metrics["node_mix_gate_std"] = gate_array.std(axis=0).tolist()
    if cf_changed:
        metrics["counterfactual"] = {
            "descendant_intervention_accuracy": float(np.mean(cf_changed)),
            "non_descendant_invariance": float(np.mean(cf_unchanged)),
            "counterfactual_verdict_consistency": float(np.mean(cf_verdict_flip)),
            "verdict_pair_order_accuracy": float(np.mean(verdict_pair_order)),
            "constraint_pair_order_accuracy": float(np.mean(constraint_pair_order)),
            "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip",
        }
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "test_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
