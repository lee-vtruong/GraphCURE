from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from graphcure.data import build_dataset, collate_manifest
from graphcure.model import GraphCURE, GraphCUREConfig


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
    with torch.inference_mode():
        for batch in loader:
            text = batch["text_embedding"].to(device, non_blocking=True)
            image = batch["image_embedding"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            logits = model(text, image, metadata)["verdict_logits"]
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
    }
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "test_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
