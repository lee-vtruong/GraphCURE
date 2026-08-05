from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from graphcure.data import build_dataset, collate_manifest
from graphcure.losses import total_loss
from graphcure.model import GraphCURE, GraphCUREConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, help="Override config seed")
    parser.add_argument("--architecture", choices=["linear", "mlp", "independent", "fully_connected", "typed_graph",
                                                   "multi_independent", "multi_fully_connected", "multi_typed_graph",
                                                   "multi_adaptive_graph"])
    parser.add_argument("--output-dir", help="Override train.output_dir")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def model_forward(model: GraphCURE, batch: dict, prefix: str = "") -> dict[str, torch.Tensor]:
    optional = {name: batch[f"{prefix}{name}"] for name in
                ("sbert_embeddings", "facenet_embeddings", "places_embeddings", "view_mask",
                 "semantic_image_embedding", "contextual_image_embedding")
                if f"{prefix}{name}" in batch}
    return model(batch[f"{prefix}text_embedding"], batch[f"{prefix}image_embedding"],
                 batch[f"{prefix}metadata"],
                 **optional)


@torch.no_grad()
def evaluate(model: GraphCURE, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    labels, predictions = [], []
    for batch in loader:
        batch = move(batch, device)
        out = model_forward(model, batch)
        labels.extend(batch["label"].cpu().tolist())
        predictions.extend(out["verdict_logits"].argmax(-1).cpu().tolist())
    return {
        "accuracy": float(np.mean(np.asarray(labels) == np.asarray(predictions))),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.architecture:
        cfg["model"]["architecture"] = args.architecture
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    seed_everything(cfg["seed"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_data = build_dataset(cfg["data"], "train")
    val_data = build_dataset(cfg["data"], "val")
    loader_args = dict(
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_manifest,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    sample = train_data[0]
    model_cfg = GraphCUREConfig(
        text_dim=sample["text_embedding"].numel(),
        vision_dim=sample["image_embedding"].numel(),
        metadata_dim=sample["metadata"].numel(),
        sbert_dim=sample.get("sbert_embeddings", torch.empty(0)).numel(),
        facenet_dim=sample.get("facenet_embeddings", torch.empty(0)).numel(),
        places_dim=sample.get("places_embeddings", torch.empty(0)).numel(),
        **{
            k: cfg["model"][k]
            for k in ("hidden_dim", "num_states", "num_labels", "graph_layers", "dropout", "architecture", "edge_dropout")
            if k in cfg["model"]
        },
    )
    model = GraphCURE(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")
    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    stale_epochs = 0
    patience = int(cfg["train"].get("early_stopping_patience", 0))
    weights = cfg["loss"]
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                out = model_forward(model, batch)
                cf_out = None
                if "cf_text_embedding" in batch:
                    cf_out = model_forward(model, batch, prefix="cf_")
                loss, _ = total_loss(out, batch, weights, cf_out)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        metrics = evaluate(model, val_loader, device)
        print(json.dumps({"epoch": epoch + 1, "loss": running / len(train_loader), **metrics}))
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            stale_epochs = 0
            torch.save(
                {"model": model.state_dict(), "config": model_cfg.__dict__, "metrics": metrics,
                 "seed": cfg["seed"], "architecture": model_cfg.architecture},
                output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        if patience and stale_epochs >= patience:
            print(json.dumps({"early_stop": epoch + 1, "patience": patience}))
            break
    (output_dir / "metrics.json").write_text(
        json.dumps({"best_val_macro_f1": best, "seed": cfg["seed"],
                    "architecture": model_cfg.architecture}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
