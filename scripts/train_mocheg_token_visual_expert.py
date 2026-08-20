"""Validation-only token-level claim--image cross-attention screen."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from graphcure.token_visual import (aggregate_pair_logits,
                                    select_candidate_ids, token_visual_loss)
from scripts.run_mocheg_visual_ensemble import aligned_gold_images
from scripts.run_mocheg_visual_retrieval import read_jsonl


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ClaimImageSets(Dataset):
    def __init__(self, manifest: Path, retrieval: Path, image_root: Path,
                 top_k: int, inject_gold: bool, limit: int = 0):
        claims, rows = read_jsonl(manifest), read_jsonl(retrieval)
        retrieval_by_id = {row["id"]: row for row in rows}
        if limit:
            claims = claims[:limit]
        corpus = {path.name: path for path in image_root.iterdir()
                  if path.is_file()}
        corpus_names = set(corpus)
        self.items = []
        for claim in claims:
            row = retrieval_by_id.get(claim["id"])
            if row is None:
                raise ValueError(f"retrieval missing claim {claim['id']}")
            gold = aligned_gold_images(claim, corpus_names)
            retrieved = [str(value) for value in
                         row.get("retrieved_image_ids", [])
                         if str(value) in corpus]
            selected = select_candidate_ids(
                claim["id"], retrieved, gold, top_k, inject_gold)
            if not selected:
                raise ValueError(f"claim {claim['id']} has no image candidates")
            self.items.append({
                "id": claim["id"], "claim": claim.get("claim", ""),
                "label": int(claim["label"]), "names": selected,
                "paths": [corpus[value] for value in selected],
                "relevance": [value in gold for value in selected],
                "ranks": [retrieved.index(value) + 1
                          if value in retrieved else 0 for value in selected],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def make_collate(processor, top_k: int, max_length: int):
    def collate(rows):
        texts, images, mask, relevance, ranks = [], [], [], [], []
        for row in rows:
            count = len(row["paths"])
            paths = row["paths"] + [row["paths"][-1]] * (top_k - count)
            texts.extend([row["claim"]] * top_k)
            images.extend(load_rgb(path) for path in paths)
            mask.append([True] * count + [False] * (top_k - count))
            relevance.append(row["relevance"] + [False] * (top_k - count))
            ranks.append(row["ranks"] + [0] * (top_k - count))
        encoded = processor(
            text=texts, images=images, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt")
        return {
            "encoded": encoded,
            "labels": torch.tensor([row["label"] for row in rows]),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "relevance": torch.tensor(relevance, dtype=torch.bool),
            "ranks": torch.tensor(ranks, dtype=torch.float32),
            "ids": [row["id"] for row in rows],
        }
    return collate


class TokenVisualExpert(nn.Module):
    """Claim tokens attend directly to pretrained image patch tokens."""
    def __init__(self, backbone: nn.Module, text_hidden_size: int,
                 image_hidden_size: int, projection: int, heads: int,
                 dropout: float, freeze_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in backbone.parameters():
                parameter.requires_grad = False
        self.text_projection = nn.Linear(text_hidden_size, projection)
        self.image_projection = nn.Linear(image_hidden_size, projection)
        self.cross_attention = nn.MultiheadAttention(
            projection, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(projection)
        self.head = nn.Sequential(
            nn.Linear(projection * 4, projection), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(projection, 4))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        context = torch.no_grad if self.freeze_backbone else torch.enable_grad
        with context():
            output = self.backbone(**encoded, return_dict=True)
            text = output.text_model_output.last_hidden_state
            image = output.vision_model_output.last_hidden_state
        text = self.text_projection(text.float())
        image = self.image_projection(image.float())
        attended, _ = self.cross_attention(text, image, image,
                                            need_weights=False)
        fused = self.norm(text + attended)
        text_mask = encoded.get("attention_mask")
        if text_mask is None:
            text_pool, fused_pool = text.mean(1), fused.mean(1)
        else:
            weight = text_mask.float().unsqueeze(-1)
            denominator = weight.sum(1).clamp_min(1)
            text_pool = (text * weight).sum(1) / denominator
            fused_pool = (fused * weight).sum(1) / denominator
        return self.head(torch.cat([
            text_pool, fused_pool, torch.abs(text_pool - fused_pool),
            text_pool * fused_pool], dim=-1))


def rank_prior(ranks: torch.Tensor) -> torch.Tensor:
    return torch.where(ranks > 0, -torch.log(ranks.clamp_min(1)),
                       torch.zeros_like(ranks))


def evaluate(model, loader, device, top_k, prior_weight):
    model.eval()
    labels, predictions, rel_y, rel_score = [], [], [], []
    selected_hits, gold_y, gold_prediction = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="validation", leave=False):
            encoded = {key: value.to(device) for key, value
                       in batch["encoded"].items()}
            mask = batch["mask"].to(device)
            pair = model(encoded).reshape(-1, top_k, 4)
            claim, attention = aggregate_pair_logits(
                pair, mask, rank_prior(batch["ranks"].to(device)), prior_weight)
            labels.extend(batch["labels"].tolist())
            predictions.extend(claim.argmax(-1).cpu().tolist())
            relation = (torch.logsumexp(pair[..., :3], -1) - pair[..., 3]).cpu()
            valid = batch["mask"]
            rel_y.extend(batch["relevance"][valid].int().tolist())
            rel_score.extend(relation[valid].tolist())
            for index in range(len(batch["ids"])):
                positions = torch.where(batch["relevance"][index]
                                        & valid[index])[0]
                if len(positions):
                    selected_hits.append(int(
                        int(attention[index].argmax().cpu()) in positions.tolist()))
                    for position in positions:
                        gold_y.append(int(batch["labels"][index]))
                        gold_prediction.append(int(
                            pair[index, position, :3].argmax().cpu()))
    binary = len(set(rel_y)) == 2
    return {
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "selection_hit_at_1": float(np.mean(selected_hits))
        if selected_hits else None,
        "gold_candidate_sets": len(selected_hits),
        "visual_stance_accuracy_gold_candidates": float(
            accuracy_score(gold_y, gold_prediction)) if gold_y else None,
        "visual_stance_macro_f1_gold_candidates": float(
            f1_score(gold_y, gold_prediction, average="macro"))
        if gold_y else None,
        "relevance_auroc": float(roc_auc_score(rel_y, rel_score))
        if binary else None,
        "relevance_average_precision": float(
            average_precision_score(rel_y, rel_score)) if binary else None,
    }


def main():
    from transformers import AutoModel, AutoProcessor
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--train-retrieval", type=Path, required=True)
    parser.add_argument("--val-retrieval", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path(
        "data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_token_visual_seed42_v9"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--projection", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--negative-weight", type=float, default=.25)
    parser.add_argument("--claim-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=.25)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone = AutoModel.from_pretrained(args.model, torch_dtype=dtype)
    text_hidden = int(backbone.config.text_config.hidden_size)
    image_hidden = int(backbone.config.vision_config.hidden_size)
    model = TokenVisualExpert(
        backbone, text_hidden, image_hidden, args.projection, args.heads,
        args.dropout,
        not args.unfreeze_backbone).to(device)
    train = ClaimImageSets(
        args.manifest_root / "train.jsonl", args.train_retrieval,
        args.raw_root / "train" / "images", args.top_k, True,
        args.limit_train)
    val = ClaimImageSets(
        args.manifest_root / "val.jsonl", args.val_retrieval,
        args.raw_root / "val" / "images", args.top_k, False,
        args.limit_val)
    collate = make_collate(processor, args.top_k, args.max_length)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=.01)
    best, bad, history = -math.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = Counter()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            encoded = {key: value.to(device, non_blocking=True)
                       for key, value in batch["encoded"].items()}
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            relevance = batch["relevance"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                pair = model(encoded).reshape(-1, args.top_k, 4)
                claim, _ = aggregate_pair_logits(
                    pair, mask, rank_prior(batch["ranks"].to(device)),
                    args.prior_weight)
                loss, _ = token_visual_loss(
                    pair, claim, labels, relevance, mask, relevance.any(1),
                    args.negative_weight, args.claim_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            totals["loss"] += float(loss)
            totals["steps"] += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
        metrics = evaluate(model, val_loader, device, args.top_k,
                           args.prior_weight)
        row = {"epoch": epoch,
               "train_loss": totals["loss"] / totals["steps"], **metrics}
        history.append(row)
        print(json.dumps(row))
        score = metrics["visual_stance_macro_f1_gold_candidates"] or -math.inf
        if score > best:
            best, bad = score, 0
            head = {key: value.cpu() for key, value in model.state_dict().items()
                    if not key.startswith("backbone.")}
            torch.save({"head": head, "args": vars(args), "metrics": metrics},
                       args.output / "best_head.pt")
            (args.output / "val_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        else:
            bad += 1
            if bad >= args.patience:
                break
    best_row = max(history, key=lambda row:
                   row["visual_stance_macro_f1_gold_candidates"] or -math.inf)
    summary = {
        "model": args.model, "backbone_frozen": not args.unfreeze_backbone,
        "best": best_row, "history": history,
        "success_gate": {
            "stance_macro_f1_at_least_0_50": best >= .50,
            "relevance_auroc_at_least_0_65":
                (best_row["relevance_auroc"] or 0) >= .65,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
