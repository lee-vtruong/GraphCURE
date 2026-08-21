"""Train a validation-only safe fusion adapter over frozen report experts."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphcure.multimodal_evidence import MultimodalEvidenceHead
from graphcure.report_fusion import SafeReportFusion, fusion_features
from scripts.train_mocheg_cached_verifier import expected_calibration_error
from scripts.train_mocheg_multimodal_verifier import (
    MultimodalEvidenceDataset, collate, validate_cache_pair)
from scripts.train_mocheg_staged_multimodal import model_batch


def build_frozen_expert(checkpoint_path: Path, metadata: dict,
                        device: torch.device) -> tuple[MultimodalEvidenceHead, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    head = MultimodalEvidenceHead(
        claim_dim=int(metadata["claim_dim"]), text_dim=int(metadata["text_dim"]),
        visual_dim=int(metadata["visual_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        visual_attention_mode=checkpoint.get("visual_attention_mode", "learned"),
        visual_prior_temperature=float(checkpoint.get(
            "visual_prior_temperature", 0.5)),
        visual_residual_scale=float(checkpoint.get("visual_residual_scale", .25)),
        visual_expert_mode=checkpoint.get("visual_expert_mode", "stance_product"),
        visual_stance_scale=float(checkpoint.get("visual_stance_scale", 1.0)),
    ).to(device)
    head.load_state_dict(checkpoint["head"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad = False
    return head, checkpoint


@torch.inference_mode()
def extract(head, dataset, batch_size, device):
    result = {key: [] for key in ("labels", "text_logits", "visual_logits",
                                  "features")}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate)
    for batch in tqdm(loader, desc=f"extract {dataset.metadata['split']}"):
        inputs, supervision = model_batch(batch, device)
        output = head(**inputs)
        result["labels"].append(supervision["labels"].cpu())
        result["text_logits"].append(output["text_verdict_logits"].float().cpu())
        result["visual_logits"].append(output["visual_expert_logits"].float().cpu())
        result["features"].append(fusion_features(output).cpu())
    return {key: torch.cat(value) for key, value in result.items()}


@torch.inference_mode()
def metrics(adapter, data, device):
    text = data["text_logits"].to(device)
    visual = data["visual_logits"].to(device)
    features = data["features"].to(device)
    labels = data["labels"].numpy()
    logits, gate = adapter(text, visual, features)
    probabilities = torch.softmax(logits.float(), -1).cpu().numpy()
    prediction = probabilities.argmax(-1)
    text_prediction = data["text_logits"].argmax(-1).numpy()
    visual_prediction = data["visual_logits"].argmax(-1).numpy()
    return {
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(labels, prediction).tolist(),
        "text_accuracy": float(accuracy_score(labels, text_prediction)),
        "text_macro_f1": float(f1_score(labels, text_prediction, average="macro")),
        "visual_accuracy": float(accuracy_score(labels, visual_prediction)),
        "visual_macro_f1": float(f1_score(labels, visual_prediction, average="macro")),
        "gate_mean": float(gate.mean().cpu()),
        "gate_std": float(gate.std().cpu()),
        "help_rate": float(np.mean((text_prediction != labels) & (prediction == labels))),
        "harm_rate": float(np.mean((text_prediction == labels) & (prediction != labels))),
        "ece_10": expected_calibration_error(probabilities, labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path(
        "data/processed/mocheg_visual_report_cache_v10"))
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/mocheg_report_fusion_seed42_v11"))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--gate-cost", type=float, default=.005)
    parser.add_argument("--anchor-kl", type=float, default=.02)
    parser.add_argument("--minimum-delta", type=float, default=.003)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    train = MultimodalEvidenceDataset(args.cache_root / "train.pt")
    val = MultimodalEvidenceDataset(args.cache_root / "val.pt")
    validate_cache_pair(train, val, expected_train_gold_injection=True)
    head, checkpoint = build_frozen_expert(
        args.expert_checkpoint, train.metadata, device)
    train_data = extract(head, train, args.batch_size, device)
    val_data = extract(head, val, args.batch_size, device)
    feature_dim = int(train_data["features"].shape[-1])
    adapter = SafeReportFusion(feature_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    counts = torch.bincount(train_data["labels"], minlength=3).float()
    class_weights = (counts.sum() / (3 * counts.clamp_min(1))).to(device)
    indices = torch.arange(len(train_data["labels"]))
    anchor = {
        "accuracy": float(accuracy_score(
            val_data["labels"], val_data["text_logits"].argmax(-1))),
        "macro_f1": float(f1_score(
            val_data["labels"], val_data["text_logits"].argmax(-1),
            average="macro")),
    }
    best_score, best_state, stale, history = anchor["macro_f1"], None, 0, []
    for epoch in range(1, args.epochs + 1):
        adapter.train(); order = indices[torch.randperm(len(indices))]; total = 0.0
        for start in range(0, len(order), args.batch_size):
            chosen = order[start:start + args.batch_size]
            text = train_data["text_logits"][chosen].to(device)
            visual = train_data["visual_logits"][chosen].to(device)
            features = train_data["features"][chosen].to(device)
            labels = train_data["labels"][chosen].to(device)
            logits, gate = adapter(text, visual, features)
            teacher = torch.softmax(text, -1)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss = loss + args.gate_cost * gate.mean()
            loss = loss + args.anchor_kl * F.kl_div(
                F.log_softmax(logits, -1), teacher, reduction="batchmean")
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step(); total += float(loss.detach()) * len(chosen)
        validation = metrics(adapter, val_data, device)
        row = {"epoch": epoch, "train_loss": total / len(order), **validation}
        history.append(row); print(json.dumps(row), flush=True)
        if validation["macro_f1"] > best_score:
            best_score = validation["macro_f1"]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in adapter.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience: break
    accepted = best_state is not None and (
        best_score - anchor["macro_f1"] >= args.minimum_delta)
    if accepted:
        adapter.load_state_dict(best_state)
        final = metrics(adapter, val_data, device)
        mode = "fused"
    else:
        final = {**anchor, "samples": len(val_data["labels"]),
                 "text_accuracy": anchor["accuracy"],
                 "text_macro_f1": anchor["macro_f1"], "gate_mean": 0.0,
                 "gate_std": 0.0, "help_rate": 0.0, "harm_rate": 0.0}
        mode = "text_anchor"
    summary = {
        "mode": mode, "accepted": accepted, "minimum_delta": args.minimum_delta,
        "anchor": anchor, "best_fused_macro_f1": best_score,
        "best_delta": best_score - anchor["macro_f1"], "final": final,
        "history": history, "test_split_used": False,
    }
    torch.save({"adapter": best_state, "summary": summary,
                "expert_checkpoint": str(args.expert_checkpoint),
                "feature_dim": feature_dim, "hidden_dim": args.hidden_dim,
                "seed": args.seed}, args.output / "best.pt")
    (args.output / "val_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
