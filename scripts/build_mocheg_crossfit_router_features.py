"""Build honest out-of-fold expert outcomes for MOCHEG router training."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from graphcure.multimodal_evidence import MultimodalEvidenceHead
from graphcure.evidence_set import EvidenceSetHead, evidence_set_loss
from scripts.train_mocheg_cached_verifier import (
    CachedEvidenceDataset,
    collate as text_collate,
)
from scripts.train_mocheg_multimodal_verifier import (
    MultimodalEvidenceDataset,
    collate as multimodal_collate,
    freeze_text_branch,
    git_commit,
    load_text_teacher,
)
from scripts.train_mocheg_set_router import extract_features, load_expert
from scripts.train_mocheg_staged_multimodal import (
    model_batch,
    optimizer_for,
    set_trainable,
    validate_router_cache,
    visual_selection_objective,
)


def save_features(
    path: Path,
    features: np.ndarray,
    gold: np.ndarray,
    text_prediction: np.ndarray,
    expert_prediction: np.ndarray,
    feature_names: list[str],
    ids: list[str],
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        features=features.astype(np.float32),
        gold=gold.astype(np.int64),
        text_prediction=text_prediction.astype(np.int64),
        expert_prediction=expert_prediction.astype(np.int64),
        feature_names=np.asarray(feature_names),
        ids=np.asarray(ids),
    )
    temporary.replace(path)


def load_features(path: Path) -> tuple:
    payload = np.load(path, allow_pickle=False)
    return (
        payload["features"], payload["gold"], payload["text_prediction"],
        payload["expert_prediction"], payload["feature_names"].tolist(),
        payload["ids"].tolist(),
    )


def build_head(
    metadata: dict,
    text_checkpoint: Path,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
) -> MultimodalEvidenceHead:
    head = MultimodalEvidenceHead(
        claim_dim=int(metadata["claim_dim"]),
        text_dim=int(metadata["text_dim"]),
        visual_dim=int(metadata["visual_dim"]),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    load_text_teacher(head, text_checkpoint)
    freeze_text_branch(head)
    return head


def train_fold_text_teacher(
    dataset: CachedEvidenceDataset,
    fit_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fold: int,
    checkpoint_path: Path,
) -> None:
    head = EvidenceSetHead(
        encoder_dim=int(dataset.metadata["embedding_dim"]),
        hidden_dim=args.hidden_dim,
        retrieval_dim=6,
        dropout=args.dropout,
    ).to(device)
    subset = Subset(dataset, fit_indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=text_collate,
        pin_memory=device.type == "cuda",
    )
    fit_labels = dataset.data["labels"][
        torch.as_tensor(fit_indices, dtype=torch.long)
    ]
    counts = torch.bincount(fit_labels, minlength=3).float()
    class_weights = (counts.sum() / (3 * counts.clamp_min(1))).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.text_learning_rate,
        weight_decay=args.weight_decay,
    )
    for epoch in range(1, args.text_epochs + 1):
        head.train()
        total = 0.0
        batches = 0
        for batch in tqdm(
            loader, desc=f"fold {fold} text {epoch}/{args.text_epochs}",
            leave=False,
        ):
            batch.pop("id")
            labels = batch.pop("labels").to(device)
            relevance = batch.pop("relevance").to(device)
            relevance_weights = batch.pop("relevance_weights").to(device)
            evidence_mask = batch["evidence_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = head(**{
                key: value.to(device) for key, value in batch.items()
            })
            loss, _ = evidence_set_loss(
                output,
                labels,
                relevance,
                evidence_mask,
                relevance_weights=relevance_weights,
                class_weights=class_weights,
                relevance_weight=args.text_relevance_weight,
                stance_weight=args.text_stance_weight,
                sufficiency_weight=args.text_sufficiency_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        print(json.dumps({
            "fold": fold, "stage": "text", "epoch": epoch,
            "train_loss": total / max(batches, 1),
        }), flush=True)
    torch.save({
        "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed + fold,
        "cache_metadata": dataset.metadata,
        "fold": fold,
        "test_split_used": False,
    }, checkpoint_path)
    del head


def train_fold_expert(
    head: MultimodalEvidenceHead,
    dataset: MultimodalEvidenceDataset,
    fit_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fold: int,
) -> None:
    subset = Subset(dataset, fit_indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=multimodal_collate,
        pin_memory=device.type == "cuda",
    )
    selector_modules = (
        head.visual_projection, head.visual_utility, head.visual_stance
    )
    set_trainable(head, selector_modules)
    selector_optimizer = optimizer_for(
        selector_modules, args.selector_learning_rate, args.weight_decay
    )
    for epoch in range(1, args.selector_epochs + 1):
        set_trainable(head, selector_modules)
        total = 0.0
        batches = 0
        for batch in tqdm(
            loader, desc=f"fold {fold} selector {epoch}/{args.selector_epochs}",
            leave=False,
        ):
            inputs, supervision = model_batch(batch, device)
            selector_optimizer.zero_grad(set_to_none=True)
            output = head(**inputs)
            loss, _ = visual_selection_objective(
                output, supervision, inputs["visual_mask"],
                args.selector_stance_weight,
            )
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                selector_optimizer.step()
            total += float(loss.detach())
            batches += 1
        print(json.dumps({
            "fold": fold, "stage": "selector", "epoch": epoch,
            "train_loss": total / max(batches, 1),
        }), flush=True)

    expert_modules = (head.sufficiency, head.visual_residual)
    set_trainable(head, expert_modules)
    expert_optimizer = optimizer_for(
        expert_modules, args.expert_learning_rate, args.weight_decay
    )
    fit_labels = dataset.data["labels"][
        torch.as_tensor(fit_indices, dtype=torch.long)
    ]
    counts = torch.bincount(fit_labels, minlength=3).float()
    class_weights = (counts.sum() / (3 * counts.clamp_min(1))).to(device)
    for epoch in range(1, args.expert_epochs + 1):
        set_trainable(head, expert_modules)
        total = 0.0
        batches = 0
        for batch in tqdm(
            loader, desc=f"fold {fold} expert {epoch}/{args.expert_epochs}",
            leave=False,
        ):
            inputs, supervision = model_batch(batch, device)
            expert_optimizer.zero_grad(set_to_none=True)
            output = head(**inputs)
            terms = F.cross_entropy(
                output["visual_expert_logits"].float(),
                supervision["labels"],
                weight=class_weights,
                reduction="none",
            )
            has_gold = supervision["visual_relevance"].bool().any(1).float()
            sample_weight = 1.0 + args.expert_gold_weight * has_gold
            verdict = torch.sum(terms * sample_weight) / sample_weight.sum()
            penalty = output["visual_residual_logits"].float().square().mean()
            loss = verdict + args.residual_penalty * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            expert_optimizer.step()
            total += float(loss.detach())
            batches += 1
        print(json.dumps({
            "fold": fold, "stage": "expert", "epoch": epoch,
            "train_loss": total / max(batches, 1),
        }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-cache-root", type=Path, required=True)
    parser.add_argument("--full-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-cache-root", type=Path, required=True)
    parser.add_argument("--router-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--selector-epochs", type=int, default=8)
    parser.add_argument("--expert-epochs", type=int, default=4)
    parser.add_argument("--text-epochs", type=int, default=2)
    parser.add_argument("--text-learning-rate", type=float, default=3e-4)
    parser.add_argument("--text-relevance-weight", type=float, default=0.25)
    parser.add_argument("--text-stance-weight", type=float, default=0.15)
    parser.add_argument("--text-sufficiency-weight", type=float, default=0.15)
    parser.add_argument("--selector-learning-rate", type=float, default=3e-4)
    parser.add_argument("--expert-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--selector-stance-weight", type=float, default=0.15)
    parser.add_argument("--expert-gold-weight", type=float, default=1.0)
    parser.add_argument("--residual-penalty", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    expert_train = MultimodalEvidenceDataset(
        args.expert_cache_root / "train.pt"
    )
    text_train = CachedEvidenceDataset(args.text_cache_root / "train.pt")
    natural_train = MultimodalEvidenceDataset(
        args.router_cache_root / "train.pt"
    )
    val = MultimodalEvidenceDataset(args.expert_cache_root / "val.pt")
    validate_router_cache(expert_train, natural_train)
    if text_train.data["ids"] != expert_train.data["ids"]:
        parser.error("text and multimodal train cache IDs do not align")
    if not torch.equal(text_train.data["labels"], expert_train.data["labels"]):
        parser.error("text and multimodal train cache labels do not align")
    labels = expert_train.data["labels"].numpy()
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_paths: list[Path] = []
    for fold, (fit_indices, held_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        fold_path = args.output_root / f"train_fold_{fold}.npz"
        fold_paths.append(fold_path)
        if fold_path.exists() and not args.no_resume:
            print(f"fold {fold}: resume {fold_path}", flush=True)
            continue
        seed = args.seed + fold
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        fold_text_checkpoint = args.output_root / f"fold_{fold}_text.pt"
        train_fold_text_teacher(
            text_train, fit_indices, args, device, fold, fold_text_checkpoint
        )
        head = build_head(
            expert_train.metadata, fold_text_checkpoint, args.hidden_dim,
            args.dropout, device,
        )
        train_fold_expert(
            head, expert_train, fit_indices, args, device, fold
        )
        held_subset = Subset(natural_train, held_indices.tolist())
        features = extract_features(
            head, held_subset, args.batch_size, device, f"fold {fold} held-out"
        )
        save_features(fold_path, *features)
        torch.save({
            "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
            "fold": fold,
            "fit_samples": len(fit_indices),
            "held_out_samples": len(held_indices),
            "test_split_used": False,
        }, args.output_root / f"fold_{fold}_expert.pt")
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_payloads = [load_features(path) for path in fold_paths]
    names = fold_payloads[0][4]
    if any(payload[4] != names for payload in fold_payloads):
        raise RuntimeError("cross-fit feature schemas do not match")
    save_features(
        args.output_root / "train_oof.npz",
        np.concatenate([payload[0] for payload in fold_payloads]),
        np.concatenate([payload[1] for payload in fold_payloads]),
        np.concatenate([payload[2] for payload in fold_payloads]),
        np.concatenate([payload[3] for payload in fold_payloads]),
        names,
        sum((payload[5] for payload in fold_payloads), []),
    )
    full_head, full_checkpoint = load_expert(
        args.full_expert_checkpoint, val.metadata, device
    )
    val_features = extract_features(
        full_head, val, args.batch_size, device, "validation full expert"
    )
    save_features(args.output_root / "val_full.npz", *val_features)
    metadata = {
        "protocol": "fold expert never observes its held-out router target",
        "folds": args.folds,
        "selector_epochs": args.selector_epochs,
        "expert_epochs": args.expert_epochs,
        "text_epochs": args.text_epochs,
        "train_samples": len(labels),
        "val_samples": len(val),
        "feature_count": len(names),
        "text_cache_root": str(args.text_cache_root),
        "full_expert_checkpoint": str(args.full_expert_checkpoint),
        "full_expert_checkpoint_stage": full_checkpoint.get("stage"),
        "git_commit": git_commit(),
        "test_split_used": False,
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
