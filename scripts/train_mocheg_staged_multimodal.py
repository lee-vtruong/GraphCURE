"""Three-stage visual selection, expert, and utility routing for MOCHEG."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from graphcure.multimodal_evidence import MultimodalEvidenceHead
from scripts.train_mocheg_multimodal_verifier import (
    MultimodalEvidenceDataset,
    collate,
    evaluate,
    freeze_text_branch,
    git_commit,
    load_text_teacher,
    validate_cache_pair,
)
from scripts.train_mocheg_cached_verifier import expected_calibration_error


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def set_trainable(model: torch.nn.Module,
                  modules: tuple[torch.nn.Module, ...]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    for module in modules:
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad = True


def model_batch(batch: dict, device: torch.device) -> tuple[dict, dict]:
    batch = dict(batch)
    batch.pop("id", None)
    supervision = {
        key: batch.pop(key).to(device)
        for key in MultimodalEvidenceDataset.SUPERVISION_KEYS
    }
    inputs = {key: value.to(device) for key, value in batch.items()}
    return inputs, supervision


def visual_selection_objective(
    output: dict[str, torch.Tensor],
    supervision: dict,
    visual_mask: torch.Tensor,
    stance_weight: float,
    prior_kl_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    relevance = supervision["visual_relevance"].bool()
    positive = relevance & visual_mask.bool()
    rows = positive.any(1)
    if not rows.any():
        zero = output["visual_attention"].new_zeros(())
        return zero, {"selection": 0.0, "stance": 0.0, "prior_kl": 0.0}
    positive_mass = torch.sum(
        output["visual_attention"] * positive.float(), dim=1
    )
    selection = -torch.log(positive_mass[rows].clamp_min(1e-8)).mean()
    targets = supervision["labels"].unsqueeze(1).expand_as(relevance)[positive]
    stance = F.cross_entropy(
        output["visual_stance_logits"][positive].float(), targets
    )
    attention = output["visual_attention"][rows].float().clamp_min(1e-8)
    prior = output["visual_prior_attention"][rows].float().clamp_min(1e-8)
    prior_kl = torch.sum(
        prior * (torch.log(prior) - torch.log(attention)), dim=1
    ).mean()
    loss = selection + stance_weight * stance + prior_kl_weight * prior_kl
    return loss, {
        "selection": float(selection.detach()),
        "stance": float(stance.detach()),
        "prior_kl": float(prior_kl.detach()),
    }


def optimizer_for(modules: tuple[torch.nn.Module, ...], learning_rate: float,
                  weight_decay: float) -> torch.optim.Optimizer:
    parameters = [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    return torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )


def hard_router_metrics(rows: list[dict], threshold: float) -> dict:
    gold = np.asarray([row["gold"] for row in rows])
    text = np.asarray([row["text_only_prediction"] for row in rows])
    expert = np.asarray([row["visual_expert_prediction"] for row in rows])
    gate = np.asarray([row["visual_modality_mass"] for row in rows])
    route = gate >= threshold
    prediction = np.where(route, expert, text)
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(gold, prediction)),
        "macro_f1": float(f1_score(gold, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(gold, prediction).tolist(),
        "visual_route_rate": float(route.mean()),
        "visual_help_rate": float(np.mean((text != gold) & (prediction == gold))),
        "visual_harm_rate": float(np.mean((text == gold) & (prediction != gold))),
    }
    if rows and all(
        "text_only_probabilities" in row
        and "visual_expert_probabilities" in row
        for row in rows
    ):
        text_probability = np.asarray([
            row["text_only_probabilities"] for row in rows
        ])
        expert_probability = np.asarray([
            row["visual_expert_probabilities"] for row in rows
        ])
        routed_probability = np.where(
            route[:, None], expert_probability, text_probability
        )
        result["ece_10"] = expected_calibration_error(
            routed_probability, gold
        )
    return result


def calibrate_hard_router(rows: list[dict]) -> dict:
    candidates = [hard_router_metrics(rows, value) for value in np.linspace(0, 1, 101)]
    return max(
        candidates,
        key=lambda row: (row["macro_f1"], row["accuracy"], -row["visual_route_rate"]),
    )


def validate_router_cache(expert_train: MultimodalEvidenceDataset,
                          router_train: MultimodalEvidenceDataset) -> None:
    if expert_train.data["ids"] != router_train.data["ids"]:
        raise ValueError("expert and router train cache IDs do not align")
    if not torch.equal(expert_train.data["labels"], router_train.data["labels"]):
        raise ValueError("expert and router train labels do not align")
    for key in ("claim_dim", "text_dim", "visual_dim", "text_top_k",
                "visual_top_k", "visual_model"):
        if expert_train.metadata.get(key) != router_train.metadata.get(key):
            raise ValueError(f"router cache mismatch for {key}")
    if router_train.metadata.get("train_gold_injection"):
        raise ValueError("router cache must preserve natural train retrieval")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/mocheg_multimodal_cache"))
    parser.add_argument(
        "--val-cache-root", type=Path,
        help="optional validation cache root when train uses a natural "
             "retrieval-only cache",
    )
    parser.add_argument("--router-cache-root", type=Path)
    parser.add_argument("--text-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expert-checkpoint", type=Path,
        help="resume a frozen selector/expert checkpoint and optionally set "
             "selector/expert epochs to zero",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/mocheg_staged_multimodal_seed42"))
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--selector-epochs", type=int, default=12)
    parser.add_argument("--expert-epochs", type=int, default=15)
    parser.add_argument("--router-epochs", type=int, default=20)
    parser.add_argument("--selector-patience", type=int, default=4)
    parser.add_argument("--expert-patience", type=int, default=5)
    parser.add_argument("--router-patience", type=int, default=6)
    parser.add_argument("--selector-learning-rate", type=float, default=3e-4)
    parser.add_argument("--expert-learning-rate", type=float, default=3e-4)
    parser.add_argument("--router-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--selector-stance-weight", type=float, default=0.15)
    parser.add_argument("--selector-prior-kl-weight", type=float, default=0.0)
    parser.add_argument(
        "--visual-attention-mode",
        choices=("learned", "retrieval", "retrieval_residual"),
        default="learned",
    )
    parser.add_argument("--visual-prior-temperature", type=float, default=0.5)
    parser.add_argument("--visual-residual-scale", type=float, default=0.25)
    parser.add_argument("--expert-gold-weight", type=float, default=1.0)
    parser.add_argument("--residual-penalty", type=float, default=0.01)
    parser.add_argument("--router-target-weight", type=float, default=0.5)
    parser.add_argument("--router-ambiguous-weight", type=float, default=0.1)
    parser.add_argument("--router-cost-margin", type=float, default=0.05)
    parser.add_argument("--router-temperature", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.router_temperature <= 0:
        parser.error("--router-temperature must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    train = MultimodalEvidenceDataset(args.cache_root / "train.pt")
    val_root = args.val_cache_root or args.cache_root
    val = MultimodalEvidenceDataset(val_root / "val.pt")
    validate_cache_pair(
        train,
        val,
        expected_train_gold_injection=(args.visual_attention_mode == "learned"),
    )
    router_train = (
        MultimodalEvidenceDataset(args.router_cache_root / "train.pt")
        if args.router_cache_root is not None else train
    )
    if args.router_cache_root is not None:
        validate_router_cache(train, router_train)
    head = MultimodalEvidenceHead(
        claim_dim=int(train.metadata["claim_dim"]),
        text_dim=int(train.metadata["text_dim"]),
        visual_dim=int(train.metadata["visual_dim"]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        visual_attention_mode=args.visual_attention_mode,
        visual_prior_temperature=args.visual_prior_temperature,
        visual_residual_scale=args.visual_residual_scale,
    ).to(device)
    teacher = load_text_teacher(head, args.text_checkpoint)
    resumed_expert = None
    if args.expert_checkpoint is not None:
        resumed_expert = torch.load(
            args.expert_checkpoint, map_location="cpu", weights_only=False
        )
        if resumed_expert.get("stage") != "expert":
            parser.error("--expert-checkpoint must be an expert_best.pt file")
        head.load_state_dict(resumed_expert["head"], strict=True)
    freeze_text_branch(head)
    counts = torch.bincount(train.data["labels"], minlength=3).float()
    class_weights = (counts.sum() / (3 * counts.clamp_min(1))).to(device)
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    router_loader = DataLoader(
        router_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    stage_summary: dict[str, dict] = {}

    # Stage 1: learn which visual candidate is useful, independently of the
    # verdict and router. Validation qrels select the checkpoint but never
    # enter a training batch.
    selector_modules = (
        head.visual_projection, head.visual_utility, head.visual_stance
    )
    set_trainable(head, selector_modules)
    selector_optimizer = optimizer_for(
        selector_modules, args.selector_learning_rate, args.weight_decay
    )
    initial, _ = evaluate(head, val, args.batch_size, device)
    best_selector = initial["visual_selection_hit_at_1"]
    best_selector_state = clone_state(head)
    stale = 0
    for epoch in range(1, args.selector_epochs + 1):
        set_trainable(head, selector_modules)
        total = 0.0
        for batch in loader:
            inputs, supervision = model_batch(batch, device)
            selector_optimizer.zero_grad(set_to_none=True)
            output = head(**inputs)
            loss, _ = visual_selection_objective(
                output, supervision, inputs["visual_mask"],
                args.selector_stance_weight,
                args.selector_prior_kl_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            selector_optimizer.step()
            total += float(loss.detach())
        metrics, _ = evaluate(head, val, args.batch_size, device)
        print(json.dumps({
            "stage": "selector", "epoch": epoch,
            "train_loss": total / len(router_loader),
            "val_visual_selection_hit_at_1":
                metrics["visual_selection_hit_at_1"],
        }), flush=True)
        score = metrics["visual_selection_hit_at_1"]
        # Retrieval-only attention has a constant ranking while its projection
        # and stance heads still learn. Keep the newest tied checkpoint so the
        # downstream expert does not receive an untrained stance head.
        if score >= best_selector:
            best_selector = score
            best_selector_state = clone_state(head)
            stale = 0
        else:
            stale += 1
            if stale >= args.selector_patience:
                break
    head.load_state_dict(best_selector_state)
    stage_summary["selector"] = {"best_val_select_at_1": best_selector}
    torch.save({
        "stage": "selector",
        "head": best_selector_state,
        "metrics": stage_summary["selector"],
        "seed": args.seed,
        "visual_attention_mode": args.visual_attention_mode,
        "visual_prior_temperature": args.visual_prior_temperature,
        "visual_residual_scale": args.visual_residual_scale,
        "test_split_used": False,
    }, args.output / "selector_best.pt")

    # Stage 2: train a full-strength visual expert while the text anchor and
    # evidence selector remain frozen. The router is deliberately absent.
    expert_modules = (head.sufficiency, head.visual_residual)
    set_trainable(head, expert_modules)
    expert_optimizer = optimizer_for(
        expert_modules, args.expert_learning_rate, args.weight_decay
    )
    initial, _ = evaluate(head, val, args.batch_size, device)
    best_expert_score = initial["oracle_router_macro_f1"]
    best_expert_metrics = {
        "visual_expert_macro_f1": initial["visual_expert_macro_f1"],
        "oracle_router_macro_f1": initial["oracle_router_macro_f1"],
    }
    best_expert_state = clone_state(head)
    stale = 0
    for epoch in range(1, args.expert_epochs + 1):
        set_trainable(head, expert_modules)
        total = 0.0
        for batch in loader:
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
        metrics, _ = evaluate(head, val, args.batch_size, device)
        print(json.dumps({
            "stage": "expert", "epoch": epoch,
            "train_loss": total / len(loader),
            "val_visual_expert_macro_f1": metrics["visual_expert_macro_f1"],
            "val_oracle_router_macro_f1": metrics["oracle_router_macro_f1"],
        }), flush=True)
        # A complementary expert can be weaker in isolation while correcting
        # precisely the text anchor's errors. Select it by oracle headroom on
        # validation; the learned router itself still sees train labels only.
        score = metrics["oracle_router_macro_f1"]
        if score > best_expert_score:
            best_expert_score = score
            best_expert_metrics = {
                "visual_expert_macro_f1": metrics["visual_expert_macro_f1"],
                "oracle_router_macro_f1": metrics["oracle_router_macro_f1"],
            }
            best_expert_state = clone_state(head)
            stale = 0
        else:
            stale += 1
            if stale >= args.expert_patience:
                break
    head.load_state_dict(best_expert_state)
    stage_summary["expert"] = {
        "selection_metric": "oracle_router_macro_f1",
        **best_expert_metrics,
    }
    torch.save({
        "stage": "expert",
        "head": best_expert_state,
        "metrics": stage_summary["expert"],
        "seed": args.seed,
        "visual_attention_mode": args.visual_attention_mode,
        "visual_prior_temperature": args.visual_prior_temperature,
        "visual_residual_scale": args.visual_residual_scale,
        "test_split_used": False,
    }, args.output / "expert_best.pt")

    # Stage 3: train only the utility gate. Its soft target is the detached
    # per-example reduction in cross-entropy supplied by the visual expert,
    # minus an explicit visual-use cost margin.
    torch.nn.init.zeros_(head.visual_gate[-1].weight)
    torch.nn.init.constant_(head.visual_gate[-1].bias, -20.0)
    anchor_metrics, _ = evaluate(head, val, args.batch_size, device)
    best_router = anchor_metrics["macro_f1"]
    best_router_state = clone_state(head)
    best_router_mode = "text_anchor"
    best_router_threshold = 1.01
    torch.nn.init.zeros_(head.visual_gate[-1].weight)
    torch.nn.init.constant_(head.visual_gate[-1].bias, -2.944439)
    router_modules = (head.visual_gate,)
    set_trainable(head, router_modules)
    router_optimizer = optimizer_for(
        router_modules, args.router_learning_rate, args.weight_decay
    )
    stale = 0
    for epoch in range(1, args.router_epochs + 1):
        set_trainable(head, router_modules)
        total = 0.0
        helpful_total = 0
        harmful_total = 0
        for batch in router_loader:
            inputs, supervision = model_batch(batch, device)
            router_optimizer.zero_grad(set_to_none=True)
            output = head(**inputs)
            labels = supervision["labels"]
            text_loss = F.cross_entropy(
                output["text_verdict_logits"].float(), labels, reduction="none"
            )
            expert_loss = F.cross_entropy(
                output["visual_expert_logits"].float(), labels, reduction="none"
            )
            text_prediction = output["text_verdict_logits"].argmax(-1)
            expert_prediction = output["visual_expert_logits"].argmax(-1)
            helpful = (text_prediction != labels) & (expert_prediction == labels)
            harmful = (text_prediction == labels) & (expert_prediction != labels)
            decisive = helpful | harmful
            helpful_total += int(helpful.sum().item())
            harmful_total += int(harmful.sum().item())
            gate = output["visual_gate"].float().clamp(1e-6, 1.0 - 1e-6)
            if decisive.any():
                decisive_target = helpful[decisive].float()
                terms = F.binary_cross_entropy(
                    gate[decisive], decisive_target, reduction="none"
                )
                positive_fraction = decisive_target.mean().clamp(1e-3, 1 - 1e-3)
                balance = torch.where(
                    decisive_target.bool(),
                    0.5 / positive_fraction,
                    0.5 / (1.0 - positive_fraction),
                )
                route = torch.sum(terms * balance) / balance.sum()
            else:
                route = gate.new_zeros(())
            soft_target = torch.sigmoid(
                (text_loss - expert_loss - args.router_cost_margin)
                / args.router_temperature
            ).detach()
            ambiguous = ~decisive
            ambiguous_route = (
                F.binary_cross_entropy(gate[ambiguous], soft_target[ambiguous])
                if ambiguous.any() else gate.new_zeros(())
            )
            verdict = F.cross_entropy(
                output["verdict_logits"].float(), labels, weight=class_weights
            )
            loss = verdict + args.router_target_weight * (
                route + args.router_ambiguous_weight * ambiguous_route
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            router_optimizer.step()
            total += float(loss.detach())
        metrics, validation_rows = evaluate(head, val, args.batch_size, device)
        hard_metrics = calibrate_hard_router(validation_rows)
        print(json.dumps({
            "stage": "router", "epoch": epoch,
            "train_loss": total / len(loader),
            "val_macro_f1": metrics["macro_f1"],
            "val_visual_help_rate": metrics["visual_help_rate"],
            "val_visual_harm_rate": metrics["visual_harm_rate"],
            "val_visual_gate_mean": metrics["visual_modality_mass_mean"],
            "val_hard_macro_f1": hard_metrics["macro_f1"],
            "val_hard_threshold": hard_metrics["threshold"],
            "val_hard_visual_route_rate": hard_metrics["visual_route_rate"],
            "train_helpful_pairs": helpful_total,
            "train_harmful_pairs": harmful_total,
        }), flush=True)
        hard_wins = hard_metrics["macro_f1"] > metrics["macro_f1"]
        score = hard_metrics["macro_f1"] if hard_wins else metrics["macro_f1"]
        if score > best_router:
            best_router = score
            best_router_state = clone_state(head)
            best_router_mode = "hard" if hard_wins else "soft"
            best_router_threshold = hard_metrics["threshold"]
            stale = 0
        else:
            stale += 1
            if stale >= args.router_patience:
                break
    head.load_state_dict(best_router_state)
    stage_summary["router"] = {
        "text_anchor_macro_f1": anchor_metrics["text_only_macro_f1"],
        "best_val_macro_f1": best_router,
        "routing_mode": best_router_mode,
        "hard_threshold": best_router_threshold
        if best_router_mode == "hard" else None,
    }
    head.to(device)
    metrics, rows = evaluate(head, val, args.batch_size, device)
    soft_summary = {
        key: metrics[key]
        for key in (
            "accuracy", "macro_f1", "visual_help_rate", "visual_harm_rate"
        )
    }
    if best_router_mode in ("hard", "text_anchor"):
        hard = hard_router_metrics(rows, best_router_threshold)
        for row in rows:
            row["soft_prediction"] = row["prediction"]
            row["soft_probabilities"] = row["probabilities"]
            use_visual = (
                row["visual_modality_mass"] >= best_router_threshold
            )
            row["prediction"] = (
                row["visual_expert_prediction"]
                if use_visual
                else row["text_only_prediction"]
            )
            row["probabilities"] = (
                row["visual_expert_probabilities"]
                if use_visual else row["text_only_probabilities"]
            )
        metrics.update({
            "accuracy": hard["accuracy"],
            "macro_f1": hard["macro_f1"],
            "confusion_matrix": hard["confusion_matrix"],
            "visual_help_rate": hard["visual_help_rate"],
            "visual_harm_rate": hard["visual_harm_rate"],
            "hard_visual_route_rate": hard["visual_route_rate"],
        })
        if "ece_10" in hard:
            metrics["ece_10"] = hard["ece_10"]
    metrics["routing_mode"] = best_router_mode
    metrics["hard_routing_threshold"] = (
        best_router_threshold if best_router_mode in ("hard", "text_anchor")
        else None
    )
    metrics["soft_router"] = soft_summary
    metrics["stages"] = stage_summary
    metrics["provenance"] = {
        "git_commit": git_commit(),
        "seed": args.seed,
        "text_checkpoint": str(args.text_checkpoint),
        "text_teacher": teacher,
        "resumed_expert_checkpoint": str(args.expert_checkpoint)
        if args.expert_checkpoint is not None else None,
        "cache_metadata": val.metadata,
        "router_train_cache_metadata": router_train.metadata,
        "test_split_used": False,
    }
    torch.save({
        "head": head.state_dict(),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "stages": stage_summary,
        "cache_metadata": train.metadata,
        "text_teacher": teacher,
        "resumed_expert_checkpoint": str(args.expert_checkpoint)
        if args.expert_checkpoint is not None else None,
        "router_train_cache_metadata": router_train.metadata,
        "routing_mode": best_router_mode,
        "hard_routing_threshold": best_router_threshold,
        "visual_attention_mode": args.visual_attention_mode,
        "visual_prior_temperature": args.visual_prior_temperature,
        "visual_residual_scale": args.visual_residual_scale,
    }, args.output / "best.pt")
    (args.output / "val_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "val_predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
