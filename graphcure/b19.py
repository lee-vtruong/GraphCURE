"""Reusable objectives and supervision rules for GraphCURE B19."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


VARIANT_DEFAULTS: dict[str, dict[str, float]] = {
    "matched_control": {"lambda_kd": 0.0, "lambda_cf": 0.0, "disagreement_alpha": 0.0},
    "ensemble_kd": {"lambda_kd": 0.5, "lambda_cf": 0.0, "disagreement_alpha": 0.0},
    "disagreement_kd": {"lambda_kd": 0.5, "lambda_cf": 0.0, "disagreement_alpha": 1.0},
    "counterfactual_only": {"lambda_kd": 0.0, "lambda_cf": 0.25, "disagreement_alpha": 0.0},
    "full": {"lambda_kd": 0.5, "lambda_cf": 0.25, "disagreement_alpha": 1.0},
}


def heterogeneous_teacher_target(
    label: int,
    direct_probabilities: list[float] | np.ndarray,
    grounded_probabilities: list[float] | np.ndarray,
    smoothing: float = 0.1,
) -> tuple[np.ndarray, str, bool]:
    """Select a train-only soft target while refusing to propagate two wrong teachers."""
    direct = np.asarray(direct_probabilities, dtype=np.float64)
    grounded = np.asarray(grounded_probabilities, dtype=np.float64)
    if direct.shape != (3,) or grounded.shape != (3,):
        raise ValueError("teacher probability vectors must each contain three classes")
    direct /= direct.sum()
    grounded /= grounded.sum()
    direct_correct = int(direct.argmax()) == int(label)
    grounded_correct = int(grounded.argmax()) == int(label)
    disagreement = int(direct.argmax()) != int(grounded.argmax())
    if direct_correct and not grounded_correct:
        return direct, "direct_only_correct", disagreement
    if grounded_correct and not direct_correct:
        return grounded, "grounded_only_correct", disagreement
    if direct_correct and grounded_correct:
        return (direct + grounded) / 2.0, "both_correct", disagreement
    target = np.full(3, smoothing / 2.0, dtype=np.float64)
    target[int(label)] = 1.0 - smoothing
    return target, "both_wrong_gold_smoothed", disagreement


def weighted_verdict_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    disagreement: torch.Tensor,
    disagreement_alpha: float,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    weights = 1.0 + float(disagreement_alpha) * disagreement.float()
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def soft_distillation_loss(
    logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    t = float(temperature)
    teacher = teacher_probabilities.clamp_min(1e-8)
    teacher = torch.softmax(torch.log(teacher) / t, dim=-1)
    student_log = torch.log_softmax(logits / t, dim=-1)
    return F.kl_div(student_log, teacher, reduction="batchmean") * (t * t)


def counterfactual_sufficiency_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float,
    nei_label: int = 2,
) -> torch.Tensor:
    """Require key evidence to raise the gold verdict and suppress false sufficiency."""
    if not bool(valid_mask.any()):
        return positive_logits.sum() * 0.0
    selected_positive = positive_logits[valid_mask]
    selected_negative = negative_logits[valid_mask]
    selected_labels = labels[valid_mask]
    indices = torch.arange(len(selected_labels), device=labels.device)
    positive_gold = selected_positive[indices, selected_labels]
    negative_gold = selected_negative[indices, selected_labels]
    verdict_margin = F.relu(float(margin) - positive_gold + negative_gold)
    positive_nei = selected_positive[:, nei_label]
    negative_nei = selected_negative[:, nei_label]
    sufficiency_margin = F.relu(float(margin) - negative_nei + positive_nei)
    return (verdict_margin + sufficiency_margin).mean()


def resolve_variant(name: str, overrides: dict[str, Any] | None = None) -> dict[str, float]:
    if name not in VARIANT_DEFAULTS:
        raise ValueError(f"unknown B19 variant: {name}")
    result = dict(VARIANT_DEFAULTS[name])
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                result[key] = float(value)
    return result
