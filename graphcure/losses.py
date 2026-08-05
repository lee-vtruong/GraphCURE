from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_constraint_loss(
    logits: torch.Tensor, targets: torch.Tensor, unknown_index: int = -100
) -> torch.Tensor:
    if not (targets != unknown_index).any():
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), ignore_index=unknown_index
    )


def typed_conflict_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = targets >= 0
    if not mask.any():
        return scores.sum() * 0.0
    return F.binary_cross_entropy(scores[mask], targets[mask].float())


def counterfactual_loss(
    factual_prob: torch.Tensor,
    counterfactual_prob: torch.Tensor,
    changed_mask: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Invariant on unchanged nodes and sensitive on annotated changed nodes."""
    eps = 1e-7
    p = factual_prob.clamp_min(eps)
    q = counterfactual_prob.clamp_min(eps)
    m = 0.5 * (p + q)
    js = 0.5 * (
        (p * (p.log() - m.log())).sum(-1)
        + (q * (q.log() - m.log())).sum(-1)
    )
    changed_mask = changed_mask.bool()
    inv = js[~changed_mask].mean() if (~changed_mask).any() else js.sum() * 0.0
    sens = (
        F.relu(margin - js[changed_mask]).mean()
        if changed_mask.any()
        else js.sum() * 0.0
    )
    return inv + sens


def total_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    cf_output: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parts = {
        "label": F.cross_entropy(output["verdict_logits"], batch["label"]),
        "constraint": masked_constraint_loss(
            output["constraint_logits"], batch["constraint_labels"]
        ),
        "conflict": typed_conflict_loss(output["conflict"], batch["conflict_labels"]),
    }
    if cf_output is not None and "changed_mask" in batch:
        parts["counterfactual"] = counterfactual_loss(
            output["constraint_prob"],
            cf_output["constraint_prob"],
            batch["changed_mask"],
        )
    else:
        parts["counterfactual"] = output["verdict_logits"].sum() * 0.0
    loss = sum(weights.get(name, 0.0) * value for name, value in parts.items())
    return loss, parts
