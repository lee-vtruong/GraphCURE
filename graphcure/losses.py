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


def directional_intervention_loss(
    factual: dict[str, torch.Tensor],
    counterfactual: dict[str, torch.Tensor],
    labels: torch.Tensor,
    cf_labels: torch.Tensor,
    changed_mask: torch.Tensor,
    verdict_margin: float = 0.5,
    constraint_margin: float = 0.1,
) -> torch.Tensor:
    """Order paired scores in the known pristine-to-falsified direction."""
    direction = (cf_labels.float() - labels.float()).sign()
    factual_score = factual["verdict_logits"][:, 1] - factual["verdict_logits"][:, 0]
    paired_score = counterfactual["verdict_logits"][:, 1] - counterfactual["verdict_logits"][:, 0]
    verdict = F.relu(verdict_margin - direction * (paired_score - factual_score)).mean()
    factual_violated = factual["constraint_prob"][..., 1]
    paired_violated = counterfactual["constraint_prob"][..., 1]
    selected_delta = (paired_violated - factual_violated)[changed_mask.bool()]
    selected_direction = direction[:, None].expand_as(changed_mask)[changed_mask.bool()]
    constraint = F.relu(
        constraint_margin - selected_direction * selected_delta
    ).mean()
    return verdict + constraint


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
        parts["directional"] = directional_intervention_loss(
            output, cf_output, batch["label"], batch["cf_label"], batch["changed_mask"]
        )
    else:
        parts["counterfactual"] = output["verdict_logits"].sum() * 0.0
        parts["directional"] = output["verdict_logits"].sum() * 0.0
    loss = sum(weights.get(name, 0.0) * value for name, value in parts.items())
    return loss, parts
