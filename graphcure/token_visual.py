"""Utilities for token-level claim--image evidence learning."""
from __future__ import annotations

import hashlib
import torch
import torch.nn.functional as F

IRRELEVANT = 3


def select_candidate_ids(sample_id: str, retrieved: list[str], gold: set[str],
                         top_k: int, inject_gold: bool) -> list[str]:
    """Select unique candidates; only training may inject one qrel positive."""
    retrieved = list(dict.fromkeys(str(value) for value in retrieved))
    if not inject_gold or not gold:
        return retrieved[:top_k]
    injected = min(gold, key=lambda value: hashlib.sha256(
        f"{sample_id}\0{value}".encode("utf-8")).digest())
    candidates = [injected] + [value for value in retrieved if value != injected]
    # Stable shuffle prevents a "positive is at position zero" shortcut.
    return sorted(candidates[:top_k], key=lambda value: hashlib.sha256(
        f"token-pair\0{sample_id}\0{value}".encode("utf-8")).digest())


def aggregate_pair_logits(pair_logits: torch.Tensor, candidate_mask: torch.Tensor,
                          rank_prior: torch.Tensor | None = None,
                          prior_weight: float = 0.25
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate four-way pair predictions into one three-way claim verdict."""
    stance = pair_logits[..., :3]
    relevance = torch.logsumexp(stance, dim=-1) - pair_logits[..., IRRELEVANT]
    if rank_prior is not None:
        relevance = relevance + prior_weight * rank_prior.float()
    relevance = relevance.masked_fill(~candidate_mask, -1e4)
    attention = torch.softmax(relevance.float(), dim=-1) * candidate_mask.float()
    attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
    return torch.sum(attention.unsqueeze(-1) * stance.float(), dim=1), attention


def token_visual_loss(pair_logits: torch.Tensor, claim_logits: torch.Tensor,
                      labels: torch.Tensor, relevance: torch.Tensor,
                      candidate_mask: torch.Tensor, claim_has_gold: torch.Tensor,
                      negative_weight: float = 0.25,
                      claim_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint pair relevance/stance and claim-verdict objective."""
    pair_targets = torch.where(
        relevance.bool(), labels[:, None].expand_as(relevance),
        torch.full_like(relevance, IRRELEVANT))
    valid = candidate_mask.bool()
    weights = (valid & relevance.bool()).float()
    weights += (valid & ~relevance.bool()).float() * float(negative_weight)
    per_pair = F.cross_entropy(
        pair_logits.float().reshape(-1, 4), pair_targets.reshape(-1),
        reduction="none").reshape_as(relevance)
    pair_loss = (per_pair * weights).sum() / weights.sum().clamp_min(1.0)
    claim_loss = (F.cross_entropy(
        claim_logits.float()[claim_has_gold], labels[claim_has_gold])
        if claim_has_gold.any() else pair_loss.new_zeros(()))
    loss = pair_loss + float(claim_weight) * claim_loss
    return loss, {"pair_loss": float(pair_loss.detach()),
                  "claim_loss": float(claim_loss.detach())}
