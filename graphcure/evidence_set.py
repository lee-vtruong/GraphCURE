"""Claim-conditioned evidence selection and verdict aggregation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the final non-padding token for right- or left-padded sequences."""
    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden and attention_mask have incompatible shapes")
    if torch.all(attention_mask[:, -1] == 1):
        return hidden[:, -1]
    indices = attention_mask.sum(dim=1).clamp_min(1) - 1
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, indices]


class EvidenceSetHead(nn.Module):
    """Jointly learn evidence utility, sufficiency, stance, and claim verdict."""

    def __init__(
        self,
        encoder_dim: int,
        hidden_dim: int = 384,
        retrieval_dim: int = 3,
        num_labels: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.claim_projection = nn.Sequential(
            nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, hidden_dim), nn.GELU()
        )
        self.evidence_projection = nn.Sequential(
            nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, hidden_dim), nn.GELU()
        )
        pair_dim = hidden_dim * 4 + retrieval_dim
        self.utility = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.stance = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
        self.sufficiency = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + num_labels),
            nn.Linear(hidden_dim * 2 + num_labels, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.verdict = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + num_labels + 1),
            nn.Linear(hidden_dim * 4 + num_labels + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        claim: torch.Tensor,
        evidence: torch.Tensor,
        evidence_mask: torch.Tensor,
        retrieval_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        q = self.claim_projection(claim.float())
        e = self.evidence_projection(evidence.float())
        qk = q.unsqueeze(1).expand_as(e)
        pair = torch.cat((qk, e, torch.abs(qk - e), qk * e, retrieval_features), -1)
        utility_logits = self.utility(pair).squeeze(-1)
        utility_logits = utility_logits.masked_fill(~evidence_mask.bool(), -1e4)
        attention = torch.softmax(utility_logits, -1)
        summary = torch.sum(attention.unsqueeze(-1) * e, dim=1)
        stance_logits = self.stance(pair)
        stance_prob = torch.softmax(stance_logits, -1)
        aggregate_stance = torch.sum(attention.unsqueeze(-1) * stance_prob, dim=1)
        sufficiency_input = torch.cat((q, summary, aggregate_stance), -1)
        sufficiency_logit = self.sufficiency(sufficiency_input).squeeze(-1)
        verdict_input = torch.cat(
            (
                q,
                summary,
                torch.abs(q - summary),
                q * summary,
                aggregate_stance,
                torch.sigmoid(sufficiency_logit).unsqueeze(-1),
            ),
            -1,
        )
        return {
            "verdict_logits": self.verdict(verdict_input),
            "utility_logits": utility_logits,
            "attention": attention,
            "stance_logits": stance_logits,
            "sufficiency_logit": sufficiency_logit,
        }


def evidence_set_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    relevance: torch.Tensor,
    evidence_mask: torch.Tensor,
    relevance_weights: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    relevance_weight: float = 0.25,
    stance_weight: float = 0.15,
    sufficiency_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-task loss without assigning the claim label to every paragraph."""
    mask = evidence_mask.bool()
    relevant = relevance.bool() & mask
    verdict = F.cross_entropy(outputs["verdict_logits"].float(), labels, weight=class_weights)
    relevance_terms = F.binary_cross_entropy_with_logits(
        outputs["utility_logits"][mask].float(),
        relevance[mask].float(),
        reduction="none",
    )
    if relevance_weights is not None:
        weights = relevance_weights[mask].float()
        relevance_loss = torch.sum(relevance_terms * weights) / weights.sum().clamp_min(1.0)
    else:
        relevance_loss = relevance_terms.mean()
    stance_mask = relevant
    if stance_mask.any():
        stance_targets = labels.unsqueeze(1).expand_as(relevance)[stance_mask]
        stance = F.cross_entropy(
            outputs["stance_logits"][stance_mask].float(), stance_targets
        )
    else:
        stance = verdict.new_zeros(())
    sufficiency_target = relevant.any(dim=1).float()
    sufficiency = F.binary_cross_entropy_with_logits(
        outputs["sufficiency_logit"].float(), sufficiency_target
    )
    total = (
        verdict
        + relevance_weight * relevance_loss
        + stance_weight * stance
        + sufficiency_weight * sufficiency
    )
    return total, {
        "verdict": verdict.detach(),
        "relevance": relevance_loss.detach(),
        "stance": stance.detach(),
        "sufficiency": sufficiency.detach(),
    }
