"""Claim-conditioned text/visual evidence aggregation for GraphCURE-R2V."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultimodalEvidenceHead(nn.Module):
    """Select evidence within each modality and reason over their conflict."""

    def __init__(
        self,
        claim_dim: int,
        text_dim: int,
        visual_dim: int,
        hidden_dim: int = 384,
        text_retrieval_dim: int = 6,
        visual_retrieval_dim: int = 3,
        num_labels: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.claim_projection = nn.Sequential(
            nn.LayerNorm(claim_dim), nn.Linear(claim_dim, hidden_dim), nn.GELU()
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU()
        )
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim), nn.GELU()
        )
        self.text_utility, self.text_stance = self._pair_heads(
            hidden_dim, text_retrieval_dim, num_labels, dropout
        )
        self.visual_utility, self.visual_stance = self._pair_heads(
            hidden_dim, visual_retrieval_dim, num_labels, dropout
        )
        # q, text summary, visual summary, |text-visual|, q*text, q*visual,
        # aggregate stance, cross-modal stance conflict, modality mass,
        # and the learned sufficiency scalar.
        verdict_dim = hidden_dim * 6 + num_labels * 2 + 3
        sufficiency_dim = hidden_dim * 3 + num_labels * 2 + 2
        self.sufficiency = nn.Sequential(
            nn.LayerNorm(sufficiency_dim),
            nn.Linear(sufficiency_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.verdict = nn.Sequential(
            nn.LayerNorm(verdict_dim),
            nn.Linear(verdict_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    @staticmethod
    def _pair_heads(hidden_dim: int, retrieval_dim: int, num_labels: int,
                    dropout: float) -> tuple[nn.Module, nn.Module]:
        pair_dim = hidden_dim * 4 + retrieval_dim

        def head(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        return head(1), head(num_labels)

    @staticmethod
    def _pair(q: torch.Tensor, evidence: torch.Tensor,
              retrieval: torch.Tensor) -> torch.Tensor:
        query = q.unsqueeze(1).expand_as(evidence)
        return torch.cat(
            (query, evidence, torch.abs(query - evidence), query * evidence,
             retrieval.float()),
            dim=-1,
        )

    @staticmethod
    def _masked_summary(values: torch.Tensor, attention: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        weights = attention * mask.float()
        return torch.sum(weights.unsqueeze(-1) * values, dim=1)

    @staticmethod
    def _modality_stance(stance: torch.Tensor, attention: torch.Tensor,
                         mask: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        probability = torch.softmax(stance, dim=-1)
        weighted = torch.sum(
            attention.unsqueeze(-1) * probability * mask.unsqueeze(-1).float(),
            dim=1,
        )
        return weighted / mass.unsqueeze(-1).clamp_min(1e-6)

    def forward(
        self,
        claim: torch.Tensor,
        text_evidence: torch.Tensor,
        text_mask: torch.Tensor,
        text_retrieval_features: torch.Tensor,
        visual_evidence: torch.Tensor,
        visual_mask: torch.Tensor,
        visual_retrieval_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        q = self.claim_projection(claim.float())
        text = self.text_projection(text_evidence.float())
        visual = self.visual_projection(visual_evidence.float())
        text_pair = self._pair(q, text, text_retrieval_features)
        visual_pair = self._pair(q, visual, visual_retrieval_features)
        text_utility = self.text_utility(text_pair).squeeze(-1)
        visual_utility = self.visual_utility(visual_pair).squeeze(-1)
        text_utility = text_utility.masked_fill(~text_mask.bool(), -1e4)
        visual_utility = visual_utility.masked_fill(~visual_mask.bool(), -1e4)
        joint_mask = torch.cat((text_mask.bool(), visual_mask.bool()), dim=1)
        joint_utility = torch.cat((text_utility, visual_utility), dim=1)
        joint_utility = joint_utility.masked_fill(~joint_mask, -1e4)
        joint_attention = torch.softmax(joint_utility, dim=1)
        text_count = text.shape[1]
        text_attention = joint_attention[:, :text_count]
        visual_attention = joint_attention[:, text_count:]
        text_mass = torch.sum(text_attention * text_mask.float(), dim=1)
        visual_mass = torch.sum(visual_attention * visual_mask.float(), dim=1)
        modality_mass = torch.stack((text_mass, visual_mass), dim=-1)
        text_summary = self._masked_summary(
            text, text_attention, text_mask.bool()
        )
        visual_summary = self._masked_summary(
            visual, visual_attention, visual_mask.bool()
        )
        text_stance_logits = self.text_stance(text_pair)
        visual_stance_logits = self.visual_stance(visual_pair)
        text_stance = self._modality_stance(
            text_stance_logits, text_attention, text_mask.bool(), text_mass
        )
        visual_stance = self._modality_stance(
            visual_stance_logits, visual_attention, visual_mask.bool(), visual_mass
        )
        aggregate_stance = (
            text_stance * text_mass.unsqueeze(-1)
            + visual_stance * visual_mass.unsqueeze(-1)
        )
        conflict = torch.abs(text_stance - visual_stance)
        both_modalities = (text_mask.any(1) & visual_mask.any(1)).unsqueeze(-1)
        conflict = conflict * both_modalities.float()
        sufficiency_input = torch.cat(
            (
                q, text_summary, visual_summary, aggregate_stance, conflict,
                modality_mass,
            ),
            dim=-1,
        )
        sufficiency_logit = self.sufficiency(sufficiency_input).squeeze(-1)
        verdict_input = torch.cat(
            (
                q,
                text_summary,
                visual_summary,
                torch.abs(text_summary - visual_summary),
                q * text_summary,
                q * visual_summary,
                aggregate_stance,
                conflict,
                modality_mass,
                torch.sigmoid(sufficiency_logit).unsqueeze(-1),
            ),
            dim=-1,
        )
        return {
            "verdict_logits": self.verdict(verdict_input),
            "text_utility_logits": text_utility,
            "visual_utility_logits": visual_utility,
            "text_attention": text_attention,
            "visual_attention": visual_attention,
            "text_stance_logits": text_stance_logits,
            "visual_stance_logits": visual_stance_logits,
            "sufficiency_logit": sufficiency_logit,
            "modality_mass": modality_mass,
            "conflict": conflict,
        }


def _masked_relevance_loss(logits: torch.Tensor, relevance: torch.Tensor,
                           mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    selected = mask.bool()
    if not selected.any():
        return logits.new_zeros(())
    terms = F.binary_cross_entropy_with_logits(
        logits[selected].float(), relevance[selected].float(), reduction="none"
    )
    chosen_weights = weights[selected].float()
    return torch.sum(terms * chosen_weights) / chosen_weights.sum().clamp_min(1.0)


def _stance_loss(logits: torch.Tensor, labels: torch.Tensor,
                 relevance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = relevance.bool() & mask.bool()
    if not selected.any():
        return logits.new_zeros(())
    targets = labels.unsqueeze(1).expand_as(relevance)[selected]
    return F.cross_entropy(logits[selected].float(), targets)


def multimodal_evidence_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    text_relevance: torch.Tensor,
    text_mask: torch.Tensor,
    text_relevance_weights: torch.Tensor,
    visual_relevance: torch.Tensor,
    visual_mask: torch.Tensor,
    visual_relevance_weights: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    relevance_weight: float = 0.25,
    stance_weight: float = 0.15,
    sufficiency_weight: float = 0.15,
    conflict_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    verdict = F.cross_entropy(
        outputs["verdict_logits"].float(), labels, weight=class_weights
    )
    text_relevance_loss = _masked_relevance_loss(
        outputs["text_utility_logits"], text_relevance, text_mask,
        text_relevance_weights,
    )
    visual_relevance_loss = _masked_relevance_loss(
        outputs["visual_utility_logits"], visual_relevance, visual_mask,
        visual_relevance_weights,
    )
    text_stance = _stance_loss(
        outputs["text_stance_logits"], labels, text_relevance, text_mask
    )
    visual_stance = _stance_loss(
        outputs["visual_stance_logits"], labels, visual_relevance, visual_mask
    )
    any_relevant = (
        (text_relevance.bool() & text_mask.bool()).any(1)
        | (visual_relevance.bool() & visual_mask.bool()).any(1)
    ).float()
    sufficiency = F.binary_cross_entropy_with_logits(
        outputs["sufficiency_logit"].float(), any_relevant
    )
    both_relevant = (
        (text_relevance.bool() & text_mask.bool()).any(1)
        & (visual_relevance.bool() & visual_mask.bool()).any(1)
    )
    conflict = (
        outputs["conflict"][both_relevant].mean()
        if both_relevant.any() else verdict.new_zeros(())
    )
    relevance = 0.5 * (text_relevance_loss + visual_relevance_loss)
    stance = 0.5 * (text_stance + visual_stance)
    total = (
        verdict
        + relevance_weight * relevance
        + stance_weight * stance
        + sufficiency_weight * sufficiency
        + conflict_weight * conflict
    )
    return total, {
        "verdict": verdict.detach(),
        "relevance": relevance.detach(),
        "stance": stance.detach(),
        "sufficiency": sufficiency.detach(),
        "conflict": conflict.detach(),
    }
