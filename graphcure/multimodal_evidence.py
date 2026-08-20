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
        self.text_sufficiency = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + num_labels),
            nn.Linear(hidden_dim * 2 + num_labels, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.text_verdict = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + num_labels + 1),
            nn.Linear(hidden_dim * 4 + num_labels + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
        # q, text summary, visual summary, |text-visual|, q*text, q*visual,
        # aggregate stance, cross-modal stance conflict, modality mass,
        # and the learned sufficiency scalar.
        verdict_dim = hidden_dim * 6 + num_labels * 2 + 3
        # Base evidence diagnostics plus probability-level conflict features:
        # p_text, p_visual, |p_text-p_visual|, two entropies, two confidences,
        # disagreement, and two top-2 margins.
        gate_dim = hidden_dim * 4 + num_labels * 2 + 6 + num_labels * 3 + 7
        sufficiency_dim = hidden_dim * 3 + num_labels * 2 + 2
        self.sufficiency = nn.Sequential(
            nn.LayerNorm(sufficiency_dim),
            nn.Linear(sufficiency_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.visual_residual = nn.Sequential(
            nn.LayerNorm(verdict_dim),
            nn.Linear(verdict_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
        self.visual_gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Begin as a conservative text-first adapter. The residual starts at
        # zero and the gate prior is 0.1, so loading a frozen text verifier
        # exactly preserves its predictions before multimodal training.
        nn.init.zeros_(self.visual_residual[-1].weight)
        nn.init.zeros_(self.visual_residual[-1].bias)
        nn.init.constant_(self.visual_gate[-1].bias, -2.1972246)

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
    def _masked_softmax(logits: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        probabilities = torch.softmax(logits.masked_fill(~mask, -1e4), dim=1)
        probabilities = probabilities * mask.float()
        return probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _normalized_entropy(attention: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        entropy = -torch.sum(
            attention * torch.log(attention.clamp_min(1e-8)), dim=1
        )
        denominator = torch.log(mask.float().sum(1).clamp_min(2.0))
        return entropy / denominator

    @staticmethod
    def _modality_stance(stance: torch.Tensor, attention: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        probability = torch.softmax(stance, dim=-1)
        weighted = torch.sum(
            attention.unsqueeze(-1) * probability * mask.unsqueeze(-1).float(),
            dim=1,
        )
        return weighted

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
        # Normalize candidates inside each modality. A single joint softmax
        # gives the modality with more candidates a spurious prior (32 visual
        # candidates versus 8 text candidates yielded ~0.8 visual mass).
        text_attention = self._masked_softmax(text_utility, text_mask)
        visual_attention = self._masked_softmax(visual_utility, visual_mask)
        text_summary = self._masked_summary(
            text, text_attention, text_mask.bool()
        )
        visual_summary = self._masked_summary(
            visual, visual_attention, visual_mask.bool()
        )
        text_stance_logits = self.text_stance(text_pair)
        visual_stance_logits = self.visual_stance(visual_pair)
        text_stance = self._modality_stance(
            text_stance_logits, text_attention, text_mask.bool()
        )
        visual_stance = self._modality_stance(
            visual_stance_logits, visual_attention, visual_mask.bool()
        )
        conflict = torch.abs(text_stance - visual_stance)
        both_modalities = (text_mask.any(1) & visual_mask.any(1)).unsqueeze(-1)
        conflict = conflict * both_modalities.float()
        text_sufficiency_input = torch.cat((q, text_summary, text_stance), dim=-1)
        text_sufficiency_logit = self.text_sufficiency(
            text_sufficiency_input
        ).squeeze(-1)
        text_verdict_input = torch.cat(
            (
                q,
                text_summary,
                torch.abs(q - text_summary),
                q * text_summary,
                text_stance,
                torch.sigmoid(text_sufficiency_logit).unsqueeze(-1),
            ),
            dim=-1,
        )
        text_verdict_logits = self.text_verdict(text_verdict_input)
        text_entropy = self._normalized_entropy(text_attention, text_mask)
        visual_entropy = self._normalized_entropy(visual_attention, visual_mask)
        visual_quality = torch.sum(
            visual_attention * visual_retrieval_features[..., 1].float(), dim=1
        )
        gate_statistics = torch.stack(
            (
                text_entropy,
                visual_entropy,
                text_mask.any(1).float(),
                visual_mask.any(1).float(),
                visual_quality,
                visual_attention.max(1).values,
            ),
            dim=-1,
        )
        has_text = text_mask.any(1).float()
        has_visual = visual_mask.any(1).float()
        available_mass = torch.stack((has_text, has_visual), dim=-1)
        expert_mass = available_mass / available_mass.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        aggregate_stance = (
            text_stance * expert_mass[:, :1]
            + visual_stance * expert_mass[:, 1:]
        )
        sufficiency_input = torch.cat(
            (
                q, text_summary, visual_summary, aggregate_stance, conflict,
                expert_mass,
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
                expert_mass,
                torch.sigmoid(sufficiency_logit).unsqueeze(-1),
            ),
            dim=-1,
        )
        visual_residual = self.visual_residual(verdict_input)
        visual_expert_logits = text_verdict_logits + visual_residual
        text_probability = torch.softmax(text_verdict_logits.float(), dim=-1)
        expert_probability = torch.softmax(visual_expert_logits.float(), dim=-1)
        text_top2 = torch.topk(text_probability, k=2, dim=-1).values
        expert_top2 = torch.topk(expert_probability, k=2, dim=-1).values
        decision_statistics = torch.cat(
            (
                text_probability,
                expert_probability,
                torch.abs(text_probability - expert_probability),
                -torch.sum(
                    text_probability * torch.log(text_probability.clamp_min(1e-8)),
                    dim=-1,
                    keepdim=True,
                ),
                -torch.sum(
                    expert_probability
                    * torch.log(expert_probability.clamp_min(1e-8)),
                    dim=-1,
                    keepdim=True,
                ),
                text_probability.max(-1, keepdim=True).values,
                expert_probability.max(-1, keepdim=True).values,
                (text_probability.argmax(-1) != expert_probability.argmax(-1))
                .float().unsqueeze(-1),
                text_top2[:, :1] - text_top2[:, 1:2],
                expert_top2[:, :1] - expert_top2[:, 1:2],
            ),
            dim=-1,
        )
        gate_input = torch.cat(
            (
                q,
                text_summary,
                visual_summary,
                torch.abs(text_summary - visual_summary),
                text_stance,
                visual_stance,
                gate_statistics,
                decision_statistics,
            ),
            dim=-1,
        )
        visual_gate = torch.sigmoid(self.visual_gate(gate_input).squeeze(-1))
        visual_gate = visual_gate * has_visual
        modality_mass = torch.stack((1.0 - visual_gate, visual_gate), dim=-1)
        verdict_logits = (
            text_verdict_logits + visual_gate.unsqueeze(-1) * visual_residual
        )
        return {
            "verdict_logits": verdict_logits,
            "text_verdict_logits": text_verdict_logits,
            "visual_residual_logits": visual_residual,
            "visual_expert_logits": visual_expert_logits,
            "visual_gate": visual_gate,
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
    del weights  # Qrels are incomplete; listwise positives are safer than BCE negatives.
    mask = mask.bool()
    positive = relevance.bool() & mask
    rows = positive.any(1)
    if not rows.any():
        return logits.new_zeros(())
    masked_logits = logits.float().masked_fill(~mask, -1e4)
    positive_logits = masked_logits.masked_fill(~positive, -1e4)
    # Maximize the probability mass assigned to any annotated positive. This
    # avoids the 31:1 negative imbalance that collapsed visual Select@1.
    losses = -(
        torch.logsumexp(positive_logits, dim=1)
        - torch.logsumexp(masked_logits, dim=1)
    )
    return losses[rows].mean()


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
    gate_weight: float = 0.10,
    visual_gate_target: float = 0.25,
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
    visual_available = (visual_relevance.bool() & visual_mask.bool()).any(1).float()
    gate_target = visual_available * visual_gate_target
    gate = F.binary_cross_entropy(
        outputs["visual_gate"].float().clamp(1e-6, 1.0 - 1e-6), gate_target
    )
    relevance = 0.5 * (text_relevance_loss + visual_relevance_loss)
    stance = 0.5 * (text_stance + visual_stance)
    total = (
        verdict
        + relevance_weight * relevance
        + stance_weight * stance
        + sufficiency_weight * sufficiency
        + conflict_weight * conflict
        + gate_weight * gate
    )
    return total, {
        "verdict": verdict.detach(),
        "relevance": relevance.detach(),
        "stance": stance.detach(),
        "sufficiency": sufficiency.detach(),
        "conflict": conflict.detach(),
        "gate": gate.detach(),
    }
