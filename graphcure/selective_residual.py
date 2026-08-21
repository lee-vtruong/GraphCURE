"""Anchor-preserving conflict-aware residual reasoning over evidence sets."""
from __future__ import annotations

import torch
from torch import nn

from graphcure.evidence_set import EvidenceSetHead


class SelectiveResidualSetVerifier(nn.Module):
    """Add bounded set-level corrections to a frozen evidence verifier."""

    def __init__(self, anchor: EvidenceSetHead, encoder_dim: int,
                 hidden_dim: int = 192, retrieval_dim: int = 6,
                 layers: int = 2, heads: int = 4, dropout: float = 0.1,
                 residual_scale: float = 2.0, gate_bias: float = -2.2) -> None:
        super().__init__()
        self.anchor = anchor
        self.residual_scale = float(residual_scale)
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self.anchor.eval()
        self.claim_projection = nn.Sequential(
            nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, hidden_dim), nn.GELU()
        )
        pair_dim = encoder_dim * 4 + retrieval_dim + 4
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 3,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.interactions = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool_score = nn.Linear(hidden_dim, 1)
        # pooled set + claim + anchor probabilities + stance disagreement +
        # entropy + top-two margin + anchor sufficiency.
        global_dim = hidden_dim * 2 + 9
        self.residual = nn.Sequential(
            nn.LayerNorm(global_dim), nn.Linear(global_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 3),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(global_dim), nn.Linear(global_dim, hidden_dim // 2),
            nn.GELU(), nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(self, claim: torch.Tensor, evidence: torch.Tensor,
                evidence_mask: torch.Tensor,
                retrieval_features: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = evidence_mask.bool()
        with torch.no_grad():
            anchor = self.anchor(claim, evidence, mask, retrieval_features)
        claim_f, evidence_f = claim.float(), evidence.float()
        repeated = claim_f.unsqueeze(1).expand_as(evidence_f)
        stance_prob = torch.softmax(anchor["stance_logits"].float(), -1)
        pair = torch.cat((
            repeated, evidence_f, torch.abs(repeated - evidence_f),
            repeated * evidence_f, retrieval_features.float(), stance_prob,
            anchor["attention"].float().unsqueeze(-1),
        ), -1)
        tokens = self.pair_projection(pair)
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        tokens = self.interactions(tokens, src_key_padding_mask=~safe_mask)
        scores = self.pool_score(tokens).squeeze(-1)
        scores += torch.log(anchor["attention"].float().clamp_min(1e-6))
        scores = scores.masked_fill(~safe_mask, -1e4)
        set_attention = torch.softmax(scores, -1)
        pooled = torch.sum(set_attention.unsqueeze(-1) * tokens, dim=1)
        anchor_prob = torch.softmax(anchor["verdict_logits"].float(), -1)
        entropy = -(anchor_prob * anchor_prob.clamp_min(1e-8).log()).sum(-1, keepdim=True)
        top_two = anchor_prob.topk(2, dim=-1).values
        margin = top_two[:, :1] - top_two[:, 1:2]
        anchor_attention = anchor["attention"].float()
        stance_mean = torch.sum(anchor_attention.unsqueeze(-1) * stance_prob, dim=1)
        disagreement = torch.sum(
            anchor_attention.unsqueeze(-1) * (stance_prob - stance_mean[:, None]) ** 2,
            dim=1,
        )
        features = torch.cat((
            pooled, self.claim_projection(claim_f), anchor_prob, disagreement,
            entropy, margin,
            torch.sigmoid(anchor["sufficiency_logit"].float()).unsqueeze(-1),
        ), -1)
        residual = self.residual(features)
        gate = torch.sigmoid(self.gate(features)).squeeze(-1)
        logits = anchor["verdict_logits"].float() + (
            self.residual_scale * gate.unsqueeze(-1) * torch.tanh(residual)
        )
        return {
            "verdict_logits": logits,
            "anchor_logits": anchor["verdict_logits"].float(),
            "residual_logits": residual,
            "residual_gate": gate,
            "set_attention": set_attention,
            "anchor_attention": anchor_attention,
            "stance_disagreement": disagreement,
        }
