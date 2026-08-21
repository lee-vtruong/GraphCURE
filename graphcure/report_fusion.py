"""Safe residual fusion for frozen text and visual-report experts."""
from __future__ import annotations

import torch
from torch import nn


def fusion_features(output: dict[str, torch.Tensor]) -> torch.Tensor:
    text = torch.softmax(output["text_verdict_logits"].float(), -1)
    visual = torch.softmax(output["visual_expert_logits"].float(), -1)
    text_top2 = torch.topk(text, 2, dim=-1).values
    visual_top2 = torch.topk(visual, 2, dim=-1).values
    attention = output["visual_attention"].float().clamp_min(1e-8)
    entropy = -torch.sum(attention * torch.log(attention), -1, keepdim=True)
    return torch.cat((
        text, visual, torch.abs(text - visual), output["conflict"].float(),
        torch.sigmoid(output["sufficiency_logit"].float()).unsqueeze(-1),
        entropy,
        text_top2[:, :1] - text_top2[:, 1:2],
        visual_top2[:, :1] - visual_top2[:, 1:2],
        (text.argmax(-1) != visual.argmax(-1)).float().unsqueeze(-1),
    ), dim=-1)


class SafeReportFusion(nn.Module):
    """Convex logit fusion initialized near the frozen text anchor."""
    def __init__(self, feature_dim: int, hidden_dim: int = 64,
                 initial_route_rate: float = 0.05) -> None:
        super().__init__()
        self.visual_calibrator = nn.Linear(3, 3)
        nn.init.eye_(self.visual_calibrator.weight)
        nn.init.zeros_(self.visual_calibrator.bias)
        self.gate = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.gate[-1].weight)
        bias = torch.logit(torch.tensor(initial_route_rate)).item()
        nn.init.constant_(self.gate[-1].bias, bias)

    def forward(self, text_logits: torch.Tensor, visual_logits: torch.Tensor,
                features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate(features.float()).squeeze(-1))
        visual = self.visual_calibrator(visual_logits.float())
        logits = text_logits.float() + gate.unsqueeze(-1) * (
            visual - text_logits.float())
        return logits, gate
