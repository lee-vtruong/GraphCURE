from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class AcquisitionDecision:
    action: int | None
    utilities: torch.Tensor
    should_stop: bool


def bayes_risk(verdict_prob: torch.Tensor) -> torch.Tensor:
    """Bayes risk under zero-one loss."""
    return 1.0 - verdict_prob.max(dim=-1).values


def choose_evi_action(
    current_prob: torch.Tensor,
    outcome_probabilities: torch.Tensor,
    posterior_if_outcome: torch.Tensor,
    action_costs: torch.Tensor,
    cost_weight: float,
) -> AcquisitionDecision:
    """One-step expected value of information.

    outcome_probabilities: [actions, outcomes]
    posterior_if_outcome: [actions, outcomes, labels]
    """
    current_risk = bayes_risk(current_prob).squeeze(0)
    future_risk = bayes_risk(posterior_if_outcome)
    expected_risk = (outcome_probabilities * future_risk).sum(dim=-1)
    utilities = current_risk - expected_risk - cost_weight * action_costs
    best = int(utilities.argmax().item())
    stop = bool(utilities[best].item() <= 0)
    return AcquisitionDecision(None if stop else best, utilities, stop)

