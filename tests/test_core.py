import torch

from graphcure.acquisition import choose_evi_action
from graphcure.losses import counterfactual_loss
from graphcure.model import GraphCURE, GraphCUREConfig


def test_model_shapes():
    model = GraphCURE(
        GraphCUREConfig(text_dim=8, vision_dim=8, metadata_dim=4, hidden_dim=16)
    )
    out = model(torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 4))
    assert out["verdict_logits"].shape == (2, 3)
    assert out["constraint_prob"].shape == (2, 4, 3)
    assert out["conflict"].shape == (2, 5)


def test_counterfactual_loss_is_finite():
    p = torch.softmax(torch.randn(3, 4, 3), -1)
    q = torch.softmax(torch.randn(3, 4, 3), -1)
    mask = torch.tensor([[0, 0, 1, 1]] * 3).bool()
    assert torch.isfinite(counterfactual_loss(p, q, mask))


def test_evi_stops_when_every_action_is_costly():
    current = torch.tensor([[0.8, 0.1, 0.1]])
    outcome = torch.full((2, 2), 0.5)
    posterior = torch.tensor(
        [[[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]] * 2
    )
    decision = choose_evi_action(current, outcome, posterior, torch.ones(2), 1.0)
    assert decision.should_stop

