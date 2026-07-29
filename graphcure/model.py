from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


CONSTRAINTS = ("semantic", "entity", "temporal", "contextual")
EDGES = ((0, 3), (1, 3), (2, 3), (1, 0), (2, 0))


@dataclass
class GraphCUREConfig:
    text_dim: int = 768
    vision_dim: int = 768
    metadata_dim: int = 16
    hidden_dim: int = 256
    num_states: int = 3
    num_labels: int = 3
    graph_layers: int = 2
    dropout: float = 0.1


class TypedGraphLayer(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_gate = nn.ModuleList([nn.Linear(dim * 2, 1) for _ in EDGES])
        self.messages = nn.ModuleList([nn.Linear(dim, dim) for _ in EDGES])
        self.update = nn.GRUCell(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_nodes, dim = nodes.shape
        incoming = torch.zeros_like(nodes)
        gates = []
        for edge_id, (src, dst) in enumerate(EDGES):
            pair = torch.cat([nodes[:, src], nodes[:, dst]], dim=-1)
            gate = torch.sigmoid(self.edge_gate[edge_id](pair))
            incoming[:, dst] += gate * self.messages[edge_id](nodes[:, src])
            gates.append(gate.squeeze(-1))
        updated = self.update(
            incoming.reshape(batch * n_nodes, dim),
            nodes.reshape(batch * n_nodes, dim),
        ).reshape(batch, n_nodes, dim)
        return self.norm(nodes + self.dropout(updated)), torch.stack(gates, dim=1)


class GraphCURE(nn.Module):
    """Core model operating on precomputed text/image/metadata embeddings.

    Encoders remain separate so large backbone embeddings can be cached and
    ablations can compare identical representations.
    """

    def __init__(self, cfg: GraphCUREConfig) -> None:
        super().__init__()
        self.cfg = cfg
        fused_dim = cfg.text_dim + cfg.vision_dim + cfg.metadata_dim
        self.constraint_tokens = nn.Parameter(torch.randn(4, cfg.hidden_dim) * 0.02)
        self.initializers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim + cfg.hidden_dim, cfg.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.LayerNorm(cfg.hidden_dim),
                )
                for _ in CONSTRAINTS
            ]
        )
        self.graph = nn.ModuleList(
            [TypedGraphLayer(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.graph_layers)]
        )
        self.state_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.num_states) for _ in CONSTRAINTS]
        )
        self.evidence_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.num_states) for _ in CONSTRAINTS]
        )
        # One compatibility tensor per typed edge. Sigmoid keeps it auditable.
        self.compatibility_logits = nn.Parameter(
            torch.zeros(len(EDGES), cfg.num_states, cfg.num_states)
        )
        verdict_dim = cfg.hidden_dim * 4 + 4 * cfg.num_states + len(EDGES)
        self.verdict = nn.Sequential(
            nn.Linear(verdict_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_labels),
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if metadata is None:
            metadata = text_embedding.new_zeros(
                text_embedding.shape[0], self.cfg.metadata_dim
            )
        fused = torch.cat([text_embedding, image_embedding, metadata], dim=-1)
        nodes = torch.stack(
            [
                init(torch.cat([fused, self.constraint_tokens[i].expand(fused.size(0), -1)], -1))
                for i, init in enumerate(self.initializers)
            ],
            dim=1,
        )
        all_gates = []
        for layer in self.graph:
            nodes, gates = layer(nodes)
            all_gates.append(gates)
        state_logits = torch.stack(
            [head(nodes[:, i]) for i, head in enumerate(self.state_heads)], dim=1
        )
        evidence = torch.stack(
            [F.softplus(head(nodes[:, i])) + 1.0 for i, head in enumerate(self.evidence_heads)],
            dim=1,
        )
        uncertainty = self.cfg.num_states / evidence.sum(dim=-1)
        state_prob = state_logits.softmax(dim=-1)
        compatibility = self.compatibility_logits.sigmoid()
        conflicts = []
        for edge_id, (src, dst) in enumerate(EDGES):
            joint = state_prob[:, src, :, None] * state_prob[:, dst, None, :]
            conflicts.append((joint * compatibility[edge_id]).sum(dim=(1, 2)))
        conflict = torch.stack(conflicts, dim=1)
        verdict_features = torch.cat(
            [nodes.flatten(1), state_prob.flatten(1), conflict], dim=1
        )
        return {
            "verdict_logits": self.verdict(verdict_features),
            "constraint_logits": state_logits,
            "constraint_prob": state_prob,
            "uncertainty": uncertainty,
            "conflict": conflict,
            "edge_gates": torch.stack(all_gates, dim=1),
            "nodes": nodes,
        }

