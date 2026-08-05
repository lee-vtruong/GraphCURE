from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


CONSTRAINTS = ("semantic", "entity", "temporal", "contextual")
EDGES = ((0, 3), (1, 3), (2, 3), (1, 0), (2, 0))
FULL_EDGES = tuple((src, dst) for src in range(4) for dst in range(4) if src != dst)


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
    architecture: str = "typed_graph"


class TypedGraphLayer(nn.Module):
    def __init__(self, dim: int, dropout: float, edges: tuple[tuple[int, int], ...]) -> None:
        super().__init__()
        self.edges = edges
        self.edge_gate = nn.ModuleList([nn.Linear(dim * 2, 1) for _ in edges])
        self.messages = nn.ModuleList([nn.Linear(dim, dim) for _ in edges])
        self.update = nn.GRUCell(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_nodes, dim = nodes.shape
        incoming = torch.zeros_like(nodes)
        gates = []
        for edge_id, (src, dst) in enumerate(self.edges):
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
        supported = {"linear", "mlp", "independent", "fully_connected", "typed_graph"}
        if cfg.architecture not in supported:
            raise ValueError(f"Unknown architecture {cfg.architecture!r}; choose from {sorted(supported)}")
        fused_dim = cfg.text_dim + cfg.vision_dim + cfg.metadata_dim
        self.direct_verdict: nn.Module | None = None
        if cfg.architecture == "linear":
            self.direct_verdict = nn.Linear(fused_dim, cfg.num_labels)
        elif cfg.architecture == "mlp":
            self.direct_verdict = nn.Sequential(
                nn.Linear(fused_dim, cfg.hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.num_labels),
            )
        self.edges = FULL_EDGES if cfg.architecture == "fully_connected" else EDGES
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
            [TypedGraphLayer(cfg.hidden_dim, cfg.dropout, self.edges) for _ in range(cfg.graph_layers)]
            if cfg.architecture in {"typed_graph", "fully_connected"}
            else []
        )
        self.state_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.num_states) for _ in CONSTRAINTS]
        )
        self.evidence_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.num_states) for _ in CONSTRAINTS]
        )
        # One compatibility tensor per typed edge. Sigmoid keeps it auditable.
        self.compatibility_logits = nn.Parameter(
            torch.zeros(len(self.edges), cfg.num_states, cfg.num_states)
        )
        verdict_dim = cfg.hidden_dim * 4 + 4 * cfg.num_states + len(self.edges)
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
        if self.direct_verdict is not None:
            batch = fused.size(0)
            zeros = fused.new_zeros
            # Preserve the common output contract; auxiliary losses must remain
            # disabled for direct baselines.
            return {
                "verdict_logits": self.direct_verdict(fused),
                "constraint_logits": zeros(batch, 4, self.cfg.num_states),
                "constraint_prob": zeros(batch, 4, self.cfg.num_states),
                "uncertainty": zeros(batch, 4),
                "conflict": zeros(batch, len(self.edges)),
                "edge_gates": zeros(batch, 0, len(self.edges)),
                "nodes": zeros(batch, 4, self.cfg.hidden_dim),
            }
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
        for edge_id, (src, dst) in enumerate(self.edges):
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
            "edge_gates": (
                torch.stack(all_gates, dim=1)
                if all_gates else nodes.new_zeros(nodes.size(0), 0, len(self.edges))
            ),
            "nodes": nodes,
        }
