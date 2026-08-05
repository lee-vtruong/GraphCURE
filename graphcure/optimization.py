from __future__ import annotations

import torch


def project_auxiliary_gradients(
    primary: tuple[torch.Tensor | None, ...],
    auxiliary: tuple[torch.Tensor | None, ...],
    eps: float = 1e-12,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Protect the primary task by removing conflicting auxiliary components."""
    pairs = [(p, a) for p, a in zip(primary, auxiliary) if p is not None and a is not None]
    if not pairs:
        return [p if a is None else a for p, a in zip(primary, auxiliary)], {
            "cosine": 0.0, "conflict": 0.0
        }
    dot = sum((p * a).sum() for p, a in pairs)
    p_norm_sq = sum((p * p).sum() for p, _ in pairs)
    a_norm_sq = sum((a * a).sum() for _, a in pairs)
    cosine = dot / (p_norm_sq.sqrt() * a_norm_sq.sqrt() + eps)
    conflict = bool(dot.detach().item() < 0)
    coefficient = dot / (p_norm_sq + eps) if conflict else dot.new_zeros(())
    combined: list[torch.Tensor | None] = []
    for p, a in zip(primary, auxiliary):
        if p is None:
            combined.append(a)
        elif a is None:
            combined.append(p)
        else:
            combined.append(p + a - coefficient * p)
    return combined, {"cosine": float(cosine.detach()), "conflict": float(conflict)}
