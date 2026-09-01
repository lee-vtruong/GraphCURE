from __future__ import annotations

import torch


def project_auxiliary_gradients(
    primary: tuple[torch.Tensor | None, ...],
    auxiliary: tuple[torch.Tensor | None, ...],
    projection_strength: float = 1.0,
    conflict_temperature: float | None = None,
    eps: float = 1e-12,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Protect the primary task with fixed or conflict-severity projection.

    ``projection_strength=1`` and ``conflict_temperature=None`` reproduce full
    PCGrad.  A strength in ``[0, 1]`` removes only that fraction of the
    conflicting auxiliary component.  When ``conflict_temperature`` is set,
    the fraction is additionally scaled by ``min(1, -cosine / temperature)``;
    weak conflicts are therefore changed less than severe conflicts.
    """
    if not 0 <= projection_strength <= 1:
        raise ValueError("projection_strength must be in [0, 1]")
    if conflict_temperature is not None and conflict_temperature <= 0:
        raise ValueError("conflict_temperature must be positive")
    pairs = [(p, a) for p, a in zip(primary, auxiliary) if p is not None and a is not None]
    if not pairs:
        return [p if a is None else a for p, a in zip(primary, auxiliary)], {
            "cosine": 0.0, "conflict": 0.0,
            "applied_projection_strength": 0.0,
        }
    dot = sum((p * a).sum() for p, a in pairs)
    p_norm_sq = sum((p * p).sum() for p, _ in pairs)
    a_norm_sq = sum((a * a).sum() for _, a in pairs)
    cosine = dot / (p_norm_sq.sqrt() * a_norm_sq.sqrt() + eps)
    conflict = bool(dot.detach().item() < 0)
    coefficient = dot / (p_norm_sq + eps) if conflict else dot.new_zeros(())
    applied_strength = projection_strength if conflict else 0.0
    if conflict and conflict_temperature is not None:
        severity = min(
            1.0, max(0.0, -float(cosine.detach()) / conflict_temperature)
        )
        applied_strength *= severity
    combined: list[torch.Tensor | None] = []
    for p, a in zip(primary, auxiliary):
        if p is None:
            combined.append(a)
        elif a is None:
            combined.append(p)
        else:
            combined.append(p + a - applied_strength * coefficient * p)
    return combined, {
        "cosine": float(cosine.detach()),
        "conflict": float(conflict),
        "applied_projection_strength": float(applied_strength),
    }
