"""Retrieval utilities shared by the modern GraphCURE evidence pipeline."""
from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np


_NEGATIONS = re.compile(
    r"\b(?:no|not|never|neither|nor|without|false|fake|hoax|deny|denied|"
    r"didn't|doesn't|isn't|wasn't|weren't|cannot|can't)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    weights: Sequence[float] | None = None,
    rank_constant: float = 60.0,
    limit: int | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked integer document IDs without requiring calibrated scores."""
    if not rankings:
        return []
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[int] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            doc_id = int(doc_id)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            fused[doc_id] = fused.get(doc_id, 0.0) + float(weight) / (
                rank_constant + rank
            )
    result = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return result if limit is None else result[:limit]


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return descending top-k indices without sorting the entire vector."""
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    k = min(max(int(k), 0), len(scores))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    candidate = np.argpartition(-scores, k - 1)[:k]
    return candidate[np.argsort(-scores[candidate], kind="stable")]


def first_relevant_rank(retrieved: Sequence[str], relevant: set[str]) -> int | None:
    return next(
        (rank for rank, evidence_id in enumerate(retrieved, start=1)
         if evidence_id in relevant),
        None,
    )


def contradiction_features(claim: str, evidence: str) -> dict[str, float | bool]:
    """Cheap diagnostics used to stratify reasoning-oriented hard negatives."""
    claim_tokens = {token.lower() for token in _TOKEN.findall(claim)}
    evidence_tokens = {token.lower() for token in _TOKEN.findall(evidence)}
    union = claim_tokens | evidence_tokens
    overlap = len(claim_tokens & evidence_tokens) / max(1, len(union))
    claim_numbers = set(_NUMBER.findall(claim))
    evidence_numbers = set(_NUMBER.findall(evidence))
    return {
        "lexical_jaccard": float(overlap),
        "negation_mismatch": bool(_NEGATIONS.search(claim))
        != bool(_NEGATIONS.search(evidence)),
        "number_mismatch": bool(claim_numbers and evidence_numbers)
        and claim_numbers.isdisjoint(evidence_numbers),
    }


def retrieval_confidence(scores: Sequence[float]) -> float:
    """Bounded confidence based on the top-score margin, not a raw model logit."""
    if not scores:
        return 0.0
    if len(scores) == 1:
        margin = float(scores[0])
    else:
        margin = float(scores[0]) - float(scores[1])
    return float(1.0 / (1.0 + math.exp(-margin)))
