"""Phase B18-B: Adaptive Sentence & Passage Selection.

Provides:
1. Training pair extraction from teacher-grounded explanations.
2. Adaptive evidence pruning policy with score thresholding and relative margin.
3. Ranking and filtering utilities for retrieval candidate lists.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class AdaptiveSelectorPolicy:
    """Selection policy for pruning retrieved evidence passages."""

    min_k: int = 1
    max_k: int = 3
    score_threshold: float = -1.0
    adaptive_margin: float = 1.5

    def select(
        self, candidate_ids: Sequence[str], scores: Sequence[float]
    ) -> tuple[list[str], list[float]]:
        """Filter candidates descending by score with margin and threshold constraints."""
        if not candidate_ids or not scores:
            return [], []

        if len(candidate_ids) != len(scores):
            raise ValueError(
                f"Length mismatch: {len(candidate_ids)} candidates vs {len(scores)} scores."
            )

        order = np.argsort(-np.asarray(scores, dtype=np.float32)).tolist()
        top_score = float(scores[order[0]])

        selected_ids: list[str] = []
        selected_scores: list[float] = []

        for rank, idx in enumerate(order):
            cid = candidate_ids[idx]
            sc = float(scores[idx])

            # Always retain up to min_k
            if len(selected_ids) < self.min_k:
                selected_ids.append(cid)
                selected_scores.append(sc)
                continue

            # Respect max_k ceiling
            if len(selected_ids) >= self.max_k:
                break

            # Check threshold and relative margin
            passes_threshold = sc >= self.score_threshold
            passes_margin = (top_score - sc) <= self.adaptive_margin

            if passes_threshold and passes_margin:
                selected_ids.append(cid)
                selected_scores.append(sc)
            else:
                # Since candidates are sorted descending, further candidates have even lower scores
                break

        return selected_ids, selected_scores


def extract_selector_pairs(
    explanations: list[dict[str, Any]],
    documents: dict[str, str],
    max_neg_per_grounded: int = 4,
    sample_ungrounded_negatives: bool = True,
    max_neg_per_ungrounded: int = 2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Construct (claim, passage, label) training pairs from teacher rationales."""
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    for row in explanations:
        cid = str(row.get("id", ""))
        claim_text = str(row.get("claim", "")).strip()
        if not claim_text:
            continue

        retrieved_ids = row.get("retrieved_evidence_ids", [])
        is_valid = bool(row.get("is_valid", False))
        grounded = bool(row.get("grounded", False))
        key_indices = set(row.get("key_evidence_ids", []))

        if is_valid and grounded:
            # Positive and hard negative extraction
            claim_negs: list[dict[str, Any]] = []
            for idx_1based, eid in enumerate(retrieved_ids, 1):
                doc_text = documents.get(eid, "").strip()
                if not doc_text:
                    continue

                if idx_1based in key_indices:
                    pairs.append({
                        "claim_id": cid,
                        "claim": claim_text,
                        "evidence_id": eid,
                        "text": doc_text,
                        "label": 1.0,
                        "is_hard_negative": False,
                    })
                else:
                    claim_negs.append({
                        "claim_id": cid,
                        "claim": claim_text,
                        "evidence_id": eid,
                        "text": doc_text,
                        "label": 0.0,
                        "is_hard_negative": True,
                    })

            if len(claim_negs) > max_neg_per_grounded:
                claim_negs = rng.sample(claim_negs, max_neg_per_grounded)
            pairs.extend(claim_negs)

        elif sample_ungrounded_negatives:
            # Teacher verified that none of the retrieved evidence grounds the claim
            ungrounded_negs: list[dict[str, Any]] = []
            for eid in retrieved_ids:
                doc_text = documents.get(eid, "").strip()
                if not doc_text:
                    continue
                ungrounded_negs.append({
                    "claim_id": cid,
                    "claim": claim_text,
                    "evidence_id": eid,
                    "text": doc_text,
                    "label": 0.0,
                    "is_hard_negative": False,
                })
            if len(ungrounded_negs) > max_neg_per_ungrounded:
                ungrounded_negs = rng.sample(ungrounded_negs, max_neg_per_ungrounded)
            pairs.extend(ungrounded_negs)

    return pairs


def mock_score_claim_evidence_pairs(
    pairs: Sequence[tuple[str, str]],
) -> list[float]:
    """Deterministic, fast heuristic scoring for offline tests without PyTorch/HuggingFace."""
    scores = []
    for claim, doc in pairs:
        claim_words = set(claim.casefold().split())
        doc_words = set(doc.casefold().split())
        if not claim_words:
            scores.append(0.0)
            continue
        overlap = len(claim_words & doc_words)
        score = float(overlap) / float(max(1, len(claim_words)))
        scores.append(score)
    return scores
