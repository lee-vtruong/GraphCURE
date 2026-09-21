"""Unit tests for Dual-Expert Routing and Selective Deferral policies."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_mocheg_expert_routing import (
    apply_confidence_routing,
    apply_deferral,
)


def test_apply_deferral_routes_to_nei_only_when_condition_met():
    # Classes: 0: Supported, 1: Refuted, 2: NEI
    b1_preds = np.array([0, 1, 0, 1])
    b18_preds = np.array([2, 2, 1, 2])
    # Case 0: B18 predicts NEI (2) with prob 0.65 >= 0.60 -> route to 2 (NEI)
    # Case 1: B18 predicts NEI (2) with prob 0.55 < 0.60 -> keep B1 prediction (1)
    # Case 2: B18 predicts Refuted (1) with prob 0.70 -> keep B1 prediction (0)
    # Case 3: B18 predicts NEI (2) with prob 0.60 >= 0.60 -> route to 2 (NEI)
    b18_probs = np.array([
        [0.15, 0.20, 0.65],
        [0.20, 0.25, 0.55],
        [0.10, 0.70, 0.20],
        [0.10, 0.30, 0.60],
    ])
    routed = apply_deferral(b1_preds, b18_preds, b18_probs, tau=0.60)
    expected = np.array([2, 1, 0, 2])
    assert np.array_equal(routed, expected)


def test_apply_confidence_routing_defers_when_b1_uncertain():
    b1_preds = np.array([0, 1, 0])
    b18_preds = np.array([1, 2, 0])
    # B1 confidence: max(probs)
    # Case 0: max prob = 0.50 < 0.65 -> defer to B18 (1)
    # Case 1: max prob = 0.80 >= 0.65 -> keep B1 (1)
    # Case 2: max prob = 0.64 < 0.65 -> defer to B18 (0)
    b1_probs = np.array([
        [0.50, 0.30, 0.20],
        [0.10, 0.80, 0.10],
        [0.64, 0.20, 0.16],
    ])
    routed = apply_confidence_routing(b1_preds, b18_preds, b1_probs, gamma=0.65)
    expected = np.array([1, 1, 0])
    assert np.array_equal(routed, expected)
