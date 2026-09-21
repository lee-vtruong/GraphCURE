"""Unit tests for validation-driven ensemble selection."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.select_mocheg_ensemble_on_val import (
    evaluate_ensemble_on_data,
    extract_ordered_data,
    locked_test_policy_names,
)


def test_extract_ordered_data():
    dict1 = {
        "c1": {"label": 0, "probabilities": [0.8, 0.1, 0.1]},
        "c2": {"label": 1, "probabilities": [0.1, 0.8, 0.1]},
    }
    dict2 = {
        "c1": {"label": 0, "probabilities": [0.7, 0.2, 0.1]},
        "c2": {"label": 1, "probabilities": [0.2, 0.7, 0.1]},
    }
    ordered_ids, y_true = extract_ordered_data([dict1, dict2])
    assert ordered_ids == ["c1", "c2"]
    assert np.array_equal(y_true, [0, 1])


def test_evaluate_ensemble_on_data():
    dict1 = {
        "c1": {"label": 0, "probabilities": [0.8, 0.1, 0.1]},
        "c2": {"label": 1, "probabilities": [0.1, 0.8, 0.1]},
    }
    dict2 = {
        "c1": {"label": 0, "probabilities": [0.7, 0.2, 0.1]},
        "c2": {"label": 1, "probabilities": [0.2, 0.7, 0.1]},
    }
    metrics = evaluate_ensemble_on_data([dict1, dict2], ["c1", "c2"], np.array([0, 1]))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["predictions"].tolist() == [0, 1]


def test_locked_test_scope_excludes_nonchampion_top_k_policies():
    assert locked_test_policy_names("top_3_val_candidates") == [
        "top_3_val_candidates",
        "full_unpruned_ensemble",
    ]
    assert locked_test_policy_names("full_unpruned_ensemble") == [
        "full_unpruned_ensemble"
    ]
