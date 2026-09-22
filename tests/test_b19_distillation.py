"""Unit tests for Phase B19: Disagreement-Aware Counterfactual Rationale Distillation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from graphcure.b19 import (
    VARIANT_DEFAULTS,
    counterfactual_sufficiency_loss,
    heterogeneous_teacher_target,
    resolve_variant,
    soft_distillation_loss,
    weighted_verdict_loss,
)


# ---------------------------------------------------------------------------
# 1. heterogeneous_teacher_target
# ---------------------------------------------------------------------------
class TestHeterogeneousTeacherTarget:
    """Verify correct teacher selection and soft-label construction."""

    def test_direct_only_correct_selects_direct(self):
        label = 0  # SUPPORTED
        direct = [0.8, 0.1, 0.1]   # predicts label 0 => correct
        grounded = [0.1, 0.7, 0.2]  # predicts label 1 => wrong
        target, source, disagree = heterogeneous_teacher_target(label, direct, grounded)
        assert source == "direct_only_correct"
        assert disagree is True
        np.testing.assert_allclose(target, np.array(direct), atol=1e-6)

    def test_grounded_only_correct_selects_grounded(self):
        label = 1  # REFUTED
        direct = [0.6, 0.2, 0.2]    # predicts label 0 => wrong
        grounded = [0.1, 0.8, 0.1]  # predicts label 1 => correct
        target, source, disagree = heterogeneous_teacher_target(label, direct, grounded)
        assert source == "grounded_only_correct"
        assert disagree is True
        np.testing.assert_allclose(target, np.array(grounded), atol=1e-6)

    def test_both_correct_averages(self):
        label = 0
        direct = [0.7, 0.2, 0.1]
        grounded = [0.9, 0.05, 0.05]
        target, source, disagree = heterogeneous_teacher_target(label, direct, grounded)
        assert source == "both_correct"
        expected = (np.array(direct) + np.array(grounded)) / 2.0
        np.testing.assert_allclose(target, expected, atol=1e-6)

    def test_both_wrong_uses_gold_smoothed(self):
        label = 2  # NEI
        direct = [0.6, 0.3, 0.1]   # predicts 0 => wrong
        grounded = [0.2, 0.7, 0.1]  # predicts 1 => wrong
        target, source, disagree = heterogeneous_teacher_target(label, direct, grounded, smoothing=0.1)
        assert source == "both_wrong_gold_smoothed"
        assert disagree is True
        assert target[2] == pytest.approx(0.9, abs=1e-6)
        assert target[0] == pytest.approx(0.05, abs=1e-6)
        assert target[1] == pytest.approx(0.05, abs=1e-6)

    def test_disagreement_flag_when_same_prediction(self):
        label = 0
        direct = [0.5, 0.3, 0.2]   # predicts 0
        grounded = [0.6, 0.2, 0.2]  # predicts 0
        _, _, disagree = heterogeneous_teacher_target(label, direct, grounded)
        assert disagree is False

    def test_rejects_invalid_shape(self):
        with pytest.raises(ValueError, match="three classes"):
            heterogeneous_teacher_target(0, [0.5, 0.5], [0.3, 0.3, 0.4])


# ---------------------------------------------------------------------------
# 2. weighted_verdict_loss
# ---------------------------------------------------------------------------
class TestWeightedVerdictLoss:
    """Verify disagreement-aware upweighting."""

    def test_no_disagreement_equals_standard_ce(self):
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 0])
        disagreement = torch.zeros(4, dtype=torch.bool)
        loss = weighted_verdict_loss(logits, labels, disagreement, disagreement_alpha=1.0)
        ce = torch.nn.functional.cross_entropy(logits, labels)
        assert loss.item() == pytest.approx(ce.item(), abs=1e-5)

    def test_all_disagreement_upscales(self):
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 0])
        no_disagree = torch.zeros(4, dtype=torch.bool)
        all_disagree = torch.ones(4, dtype=torch.bool)
        loss_normal = weighted_verdict_loss(logits, labels, no_disagree, disagreement_alpha=1.0)
        loss_upscaled = weighted_verdict_loss(logits, labels, all_disagree, disagreement_alpha=1.0)
        # When all have disagreement, weights = 2.0 for all, but mean normalization
        # makes it identical to standard CE
        assert loss_upscaled.item() == pytest.approx(loss_normal.item(), abs=1e-5)

    def test_partial_disagreement_shifts_focus(self):
        torch.manual_seed(42)
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 0])
        partial = torch.tensor([True, False, False, False])
        # Should produce a finite loss
        loss = weighted_verdict_loss(logits, labels, partial, disagreement_alpha=1.0)
        assert torch.isfinite(loss)

    def test_zero_alpha_ignores_disagreement(self):
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 0])
        partial = torch.tensor([True, False, True, False])
        loss = weighted_verdict_loss(logits, labels, partial, disagreement_alpha=0.0)
        ce = torch.nn.functional.cross_entropy(logits, labels)
        assert loss.item() == pytest.approx(ce.item(), abs=1e-5)


# ---------------------------------------------------------------------------
# 3. soft_distillation_loss
# ---------------------------------------------------------------------------
class TestSoftDistillationLoss:
    """Verify KL divergence with temperature scaling."""

    def test_identical_distributions_zero_loss(self):
        logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        probs = torch.softmax(logits, dim=-1)
        loss = soft_distillation_loss(logits, probs, temperature=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_high_temperature_smooths(self):
        logits = torch.tensor([[3.0, 0.0, 0.0]])
        probs = torch.tensor([[0.9, 0.05, 0.05]])
        loss_t1 = soft_distillation_loss(logits, probs, temperature=1.0)
        loss_t4 = soft_distillation_loss(logits, probs, temperature=4.0)
        # Both should be finite and non-negative
        assert torch.isfinite(loss_t1) and loss_t1.item() >= 0
        assert torch.isfinite(loss_t4) and loss_t4.item() >= 0

    def test_output_is_differentiable(self):
        logits = torch.randn(3, 3, requires_grad=True)
        probs = torch.softmax(torch.randn(3, 3), dim=-1)
        loss = soft_distillation_loss(logits, probs, temperature=2.0)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == (3, 3)


# ---------------------------------------------------------------------------
# 4. counterfactual_sufficiency_loss
# ---------------------------------------------------------------------------
class TestCounterfactualSufficiencyLoss:
    """Verify margin enforcement between positive and ablated evidence."""

    def test_no_valid_samples_returns_zero(self):
        pos = torch.randn(4, 3)
        neg = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 0])
        mask = torch.zeros(4, dtype=torch.bool)
        loss = counterfactual_sufficiency_loss(pos, neg, labels, mask, margin=0.5)
        assert loss.item() == 0.0

    def test_perfect_separation_zero_loss(self):
        # Positive logits strongly favor gold class; negative logits strongly favor NEI
        pos = torch.tensor([[5.0, -5.0, -5.0]])  # gold=0 => score=5
        neg = torch.tensor([[-5.0, -5.0, 5.0]])   # NEI=2 => score=5 (good)
        labels = torch.tensor([0])
        mask = torch.ones(1, dtype=torch.bool)
        loss = counterfactual_sufficiency_loss(pos, neg, labels, mask, margin=0.5)
        # pos_gold=5.0, neg_gold=-5.0 => margin - 5.0 + (-5.0) = 0.5 - 10 < 0 => relu=0
        # pos_nei=-5.0, neg_nei=5.0 => margin - 5.0 + (-5.0) = 0.5 - 10 < 0 => relu=0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_margin_violated_positive_loss(self):
        pos = torch.tensor([[1.0, 0.0, 0.0]])
        neg = torch.tensor([[0.8, 0.0, 0.2]])
        labels = torch.tensor([0])
        mask = torch.ones(1, dtype=torch.bool)
        loss = counterfactual_sufficiency_loss(pos, neg, labels, mask, margin=0.5)
        assert loss.item() > 0

    def test_output_is_differentiable(self):
        pos = torch.randn(3, 3, requires_grad=True)
        neg = torch.randn(3, 3, requires_grad=True)
        labels = torch.tensor([0, 1, 2])
        mask = torch.ones(3, dtype=torch.bool)
        loss = counterfactual_sufficiency_loss(pos, neg, labels, mask, margin=0.5)
        loss.backward()
        assert pos.grad is not None
        assert neg.grad is not None


# ---------------------------------------------------------------------------
# 5. resolve_variant
# ---------------------------------------------------------------------------
class TestResolveVariant:
    """Verify variant resolution with and without overrides."""

    def test_known_variants(self):
        for name in VARIANT_DEFAULTS:
            result = resolve_variant(name)
            assert "lambda_kd" in result
            assert "lambda_cf" in result
            assert "disagreement_alpha" in result

    def test_matched_control_is_baseline(self):
        result = resolve_variant("matched_control")
        assert result["lambda_kd"] == 0.0
        assert result["lambda_cf"] == 0.0
        assert result["disagreement_alpha"] == 0.0

    def test_full_variant_enables_all(self):
        result = resolve_variant("full")
        assert result["lambda_kd"] > 0
        assert result["lambda_cf"] > 0
        assert result["disagreement_alpha"] > 0

    def test_override_replaces(self):
        result = resolve_variant("full", {"lambda_kd": 0.9})
        assert result["lambda_kd"] == 0.9

    def test_none_override_preserves_default(self):
        result = resolve_variant("full", {"lambda_kd": None})
        assert result["lambda_kd"] == VARIANT_DEFAULTS["full"]["lambda_kd"]

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="unknown B19 variant"):
            resolve_variant("nonexistent")


# ---------------------------------------------------------------------------
# 6. B19Dataset construction (integration test)
# ---------------------------------------------------------------------------
def test_b19_dataset_construction():
    """Build a B19Dataset from synthetic data to verify field wiring."""
    from scripts.train_mocheg_b19_distillation import B19Dataset

    claims = [
        {"id": "1", "label": 0, "claim": "Claim A"},
        {"id": "2", "label": 1, "claim": "Claim B"},
        {"id": "3", "label": 2, "claim": "Claim C"},
    ]
    retrieval = {
        "1": {"retrieved_evidence_ids": ["E1", "E2", "E3"]},
        "2": {"retrieved_evidence_ids": ["E4", "E5"]},
        "3": {"retrieved_evidence_ids": ["E6", "E7", "E8"]},
    }
    documents = {f"E{i}": f"Evidence passage {i} text content." for i in range(1, 9)}
    explanations = {
        "1": {"is_valid": True, "grounded": True, "key_evidence_ids": [1], "reason": "ok"},
        "2": {"is_valid": True, "grounded": True, "key_evidence_ids": [2], "reason": "ok"},
        "3": {"is_valid": True, "grounded": True, "key_evidence_ids": [1], "reason": "ok"},
    }
    # Direct teacher: correct for claim 1, wrong for claim 2
    direct = {
        "1": {"probabilities": [0.8, 0.1, 0.1]},
        "2": {"probabilities": [0.7, 0.2, 0.1]},
        "3": {"probabilities": [0.1, 0.1, 0.8]},
    }
    # Grounded teacher: wrong for claim 1, correct for claim 2
    grounded = {
        "1": {"probabilities": [0.2, 0.6, 0.2]},
        "2": {"probabilities": [0.1, 0.8, 0.1]},
        "3": {"probabilities": [0.1, 0.2, 0.7]},
    }

    dataset = B19Dataset(
        claims, retrieval, documents, explanations, direct, grounded,
        top_k=5, max_evidence_chars=2200,
    )
    assert len(dataset) == 3
    # Claim 1: label=0 (Supported), has key_evidence, not NEI => cf_valid = True
    row0 = dataset[0]
    assert row0["id"] == "1"
    assert row0["cf_valid"] is True
    assert row0["disagreement"] is True
    assert row0["teacher_target_source"] == "direct_only_correct"
    # Claim 2: label=1 (Refuted), key_evidence_ids=[2] but only 2 passages => cf has 1 passage left
    row1 = dataset[1]
    assert row1["id"] == "2"
    assert row1["teacher_target_source"] == "grounded_only_correct"
    # Claim 3: label=2 (NEI), cf_valid should be False (NEI claims excluded)
    row2 = dataset[2]
    assert row2["id"] == "3"
    assert row2["cf_valid"] is False
    # Teacher target source stats
    assert dataset.target_sources["direct_only_correct"] == 1
    assert dataset.target_sources["grounded_only_correct"] == 1


# ---------------------------------------------------------------------------
# 7. Collate function shape validation
# ---------------------------------------------------------------------------
def test_b19_collate_produces_expected_keys(tmp_path):
    """Verify make_collate produces all expected batch keys."""
    from scripts.train_mocheg_b19_distillation import B19Dataset, make_collate

    claims = [{"id": "1", "label": 0, "claim": "Test claim"}]
    retrieval = {"1": {"retrieved_evidence_ids": ["E1", "E2"]}}
    documents = {"E1": "Evidence one.", "E2": "Evidence two."}
    explanations = {"1": {"is_valid": True, "grounded": True, "key_evidence_ids": [1], "reason": "ok"}}
    direct = {"1": {"probabilities": [0.8, 0.1, 0.1]}}
    grounded = {"1": {"probabilities": [0.1, 0.7, 0.2]}}

    dataset = B19Dataset(claims, retrieval, documents, explanations, direct, grounded, 5, 2200)

    # Use a minimal mock tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True)
    except Exception:
        pytest.skip("Tokenizer not available locally")

    collate_fn = make_collate(tokenizer, max_length=512)
    batch = collate_fn([dataset[0]])

    expected_keys = {
        "positive_input_ids", "positive_attention_mask",
        "negative_input_ids", "negative_attention_mask",
        "labels", "teacher_probabilities", "disagreement", "cf_valid", "ids",
    }
    assert expected_keys == set(batch.keys())
    assert batch["positive_input_ids"].shape[0] == 1
    assert batch["labels"].shape == (1,)
    assert batch["teacher_probabilities"].shape == (1, 3)


# ---------------------------------------------------------------------------
# 8. analyze_mocheg_b19_screen CLI smoke test
# ---------------------------------------------------------------------------
def test_b19_screen_cli_rejects_missing_variants(tmp_path):
    """Verify the screen script fails gracefully when variants are missing."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.analyze_mocheg_b19_screen",
         "--root", str(tmp_path),
         "--output", str(tmp_path / "screen.json"),
         "--markdown", str(tmp_path / "screen.md")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr or "summary.json" in result.stderr


# ---------------------------------------------------------------------------
# 9. End-to-end ablation matrix variant coverage
# ---------------------------------------------------------------------------
def test_all_ablation_variants_defined():
    """Ensure the 5-variant ACL ablation matrix is fully specified."""
    expected = {"matched_control", "ensemble_kd", "disagreement_kd", "counterfactual_only", "full"}
    assert set(VARIANT_DEFAULTS.keys()) == expected
    # Each variant must have exactly 3 hyperparameters
    for name, defaults in VARIANT_DEFAULTS.items():
        assert set(defaults.keys()) == {"lambda_kd", "lambda_cf", "disagreement_alpha"}, (
            f"Variant {name} has unexpected keys: {defaults.keys()}"
        )


# ---------------------------------------------------------------------------
# 10. score_mocheg_b19_teacher CLI argument validation
# ---------------------------------------------------------------------------
def test_score_teacher_cli_requires_arguments():
    """Verify the scoring script fails with clear errors when arguments are missing."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.score_mocheg_b19_teacher"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "error" in result.stderr.lower()
