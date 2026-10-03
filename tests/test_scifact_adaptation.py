"""Unit and integration tests for SciFact domain adaptation pipeline."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from graphcure.explanation import validate_explanation
from scripts.evaluate_scifact_adaptation import (
    apply_and_routing,
    apply_self_deferral,
    compute_metrics,
    tune_tau_on_oof,
    tune_tau_self_on_oof,
)
from scripts.generate_scifact_teacher_explanations import mock_teacher_generate
from scripts.prepare_scifact_adaptation_protocol import (
    build_corpus_documents,
    build_stratified_folds,
    compute_tfidf_retrieval,
    label_for_claim,
)


# -----------------------------------------------------------------------------
# 1. Label Mapping Tests
# -----------------------------------------------------------------------------
class TestSciFactLabelMapping:
    def test_supports_maps_to_zero(self):
        claim = {"id": 1, "evidence": {"100": [{"label": "SUPPORT", "sentences": [0]}]}}
        assert label_for_claim(claim) == 0

    def test_contradict_maps_to_one(self):
        claim = {"id": 2, "evidence": {"101": [{"label": "CONTRADICT", "sentences": [1]}]}}
        assert label_for_claim(claim) == 1

    def test_empty_evidence_maps_to_two_nei(self):
        claim = {"id": 3, "evidence": {}}
        assert label_for_claim(claim) == 2

    def test_missing_evidence_key_maps_to_nei(self):
        claim = {"id": 4}
        assert label_for_claim(claim) == 2

    def test_conflicting_labels_raises(self):
        claim = {
            "id": 5,
            "evidence": {
                "100": [{"label": "SUPPORT", "sentences": [0]}],
                "101": [{"label": "CONTRADICT", "sentences": [1]}],
            },
        }
        with pytest.raises(ValueError, match="unexpected or conflicting"):
            label_for_claim(claim)


# -----------------------------------------------------------------------------
# 2. Stratified Cross-Validation Splitting Tests
# -----------------------------------------------------------------------------
class TestStratifiedFoldSplitting:
    def test_folds_are_partition_and_stratified(self):
        # 100 claims: 40 Supported (0), 20 Refuted (1), 40 NEI (2)
        claims = [{"id": i, "claim": f"Claim {i}"} for i in range(100)]
        labels = [0] * 40 + [1] * 20 + [2] * 40
        folds = build_stratified_folds(claims, labels, num_folds=5, seed=2040)

        assert len(folds) == 5
        all_val_ids = []
        for f in folds:
            train_ids = set(f["train_ids"])
            val_ids = set(f["val_ids"])
            assert len(train_ids & val_ids) == 0  # No leakage between train and val
            assert len(train_ids) + len(val_ids) == 100
            all_val_ids.extend(f["val_ids"])

            # Verify stratified distribution in val: roughly 8 S, 4 R, 8 N
            assert f["val_label_counts"]["0"] == 8
            assert f["val_label_counts"]["1"] == 4
            assert f["val_label_counts"]["2"] == 8

        # Val sets must cover exactly all 100 claims once
        assert sorted(all_val_ids) == sorted([f"scifact-train-{i}" for i in range(100)])


# -----------------------------------------------------------------------------
# 3. TF-IDF Retrieval Determinism Tests
# -----------------------------------------------------------------------------
class TestTfidfRetrieval:
    def test_retrieval_ranking_matches_content(self):
        corpus = [
            {"doc_id": 1, "title": "CRISPR gene editing", "abstract": ["Cas9 endonuclease allows targeted mutation."]},
            {"doc_id": 2, "title": "COVID-19 epidemiology", "abstract": ["SARS-CoV-2 transmission dynamics."]},
            {"doc_id": 3, "title": "Quantum computing", "abstract": ["Superconducting qubits and coherence time."]},
        ]
        doc_ids, texts = build_corpus_documents(corpus)
        claims = [
            {"id": "c1", "claim": "CRISPR Cas9 causes mutations"},
            {"id": "c2", "claim": "SARS-CoV-2 spreads rapidly"},
        ]
        ret_ids, ret_scores = compute_tfidf_retrieval(texts, doc_ids, claims, top_k=2)

        # Claim 1 must retrieve doc 1 first
        assert ret_ids[0][0] == "1"
        assert ret_scores[0][0] > ret_scores[0][1]

        # Claim 2 must retrieve doc 2 first
        assert ret_ids[1][0] == "2"
        assert ret_scores[1][0] > ret_scores[1][1]


# -----------------------------------------------------------------------------
# 4. AND Routing & Self-Deferral Logic Tests
# -----------------------------------------------------------------------------
class TestRoutingLogic:
    def test_and_routing_only_defers_when_both_conditions_met(self):
        # Case 0: Rationale != 2 -> preserves direct prediction
        # Case 1: Rationale == 2 but prob < tau -> preserves direct prediction
        # Case 2: Rationale == 2 and prob >= tau -> defers to 2 (NEI)
        direct_probs = np.array([
            [0.8, 0.1, 0.1],  # Pred: 0
            [0.8, 0.1, 0.1],  # Pred: 0
            [0.8, 0.1, 0.1],  # Pred: 0
        ])
        rationale_probs = np.array([
            [0.7, 0.2, 0.1],  # Pred: 0 (not 2)
            [0.1, 0.4, 0.5],  # Pred: 2, but p(2) = 0.50 < tau (0.60)
            [0.1, 0.1, 0.8],  # Pred: 2, and p(2) = 0.80 >= tau (0.60)
        ])
        routed = apply_and_routing(direct_probs, rationale_probs, tau=0.60)
        assert routed[0] == 0  # direct preserved
        assert routed[1] == 0  # direct preserved because prob < 0.60
        assert routed[2] == 2  # routed to NEI

    def test_self_deferral_rule(self):
        direct_probs = np.array([
            [0.7, 0.2, 0.1],  # Pred: 0, p(NEI) = 0.1 < 0.3 -> stays 0
            [0.5, 0.1, 0.4],  # Pred: 0, p(NEI) = 0.4 >= 0.3 -> defers to 2
            [0.1, 0.1, 0.8],  # Pred: 2, stays 2
        ])
        self_routed = apply_self_deferral(direct_probs, tau_self=0.30)
        assert self_routed[0] == 0
        assert self_routed[1] == 2
        assert self_routed[2] == 2

    def test_tune_tau_on_oof(self):
        # Synthetic setup where tau=0.50 yields highest Macro-F1
        labels = np.array([0, 1, 2, 2, 2])
        direct_probs = np.array([
            [0.9, 0.05, 0.05],  # Correct 0
            [0.05, 0.9, 0.05],  # Correct 1
            [0.6, 0.2, 0.2],    # Direct predicts 0 (wrong)
            [0.7, 0.1, 0.2],    # Direct predicts 0 (wrong)
            [0.1, 0.1, 0.8],    # Direct predicts 2 (correct)
        ])
        rationale_probs = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],  # Rationale says NEI (p=0.8)
            [0.1, 0.1, 0.6],  # Rationale says NEI (p=0.6)
            [0.1, 0.1, 0.8],
        ])
        best_tau, best_f1, _ = tune_tau_on_oof(labels, direct_probs, rationale_probs)
        assert 0.20 <= best_tau <= 0.60
        assert best_f1 == 1.0  # Perfect recovery when routing both errors to NEI


# -----------------------------------------------------------------------------
# 5. Teacher Explanation Schema & Validation Tests
# -----------------------------------------------------------------------------
class TestTeacherExplanations:
    def test_mock_teacher_produces_grounded_supported(self):
        claim = "Gene expression is altered by temperature."
        evidence = ["Temperature causes gene expression to change significantly."]
        exp = mock_teacher_generate(claim, evidence, "SUPPORTED")
        is_valid, parsed, issues = validate_explanation(exp, len(evidence), evidence)
        assert is_valid
        assert parsed is not None
        assert parsed.grounded is True
        assert parsed.verdict == "SUPPORTED"
        assert parsed.key_evidence_ids == [1]

    def test_mock_teacher_produces_nei(self):
        claim = "Astrophysical dark matter composed of axions."
        evidence = ["Cellular mitosis occurs in eukaryotic cells."]
        exp = mock_teacher_generate(claim, evidence, "SUPPORTED")
        is_valid, parsed, issues = validate_explanation(exp, len(evidence), evidence)
        assert is_valid
        assert parsed is not None
        assert parsed.grounded is False
        assert parsed.verdict == "NEI"


# -----------------------------------------------------------------------------
# 6. CLI Argument Validation Tests
# -----------------------------------------------------------------------------
class TestCliArguments:
    def test_prepare_protocol_cli_requires_arguments(self):
        res = subprocess.run(
            [sys.executable, "-m", "scripts.prepare_scifact_adaptation_protocol"],
            capture_output=True, text=True,
        )
        assert res.returncode != 0
        assert "required" in res.stderr.lower() or "error" in res.stderr.lower()

    def test_train_scifact_expert_cli_requires_arguments(self):
        res = subprocess.run(
            [sys.executable, "-m", "scripts.train_scifact_expert"],
            capture_output=True, text=True,
        )
        assert res.returncode != 0
        assert "required" in res.stderr.lower() or "error" in res.stderr.lower()

    def test_evaluate_adaptation_cli_requires_arguments(self):
        res = subprocess.run(
            [sys.executable, "-m", "scripts.evaluate_scifact_adaptation"],
            capture_output=True, text=True,
        )
        assert res.returncode != 0
        assert "required" in res.stderr.lower() or "error" in res.stderr.lower()
