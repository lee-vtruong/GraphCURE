"""Unit tests for Phase B18 explanation schema, prompts, validation, folds, and analysis."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from graphcure.explanation import (
    StructuredExplanation,
    compose_student_explanation_prompt,
    compose_student_verdict_prompt,
    compose_teacher_prompt,
    resolve_corpus_path,
    validate_explanation,
)
from scripts.analyze_mocheg_b18_explanation_verifier import analyze_b18_run
from scripts.prepare_mocheg_b18_folds import build_b18_folds


def test_valid_supported_explanation_passes_validation():
    payload = {
        "grounded": True,
        "verdict": "SUPPORTED",
        "key_evidence_ids": [1, 2],
        "reason": "Evidence [1] and [2] directly confirm the claim assertions.",
        "missing_information": None,
    }
    evidence_texts = ["First evidence text.", "Second evidence text."]
    is_valid, exp, issues = validate_explanation(payload, num_evidence=2, evidence_texts=evidence_texts)
    assert is_valid is True
    assert issues == []
    assert isinstance(exp, StructuredExplanation)
    assert exp.grounded is True
    assert exp.verdict == "SUPPORTED"
    assert exp.key_evidence_ids == [1, 2]


def test_valid_nei_explanation_requires_missing_info():
    # Valid NEI with missing_information
    valid_nei = {
        "grounded": True,
        "verdict": "NEI",
        "key_evidence_ids": [1],
        "reason": "Evidence mentions the person but not the alleged event.",
        "missing_information": "Official verification of attendance.",
    }
    is_valid, exp, issues = validate_explanation(valid_nei, num_evidence=2)
    assert is_valid is True
    assert exp.missing_information == "Official verification of attendance."

    # Invalid NEI without missing_information
    invalid_nei = {
        "grounded": True,
        "verdict": "NEI",
        "key_evidence_ids": [1],
        "reason": "Evidence is incomplete.",
        "missing_information": None,
    }
    is_valid, exp, issues = validate_explanation(invalid_nei, num_evidence=2)
    assert is_valid is False
    assert any("missing_information" in err for err in issues)


def test_ungrounded_explanation_flags_safely():
    payload = {
        "grounded": False,
        "verdict": "NEI",
        "key_evidence_ids": [],
        "reason": "The retrieved documents are completely unrelated.",
        "missing_information": "Any relevant source.",
    }
    is_valid, exp, issues = validate_explanation(payload, num_evidence=3)
    assert is_valid is True
    assert exp.grounded is False
    assert exp.key_evidence_ids == []


def test_out_of_bounds_evidence_id_rejected():
    payload = {
        "grounded": True,
        "verdict": "REFUTED",
        "key_evidence_ids": [5],  # only 2 passages available!
        "reason": "Evidence refutes claim.",
        "missing_information": None,
    }
    is_valid, exp, issues = validate_explanation(payload, num_evidence=2)
    assert is_valid is False
    assert any("out of range" in err for err in issues)


def test_hallucinated_quote_rejected():
    payload = {
        "grounded": True,
        "verdict": "SUPPORTED",
        "key_evidence_ids": [1],
        "reason": 'Evidence states that "completely fabricated text never in passage" occurred.',
        "missing_information": None,
    }
    evidence_texts = ["The prime minister visited the hospital on Tuesday morning."]
    is_valid, exp, issues = validate_explanation(payload, num_evidence=1, evidence_texts=evidence_texts)
    assert is_valid is False
    assert any("Quoted text" in err for err in issues)


def test_prompt_composition():
    claim = "The event took place in Hanoi."
    evidence = ["Passage 1 content.", "Passage 2 content."]

    teacher_prompt = compose_teacher_prompt(claim, evidence, "SUPPORTED")
    assert "The event took place in Hanoi." in teacher_prompt
    assert "[1] Passage 1 content." in teacher_prompt
    assert "[2] Passage 2 content." in teacher_prompt
    assert "Reference Gold Verdict: SUPPORTED" in teacher_prompt

    v_prompt = compose_student_verdict_prompt(claim, evidence)
    assert "Return only A, B, or C." in v_prompt

    e_prompt = compose_student_explanation_prompt(claim, evidence)
    assert "Provide a structured explanation JSON" in e_prompt


def test_duplicate_safe_folds_prevent_leakage():
    mock_rows = [
        {"id": f"claim_{i}", "claim": f"Claim {i}", "label": i % 3, "snopes_url": f"https://example.com/art_{i // 2}"}
        for i in range(20)
    ]
    folds = build_b18_folds(mock_rows, folds=5, seed=2040)
    assert len(folds) == 5
    for fold in folds:
        train_set = set(fold["train_ids"])
        val_set = set(fold["val_ids"])
        assert len(train_set & val_set) == 0


def test_analyze_b18_run_computes_deltas_and_gates():
    claims = [
        {"id": str(i), "label": i % 3, "source": "politifact"}
        for i in range(30)
    ]
    # Candidate gets all correct
    cand_preds = [{"id": str(i), "label": i % 3, "prediction": i % 3} for i in range(30)]
    # Control gets some wrong
    ctrl_preds = [{"id": str(i), "label": i % 3, "prediction": 0 if i < 10 else (i % 3)} for i in range(30)]

    result = analyze_b18_run(
        cand_preds, ctrl_preds, anchor_preds=None, manifest_claims=claims, iterations=200, seed=42
    )

    assert result["samples"] == 30
    assert result["delta_metrics"]["macro_f1"] > 0
    assert result["paired_analysis"]["helpful"] > 0
    assert result["paired_analysis"]["harmful"] == 0
    assert result["gate_checks"]["helpful_exceeds_harmful"] is True


def test_resolve_corpus_path_finds_nested_file(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    corpus_file = train_dir / "Corpus2.csv"
    corpus_file.write_text("header\n1", encoding="utf-8")

    # Pass directory directly
    resolved = resolve_corpus_path(tmp_path)
    assert resolved == corpus_file

    # Pass file directly
    assert resolve_corpus_path(corpus_file) == corpus_file

    # Non-existent raises
    with pytest.raises(FileNotFoundError):
        resolve_corpus_path(Path("nonexistent_top_level_folder_xyz_123") / "corpus.csv")


def test_multi_seed_summarize(tmp_path):
    from scripts.summarize_mocheg_b18_seeds import compute_metrics, load_seed_predictions
    import numpy as np

    cand_dir = tmp_path / "cand"
    cand_dir.mkdir()
    preds = [{"id": str(i), "label": i % 3, "prediction": i % 3, "probabilities": [0.8, 0.1, 0.1]} for i in range(10)]
    (cand_dir / "val_predictions.jsonl").write_text("\n".join(json.dumps(p) for p in preds) + "\n")

    loaded = load_seed_predictions(cand_dir)
    assert len(loaded) == 10

    m = compute_metrics(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0



