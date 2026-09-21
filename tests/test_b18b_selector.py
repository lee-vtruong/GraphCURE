"""Unit tests for Phase B18-B: Explanation-Derived Sentence & Passage Selector."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from graphcure.selector import (
    AdaptiveSelectorPolicy,
    extract_selector_pairs,
    mock_score_claim_evidence_pairs,
)
from scripts.prepare_mocheg_b18b_selected_evidence import select_retrieval_top_k
from scripts.calibrate_mocheg_b18b_selector_policy import (
    attribution_metrics,
    choose_policy,
)
from scripts.train_mocheg_b18b_sentence_selector import split_pairs_by_claim


def test_adaptive_selector_policy_min_max_k() -> None:
    policy = AdaptiveSelectorPolicy(min_k=1, max_k=2, score_threshold=0.0, adaptive_margin=10.0)
    candidates = ["doc_A", "doc_B", "doc_C", "doc_D"]
    scores = [0.8, 0.7, 0.6, 0.5]
    selected_ids, selected_scores = policy.select(candidates, scores)
    assert len(selected_ids) == 2
    assert selected_ids == ["doc_A", "doc_B"]
    assert selected_scores == [0.8, 0.7]


def test_adaptive_selector_policy_margin_pruning() -> None:
    # doc_B is within margin 1.0 (2.5 - 1.8 = 0.7 <= 1.0)
    # doc_C exceeds margin (2.5 - 1.2 = 1.3 > 1.0)
    policy = AdaptiveSelectorPolicy(min_k=1, max_k=3, score_threshold=-5.0, adaptive_margin=1.0)
    candidates = ["doc_A", "doc_B", "doc_C"]
    scores = [2.5, 1.8, 1.2]
    selected_ids, selected_scores = policy.select(candidates, scores)
    assert selected_ids == ["doc_A", "doc_B"]
    assert selected_scores == [2.5, 1.8]


def test_adaptive_selector_policy_threshold_pruning() -> None:
    # doc_B passes threshold 0.0, doc_C fails threshold 0.0 (-0.2 < 0.0)
    policy = AdaptiveSelectorPolicy(min_k=1, max_k=3, score_threshold=0.0, adaptive_margin=5.0)
    candidates = ["doc_A", "doc_B", "doc_C"]
    scores = [1.5, 0.2, -0.2]
    selected_ids, selected_scores = policy.select(candidates, scores)
    assert selected_ids == ["doc_A", "doc_B"]


def test_adaptive_selector_policy_always_respects_min_k() -> None:
    # Even if scores are very low or far from top, min_k=2 guarantees 2 items
    policy = AdaptiveSelectorPolicy(min_k=2, max_k=3, score_threshold=10.0, adaptive_margin=0.01)
    candidates = ["doc_A", "doc_B", "doc_C"]
    scores = [1.0, 0.0, -1.0]
    selected_ids, _ = policy.select(candidates, scores)
    assert len(selected_ids) == 2
    assert selected_ids == ["doc_A", "doc_B"]


def test_extract_selector_pairs_grounded_and_hard_negatives() -> None:
    documents = {
        "ev_1": "The event happened on Tuesday morning in Boston.",
        "ev_2": "Boston police reported the suspect was arrested.",
        "ev_3": "Weather in California was sunny and warm.",
        "ev_4": "Stock prices surged during the afternoon trade.",
    }
    explanations = [
        {
            "id": "c1",
            "claim": "Suspect was arrested Tuesday morning in Boston.",
            "label": 0,
            "retrieved_evidence_ids": ["ev_1", "ev_2", "ev_3", "ev_4"],
            "grounded": True,
            "is_valid": True,
            "verdict": "SUPPORTED",
            "key_evidence_ids": [1, 2],  # 1-based index: ev_1 and ev_2
            "reason": "Evidence 1 and 2 confirm the arrest in Boston.",
        }
    ]

    pairs = extract_selector_pairs(explanations, documents, seed=42)
    assert len(pairs) == 4

    positives = [p for p in pairs if p["label"] == 1.0]
    negatives = [p for p in pairs if p["label"] == 0.0]

    assert len(positives) == 2
    assert {p["evidence_id"] for p in positives} == {"ev_1", "ev_2"}

    assert len(negatives) == 2
    assert {p["evidence_id"] for p in negatives} == {"ev_3", "ev_4"}
    assert all(p["is_hard_negative"] for p in negatives)


def test_extract_selector_pairs_ungrounded() -> None:
    documents = {
        "ev_1": "Completely unrelated news article.",
        "ev_2": "Another random passage.",
    }
    explanations = [
        {
            "id": "c2",
            "claim": "Alien spaceship landed in Central Park.",
            "label": 2,
            "retrieved_evidence_ids": ["ev_1", "ev_2"],
            "grounded": False,
            "is_valid": True,
            "verdict": "NEI",
            "key_evidence_ids": [],
            "reason": "Retrieved evidence does not ground the claim.",
        }
    ]

    pairs = extract_selector_pairs(
        explanations, documents, sample_ungrounded_negatives=True, seed=42
    )
    assert len(pairs) == 2
    assert all(p["label"] == 0.0 for p in pairs)
    assert all(not p["is_hard_negative"] for p in pairs)


def test_mock_score_claim_evidence_pairs() -> None:
    pairs = [
        ("The cat sat on the mat", "A cat is on the mat sleeping"),
        ("Rocket launched to orbit", "Completely unrelated garden vegetables"),
    ]
    scores = mock_score_claim_evidence_pairs(pairs)
    assert scores[0] > scores[1]
    assert scores[1] == 0.0


def test_selector_dev_split_is_claim_disjoint() -> None:
    pairs = [
        {"claim_id": f"c{claim}", "label": float(index % 2)}
        for claim in range(10)
        for index in range(3)
    ]
    train_pairs, dev_pairs = split_pairs_by_claim(pairs, dev_fraction=0.2, seed=42)
    train_claims = {row["claim_id"] for row in train_pairs}
    dev_claims = {row["claim_id"] for row in dev_pairs}
    assert train_claims.isdisjoint(dev_claims)
    assert len(dev_claims) == 2
    assert len(train_pairs) + len(dev_pairs) == len(pairs)


def test_retrieval_top_k_preserves_upstream_order_and_scores() -> None:
    candidates = ["doc_c", "doc_a", "doc_b"]
    scores = [0.91, 0.72, 0.61]
    selected_ids, selected_scores = select_retrieval_top_k(candidates, scores, 2)
    assert selected_ids == ["doc_c", "doc_a"]
    assert selected_scores == [0.91, 0.72]


def test_calibration_prefers_recall_safe_policy() -> None:
    records = [
        {
            "candidate_ids": ["a", "b", "c"],
            "scores": [2.0, 1.2, -2.0],
            "key_evidence_ids": {"a", "b"},
        },
        {
            "candidate_ids": ["d", "e", "f"],
            "scores": [1.8, 1.0, -2.0],
            "key_evidence_ids": {"d", "e"},
        },
    ]
    strict = attribution_metrics(
        records,
        score_threshold=1.5,
        adaptive_margin=0.5,
        min_k=1,
        max_k=3,
    )
    recall_safe = attribution_metrics(
        records,
        score_threshold=0.0,
        adaptive_margin=1.0,
        min_k=1,
        max_k=3,
    )
    chosen, satisfied = choose_policy([strict, recall_safe], minimum_teacher_recall=0.9)
    assert satisfied is True
    assert chosen["micro_teacher_key_recall"] == 1.0
    assert chosen["mean_selected_k"] == 2.0


def test_end_to_end_train_mock_selector_cli(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "Corpus2.csv"
    with corpus_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["evidence_id", "Evidence"])
        writer.writerow(["ev_1", "Vaccines are safe and thoroughly tested."])
        writer.writerow(["ev_2", "Side effects are rare and usually mild."])
        writer.writerow(["ev_3", "Traffic on the interstate was heavily congested."])

    exp_jsonl = tmp_path / "explanations.jsonl"
    exp_rows = [
        {
            "id": "c1",
            "claim": "Vaccines are safe.",
            "label": 0,
            "retrieved_evidence_ids": ["ev_1", "ev_2", "ev_3"],
            "grounded": True,
            "is_valid": True,
            "verdict": "SUPPORTED",
            "key_evidence_ids": [1],
        }
    ]
    with exp_jsonl.open("w", encoding="utf-8") as f:
        for r in exp_rows:
            f.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "selector_out"
    cmd = [
        sys.executable,
        "-m",
        "scripts.train_mocheg_b18b_sentence_selector",
        "--explanations",
        str(exp_jsonl),
        "--corpus",
        str(corpus_csv),
        "--output",
        str(out_dir),
        "--mock",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert out_dir.is_dir()
    assert (out_dir / "mock_selector_config.json").is_file()
    assert (out_dir / "training_summary.json").is_file()
    summary = json.loads((out_dir / "training_summary.json").read_text())
    assert summary["positives"] == 1
    assert summary["negatives"] == 2
    assert summary["claim_split_overlap"] == 0
    assert summary["audit"]["validation_labels_used"] is False


def test_end_to_end_prepare_selected_evidence_cli(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "Corpus2.csv"
    with corpus_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["evidence_id", "Evidence"])
        writer.writerow(["ev_1", "Vaccines are rigorously evaluated for safety."])
        writer.writerow(["ev_2", "Vaccines are safe according to public health officials."])
        writer.writerow(["ev_3", "Sports scores for last night games."])

    manifest_jsonl = tmp_path / "claims.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "c1", "claim": "Vaccines are safe", "label": 0}) + "\n")

    retrieval_jsonl = tmp_path / "retrieval.jsonl"
    with retrieval_jsonl.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps({
                "id": "c1",
                "retrieved_evidence_ids": ["ev_3", "ev_2", "ev_1"],
                "gold_evidence_ids": ["ev_2"],
            })
            + "\n"
        )

    selector_dir = tmp_path / "dummy_selector"
    selector_dir.mkdir(parents=True)
    (selector_dir / "mock_selector_config.json").write_text(json.dumps({"is_mock": True}))

    out_retrieval = tmp_path / "filtered_retrieval.jsonl"
    out_summary = tmp_path / "summary.json"

    cmd = [
        sys.executable,
        "-m",
        "scripts.prepare_mocheg_b18b_selected_evidence",
        "--selector",
        str(selector_dir),
        "--retrieval",
        str(retrieval_jsonl),
        "--manifest",
        str(manifest_jsonl),
        "--corpus",
        str(corpus_csv),
        "--output",
        str(out_retrieval),
        "--summary",
        str(out_summary),
        "--min-k",
        "1",
        "--max-k",
        "2",
        "--adaptive-margin",
        "1.5",
        "--mock",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    assert out_retrieval.is_file()
    assert out_summary.is_file()

    rows = [json.loads(line) for line in out_retrieval.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    # Check that ev_2 and ev_1 (with high word overlap with 'Vaccines are safe') are ranked ahead of ev_3 (sports scores)
    selected_ids = rows[0]["retrieved_evidence_ids"]
    assert len(selected_ids) <= 2
    assert "ev_2" in selected_ids or "ev_1" in selected_ids
    assert selected_ids[0] != "ev_3"  # ev_3 should not be top rank

    summary = json.loads(out_summary.read_text())
    assert summary["total_claims"] == 1
    assert summary["avg_selected_passages"] >= 1.0


def test_retrieval_control_cli_needs_no_selector_and_preserves_order(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "Corpus2.csv"
    with corpus_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["evidence_id", "Evidence"])
        writer.writerow(["ev_1", "Low lexical overlap."])
        writer.writerow(["ev_2", "Claim words appear here strongly."])
        writer.writerow(["ev_3", "Another passage."])

    manifest = tmp_path / "claims.jsonl"
    manifest.write_text(json.dumps({"id": "c1", "claim": "Claim words"}) + "\n")
    retrieval = tmp_path / "retrieval.jsonl"
    retrieval.write_text(
        json.dumps({
            "id": "c1",
            "retrieved_evidence_ids": ["ev_1", "ev_2", "ev_3"],
            "retrieved_scores": [0.9, 0.8, 0.7],
        }) + "\n"
    )
    output = tmp_path / "top2.jsonl"
    summary = tmp_path / "summary.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.prepare_mocheg_b18b_selected_evidence",
        "--retrieval", str(retrieval),
        "--manifest", str(manifest),
        "--corpus", str(corpus_csv),
        "--output", str(output),
        "--summary", str(summary),
        "--policy-mode", "retrieval_top_3",
        "--top-k-candidates", "3",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    row = json.loads(output.read_text().strip())
    assert row["retrieved_evidence_ids"] == ["ev_1", "ev_2", "ev_3"]
    assert row["retrieved_scores"] == [0.9, 0.8, 0.7]
    audit = json.loads(summary.read_text())["audit"]
    assert audit["preserves_upstream_order"] is True
    assert audit["selector_scores_used"] is False


def test_prepare_selected_evidence_with_teacher_attribution(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "Corpus2.csv"
    with corpus_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["evidence_id", "Evidence"])
        writer.writerow(["ev_1", "The president signed the treaty yesterday."])
        writer.writerow(["ev_2", "Foreign ministers praised the bilateral agreement."])
        writer.writerow(["ev_3", "Heavy snow caused delays at the international airport."])

    manifest_jsonl = tmp_path / "claims.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "c1", "claim": "The president signed the treaty."}) + "\n")

    retrieval_jsonl = tmp_path / "retrieval.jsonl"
    with retrieval_jsonl.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps({"id": "c1", "retrieved_evidence_ids": ["ev_1", "ev_2", "ev_3"]}) + "\n"
        )

    teacher_exp_jsonl = tmp_path / "teacher_explanations.jsonl"
    with teacher_exp_jsonl.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps({
                "id": "c1",
                "is_valid": True,
                "grounded": True,
                "verdict": "SUPPORTED",
                "key_evidence_ids": [1],  # ev_1
                "retrieved_evidence_ids": ["ev_1", "ev_2", "ev_3"],
            })
            + "\n"
        )

    selector_dir = tmp_path / "dummy_selector"
    selector_dir.mkdir(parents=True, exist_ok=True)
    (selector_dir / "mock_selector_config.json").write_text(json.dumps({"is_mock": True}))

    out_retrieval = tmp_path / "filtered.jsonl"
    out_summary = tmp_path / "summary.json"
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "protocol": "train_side_claim_disjoint_teacher_attribution_calibration_v1",
        "policy_name": "distilled_ce_adaptive",
        "selected_policy": {
            "score_threshold": -5.0,
            "adaptive_margin": 4.0,
        },
        "audit": {
            "official_validation_used": False,
            "test_split_used": False,
        },
    }))

    cmd = [
        sys.executable,
        "-m",
        "scripts.prepare_mocheg_b18b_selected_evidence",
        "--selector",
        str(selector_dir),
        "--retrieval",
        str(retrieval_jsonl),
        "--manifest",
        str(manifest_jsonl),
        "--corpus",
        str(corpus_csv),
        "--output",
        str(out_retrieval),
        "--summary",
        str(out_summary),
        "--teacher-explanations",
        str(teacher_exp_jsonl),
        "--policy-calibration",
        str(calibration),
        "--policy-mode",
        "adaptive",
        "--mock",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    summary = json.loads(out_summary.read_text())
    assert summary["requested_policy_mode"] == "adaptive"
    assert summary["policy_mode"] == "distilled_ce_adaptive"
    assert summary["policy"]["score_threshold"] == -5.0
    assert summary["policy"]["adaptive_margin"] == 4.0
    assert summary["audit"]["policy_calibration_sha256"] is not None
    assert summary["audit"]["policy_calibration_uses_test"] is False
    assert "teacher_attribution" in summary
    attr = summary["teacher_attribution"]
    assert attr["grounded_claims_evaluated"] == 1
    assert attr["teacher_key_coverage_mean"] == 1.0
    assert attr["pseudo_recall_at_1"] == 1.0
