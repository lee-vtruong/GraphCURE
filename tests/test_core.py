import torch
import json
import numpy as np

from graphcure.acquisition import choose_evi_action
from graphcure.losses import counterfactual_loss, directional_intervention_loss
from graphcure.model import GraphCURE, GraphCUREConfig
from graphcure.data import PackedEmbeddingDataset, resolve_newsclippings_source
from graphcure.optimization import project_auxiliary_gradients
from scripts.prepare_mocheg_protocols import close_row
from graphcure.evidence_set import EvidenceSetHead, evidence_set_loss, last_token_pool
from graphcure.retrieval import (
    contradiction_features,
    evidence_candidate_features,
    reciprocal_rank_fusion,
    retrieval_confidence,
    top_indices,
)
from scripts.train_mocheg_cached_verifier import (
    CachedEvidenceDataset,
    validate_cache_pair,
)
from scripts.run_mocheg_cached_validation import (
    markdown_summary,
    summarize_validation,
)


def test_model_shapes():
    model = GraphCURE(
        GraphCUREConfig(text_dim=8, vision_dim=8, metadata_dim=4, hidden_dim=16)
    )
    out = model(torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 4))
    assert out["verdict_logits"].shape == (2, 3)
    assert out["constraint_prob"].shape == (2, 4, 3)
    assert out["conflict"].shape == (2, 5)


def test_all_ablation_architectures_share_output_contract():
    for architecture, conflicts in (("linear", 5), ("mlp", 5),
                                    ("independent", 5), ("fully_connected", 12),
                                    ("typed_graph", 5)):
        model = GraphCURE(GraphCUREConfig(
            text_dim=8, vision_dim=8, metadata_dim=4, hidden_dim=16,
            architecture=architecture,
        ))
        out = model(torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 4))
        assert out["verdict_logits"].shape == (2, 3)
        assert out["conflict"].shape == (2, conflicts)


def test_counterfactual_loss_is_finite():
    p = torch.softmax(torch.randn(3, 4, 3), -1)
    q = torch.softmax(torch.randn(3, 4, 3), -1)
    mask = torch.tensor([[0, 0, 1, 1]] * 3).bool()
    assert torch.isfinite(counterfactual_loss(p, q, mask))


def test_directional_loss_rewards_correct_pair_order():
    mask = torch.tensor([[1, 0, 0, 0]], dtype=torch.bool)
    labels, cf_labels = torch.tensor([0]), torch.tensor([1])
    good = {
        "verdict_logits": torch.tensor([[2.0, 0.0]]),
        "constraint_prob": torch.tensor([[[0.9, 0.1, 0.0]] * 4]),
    }
    good_cf = {
        "verdict_logits": torch.tensor([[0.0, 2.0]]),
        "constraint_prob": torch.tensor([[[0.1, 0.9, 0.0]] * 4]),
    }
    bad_cf = {
        "verdict_logits": torch.tensor([[3.0, 0.0]]),
        "constraint_prob": torch.tensor([[[0.95, 0.05, 0.0]] * 4]),
    }
    assert directional_intervention_loss(good, good_cf, labels, cf_labels, mask) < \
           directional_intervention_loss(good, bad_cf, labels, cf_labels, mask)


def test_primary_projection_removes_negative_gradient_dot_product():
    primary = (torch.tensor([1.0, 0.0]),)
    auxiliary = (torch.tensor([-1.0, 1.0]),)
    combined, diagnostics = project_auxiliary_gradients(primary, auxiliary)
    projected_auxiliary = combined[0] - primary[0]
    assert diagnostics["conflict"] == 1.0
    assert torch.dot(primary[0], projected_auxiliary).abs() < 1e-6
    assert torch.equal(combined[0], torch.tensor([1.0, 1.0]))


def test_evi_stops_when_every_action_is_costly():
    current = torch.tensor([[0.8, 0.1, 0.1]])
    outcome = torch.full((2, 2), 0.5)
    posterior = torch.tensor(
        [[[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]] * 2
    )
    decision = choose_evi_action(current, outcome, posterior, torch.ones(2), 1.0)
    assert decision.should_stop


def test_packed_embedding_dataset(tmp_path):
    np.save(tmp_path / "text_embeddings.npy", np.ones((2, 4), dtype=np.float32))
    np.save(tmp_path / "image_embeddings.npy", np.arange(12, dtype=np.float32).reshape(2, 6))
    np.save(tmp_path / "labels.npy", np.array([0, 1], dtype=np.int64))
    with (tmp_path / "records.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(2):
            handle.write(json.dumps({"sample_id": f"x-{index}"}) + "\n")
    dataset = PackedEmbeddingDataset(tmp_path)
    assert len(dataset) == 2
    assert dataset[0]["text_embedding"].shape == (4,)
    assert dataset[0]["image_embedding"].shape == (6,)


def test_packed_dataset_loads_optional_constraint_targets(tmp_path):
    np.save(tmp_path / "text_embeddings.npy", np.ones((1, 4), dtype=np.float32))
    np.save(tmp_path / "image_embeddings.npy", np.ones((1, 6), dtype=np.float32))
    np.save(tmp_path / "labels.npy", np.array([1], dtype=np.int64))
    np.save(tmp_path / "constraint_labels.npy", np.array([[1, -100, -100, -100]]))
    (tmp_path / "records.jsonl").write_text(json.dumps({"sample_id": "x"}) + "\n")
    dataset = PackedEmbeddingDataset(tmp_path)
    assert dataset[0]["constraint_labels"].tolist() == [1, -100, -100, -100]


def test_packed_dataset_resolves_counterfactual_pair(tmp_path):
    np.save(tmp_path / "text_embeddings.npy", np.arange(8, dtype=np.float32).reshape(2, 4))
    np.save(tmp_path / "image_embeddings.npy", np.arange(12, dtype=np.float32).reshape(2, 6))
    np.save(tmp_path / "labels.npy", np.array([0, 1], dtype=np.int64))
    np.save(tmp_path / "counterfactual_indices.npy", np.array([1, 0]))
    np.save(tmp_path / "changed_masks.npy", np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=bool))
    (tmp_path / "records.jsonl").write_text(
        "\n".join(json.dumps({"sample_id": str(i)}) for i in range(2)) + "\n"
    )
    item = PackedEmbeddingDataset(tmp_path)[0]
    assert item["cf_label"].item() == 1
    assert item["cf_text_embedding"].tolist() == [4.0, 5.0, 6.0, 7.0]
    assert item["changed_mask"].tolist() == [True, False, False, False]
    minimal = PackedEmbeddingDataset(tmp_path, counterfactual_mode="minimal")[0]
    assert torch.equal(minimal["cf_image_embedding"], minimal["image_embedding"])
    assert minimal["cf_semantic_image_embedding"].tolist() == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]


def test_constraint_source_supports_list_and_json_dictionary():
    assert resolve_newsclippings_source(1, ["semantic", "entity"]) == "entity"
    assert resolve_newsclippings_source(2, {"2": "scene_resnet_place"}) == "scene_resnet_place"


def test_multiview_architectures_use_specialized_inputs():
    for architecture, edges in (("multi_independent", 5),
                                ("multi_fully_connected", 12),
                                ("multi_typed_graph", 5),
                                ("multi_adaptive_graph", 5)):
        model = GraphCURE(GraphCUREConfig(
            text_dim=8, vision_dim=8, metadata_dim=4, hidden_dim=16,
            sbert_dim=6, facenet_dim=5, places_dim=7,
            architecture=architecture,
        ))
        out = model(
            torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 4),
            sbert_embeddings=torch.randn(2, 6),
            facenet_embeddings=torch.randn(2, 5),
            places_embeddings=torch.randn(2, 7),
            view_mask=torch.ones(2, 5),
        )
        assert out["verdict_logits"].shape == (2, 3)
        assert out["conflict"].shape == (2, edges)
        if architecture == "multi_adaptive_graph":
            assert out["node_mix_gates"].shape == (2, 4)
            assert torch.all((out["node_mix_gates"] > 0) &
                             (out["node_mix_gates"] < 0.5))


def test_mocheg_close_protocol_removes_qrel_evidence():
    source = {
        "id": "mocheg-test-claim-1",
        "claim_id": "1",
        "label": 0,
        "label_name": "supported",
        "claim": "A claim",
        "evidence_texts": ["gold text"],
        "text_evidence_ids": ["e-1"],
        "image_paths": ["gold.jpg"],
        "image_evidence_ids": ["gold.jpg"],
        "ruling_outline": "leaky fact-check text",
    }
    closed = close_row(source)
    assert closed["claim"] == "A claim"
    assert closed["evidence_texts"] == []
    assert closed["text_evidence_ids"] == []
    assert closed["image_paths"] == []
    assert closed["image_evidence_ids"] == []
    assert "ruling_outline" not in closed
    assert closed["evidence_provenance"] == "none"


def test_cached_validation_summary_is_validation_only(tmp_path):
    paths = {}
    for seed, f1 in ((13, 0.52), (42, 0.54)):
        run = tmp_path / str(seed)
        run.mkdir()
        path = run / "val_metrics.json"
        path.write_text(json.dumps({
            "accuracy": f1 + 0.01,
            "macro_f1": f1,
            "evidence_selection_hit_at_1": 0.8,
            "ece_10": 0.05,
            "best_val_macro_f1": f1,
            "retrieval_gold_coverage": 0.94,
            "provenance": {"cache_metadata": {
                "manifest_sha256": "manifest",
                "retrieval_sha256": "retrieval",
                "encoder": "encoder",
                "top_k": 8,
            }},
        }), encoding="utf-8")
        paths[seed] = path
    summary = summarize_validation(paths)
    assert summary["test_split_used"] is False
    assert summary["stability_gate"]["passed"]
    assert summary["aggregate"]["macro_f1"]["mean"] == 0.53
    assert "Test split used: **no**" in markdown_summary(summary)


def test_reciprocal_rank_fusion_rewards_cross_retriever_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], rank_constant=1)
    assert fused[0][0] == 1
    assert {item[0] for item in fused} == {1, 2, 3, 4}


def test_top_indices_and_retrieval_confidence():
    assert top_indices(np.array([0.1, 0.8, 0.4]), 2).tolist() == [1, 2]
    assert retrieval_confidence([4.0, 1.0]) > retrieval_confidence([1.1, 1.0])


def test_contradiction_features_detect_fact_traps():
    features = contradiction_features(
        "The event killed 12 people.",
        "The report says the event did not kill 20 people.",
    )
    assert features["negation_mismatch"]
    assert features["number_mismatch"]


def test_candidate_features_upweight_non_gold_reasoning_traps():
    features, weight = evidence_candidate_features(
        "The event killed 12 people.",
        "The event did not kill 20 people.",
        score=0.8,
        top_score=0.9,
        rank=2,
        is_gold=False,
    )
    assert len(features) == 6
    assert weight > 1.0
    _, gold_weight = evidence_candidate_features(
        "claim", "evidence", 0.9, 0.9, 1, True
    )
    assert gold_weight == 1.0


def test_last_token_pool_supports_right_padding():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pooled = last_token_pool(hidden, mask)
    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 2])


def test_evidence_set_head_and_multitask_loss_are_finite():
    head = EvidenceSetHead(encoder_dim=8, hidden_dim=16, retrieval_dim=6)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    relevance = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
    output = head(
        torch.randn(2, 8),
        torch.randn(2, 3, 8),
        mask,
        torch.randn(2, 3, 6),
    )
    loss, parts = evidence_set_loss(
        output,
        torch.tensor([0, 1]),
        relevance,
        mask,
        relevance_weights=torch.tensor([[1.0, 1.25, 0.0], [1.0, 1.1, 1.25]]),
    )
    assert output["verdict_logits"].shape == (2, 3)
    assert output["attention"].shape == (2, 3)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_cached_evidence_dataset_alignment(tmp_path):
    payload = {
        "ids": ["a", "b"],
        "claim_embeddings": torch.randn(2, 8).half(),
        "evidence_embeddings": torch.randn(2, 3, 8).half(),
        "evidence_mask": torch.ones(2, 3, dtype=torch.bool),
        "retrieval_features": torch.randn(2, 3, 6),
        "relevance": torch.zeros(2, 3),
        "relevance_weights": torch.ones(2, 3),
        "labels": torch.tensor([0, 1]),
        "metadata": {
            "encoder": "test", "embedding_dim": 8, "top_k": 3,
            "max_length": 16,
        },
    }
    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    torch.save(payload, train_path)
    torch.save(payload, val_path)
    train = CachedEvidenceDataset(train_path)
    val = CachedEvidenceDataset(val_path)
    validate_cache_pair(train, val)
    assert len(train) == 2
    assert train[0]["retrieval_features"].shape == (3, 6)
