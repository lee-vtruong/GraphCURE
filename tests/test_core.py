import torch
import json
import numpy as np
from pathlib import Path

from graphcure.acquisition import choose_evi_action
from graphcure.losses import counterfactual_loss, directional_intervention_loss
from graphcure.model import GraphCURE, GraphCUREConfig
from graphcure.data import PackedEmbeddingDataset, resolve_newsclippings_source
from graphcure.optimization import project_auxiliary_gradients
from scripts.prepare_mocheg_protocols import close_row
from graphcure.evidence_set import EvidenceSetHead, evidence_set_loss, last_token_pool
from graphcure.multimodal_evidence import (
    MultimodalEvidenceHead,
    multimodal_evidence_loss,
)
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
from scripts.evaluate_mocheg_frozen_r2v import validate_freeze
from scripts.run_mocheg_visual_retrieval import (
    encode_images,
    encode_queries,
    is_torchvision_supported_image,
    normalize_image_paths,
    retrieval_summary,
)
from scripts.run_mocheg_visual_ensemble import (
    aligned_gold_images,
    fuse_visual_orders,
)
from scripts.caption_mocheg_images import (
    clean_descriptor,
    conversations,
    descriptor_signature,
    read_completed,
)
from scripts.run_mocheg_caption_fusion import (
    candidate_union_diagnostics,
    read_descriptors,
)
from scripts.rerank_mocheg_visual_qwen3 import (
    read_resumable_jsonl,
    stable_rerank,
)
from scripts.cache_mocheg_multimodal_features import (
    select_visual_candidates,
    visual_candidate_features,
)
from scripts.train_mocheg_multimodal_verifier import (
    MultimodalEvidenceDataset,
    load_text_teacher,
    training_objective,
    validate_cache_pair as validate_multimodal_cache_pair,
)
from scripts.train_mocheg_staged_multimodal import visual_selection_objective


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


def test_frozen_evaluator_requires_matching_passed_validation_seeds():
    freeze = {
        "status": "frozen_after_validation",
        "frozen_seeds": [13, 42],
        "validation_gate": {
            "minimum_mean_macro_f1": 0.5,
            "maximum_macro_f1_std": 0.02,
        },
    }
    validation = {
        "split": "val",
        "test_split_used": False,
        "seeds": [13, 42],
        "stability_gate": {"passed": True},
        "aggregate": {"macro_f1": {"mean": 0.53, "std": 0.01}},
    }
    assert validate_freeze(freeze, validation) == [13, 42]

    validation["test_split_used"] = True
    try:
        validate_freeze(freeze, validation)
    except ValueError as error:
        assert "validation-only" in str(error)
    else:
        raise AssertionError("test-contaminated validation summary was accepted")


def test_visual_retrieval_reports_raw_and_annotation_conditional_recall():
    result = retrieval_summary(
        ranks=[1, None, 7, None],
        annotated=[True, False, True, True],
        cutoffs=[1, 10],
    )
    assert result["claims"] == 4
    assert result["claims_with_gold_images"] == 3
    assert result["recall@1"] == 0.25
    assert result["conditional_recall@1"] == 1 / 3
    assert result["recall@10"] == 0.5
    assert result["conditional_recall@10"] == 2 / 3


def test_visual_retrieval_normalizes_unsupported_image_magic(tmp_path):
    from PIL import Image

    jpeg = tmp_path / "valid.jpg"
    disguised_bmp = tmp_path / "actually-bmp.jpg"
    Image.new("RGB", (4, 4), "red").save(jpeg, format="JPEG")
    Image.new("RGB", (4, 4), "blue").save(disguised_bmp, format="BMP")

    assert is_torchvision_supported_image(jpeg)
    assert not is_torchvision_supported_image(disguised_bmp)
    paths, converted = normalize_image_paths(
        [str(jpeg), str(disguised_bmp)], tmp_path / "cache"
    )
    assert converted == 1
    assert paths[0] == str(jpeg)
    assert Path(paths[1]).suffix == ".png"
    assert is_torchvision_supported_image(Path(paths[1]))


def test_visual_retrieval_letterboxes_extreme_aspect_ratio(tmp_path):
    from PIL import Image

    for name, size in (("wide.png", (232, 1)), ("tall.png", (1, 232))):
        source = tmp_path / name
        Image.new("RGB", size, "red").save(source)
        normalized, converted = normalize_image_paths(
            [str(source)], tmp_path / "cache"
        )
        assert converted == 1
        assert Path(normalized[0]) != source
        with Image.open(normalized[0]) as result:
            width, height = result.size
        assert max(width, height) / min(width, height) < 200
        assert max(width, height) == 232


def test_visual_retrieval_image_embedding_cache_is_resumable(tmp_path):
    from PIL import Image

    paths = []
    for index in range(5):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (2, 2), (index, 0, 0)).save(path)
        paths.append(str(path))

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def encode(self, inputs, **kwargs):
            self.calls += 1
            return np.full((len(inputs), 3), self.calls, dtype=np.float32)

    model = FakeModel()
    names = [Path(path).name for path in paths]
    first = encode_images(
        model, names, paths, tmp_path / "cache", "val", "fake", 2, 2
    )
    assert first.shape == (5, 3)
    assert model.calls == 3

    cached_model = FakeModel()
    second = encode_images(
        cached_model, names, paths, tmp_path / "cache", "val", "fake", 2, 2
    )
    assert np.array_equal(first, second)
    assert cached_model.calls == 0


def test_visual_retrieval_query_instruction_uses_prompt_api(tmp_path):
    class FakeModel:
        def __init__(self):
            self.inputs = None
            self.kwargs = None

        def encode(self, inputs, **kwargs):
            self.inputs = inputs
            self.kwargs = kwargs
            return np.ones((len(inputs), 4), dtype=np.float32)

    claims = [
        {"id": "a", "claim": "first claim"},
        {"id": "b", "claim": "second claim"},
    ]
    model = FakeModel()
    result = encode_queries(
        model, claims, tmp_path, "val", "fake", "Retrieve evidence.", 2
    )
    assert result.shape == (2, 4)
    assert model.inputs == ["first claim", "second claim"]
    assert model.kwargs["prompt"] == "Retrieve evidence."


def test_constraint_visual_fusion_rewards_cross_view_agreement():
    fused = fuse_visual_orders(
        [np.array([3, 1, 2]), np.array([1, 3, 4])],
        [1.0, 1.0],
        rank_constant=1.0,
        limit=4,
    )
    assert fused[0][0] in {1, 3}
    assert {index for index, _ in fused} == {1, 2, 3, 4}


def test_aligned_gold_images_ignores_non_corpus_ids():
    gold = aligned_gold_images({
        "image_candidate_names": ["candidate.jpg", "missing.jpg"],
        "image_evidence_ids": ["proof.jpg"],
    }, {"candidate.jpg", "proof.jpg"})
    assert gold == {"candidate.jpg", "proof.jpg"}


def test_pixel_descriptor_signature_and_prompt_are_deterministic(tmp_path):
    first = descriptor_signature("model", "prompt", "corpus", 10, 20)
    second = descriptor_signature("model", "prompt", "corpus", 10, 20)
    changed = descriptor_signature("model", "prompt changed", "corpus", 10, 20)
    assert first == second
    assert first != changed
    chat = conversations(["image.png"], "describe")[0][0]
    assert chat["content"][0] == {"type": "image", "image": "image.png"}
    assert chat["content"][1] == {"type": "text", "text": "describe"}


def test_pixel_descriptor_resume_and_alignment(tmp_path):
    path = tmp_path / "val.jsonl"
    rows = [
        {"image_id": "a.jpg", "descriptor": "red car",
         "descriptor_signature": "sig"},
        {"image_id": "b.jpg", "descriptor": "blue bus",
         "descriptor_signature": "sig"},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert set(read_completed(path)) == {"a.jpg", "b.jpg"}
    descriptors, signature = read_descriptors(path, ["b.jpg", "a.jpg"])
    assert descriptors == ["blue bus", "red car"]
    assert signature == "sig"


def test_pixel_descriptor_cleanup_is_single_line():
    assert clean_descriptor(" Visual: car\n\nText: 42\x00 Type/Clues：photo ") == \
           "Visual: car Text: 42 Type/Clues:photo"


def test_caption_candidate_union_reports_complementary_recoveries():
    result = candidate_union_diagnostics(
        direct_hits=[True, False, False, False],
        caption_hits=[False, True, False, True],
        union_hits=[True, True, False, True],
        annotated=[True, True, True, False],
    )
    assert result["conditional_direct_candidate_recall"] == 1 / 3
    assert result["conditional_caption_candidate_recall"] == 1 / 3
    assert result["conditional_union_candidate_recall"] == 2 / 3
    assert result["caption_only_gold_recoveries"] == 1


def test_visual_reranker_is_stable_and_keeps_scores_aligned():
    ids, scores = stable_rerank(
        ["a.jpg", "b.jpg", "c.jpg"],
        np.asarray([0.2, 0.8, 0.8], dtype=np.float32),
        output_k=2,
    )
    assert ids == ["b.jpg", "c.jpg"]
    assert np.allclose(scores, [0.8, 0.8])


def test_visual_reranker_resume_ignores_only_partial_final_line(tmp_path):
    path = tmp_path / "val.jsonl"
    path.write_text('{"id":"one"}\n{"id":', encoding="utf-8")
    assert read_resumable_jsonl(path) == [{"id": "one"}]


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


def test_multimodal_evidence_head_and_loss_are_finite():
    head = MultimodalEvidenceHead(
        claim_dim=8,
        text_dim=8,
        visual_dim=10,
        hidden_dim=16,
    )
    text_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    visual_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    text_relevance = torch.tensor(
        [[1, 0, 0], [0, 1, 0]], dtype=torch.float32
    )
    visual_relevance = torch.tensor(
        [[0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32
    )
    output = head(
        claim=torch.randn(2, 8),
        text_evidence=torch.randn(2, 3, 8),
        text_mask=text_mask,
        text_retrieval_features=torch.randn(2, 3, 6),
        visual_evidence=torch.randn(2, 4, 10),
        visual_mask=visual_mask,
        visual_retrieval_features=torch.randn(2, 4, 3),
    )
    loss, parts = multimodal_evidence_loss(
        output,
        labels=torch.tensor([0, 1]),
        text_relevance=text_relevance,
        text_mask=text_mask,
        text_relevance_weights=torch.ones(2, 3),
        visual_relevance=visual_relevance,
        visual_mask=visual_mask,
        visual_relevance_weights=torch.ones(2, 4),
    )
    assert output["verdict_logits"].shape == (2, 3)
    assert output["text_attention"].shape == (2, 3)
    assert output["visual_attention"].shape == (2, 4)
    assert output["visual_expert_logits"].shape == (2, 3)
    assert torch.allclose(
        output["verdict_logits"], output["text_verdict_logits"]
    )
    assert torch.allclose(
        output["modality_mass"].sum(-1), torch.ones(2), atol=1e-5
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    selector_loss, selector_parts = visual_selection_objective(
        output,
        {"visual_relevance": visual_relevance, "labels": torch.tensor([0, 1])},
        visual_mask,
        stance_weight=0.15,
    )
    assert torch.isfinite(selector_loss)
    assert all(np.isfinite(value) for value in selector_parts.values())


def test_visual_train_candidate_injection_is_deterministic():
    retrieved = ["a.jpg", "b.jpg", "c.jpg"]
    gold = {"gold-1.jpg", "gold-2.jpg"}
    first = select_visual_candidates("claim-1", retrieved, gold, 4, True)
    second = select_visual_candidates("claim-1", retrieved, gold, 4, True)
    assert first == second
    assert gold.issubset(first)
    assert len(first) == len(set(first)) == 4
    assert select_visual_candidates(
        "claim-1", retrieved, gold, 2, False
    ) == retrieved[:2]
    assert visual_candidate_features("injected.jpg", retrieved, [3, 2, 1]) == \
        [0.0, 0.0, 0.0]
    assert visual_candidate_features("a.jpg", retrieved, [3, 2, 1])[0] == 1.0


def test_multimodal_dataset_maps_cache_names_to_model_arguments(tmp_path):
    payload = {
        "ids": ["a", "b"],
        "claim_embeddings": torch.randn(2, 8).half(),
        "text_evidence_embeddings": torch.randn(2, 3, 8).half(),
        "text_mask": torch.ones(2, 3, dtype=torch.bool),
        "text_retrieval_features": torch.randn(2, 3, 6),
        "text_relevance": torch.zeros(2, 3),
        "text_relevance_weights": torch.ones(2, 3),
        "visual_evidence_embeddings": torch.randn(2, 4, 10).half(),
        "visual_mask": torch.ones(2, 4, dtype=torch.bool),
        "visual_retrieval_features": torch.randn(2, 4, 3),
        "visual_relevance": torch.zeros(2, 4),
        "visual_relevance_weights": torch.ones(2, 4),
        "labels": torch.tensor([0, 1]),
        "metadata": {
            "claim_dim": 8,
            "text_dim": 8,
            "visual_dim": 10,
            "text_top_k": 3,
            "visual_top_k": 4,
            "visual_model": "test-model",
            "train_gold_injection": True,
            "validation_gold_injection": False,
        },
    }
    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    torch.save(payload, train_path)
    val_payload = dict(payload)
    val_payload["metadata"] = dict(payload["metadata"])
    val_payload["metadata"]["train_gold_injection"] = False
    torch.save(val_payload, val_path)
    train = MultimodalEvidenceDataset(train_path)
    val = MultimodalEvidenceDataset(val_path)
    validate_multimodal_cache_pair(train, val)
    assert train[0]["claim"].shape == (8,)
    assert train[0]["text_evidence"].shape == (3, 8)
    assert train[0]["visual_evidence"].shape == (4, 10)


def test_multimodal_training_objective_preserves_both_masks():
    head = MultimodalEvidenceHead(
        claim_dim=8, text_dim=8, visual_dim=10, hidden_dim=16
    )
    batch = {
        "id": ["a", "b"],
        "claim": torch.randn(2, 8),
        "text_evidence": torch.randn(2, 3, 8),
        "text_mask": torch.ones(2, 3, dtype=torch.bool),
        "text_retrieval_features": torch.randn(2, 3, 6),
        "visual_evidence": torch.randn(2, 4, 10),
        "visual_mask": torch.ones(2, 4, dtype=torch.bool),
        "visual_retrieval_features": torch.randn(2, 4, 3),
        "text_relevance": torch.tensor([[1, 0, 0], [0, 1, 0]]).float(),
        "text_relevance_weights": torch.ones(2, 3),
        "visual_relevance": torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0]]
        ).float(),
        "visual_relevance_weights": torch.ones(2, 4),
        "labels": torch.tensor([0, 1]),
    }
    loss, parts = training_objective(
        head, batch, torch.device("cpu"), torch.ones(3)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_multimodal_text_teacher_transfer_preserves_predictions(tmp_path):
    teacher = EvidenceSetHead(
        encoder_dim=8, hidden_dim=16, retrieval_dim=6, dropout=0.0
    ).eval()
    checkpoint = tmp_path / "teacher.pt"
    torch.save({
        "head": teacher.state_dict(),
        "seed": 42,
        "best_val_macro_f1": 0.55,
    }, checkpoint)
    student = MultimodalEvidenceHead(
        claim_dim=8,
        text_dim=8,
        visual_dim=10,
        hidden_dim=16,
        dropout=0.0,
    ).eval()
    provenance = load_text_teacher(student, checkpoint)
    claim = torch.randn(2, 8)
    evidence = torch.randn(2, 3, 8)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    retrieval = torch.randn(2, 3, 6)
    teacher_output = teacher(claim, evidence, mask, retrieval)
    student_output = student(
        claim=claim,
        text_evidence=evidence,
        text_mask=mask,
        text_retrieval_features=retrieval,
        visual_evidence=torch.zeros(2, 4, 10),
        visual_mask=torch.zeros(2, 4, dtype=torch.bool),
        visual_retrieval_features=torch.zeros(2, 4, 3),
    )
    assert provenance["transferred_tensors"] == len(teacher.state_dict())
    assert torch.allclose(
        student_output["verdict_logits"],
        teacher_output["verdict_logits"],
        atol=1e-6,
    )
    assert torch.allclose(
        student_output["text_attention"], teacher_output["attention"], atol=1e-6
    )


def test_visual_expert_is_independent_of_router_gate():
    head = MultimodalEvidenceHead(
        claim_dim=8, text_dim=8, visual_dim=10, hidden_dim=16, dropout=0.0
    ).eval()
    with torch.no_grad():
        head.visual_residual[-1].weight.zero_()
        head.visual_residual[-1].bias.copy_(torch.tensor([1.0, -1.0, 0.5]))
        head.visual_gate[-1].weight.zero_()
        head.visual_gate[-1].bias.fill_(-10.0)
    inputs = {
        "claim": torch.randn(2, 8),
        "text_evidence": torch.randn(2, 3, 8),
        "text_mask": torch.ones(2, 3, dtype=torch.bool),
        "text_retrieval_features": torch.randn(2, 3, 6),
        "visual_evidence": torch.randn(2, 4, 10),
        "visual_mask": torch.ones(2, 4, dtype=torch.bool),
        "visual_retrieval_features": torch.randn(2, 4, 3),
    }
    low_gate = head(**inputs)
    with torch.no_grad():
        head.visual_gate[-1].bias.fill_(10.0)
    high_gate = head(**inputs)
    assert torch.allclose(
        low_gate["visual_expert_logits"], high_gate["visual_expert_logits"]
    )
    assert not torch.allclose(
        low_gate["verdict_logits"], high_gate["verdict_logits"]
    )
