import torch
import json
import numpy as np
import pytest
from pathlib import Path

from graphcure.acquisition import choose_evi_action
from graphcure.losses import counterfactual_loss, directional_intervention_loss
from graphcure.model import GraphCURE, GraphCUREConfig
from graphcure.data import PackedEmbeddingDataset, resolve_newsclippings_source
from graphcure.optimization import project_auxiliary_gradients
from scripts.prepare_mocheg_protocols import close_row
from graphcure.evidence_set import EvidenceSetHead, evidence_set_loss, last_token_pool
from graphcure.selective_residual import SelectiveResidualSetVerifier
from graphcure.multimodal_evidence import (
    MultimodalEvidenceHead,
    multimodal_evidence_loss,
)
from graphcure.token_visual import (
    aggregate_pair_logits,
    select_candidate_ids as select_token_visual_candidates,
    token_visual_loss,
)
from scripts.analyze_mocheg_claim_images import (
    PROMPT as CLAIM_IMAGE_PROMPT,
    conversations as claim_image_conversations,
    report_signature,
)
from scripts.cache_mocheg_visual_report_features import report_features
from graphcure.report_fusion import SafeReportFusion, fusion_features
from scripts.train_mocheg_long_context_verifier import compose_example
from scripts.train_mocheg_nli_set_verifier import (
    aggregate_nli_logits,
    select_evidence_candidates,
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
from scripts.train_mocheg_staged_multimodal import (
    hard_router_metrics,
    validate_router_cache,
    visual_selection_objective,
)
from scripts.audit_mocheg_router import audit, exact_mcnemar_p
from scripts.train_mocheg_set_router import (
    route_metrics,
    select_threshold,
    utility_labels,
)
from scripts.audit_mocheg_visual_selector import rank_summary
from scripts.audit_mocheg_visual_expert import audit as audit_visual_expert
from scripts.analyze_mocheg_expert_complementarity import (
    diagnose,
    prediction_metrics,
)
from scripts.summarize_mocheg_packet_ensemble import summarize as summarize_packet
from scripts.cache_mocheg_reasoning_features import inject_gold_candidate
from scripts.prepare_mocheg_atomic_evidence import (
    map_unit_id,
    normalized_text,
    official_context_windows,
    pack_context,
    split_atomic_units,
    stable_atom_id,
)
from scripts.run_mocheg_atomic_retrieval import (
    diverse_order,
    file_sha256,
    token_overlap,
)
from scripts.train_mocheg_qwen3_lora_verifier import (
    as_token_id_list,
    compose_user_prompt,
)
from scripts.prepare_mocheg_sufficiency_targets import sufficiency_target
from scripts.train_mocheg_qwen3_hierarchical_lora import (
    B6TrainingTasks,
    add_gradient_buffers,
    blend_probabilities as blend_b6_probabilities,
    hierarchical_probabilities,
    load_frozen_anchor_predictions,
    probability_metrics as b6_probability_metrics,
)
from scripts.summarize_mocheg_b6_hierarchical import summarize as summarize_b6
from scripts.analyze_mocheg_b6_auxiliary_control import analyze as analyze_b6_auxiliary
from scripts.summarize_mocheg_b6a_confirmation import summarize as summarize_b6a
from scripts.analyze_mocheg_b6b_pcgrad_screen import summarize as summarize_b6b
from scripts.prepare_mocheg_sv_folds import build_folds, claim_family
from scripts.train_mocheg_sv_lora import (
    GroupDROState,
    balanced_class_weights,
    deterministic_fraction,
    hierarchical_verification_loss,
    robust_group_key,
)
from scripts.analyze_mocheg_sv_complementarity import add_logit_bias
from scripts.summarize_mocheg_sv_confirmation import paired_fold
from scripts.summarize_mocheg_qwen3_lora import (
    apply_temperature,
    fit_temperature,
    probability_metrics,
)
from scripts.evaluate_mocheg_qwen3_frozen_test import (
    bootstrap_ci,
    validate_protocol_inputs,
)


def test_selective_residual_starts_as_exact_frozen_anchor():
    anchor = EvidenceSetHead(
        encoder_dim=8, hidden_dim=16, retrieval_dim=6, dropout=0.0
    ).eval()
    model = SelectiveResidualSetVerifier(
        anchor, encoder_dim=8, hidden_dim=16, layers=1, heads=4, dropout=0.0
    ).eval()
    inputs = {
        "claim": torch.randn(3, 8),
        "evidence": torch.randn(3, 4, 8),
        "evidence_mask": torch.tensor(
            [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool
        ),
        "retrieval_features": torch.randn(3, 4, 6),
    }
    expected = anchor(**inputs)["verdict_logits"]
    actual = model(**inputs)
    assert torch.allclose(actual["verdict_logits"], expected, atol=1e-6)
    assert all(not parameter.requires_grad for parameter in model.anchor.parameters())
    assert actual["set_attention"].shape == (3, 4)


def test_train_gold_injection_is_deterministic_and_never_changes_covered_rows():
    claim = {"id": "claim-7", "text_evidence_ids": ["gold-b", "gold-a"]}
    documents = {"gold-a": "a", "gold-b": "b", "negative": "n"}
    first, changed = inject_gold_candidate(claim, ["negative"], documents, 2)
    second, repeated = inject_gold_candidate(claim, ["negative"], documents, 2)
    assert changed and repeated and first == second
    assert set(first) & {"gold-a", "gold-b"}
    covered, changed = inject_gold_candidate(
        claim, ["gold-a", "negative"], documents, 2
    )
    assert covered == ["gold-a", "negative"] and not changed


def test_qwen_verifier_prompt_contains_no_supervision_metadata():
    prompt = compose_user_prompt(
        "The claim text", ["first article", "second article"], 100
    )
    assert "The claim text" in prompt and "[1] first article" in prompt
    assert "[2] second article" in prompt and "A, B, or C" in prompt
    assert "qrel" not in prompt.lower() and "gold" not in prompt.lower()


def test_qwen_tokenizer_output_normalization_supports_encoding_objects():
    class Encoding:
        ids = [11, 12, 13]

    class BatchEncoding:
        input_ids = torch.tensor([[21, 22]])

    assert as_token_id_list(Encoding()) == [11, 12, 13]
    assert as_token_id_list([Encoding()]) == [11, 12, 13]
    assert as_token_id_list([7, Encoding()]) == [7, 11, 12, 13]
    assert as_token_id_list(BatchEncoding()) == [21, 22]


def test_sv_hierarchical_loss_rewards_correct_sufficiency_and_polarity():
    labels = torch.tensor([0, 1, 2])
    sample_weights = torch.ones(3)
    good = torch.tensor([[5.0, 0.0, -2.0], [0.0, 5.0, -2.0], [-2.0, -2.0, 5.0]])
    bad = good.roll(1, dims=0)
    good_loss, parts = hierarchical_verification_loss(
        good, labels, sample_weights, torch.ones(3), 0.5, 0.25
    )
    bad_loss, _ = hierarchical_verification_loss(
        bad, labels, sample_weights, torch.ones(3), 0.5, 0.25
    )
    assert good_loss < bad_loss
    assert set(parts) == {"verdict", "sufficiency", "polarity"}
    assert all(torch.isfinite(value) for value in parts.values())


def test_sv_counterfactual_sampling_and_weights_are_deterministic():
    assert deterministic_fraction("claim-1", 42) == deterministic_fraction("claim-1", 42)
    weights = balanced_class_weights([0, 0, 0, 1, 2], "sqrt")
    assert weights[0] < weights[1] and weights[0] < weights[2]
    assert torch.isclose(weights.mean(), torch.tensor(1.0))


def test_sv_folds_keep_duplicate_claim_families_together():
    rows = []
    for index in range(30):
        rows.append({
            "id": f"row-{index}", "label": index % 3,
            "claim": f"Claim family {index // 2}", "snopes_url": "",
        })
    folds = build_folds(rows, folds=3, seed=7)
    by_id = {row["id"]: row for row in rows}
    for fold in folds:
        train_families = {claim_family(by_id[value]) for value in fold["train_ids"]}
        val_families = {claim_family(by_id[value]) for value in fold["val_ids"]}
        assert not train_families & val_families


def test_sv_nei_logit_bias_changes_only_relative_nei_odds():
    probability = np.asarray([[0.4, 0.3, 0.3], [0.1, 0.2, 0.7]])
    adjusted = add_logit_bias(probability, class_index=2, bias=-1.0)
    assert np.allclose(adjusted.sum(-1), 1.0)
    assert np.all(adjusted[:, 2] < probability[:, 2])
    assert np.allclose(
        adjusted[:, 0] / adjusted[:, 1],
        probability[:, 0] / probability[:, 1],
    )


def test_sv_frozen_confirmation_aligns_ids_before_interpolation(tmp_path):
    flat = [
        {"id": "b", "gold": 1, "probabilities": [0.1, 0.8, 0.1]},
        {"id": "a", "gold": 0, "probabilities": [0.6, 0.2, 0.2]},
        {"id": "c", "gold": 2, "probabilities": [0.2, 0.2, 0.6]},
    ]
    hierarchical = [
        {"id": "c", "gold": 2, "probabilities": [0.1, 0.2, 0.7]},
        {"id": "a", "gold": 0, "probabilities": [0.8, 0.1, 0.1]},
        {"id": "b", "gold": 1, "probabilities": [0.2, 0.7, 0.1]},
    ]
    flat_path = tmp_path / "flat.jsonl"
    hierarchical_path = tmp_path / "hierarchical.jsonl"
    flat_path.write_text("\n".join(map(json.dumps, flat)) + "\n")
    hierarchical_path.write_text(
        "\n".join(map(json.dumps, hierarchical)) + "\n"
    )
    result = paired_fold(flat_path, hierarchical_path, alpha=0.63)
    assert result["samples"] == 3
    assert result["frozen_ensemble"]["macro_f1"] == 1.0


def test_evidence_availability_group_dro_upweights_high_loss_group():
    state = GroupDROState(
        ["snopes|qrel_absent", "snopes|qrel_available"] * 4,
        eta=0.5,
        device=torch.device("cpu"),
    )
    loss = state.loss(
        torch.tensor([3.0, 0.2]),
        ["snopes|qrel_absent", "snopes|qrel_available"],
    )
    assert torch.isfinite(loss)
    state.update()
    weights = state.as_dict()
    assert weights["snopes|qrel_absent"] > weights["snopes|qrel_available"]
    assert robust_group_key("snopes", False, "source_qrel") == \
        "snopes|qrel_absent"
    assert as_token_id_list(np.asarray([31, 32])) == [31, 32]


def test_qwen_temperature_scaling_is_positive_and_preserves_predictions():
    probabilities = np.asarray([[0.98, 0.01, 0.01], [0.05, 0.90, 0.05]])
    labels = np.asarray([0, 1])
    temperature = fit_temperature(probabilities, labels)
    calibrated = apply_temperature(probabilities, temperature)
    assert temperature > 0
    assert np.array_equal(calibrated.argmax(1), probabilities.argmax(1))
    assert probability_metrics(calibrated, labels)["accuracy"] == 1.0


def test_frozen_test_bootstrap_is_deterministic():
    probabilities = np.asarray([[0.9,0.05,0.05],[0.1,0.8,0.1],[0.1,0.2,0.7]])
    labels = np.asarray([0,1,2])
    first = bootstrap_ci(probabilities, labels, iterations=20, seed=7)
    second = bootstrap_ci(probabilities, labels, iterations=20, seed=7)
    assert first == second


def test_frozen_test_protocol_rejects_missing_retrieval(tmp_path):
    manifest=tmp_path / "manifest.jsonl"
    retrieval=tmp_path / "retrieval.jsonl"
    manifest.write_text('{"id":"a","label":0}\n{"id":"b","label":1}\n')
    retrieval.write_text('{"id":"a","label":0}\n')
    with pytest.raises(ValueError, match="do not match exactly"):
        validate_protocol_inputs(manifest,retrieval,expected_samples=2)


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


def test_token_visual_candidates_and_joint_loss_are_finite():
    selected = select_token_visual_candidates(
        "claim-1", ["a.jpg", "b.jpg", "c.jpg"], {"gold.jpg"}, 3, True
    )
    assert "gold.jpg" in selected and len(selected) == len(set(selected)) == 3
    assert select_token_visual_candidates(
        "claim-1", ["a.jpg", "b.jpg"], {"gold.jpg"}, 2, False
    ) == ["a.jpg", "b.jpg"]
    pair = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    relevance = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool)
    labels = torch.tensor([0, 2])
    claim, attention = aggregate_pair_logits(
        pair, mask, torch.log(torch.tensor([[1., .5, .25], [1., .5, 1.]]))
    )
    loss, parts = token_visual_loss(
        pair, claim, labels, relevance, mask, relevance.any(1), .25, 1.0
    )
    loss.backward()
    assert claim.shape == (2, 3)
    assert torch.allclose(attention.sum(-1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(loss) and all(np.isfinite(x) for x in parts.values())


def test_claim_image_analyzer_prompt_has_no_label_or_qrel_input():
    chats = claim_image_conversations(
        ["A dated photograph shows an event."], ["image.jpg"]
    )
    prompt = chats[0][0]["content"][1]["text"]
    assert "A dated photograph" in prompt
    assert "final supported/refuted/NEI label" in CLAIM_IMAGE_PROMPT
    assert "qrel" not in prompt.lower()
    assert report_signature("model", 2, 100, 20) == report_signature(
        "model", 2, 100, 20
    )


def test_visual_report_features_preserve_injected_unknown_rank():
    features = report_features([0, 2], [0.0, 8.0])
    assert features[0] == [0.0, 0.0, 0.0]
    assert features[1] == [0.5, 1.0, 0.5]


def test_safe_report_fusion_outputs_bounded_gate():
    output = {
        "text_verdict_logits": torch.randn(4, 3),
        "visual_expert_logits": torch.randn(4, 3),
        "conflict": torch.rand(4, 3),
        "sufficiency_logit": torch.randn(4),
        "visual_attention": torch.softmax(torch.randn(4, 2), -1),
    }
    features = fusion_features(output)
    fusion = SafeReportFusion(features.shape[-1], hidden_dim=8)
    logits, gate = fusion(
        output["text_verdict_logits"], output["visual_expert_logits"], features
    )
    assert logits.shape == (4, 3)
    assert torch.all((gate > 0) & (gate < 1))


def test_long_context_input_contains_evidence_without_gold_metadata():
    text = compose_example(
        "A claim", ["First document", "Second document"],
        ["Visible report"], 100
    )
    assert "Claim:\nA claim" in text
    assert "Retrieved text evidence 2" in text
    assert "Retrieved visual evidence report 1" in text
    assert "qrel" not in text.lower() and "gold" not in text.lower()


def test_nli_set_train_injection_and_aggregation():
    selected = select_evidence_candidates(
        "claim", ["a", "b", "c"], {"gold"}, 3, True
    )
    assert "gold" in selected and len(selected) == len(set(selected)) == 3
    pair = torch.randn(2, 3, 3, requires_grad=True)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    claim = aggregate_nli_logits(pair, mask, temperature=.5)
    assert claim.shape == (2, 3) and torch.isfinite(claim).all()
    claim.sum().backward()


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
    validate_router_cache(train, val)
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


def test_retrieval_attention_preserves_upstream_rank_order():
    head = MultimodalEvidenceHead(
        claim_dim=8, text_dim=8, visual_dim=10, hidden_dim=16, dropout=0.0,
        visual_attention_mode="retrieval",
    ).eval()
    inputs = {
        "claim": torch.randn(1, 8),
        "text_evidence": torch.randn(1, 2, 8),
        "text_mask": torch.ones(1, 2, dtype=torch.bool),
        "text_retrieval_features": torch.randn(1, 2, 6),
        "visual_evidence": torch.randn(1, 3, 10),
        "visual_mask": torch.ones(1, 3, dtype=torch.bool),
        "visual_retrieval_features": torch.tensor([[
            [1.0, 1.0, 1.0],
            [0.5, 0.7, 0.5],
            [1 / 3, 0.2, 0.0],
        ]]),
    }
    first = head(**inputs)["visual_attention"]
    with torch.no_grad():
        for parameter in head.visual_utility.parameters():
            parameter.add_(100 * torch.randn_like(parameter))
    second = head(**inputs)["visual_attention"]
    assert torch.allclose(first, second)
    assert first.argmax(-1).item() == 0
    assert first[0, 0] > first[0, 1] > first[0, 2]


def test_stance_product_falls_back_to_text_when_insufficient():
    head = MultimodalEvidenceHead(
        claim_dim=8, text_dim=8, visual_dim=10, hidden_dim=16, dropout=0.0,
        visual_attention_mode="retrieval", visual_expert_mode="stance_product",
    ).eval()
    inputs = {
        "claim": torch.randn(2, 8),
        "text_evidence": torch.randn(2, 3, 8),
        "text_mask": torch.ones(2, 3, dtype=torch.bool),
        "text_retrieval_features": torch.randn(2, 3, 6),
        "visual_evidence": torch.randn(2, 4, 10),
        "visual_mask": torch.ones(2, 4, dtype=torch.bool),
        "visual_retrieval_features": torch.rand(2, 4, 3).clamp_min(0.1),
    }
    with torch.no_grad():
        head.sufficiency[-1].weight.zero_()
        head.sufficiency[-1].bias.fill_(-20.0)
    insufficient = head(**inputs)
    assert torch.allclose(
        insufficient["visual_expert_logits"],
        insufficient["text_verdict_logits"],
        atol=1e-6,
    )
    with torch.no_grad():
        head.sufficiency[-1].bias.fill_(20.0)
    sufficient = head(**inputs)
    assert not torch.allclose(
        sufficient["visual_expert_logits"],
        sufficient["text_verdict_logits"],
    )


def test_hard_router_selects_specialist_without_logit_blending():
    rows = [
        {"gold": 0, "text_only_prediction": 1,
         "visual_expert_prediction": 0, "visual_modality_mass": 0.9},
        {"gold": 1, "text_only_prediction": 1,
         "visual_expert_prediction": 0, "visual_modality_mass": 0.2},
        {"gold": 2, "text_only_prediction": 2,
         "visual_expert_prediction": 1, "visual_modality_mass": 0.1},
    ]
    metrics = hard_router_metrics(rows, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["visual_route_rate"] == 1 / 3


def test_hard_router_uses_selected_expert_probabilities_for_ece():
    rows = [
        {"gold": 0, "text_only_prediction": 1,
         "visual_expert_prediction": 0, "visual_modality_mass": 0.9,
         "text_only_probabilities": [0.1, 0.8, 0.1],
         "visual_expert_probabilities": [0.9, 0.05, 0.05]},
        {"gold": 1, "text_only_prediction": 1,
         "visual_expert_prediction": 0, "visual_modality_mass": 0.2,
         "text_only_probabilities": [0.1, 0.8, 0.1],
         "visual_expert_probabilities": [0.9, 0.05, 0.05]},
    ]
    metrics = hard_router_metrics(rows, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert abs(metrics["ece_10"] - 0.15) < 1e-6


def test_router_audit_reports_paired_help_and_harm():
    rows = [
        {"gold": 0, "text_only_prediction": 1,
         "visual_expert_prediction": 0, "visual_modality_mass": 0.9},
        {"gold": 1, "text_only_prediction": 1,
         "visual_expert_prediction": 0, "visual_modality_mass": 0.8},
        {"gold": 2, "text_only_prediction": 2,
         "visual_expert_prediction": 1, "visual_modality_mass": 0.1},
        {"gold": 0, "text_only_prediction": 0,
         "visual_expert_prediction": 0, "visual_modality_mass": 0.2},
    ]
    result = audit(rows, threshold=0.5, iterations=50, seed=7)
    assert result["hard_router"]["helpful"] == 1
    assert result["hard_router"]["harmful"] == 1
    assert result["hard_router"]["route_count"] == 2
    assert result["gate_ranking"]["decisive_help_vs_harm"]["auroc"] == 1.0
    assert exact_mcnemar_p(1, 1) == 1.0


def test_set_router_utility_targets_and_train_only_threshold():
    gold = np.asarray([0, 1, 2, 0])
    text = np.asarray([1, 1, 2, 0])
    expert = np.asarray([0, 0, 1, 0])
    labels = utility_labels(gold, text, expert)
    assert labels.tolist() == [2, 0, 0, 1]
    score = np.asarray([0.9, 0.8, -0.5, -0.2])
    selected = select_threshold(gold, text, expert, score)
    assert selected["helpful"] == 1
    assert selected["harmful"] == 0
    assert selected["macro_f1"] == 1.0
    repeated = route_metrics(
        gold, text, expert, score, selected["threshold"]
    )
    assert repeated == selected


def test_visual_selector_audit_compares_orders_on_same_candidates():
    relevance = np.asarray([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    attention = np.asarray([
        [0.1, 0.2, 0.7],
        [0.2, 0.7, 0.1],
        [0.3, 0.3, 0.4],
    ])
    mask = np.ones_like(relevance, dtype=bool)
    result = rank_summary(relevance, attention, mask)
    assert result["claims_with_gold_in_candidates"] == 2
    assert result["upstream_order"]["hit_at_1"] == 0.5
    assert result["learned_attention_order"]["hit_at_1"] == 0.0
    assert result["pairwise_order_comparison"] == {
        "upstream_better": 1, "learned_better": 0, "tie": 1,
    }


def test_visual_expert_audit_stratifies_help_and_harm():
    rows = [
        {"gold": 0, "text_only_prediction": 1,
         "visual_expert_prediction": 0, "visual_gold_in_candidates": True,
         "visual_selected_gold": True},
        {"gold": 1, "text_only_prediction": 1,
         "visual_expert_prediction": 0, "visual_gold_in_candidates": False,
         "visual_selected_gold": False},
        {"gold": 2, "text_only_prediction": 2,
         "visual_expert_prediction": 2, "visual_gold_in_candidates": True,
         "visual_selected_gold": False},
    ]
    result = audit_visual_expert(rows)
    assert result["gold_in_candidates"]["helpful"] == 1
    assert result["gold_in_candidates"]["harmful"] == 0
    assert result["no_gold_in_candidates"]["helpful"] == 0
    assert result["no_gold_in_candidates"]["harmful"] == 1
    assert result["qrel_oracle_sufficiency_policy"]["accuracy"] == 1.0


def test_atomic_evidence_split_is_clean_and_deterministic():
    article = (
        "<p>Officials said the event happened in Paris on Monday.</p>"
        "<p>However, the viral photograph was first published in 2018! "
        "It did not show the claimed 2024 event.</p>"
    )
    units = split_atomic_units(article)
    assert units == [
        "Officials said the event happened in Paris on Monday.",
        "However, the viral photograph was first published in 2018!",
        "It did not show the claimed 2024 event.",
    ]
    assert stable_atom_id("doc-7", units[0]) == stable_atom_id("doc-7", units[0])
    assert normalized_text("Café <b>FALSE</b>!") == "caf false"


def test_atomic_evidence_prefers_unique_official_sentence_id():
    text = "The image was taken in 2019, not 2024."
    lookup = {("17", normalized_text(text)): ["17-82-3"]}
    assert map_unit_id("17", "article-1", text, lookup) == ("17-82-3", True)
    ambiguous = {("17", normalized_text(text)): ["17-82-3", "17-91-0"]}
    atom_id, authoritative = map_unit_id("17", "article-1", text, ambiguous)
    assert atom_id.startswith("atomic-")
    assert not authoritative


def test_atomic_diversity_and_overlap_are_inference_only():
    order = [0, 1, 2, 3, 4]
    parents = ["a", "a", "a", "b", "c"]
    assert diverse_order(order, parents, limit=4, max_per_parent=2) == [0, 1, 3, 4]
    assert token_overlap("Paris fire 2024", "Fire reported in Paris") > 0
    assert token_overlap("Paris fire", "unrelated dolphins") == 0


def test_atomic_context_packet_marks_selected_sentence_without_changing_id():
    units = ["Before sentence.", "Selected sentence.", "After sentence."]
    assert pack_context(units, 1, 0) == "Selected sentence."
    assert pack_context(units, 1, 1) == (
        "[Local context] Before sentence.\n"
        "[Selected evidence] Selected sentence.\n"
        "[Local context] After sentence."
    )
    with pytest.raises(ValueError):
        pack_context(units, 1, -1)


def test_official_context_windows_do_not_cross_article_boundaries():
    rows = {
        "7-3-0": {"claim_id": "7", "relevant_document_id": "3",
                  "paragraph_id": "0", "paragraph": "First."},
        "7-3-1": {"claim_id": "7", "relevant_document_id": "3",
                  "paragraph_id": "1", "paragraph": "Second."},
        "7-4-0": {"claim_id": "7", "relevant_document_id": "4",
                  "paragraph_id": "0", "paragraph": "Other article."},
    }
    windows = official_context_windows(rows, 1)
    assert "Second." in windows["7-3-0"]
    assert "Other article." not in windows["7-3-0"]


def test_atomic_input_hash_changes_with_evidence_text(tmp_path):
    path = tmp_path / "Corpus2.csv"
    path.write_text("evidence_id,Evidence\n1,first\n", encoding="utf-8")
    first = file_sha256(path)
    path.write_text("evidence_id,Evidence\n1,changed\n", encoding="utf-8")
    assert file_sha256(path) != first


def test_expert_complementarity_reports_packet_only_corrections():
    anchor = {
        "a": {"id": "a", "gold": 0, "probabilities": [.1, .8, .1]},
        "b": {"id": "b", "gold": 1, "probabilities": [.1, .8, .1]},
        "c": {"id": "c", "gold": 2, "probabilities": [.1, .1, .8]},
    }
    packet = {
        "a": {"id": "a", "gold": 0, "probabilities": [.9, .05, .05]},
        "b": {"id": "b", "gold": 1, "probabilities": [.7, .2, .1]},
        "c": {"id": "c", "gold": 2, "probabilities": [.1, .1, .8]},
    }
    result = diagnose(anchor, packet, minimum_delta=0, bootstrap_iterations=20,
                      seed=3)
    assert result["outcome_overlap"]["packet_only_correct"] == 1
    assert result["outcome_overlap"]["anchor_only_correct"] == 1
    assert result["oracle_expert_selection"]["accuracy"] == 1.0
    assert not result["test_split_used"]


def test_packet_multiseed_summary_uses_frozen_weight_and_no_test():
    def rows(first):
        return {
            "a": {"id": "a", "gold": 0, "probabilities": first},
            "b": {"id": "b", "gold": 1, "probabilities": [.1, .8, .1]},
            "c": {"id": "c", "gold": 2, "probabilities": [.1, .1, .8]},
        }
    pairs = [
        (13, rows([.1, .8, .1]), rows([.9, .05, .05])),
        (21, rows([.1, .8, .1]), rows([.9, .05, .05])),
    ]
    result = summarize_packet(
        pairs, packet_weight=.8, minimum_delta=0,
        maximum_delta_std=1, minimum_positive_seeds=2,
        bootstrap_iterations=20, bootstrap_seed=4,
    )
    assert result["packet_weight"] == .8
    assert result["paired_macro_f1_delta"]["positive_seeds"] == 2
    assert result["promotion_gate"]["passed"]
    assert not result["test_split_used"]


def test_b6_sufficiency_targets_are_evidence_conditioned():
    assert sufficiency_target(2, [], []) == 0
    assert sufficiency_target(2, ["gold"], ["gold"]) == 0
    assert sufficiency_target(0, [], ["candidate"]) is None
    assert sufficiency_target(1, ["gold"], ["other"]) == 0
    assert sufficiency_target(0, ["gold"], ["other", "gold"]) == 1


def test_b6_hierarchical_probability_product_and_blend():
    sufficiency = np.asarray([[.8, .2], [.1, .9]])
    polarity = np.asarray([[.75, .25], [.4, .6]])
    hierarchical = hierarchical_probabilities(sufficiency, polarity)
    assert np.allclose(hierarchical, [
        [.6, .2, .2], [.04, .06, .9],
    ])
    direct = np.asarray([[.3, .3, .4], [.2, .7, .1]])
    assert np.allclose(
        blend_b6_probabilities(direct, hierarchical, 0), direct
    )
    assert np.allclose(
        blend_b6_probabilities(direct, hierarchical, 1), hierarchical
    )
    with pytest.raises(ValueError):
        blend_b6_probabilities(direct, hierarchical, 1.1)


def test_b6_frozen_anchor_predictions_require_exact_alignment(tmp_path):
    path = tmp_path / "val_predictions.jsonl"
    rows = [
        {"id": "claim-a", "gold": 0, "probabilities": [.8, .1, .1]},
        {"id": "claim-b", "gold": 1, "probabilities": [.1, .8, .1]},
        {"id": "claim-c", "gold": 2, "probabilities": [.1, .1, .8]},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    metrics, probabilities = load_frozen_anchor_predictions(
        path, ["claim-a", "claim-b", "claim-c"], np.asarray([0, 1, 2])
    )
    assert metrics["macro_f1"] == 1
    assert probabilities.shape == (3, 3)
    with pytest.raises(ValueError, match="not aligned"):
        load_frozen_anchor_predictions(
            path, ["claim-b", "claim-a", "claim-c"], np.asarray([1, 0, 2])
        )


def test_b6_training_tasks_do_not_fabricate_unknown_sufficiency():
    claims = type("Claims", (), {})()
    claims.rows = [
        {
            "id": "supported-no-qrel", "label": 0,
            "sufficiency_target": None, "polarity_target": None,
            "prompts": {task: task for task in (
                "verdict", "sufficiency", "ablation", "polarity"
            )},
        },
        {
            "id": "refuted-with-gold", "label": 1,
            "sufficiency_target": 1, "polarity_target": 1,
            "prompts": {task: task for task in (
                "verdict", "sufficiency", "ablation", "polarity"
            )},
        },
        {
            "id": "nei", "label": 2,
            "sufficiency_target": 0, "polarity_target": None,
            "prompts": {task: task for task in (
                "verdict", "sufficiency", "ablation", "polarity"
            )},
        },
    ]
    tasks = B6TrainingTasks(
        claims, ablation_ratio=1, seed=42, verdict_weight=1,
        sufficiency_weight=.5, polarity_weight=.5, ablation_weight=.5,
    )
    assert tasks.counts == {
        "verdict": 3, "sufficiency": 2, "polarity": 1, "ablation": 1,
    }
    unknown_tasks = [
        row["task"] for row in tasks.rows
        if row["id"] == "supported-no-qrel"
    ]
    assert unknown_tasks == ["verdict"]


def test_b6_zero_auxiliary_weights_create_matched_direct_control():
    claims = type("Claims", (), {})()
    claims.rows = [{
        "id": "claim-1", "label": 0,
        "sufficiency_target": 1, "polarity_target": 0,
        "prompts": {task: task for task in (
            "verdict", "sufficiency", "ablation", "polarity"
        )},
    }]
    tasks = B6TrainingTasks(
        claims, ablation_ratio=1, seed=42, verdict_weight=1,
        sufficiency_weight=0, polarity_weight=0, ablation_weight=0,
    )
    assert tasks.counts == {"verdict": 1}
    tasks.repeat_verdict_to_length(5)
    assert len(tasks) == 5
    assert tasks.counts == {"verdict": 5}
    with pytest.raises(ValueError, match="non-negative"):
        B6TrainingTasks(
            claims, ablation_ratio=0, seed=42, verdict_weight=1,
            sufficiency_weight=-1, polarity_weight=0, ablation_weight=0,
        )


def test_b6_auxiliary_control_requires_gain_over_direct(tmp_path):
    labels = [0, 1, 2, 0, 1, 2]
    anchor_predictions = [0, 1, 1, 0, 1, 1]
    control_predictions = [0, 1, 2, 1, 1, 1]
    auxiliary_predictions = labels
    paths = {}
    for name, predictions in (
        ("anchor", anchor_predictions),
        ("direct_control", control_predictions),
        ("auxiliary", auxiliary_predictions),
    ):
        path = tmp_path / f"{name}.jsonl"
        rows = [
            {
                "id": f"claim-{index}", "gold": gold,
                "probabilities": np.eye(3)[prediction].tolist(),
            }
            for index, (gold, prediction) in enumerate(zip(labels, predictions))
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        paths[name] = path
    result = analyze_b6_auxiliary(
        paths, minimum_anchor_delta=0, minimum_control_delta=0,
        minimum_bootstrap_probability=0, iterations=20, seed=7,
    )
    assert result["auxiliary_vs_direct_control"]["macro_f1_delta"] > 0
    assert result["promotion_gate"]["passed"]
    assert not result["test_split_used"]


def test_b6a_confirmation_requires_consistent_auxiliary_gain():
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    anchor = np.eye(3)[[0, 1, 1, 0, 1, 1]]
    control = np.eye(3)[[0, 1, 2, 1, 1, 1]]
    auxiliary = np.eye(3)[labels]
    runs = [{
        "seed": seed,
        "labels": labels,
        "probabilities": {
            "anchor": anchor,
            "direct_control": control,
            "auxiliary": auxiliary,
        },
        "metrics": {
            "anchor": prediction_metrics(labels, anchor),
            "direct_control": prediction_metrics(labels, control),
            "auxiliary": prediction_metrics(labels, auxiliary),
        },
    } for seed in (13, 21)]
    result = summarize_b6a(
        runs, minimum_anchor_mean_delta=0,
        minimum_control_mean_delta=0, minimum_ensemble_delta=0,
        minimum_ensemble_f1=.9, minimum_positive_seeds=2,
        minimum_bootstrap_probability=0, bootstrap_iterations=20,
        bootstrap_seed=3,
    )
    assert result["paired_auxiliary_vs_direct_control"]["positive_seeds"] == 2
    assert result["promotion_gate"]["passed"]
    assert not result["test_split_used"]


def test_b6b_gradient_buffers_preserve_missing_components():
    first = (torch.tensor([1.0]), None, torch.tensor([2.0]))
    second = (torch.tensor([3.0]), torch.tensor([4.0]), None)
    combined = add_gradient_buffers(first, second)
    assert torch.equal(combined[0], torch.tensor([4.0]))
    assert torch.equal(combined[1], torch.tensor([4.0]))
    assert torch.equal(combined[2], torch.tensor([2.0]))


def test_b6b_screen_requires_active_conflict_protection_and_seed87_repair():
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    anchor = np.eye(3)[[0, 1, 1, 0, 1, 1]]
    control = np.eye(3)[[0, 1, 2, 1, 1, 1]]
    standard = np.eye(3)[[0, 1, 2, 0, 1, 1]]
    conflict = np.eye(3)[labels]
    runs = [{
        "seed": seed,
        "labels": labels,
        "probabilities": {
            "anchor": anchor,
            "direct_control": control,
            "standard_auxiliary": standard,
            "conflict_auxiliary": conflict,
        },
        "metrics": {
            "anchor": prediction_metrics(labels, anchor),
            "direct_control": prediction_metrics(labels, control),
            "standard_auxiliary": prediction_metrics(labels, standard),
            "conflict_auxiliary": prediction_metrics(labels, conflict),
        },
        "history": [{"epoch": 1, "gradient_conflict_rate": .2}],
    } for seed in (42, 87)]
    result = summarize_b6b(
        runs, minimum_control_mean_delta=0,
        minimum_seed87_control_delta=0,
        minimum_standard_mean_delta=0,
        minimum_bootstrap_probability=0,
        bootstrap_iterations=20, bootstrap_seed=9,
    )
    assert result["promotion_gate"]["pcgrad_active_in_both_seeds"]
    assert result["promotion_gate"]["passed"]
    assert not result["test_split_used"]


def test_b6_multiseed_summary_requires_ensemble_and_class_gains():
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    article = np.asarray([
        [.8, .1, .1], [.1, .8, .1], [.1, .7, .2],
        [.8, .1, .1], [.1, .8, .1], [.2, .7, .1],
    ])
    candidate = np.asarray([
        [.8, .1, .1], [.1, .8, .1], [.1, .1, .8],
        [.8, .1, .1], [.1, .8, .1], [.1, .1, .8],
    ])
    runs = [{
        "seed": seed, "hierarchical_weight": .5, "labels": labels,
        "article_probabilities": article,
        "candidate_probabilities": candidate,
        "article_metrics": b6_probability_metrics(article, labels),
        "candidate_metrics": b6_probability_metrics(candidate, labels),
    } for seed in (13, 21)]
    result = summarize_b6(
        runs, minimum_mean_delta=0, minimum_ensemble_delta=0,
        minimum_ensemble_f1=.9, minimum_nei_delta=0,
        maximum_supported_drop=0, minimum_positive_seeds=2,
    )
    assert result["promotion_gate"]["passed"]
    assert result["raw_ensemble_deltas"]["nei_f1"] > 0
    assert not result["test_split_used"]
