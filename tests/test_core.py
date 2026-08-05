import torch
import json
import numpy as np

from graphcure.acquisition import choose_evi_action
from graphcure.losses import counterfactual_loss
from graphcure.model import GraphCURE, GraphCUREConfig
from graphcure.data import PackedEmbeddingDataset, resolve_newsclippings_source


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
    np.save(tmp_path / "image_embeddings.npy", np.ones((2, 6), dtype=np.float32))
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
    np.save(tmp_path / "image_embeddings.npy", np.ones((2, 6), dtype=np.float32))
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
