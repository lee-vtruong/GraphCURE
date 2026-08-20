"""Compare learned visual attention with the frozen upstream retrieval order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.train_mocheg_multimodal_verifier import (
    MultimodalEvidenceDataset,
    collate,
)
from scripts.train_mocheg_set_router import load_expert


def rank_summary(
    relevance: np.ndarray,
    attention: np.ndarray,
    mask: np.ndarray,
) -> dict:
    rows = relevance.astype(bool).any(1)
    if not rows.any():
        raise ValueError("no visual-positive rows to audit")
    upstream_ranks: list[int] = []
    learned_ranks: list[int] = []
    positive_mass: list[float] = []
    upstream_better = 0
    learned_better = 0
    ties = 0
    for row in np.flatnonzero(rows):
        valid = np.flatnonzero(mask[row])
        gold = relevance[row].astype(bool)
        upstream_rank = int(np.flatnonzero(gold)[0]) + 1
        learned_order = valid[np.argsort(-attention[row, valid], kind="stable")]
        learned_rank = int(np.flatnonzero(gold[learned_order])[0]) + 1
        upstream_ranks.append(upstream_rank)
        learned_ranks.append(learned_rank)
        positive_mass.append(float(attention[row, gold].sum()))
        if upstream_rank < learned_rank:
            upstream_better += 1
        elif learned_rank < upstream_rank:
            learned_better += 1
        else:
            ties += 1

    def metrics(ranks: list[int]) -> dict:
        values = np.asarray(ranks)
        result = {
            "mrr": float(np.mean(1.0 / values)),
            "mean_first_gold_rank": float(values.mean()),
        }
        for cutoff in (1, 5, 10, 20, 32):
            result[f"hit_at_{cutoff}"] = float(np.mean(values <= cutoff))
        return result

    return {
        "samples": int(len(relevance)),
        "claims_with_gold_in_candidates": int(rows.sum()),
        "upstream_order": metrics(upstream_ranks),
        "learned_attention_order": metrics(learned_ranks),
        "learned_positive_attention_mass_mean": float(np.mean(positive_mass)),
        "pairwise_order_comparison": {
            "upstream_better": upstream_better,
            "learned_better": learned_better,
            "tie": ties,
        },
        "learned_minus_upstream": {
            "hit_at_1": float(
                np.mean(np.asarray(learned_ranks) <= 1)
                - np.mean(np.asarray(upstream_ranks) <= 1)
            ),
            "mrr": float(
                np.mean(1.0 / np.asarray(learned_ranks))
                - np.mean(1.0 / np.asarray(upstream_ranks))
            ),
        },
        "test_split_used": False,
    }


@torch.inference_mode()
def collect(
    checkpoint: Path,
    cache: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    dataset = MultimodalEvidenceDataset(cache)
    if dataset.metadata.get("split") == "test":
        raise ValueError("selector development audit must not use test")
    head, saved = load_expert(checkpoint, dataset.metadata, device)
    relevance: list[np.ndarray] = []
    attention: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    for batch in tqdm(loader, desc="audit visual selector"):
        batch.pop("id")
        batch.pop("labels")
        batch.pop("text_relevance")
        batch.pop("text_relevance_weights")
        visual_relevance = batch.pop("visual_relevance")
        batch.pop("visual_relevance_weights")
        model_input = {key: value.to(device) for key, value in batch.items()}
        output = head(**model_input)
        relevance.append(visual_relevance.numpy())
        attention.append(output["visual_attention"].float().cpu().numpy())
        masks.append(batch["visual_mask"].numpy())
    return (
        np.concatenate(relevance),
        np.concatenate(attention),
        np.concatenate(masks),
        saved,
    )


def markdown(result: dict) -> str:
    upstream = result["upstream_order"]
    learned = result["learned_attention_order"]
    comparison = result["pairwise_order_comparison"]
    return "\n".join([
        "# MOCHEG visual-selector validation audit",
        "",
        "Test split used: **no**",
        "",
        "| Ordering | Hit@1 | Hit@5 | Hit@10 | MRR |",
        "|---|---:|---:|---:|---:|",
        f"| Upstream retrieval | {upstream['hit_at_1']:.4f} | "
        f"{upstream['hit_at_5']:.4f} | {upstream['hit_at_10']:.4f} | "
        f"{upstream['mrr']:.4f} |",
        f"| Learned attention | {learned['hit_at_1']:.4f} | "
        f"{learned['hit_at_5']:.4f} | {learned['hit_at_10']:.4f} | "
        f"{learned['mrr']:.4f} |",
        "",
        f"- Gold-containing candidate sets: "
        f"`{result['claims_with_gold_in_candidates']}`.",
        f"- Upstream better / learned better / tie: "
        f"`{comparison['upstream_better']} / {comparison['learned_better']} / "
        f"{comparison['tie']}`.",
        f"- Learned minus upstream Hit@1: "
        f"`{result['learned_minus_upstream']['hit_at_1']:+.6f}`.",
        f"- Learned minus upstream MRR: "
        f"`{result['learned_minus_upstream']['mrr']:+.6f}`.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    relevance, attention, mask, checkpoint = collect(
        args.checkpoint, args.cache, args.batch_size, device
    )
    result = rank_summary(relevance, attention, mask)
    result["provenance"] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage": checkpoint.get("stage"),
        "cache": str(args.cache),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(
        markdown(result), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
