"""Phase B18-B: Train an Adaptive Sentence & Passage Selector using Teacher Rationales.

Trains a Cross-Encoder on (claim, passage) pairs where:
- Positive: passage cited in teacher explanation key_evidence_ids.
- Hard Negative: retrieved top-5 passages unselected by teacher.
- Distractor Negative: retrieved passages from ungrounded claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from graphcure.explanation import resolve_corpus_path
from graphcure.selector import extract_selector_pairs
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_qwen3_lora_verifier import read_documents


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_pairs_by_claim(
    pairs: list[dict[str, Any]], dev_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic claim-disjoint train/dev split.

    Pair-level random splitting leaks the same claim and often the same passage
    family into both sets.  This helper keeps every pair for a claim together.
    """
    claim_ids = sorted({str(pair["claim_id"]) for pair in pairs})
    if dev_fraction <= 0.0 or len(claim_ids) < 2:
        return list(pairs), []
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in [0, 1)")
    rng = random.Random(seed)
    rng.shuffle(claim_ids)
    dev_count = min(len(claim_ids) - 1, max(1, round(len(claim_ids) * dev_fraction)))
    dev_ids = set(claim_ids[:dev_count])
    train_pairs = [pair for pair in pairs if str(pair["claim_id"]) not in dev_ids]
    dev_pairs = [pair for pair in pairs if str(pair["claim_id"]) in dev_ids]
    return train_pairs, dev_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase B18-B sentence/passage selector")
    parser.add_argument("--explanations", type=Path, required=True, help="Teacher explanations JSONL")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus2.csv path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory to save model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Pretrained cross-encoder backbone",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.1,
        help="Claim-level held-out fraction used to select the best selector checkpoint",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on explanations for debug")
    parser.add_argument("--mock", action="store_true", help="Deterministic mock mode for unit tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    logging.info("Loading explanations from: %s", args.explanations)
    explanations = read_jsonl(args.explanations)
    if args.limit > 0:
        explanations = explanations[:args.limit]
    logging.info("Loaded %d explanation records", len(explanations))

    corpus_path = resolve_corpus_path(args.corpus)
    logging.info("Reading corpus from: %s", corpus_path)
    documents = read_documents(corpus_path)
    logging.info("Loaded %d documents from corpus", len(documents))

    logging.info("Extracting selector training pairs...")
    pairs = extract_selector_pairs(
        explanations=explanations,
        documents=documents,
        max_neg_per_grounded=4,
        sample_ungrounded_negatives=True,
        max_neg_per_ungrounded=2,
        seed=args.seed,
    )

    positives = sum(1 for p in pairs if p["label"] == 1.0)
    negatives = sum(1 for p in pairs if p["label"] == 0.0)
    logging.info(
        "Extracted %d total training pairs (%d positive, %d negative, ratio=%.2f:1)",
        len(pairs),
        positives,
        negatives,
        negatives / max(1, positives),
    )

    train_pairs, dev_pairs = split_pairs_by_claim(pairs, args.dev_fraction, args.seed)
    train_claim_ids = {str(pair["claim_id"]) for pair in train_pairs}
    dev_claim_ids = {str(pair["claim_id"]) for pair in dev_pairs}
    if train_claim_ids & dev_claim_ids:
        raise RuntimeError("claim leakage detected between selector train and dev sets")
    logging.info(
        "Claim-disjoint split: train=%d pairs/%d claims, dev=%d pairs/%d claims",
        len(train_pairs),
        len(train_claim_ids),
        len(dev_pairs),
        len(dev_claim_ids),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    dev_metrics: dict[str, float] | None = None

    if args.mock:
        logging.info("Mock mode enabled: Saving mock model metadata without PyTorch fit")
        mock_meta = {
            "is_mock": True,
            "base_model": args.base_model,
            "num_pairs": len(pairs),
            "positives": positives,
            "negatives": negatives,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "train_pairs": len(train_pairs),
            "dev_pairs": len(dev_pairs),
        }
        (args.output / "mock_selector_config.json").write_text(
            json.dumps(mock_meta, indent=2) + "\n", encoding="utf-8"
        )
    else:
        import torch
        from sentence_transformers import CrossEncoder, InputExample
        from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
        from torch.utils.data import DataLoader

        if torch.cuda.is_available() and "cuda" in args.device:
            torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)

        train_examples = [
            InputExample(texts=[p["claim"], p["text"]], label=float(p["label"]))
            for p in train_pairs
        ]

        logging.info("Initializing CrossEncoder: %s (device=%s)", args.base_model, args.device)
        model = CrossEncoder(args.base_model, num_labels=1, device=args.device)

        train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        total_steps = len(train_loader) * args.epochs
        warmup_steps = max(1, int(total_steps * 0.1))

        evaluator = None
        if dev_pairs:
            evaluator = CrossEncoderClassificationEvaluator(
                [[p["claim"], p["text"]] for p in dev_pairs],
                [int(p["label"]) for p in dev_pairs],
                name="claim-disjoint-dev",
                batch_size=args.batch_size,
            )

        logging.info(
            "Fitting CrossEncoder: %d steps over %d epochs (warmup=%d, lr=%s)",
            total_steps,
            args.epochs,
            warmup_steps,
            args.lr,
        )
        model.fit(
            train_dataloader=train_loader,
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": args.lr},
            evaluator=evaluator,
            evaluation_steps=len(train_loader) if evaluator is not None else 0,
            output_path=str(args.output) if evaluator is not None else None,
            save_best_model=evaluator is not None,
            show_progress_bar=True,
        )

        if evaluator is None:
            logging.info("Saving final CrossEncoder to: %s", args.output)
            model.save(str(args.output))
        else:
            logging.info("Best claim-disjoint dev checkpoint saved to: %s", args.output)
            best_model = CrossEncoder(str(args.output), device=args.device)
            dev_metrics = {
                key: float(value)
                for key, value in evaluator(
                    best_model,
                    output_path=str(args.output),
                    epoch=args.epochs,
                    steps=total_steps,
                ).items()
            }

    summary_payload: dict[str, Any] = {
        "phase": "B18-B",
        "task": "sentence_selector_training",
        "explanations_file": str(args.explanations),
        "corpus_file": str(corpus_path),
        "base_model": args.base_model,
        "is_mock": args.mock,
        "total_pairs": len(pairs),
        "positives": positives,
        "negatives": negatives,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "train_claims": len(train_claim_ids),
        "dev_claims": len(dev_claim_ids),
        "claim_split_overlap": len(train_claim_ids & dev_claim_ids),
        "checkpoint_selection": "claim_disjoint_dev" if dev_pairs else "final_epoch",
        "best_checkpoint_dev_metrics": dev_metrics,
        "audit": {
            "explanations_sha256": sha256_file(args.explanations),
            "corpus_sha256": sha256_file(corpus_path),
            "teacher_pseudo_labels_used_for_training": True,
            "gold_evidence_used_for_training": False,
            "validation_labels_used": False,
            "test_split_used": False,
        },
    }
    summary_path = args.output / "training_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    logging.info("Training summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
