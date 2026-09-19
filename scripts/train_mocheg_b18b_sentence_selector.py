"""Phase B18-B: Train an Adaptive Sentence & Passage Selector using Teacher Rationales.

Trains a Cross-Encoder on (claim, passage) pairs where:
- Positive: passage cited in teacher explanation key_evidence_ids.
- Hard Negative: retrieved top-5 passages unselected by teacher.
- Distractor Negative: retrieved passages from ungrounded claims.
"""
from __future__ import annotations

import argparse
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

    args.output.mkdir(parents=True, exist_ok=True)

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
        }
        (args.output / "mock_selector_config.json").write_text(
            json.dumps(mock_meta, indent=2) + "\n", encoding="utf-8"
        )
    else:
        import torch
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader

        if torch.cuda.is_available() and "cuda" in args.device:
            torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)

        examples = [
            InputExample(texts=[p["claim"], p["text"]], label=float(p["label"]))
            for p in pairs
        ]

        logging.info("Initializing CrossEncoder: %s (device=%s)", args.base_model, args.device)
        model = CrossEncoder(args.base_model, num_labels=1, device=args.device)

        train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
        total_steps = len(train_loader) * args.epochs
        warmup_steps = max(1, int(total_steps * 0.1))

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
            show_progress_bar=True,
        )

        logging.info("Saving trained CrossEncoder to: %s", args.output)
        model.save(str(args.output))

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
    }
    summary_path = args.output / "training_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    logging.info("Training summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
