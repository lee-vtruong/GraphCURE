"""Score a B18 LoRA teacher on the train or held-out side of a frozen B18 fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from graphcure.explanation import resolve_corpus_path
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_b18_explanation_verifier import (
    B18VerifierDataset,
    evaluate_direct_verdict,
    make_b18_collate,
)
from scripts.train_mocheg_qwen3_lora_verifier import label_token_ids, read_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--subset", choices=["train", "heldout", "all"], default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--max-evidence-chars", type=int, default=2200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output.is_file() and args.summary.is_file():
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        if summary.get("complete"):
            print(f"Cached complete teacher scores: {args.output}")
            return
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    entry = next(row for row in folds["folds"] if int(row["fold"]) == args.fold)
    allowed = None
    if args.subset == "train":
        allowed = set(map(str, entry["train_ids"]))
    elif args.subset == "heldout":
        allowed = set(map(str, entry["val_ids"]))
    claims = read_jsonl(args.manifest)
    if allowed is not None:
        claims = [row for row in claims if str(row["id"]) in allowed]
    retrieval = {str(row["id"]): row for row in read_jsonl(args.retrieval)}
    documents = read_documents(resolve_corpus_path(args.corpus))

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = B18VerifierDataset(
        claims, retrieval, documents, None, args.top_k, args.max_evidence_chars, False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_b18_collate(tokenizer, args.max_length, False, "matched_control"),
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, str(args.adapter)).to(device)
    metrics = evaluate_direct_verdict(model, loader, label_token_ids(tokenizer), device)
    predictions = metrics.pop("predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    payload = {
        "phase": "B19",
        "adapter": str(args.adapter),
        "fold": args.fold,
        "subset": args.subset,
        "samples": len(predictions),
        "complete": len(predictions) == len(dataset),
        "metrics": metrics,
        "official_validation_used": False,
        "test_split_used": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
