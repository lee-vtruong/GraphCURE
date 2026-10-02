"""Inference-only external evaluator using the B18 rationale-trained verdict prompt.

The rationale-trained adapters were optimized with the B18 Prompt-A verdict
interface, so external inference must retain that interface rather than reuse
the legacy direct dataset/collator.  No explanation prompt is generated at
inference time.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.evaluate_mocheg_qwen3_direct_runs import read_jsonl, validate_inputs
from scripts.train_mocheg_b18_explanation_verifier import (
    B18VerifierDataset, evaluate_direct_verdict, make_b18_collate,
)
from scripts.train_mocheg_qwen3_lora_verifier import label_token_ids, read_documents


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-evidence-chars", type=int, default=2200)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    validate_inputs(args.manifest, args.retrieval, args.expected_samples)
    summaries = []
    for run in args.runs:
        summary_path, adapter = run / "summary.json", run / "best_adapter"
        if not summary_path.is_file() or not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"incomplete rationale-trained run: {run}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # These adapters may have previously been evaluated on the *MOCHEG*
        # test split.  That historical evaluation is not a selection signal
        # for this external SciFact transfer: the checkpoint paths, three
        # seeds, Prompt-A interface, K=5, and AND tau are fixed before any
        # SciFact prediction is read.  Record the source-run status for the
        # audit rather than rejecting a valid frozen checkpoint.
        summary["source_mocheg_test_split_used"] = summary.get("test_split_used")
        summaries.append(summary)
    claims = read_jsonl(args.manifest)
    retrieval = {str(row["id"]): row for row in read_jsonl(args.retrieval)}
    documents = read_documents(args.corpus)
    dataset = B18VerifierDataset(claims, retrieval, documents, None, args.top_k,
                                 args.max_evidence_chars, training=False)
    if len(dataset) != args.expected_samples:
        raise ValueError("B18 external dataset dropped one or more claims")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_b18_collate(tokenizer, args.max_length, False, "matched_control"),
        pin_memory=device.type == "cuda",
    )
    answer_ids = label_token_ids(tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "external_B18_promptA_rationale_member_inference",
        "manifest_sha256": sha256(args.manifest), "retrieval_sha256": sha256(args.retrieval),
        "corpus_sha256": sha256(args.corpus), "expected_samples": args.expected_samples,
        "members": [], "test_labels_used_for_selection": False,
        "external_checkpoint_or_seed_selection": False,
        "inference_prompt": "B18 Prompt-A verdict interface; no explanation prompt",
    }
    for run, summary in zip(args.runs, summaries):
        prediction_path = run / f"test_predictions_{args.tag}.jsonl"
        metrics_path = run / f"test_metrics_{args.tag}.json"
        if prediction_path.is_file() and metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            adapter = run / "best_adapter"
            base = AutoModelForCausalLM.from_pretrained(
                args.model, trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            ).to(device)
            base.config.use_cache = False
            model = PeftModel.from_pretrained(base, adapter).to(device)
            started = time.perf_counter()
            metrics = evaluate_direct_verdict(model, loader, answer_ids, device)
            rows = metrics.pop("predictions")
            elapsed_seconds = time.perf_counter() - started
            metrics.update({"elapsed_seconds": elapsed_seconds,
                            "milliseconds_per_sample": 1000 * elapsed_seconds / len(dataset),
                            "adapter_config_sha256": sha256(adapter / "adapter_config.json")})
            prediction_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            del model, base
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if len(read_jsonl(prediction_path)) != args.expected_samples:
            raise ValueError(f"incomplete predictions in {prediction_path}")
        report["members"].append({"run": str(run), "metrics": metrics,
                                  "source_mocheg_test_split_used": summary.get("source_mocheg_test_split_used"),
                                  "prediction": str(prediction_path),
                                  "prediction_sha256": sha256(prediction_path)})
    (args.output_dir / f"inference_{args.tag}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
