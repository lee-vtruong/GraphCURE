"""Inference-only evaluator for predeclared Qwen3 direct-verdict LoRA runs.

Unlike the legacy frozen-test evaluator, this utility does not select a seed,
checkpoint, temperature, or policy.  It simply materializes one prediction
file per already-trained adapter and verifies that all requested runs share a
single explicit direct-verdict recipe.  It is intended for fixed-size ensemble
controls, where excluding a validation-weak predeclared seed would itself be
a post-hoc selection decision.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_mocheg_qwen3_lora_verifier import (
    QwenVerifierDataset,
    evaluate,
    label_token_ids,
    make_collate,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_inputs(manifest: Path, retrieval: Path, expected_samples: int) -> None:
    claims = read_jsonl(manifest)
    evidence = read_jsonl(retrieval)
    claim_ids = [str(row["id"]) for row in claims]
    evidence_ids = [str(row["id"]) for row in evidence]
    if len(claim_ids) != expected_samples:
        raise ValueError(f"manifest has {len(claim_ids)} rows, expected {expected_samples}")
    if len(set(claim_ids)) != len(claim_ids) or len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("duplicate manifest or retrieval IDs")
    if set(claim_ids) != set(evidence_ids):
        raise ValueError("manifest and retrieval IDs do not match exactly")
    by_id = {str(row["id"]): row for row in evidence}
    for row in claims:
        if int(row["label"]) != int(by_id[str(row["id"])]["label"]):
            raise ValueError(f"label mismatch for {row['id']}")


def settings_signature(summary: dict) -> dict:
    settings = summary.get("settings", {})
    keys = (
        "model", "top_k", "max_evidence_chars", "max_length", "lora_r",
        "lora_alpha", "lora_dropout", "epochs", "batch_size",
        "gradient_accumulation", "learning_rate", "weight_decay", "warmup_ratio",
    )
    signature = {key: settings.get(key) for key in keys}
    signature["train_gold_injection"] = summary.get("train_gold_injection")
    return signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run in args.runs:
        summary_path = run / "summary.json"
        adapter = run / "best_adapter"
        if not summary_path.is_file() or not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"incomplete direct run: {run}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("test_split_used") is not False:
            raise ValueError(f"run is not test-clean: {run}")
        summaries.append(summary)
    signatures = [settings_signature(summary) for summary in summaries]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("direct runs do not share one training recipe")
    if signatures[0]["model"] != args.model:
        raise ValueError("run model does not match --model")
    if int(signatures[0]["top_k"]) != args.top_k:
        raise ValueError("run top-k does not match --top-k")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = QwenVerifierDataset(
        args.manifest, args.retrieval, args.corpus, args.top_k,
        args.max_evidence_chars, False, 0,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(tokenizer, args.max_length, False),
        pin_memory=device.type == "cuda",
    )
    token_ids = label_token_ids(tokenizer)
    report = {
        "protocol": "B22_predeclared_direct_only_member_inference",
        "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "retrieval": str(args.retrieval), "retrieval_sha256": sha256_file(args.retrieval),
        "corpus": str(args.corpus), "corpus_sha256": sha256_file(args.corpus),
        "expected_samples": args.expected_samples, "training_recipe": signatures[0],
        "members": [], "test_labels_used_for_selection": False,
        "test_labels_used_for_evaluation": True,
    }
    for run, summary in zip(args.runs, summaries):
        prediction_path = run / f"test_predictions_{args.tag}.jsonl"
        metrics_path = run / f"test_metrics_{args.tag}.json"
        if prediction_path.is_file() and metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            adapter = run / "best_adapter"
            base = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                attn_implementation="sdpa",
            ).to(device)
            base.config.use_cache = False
            model = PeftModel.from_pretrained(base, adapter).to(device)
            started = time.perf_counter()
            metrics, rows = evaluate(model, loader, token_ids, device)
            elapsed = time.perf_counter() - started
            metrics.update({
                "seed": summary["settings"]["seed"], "elapsed_seconds": elapsed,
                "milliseconds_per_sample": 1000 * elapsed / len(dataset),
                "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            })
            prediction_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            del model, base
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows = read_jsonl(prediction_path)
        if len(rows) != args.expected_samples:
            raise ValueError(f"incomplete predictions in {prediction_path}")
        report["members"].append({
            "run": str(run), "seed": summary["settings"]["seed"], "metrics": metrics,
            "prediction": str(prediction_path), "prediction_sha256": sha256_file(prediction_path),
        })
    report_path = args.output_dir / f"inference_{args.tag}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
