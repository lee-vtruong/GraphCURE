"""Reproducible inference-efficiency benchmark for the frozen MOCHEG systems."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from graphcure.explanation import resolve_corpus_path
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_b18_explanation_verifier import (
    B18VerifierDataset,
    evaluate_direct_verdict,
    make_b18_collate,
)
from scripts.train_mocheg_qwen3_lora_verifier import label_token_ids, read_documents


def adapter_dir(path: Path) -> Path:
    if (path / "adapter_config.json").is_file():
        return path
    candidate = path / "best_adapter"
    if (candidate / "adapter_config.json").is_file():
        return candidate
    raise FileNotFoundError(f"No LoRA adapter_config.json under {path}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warm_up(model: Any, loader: DataLoader, device: torch.device, batches: int) -> None:
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            input_ids = batch["verdict_input_ids"].to(device, non_blocking=True)
            attention_mask = batch["verdict_attention_mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                model(input_ids=input_ids, attention_mask=attention_mask)
    synchronize(device)


def benchmark_adapter(
    name: str,
    path: Path,
    args: argparse.Namespace,
    loader: DataLoader,
    answer_ids: list[int],
    samples: int,
) -> dict[str, Any]:
    device = torch.device(args.device)
    resolved = adapter_dir(path)
    load_start = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, resolved).to(device)
    model.config.use_cache = False
    synchronize(device)
    load_seconds = time.perf_counter() - load_start

    warm_up(model, loader, device, args.warmup_batches)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        evaluation = evaluate_direct_verdict(model, loader, answer_ids, device)
    synchronize(device)
    elapsed = time.perf_counter() - start

    result = {
        "name": name,
        "adapter": str(resolved),
        "samples": samples,
        "load_seconds": load_seconds,
        "inference_seconds": elapsed,
        "milliseconds_per_sample": 1000.0 * elapsed / samples,
        "samples_per_second": samples / elapsed,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "subset_diagnostic": {
            key: evaluation[key]
            for key in ("accuracy", "macro_f1", "f1_supported", "f1_refuted", "f1_nei")
        },
    }
    del evaluation, model, base
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)
    return result


def aggregate(name: str, members: list[dict[str, Any]], samples: int) -> dict[str, Any]:
    elapsed = sum(row["inference_seconds"] for row in members)
    return {
        "name": name,
        "model_forward_passes_per_claim": len(members),
        "members": [row["name"] for row in members],
        "inference_seconds": elapsed,
        "milliseconds_per_sample": 1000.0 * elapsed / samples,
        "samples_per_second": samples / elapsed,
        "sequential_peak_allocated_gib": max(row["peak_allocated_gib"] for row in members),
        "sequential_peak_reserved_gib": max(row["peak_reserved_gib"] for row in members),
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MOCHEG frozen-system efficiency benchmark",
        "",
        f"- Device: `{summary['device_name']}`",
        f"- Claims: `{summary['samples']}`",
        f"- Batch size: `{summary['settings']['batch_size']}`",
        f"- Warm-up batches: `{summary['settings']['warmup_batches']}`",
        "- Model loading is excluded from inference latency.",
        "- Ensembles are measured as sequential member inference; peak VRAM is the maximum member peak.",
        "- Subset scores are diagnostics only, not paper effectiveness results.",
        "",
        "| System | Passes/claim | ms/claim | claims/s | Peak allocated GiB |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["systems"]:
        lines.append(
            f"| {row['name']} | {row['model_forward_passes_per_claim']} | "
            f"{row['milliseconds_per_sample']:.2f} | {row['samples_per_second']:.3f} | "
            f"{row['sequential_peak_allocated_gib']:.2f} |"
        )
    lines += ["", "## Individual checkpoints", "",
              "| Checkpoint | Load s | Infer s | ms/claim | Peak GiB | Diagnostic F1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["checkpoints"]:
        lines.append(
            f"| {row['name']} | {row['load_seconds']:.2f} | {row['inference_seconds']:.2f} | "
            f"{row['milliseconds_per_sample']:.2f} | {row['peak_allocated_gib']:.2f} | "
            f"{row['subset_diagnostic']['macro_f1']:.4f} |"
        )
    if summary["missing_optional"]:
        lines += ["", "## Missing optional systems", ""]
        lines += [f"- {item}" for item in summary["missing_optional"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-adapters", type=Path, nargs="+", required=True)
    parser.add_argument("--grounded-adapters", type=Path, nargs="+", required=True)
    parser.add_argument("--b19-adapter", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-evidence-chars", type=int, default=2200)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    claims = read_jsonl(args.manifest)[: args.limit]
    retrieval_by_id = {str(row["id"]): row for row in read_jsonl(args.retrieval)}
    documents = read_documents(resolve_corpus_path(args.corpus))
    dataset = B18VerifierDataset(
        claims, retrieval_by_id, documents, None, args.top_k,
        args.max_evidence_chars, training=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=make_b18_collate(tokenizer, args.max_length, False, "matched_control"),
    )
    answers = label_token_ids(tokenizer)
    checkpoints: list[dict[str, Any]] = []
    direct, grounded = [], []
    for index, path in enumerate(args.direct_adapters):
        row = benchmark_adapter(f"direct_{index + 1}", path, args, loader, answers, len(dataset))
        checkpoints.append(row); direct.append(row)
    for index, path in enumerate(args.grounded_adapters):
        row = benchmark_adapter(f"grounded_{index + 1}", path, args, loader, answers, len(dataset))
        checkpoints.append(row); grounded.append(row)

    systems = [
        aggregate("direct_ensemble", direct, len(dataset)),
        aggregate("grounded_ensemble", grounded, len(dataset)),
        aggregate("B18B_dual_ensemble", direct + grounded, len(dataset)),
    ]
    missing = []
    if args.b19_adapter:
        try:
            row = benchmark_adapter("B19_single", args.b19_adapter, args, loader, answers, len(dataset))
            checkpoints.append(row)
            systems.append(aggregate("B19_single", [row], len(dataset)))
        except FileNotFoundError as error:
            missing.append(str(error))

    summary = {
        "protocol": "MOCHEG_frozen_efficiency_common_subset_v1",
        "samples": len(dataset),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "settings": vars(args) | {"manifest": str(args.manifest), "retrieval": str(args.retrieval),
                                  "corpus": str(args.corpus), "output": str(args.output),
                                  "markdown": str(args.markdown),
                                  "direct_adapters": [str(x) for x in args.direct_adapters],
                                  "grounded_adapters": [str(x) for x in args.grounded_adapters],
                                  "b19_adapter": str(args.b19_adapter) if args.b19_adapter else None},
        "checkpoints": checkpoints,
        "systems": systems,
        "missing_optional": missing,
        "official_validation_used_for_efficiency_only": True,
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = markdown(summary)
    args.markdown.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
