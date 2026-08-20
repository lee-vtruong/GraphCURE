"""Generate resumable claim-conditioned visual evidence reports.

The VLM is an analyzer, not the final fact-checker: it receives no labels or
qrel flags and must describe the visible relation between one claim and one
retrieved image. Training may inject one qrel image; validation never does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from graphcure.token_visual import select_candidate_ids
from scripts.caption_mocheg_images import clean_descriptor
from scripts.run_mocheg_visual_ensemble import aligned_gold_images
from scripts.run_mocheg_visual_retrieval import (normalize_image_paths,
                                                  read_jsonl)


PROMPT_VERSION = "claim-visual-constraints-v1"
PROMPT = """You are a visual evidence analyzer, not the final fact checker.
Claim: {claim}
Using only visible pixels, write one compact line (maximum 90 words) containing:
Visual facts: people/entities, objects, actions and scene;
OCR: exact readable words, dates, numbers and logos, or none;
Constraint relation: semantic, entity, temporal/provenance and contextual clues that connect or fail to connect the image to the claim;
Evidence relation: corroborates, contradicts, context-only, unrelated, or uncertain;
Sufficiency: sufficient or insufficient, with a short visible reason.
Do not use outside knowledge. Do not output a final supported/refuted/NEI label."""


def report_signature(model: str, top_k: int, max_pixels: int,
                     max_new_tokens: int) -> str:
    payload = {"model": model, "top_k": top_k, "max_pixels": max_pixels,
               "max_new_tokens": max_new_tokens, "prompt": PROMPT,
               "prompt_version": PROMPT_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def read_completed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid partial JSON at {path}:{number}") from error
        if not row.get("id") or row["id"] in result:
            raise ValueError(f"missing or duplicate id at {path}:{number}")
        result[row["id"]] = row
    return result


def conversations(claims: list[str], paths: list[str]) -> list[list[dict]]:
    return [[{"role": "user", "content": [
        {"type": "image", "image": path},
        {"type": "text", "text": PROMPT.format(claim=claim.strip())},
    ]}] for claim, path in zip(claims, paths, strict=True)]


def generate(model: Any, processor: Any, claims: list[str], paths: list[str],
             device: str, max_new_tokens: int) -> list[str]:
    chats = conversations(claims, paths)
    try:
        try:
            inputs = processor.apply_chat_template(
                chats, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
                processor_kwargs={"padding": True}).to(device)
        except TypeError:
            inputs = processor.apply_chat_template(
                chats, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, repetition_penalty=1.05,
                no_repeat_ngram_size=4)
        trimmed = [tokens[len(input_ids):] for input_ids, tokens in
                   zip(inputs.input_ids, output, strict=True)]
        return [clean_descriptor(value) for value in processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)]
    except torch.cuda.OutOfMemoryError:
        if len(paths) == 1:
            raise
        torch.cuda.empty_cache()
        middle = len(paths) // 2
        return (generate(model, processor, claims[:middle], paths[:middle],
                         device, max_new_tokens)
                + generate(model, processor, claims[middle:], paths[middle:],
                           device, max_new_tokens))


def main() -> None:
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForMultimodalLM as ModelClass
    except ImportError:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--train-retrieval", type=Path, required=True)
    parser.add_argument("--val-retrieval", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path(
        "data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "data/processed/mocheg_claim_visual_reports_v10"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/retrieval_cache"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-pixels", type=int, default=501760)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"),
                        default=("train", "val"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if min(args.top_k, args.batch_size, args.max_new_tokens, args.max_pixels) <= 0:
        parser.error("top-k, batch-size, max tokens and max pixels must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    signature = report_signature(args.model, args.top_k, args.max_pixels,
                                 args.max_new_tokens)
    processor = AutoProcessor.from_pretrained(args.model,
                                               max_pixels=args.max_pixels)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = ModelClass.from_pretrained(args.model, torch_dtype=dtype).to(
        args.device).eval()
    retrieval_paths = {"train": args.train_retrieval, "val": args.val_retrieval}

    for split in args.splits:
        claims = read_jsonl(args.manifest_root / f"{split}.jsonl")
        retrieval = {row["id"]: row for row in read_jsonl(
            retrieval_paths[split])}
        image_root = args.raw_root / split / "images"
        image_paths = {path.name: str(path.resolve()) for path in
                       image_root.iterdir() if path.is_file()}
        corpus_names = set(image_paths)
        target = args.output_root / f"{split}.jsonl"
        metadata_path = args.output_root / f"{split}.metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("report_signature") != signature:
                raise ValueError(f"settings differ from existing {metadata_path}")
        else:
            metadata_path.write_text(json.dumps({
                "split": split, "model": args.model, "top_k": args.top_k,
                "max_pixels": args.max_pixels,
                "max_new_tokens": args.max_new_tokens,
                "prompt_version": PROMPT_VERSION,
                "report_signature": signature,
                "train_gold_injection": split == "train",
                "validation_gold_injection": False,
            }, indent=2) + "\n", encoding="utf-8")
        completed = read_completed(target)
        pending = [row for row in claims if row["id"] not in completed]
        if args.limit is not None:
            pending = pending[:max(0, args.limit - len(completed))]
        print(json.dumps({"split": split, "claims": len(claims),
                          "completed": len(completed),
                          "scheduled": len(pending), "signature": signature},
                         indent=2))
        with target.open("a", encoding="utf-8", buffering=1) as handle:
            for claim in tqdm(pending, desc=f"{split} claim-image reports"):
                row = retrieval.get(claim["id"])
                if row is None:
                    raise ValueError(f"retrieval missing {claim['id']}")
                gold = aligned_gold_images(claim, corpus_names)
                retrieved = [str(value) for value in
                             row.get("retrieved_image_ids", [])
                             if str(value) in image_paths]
                selected = select_candidate_ids(
                    claim["id"], retrieved, gold, args.top_k, split == "train")
                original = [image_paths[value] for value in selected]
                normalized, _ = normalize_image_paths(original, args.cache_root)
                reports = []
                for start in range(0, len(selected), args.batch_size):
                    batch_paths = normalized[start:start + args.batch_size]
                    reports.extend(generate(
                        model, processor, [claim.get("claim", "")] * len(batch_paths),
                        batch_paths, args.device, args.max_new_tokens))
                result = {
                    "id": claim["id"], "claim_id": claim.get("claim_id"),
                    "label": int(claim["label"]), "image_ids": selected,
                    "retrieval_ranks": [retrieved.index(value) + 1
                                        if value in retrieved else 0
                                        for value in selected],
                    "retrieval_scores": [float(row.get(
                        "retrieved_image_scores", [])[retrieved.index(value)])
                        if value in retrieved and retrieved.index(value) < len(
                            row.get("retrieved_image_scores", [])) else 0.0
                        for value in selected],
                    "reports": reports,
                    # Retained for train objectives/audit only; never prompted.
                    "gold_flags": [value in gold for value in selected],
                    "report_signature": signature,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        total = len(read_completed(target))
        summary = {"split": split, "claims": len(claims), "reports": total,
                   "complete": total == len(claims), "model": args.model,
                   "top_k": args.top_k, "report_signature": signature,
                   "validation_gold_injection": False}
        (args.output_root / f"{split}.summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
