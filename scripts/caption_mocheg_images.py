"""Generate resumable, pixel-derived MOCHEG image descriptors.

Descriptors are generated without claim text, qrels, or filename-derived topic
identifiers. They form a leakage-safe contextual/OCR retrieval view.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from scripts.run_mocheg_visual_retrieval import (
    corpus_fingerprint,
    image_corpus,
    normalize_image_paths,
)


DESCRIPTOR_GENERATION_VERSION = "v3-left-padding-compact"
DESCRIPTOR_TEXT_NORMALIZATION_VERSION = "unicode-colon-v1"
DESCRIPTOR_PROMPT = """Using only visible information, output exactly one line of at most 55 words with three fields: Visual: <main people or entities, objects, actions, event, scene>; Text: <exact readable words, dates, numbers, logos, or none>; Type/Clues: <photo, screenshot, meme, document, chart or map, plus visible place, time, or source-reuse clues>. Never repeat. Identify a person or place only when visually unmistakable."""


def descriptor_signature(model: str, prompt: str, fingerprint: str,
                         max_pixels: int, max_new_tokens: int,
                         repetition_penalty: float = 1.0,
                         no_repeat_ngram_size: int = 0) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "corpus_fingerprint": fingerprint,
        "max_pixels": max_pixels,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "generation_version": DESCRIPTOR_GENERATION_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def read_completed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    completed: dict[str, dict] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSONL at {path}:{line_number}; repair the partial line"
            ) from error
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in completed:
            raise ValueError(f"missing or duplicate image_id at {path}:{line_number}")
        completed[image_id] = row
    return completed


def conversations(paths: list[str], prompt: str) -> list[list[dict]]:
    return [[{
        "role": "user",
        "content": [
            {"type": "image", "image": path},
            {"type": "text", "text": prompt},
        ],
    }] for path in paths]


def clean_descriptor(text: str) -> str:
    """Make model output a stable single-line retrieval document."""
    return " ".join(
        text.replace("\x00", " ").replace("：", ":").split()
    ).strip()


def generate_descriptors(model: Any, processor: Any, paths: list[str],
                         prompt: str, device: str,
                         max_new_tokens: int, repetition_penalty: float,
                         no_repeat_ngram_size: int) -> list[str]:
    import torch

    chats = conversations(paths, prompt)
    try:
        template_args = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            inputs = processor.apply_chat_template(
                chats,
                processor_kwargs={"padding": True},
                **template_args,
            ).to(device)
        except TypeError:
            # Compatibility with older Transformers releases.
            inputs = processor.apply_chat_template(
                chats, padding=True, **template_args
            ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, generated, strict=True)
        ]
        return [
            clean_descriptor(text) for text in processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]
    except torch.cuda.OutOfMemoryError:
        if len(paths) == 1:
            raise
        torch.cuda.empty_cache()
        middle = len(paths) // 2
        return (
            generate_descriptors(
                model, processor, paths[:middle], prompt, device,
                max_new_tokens, repetition_penalty, no_repeat_ngram_size,
            )
            + generate_descriptors(
                model, processor, paths[middle:], prompt, device,
                max_new_tokens, repetition_penalty, no_repeat_ngram_size,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/processed/mocheg_visual_descriptors"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/retrieval_cache"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--prompt", default=DESCRIPTOR_PROMPT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.max_pixels <= 0:
        parser.error("batch size, max tokens, and max pixels must be positive")
    if args.repetition_penalty < 1.0:
        parser.error("--repetition-penalty must be at least 1.0")
    if args.no_repeat_ngram_size < 0:
        parser.error("--no-repeat-ngram-size cannot be negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if "test" in args.splits:
        parser.error("test is locked until the multimodal configuration freezes")

    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForMultimodalLM as ModelClass
    except ImportError:
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError as error:
            raise ImportError(
                "Qwen3-VL requires a Transformers build containing "
                "AutoModelForMultimodalLM or Qwen3VLForConditionalGeneration"
            ) from error

    args.output_root.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=args.max_pixels
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = ModelClass.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(args.device).eval()

    for split in args.splits:
        names, original_paths = image_corpus(args.raw_root / split / "images")
        fingerprint = corpus_fingerprint(names, original_paths)
        signature = descriptor_signature(
            args.model, args.prompt, fingerprint,
            args.max_pixels, args.max_new_tokens,
            args.repetition_penalty, args.no_repeat_ngram_size,
        )
        output_path = args.output_root / f"{split}.jsonl"
        metadata_path = args.output_root / f"{split}.metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("descriptor_signature") != signature:
                raise ValueError(
                    f"descriptor settings differ from existing {metadata_path}; "
                    "use a new --output-root"
                )
        else:
            metadata_path.write_text(json.dumps({
                "split": split,
                "model": args.model,
                "prompt": args.prompt,
                "max_pixels": args.max_pixels,
                "max_new_tokens": args.max_new_tokens,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "generation_version": DESCRIPTOR_GENERATION_VERSION,
                "images": len(names),
                "corpus_fingerprint": fingerprint,
                "descriptor_signature": signature,
            }, indent=2) + "\n", encoding="utf-8")

        normalized_paths, converted = normalize_image_paths(
            original_paths, args.cache_root
        )
        completed = read_completed(output_path)
        selected = [
            (name, path) for name, path in zip(names, normalized_paths, strict=True)
            if name not in completed
        ]
        if args.limit is not None:
            selected = selected[:max(0, args.limit - len(completed))]
        print(json.dumps({
            "split": split,
            "images": len(names),
            "already_completed": len(completed),
            "scheduled": len(selected),
            "normalized_unsupported": converted,
            "descriptor_signature": signature,
        }, indent=2))

        with output_path.open("a", encoding="utf-8") as handle:
            for start in tqdm(
                range(0, len(selected), args.batch_size),
                desc=f"{split} pixel descriptors",
            ):
                batch = selected[start:start + args.batch_size]
                batch_names = [item[0] for item in batch]
                batch_paths = [item[1] for item in batch]
                descriptions = generate_descriptors(
                    model, processor, batch_paths, args.prompt,
                    args.device, args.max_new_tokens,
                    args.repetition_penalty, args.no_repeat_ngram_size,
                )
                for image_id, descriptor in zip(
                    batch_names, descriptions, strict=True
                ):
                    if not descriptor:
                        descriptor = "No reliable visible description produced."
                    handle.write(json.dumps({
                        "image_id": image_id,
                        "descriptor": descriptor,
                        "model": args.model,
                        "descriptor_signature": signature,
                    }, ensure_ascii=False) + "\n")
                handle.flush()

        total = len(read_completed(output_path))
        summary = {
            "split": split,
            "images": len(names),
            "descriptors": total,
            "complete": total == len(names),
            "model": args.model,
            "descriptor_signature": signature,
        }
        (args.output_root / f"{split}.summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
