"""Score matched rank-control and constraint-selected C3 evidence sets."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from graphcure.open_web import load_jsonl, sha256_file, sha256_text
from scripts.score_mocheg_open_constraints import as_token_ids, head_tail_truncate
from scripts.score_mocheg_open_verdicts import SYSTEM_DIRECT, compose_verdict_prompt


MODES = {
    "rank_control": "rank_control_evidence",
    "constraint_selector": "constraint_selected_evidence",
}


def build_examples(rows: list[dict], max_evidence_chars: int) -> list[dict]:
    result = []
    for row in rows:
        for mode, field in MODES.items():
            evidence = row.get(field, [])
            result.append({
                "id": str(row["id"]),
                "mode": mode,
                "prompt": compose_verdict_prompt(
                    row.get("claim", ""), evidence, {}, "direct",
                    max_evidence_chars,
                ),
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-evidence-chars", type=int, default=2200)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if min(args.max_evidence_chars, args.max_length, args.batch_size) <= 0:
        parser.error("lengths and batch size must be positive")
    if not args.adapter.is_dir():
        parser.error(f"adapter directory does not exist: {args.adapter}")

    selection_path = args.selection_root / "val.jsonl"
    selection_summary_path = args.selection_root / "summary.json"
    selection_summary = json.loads(
        selection_summary_path.read_text(encoding="utf-8")
    )
    if any(selection_summary.get(key) for key in (
        "label_used", "gold_evidence_used", "test_split_used"
    )):
        parser.error("C3 selection contains forbidden supervision")
    if sha256_file(selection_path) != selection_summary.get("output_sha256"):
        parser.error("C3 selection file hash mismatch")
    if selection_summary.get("top_k") != 5:
        parser.error("C3 preregistration requires top-k 5")
    alignment = selection_summary.get("verifier_alignment", {})
    if alignment.get("max_evidence_chars") != args.max_evidence_chars:
        parser.error("evidence character budget differs from C3 preregistration")
    if alignment.get("constraint_annotations_in_prompt") is not False:
        parser.error("C3 treatment must not put constraint annotations in prompt")

    rows = load_jsonl(selection_path)
    if args.limit:
        rows = rows[:args.limit]
    examples = build_examples(rows, args.max_evidence_chars)
    args.output_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "protocol": "P2_open_web_C3_matched_evidence_selection_screen",
        "selection_sha256": selection_summary["output_sha256"],
        "model": args.model,
        "adapter": str(args.adapter),
        "max_evidence_chars": args.max_evidence_chars,
        "max_length": args.max_length,
        "modes": list(MODES),
    }
    signature = sha256_text(json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ))
    run_manifest_path = args.output_root / "run_manifest.json"
    if run_manifest_path.exists():
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if previous.get("configuration_signature") != signature:
            parser.error("output-root belongs to another C3 configuration")
    else:
        payload = {
            **configuration, "configuration_signature": signature,
            "label_used": False, "gold_evidence_used": False,
            "test_split_used": False,
        }
        temporary = run_manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(run_manifest_path)

    output_path = args.output_root / "verdict_scores.jsonl"
    existing = load_jsonl(output_path)
    completed = {(str(row["id"]), str(row["mode"])) for row in existing}
    if len(completed) != len(existing):
        parser.error("duplicate rows in resumable C3 scores")
    pending = [
        row for row in examples if (row["id"], row["mode"]) not in completed
    ]

    import torch
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    device = torch.device(args.device)
    model.to(device).eval()
    answer_ids = []
    for code in "ABC":
        ids = as_token_ids(tokenizer.encode(code, add_special_tokens=False))
        if len(ids) != 1:
            raise ValueError(f"verdict code {code} is not one token: {ids}")
        answer_ids.append(ids[0])
    answer_index = torch.tensor(answer_ids, device=device)

    def collate(batch: list[dict]) -> dict:
        sequences = []
        for row in batch:
            chat = tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM_DIRECT},
                {"role": "user", "content": row["prompt"]},
            ], tokenize=False, add_generation_prompt=True)
            ids = as_token_ids(tokenizer.encode(chat, add_special_tokens=False))
            sequences.append(head_tail_truncate(ids, args.max_length))
        width = max(map(len, sequences))
        input_ids = torch.full(
            (len(batch), width), tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, -len(sequence):] = torch.tensor(sequence)
            attention_mask[index, -len(sequence):] = 1
        return {
            "rows": batch,
            "encoded": {"input_ids": input_ids, "attention_mask": attention_mask},
        }

    loader = DataLoader(
        pending, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )
    started = time.perf_counter()
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        for batch in tqdm(loader, desc="C3 evidence-selection verdicts"):
            encoded = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["encoded"].items()
            }
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                try:
                    logits = model(**encoded, logits_to_keep=1).logits.float()
                except TypeError:
                    logits = model(**encoded).logits.float()
            probability = torch.softmax(
                logits[:, -1, answer_index], dim=-1
            ).cpu()
            for index, row in enumerate(batch["rows"]):
                values = probability[index].tolist()
                handle.write(json.dumps({
                    "id": row["id"], "mode": row["mode"],
                    "prediction": int(max(range(3), key=values.__getitem__)),
                    "probabilities": values,
                    "label_used": False, "gold_evidence_used": False,
                }) + "\n")

    all_rows = load_jsonl(output_path)
    expected = {(row["id"], row["mode"]) for row in examples}
    relevant = [
        row for row in all_rows
        if (str(row["id"]), str(row["mode"])) in expected
    ]
    invalid = 0
    for row in relevant:
        values = row.get("probabilities", [])
        if (
            len(values) != 3
            or not all(math.isfinite(float(value)) for value in values)
            or abs(sum(map(float, values)) - 1.0) > 1e-4
        ):
            invalid += 1
    mode_counts = {
        mode: sum(row["mode"] == mode for row in relevant) for mode in MODES
    }
    prediction_counts = {
        mode: {
            str(label): sum(
                row["mode"] == mode and int(row["prediction"]) == label
                for row in relevant
            )
            for label in range(3)
        }
        for mode in MODES
    }
    summary = {
        **configuration,
        "configuration_signature": signature,
        "claims": len(rows), "expected_scores": len(examples),
        "completed_scores": len(relevant),
        "complete": len(relevant) == len(examples),
        "mode_counts": mode_counts,
        "prediction_counts": prediction_counts,
        "invalid_probability_rows": invalid,
        "label_token_ids": answer_ids,
        "new_elapsed_seconds": time.perf_counter() - started,
        "label_used": False, "gold_evidence_used": False,
        "test_split_used": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
