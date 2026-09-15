"""Score C2b claim-evidence constraints with a frozen instruction model."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from graphcure.open_web import load_jsonl, sha256_file, sha256_text


TASKS = {
    "stance": (
        "A: the evidence supports the claim; B: the evidence refutes or "
        "contradicts the claim; C: the evidence establishes neither."
    ),
    "sufficiency": (
        "A: sufficient on its own for a verdict; B: relevant but only partial; "
        "C: irrelevant, unusable, or unrelated."
    ),
    "entity": (
        "A: the central entities match the claim; B: a central entity conflicts "
        "or is mismatched; C: unknown or not applicable."
    ),
    "temporal": (
        "A: the time information is compatible with the claim; B: the time "
        "information conflicts with the claim; C: unknown or not applicable."
    ),
}
SYSTEM_PROMPT = (
    "You are a conservative evidence analyst. Use only the supplied claim and "
    "one evidence item. Do not use outside knowledge. Classify exactly the "
    "requested dimension and answer with one letter only: A, B, or C."
)


def as_token_ids(value) -> list[int]:
    """Normalize tokenizers/transformers encoding variants to integer IDs."""
    if hasattr(value, "ids"):
        value = value.ids
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict) and "input_ids" in value:
        value = value["input_ids"]
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(
        value[0], (list, tuple)
    ):
        value = list(value[0])
    if not isinstance(value, list):
        raise TypeError(f"cannot normalize token IDs from {type(value).__name__}")
    return [int(token_id) for token_id in value]


def head_tail_truncate(token_ids: list[int], max_length: int) -> list[int]:
    """Preserve the claim/system prefix and task/classes suffix of long prompts."""
    if len(token_ids) <= max_length:
        return token_ids
    prefix = min(512, max(1, max_length // 3))
    return token_ids[:prefix] + token_ids[-(max_length - prefix):]


def compose_prompt(claim: str, evidence: dict, task: str,
                   max_evidence_chars: int) -> str:
    if task not in TASKS:
        raise ValueError(f"unknown constraint task: {task}")
    metadata = (
        f"Domain: {evidence.get('domain') or 'unknown'}\n"
        f"Source family: {evidence.get('source_family') or 'unknown'}\n"
        f"Published: {evidence.get('published_at') or 'unknown'}\n"
        f"Title: {evidence.get('title') or ''}"
    )
    text = str(evidence.get("text") or "")[:max_evidence_chars].strip()
    return (
        f"Claim:\n{claim.strip()}\n\nEvidence metadata:\n{metadata}\n\n"
        f"Evidence text:\n{text}\n\nRequested dimension: {task}\n"
        f"Classes:\n{TASKS[task]}\n\nReturn only A, B, or C."
    )


def build_examples(rows: list[dict], tasks: list[str], top_k: int,
                   max_evidence_chars: int) -> list[dict]:
    examples = []
    for row in rows:
        for evidence in row.get("evidence_shortlist", [])[:top_k]:
            for task in tasks:
                examples.append({
                    "id": row["id"],
                    "claim_id": row.get("claim_id"),
                    "evidence_id": evidence["evidence_id"],
                    "evidence_rank": evidence.get("shortlist_rank"),
                    "task": task,
                    "prompt": compose_prompt(
                        row.get("claim", ""), evidence, task,
                        max_evidence_chars,
                    ),
                })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS),
                        default=list(TASKS))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-evidence-chars", type=int, default=1800)
    parser.add_argument("--max-length", type=int, default=2304)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit claims, not pair-task examples")
    args = parser.parse_args()
    if min(args.top_k, args.max_evidence_chars, args.max_length, args.batch_size) <= 0:
        parser.error("top-k, lengths and batch size must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("constraint tasks must be unique")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("label_used") or audit.get("gold_evidence_used"):
        parser.error("C2b input audit contains forbidden supervision")
    if audit.get("test_split_used") or audit.get("split") == "test":
        parser.error("C2b development scoring must not consume the test split")
    if sha256_file(args.shortlist) != audit.get("shortlist_sha256"):
        parser.error("shortlist hash does not match the C2a audit")
    rows = load_jsonl(args.shortlist)
    if args.limit:
        rows = rows[:args.limit]
    examples = build_examples(
        rows, args.tasks, args.top_k, args.max_evidence_chars
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "protocol": "P2_open_web_C2b_frozen_constraint_scorer",
        "shortlist_sha256": audit["shortlist_sha256"],
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "tasks": args.tasks,
        "top_k": args.top_k,
        "max_evidence_chars": args.max_evidence_chars,
        "max_length": args.max_length,
    }
    configuration_signature = sha256_text(json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ))
    run_manifest_path = args.output_root / "run_manifest.json"
    if run_manifest_path.exists():
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if previous.get("configuration_signature") != configuration_signature:
            parser.error(
                "output-root contains scores from another configuration; "
                "choose a new output-root"
            )
    else:
        run_manifest = {
            **configuration,
            "configuration_signature": configuration_signature,
            "label_used": False,
            "gold_evidence_used": False,
            "test_split_used": False,
        }
        temporary = run_manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(run_manifest_path)
    output_path = args.output_root / "constraint_scores.jsonl"
    completed_rows = load_jsonl(output_path)
    completed = {
        (row["id"], row["evidence_id"], row["task"])
        for row in completed_rows
    }
    if len(completed) != len(completed_rows):
        parser.error("duplicate rows in resumable C2b output")
    pending = [
        row for row in examples
        if (row["id"], row["evidence_id"], row["task"]) not in completed
    ]

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    device = torch.device(args.device)
    model.to(device).eval()

    answer_ids = []
    for code in ("A", "B", "C"):
        ids = as_token_ids(tokenizer.encode(code, add_special_tokens=False))
        if len(ids) != 1:
            raise ValueError(f"answer code {code} is not one token: {ids}")
        answer_ids.append(int(ids[0]))
    if len(set(answer_ids)) != 3:
        raise ValueError("answer token IDs are not distinct")
    answer_index = torch.tensor(answer_ids, device=device)

    def collate(batch: list[dict]) -> dict:
        chats = [tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        ) for row in batch]
        sequences = [head_tail_truncate(as_token_ids(tokenizer.encode(
            chat, add_special_tokens=False
        )), args.max_length) for chat in chats]
        width = max(map(len, sequences))
        input_ids = torch.full(
            (len(sequences), width), tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, -len(sequence):] = torch.tensor(
                sequence, dtype=torch.long
            )
            attention_mask[index, -len(sequence):] = 1
        encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"rows": batch, "encoded": encoded}

    loader = DataLoader(
        pending, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )
    started = time.perf_counter()
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        for batch in tqdm(loader, desc="C2b constraint scoring"):
            encoded = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["encoded"].items()
            }
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                try:
                    logits = model(**encoded, logits_to_keep=1).logits.float()
                except TypeError:
                    logits = model(**encoded).logits.float()
            # Batches are left padded, so the final column is the next-token
            # prediction position for every row (also with logits_to_keep=1).
            final_logits = logits[:, -1]
            probability = torch.softmax(final_logits[:, answer_index], -1).cpu()
            for index, row in enumerate(batch["rows"]):
                probs = probability[index].tolist()
                record = {
                    "id": row["id"], "claim_id": row.get("claim_id"),
                    "evidence_id": row["evidence_id"],
                    "evidence_rank": row.get("evidence_rank"),
                    "task": row["task"],
                    "prediction": "ABC"[max(range(3), key=probs.__getitem__)],
                    "probabilities": {code: value for code, value in zip(
                        "ABC", probs, strict=True
                    )},
                    "label_used": False, "gold_evidence_used": False,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    all_rows = load_jsonl(output_path)
    expected_keys = {
        (row["id"], row["evidence_id"], row["task"]) for row in examples
    }
    relevant = [
        row for row in all_rows
        if (row["id"], row["evidence_id"], row["task"]) in expected_keys
    ]
    task_counts = {
        task: sum(row["task"] == task for row in relevant) for task in args.tasks
    }
    prediction_counts = {
        task: {
            code: sum(
                row["task"] == task and row.get("prediction") == code
                for row in relevant
            )
            for code in "ABC"
        }
        for task in args.tasks
    }
    invalid_probability_rows = 0
    for row in relevant:
        probabilities = list(row.get("probabilities", {}).values())
        if (
            len(probabilities) != 3
            or not all(math.isfinite(float(value)) for value in probabilities)
            or abs(sum(map(float, probabilities)) - 1.0) > 1e-4
        ):
            invalid_probability_rows += 1
    summary = {
        "protocol": "P2_open_web_C2b_frozen_constraint_scorer",
        "claims": len(rows), "top_k": args.top_k,
        "tasks": args.tasks, "expected_scores": len(examples),
        "completed_scores": len(relevant),
        "complete": len(relevant) == len(examples),
        "task_counts": task_counts,
        "prediction_counts": prediction_counts,
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "label_token_ids": answer_ids,
        "shortlist_sha256": audit["shortlist_sha256"],
        "configuration_signature": configuration_signature,
        "invalid_probability_rows": invalid_probability_rows,
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
