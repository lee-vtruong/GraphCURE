"""Score matched direct and constraint-aware C2c open-web verdicts."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from graphcure.open_web import load_jsonl, sha256_file, sha256_text
from scripts.score_mocheg_open_constraints import (
    as_token_ids,
    head_tail_truncate,
)


SYSTEM_DIRECT = (
    "You are an evidence-grounded fact verifier. Use only the supplied web "
    "evidence. Decide whether the claim is A: supported, B: refuted, or C: "
    "not enough information. Conflicting, irrelevant, or insufficient "
    "evidence must not be treated as support. Answer with exactly one letter: "
    "A, B, or C."
)
SYSTEM_CONSTRAINT = (
    SYSTEM_DIRECT + " Each evidence item also has frozen diagnostic scores. "
    "Use stance only when sufficiency is credible. Entity or temporal conflict "
    "must reduce that item's influence. Do not count duplicated claims across "
    "sources as independent corroboration."
)
TASK_LABELS = {
    "stance": {"A": "support", "B": "refute", "C": "neither"},
    "sufficiency": {"A": "sufficient", "B": "partial", "C": "irrelevant"},
    "entity": {"A": "match", "B": "conflict", "C": "unknown"},
    "temporal": {"A": "compatible", "B": "conflict", "C": "unknown"},
}


def constraint_text(scores: dict[str, dict]) -> str:
    fields = []
    for task in TASK_LABELS:
        row = scores[task]
        probabilities = row["probabilities"]
        prediction = row["prediction"]
        fields.append(
            f"{task}={TASK_LABELS[task][prediction]} "
            f"(A={probabilities['A']:.3f},B={probabilities['B']:.3f},"
            f"C={probabilities['C']:.3f})"
        )
    return "; ".join(fields)


def compose_verdict_prompt(
    claim: str,
    evidence: list[dict],
    constraints: dict[tuple[str, str], dict[str, dict]],
    mode: str,
    max_evidence_chars: int,
) -> str:
    sections = [f"Claim:\n{claim.strip()}", "Retrieved web evidence:"]
    for index, item in enumerate(evidence, 1):
        metadata = (
            f"domain={item.get('domain') or 'unknown'}; "
            f"family={item.get('source_family') or 'unknown'}; "
            f"published={item.get('published_at') or 'unknown'}"
        )
        block = f"[{index}] {metadata}\n{str(item.get('text') or '')[:max_evidence_chars]}"
        if mode == "constraint":
            key = (str(item["evidence_id"]), str(item.get("shortlist_rank")))
            block += "\nFrozen diagnostics: " + constraint_text(constraints[key])
        sections.append(block)
    sections.append(
        "Return only A for supported, B for refuted, or C for not enough information."
    )
    return "\n\n".join(sections)


def load_constraint_map(path: Path) -> dict[str, dict[tuple[str, str], dict]]:
    result: dict[str, dict[tuple[str, str], dict]] = {}
    for row in load_jsonl(path):
        claim = result.setdefault(str(row["id"]), {})
        key = (str(row["evidence_id"]), str(row.get("evidence_rank")))
        tasks = claim.setdefault(key, {})
        task = str(row["task"])
        if task in tasks:
            raise ValueError(f"duplicate constraint score: {row['id']} {key} {task}")
        tasks[task] = row
    return result


def build_examples(
    rows: list[dict],
    constraints: dict[str, dict],
    top_k: int,
    max_evidence_chars: int,
) -> list[dict]:
    examples = []
    for row in rows:
        evidence = row.get("evidence_shortlist", [])[:top_k]
        claim_constraints = constraints.get(str(row["id"]), {})
        for item in evidence:
            key = (str(item["evidence_id"]), str(item.get("shortlist_rank")))
            observed = set(claim_constraints.get(key, {}))
            if observed != set(TASK_LABELS):
                raise ValueError(
                    f"incomplete constraints for {row['id']} evidence {key}: {observed}"
                )
        for mode, system in (
            ("direct", SYSTEM_DIRECT), ("constraint", SYSTEM_CONSTRAINT)
        ):
            examples.append({
                "id": str(row["id"]),
                "mode": mode,
                "system": system,
                "prompt": compose_verdict_prompt(
                    row.get("claim", ""), evidence, claim_constraints,
                    mode, max_evidence_chars,
                ),
            })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--constraint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-evidence-chars", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if min(args.top_k, args.max_evidence_chars, args.max_length, args.batch_size) <= 0:
        parser.error("top-k, lengths and batch size must be positive")
    if not args.adapter.is_dir():
        parser.error(f"adapter directory does not exist: {args.adapter}")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    constraint_summary_path = args.constraint_root / "summary.json"
    constraint_path = args.constraint_root / "constraint_scores.jsonl"
    constraint_summary = json.loads(
        constraint_summary_path.read_text(encoding="utf-8")
    )
    forbidden = ("label_used", "gold_evidence_used", "test_split_used")
    if any(audit.get(key) or constraint_summary.get(key) for key in forbidden):
        parser.error("C2c inputs contain forbidden label, gold, or test use")
    if not constraint_summary.get("complete"):
        parser.error("C2b constraint scoring is incomplete")
    if constraint_summary.get("invalid_probability_rows"):
        parser.error("C2b contains invalid probability rows")
    shortlist_hash = sha256_file(args.shortlist)
    if shortlist_hash != audit.get("shortlist_sha256"):
        parser.error("shortlist hash does not match C2a audit")
    if shortlist_hash != constraint_summary.get("shortlist_sha256"):
        parser.error("C2b scores do not match the shortlist")

    rows = load_jsonl(args.shortlist)
    if args.limit:
        rows = rows[:args.limit]
    constraints = load_constraint_map(constraint_path)
    examples = build_examples(
        rows, constraints, args.top_k, args.max_evidence_chars
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "protocol": "P2_open_web_C2c_matched_verdict_screen",
        "shortlist_sha256": shortlist_hash,
        "constraint_scores_sha256": sha256_file(constraint_path),
        "model": args.model,
        "adapter": str(args.adapter),
        "top_k": args.top_k,
        "max_evidence_chars": args.max_evidence_chars,
        "max_length": args.max_length,
    }
    signature = sha256_text(json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ))
    run_manifest_path = args.output_root / "run_manifest.json"
    if run_manifest_path.exists():
        old = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if old.get("configuration_signature") != signature:
            parser.error("output-root belongs to another C2c configuration")
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
    old_rows = load_jsonl(output_path)
    completed = {(str(row["id"]), str(row["mode"])) for row in old_rows}
    if len(completed) != len(old_rows):
        parser.error("duplicate rows in resumable C2c output")
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
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["prompt"]},
            ], tokenize=False, add_generation_prompt=True)
            sequences.append(head_tail_truncate(
                as_token_ids(tokenizer.encode(chat, add_special_tokens=False)),
                args.max_length,
            ))
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
        for batch in tqdm(loader, desc="C2c matched verdict scoring"):
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
                probs = probability[index].tolist()
                record = {
                    "id": row["id"], "mode": row["mode"],
                    "prediction": int(max(range(3), key=probs.__getitem__)),
                    "probabilities": probs,
                    "label_used": False, "gold_evidence_used": False,
                }
                handle.write(json.dumps(record) + "\n")

    result_rows = load_jsonl(output_path)
    expected = {(row["id"], row["mode"]) for row in examples}
    relevant = [
        row for row in result_rows
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
        mode: sum(row["mode"] == mode for row in relevant)
        for mode in ("direct", "constraint")
    }
    prediction_counts = {
        mode: {
            str(label): sum(
                row["mode"] == mode and int(row["prediction"]) == label
                for row in relevant
            )
            for label in range(3)
        }
        for mode in ("direct", "constraint")
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
