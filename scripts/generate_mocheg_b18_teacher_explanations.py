"""Generate structured, grounded explanations using a teacher LLM for Phase B18.

CRITICAL PROTOCOL RULES:
1. The teacher is ONLY allowed to see P1 system-retrieved evidence. Gold qrel injection
   is strictly prohibited to avoid evaluation-distribution mismatch.
2. If the retrieved evidence is insufficient to justify the gold label, the teacher
   must output grounded=false.
3. Every generated explanation is passed through automated schema and grounding filters.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from graphcure.explanation import (
    LETTER_TO_VERDICT,
    TEACHER_SYSTEM_PROMPT,
    compose_teacher_prompt,
    validate_explanation,
)
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_qwen3_lora_verifier import read_documents


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def mock_teacher_generate(claim: str, evidence: list[str], gold_verdict: str) -> dict[str, Any]:
    """Deterministic mock generator for testing and pipeline dry-runs."""
    norm_gold = gold_verdict.strip().upper()
    if norm_gold in LETTER_TO_VERDICT:
        norm_gold = LETTER_TO_VERDICT[norm_gold]

    if not evidence:
        return {
            "grounded": False,
            "verdict": "NEI",
            "key_evidence_ids": [],
            "reason": "No retrieved evidence was available to verify the claim.",
            "missing_information": "Primary evidence supporting or refuting the claim.",
        }

    if norm_gold == "NEI":
        return {
            "grounded": True,
            "verdict": "NEI",
            "key_evidence_ids": [1],
            "reason": f"Evidence [1] mentions aspects of the claim but does not confirm the core assertion.",
            "missing_information": "Direct verification of the claim from reliable sources.",
        }

    # If first evidence mentions words from claim, ground it
    claim_words = set(claim.casefold().split())
    ev1_words = set(evidence[0].casefold().split())
    if len(claim_words & ev1_words) >= 2:
        return {
            "grounded": True,
            "verdict": norm_gold,
            "key_evidence_ids": [1],
            "reason": f"Evidence [1] directly addresses the key entities in the claim.",
            "missing_information": None,
        }

    return {
        "grounded": False,
        "verdict": "NEI",
        "key_evidence_ids": [],
        "reason": "The retrieved evidence passages do not contain enough relevant information.",
        "missing_information": "Verification evidence linking the entities and assertions.",
    }


def generate_with_model(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 512,
) -> str:
    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    import torch
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_tokens = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B18 Teacher Explanation Generator")
    parser.add_argument("--manifest", type=Path, required=True, help="Input claims manifest (.jsonl)")
    parser.add_argument("--retrieval", type=Path, required=True, help="Retrieved evidence manifest (.jsonl)")
    parser.add_argument("--corpus", type=Path, required=True, help="Evidence documents CSV/corpus")
    parser.add_argument("--output", type=Path, required=True, help="Output explanations (.jsonl)")
    parser.add_argument("--summary", type=Path, required=True, help="Output summary audit (.json)")
    parser.add_argument("--folds", type=Path, default=None, help="Optional fold file to filter IDs")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to generate train explanations for")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Teacher model name/path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved evidence passages")
    parser.add_argument("--max-evidence-chars", type=int, default=2200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock teacher for testing")
    args = parser.parse_args()

    allowed_ids = None
    if args.folds is not None and args.folds.is_file():
        fold_data = json.loads(args.folds.read_text(encoding="utf-8"))
        fold_entry = next(f for f in fold_data["folds"] if f["fold"] == args.fold)
        allowed_ids = set(fold_entry["train_ids"])
        logging.info("Restricted generation to fold %d train set (%d claims)", args.fold, len(allowed_ids))

    all_claims = read_jsonl(args.manifest)
    if allowed_ids is not None:
        claims = [c for c in all_claims if str(c["id"]) in allowed_ids]
    else:
        claims = all_claims

    if args.limit > 0:
        claims = claims[:args.limit]

    logging.info("Processing %d claims", len(claims))
    retrieval_by_id = {str(row["id"]): row for row in read_jsonl(args.retrieval)}
    documents = read_documents(args.corpus)

    model = None
    tokenizer = None
    if not args.mock:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logging.info("Loading teacher model: %s", args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and "cuda" in args.device else torch.float32,
            device_map=args.device,
            trust_remote_code=True,
        )
        model.eval()

    results = []
    audit_counter = Counter()

    for claim_row in tqdm(claims, desc="Generating explanations"):
        cid = str(claim_row["id"])
        claim_text = claim_row.get("claim", "")
        raw_label = int(claim_row["label"])
        gold_code = {0: "A", 1: "B", 2: "C"}.get(raw_label, "C")
        gold_verdict = LETTER_TO_VERDICT.get(gold_code, "NEI")

        retrieved_row = retrieval_by_id.get(cid)
        if retrieved_row is None:
            logging.warning("Missing retrieval for claim %s", cid)
            candidate_ids = []
        else:
            candidate_ids = [
                eid for eid in retrieved_row.get("retrieved_evidence_ids", [])[:args.top_k]
                if eid in documents
            ]
        evidence_texts = [documents[eid] for eid in candidate_ids]

        if args.mock:
            raw_output = mock_teacher_generate(claim_text, evidence_texts, gold_verdict)
        else:
            prompt = compose_teacher_prompt(
                claim=claim_text,
                evidence_texts=evidence_texts,
                gold_verdict=gold_verdict,
                max_evidence_chars=args.max_evidence_chars,
            )
            raw_text = generate_with_model(model, tokenizer, prompt, device=args.device)
            raw_output = raw_text

        is_valid, explanation, issues = validate_explanation(
            raw_output, num_evidence=len(evidence_texts), evidence_texts=evidence_texts
        )

        audit_counter["total"] += 1
        if is_valid and explanation is not None:
            audit_counter["valid"] += 1
            if explanation.grounded:
                audit_counter["grounded_true"] += 1
            else:
                audit_counter["grounded_false"] += 1
            audit_counter[f"verdict_{explanation.verdict}"] += 1
            record = {
                "id": cid,
                "claim": claim_text,
                "label": raw_label,
                "retrieved_evidence_ids": candidate_ids,
                "grounded": explanation.grounded,
                "verdict": explanation.verdict,
                "key_evidence_ids": explanation.key_evidence_ids,
                "reason": explanation.reason,
                "missing_information": explanation.missing_information,
                "is_valid": True,
                "validation_issues": [],
            }
        else:
            audit_counter["invalid"] += 1
            record = {
                "id": cid,
                "claim": claim_text,
                "label": raw_label,
                "retrieved_evidence_ids": candidate_ids,
                "grounded": False,
                "verdict": "NEI",
                "key_evidence_ids": [],
                "reason": "Failed automated schema validation.",
                "missing_information": "Uncorrupted teacher rationale.",
                "is_valid": False,
                "validation_issues": issues,
            }
        results.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_payload = {
        "phase": "B18-A",
        "teacher_model": args.model if not args.mock else "mock",
        "mock": args.mock,
        "total_claims": len(claims),
        "statistics": dict(audit_counter),
        "top_k": args.top_k,
        "max_evidence_chars": args.max_evidence_chars,
        "manifest": str(args.manifest),
        "output": str(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    logging.info("Saved %d explanations to %s", len(results), args.output)
    logging.info("Summary: %s", json.dumps(dict(audit_counter), indent=2))


if __name__ == "__main__":
    main()
