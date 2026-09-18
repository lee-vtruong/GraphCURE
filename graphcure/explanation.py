"""Structured grounded explanation schemas and validation logic for Phase B18.

This module provides:
1. StructuredExplanation dataclass and serialization.
2. Prompt builders for teacher generation, multitask explanation training, and direct verdict inference.
3. Strict validation and verification logic to filter hallucinated, ungrounded, or malformed explanations.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


VALID_VERDICTS = {"SUPPORTED", "REFUTED", "NEI"}
VERDICT_TO_LETTER = {"SUPPORTED": "A", "REFUTED": "B", "NEI": "C"}
LETTER_TO_VERDICT = {"A": "SUPPORTED", "B": "REFUTED", "C": "NEI"}

TEACHER_SYSTEM_PROMPT = (
    "You are an expert fact-checking rationale annotator. Your task is to analyze a claim "
    "and a numbered list of retrieved evidence passages to produce a structured explanation.\n"
    "CRITICAL RULES:\n"
    "1. You must judge whether the supplied retrieved evidence ACTUALLY GROUNDS the claim verdict. "
    "Do NOT fabricate facts or assume external knowledge not in the evidence.\n"
    "2. If the retrieved evidence is insufficient, irrelevant, or missing key facts to justify the verdict, "
    "set \"grounded\": false.\n"
    "3. Only reference evidence passage numbers that actually exist in the prompt (1-based index).\n"
    "4. Output MUST be valid JSON conforming strictly to the required schema."
)

STUDENT_EXPLANATION_SYSTEM_PROMPT = (
    "You are an evidence-grounded reasoning assistant. Given a claim and retrieved evidence, "
    "provide a concise structured explanation identifying the key evidence passages, the reasoning, "
    "and any missing information. Output strictly as JSON."
)

STUDENT_VERDICT_SYSTEM_PROMPT = (
    "You are an evidence-grounded fact verifier. Use only the retrieved evidence supplied by the user. "
    "Decide whether the claim is A: supported, B: refuted, or C: not enough information. "
    "Conflicting, irrelevant, or insufficient evidence must not be treated as support. "
    "Answer with exactly one letter: A, B, or C."
)


@dataclass
class StructuredExplanation:
    grounded: bool
    verdict: str
    key_evidence_ids: list[int]
    reason: str
    missing_information: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def compose_evidence_block(evidence_texts: list[str], max_evidence_chars: int = 2200) -> str:
    sections = []
    for index, text in enumerate(evidence_texts, 1):
        clean_text = text.strip()[:max_evidence_chars].strip()
        sections.append(f"[{index}] {clean_text}")
    return "\n\n".join(sections)


def compose_teacher_prompt(
    claim: str,
    evidence_texts: list[str],
    gold_verdict: str,
    max_evidence_chars: int = 2200,
) -> str:
    """Prompt for the larger teacher model to synthesize a structured explanation."""
    norm_gold = gold_verdict.strip().upper()
    if norm_gold in LETTER_TO_VERDICT:
        norm_gold = LETTER_TO_VERDICT[norm_gold]

    evidence_str = compose_evidence_block(evidence_texts, max_evidence_chars=max_evidence_chars)
    num_evidence = len(evidence_texts)

    schema_example_supported = json.dumps({
        "grounded": True,
        "verdict": "SUPPORTED",
        "key_evidence_ids": [1],
        "reason": "Evidence [1] directly confirms the claim details.",
        "missing_information": None,
    }, indent=2)

    schema_example_nei = json.dumps({
        "grounded": True,
        "verdict": "NEI",
        "key_evidence_ids": [1],
        "reason": "Evidence [1] discusses the topic but does not verify the specific assertion.",
        "missing_information": "Verification of the specific date and location asserted in the claim.",
    }, indent=2)

    schema_example_ungrounded = json.dumps({
        "grounded": False,
        "verdict": "NEI",
        "key_evidence_ids": [],
        "reason": "The retrieved evidence is completely off-topic and cannot ground any verdict.",
        "missing_information": "Primary evidence addressing the claim.",
    }, indent=2)

    prompt = f"""Claim:
{claim.strip()}

Retrieved evidence ({num_evidence} passages):
{evidence_str}

Reference Gold Verdict: {norm_gold}

Instructions:
1. Examine if the retrieved evidence passages ([1] to [{num_evidence}]) can genuinely ground the verdict.
2. If the evidence supports or refutes the claim, set "grounded": true and specify "key_evidence_ids" (e.g. [1, 2]).
3. If the evidence is insufficient or does not match the gold verdict, set "grounded": false.
4. If the verdict is NEI, state what specific information is missing in "missing_information".
5. Return ONLY a JSON object matching one of these schemas:

Example SUPPORTED:
{schema_example_supported}

Example NEI:
{schema_example_nei}

Example UNGROUNDED (evidence does not justify claim):
{schema_example_ungrounded}

Output JSON:"""
    return prompt


def compose_student_verdict_prompt(
    claim: str,
    evidence_texts: list[str],
    max_evidence_chars: int = 2200,
) -> str:
    """Direct verdict prompt (Prompt A) used for both training and inference."""
    sections = [f"Claim:\n{claim.strip()}", "Retrieved evidence:"]
    sections.append(compose_evidence_block(evidence_texts, max_evidence_chars=max_evidence_chars))
    sections.append("Return only A, B, or C.")
    return "\n\n".join(sections)


def compose_student_explanation_prompt(
    claim: str,
    evidence_texts: list[str],
    max_evidence_chars: int = 2200,
) -> str:
    """Multitask explanation prompt (Prompt B) used ONLY during training."""
    num_evidence = len(evidence_texts)
    sections = [
        f"Claim:\n{claim.strip()}",
        f"Retrieved evidence ({num_evidence} passages):",
        compose_evidence_block(evidence_texts, max_evidence_chars=max_evidence_chars),
        "Provide a structured explanation JSON with keys: 'verdict', 'key_evidence_ids', 'reason', 'missing_information'.",
    ]
    return "\n\n".join(sections)


def extract_json_payload(raw_text: str) -> str:
    """Extract first JSON block from markdown codeblocks or raw text."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def validate_explanation(
    payload: Any,
    num_evidence: int,
    evidence_texts: list[str] | None = None,
) -> tuple[bool, StructuredExplanation | None, list[str]]:
    """Strictly validates an explanation payload according to Phase B18 criteria.
    
    Returns:
        (is_valid, structured_explanation_or_none, list_of_issues)
    """
    issues = []
    data = payload
    if isinstance(data, str):
        cleaned = extract_json_payload(data)
        try:
            data = json.loads(cleaned)
        except Exception as exc:
            return False, None, [f"JSONDecodeError: {exc}"]

    if not isinstance(data, dict):
        return False, None, ["Root payload is not a JSON object"]

    # Check required keys
    required_keys = {"grounded", "verdict", "key_evidence_ids", "reason", "missing_information"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        return False, None, [f"Missing required keys: {sorted(missing_keys)}"]

    # Check grounded
    grounded = data.get("grounded")
    if not isinstance(grounded, bool):
        issues.append(f"'grounded' must be boolean, got {type(grounded).__name__}")

    # Check verdict
    verdict = str(data.get("verdict", "")).strip().upper()
    if verdict in LETTER_TO_VERDICT:
        verdict = LETTER_TO_VERDICT[verdict]
    if verdict not in VALID_VERDICTS:
        issues.append(f"Invalid verdict '{verdict}'; must be one of {VALID_VERDICTS}")

    # Check key_evidence_ids
    key_ids = data.get("key_evidence_ids")
    validated_ids: list[int] = []
    if not isinstance(key_ids, list):
        issues.append("'key_evidence_ids' must be a list")
    else:
        for item in key_ids:
            try:
                val = int(item)
                if val < 1 or val > num_evidence:
                    issues.append(f"key_evidence_id {val} out of range [1, {num_evidence}]")
                else:
                    validated_ids.append(val)
            except (ValueError, TypeError):
                issues.append(f"Invalid evidence id {item}")

    # Check reason
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        issues.append("'reason' must be a non-empty string")
    else:
        reason = reason.strip()

    # Check missing_information
    missing_info = data.get("missing_information")
    if missing_info is not None:
        if not isinstance(missing_info, str):
            issues.append("'missing_information' must be string or null")
        else:
            missing_info = missing_info.strip() or None

    # Logic consistency checks
    if grounded is True and verdict in {"SUPPORTED", "REFUTED"}:
        if not validated_ids:
            issues.append(f"Grounded {verdict} requires at least 1 key_evidence_id")

    if verdict == "NEI" and grounded is True:
        if not missing_info:
            issues.append("NEI verdict requires non-empty 'missing_information'")

    # Optional quote grounding check
    if evidence_texts and reason and validated_ids:
        # Check if quotes in reason correspond to texts in key_evidence_ids
        quoted_spans = re.findall(r'"([^"]{10,})"', reason)
        active_evidence_text = " ".join(
            evidence_texts[i - 1] for i in validated_ids if 1 <= i <= len(evidence_texts)
        ).casefold()
        for quote in quoted_spans:
            clean_q = " ".join(quote.casefold().split())
            clean_corpus = " ".join(active_evidence_text.split())
            if clean_q not in clean_corpus:
                issues.append(f"Quoted text '{quote[:30]}...' not found in referenced evidence")

    if issues:
        return False, None, issues

    explanation = StructuredExplanation(
        grounded=bool(grounded),
        verdict=verdict,
        key_evidence_ids=sorted(set(validated_ids)),
        reason=reason,
        missing_information=missing_info,
    )
    return True, explanation, []
