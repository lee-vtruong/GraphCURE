"""Prepare leakage-safe sufficiency targets for MOCHEG Phase B6.

The target is intentionally evidence-conditioned.  A supported/refuted claim
is sufficient only when a labelled relevant article is present in the supplied
candidate set.  NEI is always insufficient for a binary polarity decision.
Claims without article qrels receive no sufficiency target unless their verdict
is NEI.  Gold evidence may be injected into *training* candidates only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.cache_mocheg_reasoning_features import inject_gold_candidate
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_qwen3_lora_verifier import read_documents


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gold_document_ids(claim: dict, documents: dict[str, str]) -> list[str]:
    return sorted({
        str(value) for value in claim.get("text_evidence_ids", [])
        if documents.get(str(value))
    })


def sufficiency_target(label: int, gold_ids: list[str],
                       candidate_ids: list[str]) -> int | None:
    """Return 1=sufficient, 0=insufficient, or None=unlabelled.

    NEI is definitionally insufficient for a supported/refuted polarity
    decision.  For decidable labels, qrels are required to supervise whether
    the current retrieved set is sufficient.
    """
    if int(label) == 2:
        return 0
    if not gold_ids:
        return None
    return int(bool(set(gold_ids).intersection(candidate_ids)))


def build_row(claim: dict, retrieved: dict, documents: dict[str, str],
              top_k: int, inject_gold: bool) -> dict:
    natural = [
        str(value)
        for value in retrieved.get("retrieved_evidence_ids", [])[:top_k]
        if documents.get(str(value))
    ]
    natural = list(dict.fromkeys(natural))[:top_k]
    candidates = natural
    injected = False
    if inject_gold:
        candidates, injected = inject_gold_candidate(
            claim, natural, documents, top_k
        )
    gold = gold_document_ids(claim, documents)
    gold_set = set(gold)
    label = int(claim["label"])
    target = sufficiency_target(label, gold, candidates)
    polarity = label if target == 1 and label in (0, 1) else None
    return {
        "id": claim["id"],
        "claim_id": claim.get("claim_id"),
        "claim": claim.get("claim", ""),
        "label": label,
        "candidate_ids": candidates,
        "natural_candidate_ids": natural,
        "ablated_candidate_ids": [
            value for value in candidates if value not in gold_set
        ],
        "gold_evidence_ids": gold,
        "qrel_available": bool(gold),
        "natural_gold_hit": bool(gold_set.intersection(natural)),
        "candidate_gold_hit": bool(gold_set.intersection(candidates)),
        "train_gold_injected": bool(injected),
        "sufficiency_target": target,
        "polarity_target": polarity,
    }


def prepare_split(split: str, manifest_root: Path, retrieval_root: Path,
                  raw_root: Path, output_root: Path, top_k: int,
                  inject_train_gold: bool, limit: int = 0) -> dict:
    manifest_path = manifest_root / f"{split}.jsonl"
    retrieval_path = retrieval_root / f"{split}.jsonl"
    corpus_path = raw_root / split / "Corpus2.csv"
    claims = read_jsonl(manifest_path)
    if limit:
        claims = claims[:limit]
    retrieval = {row["id"]: row for row in read_jsonl(retrieval_path)}
    documents = read_documents(corpus_path)
    rows = []
    for claim in claims:
        if claim["id"] not in retrieval:
            raise ValueError(f"{split}: retrieval missing {claim['id']}")
        rows.append(build_row(
            claim, retrieval[claim["id"]], documents, top_k,
            inject_train_gold and split == "train",
        ))

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{split}.jsonl"
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    target_counts = Counter(
        "unknown" if row["sufficiency_target"] is None
        else "sufficient" if row["sufficiency_target"] == 1
        else "insufficient"
        for row in rows
    )
    summary = {
        "split": split,
        "samples": len(rows),
        "top_k": top_k,
        "label_counts": dict(Counter(str(row["label"]) for row in rows)),
        "sufficiency_target_counts": dict(target_counts),
        "qrel_available": sum(row["qrel_available"] for row in rows),
        "natural_gold_hits": sum(row["natural_gold_hit"] for row in rows),
        "candidate_gold_hits": sum(row["candidate_gold_hit"] for row in rows),
        "train_gold_injected": sum(row["train_gold_injected"] for row in rows),
        "polarity_targets": sum(row["polarity_target"] is not None for row in rows),
        "validation_gold_injection": False,
        "manifest_sha256": file_sha256(manifest_path),
        "retrieval_sha256": file_sha256(retrieval_path),
        "corpus_sha256": file_sha256(corpus_path),
        "output_sha256": file_sha256(output_path),
    }
    (output_root / f"{split}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--retrieval-root", type=Path,
                        default=Path("outputs/retrieval_mocheg_qwen3_reranked"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/processed/mocheg_b6_targets"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--splits", nargs="+", choices=["train", "val"],
                        default=["train", "val"])
    parser.add_argument("--inject-train-gold", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    summaries = {}
    for split in args.splits:
        summaries[split] = prepare_split(
            split, args.manifest_root, args.retrieval_root, args.raw_root,
            args.output_root, args.top_k, args.inject_train_gold, args.limit,
        )
        print(json.dumps(summaries[split], indent=2), flush=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
