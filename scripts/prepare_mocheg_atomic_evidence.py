"""Build claim-conditioned atomic evidence from MOCHEG article candidates.

The official sentence corpus is used as the canonical ID space.  Retrieved
articles are split into short evidence units; exact text matches inherit their
official ``corpus_id`` while unmatched units receive deterministic synthetic
IDs.  Gold sentence IDs are copied into a separate manifest, but are never
inserted into validation/test candidates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


TAG = re.compile(r"<[^>]+>")
BREAK = re.compile(r"(?i)<(?:p|br|li|h[1-6]|div)[^>]*>")
SPACE = re.compile(r"\s+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'([{A-Z0-9])")
TOKEN = re.compile(r"[a-z0-9]+")


def raise_csv_field_limit() -> int:
    """Raise Python's small default while remaining portable across builds."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 10


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def normalized_text(value: str) -> str:
    value = TAG.sub(" ", html.unescape(value))
    return " ".join(TOKEN.findall(value.lower()))


def clean_html(value: str) -> str:
    value = BREAK.sub("\n", value)
    value = TAG.sub(" ", value)
    value = html.unescape(value).replace("\r", "\n")
    return "\n".join(SPACE.sub(" ", part).strip()
                     for part in value.split("\n") if part.strip())


def split_atomic_units(value: str, min_chars: int = 24,
                       max_chars: int = 900) -> list[str]:
    """Split an article into stable paragraph/sentence-size verification units."""
    units: list[str] = []
    for paragraph in clean_html(value).splitlines():
        pieces = SENTENCE_BOUNDARY.split(paragraph)
        buffer = ""
        for piece in pieces:
            piece = SPACE.sub(" ", piece).strip()
            if not piece:
                continue
            if len(piece) > max_chars:
                words = piece.split()
                chunks, current = [], []
                for word in words:
                    if current and len(" ".join(current + [word])) > max_chars:
                        chunks.append(" ".join(current)); current = []
                    current.append(word)
                if current:
                    chunks.append(" ".join(current))
            else:
                chunks = [piece]
            for chunk in chunks:
                if len(chunk) < min_chars:
                    buffer = f"{buffer} {chunk}".strip()
                    continue
                if buffer:
                    chunk = f"{buffer} {chunk}"; buffer = ""
                units.append(chunk)
        if buffer:
            if units and len(units[-1]) + len(buffer) + 1 <= max_chars:
                units[-1] = f"{units[-1]} {buffer}"
            elif len(buffer) >= min_chars:
                units.append(buffer)
    return list(dict.fromkeys(unit for unit in units if normalized_text(unit)))


def stable_atom_id(parent_id: str, text: str) -> str:
    digest = hashlib.sha256(
        f"{parent_id}\0{normalized_text(text)}".encode("utf-8")
    ).hexdigest()[:24]
    return f"atomic-{digest}"


def pack_context(units: list[str], selected_index: int, radius: int) -> str:
    """Pack a selected sentence with local context while marking its role."""
    if radius < 0:
        raise ValueError("context radius must be non-negative")
    if not 0 <= selected_index < len(units):
        raise IndexError("selected sentence index is outside the article")
    if radius == 0:
        return units[selected_index]
    sections = []
    start, end = max(0, selected_index - radius), min(
        len(units), selected_index + radius + 1
    )
    for index in range(start, end):
        role = "Selected evidence" if index == selected_index else "Local context"
        sections.append(f"[{role}] {units[index]}")
    return "\n".join(sections)


def official_context_windows(sentence_rows: dict[str, dict],
                             radius: int) -> dict[str, str]:
    groups: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    for sentence_id, row in sentence_rows.items():
        key = (row.get("claim_id", "").strip(),
               row.get("relevant_document_id", "").strip())
        groups[key].append((sentence_id, row))

    def position(item: tuple[str, dict]) -> tuple[int, str]:
        raw = item[1].get("paragraph_id", "").strip()
        return (int(raw), raw) if raw.isdigit() else (sys.maxsize, raw)

    result = {}
    for rows in groups.values():
        rows.sort(key=position)
        units = [row.get("paragraph", "").strip() for _, row in rows]
        for index, (sentence_id, _) in enumerate(rows):
            result[sentence_id] = pack_context(units, index, radius)
    return result


def read_sentence_corpus(path: Path) -> tuple[dict[str, dict], dict[tuple[str, str], list[str]]]:
    raise_csv_field_limit()
    rows: dict[str, dict] = {}
    lookup: dict[tuple[str, str], list[str]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            sentence_id = row.get("corpus_id", "").strip()
            text = row.get("paragraph", "").strip()
            owner = row.get("claim_id", "").strip()
            key = (owner, normalized_text(text))
            if sentence_id and key[1]:
                rows.setdefault(sentence_id, row)
                if sentence_id not in lookup[key]:
                    lookup[key].append(sentence_id)
    return rows, lookup


def map_unit_id(owner_claim_id: str, parent_id: str, text: str,
                official_lookup: dict[tuple[str, str], list[str]]) -> tuple[str, bool]:
    matches = official_lookup.get((owner_claim_id, normalized_text(text)), [])
    if len(matches) == 1:
        return matches[0], True
    return stable_atom_id(parent_id, text), False


def build_split(split: str, manifest_root: Path, article_retrieval_root: Path,
                raw_root: Path, sentence_rows: dict[str, dict],
                official_lookup: dict[tuple[str, str], list[str]],
                output_manifest_root: Path, output_corpus_root: Path,
                output_candidates_root: Path, article_top_k: int,
                max_units_per_article: int, context_radius: int = 0) -> dict:
    manifest = read_jsonl(manifest_root / f"{split}.jsonl")
    claims = {row["id"]: row for row in manifest}
    retrieved = read_jsonl(article_retrieval_root / f"{split}.jsonl")
    articles: dict[str, dict] = {}
    raise_csv_field_limit()
    with (raw_root / split / "Corpus2.csv").open(
        encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            evidence_id = row.get("evidence_id", "").strip()
            if evidence_id and evidence_id not in articles:
                articles[evidence_id] = row

    atomic_rows: dict[str, dict] = {}
    parent_atoms: dict[str, list[str]] = {}
    authoritative_matches = 0
    for parent_id in dict.fromkeys(
        value for row in retrieved
        for value in row.get("retrieved_evidence_ids", [])[:article_top_k]
    ):
        article = articles.get(str(parent_id))
        if not article:
            parent_atoms[str(parent_id)] = []
            continue
        owner = article.get("claim_id", "").strip()
        atom_ids: list[str] = []
        units = split_atomic_units(
            article.get("Evidence", "")
        )[:max_units_per_article]
        for position, text in enumerate(units):
            atom_id, authoritative = map_unit_id(
                owner, str(parent_id), text, official_lookup
            )
            authoritative_matches += int(authoritative)
            atomic_rows.setdefault(atom_id, {
                "evidence_id": atom_id,
                "Evidence": pack_context(units, position, context_radius),
                "parent_evidence_id": str(parent_id),
                "owner_claim_id": owner,
                "sentence_position": position,
                "authoritative_sentence": int(authoritative),
            })
            atom_ids.append(atom_id)
        parent_atoms[str(parent_id)] = list(dict.fromkeys(atom_ids))

    # Gold rows are included in the corpus solely so the existing train-only
    # injection path can access them. They are not added to natural candidates.
    missing_gold = 0
    official_windows = official_context_windows(sentence_rows, context_radius)
    for claim in manifest:
        for sentence_id in map(str, claim.get("text_sentence_ids", [])):
            source = sentence_rows.get(sentence_id)
            if not source:
                missing_gold += 1
                continue
            atomic_rows.setdefault(sentence_id, {
                "evidence_id": sentence_id,
                "Evidence": official_windows.get(
                    sentence_id, source.get("paragraph", "")
                ),
                "parent_evidence_id": (
                    f"{source.get('claim_id', '')}-{source.get('relevant_document_id', '')}"
                ),
                "owner_claim_id": source.get("claim_id", ""),
                "sentence_position": source.get("paragraph_id", ""),
                "authoritative_sentence": 1,
            })

    output_manifest_root.mkdir(parents=True, exist_ok=True)
    atomic_manifest = []
    for claim in manifest:
        row = dict(claim)
        row["article_text_evidence_ids"] = row.get("text_evidence_ids", [])
        row["text_evidence_ids"] = [
            str(value) for value in row.get("text_sentence_ids", [])
            if str(value) in atomic_rows
        ]
        row["atomic_evidence_protocol"] = (
            f"official_sentence_qrels_context_radius_{context_radius}_v1"
        )
        atomic_manifest.append(row)
    (output_manifest_root / f"{split}.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in atomic_manifest) + "\n",
        encoding="utf-8",
    )

    candidate_rows = []
    natural_ranks: list[int | None] = []
    for retrieval in retrieved:
        claim = claims[retrieval["id"]]
        parent_ids = [str(value) for value in
                      retrieval.get("retrieved_evidence_ids", [])[:article_top_k]]
        parent_scores = retrieval.get("retrieved_scores", [])[:article_top_k]
        ids, parent_for_atom, parent_rank, position, score = [], [], [], [], []
        seen = set()
        for rank, parent_id in enumerate(parent_ids, 1):
            parent_score = float(parent_scores[rank - 1]) if rank <= len(parent_scores) else 0.0
            for atom_position, atom_id in enumerate(parent_atoms.get(parent_id, []), 1):
                if atom_id in seen:
                    continue
                seen.add(atom_id); ids.append(atom_id); parent_for_atom.append(parent_id)
                parent_rank.append(rank); position.append(atom_position); score.append(parent_score)
        gold = set(map(str, claim.get("text_sentence_ids", [])))
        rank = next((index for index, value in enumerate(ids, 1) if value in gold), None)
        natural_ranks.append(rank)
        candidate_rows.append({
            "id": retrieval["id"], "claim_id": claim["claim_id"],
            "label": int(claim["label"]), "retrieved_evidence_ids": ids,
            "parent_evidence_ids": parent_for_atom, "parent_ranks": parent_rank,
            "sentence_positions": position, "parent_scores": score,
            "gold_evidence_ids": sorted(gold), "first_gold_rank": rank,
            "train_gold_injection": False, "validation_gold_injection": False,
        })
    output_candidates_root.mkdir(parents=True, exist_ok=True)
    (output_candidates_root / f"{split}.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in candidate_rows) + "\n",
        encoding="utf-8",
    )

    split_corpus = output_corpus_root / split
    split_corpus.mkdir(parents=True, exist_ok=True)
    fields = ["evidence_id", "Evidence", "parent_evidence_id", "owner_claim_id",
              "sentence_position", "authoritative_sentence"]
    with (split_corpus / "Corpus2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(atomic_rows.values())

    summary = {
        "split": split, "claims": len(manifest), "atomic_documents": len(atomic_rows),
        "natural_candidate_atoms": sum(len(row["retrieved_evidence_ids"])
                                       for row in candidate_rows),
        "authoritative_article_sentence_matches": authoritative_matches,
        "missing_gold_sentence_rows": missing_gold,
        "natural_gold_coverage": sum(rank is not None for rank in natural_ranks) / max(1, len(natural_ranks)),
        "train_gold_injection": False, "validation_gold_injection": False,
        "context_radius": context_radius,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--article-retrieval-root", type=Path, default=Path("outputs/retrieval_mocheg_qwen3_reranked"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--sentence-corpus", type=Path, default=Path("data/raw/mocheg_dataset/extracted/mocheg/supplementary/Corpus3_sentence_level.csv"))
    parser.add_argument("--output-manifest-root", type=Path, default=Path("data/processed/mocheg_atomic_manifest"))
    parser.add_argument("--output-corpus-root", type=Path, default=Path("data/processed/mocheg_atomic_corpus"))
    parser.add_argument("--output-candidates-root", type=Path, default=Path("outputs/retrieval_mocheg_atomic_candidates"))
    parser.add_argument("--article-top-k", type=int, default=10)
    parser.add_argument("--max-units-per-article", type=int, default=32)
    parser.add_argument("--context-radius", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    sentence_rows, lookup = read_sentence_corpus(args.sentence_corpus)
    summary = {}
    for split in args.splits:
        summary[split] = build_split(
            split, args.manifest_root, args.article_retrieval_root, args.raw_root,
            sentence_rows, lookup, args.output_manifest_root,
            args.output_corpus_root, args.output_candidates_root,
            args.article_top_k, args.max_units_per_article, args.context_radius,
        )
        print(json.dumps({split: summary[split]}, indent=2))
    args.output_candidates_root.mkdir(parents=True, exist_ok=True)
    (args.output_candidates_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
