"""Instruction-aware Qwen3 reranking for MOCHEG verification evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from graphcure.retrieval import first_relevant_rank, retrieval_confidence


DEFAULT_INSTRUCTION = (
    "Rank the document by its utility for verifying the claim. A useful "
    "document must provide independent information that can support, refute, "
    "or resolve the claim. Match entities, time, location, quantities, and "
    "negation; lexical similarity alone is insufficient."
)

PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements '
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_docs(path: Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            evidence_id = row["evidence_id"].strip()
            text = row.get("Evidence", "").replace("<p>", " ").replace("</p>", " ").strip()
            if evidence_id and text:
                documents.setdefault(evidence_id, text)
    return documents


class QwenReranker:
    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device).eval()
        self.yes_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        self.no_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.content_length = max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if self.content_length <= 0:
            raise ValueError("max_length is shorter than the fixed reranker prompt")

    @staticmethod
    def prompt(query: str, document: str, instruction: str) -> str:
        return (
            f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
        )

    @torch.inference_mode()
    def score(
        self,
        pairs: list[tuple[str, str]],
        instruction: str,
        batch_size: int,
    ) -> np.ndarray:
        prompts = [self.prompt(query, document, instruction) for query, document in pairs]
        scores: list[float] = []
        for start in range(0, len(prompts), batch_size):
            batch = self.tokenizer(
                prompts[start:start + batch_size],
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.content_length,
            )
            batch["input_ids"] = [
                self.prefix_tokens + tokens + self.suffix_tokens
                for tokens in batch["input_ids"]
            ]
            batch = self.tokenizer.pad(
                batch,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**batch).logits[:, -1]
            binary = torch.stack((logits[:, self.no_id], logits[:, self.yes_id]), -1)
            scores.extend(torch.log_softmax(binary.float(), -1)[:, 1].exp().cpu().tolist())
        return np.asarray(scores, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/retrieval_mocheg_qwen3_reranked"))
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    if args.top_k > args.candidate_k:
        parser.error("--top-k cannot exceed --candidate-k")

    args.output_root.mkdir(parents=True, exist_ok=True)
    reranker = QwenReranker(args.model, args.device, args.max_length)
    summary_path = args.output_root / "summary.json"
    summary: dict[str, dict] = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists() else {}
    )
    for split in args.splits:
        documents = read_docs(args.raw_root / split / "Corpus2.csv")
        claims = {
            row["id"]: row
            for row in read_jsonl(args.manifest_root / f"{split}.jsonl")
        }
        retrieved = read_jsonl(args.retrieval_root / f"{split}.jsonl")
        output: list[dict] = []
        ranks: list[int | None] = []
        for row in tqdm(retrieved, desc=f"{split} Qwen3 reranking"):
            claim = claims[row["id"]]
            candidates = [
                evidence_id
                for evidence_id in row.get("retrieved_evidence_ids", [])[:args.candidate_k]
                if documents.get(evidence_id)
            ]
            scores = reranker.score(
                [(claim.get("claim", ""), documents[evidence_id])
                 for evidence_id in candidates],
                args.instruction,
                args.batch_size,
            ) if candidates else np.empty(0, dtype=np.float32)
            order = np.argsort(-scores, kind="stable")[:args.top_k]
            ids = [candidates[index] for index in order]
            values = [float(scores[index]) for index in order]
            gold = {str(value) for value in claim.get("text_evidence_ids", [])}
            rank = first_relevant_rank(ids, gold)
            ranks.append(rank)
            output.append({
                **row,
                "retrieved_evidence_ids": ids,
                "retrieved_scores": values,
                "reranker_scores": values,
                "retrieval_confidence": retrieval_confidence(values),
                "first_gold_rank": rank,
                "reranker_model": args.model,
            })
        (args.output_root / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
            encoding="utf-8",
        )
        result = {
            "claims": len(output),
            "model": args.model,
            "mrr": float(np.mean([
                1.0 / rank if rank is not None else 0.0 for rank in ranks
            ])),
        }
        for cutoff in sorted({1, 5, args.top_k}):
            result[f"recall@{cutoff}"] = float(np.mean([
                rank is not None and rank <= cutoff for rank in ranks
            ]))
        summary[split] = result
        print(json.dumps({split: result}, indent=2))
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
