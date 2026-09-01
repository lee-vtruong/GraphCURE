"""Phase-B6 Qwen3 LoRA verifier with sufficiency/polarity decomposition.

This script is validation-only by construction.  It continues a frozen article
verifier adapter and trains three task prompts with one shared LoRA:

* direct verdict: supported / refuted / NEI;
* evidence sufficiency: sufficient for a binary verdict or insufficient;
* polarity: supported or refuted, only when sufficient evidence is labelled.

At inference the hierarchical distribution is
``[p(sufficient)*p(support), p(sufficient)*p(refute), p(insufficient)]``.
It can be blended with the direct verdict distribution using a small, explicit
validation grid.  Freeze the selected blend weight before multi-seed runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from graphcure.optimization import project_auxiliary_gradients
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_cached_verifier import expected_calibration_error
from scripts.train_mocheg_qwen3_lora_verifier import (
    LABEL_CODES,
    SYSTEM_PROMPT,
    as_token_id_list,
    load_fold,
    read_documents,
)
from scripts.prepare_mocheg_sv_folds import sha256


SUFFICIENCY_SYSTEM_PROMPT = (
    "You are checking whether the supplied evidence is sufficient to decide "
    "the claim as supported or refuted. Relevant-looking evidence that does "
    "not establish either polarity is insufficient. Answer with exactly one "
    "letter: Y for sufficient or N for insufficient."
)
POLARITY_SYSTEM_PROMPT = (
    "Assume the supplied evidence is sufficient for a binary fact-checking "
    "decision. Decide whether it supports or refutes the claim. Answer with "
    "exactly one letter: A for supported or B for refuted."
)
TASK_SYSTEM_PROMPTS = {
    "verdict": SYSTEM_PROMPT,
    "sufficiency": SUFFICIENCY_SYSTEM_PROMPT,
    "ablation": SUFFICIENCY_SYSTEM_PROMPT,
    "polarity": POLARITY_SYSTEM_PROMPT,
}


def deterministic_fraction(sample_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_signature(path: Path) -> dict | None:
    if not path.exists():
        return None
    result = {"path": str(path)}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        candidate = path / name
        if candidate.exists():
            result[f"{name}_sha256"] = file_sha256(candidate)
    return result


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evidence_prompt(claim: str, evidence: list[str], task: str,
                    max_evidence_chars: int) -> str:
    sections = [f"Claim:\n{claim.strip()}", "Retrieved evidence:"]
    if evidence:
        for index, text in enumerate(evidence, 1):
            sections.append(f"[{index}] {text[:max_evidence_chars].strip()}")
    else:
        sections.append("[No usable evidence retrieved]")
    instruction = {
        "verdict": "Return only A, B, or C.",
        "sufficiency": "Return only Y or N.",
        "ablation": "Return only Y or N.",
        "polarity": "Return only A or B.",
    }[task]
    sections.append(instruction)
    return "\n\n".join(sections)


def task_prompt_ids(tokenizer, task: str, user: str, budget: int) -> list[int]:
    messages = [
        {"role": "system", "content": TASK_SYSTEM_PROMPTS[task]},
        {"role": "user", "content": user},
    ]
    ids = as_token_id_list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    ))
    if len(ids) <= budget:
        return ids
    prefix = min(512, budget // 3)
    return ids[:prefix] + ids[-(budget - prefix):]


def verbalizer_token_ids(tokenizer) -> dict[str, int]:
    result = {}
    for code in ("A", "B", "C", "Y", "N"):
        ids = as_token_id_list(tokenizer.encode(code, add_special_tokens=False))
        if len(ids) != 1:
            raise ValueError(f"verbalizer {code!r} is not one token: {ids}")
        result[code] = ids[0]
    if len(set(result.values())) != len(result):
        raise ValueError(f"verbalizer tokens are not distinct: {result}")
    return result


def hierarchical_probabilities(sufficiency: np.ndarray,
                               polarity: np.ndarray) -> np.ndarray:
    """Compose [supported, refuted, NEI] from Y/N and A/B probabilities."""
    sufficiency = np.asarray(sufficiency, dtype=float)
    polarity = np.asarray(polarity, dtype=float)
    if sufficiency.ndim != 2 or sufficiency.shape[1] != 2:
        raise ValueError("sufficiency probabilities must have shape [N, 2]")
    if polarity.ndim != 2 or polarity.shape != sufficiency.shape:
        raise ValueError("polarity probabilities must have shape [N, 2]")
    sufficient = sufficiency[:, 0]
    result = np.column_stack([
        sufficient * polarity[:, 0],
        sufficient * polarity[:, 1],
        sufficiency[:, 1],
    ])
    return result / np.clip(result.sum(1, keepdims=True), 1e-12, None)


def blend_probabilities(direct: np.ndarray, hierarchical: np.ndarray,
                        hierarchical_weight: float) -> np.ndarray:
    if not 0 <= hierarchical_weight <= 1:
        raise ValueError("hierarchical weight must be in [0, 1]")
    result = ((1 - hierarchical_weight) * np.asarray(direct, dtype=float)
              + hierarchical_weight * np.asarray(hierarchical, dtype=float))
    return result / np.clip(result.sum(1, keepdims=True), 1e-12, None)


def probability_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    prediction = np.asarray(probabilities).argmax(-1)
    class_f1 = f1_score(
        labels, prediction, labels=[0, 1, 2], average=None,
        zero_division=0,
    )
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(
            labels, prediction, labels=[0, 1, 2], average="macro",
            zero_division=0,
        )),
        "class_f1": {
            "supported": float(class_f1[0]),
            "refuted": float(class_f1[1]),
            "nei": float(class_f1[2]),
        },
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=[0, 1, 2]
        ).tolist(),
        "ece_10": expected_calibration_error(probabilities, labels),
    }


def load_frozen_anchor_predictions(path: Path, expected_ids: list[str],
                                   expected_labels: np.ndarray) -> tuple[dict, np.ndarray]:
    rows = read_jsonl(path)
    ids = [row["id"] for row in rows]
    labels = np.asarray([int(row["gold"]) for row in rows])
    probabilities = np.asarray(
        [row["probabilities"] for row in rows], dtype=float
    )
    if ids != expected_ids or not np.array_equal(labels, expected_labels):
        raise ValueError(
            f"frozen anchor predictions are not aligned with B6 validation: {path}"
        )
    return probability_metrics(probabilities, labels), probabilities


class B6Claims:
    def __init__(self, target_path: Path, corpus_path: Path,
                 max_evidence_chars: int, limit: int = 0,
                 allowed_ids: set[str] | None = None) -> None:
        rows = read_jsonl(target_path)
        if allowed_ids is not None:
            rows = [row for row in rows if row["id"] in allowed_ids]
            missing = allowed_ids - {row["id"] for row in rows}
            if missing:
                raise ValueError(
                    f"fold IDs missing from B6 targets: {len(missing)}"
                )
        if limit:
            rows = rows[:limit]
        documents = read_documents(corpus_path)
        self.rows = []
        for row in rows:
            candidates = [
                value for value in row.get("candidate_ids", [])
                if documents.get(str(value))
            ]
            ablated = [
                value for value in row.get("ablated_candidate_ids", [])
                if documents.get(str(value))
            ]
            item = dict(row)
            item["evidence"] = [documents[str(value)] for value in candidates]
            item["ablated_evidence"] = [
                documents[str(value)] for value in ablated
            ]
            item["prompts"] = {
                task: evidence_prompt(
                    item.get("claim", ""),
                    item["ablated_evidence"] if task == "ablation"
                    else item["evidence"],
                    task, max_evidence_chars,
                )
                for task in ("verdict", "sufficiency", "ablation", "polarity")
            }
            self.rows.append(item)


class B6TrainingTasks(Dataset):
    def __init__(self, claims: B6Claims, ablation_ratio: float, seed: int,
                 verdict_weight: float, sufficiency_weight: float,
                 polarity_weight: float, ablation_weight: float) -> None:
        weights = {
            "verdict": verdict_weight,
            "sufficiency": sufficiency_weight,
            "polarity": polarity_weight,
            "ablation": ablation_weight,
        }
        if verdict_weight <= 0:
            raise ValueError("verdict_weight must be positive")
        if any(value < 0 for value in weights.values()):
            raise ValueError("training task weights must be non-negative")
        self.rows = []
        for row in claims.rows:
            self.rows.append({
                "id": row["id"], "task": "verdict",
                "user": row["prompts"]["verdict"],
                "target_code": LABEL_CODES[int(row["label"])],
                "weight": verdict_weight, "label": int(row["label"]),
            })
            target = row.get("sufficiency_target")
            if target is not None and sufficiency_weight > 0:
                self.rows.append({
                    "id": row["id"], "task": "sufficiency",
                    "user": row["prompts"]["sufficiency"],
                    "target_code": "Y" if int(target) == 1 else "N",
                    "weight": sufficiency_weight, "label": int(row["label"]),
                })
            polarity = row.get("polarity_target")
            if polarity is not None and polarity_weight > 0:
                self.rows.append({
                    "id": row["id"], "task": "polarity",
                    "user": row["prompts"]["polarity"],
                    "target_code": "A" if int(polarity) == 0 else "B",
                    "weight": polarity_weight, "label": int(row["label"]),
                })
                if (ablation_ratio > 0 and ablation_weight > 0 and
                        deterministic_fraction(row["id"], seed) < ablation_ratio):
                    self.rows.append({
                        "id": row["id"], "task": "ablation",
                        "user": row["prompts"]["ablation"],
                        "target_code": "N", "weight": ablation_weight,
                        "label": int(row["label"]),
                    })
        self.counts = dict(Counter(row["task"] for row in self.rows))

    def repeat_verdict_to_length(self, target_length: int) -> None:
        if set(self.counts) != {"verdict"}:
            raise ValueError(
                "compute matching is only valid for a verdict-only control"
            )
        if target_length < len(self.rows):
            raise ValueError(
                "matched training length cannot be shorter than the control"
            )
        original = self.rows
        self.rows = [dict(original[index % len(original)])
                     for index in range(target_length)]
        self.counts = {"verdict": len(self.rows)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class B6EvaluationTask(Dataset):
    def __init__(self, claims: B6Claims, task: str) -> None:
        self.rows = [{
            "id": row["id"], "label": int(row["label"]), "task": task,
            "user": row["prompts"][task],
        } for row in claims.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def make_collate(tokenizer, token_ids: dict[str, int], max_length: int,
                 training: bool):
    pad = tokenizer.pad_token_id

    def collate(rows: list[dict]) -> dict:
        sequences = [
            task_prompt_ids(tokenizer, row["task"], row["user"], max_length)
            for row in rows
        ]
        width = max(len(value) for value in sequences)
        input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            input_ids[index, :len(sequence)] = torch.tensor(
                sequence, dtype=torch.long
            )
            attention[index, :len(sequence)] = 1
        result = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "prediction_index": attention.sum(-1) - 1,
            "ids": [row["id"] for row in rows],
            "tasks": [row["task"] for row in rows],
            "class_labels": torch.tensor(
                [int(row["label"]) for row in rows], dtype=torch.long
            ),
        }
        if training:
            result["target_ids"] = torch.tensor(
                [token_ids[row["target_code"]] for row in rows],
                dtype=torch.long,
            )
            result["loss_weights"] = torch.tensor(
                [float(row["weight"]) for row in rows], dtype=torch.float32
            )
        return result
    return collate


def captured_gradients(parameters: list[torch.nn.Parameter]) -> tuple[
        torch.Tensor | None, ...]:
    return tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )


def add_gradient_buffers(
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
) -> tuple[torch.Tensor | None, ...]:
    result = []
    for left, right in zip(first, second):
        if left is None:
            result.append(right)
        elif right is None:
            result.append(left)
        else:
            result.append(left + right)
    return tuple(result)


@torch.inference_mode()
def score_task(model, loader, answer_ids: list[int], device) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    ids, labels, probabilities = [], [], []
    token_index = torch.tensor(answer_ids, device=device)
    for batch in tqdm(loader, desc=f"validation {loader.dataset.rows[0]['task']}",
                      leave=False):
        inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(
                device, non_blocking=True
            ),
        }
        prediction_index = batch["prediction_index"].to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(**inputs).logits.float()
        selected = logits[
            torch.arange(len(prediction_index), device=device), prediction_index
        ][:, token_index]
        probabilities.extend(torch.softmax(selected, -1).cpu().tolist())
        ids.extend(batch["ids"])
        labels.extend(batch["class_labels"].tolist())
    return ids, np.asarray(labels), np.asarray(probabilities, dtype=float)


def evaluate_all(model, loaders: dict, token_ids: dict[str, int], device,
                 weights: list[float]) -> tuple[dict, list[dict]]:
    ids, labels, direct = score_task(
        model, loaders["verdict"],
        [token_ids["A"], token_ids["B"], token_ids["C"]], device,
    )
    suff_ids, suff_labels, sufficiency = score_task(
        model, loaders["sufficiency"],
        [token_ids["Y"], token_ids["N"]], device,
    )
    polarity_ids, polarity_labels, polarity = score_task(
        model, loaders["polarity"],
        [token_ids["A"], token_ids["B"]], device,
    )
    if ids != suff_ids or ids != polarity_ids:
        raise ValueError("validation task IDs are not aligned")
    if not np.array_equal(labels, suff_labels) or not np.array_equal(
        labels, polarity_labels
    ):
        raise ValueError("validation task labels are not aligned")
    hierarchical = hierarchical_probabilities(sufficiency, polarity)
    direct_metrics = probability_metrics(direct, labels)
    hierarchical_metrics = probability_metrics(hierarchical, labels)
    candidates = []
    for weight in sorted(set(weights)):
        probabilities = blend_probabilities(direct, hierarchical, weight)
        candidates.append({
            "hierarchical_weight": float(weight),
            "metrics": probability_metrics(probabilities, labels),
            "probabilities": probabilities,
        })
    best = max(candidates, key=lambda row: (
        row["metrics"]["macro_f1"], -row["hierarchical_weight"]
    ))
    rows = []
    for index, sample_id in enumerate(ids):
        probability = best["probabilities"][index]
        rows.append({
            "id": sample_id, "gold": int(labels[index]),
            "prediction": int(probability.argmax()),
            "probabilities": probability.tolist(),
            "direct_probabilities": direct[index].tolist(),
            "hierarchical_probabilities": hierarchical[index].tolist(),
            "sufficiency_probabilities": sufficiency[index].tolist(),
            "polarity_probabilities": polarity[index].tolist(),
        })
    result = {
        "direct": direct_metrics,
        "hierarchical": hierarchical_metrics,
        "weight_grid": [{
            "hierarchical_weight": row["hierarchical_weight"],
            **row["metrics"],
        } for row in candidates],
        "selected_hierarchical_weight": best["hierarchical_weight"],
        "selected": best["metrics"],
    }
    return result, rows


def parse_weights(value: str) -> list[float]:
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not weights or any(weight < 0 or weight > 1 for weight in weights):
        raise ValueError("hierarchical weights must be comma-separated values in [0, 1]")
    return weights


def validate_target_metadata(target_root: Path) -> dict:
    summaries = {}
    for split in ("train", "val"):
        path = target_root / f"{split}.summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing target audit: {path}")
        summaries[split] = json.loads(path.read_text(encoding="utf-8"))
    if summaries["val"].get("validation_gold_injection") is not False:
        raise ValueError("validation targets are not leakage-safe")
    if summaries["train"]["top_k"] != summaries["val"]["top_k"]:
        raise ValueError("train/validation target top-k mismatch")
    return summaries


def validate_fold_target_metadata(
    training_root: Path, held_root: Path, fold_payload: dict
) -> dict:
    summaries = {}
    for name, root in (("training", training_root), ("held", held_root)):
        path = root / "train.summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing fold target audit: {path}")
        summaries[name] = json.loads(path.read_text(encoding="utf-8"))
    expected = fold_payload["manifest_sha256"]
    if any(row.get("manifest_sha256") != expected for row in summaries.values()):
        raise ValueError("B6 fold targets do not match the fold manifest")
    if summaries["training"]["top_k"] != summaries["held"]["top_k"]:
        raise ValueError("training and held-fold target top-k mismatch")
    if summaries["held"].get("train_gold_injected", 0) != 0:
        raise ValueError("held-fold targets contain forbidden gold injection")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", type=Path,
                        default=Path("data/processed/mocheg_b6_targets"))
    parser.add_argument("--held-target-root", type=Path, default=None,
                        help="Natural, non-injected targets used by held folds")
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
    parser.add_argument("--fold-spec", type=Path, default=None)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--fixed-checkpoint-epoch", type=int, default=0,
                        help="Select exactly this epoch; required for folds")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--initial-adapter", type=Path,
                        default=Path("outputs/mocheg_qwen3_lora_seed42_v16/best_adapter"))
    parser.add_argument("--resume-from-best", action="store_true",
                        help="continue from OUTPUT/best_adapter after interruption")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/mocheg_b6_hierarchical_seed42"))
    parser.add_argument("--max-evidence-chars", type=int, default=2200)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--warmup-ratio", type=float, default=.05)
    parser.add_argument("--ablation-ratio", type=float, default=.25)
    parser.add_argument("--verdict-loss-weight", type=float, default=1.0)
    parser.add_argument("--sufficiency-loss-weight", type=float, default=.5)
    parser.add_argument("--polarity-loss-weight", type=float, default=.5)
    parser.add_argument("--ablation-loss-weight", type=float, default=.5)
    parser.add_argument("--hierarchical-weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--gradient-mode", choices=("standard", "pcgrad"),
                        default="standard")
    parser.add_argument("--auxiliary-gradient-scale", type=float, default=1.0)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--conflict-temperature", type=float, default=None,
                        help="Enable conflict-severity adaptive projection")
    parser.add_argument("--match-training-examples-from", type=Path,
                        default=None, help=(
                            "B6 summary whose task-count total is used for a "
                            "compute-matched verdict-only control"
                        ))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--anchor-macro-f1", type=float,
                        default=.670470583003719)
    parser.add_argument("--anchor-predictions", type=Path, default=None,
                        help="defaults to INITIAL_ADAPTER/../val_predictions.jsonl")
    parser.add_argument("--maximum-anchor-reproduction-delta", type=float,
                        default=.01)
    parser.add_argument("--minimum-delta", type=float, default=.005)
    parser.add_argument("--minimum-nei-delta", type=float, default=.02)
    parser.add_argument("--maximum-supported-drop", type=float, default=.005)
    parser.add_argument("--maximum-accuracy-drop", type=float, default=.003)
    parser.add_argument("--skip-anchor-check", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.ablation_ratio <= 1:
        raise ValueError("ablation ratio must be in [0, 1]")
    if not 0 <= args.projection_strength <= 1:
        raise ValueError("projection strength must be in [0, 1]")
    if args.conflict_temperature is not None and args.conflict_temperature <= 0:
        raise ValueError("conflict temperature must be positive")
    fit_ids, held_ids, fold_payload = load_fold(
        args.fold_spec, args.fold_index
    )
    cv_mode = fold_payload is not None
    if cv_mode and not 1 <= args.fixed_checkpoint_epoch <= args.epochs:
        raise ValueError(
            "train-only folds require --fixed-checkpoint-epoch in [1, epochs]"
        )
    if args.fixed_checkpoint_epoch < 0 or args.fixed_checkpoint_epoch > args.epochs:
        raise ValueError("fixed checkpoint epoch must be in [0, epochs]")
    if cv_mode and args.held_target_root is None:
        raise ValueError("train-only folds require --held-target-root")
    weights = parse_weights(args.hierarchical_weights)
    if cv_mode and weights != [0.0]:
        raise ValueError(
            "B6-C fold runs require --hierarchical-weights 0; no held-fold "
            "blend search is allowed"
        )
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    target_metadata = (
        validate_fold_target_metadata(
            args.target_root, args.held_target_root, fold_payload
        ) if cv_mode else validate_target_metadata(args.target_root)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_ids = verbalizer_token_ids(tokenizer)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    adapter_path = args.output / "best_adapter" if args.resume_from_best else args.initial_adapter
    if args.resume_from_best and not adapter_path.exists():
        raise FileNotFoundError(f"resume adapter does not exist: {adapter_path}")
    if adapter_path.exists():
        model = PeftModel.from_pretrained(
            base_model, str(adapter_path), is_trainable=True
        )
    else:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(base_model, config)
    model.print_trainable_parameters()

    train_corpus = args.raw_root / "train" / "Corpus2.csv"
    val_target = (
        args.held_target_root / "train.jsonl" if cv_mode
        else args.target_root / "val.jsonl"
    )
    val_corpus = train_corpus if cv_mode else (
        args.raw_root / "val" / "Corpus2.csv"
    )
    train_claims = B6Claims(
        args.target_root / "train.jsonl", train_corpus,
        args.max_evidence_chars, args.limit_train, fit_ids,
    )
    val_claims = B6Claims(
        val_target, val_corpus,
        args.max_evidence_chars, args.limit_val, held_ids,
    )
    train = B6TrainingTasks(
        train_claims, args.ablation_ratio, args.seed,
        args.verdict_loss_weight, args.sufficiency_loss_weight,
        args.polarity_loss_weight, args.ablation_loss_weight,
    )
    if args.match_training_examples_from is not None:
        reference = json.loads(args.match_training_examples_from.read_text(
            encoding="utf-8"
        ))
        counts = reference.get("training_task_counts")
        if not isinstance(counts, dict) or not counts:
            raise ValueError(
                "matched B6 summary has no training_task_counts: "
                f"{args.match_training_examples_from}"
            )
        train.repeat_verdict_to_length(sum(int(value) for value in counts.values()))
    if args.gradient_mode == "pcgrad":
        if args.batch_size != 1:
            raise ValueError("pcgrad currently requires --batch-size 1")
        if args.auxiliary_gradient_scale <= 0:
            raise ValueError("auxiliary_gradient_scale must be positive")
    elif args.projection_strength != 1 or args.conflict_temperature is not None:
        raise ValueError(
            "projection controls require --gradient-mode pcgrad"
        )
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate(
            tokenizer, token_ids, args.max_length, training=True
        ),
        pin_memory=device.type == "cuda",
    )
    validation_loaders = {
        task: DataLoader(
            B6EvaluationTask(val_claims, task),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            collate_fn=make_collate(
                tokenizer, token_ids, args.max_length, training=False
            ),
            pin_memory=device.type == "cuda",
        )
        for task in ("verdict", "sufficiency", "polarity")
    }

    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = max(
        1, int(np.ceil(len(train_loader) / args.gradient_accumulation))
    )
    updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(updates * args.warmup_ratio), updates
    )

    initial, initial_rows = evaluate_all(
        model, validation_loaders, token_ids, device, weights
    )
    initial_direct = initial["direct"]
    anchor_path = args.anchor_predictions
    if anchor_path is None:
        anchor_path = args.initial_adapter.parent / "val_predictions.jsonl"
    anchor_source = "current_reinference"
    anchor_metrics = initial_direct
    if not args.limit_val and anchor_path.exists():
        expected_ids = [row["id"] for row in val_claims.rows]
        expected_labels = np.asarray([
            int(row["label"]) for row in val_claims.rows
        ])
        anchor_metrics, _ = load_frozen_anchor_predictions(
            anchor_path, expected_ids, expected_labels
        )
        anchor_source = "frozen_val_predictions"
    if cv_mode and anchor_source != "frozen_val_predictions":
        raise FileNotFoundError(
            "fold anchor predictions are required for aligned comparison: "
            f"{anchor_path}"
        )
    if cv_mode and abs(
        initial_direct["macro_f1"] - anchor_metrics["macro_f1"]
    ) > args.maximum_anchor_reproduction_delta:
        raise ValueError(
            "fold anchor reproduction drift exceeds the fixed tolerance: "
            f"reinference={initial_direct['macro_f1']:.6f}, "
            f"artifact={anchor_metrics['macro_f1']:.6f}"
        )
    if not cv_mode and not args.skip_anchor_check and not args.limit_val:
        if abs(anchor_metrics["macro_f1"] - args.anchor_macro_f1) > .002:
            raise ValueError(
                "frozen anchor artifact does not match its registered score: "
                f"artifact={anchor_metrics['macro_f1']:.6f}, "
                f"registered={args.anchor_macro_f1:.6f}"
            )
        reproduction_delta = (
            initial_direct["macro_f1"] - anchor_metrics["macro_f1"]
        )
        if abs(reproduction_delta) > args.maximum_anchor_reproduction_delta:
            raise ValueError(
                "current libraries reproduce the frozen adapter outside the "
                "allowed drift: "
                f"observed={initial_direct['macro_f1']:.6f}, "
                f"frozen={anchor_metrics['macro_f1']:.6f}, "
                f"delta={reproduction_delta:+.6f}"
            )
    history = [{"epoch": 0, "train_loss": None, **initial}]
    print(json.dumps(history[0]), flush=True)
    fixed_epoch = args.fixed_checkpoint_epoch or None
    best = (
        float("-inf") if fixed_epoch is not None
        else initial["selected"]["macro_f1"]
    )
    best_epoch = 0
    best_weight = 0.0 if fixed_epoch is not None else (
        initial["selected_hierarchical_weight"]
    )
    best_state = None if fixed_epoch is not None else {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters() if value.requires_grad
    }
    stale = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        gradient_cosines = []
        gradient_conflicts = []
        applied_projection_strengths = []
        protected_windows = 0
        unprotected_auxiliary_windows = 0
        empty_gradients = tuple(None for _ in parameters)
        primary_buffer = empty_gradients
        auxiliary_buffer = empty_gradients
        primary_in_window = auxiliary_in_window = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, 1):
            inputs = {
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(
                    device, non_blocking=True
                ),
            }
            positions = batch["prediction_index"].to(device)
            targets = batch["target_ids"].to(device)
            loss_weights = batch["loss_weights"].to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(**inputs).logits
                selected = logits[
                    torch.arange(len(positions), device=device), positions
                ].float()
                per_example = F.cross_entropy(
                    selected, targets, reduction="none"
                )
                loss = (per_example * loss_weights).sum() / loss_weights.sum()
            if args.gradient_mode == "standard":
                (loss / args.gradient_accumulation).backward()
            else:
                task = batch["tasks"][0]
                scaled_loss = loss if task == "verdict" else (
                    loss * args.auxiliary_gradient_scale
                )
                (scaled_loss / args.gradient_accumulation).backward()
                observed = captured_gradients(parameters)
                optimizer.zero_grad(set_to_none=True)
                if task == "verdict":
                    primary_buffer = add_gradient_buffers(
                        primary_buffer, observed
                    )
                    primary_in_window += 1
                else:
                    auxiliary_buffer = add_gradient_buffers(
                        auxiliary_buffer, observed
                    )
                    auxiliary_in_window += 1
            running += float(loss.detach())
            if (step % args.gradient_accumulation == 0 or
                    step == len(train_loader)):
                if args.gradient_mode == "pcgrad":
                    combined, diagnostics = project_auxiliary_gradients(
                        primary_buffer, auxiliary_buffer,
                        projection_strength=args.projection_strength,
                        conflict_temperature=args.conflict_temperature,
                    )
                    for parameter, gradient in zip(parameters, combined):
                        parameter.grad = gradient
                    gradient_cosines.append(diagnostics["cosine"])
                    gradient_conflicts.append(diagnostics["conflict"])
                    applied_projection_strengths.append(
                        diagnostics["applied_projection_strength"]
                    )
                    if primary_in_window and auxiliary_in_window:
                        protected_windows += 1
                    elif auxiliary_in_window and not primary_in_window:
                        unprotected_auxiliary_windows += 1
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                primary_buffer = empty_gradients
                auxiliary_buffer = empty_gradients
                primary_in_window = auxiliary_in_window = 0
            progress.set_postfix(loss=f"{loss.item():.4f}")
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
        }
        if args.gradient_mode == "pcgrad":
            row["gradient_cosine"] = float(np.mean(gradient_cosines))
            row["gradient_conflict_rate"] = float(np.mean(gradient_conflicts))
            row["applied_projection_strength_mean"] = float(
                np.mean(applied_projection_strengths)
            )
            row["protected_windows"] = protected_windows
            row["unprotected_auxiliary_windows"] = (
                unprotected_auxiliary_windows
            )
        if fixed_epoch is not None and epoch < fixed_epoch:
            row["held_fold_evaluated"] = False
            history.append(row)
            print(json.dumps(row), flush=True)
            continue
        evaluation, _ = evaluate_all(
            model, validation_loaders, token_ids, device, weights
        )
        row.update(evaluation)
        row["held_fold_evaluated"] = True
        history.append(row)
        print(json.dumps(row), flush=True)
        score = evaluation["selected"]["macro_f1"]
        selected = (
            epoch == fixed_epoch if fixed_epoch is not None else score > best
        )
        if selected:
            best = score
            best_epoch = epoch
            best_weight = evaluation["selected_hierarchical_weight"]
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.named_parameters()
                if value.requires_grad
            }
            temporary = args.output / "best.tmp"
            target = args.output / "best_adapter"
            if temporary.exists():
                shutil.rmtree(temporary)
            model.save_pretrained(temporary, safe_serialization=True)
            tokenizer.save_pretrained(temporary)
            if target.exists():
                shutil.rmtree(target)
            temporary.replace(target)
            if fixed_epoch is not None:
                break
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("fixed checkpoint epoch was not reached")
    model.load_state_dict(best_state, strict=False)
    final, prediction_rows = evaluate_all(
        model, validation_loaders, token_ids, device, [best_weight]
    )
    candidate = final["selected"]
    macro_delta = candidate["macro_f1"] - anchor_metrics["macro_f1"]
    nei_delta = (
        candidate["class_f1"]["nei"] - anchor_metrics["class_f1"]["nei"]
    )
    supported_delta = (
        candidate["class_f1"]["supported"]
        - anchor_metrics["class_f1"]["supported"]
    )
    accuracy_delta = candidate["accuracy"] - anchor_metrics["accuracy"]
    promotion_gate = {
        "decomposition_active": best_weight > 0,
        "macro_f1_delta_at_least_minimum": macro_delta >= args.minimum_delta,
        "nei_f1_delta_at_least_minimum": nei_delta >= args.minimum_nei_delta,
        "supported_f1_drop_within_limit": (
            supported_delta >= -args.maximum_supported_drop
        ),
        "accuracy_drop_within_limit": (
            accuracy_delta >= -args.maximum_accuracy_drop
        ),
    }
    promotion_gate["passed"] = all(promotion_gate.values())
    summary = {
        "method": "GraphCURE-B6-sufficiency-polarity",
        "protocol": (
            "train_only_duplicate_safe_fixed_epoch_cv" if cv_mode
            else "external_validation"
        ),
        "fold": args.fold_index if cv_mode else None,
        "fixed_checkpoint_epoch": fixed_epoch,
        "official_validation_used_for_selection": not cv_mode,
        "mode": "candidate" if promotion_gate["passed"]
        else "rejected_keep_article_anchor",
        "accepted": promotion_gate["passed"],
        "model": args.model,
        "initial_adapter": str(args.initial_adapter),
        "loaded_adapter": str(adapter_path) if adapter_path.exists() else None,
        "loaded_adapter_signature": adapter_signature(adapter_path),
        "resumed_from_best": args.resume_from_best,
        "git_commit": current_git_commit(),
        "verbalizer_token_ids": token_ids,
        "training_task_counts": train.counts,
        "best_epoch": best_epoch,
        "selected_hierarchical_weight": best_weight,
        "anchor": anchor_metrics,
        "anchor_source": anchor_source,
        "anchor_predictions": str(anchor_path) if anchor_path.exists() else None,
        "anchor_reinference": initial_direct,
        "anchor_reproduction_macro_f1_delta": (
            initial_direct["macro_f1"] - anchor_metrics["macro_f1"]
        ),
        "final_candidate": candidate,
        "direct_at_best": final["direct"],
        "hierarchical_at_best": final["hierarchical"],
        "deltas": {
            "accuracy": accuracy_delta,
            "macro_f1": macro_delta,
            "supported_f1": supported_delta,
            "refuted_f1": (
                candidate["class_f1"]["refuted"]
                - anchor_metrics["class_f1"]["refuted"]
            ),
            "nei_f1": nei_delta,
        },
        "promotion_gate": promotion_gate,
        "history": history,
        "target_metadata": target_metadata,
        "test_split_used": False,
        "provenance": {
            "fold_spec_sha256": sha256(args.fold_spec)
            if args.fold_spec else None,
            "training_target_sha256": file_sha256(
                args.target_root / "train.jsonl"
            ),
            "held_target_sha256": file_sha256(val_target),
        },
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "val_predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
