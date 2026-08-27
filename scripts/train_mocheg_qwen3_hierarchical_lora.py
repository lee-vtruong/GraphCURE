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

from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_cached_verifier import expected_calibration_error
from scripts.train_mocheg_qwen3_lora_verifier import (
    LABEL_CODES,
    SYSTEM_PROMPT,
    as_token_id_list,
    read_documents,
)


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


class B6Claims:
    def __init__(self, target_path: Path, corpus_path: Path,
                 max_evidence_chars: int, limit: int = 0) -> None:
        rows = read_jsonl(target_path)
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
        self.rows = []
        for row in claims.rows:
            self.rows.append({
                "id": row["id"], "task": "verdict",
                "user": row["prompts"]["verdict"],
                "target_code": LABEL_CODES[int(row["label"])],
                "weight": verdict_weight, "label": int(row["label"]),
            })
            target = row.get("sufficiency_target")
            if target is not None:
                self.rows.append({
                    "id": row["id"], "task": "sufficiency",
                    "user": row["prompts"]["sufficiency"],
                    "target_code": "Y" if int(target) == 1 else "N",
                    "weight": sufficiency_weight, "label": int(row["label"]),
                })
            polarity = row.get("polarity_target")
            if polarity is not None:
                self.rows.append({
                    "id": row["id"], "task": "polarity",
                    "user": row["prompts"]["polarity"],
                    "target_code": "A" if int(polarity) == 0 else "B",
                    "weight": polarity_weight, "label": int(row["label"]),
                })
                if (ablation_ratio > 0 and
                        deterministic_fraction(row["id"], seed) < ablation_ratio):
                    self.rows.append({
                        "id": row["id"], "task": "ablation",
                        "user": row["prompts"]["ablation"],
                        "target_code": "N", "weight": ablation_weight,
                        "label": int(row["label"]),
                    })
        self.counts = dict(Counter(row["task"] for row in self.rows))

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", type=Path,
                        default=Path("data/processed/mocheg_b6_targets"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/mocheg_dataset/extracted/mocheg"))
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
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--anchor-macro-f1", type=float,
                        default=.670470583003719)
    parser.add_argument("--minimum-delta", type=float, default=.005)
    parser.add_argument("--minimum-nei-delta", type=float, default=.02)
    parser.add_argument("--maximum-supported-drop", type=float, default=.005)
    parser.add_argument("--maximum-accuracy-drop", type=float, default=.003)
    parser.add_argument("--skip-anchor-check", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.ablation_ratio <= 1:
        raise ValueError("ablation ratio must be in [0, 1]")
    weights = parse_weights(args.hierarchical_weights)
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
    target_metadata = validate_target_metadata(args.target_root)

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

    train_claims = B6Claims(
        args.target_root / "train.jsonl",
        args.raw_root / "train" / "Corpus2.csv",
        args.max_evidence_chars, args.limit_train,
    )
    val_claims = B6Claims(
        args.target_root / "val.jsonl",
        args.raw_root / "val" / "Corpus2.csv",
        args.max_evidence_chars, args.limit_val,
    )
    train = B6TrainingTasks(
        train_claims, args.ablation_ratio, args.seed,
        args.verdict_loss_weight, args.sufficiency_loss_weight,
        args.polarity_loss_weight, args.ablation_loss_weight,
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
    if (not args.skip_anchor_check and not args.limit_val and
            abs(initial_direct["macro_f1"] - args.anchor_macro_f1) > .002):
        raise ValueError(
            "initial adapter does not reproduce the frozen anchor: "
            f"observed={initial_direct['macro_f1']:.6f}, "
            f"expected={args.anchor_macro_f1:.6f}"
        )
    history = [{"epoch": 0, "train_loss": None, **initial}]
    print(json.dumps(history[0]), flush=True)
    best = initial["selected"]["macro_f1"]
    best_epoch = 0
    best_weight = initial["selected_hierarchical_weight"]
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters() if value.requires_grad
    }
    stale = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
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
            (loss / args.gradient_accumulation).backward()
            running += float(loss.detach())
            if (step % args.gradient_accumulation == 0 or
                    step == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        evaluation, _ = evaluate_all(
            model, validation_loaders, token_ids, device, weights
        )
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            **evaluation,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        score = evaluation["selected"]["macro_f1"]
        if score > best:
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
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state, strict=False)
    final, prediction_rows = evaluate_all(
        model, validation_loaders, token_ids, device, [best_weight]
    )
    candidate = final["selected"]
    macro_delta = candidate["macro_f1"] - initial_direct["macro_f1"]
    nei_delta = (
        candidate["class_f1"]["nei"] - initial_direct["class_f1"]["nei"]
    )
    supported_delta = (
        candidate["class_f1"]["supported"]
        - initial_direct["class_f1"]["supported"]
    )
    accuracy_delta = candidate["accuracy"] - initial_direct["accuracy"]
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
        "anchor": initial_direct,
        "final_candidate": candidate,
        "direct_at_best": final["direct"],
        "hierarchical_at_best": final["hierarchical"],
        "deltas": {
            "accuracy": accuracy_delta,
            "macro_f1": macro_delta,
            "supported_f1": supported_delta,
            "refuted_f1": (
                candidate["class_f1"]["refuted"]
                - initial_direct["class_f1"]["refuted"]
            ),
            "nei_f1": nei_delta,
        },
        "promotion_gate": promotion_gate,
        "history": history,
        "target_metadata": target_metadata,
        "test_split_used": False,
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
