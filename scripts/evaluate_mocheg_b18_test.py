"""Phase B18 Official Test Set Evaluation & Champion Ensembling.

Evaluates trained Qwen3-4B LoRA verifier models on the official locked test set,
using their corresponding evidence streams (raw top-5, B18-B Top-1, or Top-3).
Saves individual test predictions and computes the multi-view heterogeneous ensemble.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphcure.explanation import (
    STUDENT_VERDICT_SYSTEM_PROMPT,
    compose_student_verdict_prompt,
    resolve_corpus_path,
)
from scripts.audit_mocheg_router import bootstrap_delta
from scripts.ensemble_mocheg_runs import compute_metrics, ensemble_predictions
from scripts.run_mocheg_visual_retrieval import read_jsonl
from scripts.train_mocheg_b18_explanation_verifier import (
    B18VerifierDataset,
    evaluate_direct_verdict,
    git_commit,
    make_b18_collate,
)
from scripts.train_mocheg_qwen3_lora_verifier import label_token_ids, read_documents


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def select_retrieval_for_model(
    model_dir: Path,
    retrieval_raw: Path,
    retrieval_b18b: Path | None,
    retrieval_top3: Path | None,
) -> Path:
    """Select appropriate test retrieval manifest matching the model's training configuration."""
    model_str = str(model_dir).replace("\\", "/")
    if "top3" in model_str and retrieval_top3 is not None and retrieval_top3.is_file():
        logging.info("Model %s matched Top-3 retrieval: %s", model_dir.name, retrieval_top3)
        return retrieval_top3
    elif "mocheg_b18b" in model_str and retrieval_b18b is not None and retrieval_b18b.is_file():
        logging.info("Model %s matched B18-B filtered retrieval: %s", model_dir.name, retrieval_b18b)
        return retrieval_b18b
    else:
        logging.info("Model %s matched raw top-5 retrieval: %s", model_dir.name, retrieval_raw)
        return retrieval_raw


def evaluate_single_run(
    model_dir: Path,
    base_model_name: str,
    manifest_path: Path,
    retrieval_path: Path,
    corpus_docs: dict[str, str],
    tokenizer: Any,
    answer_ids: list[int],
    device: torch.device,
    batch_size: int = 4,
    top_k: int = 5,
    max_evidence_chars: int = 2200,
    max_length: int = 2500,
    tag: str = "",
    force: bool = False,
) -> dict[str, Any]:
    if model_dir.name == "best_adapter":
        model_dir = model_dir.parent
    adapter_dir = model_dir / "best_adapter"
    if not adapter_dir.is_dir():
        # Fallback to model_dir itself if adapter is directly in root
        if (model_dir / "adapter_config.json").is_file():
            adapter_dir = model_dir
        else:
            raise FileNotFoundError(f"Adapter directory missing: {adapter_dir}")

    pred_out_path = model_dir / (f"test_predictions_{tag}.jsonl" if tag else "test_predictions.jsonl")
    summary_out_path = model_dir / (f"test_summary_{tag}.json" if tag else "test_summary.json")

    # If predictions already exist and not force, reload them
    if not force and pred_out_path.is_file() and summary_out_path.is_file():
        logging.info("Found cached test predictions in %s, reloading...", model_dir)
        pred_rows = read_jsonl(pred_out_path)
        summary = json.loads(summary_out_path.read_text(encoding="utf-8"))
        summary["predictions"] = pred_rows
        return summary

    logging.info("Evaluating model: %s on %s", model_dir.name, retrieval_path.name)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    claims = read_jsonl(manifest_path)
    retrieval_by_id = {str(r["id"]): r for r in read_jsonl(retrieval_path)}

    dataset = B18VerifierDataset(
        claims=claims,
        retrieval_by_id=retrieval_by_id,
        documents=corpus_docs,
        explanations_by_id=None,
        top_k=top_k,
        max_evidence_chars=max_evidence_chars,
        training=False,
    )

    collate_fn = make_b18_collate(tokenizer, max_length=max_length, training=False, mode="matched_control")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    logging.info("Loading base model: %s", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    base_model.config.use_cache = False

    logging.info("Attaching LoRA adapter from: %s", adapter_dir)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir)).to(device)

    metrics = evaluate_direct_verdict(model, loader, answer_ids, device)

    # Free GPU memory immediately
    del model, base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Save individual predictions
    pred_rows = metrics.pop("predictions")
    with pred_out_path.open("w", encoding="utf-8") as f:
        for r in pred_rows:
            f.write(json.dumps(r) + "\n")

    summary_payload = {
        "split": "test",
        "model_dir": str(model_dir),
        "adapter_dir": str(adapter_dir),
        "manifest": str(manifest_path),
        "retrieval": str(retrieval_path),
        "sample_count": len(pred_rows),
        **metrics,
    }
    summary_out_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    logging.info("Saved test evaluation for %s (Macro-F1=%.5f)", model_dir.name, metrics["macro_f1"])

    summary_payload["predictions"] = pred_rows
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B18 models on official test set and ensemble")
    parser.add_argument("--runs", "--checkpoint", dest="runs", type=Path, nargs="+", required=True, help="List of model run directories or checkpoints to evaluate")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/mocheg_manifest_strict/test.jsonl"))
    parser.add_argument("--retrieval", "--retrieval-raw", dest="retrieval_raw", type=Path, default=Path("outputs/retrieval_mocheg_dense_top50/test.jsonl"), help="Path to test retrieval jsonl")
    parser.add_argument("--retrieval-b18b", type=Path, default=Path("outputs/mocheg_b18b_filtered_retrieval/test.jsonl"))
    parser.add_argument("--retrieval-top3", type=Path, default=Path("outputs/mocheg_b18b_top3_retrieval/test.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/raw/mocheg_dataset/extracted/mocheg/test/Corpus2.csv"))
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tag", type=str, default="", help="Tag suffix for output prediction files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mocheg_b18_official_test"))
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if predictions already exist")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    corpus_path = resolve_corpus_path(args.corpus)
    logging.info("Reading test corpus: %s", corpus_path)
    corpus_docs = read_documents(corpus_path)
    logging.info("Loaded %d documents from test corpus", len(corpus_docs))

    from transformers import AutoTokenizer
    logging.info("Loading tokenizer from base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    answer_ids = label_token_ids(tokenizer)

    individual_summaries = []
    runs_predictions: list[dict[str, dict]] = []

    for model_dir in args.runs:
        retrieval_path = select_retrieval_for_model(
            model_dir=model_dir,
            retrieval_raw=args.retrieval_raw,
            retrieval_b18b=args.retrieval_b18b,
            retrieval_top3=args.retrieval_top3,
        )
        res = evaluate_single_run(
            model_dir=model_dir,
            base_model_name=args.base_model,
            manifest_path=args.manifest,
            retrieval_path=retrieval_path,
            corpus_docs=corpus_docs,
            tokenizer=tokenizer,
            answer_ids=answer_ids,
            device=device,
            batch_size=args.batch_size,
            tag=args.tag,
            force=args.force,
        )
        individual_summaries.append(res)
        p_dict = {str(r["id"]): r for r in res["predictions"]}
        runs_predictions.append(p_dict)

    # Ensembling across all evaluated models on the Test Set
    all_cids = sorted(list(runs_predictions[0].keys()))
    y_true = np.asarray([int(runs_predictions[0][cid]["label"]) for cid in all_cids])

    avg_probs, ensemble_preds = ensemble_predictions(runs_predictions, all_cids)
    ens_metrics = compute_metrics(y_true, ensemble_preds, avg_probs)

    # Bootstrap confidence interval for test ensemble Macro-F1
    boot = bootstrap_delta(y_true, ensemble_preds, ensemble_preds, iterations=args.bootstrap_iterations, seed=2026)
    ens_ci = boot.get("ci_95_percentile", [0.0, 0.0])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save final ensemble predictions JSONL
    test_ens_pred_path = args.output_dir / "test_ensemble_predictions.jsonl"
    with test_ens_pred_path.open("w", encoding="utf-8") as f:
        for i, cid in enumerate(all_cids):
            f.write(json.dumps({
                "id": cid,
                "label": int(y_true[i]),
                "prediction": int(ensemble_preds[i]),
                "probabilities": avg_probs[i].tolist(),
            }) + "\n")

    summary_payload = {
        "benchmark": "MOCHEG Official Test Set",
        "git_commit": git_commit(),
        "num_models": len(args.runs),
        "models": [str(p) for p in args.runs],
        "sample_count": len(all_cids),
        "individual_models": [
            {
                "name": p.name,
                "accuracy": s["accuracy"],
                "macro_f1": s["macro_f1"],
                "f1_supported": s["f1_supported"],
                "f1_refuted": s["f1_refuted"],
                "f1_nei": s["f1_nei"],
                "ece": s.get("ece"),
            }
            for p, s in zip(args.runs, individual_summaries)
        ],
        "official_test_ensemble": {
            "accuracy": ens_metrics["accuracy"],
            "macro_f1": ens_metrics["macro_f1"],
            "f1_supported": ens_metrics["f1_supported"],
            "f1_refuted": ens_metrics["f1_refuted"],
            "f1_nei": ens_metrics["f1_nei"],
            "ece": ens_metrics["ece"],
            "confusion_matrix": ens_metrics["confusion_matrix"],
            "bootstrap_ci_95": ens_ci,
        },
    }

    final_summary_path = args.output_dir / "test_summary.json"
    final_summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    # Generate Test Markdown Report with accurate protocol title
    n_samples = len(all_cids)
    if n_samples == 2434:
        protocol_title = "P1 Strict Test Set Benchmark Report (Strict Deduplicated Track)"
        protocol_name = "P1 strict test (leakage-controlled deduplicated track, n=2434)"
    elif n_samples == 2442:
        protocol_title = "P1 Official Test Set Benchmark Report (Raw Official Benchmark Track)"
        protocol_name = "P1 official test (raw official benchmark track, n=2442)"
    else:
        protocol_title = "MOCHEG Test Set Benchmark Report"
        protocol_name = f"Locked Test Evaluation (n={n_samples} claims)"

    lines = [
        f"# {protocol_title}",
        "",
        f"- **Protocol:** {protocol_name}",
        f"- **Git Commit:** `{git_commit()}`",
        f"- **Total Ensemble Models:** {len(args.runs)}",
        "",
        f"## 1. Test Set Ensemble Performance ({protocol_name})",
        "",
        "| Metric | Ensemble Score |",
        "|---|---:|",
        f"| **Macro-F1** | **{ens_metrics['macro_f1']:.5f}** |",
        f"| **Accuracy** | **{ens_metrics['accuracy']:.5f}** |",
        f"| F1 Supported | {ens_metrics['f1_supported']:.5f} |",
        f"| F1 Refuted | {ens_metrics['f1_refuted']:.5f} |",
        f"| F1 NEI | {ens_metrics['f1_nei']:.5f} |",
        f"| ECE (Calibration Error) | {ens_metrics['ece']:.5f} |",
        "",
        "## 2. Individual Member Model Scores on Test",
        "",
        "| Model Run | Macro-F1 | Accuracy | F1 Supp | F1 Ref | F1 NEI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summary_payload["individual_models"]:
        lines.append(
            f"| `{s['name']}` | {s['macro_f1']:.5f} | {s['accuracy']:.5f} | {s['f1_supported']:.5f} | {s['f1_refuted']:.5f} | {s['f1_nei']:.5f} |"
        )

    lines.extend([
        "",
        "## 3. Confusion Matrix",
        "```",
        json.dumps(ens_metrics["confusion_matrix"]),
        "```",
        "",
        f"Predictions saved to: `{test_ens_pred_path}`",
    ])

    report_text = "\n".join(lines) + "\n"
    markdown_path = args.output_dir / "test_summary.md"
    markdown_path.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)


if __name__ == "__main__":
    main()
