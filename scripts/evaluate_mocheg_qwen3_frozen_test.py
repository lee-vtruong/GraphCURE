"""One-shot frozen test evaluation for the accepted Qwen3 LoRA ensemble."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.summarize_mocheg_qwen3_lora import (
    aggregate,
    apply_temperature,
    probability_metrics,
)
from scripts.train_mocheg_qwen3_lora_verifier import (
    QwenVerifierDataset,
    evaluate,
    label_token_ids,
    make_collate,
)


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git","rev-parse","HEAD"],text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError,subprocess.CalledProcessError):
        return "unknown"


def read_predictions(path: Path) -> tuple[list[str],np.ndarray,np.ndarray]:
    rows=[json.loads(line) for line in path.read_text().splitlines() if line]
    return (
        [row["id"] for row in rows],
        np.asarray([int(row["gold"]) for row in rows]),
        np.asarray([row["probabilities"] for row in rows],dtype=float),
    )


def bootstrap_ci(probabilities: np.ndarray, labels: np.ndarray,
                 iterations: int, seed: int) -> dict:
    rng=np.random.default_rng(seed); accuracy=[]; macro_f1=[]; size=len(labels)
    for _ in range(iterations):
        indices=rng.integers(0,size,size=size)
        metrics=probability_metrics(probabilities[indices],labels[indices])
        accuracy.append(metrics["accuracy"]); macro_f1.append(metrics["macro_f1"])
    return {
        "iterations":iterations,"seed":seed,
        "accuracy_ci_95_percentile":np.percentile(accuracy,[2.5,97.5]).tolist(),
        "macro_f1_ci_95_percentile":np.percentile(macro_f1,[2.5,97.5]).tolist(),
    }


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p=argparse.ArgumentParser()
    p.add_argument("--validation-summary",type=Path,default=Path("outputs/mocheg_qwen3_lora_validation_summary.json"))
    p.add_argument("--run-template",default="outputs/mocheg_qwen3_lora_seed{seed}_v16")
    p.add_argument("--manifest",type=Path,default=Path("data/processed/mocheg_manifest_strict/test.jsonl")); p.add_argument("--retrieval",type=Path,default=Path("outputs/retrieval_mocheg_qwen3_reranked/test.jsonl")); p.add_argument("--corpus",type=Path,default=Path("data/raw/mocheg_dataset/extracted/mocheg/test/Corpus2.csv"))
    p.add_argument("--output-root",type=Path,default=Path("outputs/mocheg_qwen3_lora_frozen_test")); p.add_argument("--batch-size",type=int,default=2); p.add_argument("--num-workers",type=int,default=2); p.add_argument("--bootstrap-iterations",type=int,default=5000); p.add_argument("--device",default="cuda")
    a=p.parse_args(); validation=json.loads(a.validation_summary.read_text())
    if validation.get("split")!="val" or validation.get("test_split_used") is not False or not validation.get("stability_gate",{}).get("passed"):
        p.error("validation summary is not an accepted, test-locked freeze artifact")
    configuration=validation["configuration"]; seeds=[int(value) for value in validation["seeds"]]
    a.output_root.mkdir(parents=True,exist_ok=True); freeze_path=a.output_root/"freeze_manifest.json"
    summary_path=a.output_root/"summary.json"
    if summary_path.exists():
        raise FileExistsError(f"final test summary already exists: {summary_path}")
    freeze={"validation_summary":str(a.validation_summary),"validation_summary_sha256":sha256(a.validation_summary),"configuration":configuration,"seeds":seeds,"temperatures":{str(run["seed"]):run["temperature"] for run in validation["per_seed"]},"manifest":str(a.manifest),"manifest_sha256":sha256(a.manifest),"retrieval":str(a.retrieval),"retrieval_sha256":sha256(a.retrieval),"corpus":str(a.corpus),"corpus_sha256":sha256(a.corpus),"primary_metric":"raw five-seed ensemble Macro-F1","secondary_metric":"validation-temperature-scaled ensemble calibration","git_commit":git_commit()}
    if freeze_path.exists():
        existing_freeze=json.loads(freeze_path.read_text())
        comparable_freeze={**freeze,"git_commit":existing_freeze.get("git_commit")}
        if existing_freeze!=comparable_freeze:
            raise ValueError("existing frozen-test manifest does not match this invocation")
    else:
        freeze_path.write_text(json.dumps(freeze,indent=2)+"\n")
    device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    dataset=QwenVerifierDataset(a.manifest,a.retrieval,a.corpus,int(configuration["top_k"]),int(configuration["max_evidence_chars"]),False,0)
    all_runs=[]; reference_ids=None; reference_labels=None
    for seed in seeds:
        prediction_path=a.output_root/f"seed_{seed}_predictions.jsonl"; metrics_path=a.output_root/f"seed_{seed}_metrics.json"
        if prediction_path.exists() and metrics_path.exists():
            ids,labels,probabilities=read_predictions(prediction_path); metrics=json.loads(metrics_path.read_text())
        else:
            root=Path(a.run_template.format(seed=seed)); adapter=root/"best_adapter"; summary=json.loads((root/"summary.json").read_text())
            if not summary.get("accepted") or summary["settings"]["model"]!=configuration["model"]:
                raise ValueError(f"seed {seed} adapter is not an accepted frozen run")
            tokenizer=AutoTokenizer.from_pretrained(adapter); tokenizer.padding_side="right"
            if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
            answer_ids=label_token_ids(tokenizer)
            loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,collate_fn=make_collate(tokenizer,int(configuration["max_length"]),False),pin_memory=device.type=="cuda")
            if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
            base=AutoModelForCausalLM.from_pretrained(configuration["model"],torch_dtype=torch.bfloat16 if device.type=="cuda" else torch.float32,attn_implementation="sdpa").to(device); base.config.use_cache=False
            model=PeftModel.from_pretrained(base,adapter).to(device); started=time.perf_counter(); metrics,rows=evaluate(model,loader,answer_ids,device); elapsed=time.perf_counter()-started
            adapter_weights=adapter/"adapter_model.safetensors"
            if not adapter_weights.exists(): adapter_weights=adapter/"adapter_model.bin"
            metrics.update({"seed":seed,"elapsed_seconds":elapsed,"milliseconds_per_sample":1000*elapsed/len(dataset),"peak_cuda_memory_mib":float(torch.cuda.max_memory_allocated(device)/2**20) if device.type=="cuda" else 0.0,"adapter":str(adapter),"adapter_config_sha256":sha256(adapter/"adapter_config.json"),"adapter_weights_sha256":sha256(adapter_weights)})
            prediction_path.write_text("\n".join(json.dumps(row) for row in rows)+"\n"); metrics_path.write_text(json.dumps(metrics,indent=2)+"\n"); ids,labels,probabilities=read_predictions(prediction_path)
            del model,base; gc.collect()
            if device.type=="cuda": torch.cuda.empty_cache()
        if len(ids)!=len(dataset): raise ValueError(f"incomplete test predictions for seed {seed}")
        if reference_ids is None: reference_ids,reference_labels=ids,labels
        elif ids!=reference_ids or not np.array_equal(labels,reference_labels): raise ValueError(f"unaligned test predictions for seed {seed}")
        temperature=float(freeze["temperatures"][str(seed)])
        all_runs.append({"seed":seed,"metrics":metrics,"temperature":temperature,"probabilities":probabilities,"calibrated_probabilities":apply_temperature(probabilities,temperature)})
    raw=np.mean([run["probabilities"] for run in all_runs],axis=0); calibrated=np.mean([run["calibrated_probabilities"] for run in all_runs],axis=0)
    result={"split":"test","protocol":"frozen Qwen3 LoRA five-seed ensemble","freeze_manifest":freeze,"per_seed":[{"seed":run["seed"],"temperature":run["temperature"],**run["metrics"]} for run in all_runs],"aggregate":{"accuracy":aggregate([run["metrics"]["accuracy"] for run in all_runs]),"macro_f1":aggregate([run["metrics"]["macro_f1"] for run in all_runs])},"raw_ensemble":probability_metrics(raw,reference_labels),"temperature_scaled_ensemble":probability_metrics(calibrated,reference_labels),"raw_ensemble_bootstrap":bootstrap_ci(raw,reference_labels,a.bootstrap_iterations,2026),"test_split_used":True,"test_fitted_parameters":0}
    markdown_path=a.output_root/"summary.md"
    summary_path.write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Frozen Qwen3 LoRA MOCHEG test summary","","Primary: raw five-seed ensemble Macro-F1. No parameter was fitted on test.","","| Seed | Accuracy | Macro-F1 | ECE-10 | ms/sample |","|---:|---:|---:|---:|---:|"]
    for run in result["per_seed"]: lines.append(f"| {run['seed']} | {run['accuracy']:.4f} | {run['macro_f1']:.4f} | {run['ece_10']:.4f} | {run['milliseconds_per_sample']:.2f} |")
    lines += ["","## Aggregate",f"- Accuracy: {result['aggregate']['accuracy']['mean']:.4f} +/- {result['aggregate']['accuracy']['std']:.4f}",f"- Macro-F1: {result['aggregate']['macro_f1']['mean']:.4f} +/- {result['aggregate']['macro_f1']['std']:.4f}",f"- **Primary raw ensemble Accuracy: {result['raw_ensemble']['accuracy']:.4f}**",f"- **Primary raw ensemble Macro-F1: {result['raw_ensemble']['macro_f1']:.4f}**",f"- Raw ensemble Macro-F1 95% bootstrap CI: {result['raw_ensemble_bootstrap']['macro_f1_ci_95_percentile']}",f"- Calibrated ensemble Macro-F1: {result['temperature_scaled_ensemble']['macro_f1']:.4f}",f"- Calibrated ensemble ECE-10: {result['temperature_scaled_ensemble']['ece_10']:.4f}","- Parameters fit on test: **0**"]
    markdown_path.write_text("\n".join(lines)+"\n"); print("\n".join(lines))


if __name__=="__main__": main()
