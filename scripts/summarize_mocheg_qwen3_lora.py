"""Aggregate frozen validation runs for the Qwen3 LoRA verifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scripts.train_mocheg_cached_verifier import expected_calibration_error


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive validation temperature by minimizing multiclass NLL."""
    logits = torch.tensor(
        np.log(np.clip(probabilities, 1e-8, 1.0)), dtype=torch.float64
    )
    targets = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def probability_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    prediction = probabilities.argmax(axis=1)
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1.0)).mean()
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(labels, prediction).tolist(),
        "ece_10": expected_calibration_error(probabilities, labels),
        "nll": float(nll),
    }


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()), "std": float(array.std()),
        "min": float(array.min()), "max": float(array.max()),
        "values": array.tolist(),
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output-template",default="outputs/mocheg_qwen3_lora_seed{seed}_v16")
    p.add_argument("--seeds",nargs="+",type=int,default=[13,21,42,87,100])
    p.add_argument("--output",type=Path,default=Path("outputs/mocheg_qwen3_lora_validation_summary.json"))
    p.add_argument("--markdown",type=Path,default=Path("outputs/mocheg_qwen3_lora_validation_summary.md"))
    a=p.parse_args(); runs=[]; reference_ids=None; reference_labels=None; signatures=[]
    selected_settings=("model","top_k","max_length","max_evidence_chars","lora_r","lora_alpha","lora_dropout","epochs","batch_size","gradient_accumulation","learning_rate")
    for seed in a.seeds:
        root=Path(a.output_template.format(seed=seed)); summary_path=root/"summary.json"; prediction_path=root/"val_predictions.jsonl"
        if not summary_path.exists() or not prediction_path.exists():
            raise FileNotFoundError(f"missing completed validation artifacts for seed {seed}: {root}")
        summary=json.loads(summary_path.read_text())
        if summary.get("test_split_used") is not False:
            raise ValueError(f"seed {seed} is not validation-only")
        if not summary.get("accepted"):
            raise ValueError(f"seed {seed} did not pass its validation gate")
        rows=[json.loads(line) for line in prediction_path.read_text().splitlines() if line]
        ids=[row["id"] for row in rows]; labels=np.asarray([int(row["gold"]) for row in rows]); probabilities=np.asarray([row["probabilities"] for row in rows],dtype=float)
        if reference_ids is None: reference_ids,reference_labels=ids,labels
        elif ids!=reference_ids or not np.array_equal(labels,reference_labels):
            raise ValueError(f"seed {seed} predictions are not aligned")
        signature={key:summary["settings"][key] for key in selected_settings}
        signatures.append(signature)
        temperature=fit_temperature(probabilities,labels); calibrated=apply_temperature(probabilities,temperature)
        runs.append({"seed":seed,"root":str(root),"best_epoch":summary["best_epoch"],"raw":probability_metrics(probabilities,labels),"temperature":temperature,"calibrated":probability_metrics(calibrated,labels),"probabilities":probabilities,"calibrated_probabilities":calibrated})
    if any(value!=signatures[0] for value in signatures[1:]):
        raise ValueError("validation runs do not share one frozen configuration")
    raw_ensemble=np.mean([run["probabilities"] for run in runs],axis=0); calibrated_ensemble=np.mean([run["calibrated_probabilities"] for run in runs],axis=0)
    result={"split":"val","seeds":a.seeds,"configuration":signatures[0],"per_seed":[{key:value for key,value in run.items() if key not in ("probabilities","calibrated_probabilities")} for run in runs],"aggregate":{"accuracy":aggregate([run["raw"]["accuracy"] for run in runs]),"macro_f1":aggregate([run["raw"]["macro_f1"] for run in runs]),"ece_10":aggregate([run["raw"]["ece_10"] for run in runs])},"raw_ensemble":probability_metrics(raw_ensemble,reference_labels),"temperature_scaled_ensemble":probability_metrics(calibrated_ensemble,reference_labels),"stability_gate":{"all_runs_accepted":True,"mean_macro_f1_at_least_0_64":float(np.mean([run["raw"]["macro_f1"] for run in runs]))>=.64,"macro_f1_std_at_most_0_02":float(np.std([run["raw"]["macro_f1"] for run in runs]))<=.02},"test_split_used":False}
    result["stability_gate"]["passed"]=all(result["stability_gate"].values())
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Qwen3 LoRA MOCHEG validation summary","","Test split used: **no**","","| Seed | Accuracy | Macro-F1 | ECE-10 | Temperature |","|---:|---:|---:|---:|---:|"]
    for run in result["per_seed"]: lines.append(f"| {run['seed']} | {run['raw']['accuracy']:.4f} | {run['raw']['macro_f1']:.4f} | {run['raw']['ece_10']:.4f} | {run['temperature']:.4f} |")
    lines += ["","## Aggregate",f"- Accuracy: {result['aggregate']['accuracy']['mean']:.4f} +/- {result['aggregate']['accuracy']['std']:.4f}",f"- Macro-F1: {result['aggregate']['macro_f1']['mean']:.4f} +/- {result['aggregate']['macro_f1']['std']:.4f}",f"- Raw ensemble Macro-F1: {result['raw_ensemble']['macro_f1']:.4f}",f"- Temperature-scaled ensemble Macro-F1: {result['temperature_scaled_ensemble']['macro_f1']:.4f}",f"- Temperature-scaled ensemble ECE-10: {result['temperature_scaled_ensemble']['ece_10']:.4f}",f"- Stability gate: **{'pass' if result['stability_gate']['passed'] else 'fail'}**"]
    a.markdown.write_text("\n".join(lines)+"\n"); print("\n".join(lines))


if __name__=="__main__": main()
