"""One-shot multi-seed test evaluation for the frozen GraphCURE-R2V model."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


TEST_METRICS = (
    "accuracy",
    "macro_f1",
    "evidence_selection_hit_at_1",
    "ece_10",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_freeze(freeze: dict, validation: dict) -> list[int]:
    """Reject test access unless the preregistered validation gate passed."""
    if freeze.get("status") != "frozen_after_validation":
        raise ValueError("experiment is not frozen after validation")
    if validation.get("split") != "val" or validation.get("test_split_used") is not False:
        raise ValueError("invalid validation-only summary")
    if not validation.get("stability_gate", {}).get("passed"):
        raise ValueError("validation stability gate did not pass")
    seeds = [int(seed) for seed in freeze.get("frozen_seeds", [])]
    if seeds != [int(seed) for seed in validation.get("seeds", [])]:
        raise ValueError("frozen seeds differ from the validation seed set")
    observed = validation["aggregate"]["macro_f1"]
    gate = freeze["validation_gate"]
    if observed["mean"] < float(gate["minimum_mean_macro_f1"]):
        raise ValueError("mean validation Macro-F1 is below the frozen gate")
    if observed["std"] > float(gate["maximum_macro_f1_std"]):
        raise ValueError("validation Macro-F1 is too unstable")
    return seeds


def validate_test_cache(cache_path: Path, freeze: dict) -> dict:
    import torch

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    metadata = cache.get("metadata", {})
    expected = freeze["cache"]
    for key in ("encoder", "top_k", "max_length", "embedding_dim"):
        if metadata.get(key) != expected.get(key):
            raise ValueError(
                f"test cache mismatch for {key}: "
                f"{metadata.get(key)!r} != {expected.get(key)!r}"
            )
    return metadata


def evaluate_seed(args: argparse.Namespace, seed: int, run: Path) -> None:
    checkpoint = run / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing frozen checkpoint: {checkpoint}")
    command = [
        sys.executable, "-m", "scripts.train_mocheg_cached_verifier",
        "--cache-root", str(args.cache_root),
        "--output", str(run),
        "--checkpoint", str(checkpoint),
        "--evaluate-test",
        "--batch-size", str(args.batch_size),
        "--device", args.device,
    ]
    print(f"\n=== frozen test seed {seed} ===", flush=True)
    subprocess.run(command, check=True)


def aggregate_test(paths: dict[int, Path], freeze: dict, freeze_sha256: str,
                   test_cache_metadata: dict) -> dict:
    rows = []
    for seed, path in sorted(paths.items()):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "seed": seed,
            **{key: float(metrics[key]) for key in TEST_METRICS},
            "retrieval_gold_coverage": float(metrics["retrieval_gold_coverage"]),
            "confusion_matrix": metrics["confusion_matrix"],
        })
    aggregate = {}
    for key in TEST_METRICS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }
    target = freeze["comparison_target"]
    return {
        "split": "test",
        "protocol": freeze["protocol"],
        "runs": len(rows),
        "seeds": [row["seed"] for row in rows],
        "per_seed": rows,
        "aggregate": aggregate,
        "comparison_target": target,
        "delta_vs_target": {
            "accuracy": aggregate["accuracy"]["mean"] - float(target["accuracy"]),
            "macro_f1": aggregate["macro_f1"]["mean"] - float(target["macro_f1"]),
        },
        "freeze_sha256": freeze_sha256,
        "test_cache_metadata": test_cache_metadata,
    }


def markdown_summary(summary: dict) -> str:
    lines = [
        "# Frozen GraphCURE-R2V MOCHEG test summary",
        "",
        f"Protocol: `{summary['protocol']}`",
        "",
        "| Seed | Accuracy | Macro-F1 | Select@1 | ECE-10 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} "
            f"| {row['evidence_selection_hit_at_1']:.4f} | {row['ece_10']:.4f} |"
        )
    aggregate = summary["aggregate"]
    delta = summary["delta_vs_target"]
    lines.extend((
        "",
        "## Aggregate",
        "",
        f"- Accuracy: {aggregate['accuracy']['mean']:.4f} +/- "
        f"{aggregate['accuracy']['std']:.4f}",
        f"- Macro-F1: {aggregate['macro_f1']['mean']:.4f} +/- "
        f"{aggregate['macro_f1']['std']:.4f}",
        f"- Select@1: {aggregate['evidence_selection_hit_at_1']['mean']:.4f} +/- "
        f"{aggregate['evidence_selection_hit_at_1']['std']:.4f}",
        f"- ECE-10: {aggregate['ece_10']['mean']:.4f} +/- "
        f"{aggregate['ece_10']['std']:.4f}",
        f"- Delta vs frozen comparison target: Accuracy {delta['accuracy']:+.4f}, "
        f"Macro-F1 {delta['macro_f1']:+.4f}",
        "",
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all frozen R2V seeds once on MOCHEG test"
    )
    parser.add_argument("--freeze", type=Path,
                        default=Path("configs/mocheg_r2v_frozen.json"))
    parser.add_argument("--validation-summary", type=Path,
                        default=Path("outputs/mocheg_cached_verifier_summary_val.json"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/mocheg_reasoning_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-prefix", default="mocheg_cached_verifier_seed")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    validation = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    seeds = validate_freeze(freeze, validation)
    test_cache_metadata = validate_test_cache(args.cache_root / "test.pt", freeze)

    paths = {}
    for seed in seeds:
        run = args.output_root / f"{args.run_prefix}{seed}"
        metrics = run / "test_metrics.json"
        if not (args.skip_existing and metrics.exists()):
            evaluate_seed(args, seed, run)
        if not metrics.exists():
            raise FileNotFoundError(f"missing test result: {metrics}")
        paths[seed] = metrics

    summary = aggregate_test(paths, freeze, file_sha256(args.freeze), test_cache_metadata)
    json_path = args.output_root / "mocheg_cached_verifier_summary_test.json"
    md_path = args.output_root / "mocheg_cached_verifier_summary_test.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_summary(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
