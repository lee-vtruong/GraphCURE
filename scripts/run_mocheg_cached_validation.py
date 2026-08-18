"""Run and summarize validation-only cached-verifier seeds.

This runner deliberately has no test-evaluation option.  It is the model
selection boundary used before the MOCHEG test split is unlocked.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


METRICS = (
    "accuracy",
    "macro_f1",
    "evidence_selection_hit_at_1",
    "ece_10",
)


def summarize_validation(paths: dict[int, Path]) -> dict:
    """Aggregate matched validation metrics and verify cache provenance."""
    rows = []
    cache_signatures = set()
    for seed, path in sorted(paths.items()):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metadata = metrics.get("provenance", {}).get("cache_metadata", {})
        cache_signatures.add((
            metadata.get("manifest_sha256"),
            metadata.get("retrieval_sha256"),
            metadata.get("encoder"),
            metadata.get("top_k"),
        ))
        row = {"seed": seed, **{key: float(metrics[key]) for key in METRICS}}
        row["best_val_macro_f1"] = float(metrics["best_val_macro_f1"])
        row["retrieval_gold_coverage"] = float(metrics["retrieval_gold_coverage"])
        rows.append(row)
    if not rows:
        raise ValueError("no validation metric files found")
    if len(cache_signatures) != 1:
        raise ValueError("validation runs do not share identical cache provenance")

    aggregate = {}
    for key in METRICS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }
    f1 = aggregate["macro_f1"]
    return {
        "split": "val",
        "runs": len(rows),
        "seeds": [row["seed"] for row in rows],
        "per_seed": rows,
        "aggregate": aggregate,
        "stability_gate": {
            "mean_macro_f1_at_least_0_50": f1["mean"] >= 0.50,
            "macro_f1_std_at_most_0_02": f1["std"] <= 0.02,
            "passed": f1["mean"] >= 0.50 and f1["std"] <= 0.02,
        },
        "cache_signature": {
            "manifest_sha256": next(iter(cache_signatures))[0],
            "retrieval_sha256": next(iter(cache_signatures))[1],
            "encoder": next(iter(cache_signatures))[2],
            "top_k": next(iter(cache_signatures))[3],
        },
        "test_split_used": False,
    }


def markdown_summary(summary: dict) -> str:
    lines = [
        "# MOCHEG cached verifier validation summary",
        "",
        "Test split used: **no**",
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
        f"- Stability gate: **{'pass' if summary['stability_gate']['passed'] else 'fail'}**",
        "",
    ))
    return "\n".join(lines)


def train_seed(args: argparse.Namespace, seed: int, output: Path) -> None:
    command = [
        sys.executable, "-m", "scripts.train_mocheg_cached_verifier",
        "--cache-root", str(args.cache_root),
        "--output", str(output),
        "--hidden-dim", str(args.hidden_dim),
        "--dropout", str(args.dropout),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--relevance-weight", str(args.relevance_weight),
        "--stance-weight", str(args.stance_weight),
        "--sufficiency-weight", str(args.sufficiency_weight),
        "--device", args.device,
        "--seed", str(seed),
    ]
    log_path = args.output_root / f"{args.run_prefix}{seed}.log"
    print(f"\n=== seed {seed}: {' '.join(command)} ===", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched cached-verifier seeds on validation only"
    )
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/mocheg_reasoning_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-prefix", default="mocheg_cached_verifier_seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42, 87, 100])
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--relevance-weight", type=float, default=0.25)
    parser.add_argument("--stance-weight", type=float, default=0.15)
    parser.add_argument("--sufficiency-weight", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    metric_paths: dict[int, Path] = {}
    for seed in args.seeds:
        output = args.output_root / f"{args.run_prefix}{seed}"
        metrics = output / "val_metrics.json"
        if not args.summary_only and not (args.skip_existing and metrics.exists()):
            train_seed(args, seed, output)
        if metrics.exists():
            metric_paths[seed] = metrics
        else:
            raise FileNotFoundError(f"missing validation result: {metrics}")

    summary = summarize_validation(metric_paths)
    json_path = args.output_root / "mocheg_cached_verifier_summary_val.json"
    md_path = args.output_root / "mocheg_cached_verifier_summary_val.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_summary(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
