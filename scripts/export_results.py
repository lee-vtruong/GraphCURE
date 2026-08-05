from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all server test metrics to a Markdown audit trail")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output", default="docs/SERVER_RESULTS.md")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fmt(value: object) -> str:
    return f"{float(value):.6f}" if value is not None else "NA"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    files = sorted(root.rglob("test_metrics.json"))
    lines = [
        "# Server results snapshot", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        f"Git commit: `{git_commit()}`", "",
        f"PyTorch: `{torch.__version__}`; CUDA runtime: `{torch.version.cuda}`", "",
        f"Discovered test metric files: {len(files)}", "",
        "| Result path | Architecture | Seed | Samples | Accuracy | Macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    loaded = []
    for path in files:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        loaded.append((path, metrics))
        provenance = metrics.get("provenance", {})
        lines.append(f"| `{path.as_posix()}` | {provenance.get('architecture', 'legacy')} | "
                     f"{provenance.get('seed', 'NA')} | {metrics.get('samples', 'NA')} | "
                     f"{fmt(metrics.get('accuracy'))} | {fmt(metrics.get('macro_f1'))} |")
    for path, metrics in loaded:
        lines.extend(["", f"## `{path.as_posix()}`", "",
                      f"- Confusion matrix: `{json.dumps(metrics.get('confusion_matrix'))}`"])
        if metrics.get("provenance"):
            lines.append(f"- Provenance: `{json.dumps(metrics['provenance'], sort_keys=True)}`")
        if "node_mix_gate_mean" in metrics:
            lines.append(f"- Node mix gate mean `[semantic, entity, temporal, contextual]`: "
                         f"`{json.dumps(metrics['node_mix_gate_mean'])}`")
            lines.append(f"- Node mix gate std: `{json.dumps(metrics.get('node_mix_gate_std'))}`")
        report = metrics.get("classification_report", {})
        for label in ("0", "1"):
            if label in report:
                row = report[label]
                lines.append(f"- Class {label}: precision={fmt(row.get('precision'))}, "
                             f"recall={fmt(row.get('recall'))}, F1={fmt(row.get('f1-score'))}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output} from {len(files)} result files")


if __name__ == "__main__":
    main()
