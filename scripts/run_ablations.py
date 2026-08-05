from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched multi-seed architecture ablations")
    parser.add_argument("--config", default="configs/newsclippings_embeddings.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42, 87, 100])
    parser.add_argument("--architectures", nargs="+", default=[
        "linear", "mlp", "independent", "fully_connected", "typed_graph"
    ])
    parser.add_argument("--output-root", default="outputs/ablations/newsclippings_clip")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    rows: list[dict] = []
    for architecture in args.architectures:
        for seed in args.seeds:
            directory = root / architecture / f"seed_{seed}"
            test_file = directory / "test_metrics.json"
            if not (args.skip_existing and test_file.exists()):
                run([sys.executable, "-m", "scripts.train", "--config", args.config,
                     "--device", args.device, "--seed", str(seed),
                     "--architecture", architecture, "--output-dir", str(directory)])
                run([sys.executable, "-m", "scripts.evaluate", "--config", args.config,
                     "--checkpoint", str(directory / "best.pt"), "--device", args.device,
                     "--output", str(test_file)])
            metrics = json.loads(test_file.read_text(encoding="utf-8"))
            rows.append({"architecture": architecture, "seed": seed,
                         "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})

    summary = []
    for architecture in args.architectures:
        selected = [row for row in rows if row["architecture"] == architecture]
        entry = {"architecture": architecture, "runs": len(selected)}
        for metric in ("accuracy", "macro_f1"):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(entry)
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
