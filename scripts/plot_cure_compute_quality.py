"""Render the descriptive CURE raw-P1 quality--latency Pareto figure.

Effectiveness values are the frozen raw-P1 point estimates reported in the
manuscript; latency values are the RTX 5090 sequential-member benchmark on
256 validation claims. The plot is descriptive because the two quantities are
measured on different, predeclared evaluation artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROWS = (
    ("Direct\nensemble", 320.72, .54531, "#4C78A8", "o"),
    ("Rationale\nensemble", 193.17, .53986, "#F58518", "o"),
    ("CURE AND", 513.88, .55470, "#54A24B", "D"),
    ("CURE-Ensemble", 513.88, .55507, "#B279A2", "D"),
    ("CURE-Student", 64.47, .54940, "#E45756", "s"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axis = plt.subplots(figsize=(5.6, 3.35))
    for name, latency, score, color, marker in ROWS:
        axis.scatter(latency, score, s=58, marker=marker, color=color, edgecolor="white", linewidth=.7, zorder=3)
        offset = (5, 5) if name != "CURE-Ensemble" else (5, -14)
        axis.annotate(name, (latency, score), xytext=offset, textcoords="offset points", fontsize=8)
    axis.set_xlabel("Sequential inference latency (ms / claim)")
    axis.set_ylabel("Raw official P1 Macro-F1")
    axis.set_xlim(35, 550)
    axis.set_ylim(.5375, .5575)
    axis.grid(alpha=.25, linewidth=.6, zorder=0)
    figure.tight_layout(pad=.4)
    figure.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
