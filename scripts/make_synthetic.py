from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/synthetic")
    parser.add_argument("--n", type=int, default=96)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    rows = []
    for i in range(args.n):
        constraints = [rng.randrange(3) for _ in range(4)]
        label = 0 if all(v == 0 for v in constraints) else 1 + (constraints.count(1) > 1)
        text = [rng.gauss(label, 1) for _ in range(32)]
        image = [rng.gauss(constraints[0], 1) for _ in range(32)]
        changed = [False, False, i % 2 == 0, i % 2 == 0]
        cf_text = [v + (0.8 if any(changed) else 0.0) for v in text]
        rows.append(
            {
                "id": f"synthetic-{i}",
                "label": int(label),
                "text_embedding": text,
                "image_embedding": image,
                "metadata": [0.0] * 16,
                "constraint_labels": constraints,
                "conflict_labels": [float(rng.randrange(2)) for _ in range(5)],
                "counterfactual": {
                    "text_embedding": cf_text,
                    "changed_mask": changed,
                },
            }
        )
    for name, subset in (
        ("train", rows[:64]),
        ("val", rows[64:80]),
        ("test", rows[80:]),
    ):
        with (root / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

