"""Create duplicate-safe, train-only folds for Phase B18 development.

This script operates strictly on the strict train split, using StratifiedGroupKFold
grouped by claim family (normalized claim text and source URLs) to eliminate
cross-fold leakage. Official validation and test splits are strictly forbidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.prepare_mocheg_sv_folds import claim_family, sha256
from scripts.run_mocheg_visual_retrieval import read_jsonl


DEFAULT_SEED = 2040


def build_b18_folds(rows: list[dict], folds: int = 5, seed: int = DEFAULT_SEED) -> list[dict]:
    labels = np.asarray([int(row["label"]) for row in rows])
    groups = np.asarray([claim_family(row) for row in rows])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_ids = np.asarray([str(row["id"]) for row in rows])
    result = []
    for fold, (fit, held) in enumerate(splitter.split(all_ids, labels, groups)):
        fit_groups = set(groups[fit])
        held_groups = set(groups[held])
        if fit_groups & held_groups:
            raise RuntimeError(f"Claim family leakage detected in fold {fold}")
        result.append({
            "fold": fold,
            "train_ids": all_ids[fit].tolist(),
            "val_ids": all_ids[held].tolist(),
            "train_label_counts": dict(Counter(map(str, labels[fit].tolist()))),
            "val_label_counts": dict(Counter(map(str, labels[held].tolist()))),
            "train_families": len(fit_groups),
            "val_families": len(held_groups),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen duplicate-safe folds for Phase B18")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/mocheg_manifest_strict/train.jsonl"),
        help="Strict train split manifest",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mocheg_b18_folds.json"),
    )
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Manifest contains duplicate IDs")

    payload = {
        "protocol": "train_only_duplicate_safe_stratified_group_cv",
        "phase": "B18",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "samples": len(rows),
        "fold_count": args.folds,
        "seed": args.seed,
        "test_split_used": False,
        "validation_split_used": False,
        "folds": build_b18_folds(rows, args.folds, args.seed),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved B18 folds to {args.output} (samples={len(rows)}, folds={args.folds}, seed={args.seed})")


if __name__ == "__main__":
    main()
