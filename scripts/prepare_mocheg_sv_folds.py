"""Create duplicate-safe, train-only folds for GraphCURE-SV development.

The official validation and test sets are deliberately not read.  All model
selection for the post-test follow-up therefore happens inside the original
strict training split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.run_mocheg_visual_retrieval import read_jsonl


def normalized_claim(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def claim_family(row: dict) -> str:
    """Group exact/formatting claim duplicates and repeated source pages."""
    claim = normalized_claim(row.get("claim", ""))
    url = row.get("snopes_url", "").strip().casefold().rstrip("/")
    key = url or claim or str(row["id"])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_folds(rows: list[dict], folds: int, seed: int) -> list[dict]:
    labels = np.asarray([int(row["label"]) for row in rows])
    groups = np.asarray([claim_family(row) for row in rows])
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    result = []
    all_ids = np.asarray([row["id"] for row in rows])
    for fold, (fit, held) in enumerate(splitter.split(all_ids, labels, groups)):
        fit_groups = set(groups[fit])
        held_groups = set(groups[held])
        if fit_groups & held_groups:
            raise RuntimeError(f"claim-family leakage in fold {fold}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/mocheg_manifest_strict/train.jsonl"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mocheg_sv_folds.json"),
    )
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("manifest contains duplicate IDs")
    payload = {
        "protocol": "train_only_duplicate_safe_stratified_group_cv",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "samples": len(rows),
        "fold_count": args.folds,
        "seed": args.seed,
        "test_split_used": False,
        "validation_split_used": False,
        "folds": build_folds(rows, args.folds, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "saved": str(args.output), "samples": len(rows),
        "folds": args.folds, "protocol": payload["protocol"],
    }, indent=2))


if __name__ == "__main__":
    main()
