"""Validate and freeze a completed C1 retrieval snapshot without unlocking test."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from graphcure.open_web import load_jsonl, sha256_file


def tree_digest(root: Path, excluded: set[str]) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
        files += 1
    return digest.hexdigest(), files


def tree_contains_secret(root: Path, secret: str) -> bool:
    if not secret:
        return False
    for pattern in ("*.json", "*.jsonl"):
        for path in root.rglob(pattern):
            try:
                if secret in path.read_text(encoding="utf-8"):
                    return True
            except UnicodeDecodeError:
                continue
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-name", default="c1_freeze_manifest.json")
    parser.add_argument("--api-key-env", default="SERPER_API_KEY")
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a file name inside the snapshot root")

    snapshot_manifest_path = args.snapshot_root / "snapshot_manifest.json"
    summary_path = args.snapshot_root / "summary.json"
    rows_path = args.snapshot_root / f"{args.split}.jsonl"
    snapshot = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_summary = summary["splits"][args.split]
    rows = load_jsonl(rows_path)
    ids = [row["id"] for row in rows]

    failures = []
    if snapshot.get("protocol") != "P2_open_web":
        failures.append("snapshot protocol is not P2_open_web")
    if not split_summary.get("complete"):
        failures.append("split summary is incomplete")
    if split_summary.get("test_split_used"):
        failures.append("C1 development freeze cannot contain test")
    if split_summary.get("gold_evidence_used"):
        failures.append("gold evidence flag is true")
    if len(ids) != len(set(ids)):
        failures.append("duplicate claim IDs")
    if len(rows) != int(split_summary.get("completed_claims", -1)):
        failures.append("row count disagrees with summary")
    if any(row.get("snapshot_signature") != snapshot.get("signature") for row in rows):
        failures.append("row snapshot signature mismatch")
    api_key = os.environ.get(args.api_key_env, "")
    if tree_contains_secret(args.snapshot_root, api_key):
        failures.append("API key found in snapshot root")

    output_relative = Path(args.output_name).as_posix()
    digest, artifact_files = tree_digest(args.snapshot_root, {output_relative})
    freeze = {
        "status": "c1_retrieval_snapshot_frozen",
        "protocol": "P2_open_web_C1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "claims": len(rows),
        "snapshot_signature": snapshot.get("signature"),
        "snapshot_manifest_sha256": sha256_file(snapshot_manifest_path),
        "summary_sha256": sha256_file(summary_path),
        "rows_sha256": sha256_file(rows_path),
        "artifact_tree_sha256": digest,
        "artifact_files": artifact_files,
        "search_calls": split_summary.get("search_calls"),
        "mean_queries_per_claim": split_summary.get("mean_queries_per_claim"),
        "usable_evidence_rate": split_summary.get("usable_evidence_rate"),
        "median_domains_per_claim": split_summary.get("median_domains_per_claim"),
        "gold_evidence_used": False,
        "test_split_used": False,
        "unlocks_test": False,
        "failures": failures,
    }
    output = args.snapshot_root / args.output_name
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(freeze, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
