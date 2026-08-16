"""Verify alignment and provenance of cached MOCHEG protocol embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


SPLITS = ("train", "val", "test")


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--close-root", type=Path, required=True)
    parser.add_argument("--retrieved-root", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data_audit/mocheg_protocol_embeddings.json"),
    )
    args = parser.parse_args()

    failures: list[str] = []
    summaries: list[dict] = []
    expected = {
        "close": (["close"], True),
        "retrieved": (["open_retrieved"], True),
        "gold": (["open_gold_oracle"], False),
    }
    roots = {
        "close": args.close_root,
        "retrieved": args.retrieved_root,
        "gold": args.gold_root,
    }

    for split in SPLITS:
        payloads = {name: load(root / f"{split}.pt") for name, root in roots.items()}
        reference = payloads["close"]
        for name, payload in payloads.items():
            if payload["ids"] != reference["ids"]:
                failures.append(f"{split}:{name}: ID order mismatch")
            if not torch.equal(payload["labels"], reference["labels"]):
                failures.append(f"{split}:{name}: labels mismatch")
            if not torch.allclose(
                payload["claim_embeddings"], reference["claim_embeddings"], atol=1e-6
            ):
                failures.append(f"{split}:{name}: claim embeddings mismatch")
            protocols, should_skip = expected[name]
            metadata = payload.get("metadata", {})
            if metadata.get("protocols") != protocols:
                failures.append(
                    f"{split}:{name}: protocol metadata {metadata.get('protocols')} != {protocols}"
                )
            if bool(metadata.get("image_encoder_skipped")) != should_skip:
                failures.append(f"{split}:{name}: image skip metadata mismatch")
            if name in {"close", "retrieved"} and payload["image_mask"].any():
                failures.append(f"{split}:{name}: unexpected visible image evidence")

        summaries.append(
            {
                "split": split,
                "samples": len(reference["ids"]),
                "embedding_dim": int(reference["claim_embeddings"].shape[1]),
                "close_visible_images": int(payloads["close"]["image_mask"].sum()),
                "retrieved_visible_images": int(payloads["retrieved"]["image_mask"].sum()),
                "gold_visible_images": int(payloads["gold"]["image_mask"].sum()),
            }
        )

    report = {
        "status": "pass" if not failures else "fail",
        "splits": summaries,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
