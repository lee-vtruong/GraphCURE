"""Encode MOCHEG manifests with cached SBERT text and CLIP image features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


def load_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/mocheg_embeddings_strict"))
    parser.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--nli-root", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)

    text_model = SentenceTransformer(args.text_model, device=str(device))
    processor = CLIPProcessor.from_pretrained(args.vision_model)
    vision_model = CLIPModel.from_pretrained(args.vision_model).to(device).eval()

    for split in args.splits:
        output = args.output_root / f"{split}.pt"
        if output.exists():
            print(f"skip existing: {output}")
            continue
        rows = load_rows(args.manifest_root / f"{split}.jsonl")
        nli = {}
        if args.nli_root:
            nli = {x["id"]: x for x in load_rows(args.nli_root / f"{split}.jsonl")}
        claims, evidences, texts = [], [], []
        for row in rows:
            evidence = " ".join(row.get("evidence_texts", []))
            claim = row.get("claim", "").strip()
            claims.append(claim)
            evidences.append(evidence)
            texts.append((claim + " [EVIDENCE] " + evidence).strip())
        claim_text = text_model.encode(claims, batch_size=args.batch_size, convert_to_numpy=True,
                                       normalize_embeddings=True, show_progress_bar=True)
        evidence_text = text_model.encode(evidences, batch_size=args.batch_size, convert_to_numpy=True,
                                          normalize_embeddings=True, show_progress_bar=True)
        text = text_model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=True)

        image_features, image_mask = [], []
        for start in tqdm(range(0, len(rows), args.batch_size), desc=f"{split} images"):
            batch, masks = [], []
            for row in rows[start:start + args.batch_size]:
                paths = row.get("image_paths", [])[: args.max_images]
                valid = []
                for path in paths:
                    try:
                        valid.append(Image.open(path).convert("RGB"))
                    except Exception:
                        continue
                masks.append(bool(valid))
                batch.append(valid[0] if valid else Image.new("RGB", (224, 224), "black"))
            inputs = processor(images=batch, return_tensors="pt").to(device)
            with torch.inference_mode():
                feats = vision_model.get_image_features(**inputs)
                # transformers 5 may return BaseModelOutputWithPooling while
                # older releases return the projected tensor directly.
                if not isinstance(feats, torch.Tensor):
                    feats = getattr(feats, "pooler_output", None)
                    if feats is None:
                        raise RuntimeError("CLIP image output has no pooler_output")
                feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            image_features.append(feats.cpu())
            image_mask.extend(masks)

        payload = {
            "ids": [row["id"] for row in rows],
            "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
            "text_embeddings": torch.from_numpy(np.asarray(text, dtype=np.float32)),
            "claim_embeddings": torch.from_numpy(np.asarray(claim_text, dtype=np.float32)),
            "evidence_embeddings": torch.from_numpy(np.asarray(evidence_text, dtype=np.float32)),
            "image_embeddings": torch.cat(image_features),
            "image_mask": torch.tensor(image_mask, dtype=torch.bool),
            "nli_metadata": torch.tensor([[nli.get(row["id"], {}).get(k, 0.0) for k in ("nli_support", "nli_contradiction", "nli_neutral", "nli_margin")] for row in rows], dtype=torch.float32),
            "metadata": {"text_model": args.text_model, "vision_model": args.vision_model,
                         "max_images": args.max_images, "device": str(device)},
        }
        torch.save(payload, output)
        print(f"saved {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
