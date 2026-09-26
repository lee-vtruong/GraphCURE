#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="outputs/mocheg_b18b_component_ablation"
mkdir -p "$OUT"

python -m scripts.analyze_mocheg_b18b_component_ablations \
  --direct-runs \
    outputs/mocheg_qwen3_lora_seed13/val_predictions.jsonl \
    outputs/mocheg_qwen3_lora_seed21/val_predictions.jsonl \
    outputs/mocheg_qwen3_lora_seed42/val_predictions.jsonl \
    outputs/mocheg_qwen3_lora_seed87/val_predictions.jsonl \
    outputs/mocheg_qwen3_lora_seed100/val_predictions.jsonl \
  --grounded-runs \
    outputs/mocheg_b18a_full/candidate_seed42/val_predictions.jsonl \
    outputs/mocheg_b18a_full/candidate_seed87/val_predictions.jsonl \
    outputs/mocheg_b18a_full/candidate_seed100/val_predictions.jsonl \
  --tau 0.49 \
  --output "$OUT/summary.json" \
  --markdown "$OUT/summary.md" \
  2>&1 | tee "$OUT/run.log"

echo "Wrote $OUT/summary.json and $OUT/summary.md"
