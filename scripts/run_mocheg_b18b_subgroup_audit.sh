#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="outputs/mocheg_b18b_subgroup_audit"
mkdir -p "$OUT"

DIRECT=()
for seed in 13 21 42 87 100; do
  DIRECT+=("outputs/mocheg_qwen3_lora_seed${seed}_v16/val_predictions.jsonl")
done
GROUNDED=()
for seed in 42 87 100; do
  GROUNDED+=("outputs/mocheg_b18a_full/candidate_seed${seed}/val_predictions.jsonl")
done

python -m scripts.analyze_mocheg_b18b_subgroups \
  --direct-runs "${DIRECT[@]}" \
  --grounded-runs "${GROUNDED[@]}" \
  --manifest data/processed/mocheg_manifest_strict/val.jsonl \
  --retrieval outputs/retrieval_mocheg_qwen3_reranked/val.jsonl \
  --tau 0.49 \
  --top-k 5 \
  --bootstrap-iterations 10000 \
  --seed 2026 \
  --output "$OUT/summary.json" \
  --markdown "$OUT/summary.md" \
  2>&1 | tee "$OUT/run.log"

echo "DONE: $OUT/summary.md"
