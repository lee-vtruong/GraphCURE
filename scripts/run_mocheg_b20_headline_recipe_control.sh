#!/usr/bin/env bash
# B20: direct-only causal control using the exact B18A headline recipe.
# This script intentionally stops after official-validation analysis.  It does
# not run any test inference and therefore cannot tune against test labels.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs/mocheg_b20_headline_recipe_control"
EXPLANATIONS="data/processed/mocheg_b18_explanations/train_full_explanations.jsonl"
SEEDS=(42 87 100)
mkdir -p "$OUT"

for required in \
  data/processed/mocheg_manifest_strict/train.jsonl \
  data/processed/mocheg_manifest_strict/val.jsonl \
  outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
  outputs/retrieval_mocheg_qwen3_reranked/val.jsonl \
  data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  data/raw/mocheg_dataset/extracted/mocheg/val/Corpus2.csv
do
  [[ -s "$required" ]] || { echo "ERROR: missing required input: $required" >&2; exit 1; }
done

start_seconds=$SECONDS
for seed in "${SEEDS[@]}"; do
  run="$OUT/direct_recipe_seed${seed}"
  if [[ -s "$run/summary.json" && -s "$run/val_predictions.jsonl" ]]; then
    echo "SKIP seed=$seed (complete)"
    continue
  fi
  mkdir -p "$run"
  echo "===== B20 direct-only headline-recipe control, seed $seed ====="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python -m scripts.train_mocheg_b18_explanation_verifier \
      --mode matched_control \
      --manifest data/processed/mocheg_manifest_strict/train.jsonl \
      --retrieval outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
      --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
      --val-manifest data/processed/mocheg_manifest_strict/val.jsonl \
      --val-retrieval outputs/retrieval_mocheg_qwen3_reranked/val.jsonl \
      --val-corpus data/raw/mocheg_dataset/extracted/mocheg/val/Corpus2.csv \
      --model Qwen/Qwen3-4B-Instruct-2507 \
      --output "$run" \
      --seed "$seed" \
      --epochs 3 \
      --batch-size 2 \
      --grad-accum 4 \
      --lr 2e-4 \
      --top-k 5 \
      --max-length 3072 \
      --max-evidence-chars 2200 \
      --device cuda \
      2>&1 | tee "$run/train.log"
done

CONTROL=()
RATIONALE=()
for seed in "${SEEDS[@]}"; do
  CONTROL+=("$OUT/direct_recipe_seed${seed}")
  RATIONALE+=("outputs/mocheg_b18a_full/candidate_seed${seed}")
done

python -m scripts.analyze_mocheg_b20_headline_recipe_control \
  --control-runs "${CONTROL[@]}" \
  --rationale-runs "${RATIONALE[@]}" \
  --iterations 10000 --seed 2026 \
  --output "$OUT/summary.json" \
  --markdown "$OUT/summary.md" \
  2>&1 | tee "$OUT/analysis.log"

echo "B20 validation-only control completed in $((SECONDS - start_seconds)) seconds"
