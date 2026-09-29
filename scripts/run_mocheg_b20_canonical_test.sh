#!/usr/bin/env bash
# Full-train raw/strict P1 confirmation of B20's headline-recipe direct control.
# All training/checkpoint decisions use official validation only.  Test labels
# are read only once for frozen evaluation after all three runs are complete.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs/mocheg_b20_canonical_test"
TRAIN_MANIFEST="data/processed/mocheg_manifest_strict/train.jsonl"
TRAIN_RETRIEVAL="outputs/retrieval_mocheg_qwen3_reranked/train.jsonl"
VAL_MANIFEST="data/processed/mocheg_manifest_strict/val.jsonl"
VAL_RETRIEVAL="outputs/retrieval_mocheg_qwen3_reranked/val.jsonl"
RAW_MANIFEST="data/processed/mocheg_manifest/test.jsonl"
RAW_RETRIEVAL="outputs/retrieval_mocheg_qwen3_reranked_official/test.jsonl"
STRICT_MANIFEST="data/processed/mocheg_manifest_strict/test.jsonl"
STRICT_RETRIEVAL="outputs/retrieval_mocheg_qwen3_reranked/test.jsonl"
CORPUS_ROOT="data/raw/mocheg_dataset/extracted/mocheg"
SEEDS=(42 87 100)

for required in "$TRAIN_MANIFEST" "$TRAIN_RETRIEVAL" "$VAL_MANIFEST" "$VAL_RETRIEVAL" \
  "$RAW_MANIFEST" "$RAW_RETRIEVAL" "$STRICT_MANIFEST" "$STRICT_RETRIEVAL" \
  "$CORPUS_ROOT/train/Corpus2.csv" "$CORPUS_ROOT/val/Corpus2.csv" "$CORPUS_ROOT/test/Corpus2.csv"; do
  [[ -s "$required" ]] || { echo "ERROR: missing required input: $required" >&2; exit 1; }
done

mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DIRECT_RUNS=()
RAW_DIRECT=()
STRICT_DIRECT=()
RAW_RATIONALE=()
STRICT_RATIONALE=()
for seed in "${SEEDS[@]}"; do
  run="$OUT/direct_recipe_seed${seed}"
  DIRECT_RUNS+=("$run")
  RAW_DIRECT+=("$run/test_predictions_b20_canonical_raw.jsonl")
  STRICT_DIRECT+=("$run/test_predictions_b20_canonical_strict.jsonl")
  RAW_RATIONALE+=("outputs/mocheg_b18a_full/candidate_seed${seed}/test_predictions_canonical_k5.jsonl")
  STRICT_RATIONALE+=("outputs/mocheg_b18a_full/candidate_seed${seed}/test_predictions_canonical_k5_strict.jsonl")

  if [[ ! -s "$run/summary.json" || ! -s "$run/val_predictions.jsonl" ]]; then
    mkdir -p "$run"
    echo "===== TRAIN B20 FULL CONTROL seed=$seed ====="
    python -m scripts.train_mocheg_b18_explanation_verifier \
      --mode matched_control \
      --manifest "$TRAIN_MANIFEST" --retrieval "$TRAIN_RETRIEVAL" --corpus "$CORPUS_ROOT/train/Corpus2.csv" \
      --val-manifest "$VAL_MANIFEST" --val-retrieval "$VAL_RETRIEVAL" --val-corpus "$CORPUS_ROOT/val/Corpus2.csv" \
      --model Qwen/Qwen3-4B-Instruct-2507 --output "$run" --seed "$seed" \
      --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4 --top-k 5 \
      --max-length 3072 --max-evidence-chars 2200 --device cuda \
      2>&1 | tee "$run/train.log"
  else
    echo "SKIP TRAIN seed=$seed (complete)"
  fi
done

for seed in "${SEEDS[@]}"; do
  run="$OUT/direct_recipe_seed${seed}"
  raw="$run/test_predictions_b20_canonical_raw.jsonl"
  strict="$run/test_predictions_b20_canonical_strict.jsonl"
  if [[ ! -s "$raw" ]]; then
    echo "===== INFER RAW P1 B20 seed=$seed ====="
    python -m scripts.evaluate_mocheg_b18_test --runs "$run" \
      --manifest "$RAW_MANIFEST" --retrieval "$RAW_RETRIEVAL" --corpus "$CORPUS_ROOT/test/Corpus2.csv" \
      --base-model Qwen/Qwen3-4B-Instruct-2507 --tag b20_canonical_raw --output-dir "$OUT/raw_inference" \
      --top-k 5 --max-evidence-chars 2200 --batch-size 4 --device cuda --bootstrap-iterations 10000 \
      2>&1 | tee "$run/raw_inference.log"
  fi
  if [[ ! -s "$strict" ]]; then
    echo "===== INFER STRICT P1 B20 seed=$seed ====="
    python -m scripts.evaluate_mocheg_b18_test --runs "$run" \
      --manifest "$STRICT_MANIFEST" --retrieval "$STRICT_RETRIEVAL" --corpus "$CORPUS_ROOT/test/Corpus2.csv" \
      --base-model Qwen/Qwen3-4B-Instruct-2507 --tag b20_canonical_strict --output-dir "$OUT/strict_inference" \
      --top-k 5 --max-evidence-chars 2200 --batch-size 4 --device cuda --bootstrap-iterations 10000 \
      2>&1 | tee "$run/strict_inference.log"
  fi
done

for path in "${RAW_RATIONALE[@]}" "${STRICT_RATIONALE[@]}" "${RAW_DIRECT[@]}" "${STRICT_DIRECT[@]}"; do
  [[ -s "$path" ]] || { echo "ERROR: missing frozen prediction: $path" >&2; exit 1; }
done

python -m scripts.analyze_mocheg_b20_canonical_test \
  --raw-manifest "$RAW_MANIFEST" --strict-manifest "$STRICT_MANIFEST" \
  --raw-retrieval "$RAW_RETRIEVAL" --strict-retrieval "$STRICT_RETRIEVAL" \
  --raw-direct "${RAW_DIRECT[@]}" --strict-direct "${STRICT_DIRECT[@]}" \
  --raw-rationale "${RAW_RATIONALE[@]}" --strict-rationale "${STRICT_RATIONALE[@]}" \
  --iterations 10000 --seed 2026 \
  --output "$OUT/canonical_comparison.json" --markdown "$OUT/canonical_comparison.md" \
  2>&1 | tee "$OUT/canonical_comparison.log"

echo "DONE: $OUT/canonical_comparison.md"
