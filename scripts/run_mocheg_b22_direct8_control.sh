#!/usr/bin/env bash
# B22: a fixed-member-count direct-only control for CURE-Ensemble.
# Three additional direct-only seeds are declared below before test inference.
# They use the legacy B1 verdict recipe, and every one is retained regardless
# of validation score.  Test labels are used only in the final frozen report.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs/mocheg_b22_direct8_control"
SEEDS=(314 2718 2026)
MODEL="Qwen/Qwen3-4B-Instruct-2507"
MANIFEST_ROOT="data/processed/mocheg_manifest_strict"
RETRIEVAL_ROOT="outputs/retrieval_mocheg_qwen3_reranked"
RAW_ROOT="data/raw/mocheg_dataset/extracted/mocheg"
RAW_MANIFEST="data/processed/mocheg_manifest/test.jsonl"
RAW_RETRIEVAL="outputs/retrieval_mocheg_qwen3_reranked_official/test.jsonl"
STRICT_MANIFEST="$MANIFEST_ROOT/test.jsonl"
STRICT_RETRIEVAL="$RETRIEVAL_ROOT/test.jsonl"

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for required in \
  "$MANIFEST_ROOT/train.jsonl" "$MANIFEST_ROOT/val.jsonl" \
  "$RETRIEVAL_ROOT/train.jsonl" "$RETRIEVAL_ROOT/val.jsonl" \
  "$RAW_MANIFEST" "$RAW_RETRIEVAL" "$STRICT_MANIFEST" "$STRICT_RETRIEVAL" \
  "$RAW_ROOT/train/Corpus2.csv" "$RAW_ROOT/val/Corpus2.csv" "$RAW_ROOT/test/Corpus2.csv"
do
  [[ -s "$required" ]] || { echo "ERROR: missing required input: $required" >&2; exit 1; }
done

# Legacy members must already have checksum-audited canonical predictions.
for seed in 13 21 42 87 100; do
  for required in \
    "outputs/mocheg_b18b_canonical_test/direct_raw/seed_${seed}_predictions.jsonl" \
    "outputs/mocheg_b18b_canonical_test/direct_strict/seed_${seed}_predictions.jsonl"
  do
    [[ -s "$required" ]] || { echo "ERROR: missing legacy frozen member: $required" >&2; exit 1; }
  done
done
for seed in 42 87 100; do
  for required in \
    "outputs/mocheg_b18a_full/candidate_seed${seed}/test_predictions_canonical_k5.jsonl" \
    "outputs/mocheg_b18a_full/candidate_seed${seed}/test_predictions_canonical_k5_strict.jsonl"
  do
    [[ -s "$required" ]] || { echo "ERROR: missing frozen rationale member: $required" >&2; exit 1; }
  done
done

RUNS=()
RAW_ADDED=()
STRICT_ADDED=()
for seed in "${SEEDS[@]}"; do
  run="$OUT/direct_seed${seed}"
  RUNS+=("$run")
  RAW_ADDED+=("$run/test_predictions_b22_raw.jsonl")
  STRICT_ADDED+=("$run/test_predictions_b22_strict.jsonl")
  if [[ ! -s "$run/summary.json" || ! -s "$run/val_predictions.jsonl" ]]; then
    mkdir -p "$run"
    echo "===== B22 LEGACY-RECIPE DIRECT TRAIN seed=$seed ====="
    python -m scripts.train_mocheg_qwen3_lora_verifier \
      --manifest-root "$MANIFEST_ROOT" --retrieval-root "$RETRIEVAL_ROOT" --raw-root "$RAW_ROOT" \
      --model "$MODEL" --output "$run" --seed "$seed" \
      --top-k 5 --max-evidence-chars 2200 --max-length 3072 \
      --lora-r 16 --lora-alpha 32 --lora-dropout .05 \
      --epochs 3 --patience 2 --batch-size 1 --gradient-accumulation 16 \
      --learning-rate 1e-4 --weight-decay .01 --warmup-ratio .05 \
      --num-workers 2 --device cuda \
      2>&1 | tee "$run/train.log"
  else
    echo "SKIP TRAIN seed=$seed (complete)"
  fi
done

if [[ ! -s "$OUT/inference_b22_raw.json" ]]; then
  python -m scripts.evaluate_mocheg_qwen3_direct_runs \
    --runs "${RUNS[@]}" --manifest "$RAW_MANIFEST" --retrieval "$RAW_RETRIEVAL" \
    --corpus "$RAW_ROOT/test/Corpus2.csv" --output-dir "$OUT" --tag b22_raw \
    --expected-samples 2442 --model "$MODEL" --top-k 5 --max-evidence-chars 2200 \
    --max-length 3072 --batch-size 4 --num-workers 2 --device cuda \
    2>&1 | tee "$OUT/raw_inference.log"
else
  echo "SKIP RAW INFERENCE (complete)"
fi

if [[ ! -s "$OUT/inference_b22_strict.json" ]]; then
  python -m scripts.evaluate_mocheg_qwen3_direct_runs \
    --runs "${RUNS[@]}" --manifest "$STRICT_MANIFEST" --retrieval "$STRICT_RETRIEVAL" \
    --corpus "$RAW_ROOT/test/Corpus2.csv" --output-dir "$OUT" --tag b22_strict \
    --expected-samples 2434 --model "$MODEL" --top-k 5 --max-evidence-chars 2200 \
    --max-length 3072 --batch-size 4 --num-workers 2 --device cuda \
    2>&1 | tee "$OUT/strict_inference.log"
else
  echo "SKIP STRICT INFERENCE (complete)"
fi

for path in "${RAW_ADDED[@]}" "${STRICT_ADDED[@]}"; do
  [[ -s "$path" ]] || { echo "ERROR: missing added direct prediction: $path" >&2; exit 1; }
done

python -m scripts.analyze_mocheg_b22_direct8_control \
  --raw-manifest "$RAW_MANIFEST" --strict-manifest "$STRICT_MANIFEST" \
  --raw-retrieval "$RAW_RETRIEVAL" --strict-retrieval "$STRICT_RETRIEVAL" \
  --raw-legacy-direct \
    outputs/mocheg_b18b_canonical_test/direct_raw/seed_13_predictions.jsonl \
    outputs/mocheg_b18b_canonical_test/direct_raw/seed_21_predictions.jsonl \
    outputs/mocheg_b18b_canonical_test/direct_raw/seed_42_predictions.jsonl \
    outputs/mocheg_b18b_canonical_test/direct_raw/seed_87_predictions.jsonl \
    outputs/mocheg_b18b_canonical_test/direct_raw/seed_100_predictions.jsonl \
  --strict-legacy-direct \
    outputs/mocheg_b18b_canonical_test/direct_strict/seed_13_predictions.jsonl \
    outputs/mocheg_b18b_canonical_test/direct_strict/seed_21_predictions.jsonl \
    outputs/mocheg_b18b_canonical_test/direct_strict/seed_42_predictions.jsonl \
    outputs/mocheg_b18b_canonical_test/direct_strict/seed_87_predictions.jsonl \
    outputs/mocheg_b18b_canonical_test/direct_strict/seed_100_predictions.jsonl \
  --raw-added-direct "${RAW_ADDED[@]}" --strict-added-direct "${STRICT_ADDED[@]}" \
  --raw-rationale \
    outputs/mocheg_b18a_full/candidate_seed42/test_predictions_canonical_k5.jsonl \
    outputs/mocheg_b18a_full/candidate_seed87/test_predictions_canonical_k5.jsonl \
    outputs/mocheg_b18a_full/candidate_seed100/test_predictions_canonical_k5.jsonl \
  --strict-rationale \
    outputs/mocheg_b18a_full/candidate_seed42/test_predictions_canonical_k5_strict.jsonl \
    outputs/mocheg_b18a_full/candidate_seed87/test_predictions_canonical_k5_strict.jsonl \
    outputs/mocheg_b18a_full/candidate_seed100/test_predictions_canonical_k5_strict.jsonl \
  --iterations 10000 --seed 2026 \
  --output "$OUT/size_control.json" --markdown "$OUT/size_control.md" \
  2>&1 | tee "$OUT/size_control.log"

echo "DONE: $OUT/size_control.md"
