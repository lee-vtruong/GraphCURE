#!/usr/bin/env bash
# B21: Frozen Qwen2.5 teacher-capacity ablation.
# The only experimental variable is the rationale teacher (3B vs the existing
# 7B arm); student recipe, seeds, prompts, retrieval and test policy remain fixed.
# B20 provides the complementary no-rationale control. Thus B21 tests whether
# the contribution within the rationale-supervised arm is capacity-dependent.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs/mocheg_b21_teacher_capacity"
EXPLANATIONS="data/processed/mocheg_b21_explanations/train_qwen25_3b.jsonl"
EXPLANATION_SUMMARY="outputs/mocheg_b21_explanations/train_qwen25_3b_summary.json"
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
  [[ -s "$required" ]] || { echo "ERROR: missing input: $required" >&2; exit 1; }
done

for seed in "${SEEDS[@]}"; do
  for required in \
    "outputs/mocheg_b18a_full/candidate_seed${seed}/test_predictions_canonical_k5.jsonl" \
    "outputs/mocheg_b18a_full/candidate_seed${seed}/test_predictions_canonical_k5_strict.jsonl"; do
    [[ -s "$required" ]] || { echo "ERROR: missing frozen 7B reference: $required" >&2; exit 1; }
  done
done

mkdir -p "$OUT" "$(dirname "$EXPLANATIONS")" "$(dirname "$EXPLANATION_SUMMARY")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -s "$EXPLANATIONS" || ! -s "$EXPLANATION_SUMMARY" ]]; then
  echo "===== GENERATE Qwen2.5-3B RATIONALES ====="
  python -m scripts.generate_mocheg_b18_teacher_explanations \
    --manifest "$TRAIN_MANIFEST" --retrieval "$TRAIN_RETRIEVAL" --corpus "$CORPUS_ROOT/train/Corpus2.csv" \
    --model Qwen/Qwen2.5-3B-Instruct --top-k 5 --max-evidence-chars 2200 --device cuda \
    --output "$EXPLANATIONS" --summary "$EXPLANATION_SUMMARY" \
    2>&1 | tee "$OUT/generate_teacher_3b.log"
else
  echo "SKIP teacher generation (complete)"
fi

RAW_3B=(); STRICT_3B=(); RAW_7B=(); STRICT_7B=()
for seed in "${SEEDS[@]}"; do
  run="$OUT/candidate_teacher3b_seed${seed}"
  RAW_3B+=("$run/test_predictions_b21_canonical_raw.jsonl")
  STRICT_3B+=("$run/test_predictions_b21_canonical_strict.jsonl")
  RAW_7B+=("outputs/mocheg_b18a_full/candidate_seed${seed}/test_predictions_canonical_k5.jsonl")
  STRICT_7B+=("outputs/mocheg_b18a_full/candidate_seed${seed}/test_predictions_canonical_k5_strict.jsonl")
  if [[ ! -s "$run/summary.json" || ! -s "$run/val_predictions.jsonl" ]]; then
    mkdir -p "$run"
    echo "===== TRAIN 3B-TEACHER STUDENT seed=$seed ====="
    python -m scripts.train_mocheg_b18_explanation_verifier \
      --mode explanation_candidate --manifest "$TRAIN_MANIFEST" --retrieval "$TRAIN_RETRIEVAL" --corpus "$CORPUS_ROOT/train/Corpus2.csv" \
      --val-manifest "$VAL_MANIFEST" --val-retrieval "$VAL_RETRIEVAL" --val-corpus "$CORPUS_ROOT/val/Corpus2.csv" \
      --explanations "$EXPLANATIONS" --model Qwen/Qwen3-4B-Instruct-2507 --output "$run" --seed "$seed" \
      --lambda-exp 0.25 --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4 --top-k 5 \
      --max-length 3072 --max-evidence-chars 2200 --device cuda \
      2>&1 | tee "$run/train.log"
  else
    echo "SKIP TRAIN seed=$seed (complete)"
  fi
done

for seed in "${SEEDS[@]}"; do
  run="$OUT/candidate_teacher3b_seed${seed}"
  raw="$run/test_predictions_b21_canonical_raw.jsonl"
  strict="$run/test_predictions_b21_canonical_strict.jsonl"
  if [[ ! -s "$raw" ]]; then
    echo "===== INFER RAW P1 3B-TEACHER seed=$seed ====="
    python -m scripts.evaluate_mocheg_b18_test --runs "$run" --manifest "$RAW_MANIFEST" --retrieval "$RAW_RETRIEVAL" --corpus "$CORPUS_ROOT/test/Corpus2.csv" \
      --base-model Qwen/Qwen3-4B-Instruct-2507 --tag b21_canonical_raw --output-dir "$OUT/raw_inference" \
      --top-k 5 --max-evidence-chars 2200 --batch-size 4 --device cuda --bootstrap-iterations 10000 \
      2>&1 | tee "$run/raw_inference.log"
  fi
  if [[ ! -s "$strict" ]]; then
    echo "===== INFER STRICT P1 3B-TEACHER seed=$seed ====="
    python -m scripts.evaluate_mocheg_b18_test --runs "$run" --manifest "$STRICT_MANIFEST" --retrieval "$STRICT_RETRIEVAL" --corpus "$CORPUS_ROOT/test/Corpus2.csv" \
      --base-model Qwen/Qwen3-4B-Instruct-2507 --tag b21_canonical_strict --output-dir "$OUT/strict_inference" \
      --top-k 5 --max-evidence-chars 2200 --batch-size 4 --device cuda --bootstrap-iterations 10000 \
      2>&1 | tee "$run/strict_inference.log"
  fi
done

for path in "${RAW_3B[@]}" "${STRICT_3B[@]}" "${RAW_7B[@]}" "${STRICT_7B[@]}"; do
  [[ -s "$path" ]] || { echo "ERROR: missing prediction: $path" >&2; exit 1; }
done

python -m scripts.analyze_mocheg_b21_teacher_capacity \
  --raw-manifest "$RAW_MANIFEST" --strict-manifest "$STRICT_MANIFEST" \
  --raw-retrieval "$RAW_RETRIEVAL" --strict-retrieval "$STRICT_RETRIEVAL" \
  --raw-3b "${RAW_3B[@]}" --strict-3b "${STRICT_3B[@]}" \
  --raw-7b "${RAW_7B[@]}" --strict-7b "${STRICT_7B[@]}" \
  --iterations 10000 --seed 2026 \
  --output "$OUT/canonical_comparison.json" --markdown "$OUT/canonical_comparison.md" \
  2>&1 | tee "$OUT/canonical_comparison.log"

echo "DONE: $OUT/canonical_comparison.md"
