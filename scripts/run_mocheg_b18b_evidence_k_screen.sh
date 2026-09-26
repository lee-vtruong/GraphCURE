#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs/mocheg_b18b_evidence_k_screen"
EXPLANATION_ROOT="data/processed/mocheg_b18_explanations"
mkdir -p "$OUT" "$EXPLANATION_ROOT"

resolve_direct_prediction() {
  local seed="$1"
  local candidate
  for candidate in \
    "outputs/mocheg_qwen3_lora_seed${seed}/val_predictions.jsonl" \
    "outputs/mocheg_qwen3_lora_seed${seed}_v16/val_predictions.jsonl"
  do
    if [[ -s "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "ERROR: missing direct validation predictions for seed $seed" >&2
  return 1
}

DIRECT_RUNS=()
for seed in 13 21 42 87 100; do
  DIRECT_RUNS+=("$(resolve_direct_prediction "$seed")")
done

generate_explanations() {
  local k="$1"
  local explanations="$EXPLANATION_ROOT/train_top${k}_explanations.jsonl"
  local summary="$OUT/teacher_top${k}_summary.json"
  local log="$OUT/teacher_top${k}.log"
  if [[ -s "$explanations" && -s "$summary" ]]; then
    echo "SKIP teacher K=$k (complete artifacts found)"
    return 0
  fi
  echo "START teacher K=$k at $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python -m scripts.generate_mocheg_b18_teacher_explanations \
      --manifest data/processed/mocheg_manifest_strict/train.jsonl \
      --retrieval outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
      --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
      --model Qwen/Qwen2.5-7B-Instruct \
      --output "$explanations" \
      --summary "$summary" \
      --top-k "$k" \
      --max-evidence-chars 2200 \
      --device cuda \
      2>&1 | tee "$log"
}

train_candidate() {
  local k="$1"
  local run_dir="$OUT/k${k}/candidate_seed42"
  local explanations="$EXPLANATION_ROOT/train_top${k}_explanations.jsonl"
  mkdir -p "$run_dir"
  if [[ -s "$run_dir/summary.json" && -s "$run_dir/val_predictions.jsonl" ]]; then
    echo "SKIP candidate K=$k seed=42 (complete artifacts found)"
    return 0
  fi
  echo "START candidate K=$k seed=42 at $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python -m scripts.train_mocheg_b18_explanation_verifier \
      --mode explanation_candidate \
      --manifest data/processed/mocheg_manifest_strict/train.jsonl \
      --retrieval outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
      --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
      --val-manifest data/processed/mocheg_manifest_strict/val.jsonl \
      --val-retrieval outputs/retrieval_mocheg_qwen3_reranked/val.jsonl \
      --val-corpus data/raw/mocheg_dataset/extracted/mocheg/val/Corpus2.csv \
      --explanations "$explanations" \
      --model Qwen/Qwen3-4B-Instruct-2507 \
      --output "$run_dir" \
      --seed 42 \
      --lambda-exp 0.25 \
      --top-k "$k" \
      --max-evidence-chars 2200 \
      --epochs 3 \
      --batch-size 2 \
      --grad-accum 4 \
      --device cuda \
      2>&1 | tee "$run_dir/train.log"
}

analyze_k() {
  local k="$1"
  local grounded
  if [[ "$k" == "5" ]]; then
    grounded="outputs/mocheg_b18a_full/candidate_seed42/val_predictions.jsonl"
  else
    grounded="$OUT/k${k}/candidate_seed42/val_predictions.jsonl"
  fi
  if [[ ! -s "$grounded" ]]; then
    echo "ERROR: missing grounded validation predictions: $grounded" >&2
    return 1
  fi
  python -m scripts.analyze_mocheg_b18b_component_ablations \
    --direct-runs "${DIRECT_RUNS[@]}" \
    --grounded-runs "$grounded" \
    --tau 0.49 \
    --output "$OUT/k${k}/ablation.json" \
    --markdown "$OUT/k${k}/ablation.md" \
    2>&1 | tee "$OUT/k${k}/analysis.log"
}

START_SECONDS=$SECONDS
for k in 1 3; do
  generate_explanations "$k"
  train_candidate "$k"
  analyze_k "$k"
done
analyze_k 5

echo "Evidence-K screening complete in $((SECONDS - START_SECONDS)) seconds"
echo "Results:"
for k in 1 3 5; do
  echo "  $OUT/k${k}/ablation.md"
done
