#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SCREEN="outputs/mocheg_b18b_evidence_k_screen"
OUT="outputs/mocheg_b18b_evidence_k_confirmation"
EXPLANATION_ROOT="data/processed/mocheg_b18_explanations"
mkdir -p "$OUT"

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

train_candidate() {
  local k="$1"
  local seed="$2"
  local explanations="$EXPLANATION_ROOT/train_top${k}_explanations.jsonl"
  local run_dir="$SCREEN/k${k}/candidate_seed${seed}"
  if [[ ! -s "$explanations" ]]; then
    echo "ERROR: missing Top-$k explanations: $explanations" >&2
    echo "Run scripts/run_mocheg_b18b_evidence_k_screen.sh first." >&2
    return 1
  fi
  mkdir -p "$run_dir"
  if [[ -s "$run_dir/summary.json" && -s "$run_dir/val_predictions.jsonl" ]]; then
    echo "SKIP K=$k seed=$seed (complete artifacts found)"
    return 0
  fi
  echo "START K=$k seed=$seed at $(date --iso-8601=seconds)"
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
      --seed "$seed" \
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
  local grounded_runs=()
  local seed
  if [[ "$k" == "5" ]]; then
    for seed in 42 87 100; do
      grounded_runs+=("outputs/mocheg_b18a_full/candidate_seed${seed}/val_predictions.jsonl")
    done
  else
    for seed in 42 87 100; do
      grounded_runs+=("$SCREEN/k${k}/candidate_seed${seed}/val_predictions.jsonl")
    done
  fi
  for candidate in "${grounded_runs[@]}"; do
    if [[ ! -s "$candidate" ]]; then
      echo "ERROR: missing grounded prediction: $candidate" >&2
      return 1
    fi
  done
  mkdir -p "$OUT/k${k}"
  python -m scripts.analyze_mocheg_b18b_component_ablations \
    --direct-runs "${DIRECT_RUNS[@]}" \
    --grounded-runs "${grounded_runs[@]}" \
    --tau 0.49 \
    --output "$OUT/k${k}/ablation.json" \
    --markdown "$OUT/k${k}/ablation.md" \
    2>&1 | tee "$OUT/k${k}/analysis.log"
}

START_SECONDS=$SECONDS
for k in 1 3; do
  for seed in 87 100; do
    train_candidate "$k" "$seed"
  done
done

for k in 1 3 5; do
  analyze_k "$k"
done

echo "Three-seed Evidence-K confirmation complete in $((SECONDS - START_SECONDS)) seconds"
for k in 1 3 5; do
  echo "===== TOP-K $k ====="
  grep 'A1_grounded_only\|A8_b18b_asymmetric' "$OUT/k${k}/ablation.md"
done
