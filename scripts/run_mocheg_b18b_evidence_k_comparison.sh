#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SCREEN="outputs/mocheg_b18b_evidence_k_screen"
OUT="outputs/mocheg_b18b_evidence_k_confirmation"
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

K1_RUNS=()
K3_RUNS=()
K5_RUNS=()
for seed in 42 87 100; do
  K1_RUNS+=("$SCREEN/k1/candidate_seed${seed}/val_predictions.jsonl")
  K3_RUNS+=("$SCREEN/k3/candidate_seed${seed}/val_predictions.jsonl")
  K5_RUNS+=("outputs/mocheg_b18a_full/candidate_seed${seed}/val_predictions.jsonl")
done

for path in "${DIRECT_RUNS[@]}" "${K1_RUNS[@]}" "${K3_RUNS[@]}" "${K5_RUNS[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: missing prediction file: $path" >&2
    exit 1
  fi
done

python -m scripts.compare_mocheg_b18b_evidence_k \
  --direct-runs "${DIRECT_RUNS[@]}" \
  --k1-runs "${K1_RUNS[@]}" \
  --k3-runs "${K3_RUNS[@]}" \
  --k5-runs "${K5_RUNS[@]}" \
  --tau 0.49 \
  --iterations 10000 \
  --seed 2026 \
  --output "$OUT/comparison.json" \
  --markdown "$OUT/comparison.md" \
  2>&1 | tee "$OUT/comparison.log"
