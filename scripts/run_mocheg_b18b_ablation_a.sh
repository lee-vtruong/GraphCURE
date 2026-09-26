#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="outputs/mocheg_b18b_component_ablation"
mkdir -p "$OUT"

resolve_val_predictions() {
  local run_root="$1"
  local candidate
  for candidate in \
    "$run_root/val_predictions.jsonl" \
    "${run_root}_v16/val_predictions.jsonl"
  do
    if [[ -s "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "ERROR: no validation predictions for $run_root" >&2
  echo "Checked: $run_root/val_predictions.jsonl" >&2
  echo "Checked: ${run_root}_v16/val_predictions.jsonl" >&2
  return 1
}

DIRECT_RUNS=()
for seed in 13 21 42 87 100; do
  DIRECT_RUNS+=("$(resolve_val_predictions "outputs/mocheg_qwen3_lora_seed${seed}")")
done

GROUNDED_RUNS=()
for seed in 42 87 100; do
  GROUNDED_RUNS+=("$(resolve_val_predictions "outputs/mocheg_b18a_full/candidate_seed${seed}")")
done

printf 'Direct predictions:\n  %s\n' "${DIRECT_RUNS[@]}"
printf 'Grounded predictions:\n  %s\n' "${GROUNDED_RUNS[@]}"

python -m scripts.analyze_mocheg_b18b_component_ablations \
  --direct-runs "${DIRECT_RUNS[@]}" \
  --grounded-runs "${GROUNDED_RUNS[@]}" \
  --tau 0.49 \
  --output "$OUT/summary.json" \
  --markdown "$OUT/summary.md" \
  2>&1 | tee "$OUT/run.log"

echo "Wrote $OUT/summary.json and $OUT/summary.md"
