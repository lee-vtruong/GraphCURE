#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="outputs/mocheg_b18b_reviewer_controls"
mkdir -p "$OUT"

direct_val=()
for seed in 13 21 42 87 100; do direct_val+=("outputs/mocheg_qwen3_lora_seed${seed}_v16/val_predictions.jsonl"); done
grounded_val=()
for seed in 42 87 100; do grounded_val+=("outputs/mocheg_b18a_full/candidate_seed${seed}/val_predictions.jsonl"); done
raw_direct=()
strict_direct=()
for seed in 13 21 42 87 100; do
  raw_direct+=("outputs/mocheg_b18b_canonical_test/direct_raw/seed_${seed}_predictions.jsonl")
  strict_direct+=("outputs/mocheg_b18b_canonical_test/direct_strict/seed_${seed}_predictions.jsonl")
done
for path in "${direct_val[@]}" "${grounded_val[@]}" "${raw_direct[@]}" "${strict_direct[@]}"; do
  [[ -s "$path" ]] || { echo "Missing required prediction: $path" >&2; exit 1; }
done

python -m scripts.analyze_mocheg_b18b_reviewer_controls \
  --direct-val-runs "${direct_val[@]}" \
  --grounded-val-runs "${grounded_val[@]}" \
  --val-manifest data/processed/mocheg_manifest_strict/val.jsonl \
  --val-retrieval outputs/retrieval_mocheg_qwen3_reranked/val.jsonl \
  --raw-direct-runs "${raw_direct[@]}" \
  --strict-direct-runs "${strict_direct[@]}" \
  --and-tau 0.49 --top-k 5 --iterations 10000 --seed 2026 \
  --output "$OUT/summary.json" --markdown "$OUT/summary.md" \
  2>&1 | tee "$OUT/run.log"
