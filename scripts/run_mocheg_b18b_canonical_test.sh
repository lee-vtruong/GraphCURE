#!/usr/bin/env bash
# Canonical, audit-only evaluation of the registered B18B AND policy.
# This script does NOT train, tune tau/K, or run model inference. It evaluates
# the already frozen raw-official prediction files, validates their provenance,
# and derives the strict report only as the verified subset of the same raw IDs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs/mocheg_b18b_canonical_test"
RAW_MANIFEST="data/processed/mocheg_manifest/test.jsonl"
STRICT_MANIFEST="data/processed/mocheg_manifest_strict/test.jsonl"
RAW_RETRIEVAL="outputs/retrieval_mocheg_qwen3_reranked_official/test.jsonl"
STRICT_RETRIEVAL="outputs/retrieval_mocheg_qwen3_reranked/test.jsonl"

DIRECT=()
for seed in 13 21 42 87 100; do
  DIRECT+=("outputs/mocheg_qwen3_lora_frozen_test/seed_${seed}_predictions.jsonl")
done
GROUNDED=(
  "outputs/mocheg_b18a_full/candidate_seed100/test_predictions_official.jsonl"
  "outputs/mocheg_b18a_full/candidate_seed87/test_predictions_official.jsonl"
  "outputs/mocheg_b18a_full/candidate_seed42/test_predictions_official.jsonl"
)

for path in "$RAW_MANIFEST" "$STRICT_MANIFEST" "$RAW_RETRIEVAL" "$STRICT_RETRIEVAL" \
  "${DIRECT[@]}" "${GROUNDED[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required canonical input is missing or empty: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUT"
python -m scripts.evaluate_mocheg_b18b_canonical_router \
  --raw-manifest "$RAW_MANIFEST" \
  --strict-manifest "$STRICT_MANIFEST" \
  --raw-retrieval "$RAW_RETRIEVAL" \
  --strict-retrieval "$STRICT_RETRIEVAL" \
  --direct-runs "${DIRECT[@]}" \
  --grounded-runs "${GROUNDED[@]}" \
  --top-k 5 \
  --tau 0.49 \
  --iterations 10000 \
  --seed 2026 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo "DONE: $OUT/canonical_router.md"
