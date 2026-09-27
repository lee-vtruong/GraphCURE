#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SCREEN="outputs/mocheg_b18b_evidence_k_screen"
OUT="outputs/mocheg_b18b_k3_official_test"
RETRIEVAL="outputs/retrieval_mocheg_qwen3_reranked_official/test.jsonl"
mkdir -p "$OUT"

K3_RUNS=()
for seed in 42 87 100; do
  run="$SCREEN/k3/candidate_seed${seed}"
  if [[ ! -d "$run/best_adapter" ]]; then
    echo "ERROR: missing K=3 adapter: $run/best_adapter" >&2
    exit 1
  fi
  K3_RUNS+=("$run")
done

DIRECT_TEST_RUNS=()
for seed in 13 21 42 87 100; do
  path="outputs/mocheg_qwen3_lora_frozen_test/seed_${seed}_predictions.jsonl"
  if [[ ! -s "$path" ]]; then
    echo "ERROR: missing frozen direct test predictions: $path" >&2
    exit 1
  fi
  DIRECT_TEST_RUNS+=("$path")
done

python -m scripts.evaluate_mocheg_b18_test \
  --runs "${K3_RUNS[@]}" \
  --manifest data/processed/mocheg_manifest/test.jsonl \
  --retrieval "$RETRIEVAL" \
  --retrieval-b18b "$RETRIEVAL" \
  --retrieval-top3 "$RETRIEVAL" \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/test/Corpus2.csv \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --top-k 3 \
  --max-evidence-chars 2200 \
  --tag official_k3 \
  --output-dir "$OUT/grounded_ensemble" \
  --batch-size 4 \
  --device cuda \
  --bootstrap-iterations 10000 \
  2>&1 | tee "$OUT/inference.log"

GROUNDED_TEST_RUNS=()
for seed in 42 87 100; do
  GROUNDED_TEST_RUNS+=(
    "$SCREEN/k3/candidate_seed${seed}/test_predictions_official_k3.jsonl"
  )
done

python -m scripts.evaluate_mocheg_b18b_frozen_router \
  --direct-runs "${DIRECT_TEST_RUNS[@]}" \
  --grounded-runs "${GROUNDED_TEST_RUNS[@]}" \
  --top-k 3 \
  --tau 0.49 \
  --iterations 10000 \
  --seed 2026 \
  --output "$OUT/frozen_router.json" \
  --markdown "$OUT/frozen_router.md" \
  2>&1 | tee "$OUT/frozen_router.log"

echo "Official K=3 evaluation complete: $OUT/frozen_router.md"
