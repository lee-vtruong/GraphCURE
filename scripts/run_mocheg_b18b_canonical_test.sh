#!/usr/bin/env bash
# Canonical, audit-only evaluation of the registered B18B AND policy.
# This script does NOT train or tune tau/K. It creates a fresh, named raw-P1
# prediction artifact for frozen adapters only when it is absent.  Raw and
# strict retrieval manifests are audited separately; if their ranked evidence
# IDs differ, strict predictions are independently inferred instead of being
# incorrectly sliced from raw predictions.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs/mocheg_b18b_canonical_test"
RAW_MANIFEST="data/processed/mocheg_manifest/test.jsonl"
STRICT_MANIFEST="data/processed/mocheg_manifest_strict/test.jsonl"
RAW_RETRIEVAL="outputs/retrieval_mocheg_qwen3_reranked_official/test.jsonl"
STRICT_RETRIEVAL="outputs/retrieval_mocheg_qwen3_reranked/test.jsonl"

DIRECT=()
STRICT_DIRECT=()
for seed in 13 21 42 87 100; do
  DIRECT+=("$OUT/direct_raw/seed_${seed}_predictions.jsonl")
  STRICT_DIRECT+=("$OUT/direct_strict/seed_${seed}_predictions.jsonl")
done
GROUNDED=(
  "outputs/mocheg_b18a_full/candidate_seed100/test_predictions_canonical_k5.jsonl"
  "outputs/mocheg_b18a_full/candidate_seed87/test_predictions_canonical_k5.jsonl"
  "outputs/mocheg_b18a_full/candidate_seed42/test_predictions_canonical_k5.jsonl"
)
STRICT_GROUNDED=(
  "outputs/mocheg_b18a_full/candidate_seed100/test_predictions_canonical_k5_strict.jsonl"
  "outputs/mocheg_b18a_full/candidate_seed87/test_predictions_canonical_k5_strict.jsonl"
  "outputs/mocheg_b18a_full/candidate_seed42/test_predictions_canonical_k5_strict.jsonl"
)

for path in "$RAW_MANIFEST" "$STRICT_MANIFEST" "$RAW_RETRIEVAL" "$STRICT_RETRIEVAL"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required canonical input is missing or empty: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUT"

# Historic direct predictions were retained on the strict ID universe only.
# Produce a dedicated raw-P1 artifact from the same frozen five B1 adapters;
# this is inference-only and prevents any implicit raw/strict intersection.
NEED_RAW_DIRECT=0
for path in "${DIRECT[@]}"; do
  [[ -s "$path" ]] || NEED_RAW_DIRECT=1
done
if [[ "$NEED_RAW_DIRECT" -eq 1 ]]; then
  echo "Creating missing canonical K=5 raw-P1 predictions for frozen direct adapters."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python -m scripts.evaluate_mocheg_qwen3_frozen_test \
    --manifest "$RAW_MANIFEST" \
    --retrieval "$RAW_RETRIEVAL" \
    --corpus data/raw/mocheg_dataset/extracted/mocheg/test/Corpus2.csv \
    --output-root "$OUT/direct_raw" \
    --protocol P1_closed_corpus_retrieved_raw_n2442_canonical_b18b \
    --expected-samples 2442 \
    --batch-size 4 \
    --bootstrap-iterations 10000 \
    --device cuda \
    2>&1 | tee "$OUT/direct_raw.log"
fi
for path in "${DIRECT[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: canonical raw direct prediction was not produced: $path" >&2
    exit 1
  fi
done

# The historic report may contain a different prediction tag or no retained
# raw prediction at all.  Always use this dedicated tag for the audit.  This
# is inference-only (three adapters, no training); cached canonical files are
# reused on restart.
NEED_INFERENCE=0
ADAPTERS=()
for seed in 100 87 42; do
  run="outputs/mocheg_b18a_full/candidate_seed${seed}"
  adapter="$run/best_adapter"
  prediction="$run/test_predictions_canonical_k5.jsonl"
  if [[ ! -d "$adapter" ]]; then
    echo "ERROR: frozen grounded adapter is missing: $adapter" >&2
    exit 1
  fi
  ADAPTERS+=("$run")
  if [[ ! -s "$prediction" ]]; then
    NEED_INFERENCE=1
  fi
done

if [[ "$NEED_INFERENCE" -eq 1 ]]; then
  echo "Creating missing canonical K=5 raw-P1 predictions for frozen grounded adapters."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python -m scripts.evaluate_mocheg_b18_test \
    --runs "${ADAPTERS[@]}" \
    --manifest "$RAW_MANIFEST" \
    --retrieval "$RAW_RETRIEVAL" \
    --corpus data/raw/mocheg_dataset/extracted/mocheg/test/Corpus2.csv \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --tag canonical_k5 \
    --output-dir "$OUT/grounded_inference" \
    --top-k 5 \
    --max-evidence-chars 2200 \
    --batch-size 4 \
    --device cuda \
    --bootstrap-iterations 10000 \
    2>&1 | tee "$OUT/grounded_inference.log"
fi

for path in "${GROUNDED[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: canonical grounded prediction was not produced: $path" >&2
    exit 1
  fi
done

# The prior audit showed at least one raw/strict ranked-evidence mismatch.
# Therefore strict predictions must be independently inferred with the same
# frozen adapters and registered K=5 policy; no model is trained here.
NEED_STRICT_DIRECT=0
for path in "${STRICT_DIRECT[@]}"; do
  [[ -s "$path" ]] || NEED_STRICT_DIRECT=1
done
if [[ "$NEED_STRICT_DIRECT" -eq 1 ]]; then
  echo "Creating missing canonical K=5 strict-P1 predictions for frozen direct adapters."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python -m scripts.evaluate_mocheg_qwen3_frozen_test \
    --manifest "$STRICT_MANIFEST" \
    --retrieval "$STRICT_RETRIEVAL" \
    --corpus data/raw/mocheg_dataset/extracted/mocheg/test/Corpus2.csv \
    --output-root "$OUT/direct_strict" \
    --protocol P1_closed_corpus_retrieved_strict_n2434_canonical_b18b \
    --expected-samples 2434 \
    --batch-size 4 \
    --bootstrap-iterations 10000 \
    --device cuda \
    2>&1 | tee "$OUT/direct_strict.log"
fi

NEED_STRICT_GROUNDED=0
for path in "${STRICT_GROUNDED[@]}"; do
  [[ -s "$path" ]] || NEED_STRICT_GROUNDED=1
done
if [[ "$NEED_STRICT_GROUNDED" -eq 1 ]]; then
  echo "Creating missing canonical K=5 strict-P1 predictions for frozen grounded adapters."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python -m scripts.evaluate_mocheg_b18_test \
    --runs "${ADAPTERS[@]}" \
    --manifest "$STRICT_MANIFEST" \
    --retrieval "$STRICT_RETRIEVAL" \
    --corpus data/raw/mocheg_dataset/extracted/mocheg/test/Corpus2.csv \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --tag canonical_k5_strict \
    --output-dir "$OUT/grounded_strict_inference" \
    --top-k 5 \
    --max-evidence-chars 2200 \
    --batch-size 4 \
    --device cuda \
    --bootstrap-iterations 10000 \
    2>&1 | tee "$OUT/grounded_strict_inference.log"
fi

for path in "${STRICT_DIRECT[@]}" "${STRICT_GROUNDED[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: canonical strict prediction was not produced: $path" >&2
    exit 1
  fi
done

python -m scripts.evaluate_mocheg_b18b_canonical_router \
  --raw-manifest "$RAW_MANIFEST" \
  --strict-manifest "$STRICT_MANIFEST" \
  --raw-retrieval "$RAW_RETRIEVAL" \
  --strict-retrieval "$STRICT_RETRIEVAL" \
  --direct-runs "${DIRECT[@]}" \
  --grounded-runs "${GROUNDED[@]}" \
  --strict-direct-runs "${STRICT_DIRECT[@]}" \
  --strict-grounded-runs "${STRICT_GROUNDED[@]}" \
  --top-k 5 \
  --tau 0.49 \
  --iterations 10000 \
  --seed 2026 \
  --prediction-provenance canonical_raw_inference \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo "DONE: $OUT/canonical_router.md"
