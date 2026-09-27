#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="outputs/mocheg_b18b_efficiency"
mkdir -p "$OUT"

DIRECT=()
for seed in 13 21 42 87 100; do
  DIRECT+=("outputs/mocheg_qwen3_lora_seed${seed}_v16/best_adapter")
done

GROUNDED=()
for seed in 42 87 100; do
  GROUNDED+=("outputs/mocheg_b18a_full/candidate_seed${seed}/best_adapter")
done

B19_ARGS=()
if [[ -n "${B19_ADAPTER:-}" ]]; then
  B19_ARGS=(--b19-adapter "$B19_ADAPTER")
else
  found="$(find outputs/mocheg_b19/full_train -type f -path '*best_adapter/adapter_config.json' \
    -ipath '*disagreement*' -ipath '*87*' -print -quit 2>/dev/null || true)"
  if [[ -n "$found" ]]; then
    B19_ARGS=(--b19-adapter "$(dirname "$found")")
    echo "Auto-detected B19 adapter: ${B19_ARGS[1]}"
  else
    echo "WARNING: B19 seed-87 adapter not found; benchmarking B18B components only."
    echo "Set B19_ADAPTER=/path/to/best_adapter to include it."
  fi
fi

python -m scripts.benchmark_mocheg_b18b_efficiency \
  --direct-adapters "${DIRECT[@]}" \
  --grounded-adapters "${GROUNDED[@]}" \
  "${B19_ARGS[@]}" \
  --manifest data/processed/mocheg_manifest_strict/val.jsonl \
  --retrieval outputs/retrieval_mocheg_qwen3_reranked/val.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/val/Corpus2.csv \
  --top-k 5 \
  --max-evidence-chars 2200 \
  --max-length 3072 \
  --batch-size "${BATCH_SIZE:-4}" \
  --limit "${LIMIT:-256}" \
  --warmup-batches 2 \
  --output "$OUT/summary.json" \
  --markdown "$OUT/summary.md" \
  2>&1 | tee "$OUT/benchmark.log"

echo "DONE: $OUT/summary.md"
