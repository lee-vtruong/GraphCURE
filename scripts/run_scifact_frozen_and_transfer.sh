#!/usr/bin/env bash
# Zero-shot external transfer of the frozen MOCHEG CURE AND policy to SciFact.
# This does not train, select a seed, tune K/tau, or use gold evidence for
# retrieval. SciFact dev labels are consumed only by the final evaluator.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs/scifact_cure_and_zero_shot"
RAW="data/external/scifact_raw"
DATA="$RAW/data"
PREP="$OUT/protocol"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
mkdir -p "$OUT" "$RAW"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Official SciFact release archive from the AllenAI project README.
if [[ ! -s "$DATA/corpus.jsonl" || ! -s "$DATA/claims_dev.jsonl" ]]; then
  archive="$RAW/data.tar.gz"
  if [[ ! -s "$archive" ]]; then
    curl -L --fail --retry 3 \
      https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz \
      -o "$archive"
  fi
  tar -xzf "$archive" -C "$RAW"
fi
for required in "$DATA/corpus.jsonl" "$DATA/claims_dev.jsonl"; do
  [[ -s "$required" ]] || { echo "ERROR: SciFact release is incomplete: $required" >&2; exit 1; }
done

if [[ ! -s "$PREP/protocol.json" ]]; then
  python -m scripts.prepare_scifact_external_protocol \
    --corpus "$DATA/corpus.jsonl" --claims "$DATA/claims_dev.jsonl" \
    --output-root "$PREP" --top-k 5 \
    2>&1 | tee "$OUT/prepare.log"
fi

SAMPLES="$(python - <<'PY'
import json
print(json.load(open('outputs/scifact_cure_and_zero_shot/protocol/protocol.json'))['claims'])
PY
)"

DIRECT=(
  outputs/mocheg_qwen3_lora_seed13_v16
  outputs/mocheg_qwen3_lora_seed21_v16
  outputs/mocheg_qwen3_lora_seed42_v16
  outputs/mocheg_qwen3_lora_seed87_v16
  outputs/mocheg_qwen3_lora_seed100_v16
)
RATIONALE=(
  outputs/mocheg_b18a_full/candidate_seed42
  outputs/mocheg_b18a_full/candidate_seed87
  outputs/mocheg_b18a_full/candidate_seed100
)
for run in "${DIRECT[@]}" "${RATIONALE[@]}"; do
  [[ -s "$run/best_adapter/adapter_config.json" ]] || { echo "ERROR: adapter missing: $run" >&2; exit 1; }
done

if [[ ! -s "$OUT/direct_inference/inference_scifact_direct.json" ]]; then
  python -m scripts.evaluate_mocheg_qwen3_direct_runs \
    --runs "${DIRECT[@]}" --manifest "$PREP/dev_manifest.jsonl" \
    --retrieval "$PREP/dev_retrieval_tfidf.jsonl" --corpus "$PREP/Corpus2.csv" \
    --output-dir "$OUT/direct_inference" --tag scifact_direct \
    --expected-samples "$SAMPLES" --model "$MODEL" --top-k 5 \
    --max-evidence-chars 2200 --max-length 3072 --batch-size 4 --num-workers 2 --device cuda \
    2>&1 | tee "$OUT/direct_inference.log"
fi

if [[ ! -s "$OUT/rationale_inference/inference_scifact_rationale.json" ]]; then
  python -m scripts.evaluate_mocheg_b18_external_runs \
    --runs "${RATIONALE[@]}" --manifest "$PREP/dev_manifest.jsonl" \
    --retrieval "$PREP/dev_retrieval_tfidf.jsonl" --corpus "$PREP/Corpus2.csv" \
    --output-dir "$OUT/rationale_inference" --tag scifact_rationale \
    --expected-samples "$SAMPLES" --model "$MODEL" --top-k 5 \
    --max-evidence-chars 2200 --max-length 3072 --batch-size 4 --num-workers 2 --device cuda \
    2>&1 | tee "$OUT/rationale_inference.log"
fi

DIRECT_PRED=()
RATIONALE_PRED=()
for run in "${DIRECT[@]}"; do DIRECT_PRED+=("$run/test_predictions_scifact_direct.jsonl"); done
for run in "${RATIONALE[@]}"; do RATIONALE_PRED+=("$run/test_predictions_scifact_rationale.jsonl"); done
for path in "${DIRECT_PRED[@]}" "${RATIONALE_PRED[@]}"; do
  [[ -s "$path" ]] || { echo "ERROR: missing external prediction: $path" >&2; exit 1; }
done

python -m scripts.evaluate_cure_external_and \
  --manifest "$PREP/dev_manifest.jsonl" --retrieval "$PREP/dev_retrieval_tfidf.jsonl" \
  --expected-samples "$SAMPLES" --direct-runs "${DIRECT_PRED[@]}" \
  --rationale-runs "${RATIONALE_PRED[@]}" --top-k 5 --tau .49 \
  --iterations 10000 --seed 2026 --output-dir "$OUT/result" \
  2>&1 | tee "$OUT/evaluate.log"

echo "DONE: $OUT/result/report.md"
