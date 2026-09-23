#!/usr/bin/env bash
set -euo pipefail

# One-shot, restart-safe B19 fold runner. Every completed stage is cached.
FOLD="${B19_FOLD:-0}"
ROOT="${B19_ROOT:-outputs/mocheg_b19/fold_${FOLD}}"
MODEL="${B19_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MANIFEST="${B19_MANIFEST:-data/processed/mocheg_manifest_strict/train.jsonl}"
RETRIEVAL="${B19_RETRIEVAL:-outputs/retrieval_mocheg_qwen3_reranked/train.jsonl}"
CORPUS="${B19_CORPUS:-data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv}"
FOLDS="${B19_FOLDS:-data/processed/mocheg_b18_folds.json}"
EXPLANATIONS="${B19_EXPLANATIONS:-data/processed/mocheg_b18_explanations/train_fold${FOLD}_explanations.jsonl}"
DIRECT_ADAPTER="${B19_DIRECT_ADAPTER:-outputs/mocheg_b18/control_fold${FOLD}/best_adapter}"
GROUNDED_ADAPTER="${B19_GROUNDED_ADAPTER:-outputs/mocheg_b18/candidate_fold${FOLD}/best_adapter}"
DEVICE="${B19_DEVICE:-cuda}"
SEED="${B19_SEED:-42}"
EPOCHS="${B19_EPOCHS:-3}"
VARIANTS="${B19_VARIANTS:-matched_control ensemble_kd disagreement_kd counterfactual_only full}"

mkdir -p "$ROOT/teachers" "$ROOT/logs"

require_file() { test -f "$1" || { echo "MISSING FILE: $1" >&2; exit 2; }; }
require_dir() { test -d "$1" || { echo "MISSING DIR: $1" >&2; exit 2; }; }
require_file "$MANIFEST"
require_file "$RETRIEVAL"
require_file "$CORPUS"
require_file "$FOLDS"
require_file "$EXPLANATIONS"
require_dir "$DIRECT_ADAPTER"
require_dir "$GROUNDED_ADAPTER"

score_teacher() {
  local name="$1" adapter="$2"
  python -m scripts.score_mocheg_b19_teacher \
    --adapter "$adapter" --manifest "$MANIFEST" --retrieval "$RETRIEVAL" \
    --corpus "$CORPUS" --folds "$FOLDS" --fold "$FOLD" --subset train \
    --output "$ROOT/teachers/${name}.jsonl" \
    --summary "$ROOT/teachers/${name}_summary.json" \
    --model "$MODEL" --device "$DEVICE" \
    2>&1 | tee -a "$ROOT/logs/teacher_${name}.log"
}

score_teacher direct "$DIRECT_ADAPTER"
score_teacher grounded "$GROUNDED_ADAPTER"

for variant in $VARIANTS; do
  mkdir -p "$ROOT/$variant"
  python -m scripts.train_mocheg_b19_distillation \
    --variant "$variant" --manifest "$MANIFEST" --retrieval "$RETRIEVAL" \
    --corpus "$CORPUS" --folds "$FOLDS" --fold "$FOLD" \
    --explanations "$EXPLANATIONS" \
    --direct-teacher "$ROOT/teachers/direct.jsonl" \
    --grounded-teacher "$ROOT/teachers/grounded.jsonl" \
    --output "$ROOT/$variant" --model "$MODEL" --epochs "$EPOCHS" \
    --seed "$SEED" --device "$DEVICE" \
    2>&1 | tee -a "$ROOT/logs/${variant}.log"
done

if test "$VARIANTS" = "matched_control ensemble_kd disagreement_kd counterfactual_only full"; then
  python -m scripts.analyze_mocheg_b19_screen \
    --root "$ROOT" --output "$ROOT/screen.json" --markdown "$ROOT/screen.md" \
    --bootstrap-iterations 5000 --seed 2026 \
    2>&1 | tee -a "$ROOT/logs/analyze.log"
  cat "$ROOT/screen.md"
fi
echo "B19 fold $FOLD pipeline finished: $ROOT"
