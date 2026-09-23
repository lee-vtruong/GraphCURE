#!/usr/bin/env bash
set -euo pipefail

# Frozen B19-B confirmation. Builds fold-specific B18 teachers, then trains the
# paired B19 control and DKD runs. Safe to rerun after interruption.
MODEL="${B19_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
TEACHER_MODEL="${B19_EXPLANATION_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MANIFEST="${B19_MANIFEST:-data/processed/mocheg_manifest_strict/train.jsonl}"
B18_RETRIEVAL="${B19_B18_RETRIEVAL:-outputs/retrieval_mocheg_dense_top50/train.jsonl}"
B19_RETRIEVAL="${B19_RETRIEVAL:-outputs/retrieval_mocheg_qwen3_reranked/train.jsonl}"
CORPUS="${B19_CORPUS:-data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv}"
FOLDS="${B19_FOLDS:-data/processed/mocheg_b18_folds.json}"
ROOT="${B19_CONFIRM_ROOT:-outputs/mocheg_b19}"
SEED="${B19_SEED:-42}"
DEVICE="${B19_DEVICE:-cuda}"

mkdir -p "$ROOT/logs" data/processed/mocheg_b18_explanations outputs/mocheg_b18_explanations

complete_run() {
  local path="$1"
  python - "$path" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
try:
    ok = bool(json.loads(p.read_text()).get("complete"))
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

for fold in 1 2 3 4; do
  echo "===== B19-B FOLD $fold ====="
  explanations="data/processed/mocheg_b18_explanations/train_fold${fold}_explanations.jsonl"
  explanation_summary="outputs/mocheg_b18_explanations/summary_fold${fold}.json"
  control="outputs/mocheg_b18/control_fold${fold}"
  candidate="outputs/mocheg_b18/candidate_fold${fold}"

  if ! complete_run "$explanation_summary"; then
    CUDA_VISIBLE_DEVICES=0 python -m scripts.generate_mocheg_b18_teacher_explanations \
      --manifest "$MANIFEST" --retrieval "$B18_RETRIEVAL" --corpus "$CORPUS" \
      --folds "$FOLDS" --fold "$fold" --model "$TEACHER_MODEL" \
      --output "$explanations" --summary "$explanation_summary" --device "$DEVICE" \
      2>&1 | tee -a "$ROOT/logs/fold${fold}_explanations.log"
  fi

  if ! complete_run "$control/summary.json"; then
    mkdir -p "$control"
    CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_b18_explanation_verifier \
      --mode matched_control --manifest "$MANIFEST" --retrieval "$B18_RETRIEVAL" \
      --corpus "$CORPUS" --folds "$FOLDS" --fold "$fold" --model "$MODEL" \
      --output "$control" --epochs 3 --batch-size 2 --grad-accum 4 --device "$DEVICE" \
      2>&1 | tee -a "$ROOT/logs/fold${fold}_b18_control.log"
  fi

  if ! complete_run "$candidate/summary.json"; then
    mkdir -p "$candidate"
    CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_b18_explanation_verifier \
      --mode explanation_candidate --manifest "$MANIFEST" --retrieval "$B18_RETRIEVAL" \
      --corpus "$CORPUS" --folds "$FOLDS" --fold "$fold" \
      --explanations "$explanations" --model "$MODEL" --output "$candidate" \
      --lambda-exp 0.25 --epochs 3 --batch-size 2 --grad-accum 4 --device "$DEVICE" \
      2>&1 | tee -a "$ROOT/logs/fold${fold}_b18_candidate.log"
  fi

  B19_FOLD="$fold" \
  B19_ROOT="$ROOT/fold_${fold}" \
  B19_EXPLANATIONS="$explanations" \
  B19_DIRECT_ADAPTER="$control/best_adapter" \
  B19_GROUNDED_ADAPTER="$candidate/best_adapter" \
  B19_RETRIEVAL="$B19_RETRIEVAL" \
  B19_VARIANTS="matched_control disagreement_kd" \
  B19_SEED="$SEED" \
  B19_DEVICE="$DEVICE" \
    bash scripts/run_mocheg_b19_fold0.sh \
      2>&1 | tee -a "$ROOT/logs/fold${fold}_b19.log"
done

python -m scripts.summarize_mocheg_b19_confirmation \
  --root "$ROOT" --folds 0 1 2 3 4 \
  --output "$ROOT/confirmation.json" --markdown "$ROOT/confirmation.md" \
  --bootstrap-iterations 10000 --seed 2026 \
  2>&1 | tee -a "$ROOT/logs/confirmation.log"

cat "$ROOT/confirmation.md"
