#!/usr/bin/env bash
# End-to-end SciFact Domain Adaptation Pipeline (Adaptation-A)
#
# Protocol:
# 1. Same SciFact TF-IDF retrieval (no dense retriever, isolates adaptation effect)
# 2. Train Direct & Rationale experts on SciFact train set
# 3. K-fold OOF cross-validation on SciFact train strictly for tau* calibration
# 4. Evaluate 4 core systems on 300-claim SciFact dev set (held-out)
#
# All steps are restart-safe and cached.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_BASE="outputs/scifact_adaptation"
PROTOCOL_DIR="$OUT_BASE/protocol"
DEV_PROTOCOL="outputs/scifact_cure_and_zero_shot/protocol"
RAW="data/external/scifact_raw"
DATA="$RAW/data"
MODEL="Qwen/Qwen3-4B-Instruct-2507"

mkdir -p "$OUT_BASE" "$PROTOCOL_DIR" "$RAW"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "======================================================================"
echo "  GraphCURE: SciFact Domain Adaptation (Adaptation-A Pipeline)"
echo "  Date: $(date)"
echo "  Model: $MODEL"
echo "======================================================================"

# -----------------------------------------------------------------------------
# Bước 0: Kiểm tra hoặc tải dữ liệu SciFact gốc
# -----------------------------------------------------------------------------
if [[ ! -s "$DATA/corpus.jsonl" || ! -s "$DATA/claims_train.jsonl" || ! -s "$DATA/claims_dev.jsonl" ]]; then
  echo ">>> Downloading official SciFact release archive..."
  archive="$RAW/data.tar.gz"
  if [[ ! -s "$archive" ]]; then
    curl -L --fail --retry 3 \
      https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz \
      -o "$archive"
  fi
  tar -xzf "$archive" -C "$RAW"
fi

for required in "$DATA/corpus.jsonl" "$DATA/claims_train.jsonl" "$DATA/claims_dev.jsonl"; do
  [[ -s "$required" ]] || { echo "ERROR: SciFact file missing: $required" >&2; exit 1; }
done

# Đảm bảo dev protocol đã tồn tại
if [[ ! -s "$DEV_PROTOCOL/protocol.json" ]]; then
  echo ">>> Preparing SciFact dev protocol (zero-shot benchmark)..."
  mkdir -p "$DEV_PROTOCOL"
  python -m scripts.prepare_scifact_external_protocol \
    --corpus "$DATA/corpus.jsonl" --claims "$DATA/claims_dev.jsonl" \
    --output-root "$DEV_PROTOCOL" --top-k 5
fi

# -----------------------------------------------------------------------------
# Bước 1: Chuẩn bị SciFact Train Protocol & 5-Fold Stratified Splits
# -----------------------------------------------------------------------------
if [[ ! -s "$PROTOCOL_DIR/protocol.json" ]]; then
  echo ">>> [Step 1] Preparing SciFact train protocol and 5-fold CV splits..."
  python -m scripts.prepare_scifact_adaptation_protocol \
    --corpus "$DATA/corpus.jsonl" \
    --claims-train "$DATA/claims_train.jsonl" \
    --output-root "$PROTOCOL_DIR" \
    --top-k 5 \
    --folds 5 \
    --seed 2040 \
    2>&1 | tee "$OUT_BASE/step1_prep.log"
else
  echo ">>> [Step 1] Protocol cached at $PROTOCOL_DIR/protocol.json"
fi

# -----------------------------------------------------------------------------
# Bước 2: Sinh Structured Teacher Explanations cho SciFact Train
# -----------------------------------------------------------------------------
if [[ ! -s "$PROTOCOL_DIR/train_explanations.jsonl" ]]; then
  echo ">>> [Step 2] Generating teacher rationales on SciFact train..."
  python -m scripts.generate_scifact_teacher_explanations \
    --manifest "$PROTOCOL_DIR/train_manifest.jsonl" \
    --retrieval "$PROTOCOL_DIR/train_retrieval_tfidf.jsonl" \
    --corpus "$PROTOCOL_DIR/Corpus2.csv" \
    --output "$PROTOCOL_DIR/train_explanations.jsonl" \
    --summary "$PROTOCOL_DIR/train_explanations_summary.json" \
    --model "$MODEL" \
    --top-k 5 \
    --device cuda \
    2>&1 | tee "$OUT_BASE/step2_teacher.log"
else
  echo ">>> [Step 2] Teacher rationales cached at $PROTOCOL_DIR/train_explanations.jsonl"
fi

# -----------------------------------------------------------------------------
# Bước 3: K-Fold Out-of-Fold (OOF) Training để Calibrate Tau* (5 folds)
# -----------------------------------------------------------------------------
echo ">>> [Step 3] Running 5-fold CV on SciFact train for OOF threshold calibration..."
OOF_DIRECT_FILES=()
OOF_RATIONALE_FILES=()

for fold in 0 1 2 3 4; do
  # 3a. Direct expert on fold
  DIR_OUT="$OUT_BASE/oof_cv/direct_fold${fold}"
  mkdir -p "$DIR_OUT"
  if [[ ! -s "$DIR_OUT/val_predictions.jsonl" ]]; then
    echo "    Training Direct expert fold ${fold}..."
    CUDA_VISIBLE_DEVICES=0 python -m scripts.train_scifact_expert \
      --mode direct \
      --manifest "$PROTOCOL_DIR/train_manifest.jsonl" \
      --retrieval "$PROTOCOL_DIR/train_retrieval_tfidf.jsonl" \
      --corpus "$PROTOCOL_DIR/Corpus2.csv" \
      --folds "$PROTOCOL_DIR/scifact_adaptation_folds.json" \
      --fold "$fold" \
      --output "$DIR_OUT" \
      --model "$MODEL" \
      --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4 --seed 42 --device cuda \
      2>&1 | tee "$DIR_OUT/train.log"
  fi
  OOF_DIRECT_FILES+=("$DIR_OUT/val_predictions.jsonl")

  # 3b. Rationale expert on fold
  RAT_OUT="$OUT_BASE/oof_cv/rationale_fold${fold}"
  mkdir -p "$RAT_OUT"
  if [[ ! -s "$RAT_OUT/val_predictions.jsonl" ]]; then
    echo "    Training Rationale expert fold ${fold}..."
    CUDA_VISIBLE_DEVICES=0 python -m scripts.train_scifact_expert \
      --mode rationale \
      --manifest "$PROTOCOL_DIR/train_manifest.jsonl" \
      --retrieval "$PROTOCOL_DIR/train_retrieval_tfidf.jsonl" \
      --corpus "$PROTOCOL_DIR/Corpus2.csv" \
      --explanations "$PROTOCOL_DIR/train_explanations.jsonl" \
      --folds "$PROTOCOL_DIR/scifact_adaptation_folds.json" \
      --fold "$fold" \
      --output "$RAT_OUT" \
      --model "$MODEL" \
      --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4 --lambda-exp 0.25 --seed 42 --device cuda \
      2>&1 | tee "$RAT_OUT/train.log"
  fi
  OOF_RATIONALE_FILES+=("$RAT_OUT/val_predictions.jsonl")
done

# -----------------------------------------------------------------------------
# Bước 4: Full-Train Deployed Models trên toàn bộ SciFact Train (3 seeds)
# -----------------------------------------------------------------------------
echo ">>> [Step 4] Training deployed experts on full SciFact train (3 seeds)..."
DEV_DIRECT_FILES=()
DEV_RATIONALE_FILES=()

for seed in 42 13 87; do
  # 4a. Deployed Direct expert
  DEP_DIR="$OUT_BASE/deployed/direct_seed${seed}"
  mkdir -p "$DEP_DIR"
  if [[ ! -s "$DEP_DIR/val_predictions.jsonl" ]]; then
    echo "    Training deployed Direct expert seed ${seed}..."
    CUDA_VISIBLE_DEVICES=0 python -m scripts.train_scifact_expert \
      --mode direct \
      --full-train \
      --manifest "$PROTOCOL_DIR/train_manifest.jsonl" \
      --retrieval "$PROTOCOL_DIR/train_retrieval_tfidf.jsonl" \
      --corpus "$PROTOCOL_DIR/Corpus2.csv" \
      --val-manifest "$DEV_PROTOCOL/dev_manifest.jsonl" \
      --val-retrieval "$DEV_PROTOCOL/dev_retrieval_tfidf.jsonl" \
      --val-corpus "$DEV_PROTOCOL/Corpus2.csv" \
      --output "$DEP_DIR" \
      --model "$MODEL" \
      --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4 --seed "$seed" --device cuda \
      2>&1 | tee "$DEP_DIR/train.log"
  fi
  DEV_DIRECT_FILES+=("$DEP_DIR/val_predictions.jsonl")

  # 4b. Deployed Rationale expert
  DEP_RAT="$OUT_BASE/deployed/rationale_seed${seed}"
  mkdir -p "$DEP_RAT"
  if [[ ! -s "$DEP_RAT/val_predictions.jsonl" ]]; then
    echo "    Training deployed Rationale expert seed ${seed}..."
    CUDA_VISIBLE_DEVICES=0 python -m scripts.train_scifact_expert \
      --mode rationale \
      --full-train \
      --manifest "$PROTOCOL_DIR/train_manifest.jsonl" \
      --retrieval "$PROTOCOL_DIR/train_retrieval_tfidf.jsonl" \
      --corpus "$PROTOCOL_DIR/Corpus2.csv" \
      --explanations "$PROTOCOL_DIR/train_explanations.jsonl" \
      --val-manifest "$DEV_PROTOCOL/dev_manifest.jsonl" \
      --val-retrieval "$DEV_PROTOCOL/dev_retrieval_tfidf.jsonl" \
      --val-corpus "$DEV_PROTOCOL/Corpus2.csv" \
      --output "$DEP_RAT" \
      --model "$MODEL" \
      --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4 --lambda-exp 0.25 --seed "$seed" --device cuda \
      2>&1 | tee "$DEP_RAT/train.log"
  fi
  DEV_RATIONALE_FILES+=("$DEP_RAT/val_predictions.jsonl")
done

# -----------------------------------------------------------------------------
# Bước 5: Đánh giá 4 hệ thống & Kiểm định Thống kê trên SciFact Dev Set
# -----------------------------------------------------------------------------
echo ">>> [Step 5] Evaluating 4 systems on SciFact dev set..."
python -m scripts.evaluate_scifact_adaptation \
  --dev-manifest "$DEV_PROTOCOL/dev_manifest.jsonl" \
  --dev-direct-runs "${DEV_DIRECT_FILES[@]}" \
  --dev-rationale-runs "${DEV_RATIONALE_FILES[@]}" \
  --oof-direct-runs "${OOF_DIRECT_FILES[@]}" \
  --oof-rationale-runs "${OOF_RATIONALE_FILES[@]}" \
  --output "$OUT_BASE/scifact_adaptation_summary.json" \
  --markdown "$OUT_BASE/scifact_adaptation_report.md" \
  --iterations 10000 \
  --seed 2026 \
  2>&1 | tee "$OUT_BASE/evaluation.log"

echo ""
echo "======================================================================"
echo "  EXPERIMENT COMPLETE!"
echo "  Report: $OUT_BASE/scifact_adaptation_report.md"
echo "======================================================================"
cat "$OUT_BASE/scifact_adaptation_report.md"
