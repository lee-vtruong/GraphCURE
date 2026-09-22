# MOCHEG Phase B19: Disagreement-Aware Counterfactual Rationale Distillation
## GraphCURE-DCRD: Chuyển Complementarity giữa hai Expert vào tham số Student

**Phase:** B19  
**Mục tiêu:** Distill tri thức từ cả hai expert (Direct Verifier B1 và Grounded Rationale B18-A) vào một verifier duy nhất thông qua heterogeneous soft-label distillation, disagreement-weighted training, và counterfactual key-evidence supervision.  
**Trạng thái:** Code sẵn sàng chạy trên GPU Server  

---

## 1. Động Lực Khoa Học

### 1.1. Kết luận từ B18-B/C/C2
- Hai expert (B1 Direct và B18-A Grounded) có tính bổ trợ nhận thức (epistemic complementarity) rõ ràng: trên Official Test ($n=2,434$), chúng bất đồng trên **490 claims (20.13%)**.
- **Oracle Router** đạt **0.61287** Macro-F1 — chứng minh tiềm năng kết hợp rất lớn.
- Nhưng **Observable Router** (confidence routing, entropy-based gating) chỉ đạt AUROC ~0.596 — không đủ để khai thác triệt để tính bổ trợ.
- **Kết luận**: Complementarity tồn tại, nhưng khó định tuyến tại inference bằng feature đơn giản.

### 1.2. Giải pháp B19
Thay vì routing tại inference, **chuyển complementarity vào tham số student** trong quá trình training:
- Inference vẫn chỉ cần: `Claim + Evidence → Qwen3 Verifier → Supported/Refuted/NEI`
- **Không cần**: router, hai expert song song, 8 model ensemble, teacher rationale tại inference.

---

## 2. Kiến Trúc Huấn Luyện B19

### 2.1. Loss Tổng Hợp
$$\mathcal{L} = \mathcal{L}_{\text{verdict}} + \lambda_{\text{KD}} \mathcal{L}_{\text{KD}} + \lambda_{\text{cf}} \mathcal{L}_{\text{cf}}$$

### 2.2. Thành phần 1: Weighted Verdict Loss
$$\mathcal{L}_{\text{verdict}} = \frac{\sum_i w_i \cdot \text{CE}(y_i, p_{\text{student}}(x_i))}{\sum_i w_i}$$
$$w_i = 1 + \alpha \cdot \mathbb{1}[\hat{y}_{\text{direct}} \neq \hat{y}_{\text{grounded}}]$$

### 2.3. Thành phần 2: Heterogeneous Soft-Label KD
$$\mathcal{L}_{\text{KD}} = T^2 \text{KL}(q_T \| p_{\text{student},T})$$

Với teacher distribution $q$ được xác định **trên từng mẫu training** (không cần gold label tại inference):
- **Direct đúng, Grounded sai** → $q = p_{\text{direct}}$
- **Grounded đúng, Direct sai** → $q = p_{\text{grounded}}$
- **Cả hai đúng** → $q = (p_{\text{direct}} + p_{\text{grounded}}) / 2$
- **Cả hai sai** → $q = \text{label-smoothed gold}$ (không distill lỗi teacher)

### 2.4. Thành phần 3: Counterfactual Key-Evidence Sufficiency
Tận dụng `key_evidence_ids` từ teacher explanations B18-A:
- $E^+$: bằng chứng đầy đủ (chứa key evidence).
- $E^-$: bằng chứng bị cắt bỏ key evidence (counterfactual ablation).
$$\mathcal{L}_{\text{cf}} = \text{mean}\left[\text{ReLU}(m - s_y(x, E^+) + s_y(x, E^-)) + \text{ReLU}(m - s_{\text{NEI}}(x, E^-) + s_{\text{NEI}}(x, E^+))\right]$$

---

## 3. Ma Trận Ablation ACL-Style (5 Variants)

| Variant | Gold CE | Soft KD | Disagreement Weighting | Counterfactual |
|---|:---:|:---:|:---:|:---:|
| `matched_control` | ✓ | | | |
| `ensemble_kd` | ✓ | ✓ | | |
| `disagreement_kd` | ✓ | ✓ | ✓ | |
| `counterfactual_only` | ✓ | | | ✓ |
| `full` | ✓ | ✓ | ✓ | ✓ |

---

## 4. Tiêu Chuẩn Promotion Gate (B19-A Fold 0)

1. B19 `full` − `matched_control` ≥ +0.005 Macro-F1
2. B19 `full` − `ensemble_kd` ≥ +0.003 Macro-F1
3. NEI F1 không giảm (so với control)
4. Bootstrap P(Δ>0) ≥ 0.90
5. Helpful > Harmful (vs control)

---

## 5. Runbook Thực Thi trên GPU Server

### Bước 0: Cập nhật mã nguồn
```bash
cd ~/whale/GraphCURE
git fetch origin
git checkout feature/mocheg-phase-b18-explanation-distillation
git pull --ff-only origin feature/mocheg-phase-b18-explanation-distillation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv

export HF_HOME=~/whale/cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
set -euo pipefail
```

### Bước 1: Xác minh tệp tiên quyết tồn tại
```bash
# Kiểm tra tất cả tệp đầu vào cần thiết
echo "=== Checking prerequisites ==="
for f in \
  data/processed/mocheg_manifest_strict/train.jsonl \
  outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
  data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  data/processed/mocheg_b18_folds.json \
  data/processed/mocheg_b18_explanations/train_fold0_explanations.jsonl \
  outputs/mocheg_b18/control_fold0/best_adapter/adapter_config.json \
  outputs/mocheg_b18/candidate_fold0/best_adapter/adapter_config.json; do
  if [ -f "$f" ]; then echo "  OK: $f"; else echo "  MISSING: $f"; fi
done
```

> **Quan trọng:** Bạn cần hai adapter đã train từ Phase B18-A:
> - `outputs/mocheg_b18/control_fold0/best_adapter` — Direct Teacher (B1 control, Fold 0)
> - `outputs/mocheg_b18/candidate_fold0/best_adapter` — Grounded Teacher (B18-A explanation candidate, Fold 0)

### Bước 2: Chấm điểm Teacher Predictions trên tập Train của Fold 0

Bước này chạy inference hai teacher đã huấn luyện trên phần **train** của Fold 0 ($n \approx 9,300$) để lấy xác suất mềm (soft probabilities) cho distillation:

```bash
mkdir -p outputs/mocheg_b19/fold_0/teachers outputs/mocheg_b19/fold_0/logs

# 2a. Direct Teacher (B1 Control Fold 0)
python -m scripts.score_mocheg_b19_teacher \
  --adapter outputs/mocheg_b18/control_fold0/best_adapter \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --retrieval outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  --folds data/processed/mocheg_b18_folds.json \
  --fold 0 \
  --subset train \
  --output outputs/mocheg_b19/fold_0/teachers/direct.jsonl \
  --summary outputs/mocheg_b19/fold_0/teachers/direct_summary.json \
  --device cuda \
  2>&1 | tee outputs/mocheg_b19/fold_0/logs/teacher_direct.log

# 2b. Grounded Teacher (B18-A Candidate Fold 0)
python -m scripts.score_mocheg_b19_teacher \
  --adapter outputs/mocheg_b18/candidate_fold0/best_adapter \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --retrieval outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  --folds data/processed/mocheg_b18_folds.json \
  --fold 0 \
  --subset train \
  --output outputs/mocheg_b19/fold_0/teachers/grounded.jsonl \
  --summary outputs/mocheg_b19/fold_0/teachers/grounded_summary.json \
  --device cuda \
  2>&1 | tee outputs/mocheg_b19/fold_0/logs/teacher_grounded.log
```

**Thời gian ước tính:** ~15-20 phút mỗi teacher trên A100/A6000 (9,300 samples × inference).

### Bước 3: Huấn luyện 5 Variants Ablation trên Fold 0

```bash
for variant in matched_control ensemble_kd disagreement_kd counterfactual_only full; do
  mkdir -p "outputs/mocheg_b19/fold_0/$variant"
  echo "=== Training variant: $variant ==="
  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_b19_distillation \
    --variant "$variant" \
    --manifest data/processed/mocheg_manifest_strict/train.jsonl \
    --retrieval outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
    --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
    --folds data/processed/mocheg_b18_folds.json \
    --fold 0 \
    --explanations data/processed/mocheg_b18_explanations/train_fold0_explanations.jsonl \
    --direct-teacher outputs/mocheg_b19/fold_0/teachers/direct.jsonl \
    --grounded-teacher outputs/mocheg_b19/fold_0/teachers/grounded.jsonl \
    --output "outputs/mocheg_b19/fold_0/$variant" \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 4 \
    --seed 42 \
    --device cuda \
    2>&1 | tee "outputs/mocheg_b19/fold_0/logs/${variant}.log"
done
```

**Thời gian ước tính:** ~40-60 phút mỗi variant trên A100 (3 epochs × ~9,300 train samples).  
**Tổng cộng 5 variants:** ~3.5-5 giờ. Có thể chạy qua đêm.

> **Lưu ý:** Script có cơ chế **resume-safe**. Nếu bị ngắt giữa chừng, chỉ cần chạy lại lệnh tương tự — nó sẽ tự động tiếp tục từ epoch cuối cùng đã lưu.

### Bước 4: Đánh giá Promotion Gate

```bash
python -m scripts.analyze_mocheg_b19_screen \
  --root outputs/mocheg_b19/fold_0 \
  --output outputs/mocheg_b19/fold_0/screen.json \
  --markdown outputs/mocheg_b19/fold_0/screen.md \
  --bootstrap-iterations 5000 \
  --seed 2026 \
  2>&1 | tee outputs/mocheg_b19/fold_0/logs/analyze.log

cat outputs/mocheg_b19/fold_0/screen.md
```

### Hoặc chạy tất cả bằng shell script tự động

```bash
bash scripts/run_mocheg_b19_fold0.sh 2>&1 | tee outputs/mocheg_b19/fold_0/pipeline.log
```

---

## 6. Cấu Trúc Tệp Đầu Ra Sau Khi Chạy Xong

```
outputs/mocheg_b19/fold_0/
├── teachers/
│   ├── direct.jsonl                  # Soft probabilities từ B1 control teacher
│   ├── direct_summary.json
│   ├── grounded.jsonl                # Soft probabilities từ B18-A explanation teacher
│   └── grounded_summary.json
├── matched_control/
│   ├── best_adapter/                 # LoRA checkpoint tốt nhất
│   ├── val_predictions.jsonl         # Predictions trên held-out Fold 0
│   └── summary.json                 # Training audit trail
├── ensemble_kd/
│   ├── best_adapter/
│   ├── val_predictions.jsonl
│   └── summary.json
├── disagreement_kd/
│   ├── best_adapter/
│   ├── val_predictions.jsonl
│   └── summary.json
├── counterfactual_only/
│   ├── best_adapter/
│   ├── val_predictions.jsonl
│   └── summary.json
├── full/
│   ├── best_adapter/
│   ├── val_predictions.jsonl
│   └── summary.json
├── screen.json                       # Promotion gate audit (JSON)
├── screen.md                         # Bảng kết quả đánh giá (Markdown)
└── logs/
    ├── teacher_direct.log
    ├── teacher_grounded.log
    ├── matched_control.log
    ├── ensemble_kd.log
    ├── disagreement_kd.log
    ├── counterfactual_only.log
    ├── full.log
    └── analyze.log
```

---

## 7. Quyết Định Sau Promotion Gate

### Nếu **PASS** (B19-A vượt qua tất cả 5 tiêu chuẩn):

**B19-B: Xác nhận Multi-Fold**
```bash
for fold in 1 2 3 4; do
  B19_ROOT="outputs/mocheg_b19/fold_${fold}" \
  B19_SEED=42 \
  bash scripts/run_mocheg_b19_fold0.sh
done
```
- Yêu cầu: ≥ 4/5 fold dương, Mean Δ ≥ +0.005, Bootstrap P(Δ>0) ≥ 0.95.

**B19-C: Multi-Seed Validation** (chỉ sau B19-B pass)
- Seeds 13, 42, 87 — So sánh B19 single model vs B1/B18-A.
- Kiểm tra B19 ensemble ≥ 0.6920 validation ensemble.

**B19-D: Official Test One-Shot** (chỉ khi mọi policy đã đóng băng)
- Mục tiêu: Single-model ≥ 0.55, 3-seed ensemble ≥ 0.558.

### Nếu **FAIL**:
- Dừng tối ưu hóa MOCHEG.
- Chuyển sang multi-dataset/generalization + paper writing.
- Giữ B18-B Routing `0.55676` làm SOTA anchor.
