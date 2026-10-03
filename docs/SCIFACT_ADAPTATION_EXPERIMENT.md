# SciFact Domain Adaptation Experiment (Adaptation-A) Runbook

Tài liệu hướng dẫn chi tiết thực hiện thí nghiệm **Domain Adaptation trên SciFact**, kiểm chứng nguyên lý định tuyến bất đối xứng (**Asymmetric NEI Routing / CURE AND**) khi được thích nghi đặc thù miền (domain-specific adaptation).

---

## 1. Động Lực Khoa Học & Giả Thuyết Thí Nghiệm

### 1.1. Bối cảnh từ Zero-Shot Audit
Trong thí nghiệm chuyển giao không mẫu (zero-shot transfer) trước đây trên SciFact dev ($n=300$):
- **Direct Ensemble**: Macro-F1 = `0.5283`
- **Frozen CURE AND ($\tau=0.49$)**: Macro-F1 = `0.4452` ($\Delta = -0.0831$, $p = 0.0113$)
- **Nguyên nhân cốt lõi**: Mô hình rationale-trained bị suy thoái nặng ở miền khoa học (Refuted F1 sụp từ 0.4694 xuống 0.2133). AND router do đó bị kích hoạt sai trên 68.8% các ca Refuted, đẩy sang NEI.

### 1.2. Giả Thuyết Thí Nghiệm Adaptation-A
> *"Zero-shot routing does not transfer, but the asymmetric routing principle recovers after domain-specific adaptation."*

Nếu huấn luyện cả hai chuyên gia trên dữ liệu khoa học của SciFact:
1. Chuyên gia rationale sẽ học được cách nhận diện bằng chứng khoa học và phát hiện thiếu bằng chứng (NEI) một cách chuẩn xác.
2. Ngưỡng $\tau^*$ được hiệu chuẩn lại trên tập SciFact train (thay vì áp đặt $\tau=0.49$ từ MOCHEG).
3. **Adapted CURE (AND)** sẽ vượt trội cả **Direct-only** lẫn **Direct self-deferral**.

### 1.3. Nguyên Tắc Cách Ly Biến Số (Isolation Principle)
- **GIỮ NGUYÊN**: Retriever TF-IDF top-5 (giống hệt zero-shot audit). **Không đổi retriever** cùng lúc với việc retrain mô hình, để reviewer thấy rõ hiệu quả đến từ domain adaptation chứ không phải do retriever.
- **GIỮ NGUYÊN**: Backbone Qwen3-4B-Instruct-2507, LoRA hyperparameters ($r=16, \alpha=32$), prompt template, và quy tắc AND routing.

---

## 2. Thiết Kế Giao Thức (Protocol Specification)

### 2.1. Phân Chia Dữ Liệu
- **SciFact Corpus**: 5,183 tóm tắt bài báo khoa học (`corpus.jsonl`).
- **SciFact Train Split**: ~809 claims (`claims_train.jsonl`):
  - Dùng để sinh teacher rationales (chỉ thấy nhãn train).
  - Dùng để chia 5-fold cross-validation nội bộ nhằm hiệu chuẩn ngưỡng $\tau^*_{\text{SciFact}}$ và $\tau^*_{\text{self}}$.
  - Dùng để train các final model adapters.
- **SciFact Dev Split**: 300 claims (`claims_dev.jsonl`):
  - **Giữ vai trò locked held-out test set**.
  - Tuyệt đối không dùng dev labels để chọn seed, model, hay tune $\tau$. Đánh giá 1 lần duy nhất.

### 2.2. Bốn Hệ Thống Báo Cáo Chuẩn Mực
1. **Direct-only**: Huấn luyện giám sát phân loại trực tiếp qua Cross-Entropy ($\mathcal{L}_{\text{verdict}}$).
2. **Rationale-trained-only**: Huấn luyện đa nhiệm giám sát phân loại + sinh giải trình có căn cứ ($\mathcal{L}_{\text{verdict}} + 0.25 \cdot \mathcal{L}_{\text{exp}}$).
3. **Direct Self-deferral**: Chỉ dùng Direct expert; nếu $p_{\text{direct}}(\text{NEI}) \ge \tau_{\text{self}}^*$, đổi dự đoán sang NEI.
4. **Adapted CURE (AND)**: Direct expert làm mỏ neo; chỉ đổi sang NEI khi:
   $$\arg\max p_{\text{rationale}} = \text{NEI} \quad \land \quad p_{\text{rationale}}(\text{NEI}) \ge \tau^*$$

### 2.3. Hai So Sánh Quyết Định
- **$\Delta_{\text{AND - Direct}} = \text{Macro-F1}(\text{AND}) - \text{Macro-F1}(\text{Direct})$**
- **$\Delta_{\text{AND - Self}} = \text{Macro-F1}(\text{AND}) - \text{Macro-F1}(\text{Self-deferral})$**

Cả hai so sánh phải được đo lường bằng:
- Paired bootstrap 95% Confidence Interval (10,000 iterations).
- Xác suất tăng trưởng có hướng $P(\Delta > 0)$.
- Kiểm định McNemar chính xác 2 phía cho độ chính xác phân loại.

---

## 3. Hướng Dẫn Thực Thi Trên Server (`hvtham-server`)

### Cách 1: Chạy Tự Động Toàn Bộ Bằng Script Một Lệnh (Khuyên Dùng)

```bash
cd ~/whale/GraphCURE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv
export HF_HOME=~/whale/cache/huggingface
set -euo pipefail

# Pull code mới nhất
git pull --ff-only origin feature/mocheg-phase-b18-explanation-distillation

# Chạy toàn bộ pipeline (có thể chạy ngầm bằng nohup)
nohup bash scripts/run_scifact_adaptation.sh > outputs/scifact_adaptation/pipeline.log 2>&1 &

# Theo dõi tiến độ:
tail -f outputs/scifact_adaptation/pipeline.log
```

---

### Cách 2: Chạy Từng Bước Thủ Công

#### Bước 1: Chuẩn Bị Protocol SciFact Train & 5-Fold Splits
```bash
python -m scripts.prepare_scifact_adaptation_protocol \
  --corpus data/external/scifact_raw/data/corpus.jsonl \
  --claims-train data/external/scifact_raw/data/claims_train.jsonl \
  --output-root outputs/scifact_adaptation/protocol \
  --top-k 5 \
  --folds 5 \
  --seed 2040
```
*Thời gian: ~1–2 phút.*

#### Bước 2: Sinh Structured Teacher Explanations Trên SciFact Train
```bash
python -m scripts.generate_scifact_teacher_explanations \
  --manifest outputs/scifact_adaptation/protocol/train_manifest.jsonl \
  --retrieval outputs/scifact_adaptation/protocol/train_retrieval_tfidf.jsonl \
  --corpus outputs/scifact_adaptation/protocol/Corpus2.csv \
  --output outputs/scifact_adaptation/protocol/train_explanations.jsonl \
  --summary outputs/scifact_adaptation/protocol/train_explanations_summary.json \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --top-k 5 \
  --device cuda
```
*Thời gian: ~25–35 phút (sinh cho ~809 claims trên A100).*

#### Bước 3: Huấn Luyện 5-Fold OOF Để Calibrate Ngưỡng (10 runs)
```bash
for fold in 0 1 2 3 4; do
  # 3a. Direct expert
  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_scifact_expert \
    --mode direct \
    --manifest outputs/scifact_adaptation/protocol/train_manifest.jsonl \
    --retrieval outputs/scifact_adaptation/protocol/train_retrieval_tfidf.jsonl \
    --corpus outputs/scifact_adaptation/protocol/Corpus2.csv \
    --folds outputs/scifact_adaptation/protocol/scifact_adaptation_folds.json \
    --fold "$fold" \
    --output "outputs/scifact_adaptation/oof_cv/direct_fold${fold}" \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4 --seed 42 --device cuda

  # 3b. Rationale expert
  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_scifact_expert \
    --mode rationale \
    --manifest outputs/scifact_adaptation/protocol/train_manifest.jsonl \
    --retrieval outputs/scifact_adaptation/protocol/train_retrieval_tfidf.jsonl \
    --corpus outputs/scifact_adaptation/protocol/Corpus2.csv \
    --explanations outputs/scifact_adaptation/protocol/train_explanations.jsonl \
    --folds outputs/scifact_adaptation/protocol/scifact_adaptation_folds.json \
    --fold "$fold" \
    --output "outputs/scifact_adaptation/oof_cv/rationale_fold${fold}" \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4 --lambda-exp 0.25 --seed 42 --device cuda
done
```
*Thời gian: ~10 phút/run $\times$ 10 runs $\approx$ 1.5–2 giờ.*

#### Bước 4: Huấn Luyện Final Deployed Models Trên Toàn Bộ SciFact Train (3 seeds)
```bash
for seed in 42 13 87; do
  # 4a. Deployed Direct
  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_scifact_expert \
    --mode direct \
    --full-train \
    --manifest outputs/scifact_adaptation/protocol/train_manifest.jsonl \
    --retrieval outputs/scifact_adaptation/protocol/train_retrieval_tfidf.jsonl \
    --corpus outputs/scifact_adaptation/protocol/Corpus2.csv \
    --val-manifest outputs/scifact_cure_and_zero_shot/protocol/dev_manifest.jsonl \
    --val-retrieval outputs/scifact_cure_and_zero_shot/protocol/dev_retrieval_tfidf.jsonl \
    --val-corpus outputs/scifact_cure_and_zero_shot/protocol/Corpus2.csv \
    --output "outputs/scifact_adaptation/deployed/direct_seed${seed}" \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4 --seed "$seed" --device cuda

  # 4b. Deployed Rationale
  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_scifact_expert \
    --mode rationale \
    --full-train \
    --manifest outputs/scifact_adaptation/protocol/train_manifest.jsonl \
    --retrieval outputs/scifact_adaptation/protocol/train_retrieval_tfidf.jsonl \
    --corpus outputs/scifact_adaptation/protocol/Corpus2.csv \
    --explanations outputs/scifact_adaptation/protocol/train_explanations.jsonl \
    --val-manifest outputs/scifact_cure_and_zero_shot/protocol/dev_manifest.jsonl \
    --val-retrieval outputs/scifact_cure_and_zero_shot/protocol/dev_retrieval_tfidf.jsonl \
    --val-corpus outputs/scifact_cure_and_zero_shot/protocol/Corpus2.csv \
    --output "outputs/scifact_adaptation/deployed/rationale_seed${seed}" \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4 --lambda-exp 0.25 --seed "$seed" --device cuda
done
```
*Thời gian: ~12 phút/run $\times$ 6 runs $\approx$ 1–1.2 giờ.*

#### Bước 5: Đánh Giá 4 Hệ Thống & Sinh Báo Cáo
```bash
python -m scripts.evaluate_scifact_adaptation \
  --dev-manifest outputs/scifact_cure_and_zero_shot/protocol/dev_manifest.jsonl \
  --dev-direct-runs \
    outputs/scifact_adaptation/deployed/direct_seed42/val_predictions.jsonl \
    outputs/scifact_adaptation/deployed/direct_seed13/val_predictions.jsonl \
    outputs/scifact_adaptation/deployed/direct_seed87/val_predictions.jsonl \
  --dev-rationale-runs \
    outputs/scifact_adaptation/deployed/rationale_seed42/val_predictions.jsonl \
    outputs/scifact_adaptation/deployed/rationale_seed13/val_predictions.jsonl \
    outputs/scifact_adaptation/deployed/rationale_seed87/val_predictions.jsonl \
  --oof-direct-runs outputs/scifact_adaptation/oof_cv/direct_fold*/val_predictions.jsonl \
  --oof-rationale-runs outputs/scifact_adaptation/oof_cv/rationale_fold*/val_predictions.jsonl \
  --output outputs/scifact_adaptation/scifact_adaptation_summary.json \
  --markdown outputs/scifact_adaptation/scifact_adaptation_report.md \
  --iterations 10000 \
  --seed 2026

cat outputs/scifact_adaptation/scifact_adaptation_report.md
```

---

## 4. Cấu Trúc Bảng Kết Quả Đưa Vào Bài Báo (Paper Table)

Sau khi chạy xong, bảng tổng kết trong bài báo sẽ có format 3 dòng kinh điển:

| Setting | Direct | Self-deferral | AND | $\Delta$ AND–Direct | $\Delta$ AND–Self |
|---|---:|---:|---:|---:|---:|
| **MOCHEG frozen** (In-domain) | 0.5449 | 0.5497 | **0.5547** | `+0.0098` | `+0.0050` |
| **SciFact zero-shot** (Cross-domain audit) | **0.5283** | — | 0.4452 | `−0.0831` | — |
| 🌟 **SciFact adapted** (Adaptation-A) | *[result]* | *[result]* | **[result]** | **[delta]** | **[delta]** |

### Cách Diễn Giải Kết Quả (Story Line):
1. **Nếu $\Delta_{\text{AND - Direct}} > 0$ và $\Delta_{\text{AND - Self}} > 0$:**
   > *"Zero-shot routing fails because cross-domain rationale models suffer feature corruption. However, once adapted to the target domain, the asymmetric routing principle (AND) recovers and significantly outperforms both direct verification and self-deferral, establishing that epistemic complementarity is domain-recoverable."*
2. **Nếu $\Delta_{\text{AND - Direct}} \le 0$:**
   > *"CURE routing proves highly effective in complex multimodal verification (MOCHEG) where multiple evidence perspectives are essential, but in homogeneous single-modality scientific abstract verification, direct verification remains a resilient standalone anchor."*
