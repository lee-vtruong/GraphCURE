# MOCHEG Phase B18-B: Explanation-Derived Adaptive Evidence Selector
## Decoupling Rationale Learning from Verdict Reasoning via Evidence Attribution Distillation

**Phase:** B18-B  
**Mục tiêu:** Chưng cất kiến thức giải thích có căn cứ (Grounded Rationale Distillation) thành bộ chọn bằng chứng thích ứng (Adaptive Evidence Selector), tách rời hoàn toàn việc học tính hữu ích của bằng chứng (evidence utility) khỏi việc phán quyết sự thật (verdict classification) nhằm triệt tiêu nhiễu distractor và bảo toàn 100% độ đa dạng hạt giống (ensemble diversity).  
**Trạng thái:** Sẵn sàng thực thi trên GPU Server (Ready for GPU Screen)  

---

## 1. Động Lực Khoa Học & Phân Tích Điểm Nghẽn B18-A

### 1.1. Chẩn đoán Thực nghiệm từ B18-A: Điểm Nghẽn Sụp Đổ Đa Dạng (Diversity Collapse)
Trong Phase B18-A, mô hình sinh giải thích đa nhiệm (multi-task generation: sinh giải thích từng bước + nhãn verdict) đã chứng minh rõ ràng:
- **Hiệu năng đơn hạt giống (Single Seed) tăng vượt trội:**
  - Seed 42: $+0.00878$ Macro-F1 so với matched control ($0.65231$ vs $0.64353$).
  - Seed 87: $+0.00319$ Macro-F1 so với matched control ($0.65497$ vs $0.65178$).
  - Mean Delta: $\mathbf{+0.00599 \pm 0.00280}$ (100% các seed đều tăng điểm).
- **Tuy nhiên, khi kết hợp Ensemble lại thất bại:**
  - Ở nhóm Matched Control (chỉ học nhãn verdict), các seed khác nhau có không gian lỗi độc lập, giúp ensemble tăng vọt $+2.3\%$ đạt **`0.67081`**.
  - Ở nhóm Candidate B18-A, việc bắt verifier phải học thuộc chuỗi token giải thích của giáo viên (`Qwen2.5-7B`) khiến các seed bị đồng bộ hóa (homogeneous reasoning), làm sụp đổ phương sai ngẫu nhiên giữa các seed. Kết quả là ensemble của B18-A chỉ đạt `0.65525` (thua control `-0.01555`).
- **Nhiễu từ Top-5 Retrieval thô:** Cả B1 và B18-A đều phải nhận nguyên vẹn 5 đoạn văn từ bước tìm kiếm thông tin. Nhiều đoạn trong số đó chỉ liên quan bề mặt (surface lexical overlap) hoặc hoàn toàn là distractor (đặc biệt là các bài báo đính chính trên Snopes/PolitiFact), làm loãng khả năng suy luận của verifier.

### 1.2. Giải Pháp B18-B: Phân Tách Kiến Trúc (Architectural Decoupling)
B18-B đặt câu hỏi cốt lõi:
> *"Làm thế nào để tận dụng tri thức sâu sắc của giáo viên 7B về việc 'bằng chứng nào thật sự quan trọng', nhưng không bắt verifier phải học sinh văn bản giải thích nữa để không làm mất tính đa dạng khi ensemble?"*

```
B18-A (Coupled):
Teacher Explanation ──> Supervise Verifier Parameters (Verdict + Explanation Loss)
                        └──> Verifier sinh giải thích tốt hơn ở từng seed
                        └──> Nhưng các seed học giống nhau ──> MẤT DIVERSITY ENSEMBLE!

B18-B (Decoupled):
Teacher Explanation ──> Supervise Lightweight Evidence Selector
                                     │ (Loại bỏ distractor, giữ 1-3 passages tinh túy)
                                     ▼
                              Clean Evidence
                                     │
                                     ▼
                            Direct Qwen3 Verifier (Chỉ học Verdict Loss)
                            └──> BẢO TOÀN 100% ENSEMBLE DIVERSITY!
```

---

## 2. Giả Thuyết Khoa Học (Core Hypotheses)

1. **Giả thuyết 1 (Rationale-to-Selection Distillation):** Chuỗi giải thích có cấu trúc của giáo viên chứa tín hiệu quy kết nhân quả (attribution signal) chính xác về việc đoạn văn nào là điều kiện tiên quyết để đưa ra phán quyết. Tín hiệu này có thể chưng cất hoàn hảo sang một bộ phân loại cặp câu (Cross-Encoder) nhẹ.
2. **Giả thuyết 2 (Decoupled Specialization):**
   - **Mục tiêu 1 (Selector):** Học *bằng chứng nào có ích (WHICH evidence matters)*.
   - **Mục tiêu 2 (Verifier):** Học *kết luận sự thật là gì (WHAT the verdict is)*.
   Việc chuyên môn hóa độc lập từng khối kiến trúc giúp verifier không bị quá tải nhiệm vụ.
3. **Giả thuyết 3 (Superiority over Fixed Contexts B4/B5):**
   - B4 (câu đơn lập): Mất toàn bộ ngữ cảnh xung quanh.
   - B5 (cửa sổ trượt $\pm 1$): Chỉ phục hồi ngữ cảnh dựa trên vị trí vật lý liền kề, không chọn được hai bằng chứng nằm cách xa nhau trong tài liệu.
   - B18-B: Lựa chọn theo giá trị ngữ nghĩa có điều kiện theo mệnh đề (claim-conditioned semantic utility), có thể linh hoạt nhặt Passage 1 và Passage 4 nếu cả hai cùng chứa sự thật quyết định.

---

## 3. Bản Chất Dữ Liệu Huấn Luyện của Selector

B18-A đã sinh ra 2,594 giải thích có cấu trúc được kiểm chứng `grounded = true`, chứa danh sách `key_evidence_ids`.

### Chuẩn mực Định nghĩa Nhãn (Scientific Label Wording):
Thay vì gọi một cách cực đoan là *true positive* hay *true negative*, nghiên cứu định nghĩa chuẩn mực học thuật:
- **Teacher-selected passage ($j \in \mathcal{K}_i$):** Gán nhãn **Positive Pseudo-label ($y = 1.0$)** — bằng chứng tối thiểu cốt lõi được giáo viên viện dẫn.
- **Teacher-unselected passage ($j \notin \mathcal{K}_i$):** Gán nhãn **Hard Pseudo-negative ($y = 0.0$)** — các đoạn văn nằm trong Top-5 tìm kiếm nhưng không đủ điều kiện làm căn cứ quyết định.
- **Ungrounded claims ($\text{grounded} = \text{false}$):** Gán nhãn **Distractor Negatives ($y = 0.0$)** — các đoạn văn hoàn toàn không chứng minh được mệnh đề.

---

## 4. Kiến Trúc Mô Hình Selector & Chính Sách Chọn Thích Ứng

### 4.1. Bộ Chọn Cross-Encoder Nhẹ
- Backbone: `cross-encoder/ms-marco-MiniLM-L-6-v2` (hoặc RoBERTa/DeBERTa lightweight).
- Đầu vào: Cặp `[Claim]` + `[Candidate Evidence Passage]`.
- Điểm đầu ra: $s_{ij} = f_\theta(c_i, e_{ij}) \in \mathbb{R}$.
- Hàm mất mát: Binary Cross-Entropy with Logits:
  $$\mathcal{L}_{\text{selector}} = -\frac{1}{N}\sum_{(c, e, y)} \left[ y \log \sigma(s) + (1 - y) \log (1 - \sigma(s)) \right]$$

### 4.2. Thuật Toán Lọc Thích Ứng (Adaptive Evidence Selection Policy)
Sau khi tính điểm cho các ứng viên $s_1 \ge s_2 \ge \dots \ge s_K$:
1. **Luôn giữ hạng 1:** $e_1 \in \mathcal{E}^*$ (mỏ neo thông tin mạnh nhất).
2. **Xét các hạng tiếp theo $k \in \{2, \dots, K_{\max}\}$:**
   Bổ sung $e_k$ vào $\mathcal{E}^*$ khi và chỉ khi thỏa mãn đồng thời hai điều kiện:
   $$s_k \ge \tau \quad \text{(Ngưỡng điểm tuyệt đối)} \quad \land \quad (s_1 - s_k) \le \delta \quad \text{(Biên độ tương đối so với đỉnh)}$$
3. **Ràng buộc cận số lượng:** $K_{\min} \le |\mathcal{E}^*| \le K_{\max}$ (mặc định $K_{\min}=1, K_{\max}=3$).
   - *Trường hợp bằng chứng hiển nhiên (0.95, 0.30, 0.15):* Chỉ giữ 1 đoạn duy nhất, loại bỏ hoàn toàn 4 đoạn rác.
   - *Trường hợp đa nguồn phức tạp (0.92, 0.89, 0.85, 0.20):* Giữ trọn vẹn 3 đoạn có độ tự tin cao.

---

## 5. Huấn Luyện Verifier: Giữ Trọn Vẹn 100% Diversity

Sau khi evidence đã được thanh lọc qua Selector, mô hình verifier (`Qwen3-4B-Instruct LoRA`) được huấn luyện ở chế độ **Direct Verdict Only** (`matched_control`):
- Đầu vào: `[Claim]` + `[Filtered Evidence (1-3 passages)]`
- Đầu ra: Token phán quyết duy nhất (`A` / `B` / `C` tương ứng `Supported` / `Refuted` / `NEI`).
- Hàm mất mát:
  $$\mathcal{L} = \mathcal{L}_{\text{verdict}} \quad (\text{Tuyệt đối không dùng } \lambda \mathcal{L}_{\text{explanation}})$$
- **Ý nghĩa then chốt:** Verifier quay lại bài toán phân loại đơn nhiệm, cho phép các seed huấn luyện tự do khám phá các vùng cực tiểu khác nhau trong không gian trọng số, khôi phục lại hiệu ứng triệt tiêu lỗi khi ensemble.

---

## 6. Hệ Thống Đo Lường & Đánh Giá Độc Lập Cho Selector

Để chứng minh selector thực sự học được khả năng quy kết bằng chứng (attribution) chứ không chỉ "vô tình làm input ngắn đi", hệ thống ghi nhận các metric chuyên biệt:
1. **Teacher-Key Coverage:** Tỷ lệ bằng chứng then chốt của giáo viên được selector giữ lại:
   $$\text{Coverage} = \frac{|\mathcal{E}^* \cap \mathcal{K}|}{|\mathcal{K}|}$$
2. **Attribution Precision:** Tỷ lệ bằng chứng được chọn thực sự là bằng chứng then chốt:
   $$\text{Precision} = \frac{|\mathcal{E}^* \cap \mathcal{K}|}{|\mathcal{E}^*|}$$
3. **Ranking Quality (Pseudo-label Recall@K):** Tần suất bằng chứng của giáo viên xuất hiện trong Top-1, Top-2, Top-3 xếp hạng điểm số của Cross-Encoder.
4. **Phân bố $K$ lọc được:** Thống kê tỷ lệ phần trăm các claim được co về $K=1$, $K=2$, $K=3$ và trung bình $\bar{K}$.

---

## 7. Bảng Đối Chứng & Thực Nghiệm Triệt Tiêu (Ablation Matrix)

Để loại trừ các cách giải thích đối nghịch từ phía reviewer, B18-B thiết lập ma trận kiểm định 6 dòng chuẩn mực:

| Cấu hình Thử nghiệm | Cơ chế Chọn | Thích ứng $K$? | Giám sát từ Teacher Rationale? | Macro-F1 Đơn Seed | Macro-F1 2-Seed Ensemble |
|---|:---:|:---:|:---:|:---:|:---:|
| **Control A (Original B1/B18)** | Không lọc (Top-5) | Không (Cố định 5) | Không | $0.64765$ | $0.67081$ |
| **Control B (Heuristic Top-1)** | Lấy Top-1 Retrieval | Không (Cố định 1) | Không | — | — |
| **Control C (Heuristic Top-3)** | Lấy Top-3 Retrieval | Không (Cố định 3) | Không | — | — |
| **Control D (Generic Cross-Encoder)** | Cross-Encoder MS-MARCO | Có ($1 \le K \le 3$) | **Không** (Chỉ dùng relevance gốc) | — | — |
| 🏆 **B18-B Candidate (Rationale-Distilled)** | Cross-Encoder Distilled | Có ($1 \le K \le 3$) | **Có** (Supervised by Key-Evidence) | **Mục tiêu $> 0.655$** | **Mục tiêu $\ge 0.67081$** |
| *Optional Oracle (Teacher Key-Evidence)* | Teacher Perfect Attribution | Có | Oracle Thượng tầng | Upper Bound | Upper Bound |

> **Giá trị phản biện:** Dòng *Control D* cực kỳ quan trọng. Nó chứng minh mức tăng hiệu năng xuất phát từ **tri thức giải thích của giáo viên (Rationale Supervision)** chứ không đơn thuần chỉ là việc cắm thêm một con Cross-Encoder reranker thông thường.

---

## 8. Tiêu Chuẩn Thăng Hạng Nghiêm Ngặt (Promotion Gate - Fold 0)

1. **Ensemble Benchmark:** 2-Seed Candidate Ensemble Macro-F1 $\ge \mathbf{0.67081}$ (phải vượt hoặc tối thiểu hòa mốc vô địch của Matched Control).
2. **Single-Seed Delta:** Mean $\Delta \text{Macro-F1} > 0$ so với Matched Control tương ứng.
3. **Harm Mitigation:** Số ca sửa đúng (Helpful) phải vượt trội số ca làm sai (Harmful).
4. **Cân bằng Lớp:** $\text{F1}_{\text{Supported}} \ge 0.60$ và $\text{F1}_{\text{Refuted}} \ge 0.82$.

---

## 9. Runbook Thực Thi Toàn Diện trên GPU Server (Step-by-Step GPU Runbook)

Chạy tuần tự các khối lệnh sau trên GPU server `hvtham-server`:

```bash
# -----------------------------------------------------------------------------
# Bước 0: Cập nhật mã nguồn và môi trường
# -----------------------------------------------------------------------------
cd ~/whale/GraphCURE
git fetch origin
git checkout feature/mocheg-phase-b18-explanation-distillation
git pull --ff-only origin feature/mocheg-phase-b18-explanation-distillation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv

export HF_HOME=~/whale/cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
set -euo pipefail

# -----------------------------------------------------------------------------
# Bước 1: Huấn luyện Selector từ Teacher Rationales (B18-A Explanations)
# -----------------------------------------------------------------------------
mkdir -p outputs/mocheg_b18b_selector
python -m scripts.train_mocheg_b18b_sentence_selector \
  --explanations data/processed/mocheg_b18_explanations/train_fold0_explanations.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  --output outputs/mocheg_b18b_selector \
  --base-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --epochs 3 \
  --batch-size 32 \
  --device cuda \
  2>&1 | tee outputs/mocheg_b18b_selector/train.log

# -----------------------------------------------------------------------------
# Bước 2: Lọc bằng chứng thích ứng & Đánh giá Attribution Metrics
# -----------------------------------------------------------------------------
mkdir -p outputs/mocheg_b18b_filtered_retrieval
python -m scripts.prepare_mocheg_b18b_selected_evidence \
  --selector outputs/mocheg_b18b_selector \
  --retrieval outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  --teacher-explanations data/processed/mocheg_b18_explanations/train_fold0_explanations.jsonl \
  --output outputs/mocheg_b18b_filtered_retrieval/train.jsonl \
  --summary outputs/mocheg_b18b_filtered_retrieval/summary.json \
  --policy-mode adaptive \
  --min-k 1 \
  --max-k 3 \
  --adaptive-margin 1.5 \
  --device cuda \
  2>&1 | tee outputs/mocheg_b18b_filtered_retrieval/filter.log

# -----------------------------------------------------------------------------
# Bước 3: Huấn luyện Verifier Direct-Verdict Seed 42 trên bằng chứng sạch
# -----------------------------------------------------------------------------
mkdir -p outputs/mocheg_b18b/candidate_seed42
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_b18_explanation_verifier \
  --mode matched_control \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --retrieval outputs/mocheg_b18b_filtered_retrieval/train.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  --folds data/processed/mocheg_b18_folds.json \
  --fold 0 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output outputs/mocheg_b18b/candidate_seed42 \
  --seed 42 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4 \
  --device cuda \
  2>&1 | tee outputs/mocheg_b18b/candidate_seed42/train.log

# -----------------------------------------------------------------------------
# Bước 4: Huấn luyện Verifier Direct-Verdict Seed 87 trên bằng chứng sạch
# -----------------------------------------------------------------------------
mkdir -p outputs/mocheg_b18b/candidate_seed87
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_b18_explanation_verifier \
  --mode matched_control \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --retrieval outputs/mocheg_b18b_filtered_retrieval/train.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  --folds data/processed/mocheg_b18_folds.json \
  --fold 0 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output outputs/mocheg_b18b/candidate_seed87 \
  --seed 87 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4 \
  --device cuda \
  2>&1 | tee outputs/mocheg_b18b/candidate_seed87/train.log

# -----------------------------------------------------------------------------
# Bước 5: Kiểm định Thống kê & So sánh Ensemble vs Matched Control
# -----------------------------------------------------------------------------
python -m scripts.summarize_mocheg_b18_seeds \
  --candidate-roots outputs/mocheg_b18b/candidate_seed42 outputs/mocheg_b18b/candidate_seed87 \
  --control-roots outputs/mocheg_b18/control_seed42 outputs/mocheg_b18/control_seed87 \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --output outputs/mocheg_b18b/summary_2seeds.json \
  --markdown outputs/mocheg_b18b/summary_2seeds.md \
  2>&1 | tee outputs/mocheg_b18b/summary_2seeds.log
```

---

## 10. Định Vị Đóng Góp Học Thuật Cho Bài Báo (Paper Contribution Framing)

Khi B18-B vượt qua Promotion Gate, câu chuyện học thuật của bài báo sẽ được nâng tầm mạnh mẽ:
> *"Thay vì dừng lại ở tuyên bố thông thường 'chúng tôi huấn luyện mô hình sinh giải thích', bài báo chứng minh một bước đột phá về kiến trúc: **Chuyển đổi các chuỗi suy luận tự nhiên (natural language rationales) thành tín hiệu quy kết hữu ích (evidence attribution) để huấn luyện bộ chọn bằng chứng thích ứng, tách rời việc xác định tính hữu ích của bằng chứng khỏi quá trình phán quyết sự thật.**"*
