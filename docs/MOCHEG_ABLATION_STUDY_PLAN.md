# MOCHEG Ablation Study Plan

**Main method:** GraphCURE-B18B Dual-Expert Asymmetric Routing  
**Primary goal:** xác định thành phần nào thực sự tạo ra cải thiện, trong khi giữ nguyên protocol đánh giá và tránh test leakage.

## 1. Quy tắc thực nghiệm

- Main method của paper là **B18B**, không phải B19.
- B18A được xem là grounded-expert baseline và là một thành phần của B18B.
- Mỗi ablation chỉ thay đổi một thành phần; giữ cố định manifest, retrieval, backbone, seed, epoch, batch size và evaluation script.
- Tất cả lựa chọn hyperparameter phải thực hiện trên validation hoặc train-only OOF.
- Official test chỉ được chạy một lần sau khi cấu hình đã đóng băng.
- Không chọn seed, threshold hoặc routing policy theo kết quả official test.
- B19 được báo cáo như secondary contribution về single-model disagreement distillation.

## 2. Main B18B component ablations

Đây là nhóm bắt buộc cho bảng ablation chính của paper.

| ID | Cấu hình | Thành phần thay đổi | Mục đích |
|---|---|---|---|
| A0 | B1 direct-only | Chỉ direct verifier | Baseline text/evidence verifier |
| A1 | B18A grounded-only | Chỉ grounded verifier | Đo tác động của grounding/rationale |
| A2 | Equal two-expert ensemble | `0.5 p_direct + 0.5 p_grounded` | Kiểm tra average có thay được router không |
| A3 | Symmetric routing | Dùng cùng rule cho mọi class | Kiểm tra lợi ích có đến từ routing bất đối xứng không |
| A4 | No asymmetric NEI rule | Bỏ nhánh NEI đặc biệt | Đo riêng đóng góp của NEI deferral |
| A5 | No confidence feature | Bỏ confidence khỏi router | Kiểm tra confidence có cần thiết không |
| A6 | No entropy/margin feature | Bỏ entropy và margin | Kiểm tra uncertainty features |
| A7 | Fixed threshold | Threshold định trước, không tune validation | Kiểm tra lợi ích của validation tuning |
| A8 | **B18B full routing** | Dual experts + asymmetric router + `τ=0.49` | **Main proposed method** |
| A9 | Oracle routing | Chọn expert đúng theo gold outcome | Upper bound/diagnostic only |

### Tập tối thiểu cần chạy

Nếu tài nguyên hạn chế, chạy trước:

```text
A0, A1, A2, A4, A7, A8, A9
```

Nếu router hiện tại dùng rõ ràng confidence, entropy và margin, bổ sung:

```text
A5, A6
```

## 3. Threshold and routing sensitivity

Chạy trên validation hoặc OOF, không chọn trên test:

| Nhóm | Giá trị |
|---|---|
| Fixed threshold | `τ=0.40`, `τ=0.49`, `τ=0.50`, `τ=0.60` |
| Routing mode | no routing, equal ensemble, asymmetric routing |
| NEI policy | route only NEI, route all classes, route when confidence advantage exceeds margin |

`τ=0.49` là cấu hình confirmatory nếu nó được khóa từ validation. `τ=0.60` chỉ được ghi là exploratory nếu threshold được nhận diện từ test peak.

## 4. Retrieval/evidence ablations

Chỉ chạy nhóm này nếu paper muốn claim rằng evidence selection/retrieval là một đóng góp quan trọng:

| ID | Evidence configuration |
|---|---|
| E1 | Top-1 evidence |
| E2 | Top-3 evidence |
| E3 | Top-5 evidence |
| E4 | Không grounded explanation/evidence signal |

Giữ nguyên B18B router khi thay đổi số lượng evidence. Không quét hàng loạt TF-IDF, BM25, dense encoder và reranker nếu retrieval không phải novelty chính; khi đó chúng trở thành một project ablation riêng.

## 5. B19 secondary ablations

B19 không thay thế main B18B. Nó chứng minh rằng complementarity giữa các expert có thể được nén vào một student duy nhất.

| ID | Variant | Thành phần |
|---|---|---|
| D0 | Matched control | Gold CE, không KD |
| D1 | Naive ensemble KD | KD từ trung bình hai teacher |
| D2 | Disagreement KD | Chọn teacher đúng khi hai teacher bất đồng |
| D3 | Counterfactual-only | Chỉ counterfactual sufficiency loss |
| D4 | Full | KD + disagreement weighting + counterfactual loss |

B19-B đã xác nhận train-only OOF cho `D2`:

- Control Macro-F1: `0.660788`
- Disagreement-KD Macro-F1: `0.667607`
- Delta: `+0.006819`
- Positive folds: `4/5`
- Bootstrap `P(Δ>0)=0.9517`

## 6. Efficiency ablation

Đây là bảng quan trọng để giải thích tại sao B19 vẫn có giá trị dù B18B có thể đạt điểm test cao hơn.

| System | Models at inference | Required forward passes | Macro-F1 | Latency | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| B1 direct | 1 | 1 | report | measure | measure |
| B18A ensemble | 8 | 8 | report | measure | measure |
| B18B routing | up to 2 | 2 | report | measure | measure |
| B19 DKD | 1 | 1 | report | measure | measure |

Đo trên cùng hardware và cùng batch size:

- milliseconds/sample;
- samples/second;
- peak GPU memory;
- trainable parameters;
- số forward passes mỗi claim.

## 7. Robustness and subgroup analysis

Không nhất thiết phải train thêm model; có thể phân tích prediction files đã có.

- Seed stability: 13, 42, 87.
- Politifact vs Snopes.
- QREL available vs absent.
- Gold evidence hit vs miss.
- Claim length quartiles.
- Retrieval confidence/margin quartiles.
- Supported, Refuted và NEI riêng biệt.
- Helpful/harmful transition matrix.

Các subgroup này dùng để giải thích failure modes, không dùng để chọn lại threshold sau khi xem test.

## 8. Dataset generalization

Sau khi B18B đã đóng băng, ưu tiên một dataset ngoài MOCHEG thay vì thêm nhiều biến thể backbone:

1. Giữ nguyên checkpoint và routing policy.
2. Không tune threshold riêng trên test dataset mới nếu muốn claim zero-shot transfer.
3. Nếu cần calibration, dùng validation split riêng của dataset đó.
4. Báo cáo domain shift, label mapping và evidence availability.

## 9. Trạng thái hiện tại

### Đã có

- B1 direct baseline.
- B18A grounded expert.
- B18B full routing.
- Oracle routing.
- Threshold sensitivity cơ bản.
- B19 D0–D4 ablation.
- B19 five-fold OOF confirmation.

### Cần ưu tiên chạy

1. A2: equal two-expert ensemble.
2. A4: bỏ asymmetric NEI rule.
3. A5: bỏ confidence feature.
4. A6: bỏ entropy/margin feature.
5. E1–E3 nếu evidence selection là claim của paper.
6. Efficiency benchmark cho B1/B18A/B18B/B19.

## 10. Cấu trúc bảng trong paper

### Main performance table

- B1 direct-only.
- B18A grounded-only.
- B18B proposed routing.
- Các hệ thống SOTA cùng protocol.

### Main ablation table

- A0, A1, A2, A4, A7, A8, A9.

### Secondary B19 table

- D0, D1, D2, D3, D4.

### Efficiency table

- Số model inference.
- Latency.
- VRAM.
- Forward passes.

## 11. Tiêu chí dừng

Có thể đóng băng ablation khi đáp ứng đủ:

- A8 vượt A0 và A2 trên OOF/validation.
- A4/A5/A6 xác định rõ thành phần có đóng góp.
- Không có source hoặc subgroup lớn nào suy giảm nghiêm trọng.
- Official test chỉ được dùng cho A8 đã khóa.
- Không tiếp tục quét biến thể chỉ vì muốn tăng thêm vài phần nghìn Macro-F1.

## 12. Kết quả frozen-validation của nhóm A

Protocol: official validation chỉ dùng để đánh giá các policy đã đóng băng;
không dùng để chọn lại policy và không đọc test.

| ID | Macro-F1 | Delta so với A8 | Ghi chú |
|---|---:|---:|---|
| A0 direct-only | 0.692045 | -0.019160 | Direct expert |
| A1 grounded-only | 0.701191 | -0.010014 | Grounded expert |
| A2 equal ensemble | 0.692773 | -0.018432 | Trung bình posterior không thay thế được router |
| A3 symmetric confidence | 0.691095 | -0.020110 | Class-agnostic confidence routing |
| A4 no asymmetric NEI | 0.704436 | -0.006769 | Ablation gần nhất và mạnh nhất |
| A5 disagreement-only | 0.701191 | -0.010014 | Trùng quyết định với A1 trên tập này |
| A6 max-confidence-only | 0.691095 | -0.020110 | Trùng nhãn dự đoán với A3 trên tập này |
| A8 full asymmetric routing | **0.711205** | **0.000000** | Main method |

Kết luận chính:

- A8 là cấu hình tốt nhất trong toàn bộ nhóm đã chạy.
- A8 hơn A0 `+0.019160`, A1 `+0.010014`, A2 `+0.018432` và A4
  `+0.006769` Macro-F1.
- So sánh A8 với A4 là bằng chứng trực tiếp nhất cho đóng góp của asymmetric
  NEI deferral vì hai cấu hình dùng cùng threshold `0.49`.
- A1/A5 và A3/A6 tạo cùng nhãn dự đoán trên validation. Đây là tính tương
  đương của decision rule trên tập này, không phải hai bằng chứng độc lập.
- Bảng chính của paper nên ưu tiên A0, A1, A2, A4, A8 và A9. Đưa A3, A5,
  A6 cùng threshold sensitivity A7 vào appendix để tránh bảng chính dư thừa.
