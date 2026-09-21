# Báo cáo tiến độ GraphCURE: từ đề xuất ban đầu đến B18

**Mốc tổng hợp:** 21/09/2026
**Bài toán chính:** kiểm chứng thông tin đa phương thức trên MOCHEG  
**Chỉ số chính:** Accuracy và Macro-F1; retrieval dùng Recall@k và MRR  
**Trạng thái dữ liệu:** B18 đã được khóa bằng validation trước khi đánh giá P1 official/strict; B6–B17 là các ablation và failure-analysis cycles trước đó

> Báo cáo này phân biệt nghiêm ngặt bốn loại số: kết quả test chính thức, test strict đã khử trùng lặp, validation, và train-only out-of-fold (OOF). Một con số tốt trên fold phát triển không được gọi là SOTA. Gold evidence cũng không được trộn với system-retrieved evidence, và live-web không được trộn với fixed-corpus retrieval.

## 1. Tóm tắt điều hành

GraphCURE khởi đầu từ giả thuyết rằng fact-checking đa phương thức không chỉ là bài toán fusion, mà là bài toán kiểm tra một tập ràng buộc phụ thuộc gồm **semantic, entity, temporal và contextual constraints**. Hệ thống dự kiến suy luận trên đồ thị ràng buộc, ước lượng bất định/xung đột, sau đó định tuyến mẫu rủi ro thấp sang closed-corpus expert và mẫu rủi ro cao sang open-web expert.

Trong quá trình thực nghiệm, dự án đã chuyển từ các encoder nhỏ và graph fusion sang một pipeline mạnh, có kiểm soát protocol:

1. Qwen3-Embedding-4B truy hồi ứng viên trong corpus cố định.
2. Qwen3-Reranker-4B sắp hạng lại bằng relevance chuyên biệt.
3. Qwen3-4B-Instruct-2507 được LoRA để đưa ra một verdict ở claim level.
4. Các nhánh visual, hierarchical constraints, domain robustness, calibration và routing được đánh giá bằng ablation và confirmation độc lập.
5. Sau khi xác định lỗi chính là **NEI/evidence-absence**, B16 đưa vào counterfactual evidence omission với ngân sách huấn luyện không đổi.
6. B18 mở một chu kỳ mới với grounded explanation distillation, chọn ensemble trên validation và đánh giá một lần trên official/strict test.

Kết quả tốt nhất đã hợp lệ trên test chính thức hiện tại là:

| Hệ thống | Protocol | Accuracy | Macro-F1 | Trạng thái |
|---|---|---:|---:|---|
| **GraphCURE-B18A heterogeneous ensemble** | **P1 official test, n=2442** | **0.57535** | **0.55507** | **Main frozen result** |
| GraphCURE-B18A heterogeneous ensemble | P1 strict test, n=2434 | 0.57477 | 0.55383 | Robustness track |
| GraphCURE-Qwen3 B1 5-seed ensemble | P1 official test, n=2442 | 0.56798 | 0.54531 | Frozen baseline |
| B16 counterfactual curriculum | train-only fold 0, n=2327 | 0.67383 | **0.65683** | Development pass nhưng confirmation fail; đã đóng |

So với Table 3 của AMuFC arXiv v2, B18-A official cao hơn **+2.94 điểm phần trăm Accuracy** và **+1.51 điểm Macro-F1**. Hai số `0.5577/0.5560` từng được ghi là “workshop report” không xuất hiện trong paper và đã bị loại. GraphCURE-B18A hiện có official P1 point estimate cao nhất trong các hàng đã xác minh. Paired bootstrap chứng minh B18-A vượt B1 (`P(Δ>0)=0.9839`, CI `[+0.00097,+0.01850]`); chưa có paired significance test trực tiếp với AMuFC vì prediction của AMuFC không được công bố.

## 2. Mục tiêu ban đầu và kiến trúc nghiên cứu

### 2.1. Mục tiêu khoa học

Mục tiêu ban đầu gồm ba câu hỏi:

- Có thể biểu diễn claim và bằng chứng thành các ràng buộc có nghĩa—semantic, entity, temporal, contextual—thay vì fusion đặc trưng phẳng hay không?
- Có thể dùng dependency-aware reasoning để phát hiện ràng buộc nào được thỏa, bị vi phạm hoặc chưa biết, đồng thời giữ được bất định hay không?
- Có thể dùng conflict-aware uncertainty routing để chỉ kích hoạt retrieval/LLM/VLM đắt tiền khi lợi ích kỳ vọng đủ lớn hay không?

Mục tiêu hệ thống cuối cùng gồm bốn giai đoạn A–D:

| Giai đoạn | Mục tiêu | Đầu ra cần khóa |
|---|---|---|
| A | Audit protocol, leakage, ID và split | Manifest và định nghĩa so sánh hợp lệ |
| B | Xây closed-corpus expert mạnh | Kết quả P1 trên official + strict test |
| C | Xây open-web expert | Chất lượng dưới live search, tách riêng P2 |
| D | Cost-aware router | Pareto chất lượng–chi phí giữa B và C |

### 2.2. Tại sao Phase B là trọng tâm hiện tại

Nếu closed-corpus expert còn yếu, router chỉ chọn giữa hai chuyên gia chưa đủ tốt. Vì vậy dự án ưu tiên Phase B: đạt một verifier P1 mạnh và ổn định, rồi mới đóng băng expert này để sang Phase C và D. Các B1–B16 dưới đây là các nhánh con liên tiếp của Phase B, không phải bốn phase A–D độc lập.

## 3. Dataset và protocol

MOCHEG được giới thiệu tại SIGIR 2023 với 15.601 claims, 33.880 đoạn văn và 12.112 ảnh bằng chứng.[^1] Bài toán có ba nhãn: supported, refuted và not enough information (NEI).

### 3.1. Hai track dữ liệu

| Track | Train | Validation | Test | Mục đích |
|---|---:|---:|---:|---|
| Official | 11.669 | 1.490 | 2.442 | So sánh trực tiếp với paper |
| Strict deduplicated | 11.631 | 1.456 | 2.434 | Robustness, loại 38/34/8 claim trùng |

Audit strict xác nhận không còn overlap claim text giữa các split. Phân bố train là 3.810 supported, 4.542 refuted và 3.279 NEI; validation là 491/487/478; test là 816/825/793.

### 3.2. Định nghĩa bằng chứng

| Tên trong code | Định nghĩa đúng | Có dùng Internet trực tiếp? | Vai trò |
|---|---|---:|---|
| P0 `close` | Claim và claim-owned image nếu release xác định; schema hiện tại không có claim image hợp lệ | Không | Diagnostic no-retrieval |
| P1 `open_retrieved` | Claim + evidence do hệ thống truy hồi từ corpus MOCHEG cố định | Không | **Phase-B closed-corpus expert** |
| P1-oracle `open_gold_oracle` | Claim + qrel gold text/image | Không | Chẩn đoán trần, không phải kết quả chính |
| P2 open-web | Search/reformulation trên Web tại inference | Có | Phase C |

Tên legacy `open_retrieved` dễ gây nhầm: nó là fixed-corpus retrieval, không phải live-web. Toàn bộ comparison chính của Phase B dùng P1.

### 3.3. Kỷ luật thực nghiệm

- Chọn kiến trúc/hyperparameter trên validation hoặc train-only folds; không dùng test để chọn.
- Từ B6 trở đi, dùng duplicate-family-safe folds, fixed epoch, matched compute control và promotion gate đăng ký trước.
- Một screen pass phải được xác nhận trên fold chưa dùng; nếu confirmation fail thì đóng nhánh.
- Báo cáo mean ± standard deviation qua seed/fold, bootstrap CI, xác suất delta dương, McNemar, helpful/harmful corrections và source diagnostics.
- Không nới threshold sau khi nhìn kết quả.

## 4. Nền tảng trước B1: các thử nghiệm đã giúp loại bỏ hướng yếu

### 4.1. NewsCLIPpings: kiểm tra giả thuyết graph constraints

Trên NewsCLIPpings, typed graph ban đầu đạt Macro-F1 0.6304. Independent fusion đạt 0.6366; multi-view independent sau đó đạt 0.6529. R6 directional graph đạt 0.6534 ở một run, nhưng 5-seed mean chỉ 0.6495 ± 0.0073 so với 0.6494 ± 0.0040 của R3. R7 minimal intervention tăng các chỉ số causal/constraint nhưng locked-test Macro-F1 giảm 0.0057. Kết luận: graph constraints tạo tín hiệu giải thích được, nhưng chưa chuyển thành verdict gain ổn định.

### 4.2. MOCHEG encoder/fusion/router thế hệ đầu

| Nhóm thử nghiệm | Kết quả nổi bật | Kết luận |
|---|---:|---|
| MPNet claim-only | Acc 0.4606, MF1 0.4350 | Baseline closed/no-retrieval tốt nhất thời kỳ đầu |
| Dense top-1 open | 0.4339 / 0.4209 | Retrieval tốt không tự động tạo verifier tốt |
| Finetuned cross-encoder retrieval + classifier | 0.4458 / 0.4326 | Reranking tăng retrieval nhưng classifier không tận dụng đủ |
| Relation-aware classifier | 0.4577 / 0.4511 | Tăng MF1; nền cho router |
| Relation router, seed 42 | 0.4782 / 0.4609 | Tốt nhất thế hệ feature model, nhưng không ổn định |
| Relation router, 5 seed | 0.4685 ± 0.0076 / 0.4492 ± 0.0084 | Gain seed 42 không lặp lại |
| Gold-evidence naive | 0.4643 / 0.4130 | Gold evidence vẫn bị classifier xử lý sai |
| Gold class-balanced | 0.4803 / 0.4389 | Bias refuted/NEI còn lớn |

Các nhánh VLM zero-shot, NLI rules, learned top-k aggregation, pair verifier, consistency loss và learned gates cũng thất bại: VLM 500 mẫu chỉ MF1 0.2266–0.2579; NLI rule tốt nhất 0.3283; learned aggregation 0.3491; pair verifier collapse về một lớp ở 0.1662; consistency khoảng 0.4340; một số learned gate collapse gần 0 hoặc 1. Những kết quả này dẫn đến B1: thay vì thêm module nhỏ lên feature yếu, cần nâng cấp đồng thời retriever, reranker và claim-level verifier.

## 5. Diễn tiến Phase B từ B1 đến B16

> Tên gọi dưới đây chuẩn hóa các nhánh trong ledger thành một câu chuyện nghiên cứu liên tục. Một số phase có nhiều sub-experiment (ví dụ B2 visual và B2-SV; B6-A/B/C).

### B1 — Modern fixed-corpus retrieval và Qwen3 claim-level verifier

**Giả thuyết.** Bottleneck nằm ở chất lượng evidence ranking và khả năng reasoning ở claim level; thay retriever/reranker/encoder nhỏ bằng Qwen3 sẽ tạo một anchor đủ mạnh.

**Đã làm.** Dùng Qwen3-Embedding-4B cho hybrid candidate retrieval, Qwen3-Reranker-4B cho article reranking và Qwen3-4B-Instruct-2507 LoRA cho verdict. Một lỗi prompt truncation từng làm reranker có MRR 0.2896; sau sửa, validation MRR đạt 0.8864.

**Retrieval.** Trên strict test, hybrid retrieval đạt R@1/R@5/R@10/R@50 = 0.7428/0.8891/0.9256/0.9548, MRR 0.8077. Reranking nâng R@1 lên 0.8221, R@5 0.9297, R@10 0.9454 và MRR 0.8700.

**Verifier.** Cached set verifier chỉ đạt test MF1 0.4711 ± 0.0118. Qwen3 LoRA tạo bước nhảy lớn: validation 5-seed MF1 0.6748 ± 0.0085, raw ensemble 0.6920. Trên strict test, raw ensemble đạt Acc 0.5690/MF1 0.5458; trên official test đạt 0.5680/0.5453.

**So SOTA và quyết định.** B1 vượt HGTMFC 0.4861/0.4678 và AMuFC arXiv v2 0.546/0.540 theo point estimate P1. Mốc AMuFC `0.5577/0.5560` trước đây là không có nguồn và đã bị loại. Retrieval text đã gần bão hòa, nên B2 kiểm tra visual evidence và structured verification.

### B2 — Visual retrieval, multimodal expert và sufficiency verification

**Giả thuyết.** MOCHEG là multimodal; ảnh có thể sửa các trường hợp text-only không đủ, đặc biệt out-of-context claims.

**Đã làm.** Xây Qwen3-VL-Embedding-2B visual retrieval, structured descriptors bằng Qwen3-VL-2B-Instruct, direct/caption/lexical fusion và Qwen3-VL-Reranker-2B. Corpus validation có 12.267 ảnh. Caption union phục hồi thêm 74 gold images.

**Retrieval visual.** Direct conditional R@50/R@200 = 0.5967/0.6972. Candidate union direct+caption đạt conditional recall 0.7790. VLM reranker đạt conditional R@1 0.3425, R@5 0.4983, R@10 0.5503, R@100 0.7050.

**Verifier visual.** Text anchor validation MF1 0.5499; visual expert thường 0.48–0.53. Oracle chọn đúng expert có thể đạt khoảng 0.648, chứng minh complementarity tồn tại, nhưng learned gate gần random và safe fusion không tăng. Structured visual report nâng selection@1 lên 0.9471 trong tập báo cáo top-2, nhưng stance MF1 trên gold candidates chỉ 0.4111: selector tìm được ảnh, song expert chưa hiểu stance đủ tốt.

**B2-SV.** Flat verifier fold 0 đạt MF1 0.6371; sufficiency-verification đạt 0.6398 (+0.0027). Hierarchical-only đạt 0.6412 và interpolation chẩn đoán 0.6492, nhưng confirmation folds 1–4 chỉ +0.0006 ± 0.0074.

**Quyết định.** B2 không vượt anchor ổn định. Bottleneck chuyển từ retrieval sang evidence–claim stance và domain imbalance. B3 cô lập source/qrel failures trước khi huấn luyện thêm.

### B3 — Failure isolation và domain robustness

**Giả thuyết.** Snopes/Politifact và qrel-present/qrel-absent tạo domain shift; GroupDRO có thể cải thiện worst group.

**Kết quả.** Evidence-availability GroupDRO tăng qrel-absent MF1 tới +0.1004 nhưng overall giảm -0.0021. Source-DRO tăng Snopes +0.0086 nhưng giảm Politifact -0.0132; overall chỉ +0.0005.

**Quyết định.** Không có một reweighting toàn cục vừa cải thiện overall vừa an toàn theo source. Chuyển B4 sang bằng chứng granular hơn để giảm nhiễu trong article dài.

### B4 — Atomic sentence evidence

**Giả thuyết.** Article-level context chứa nhiều câu không liên quan; truy hồi sentence-level sẽ tạo packet sạch hơn cho verifier.

**Kết quả.** Validation sentence retrieval đạt R@1 0.5680, R@5 0.8324, R@8 0.8661, MRR 0.6803. Qwen3 sentence verifier đạt Acc 0.6545/MF1 0.6435, thấp hơn article anchor 0.6705 khoảng -0.0270 MF1.

**Quyết định.** Sentence đơn lẻ thiếu discourse/context. B5 gom câu thành context packet thay vì bỏ atomic retrieval hoàn toàn.

### B5 — Context packets và complementary ensemble

**Giả thuyết.** Packet quanh câu hit giữ được precision của sentence retrieval và context của article.

**Kết quả.** Validation packet retrieval R@1/R@5/R@8 = 0.5845/0.8482/0.8743, MRR 0.6896. Packet verifier đạt 0.6614/0.6486, vẫn thấp hơn anchor 0.0219 MF1. Tuy nhiên best interpolation weight 0.44 đạt MF1 0.6806, +0.0101 trên seed-42 anchor; oracle expert selection đạt 0.7492.

**Confirmation.** Qua 5 seed, paired gain article+packet là +0.00387 ± 0.00862. Raw packet ensemble MF1 0.6890 thấp hơn raw article ensemble 0.6920; promotion gate fail.

**Quyết định.** Packet có thông tin bổ sung nhưng không ổn định. B6 đưa cấu trúc sufficiency/polarity vào quá trình học thay vì chỉ ensemble hai verifier.

### B6 — Hierarchical sufficiency/polarity supervision

**Giả thuyết.** Tách “evidence có đủ không?” khỏi “nếu đủ thì support hay refute?” sẽ giúp NEI và tăng tính giải thích.

**B6 gốc.** Seed 42, auxiliary direct inference đạt Acc 0.6944/MF1 0.6863, +0.0158 so với anchor. Nhưng decomposition weight tối ưu bằng 0 và NEI-F1 gain chỉ +0.0087, không đạt gate +0.020.

**B6-A matched causal control.** 5-seed auxiliary hơn anchor +0.01283 ± 0.00773 và hơn direct compute control +0.00886 ± 0.01081. Raw auxiliary ensemble đạt MF1 **0.7080**, so với control 0.6981 và anchor 0.6920. Dù vậy bootstrap probability auxiliary > control chỉ 0.9354, dưới gate 0.95; một seed âm.

**B6-B PCGrad.** Gradient conflict thực sự tồn tại (epoch đầu khoảng 45–46%), nhưng conflict auxiliary không hơn standard auxiliary; ensemble thua control -0.0105.

**B6-C soft/severity projection.** Trên train-only fold 0: anchor 0.6411, control 0.6246, standard auxiliary 0.6190, soft-0.25 0.6213, soft-0.50 0.6215, severity 0.6085. Soft-0.50 vẫn thua anchor -0.0195.

**Quyết định.** Auxiliary supervision có tín hiệu mạnh nhưng làm hỏng anchor khi áp dụng rộng. B7 đóng băng anchor và chỉ route những trường hợp có khả năng hưởng lợi.

### B7 — Frozen-anchor selective router

**Giả thuyết.** Không fine-tune tiếp anchor; chỉ dùng expert B6-C khi confidence/disagreement thỏa điều kiện.

**Kết quả.** Route 61/2.327 mẫu (2,62%), tạo 35 helpful và 17 harmful corrections. Acc tăng 0.6571→0.6648; MF1 0.6411→0.6457 (+0.00460). McNemar p=0.0175, nhưng bootstrap probability chỉ 0.9294 và gain dưới gate +0.005.

**Quyết định.** Router có precision tốt nhưng coverage/gain quá nhỏ. B8 kiểm tra liệu lỗi chỉ là decision boundary/prior shift hay không.

### B8 — Prior-robust logit adjustment

**Giả thuyết.** Bias lớp ổn định có thể sửa bằng offset mà không cần thêm model.

**Kết quả.** Bias [supported +0.30, refuted 0, NEI +0.15] nâng MF1 0.6411→0.6432 (+0.00216), Acc +0.00258, helpful/harmful 25/19; bootstrap probability 0.7604.

**Quyết định.** Class prior không giải thích đủ lỗi. B9 kiểm tra variance theo seed và ổn định anchor trước khi thử calibration phức tạp.

### B9 — Fixed-seed anchor ensemble

**Giả thuyết.** Ensemble 13/42/87 giảm variance của LoRA anchor.

**Fold-0 screen.** Ensemble đạt MF1 0.6545 so với seed42 0.6411 (+0.01348), Acc +0.01289, bootstrap probability 0.9902; cả Politifact và Snopes tăng.

**Confirmation folds 1–4.** Delta lần lượt +0.00648, -0.00064, -0.00241, +0.01248; mean +0.00398 ± 0.00593, chỉ 2/4 fold dương. Aggregate MF1 0.66710 so với seed42 0.66322 (+0.00387), probability 0.905.

**Quyết định.** Ensemble là baseline ổn định hữu ích nhưng không đủ promotion. B10 học cách kết hợp seed theo disagreement thay vì trung bình đều.

### B10 — Nested cross-fitted disagreement calibration

**Giả thuyết.** Confidence, entropy và bất đồng seed dự báo model nào đáng tin.

**Kết quả.** Nested OOF calibrator đạt MF1 0.67054: +0.00732 so seed42 và +0.00345 so unweighted ensemble. Ba fold dương, nhưng bootstrap probability so ensemble chỉ 0.8702. Politifact giảm -0.00185 và Snopes giảm -0.00543—một dấu hiệu Simpson/mixture-prior effect.

**Quyết định.** Không promote aggregate gain khi cả source đều giảm. B11 thử condition trực tiếp theo provenance.

### B11 — Provenance-conditioned calibration

**Giả thuyết.** Calibrator riêng cho Politifact/Snopes loại bỏ mixture-prior artifact.

**Kết quả.** MF1 0.66534, thấp hơn unweighted ensemble 0.66710 và global calibrator 0.67054; Accuracy giảm 0.00806; helpful/harmful 418/493; chỉ 2 fold tăng.

**Quyết định.** Đóng nhánh calibration/stacking. B12 quay lại thay đổi representation/training trajectory từ base với constraint tasks.

### B12 — Joint-from-base constraints

**Giả thuyết.** Auxiliary constraints cần được học đồng thời từ base, không phải continuation trên anchor đã hội tụ.

**Thiết kế.** LoRA mới học verdict, sufficiency, polarity và ablation; direct-only control có đúng số optimizer updates; fold assignment mới seed 2027, fixed epoch 3.

**Kết quả.** Standard anchor MF1 0.66106, compute-matched control 0.64191, joint 0.64729. Joint hơn control +0.00538 nhưng thua anchor -0.01376; Politifact -0.02416, Snopes +0.00403; probability joint > control 0.7282.

**Quyết định.** Auxiliary task có ích tương đối, nhưng optimization dài gây catastrophic drift. B13 không đoán thêm architecture; trước tiên tạo failure atlas.

### B13 — Failure atlas, compute-neutral curriculum và frozen blend

**Chẩn đoán.** 150/207 harmful changes là các dự đoán supported/refuted đúng bị đẩy sang NEI. Sufficiency head tạo 282 false-insufficient trên 1.120 sufficient targets; polarity sai 94/320 supported targets. Extra optimization làm mất 0.01915 MF1, auxiliary chỉ phục hồi 28,1%.

**Intervention.** Epoch 1 thay một phần verdict examples bằng balanced auxiliary examples; epochs 2–3 verdict-only recovery; tổng 9.305 examples/epoch không đổi.

**Kết quả direct.** Candidate MF1 0.65139 so anchor 0.66106 (-0.00966); cả hai source đều giảm. Standalone hierarchical MF1 0.62278. Fold-0 interpolation anchor+hierarchical weight 0.11 tăng +0.00510 nhưng chỉ exploratory.

**Confirmation.** Folds 1–4 có paired delta -0.00017 ± 0.00271; aggregate blend 0.65769 so anchor 0.65790, bootstrap probability 0.4474. Nhánh hierarchical blend đóng.

**Quyết định.** Direct curriculum có pattern bất đối xứng đáng chú ý nhưng không ổn định. B14 gom toàn bộ OOF để xác định chính xác transitions và nhóm lỗi.

### B14 — Five-fold atlas và asymmetric NEI escape

**Atlas.** Trên 11.631 OOF samples, direct curriculum chỉ tăng +0.00100 MF1. Supported tăng +0.00746, refuted +0.00181 nhưng NEI giảm -0.00627. Từ anchor NEI sang determinate có 457 helpful/326 harmful; chiều ngược lại có 253 helpful/360 harmful.

**Policy.** Giữ mọi determinate anchor decision, chỉ cho candidate thay anchor khi anchor dự đoán NEI và candidate dự đoán supported/refuted.

**Kết quả.** Accuracy tăng +0.01126 và MF1 +0.00209; helpful/harmful 457/326, nhưng bootstrap probability 0.7958. Snopes +0.00575, Politifact -0.00276. Đặc biệt qrel-available +0.00611 nhưng qrel-absent -0.02068.

**Quyết định.** Evidence availability là nút thắt, nhưng qrel không được quan sát khi inference nên không thể route bằng qrel. B15 học value-of-information gate chỉ từ inference-time features.

### B15 — Cross-fitted value-of-information gate

**Giả thuyết.** Retrieval confidence/margin, claim length, entropy, disagreement và predicted auxiliary state có thể dự báo candidate sẽ helpful hay harmful.

**Kết quả.** Gate chọn 451/11.631 samples, nâng Accuracy +0.00834 và MF1 +0.00380; helpful/harmful 223/126; bootstrap CI dương và 4/5 fold dương. Nhưng value-ranking AUROC chỉ 0.5660, dưới 0.60; fold 3 âm; Politifact -0.00036; gain dưới +0.005.

**Quyết định.** Routing không giải quyết được khi expert chưa học đúng evidence absence. Nhánh anchor-vs-B13 routing đóng. B16 sửa trực tiếp training target bằng counterfactual evidence omission.

### B16 — Counterfactual evidence-omission verdict curriculum

**Giả thuyết.** Nếu bỏ gold evidence khỏi một claim supported/refuted thì target hợp lệ phải là NEI. Huấn luyện trực tiếp cặp phản thực này giúp model học sufficiency mà không cần auxiliary head dễ gây negative transfer.

**Thiết kế.** Trên fold assignment mới seed 2039, 15% verdict examples epoch 1 được thay bằng counterfactual thiếu evidence và gán token C/NEI. Epochs 2–3 verdict-only recovery. Epoch size 9.304 và số update không đổi. So sánh với fresh standard anchor và matched verdict-only control, fixed epoch 3.

**Fold-0 result.** Candidate đạt Acc 0.67383/MF1 0.65683, so với anchor 0.66094/0.64029 và control 0.65320/0.63488. Delta MF1 là +0.01654 so anchor và +0.02195 so control. Class-F1 đều tăng: supported +0.01115, refuted +0.00875, NEI +0.02973. Politifact +0.01980 và Snopes +0.01212. Bootstrap probability là 0.9828/0.9976; mọi gate pass.

**Trạng thái sau confirmation.** B16 không tái lập được fold-0 gain và đã đóng. Trên folds 1–4, candidate đạt MF1 0.66517, so với anchor 0.66051 (+0.00467) nhưng thấp hơn matched control 0.66621 (-0.00104). Mean fold delta là +0.00461 ± 0.00647 so anchor và -0.00115 ± 0.00469 so control; bootstrap probability lần lượt 0.8898 và 0.3814. Chỉ 2/4 folds hơn control và NEI-F1 giảm 0.00202 so anchor. Không mở official validation/test. B17 được đăng ký như một failure atlas diagnostic-only với matched control là causal baseline chính.

### B17 — B16 confirmation failure atlas

**Mục tiêu.** Xác định B16 thất bại ở đâu bằng held predictions của folds 1–4,
với matched direct control là causal baseline; không huấn luyện, không chọn
threshold và không đọc validation/test.

**Kết quả.** Candidate/control đạt MF1 0.66517/0.66621, tức -0.00104; có
550/561 helpful/harmful corrections. Class-F1 so với control thay đổi
+0.00140 supported, -0.00029 refuted và -0.00422 NEI. Harm mạnh nhất ở retrieval
confidence/margin q1 (-0.01193), control confidence 0.70–0.90 (-0.01060) và
gold rank 2–5 (-0.00992). Cả Politifact (-0.00057) lẫn Snopes (-0.00046) đều
âm nhẹ.

**Kết luận.** Các chuyển đổi supported↔NEI gần đối xứng, không chứng minh model
học được sufficiency. Không có subgroup ổn định đủ mạnh để đăng ký B18; B16/B17
đóng và B1 Qwen3 five-seed ensemble được giữ làm frozen Phase-B expert.

## 6. Bảng tóm tắt B1–B17

| Phase | Best signal | So với anchor/control | Kết luận | Lý do sang phase kế |
|---|---|---|---|---|
| B1 | Official test 0.5680/0.5453 | Vượt HGTMFC và AMuFC-v2 point estimate | Anchor P1 mạnh | Text retrieval gần bão hòa; thử visual |
| B2 | Oracle router ~0.648; B2-SV +0.0027 | Confirmation +0.0006 ± 0.0074 | Fail | Stance visual/domain là bottleneck |
| B3 | qrel-absent +0.1004 | Overall -0.0021 | Fail | Reweighting không source-safe |
| B4 | Sentence retrieval R@8 0.8661 | Verifier -0.0270 MF1 | Fail | Thiếu discourse context |
| B5 | Interpolation +0.0101 | 5-seed +0.0039 ± 0.0086 | Fail | Complementary nhưng bất ổn |
| B6 | Aux ensemble MF1 0.7080 val | Control gain +0.0099, P=0.9354 | Fail gate | Negative transfer |
| B7 | Router +0.00460 | P=0.9294 | Promising, not promoted | Kiểm decision bias |
| B8 | Logit bias +0.00216 | P=0.7604 | Fail | Không chỉ prior shift |
| B9 | Fold0 +0.01348 | Confirmation +0.00398 | Fail | Học disagreement |
| B10 | OOF MF1 0.67054 | +0.00345 vs ensemble | Fail source/bootstrap | Condition provenance |
| B11 | MF1 0.66534 | -0.00176 vs ensemble | Fail | Quay lại training objective |
| B12 | +0.00538 vs control | -0.01376 vs anchor | Fail | Cần failure atlas |
| B13 | Hier blend fold0 +0.00510 | Confirmation -0.00017 | Fail | Audit 5-fold transition |
| B14 | NEI escape Acc +0.01126 | MF1 +0.00209 | Fail | Học VOI từ observable features |
| B15 | Gate MF1 +0.00380 | AUROC 0.566 | Fail | Sửa expert, không sửa router |
| B16 | Fold0 MF1 +0.01654 | Confirmation -0.00104 vs control | **Confirmation fail; closed** | B17 phân tích cơ chế lỗi |
| B17 | Candidate/control 0.66517/0.66621 | NEI -0.00422; help/harm 550/561 | **Diagnostic complete; closed** | Đóng băng B1; chuyển Phase C |

## 7. Ablation và bài học kỹ thuật

### 7.1. Retrieval không còn là bottleneck text chính

Qwen3 reranker strict-test R@1 0.8221 và R@10 0.9454. Trong khi đó cached verifier chỉ MF1 0.4711. Khoảng cách này chứng minh “gold evidence có trong top-k” không đồng nghĩa “verdict đúng”. Vì vậy các bước sau ưu tiên claim-level reasoning và sufficiency thay vì tiếp tục tối ưu Recall@50.

### 7.2. Visual bottleneck gồm stance và utility, không chỉ retrieval

Caption/direct union tăng conditional candidate recall tới 0.7790; visual reranker tăng conditional R@10 lên 0.5503. Tuy nhiên stance MF1 trên gold-containing cases còn thấp và learned router gần random. Visual expert có oracle complementarity cao nhưng utility không dự báo được. Đây là lý do visual không được ép fusion vào current best.

### 7.3. Auxiliary supervision có tín hiệu nhưng dễ negative transfer

B6-A raw ensemble 0.7080 validation cho thấy sufficiency/polarity hữu ích. B12 cũng hơn matched control +0.00538. Nhưng full multitask training làm anchor drift, còn PCGrad/soft projection không khắc phục ổn định. B16 thành công hơn vì chuyển constraint thành **counterfactual verdict supervision cùng output space**, tránh auxiliary inference head.

### 7.4. Confirmation đã ngăn nhiều false discoveries

B9 fold0 +0.01348 nhưng confirmation chỉ +0.00398; B13 blend fold0 +0.00510 nhưng confirmation âm; B6-A có 5-seed mean tốt nhưng bootstrap gate fail. Nếu mở test ngay sau screen, nghiên cứu dễ overfit protocol. Việc khóa test và dùng fresh folds là một đóng góp phương pháp luận quan trọng.

### 7.5. NEI là lớp quyết định

Anchor mạnh ở refuted nhưng yếu hơn ở NEI. B12/B13 thường chuyển determinate đúng sang NEI; B14 lại over-escape NEI khi qrel absent. Trên development fold, B16 là lần đầu tăng đồng thời supported, refuted và NEI; tuy nhiên confirmation cho thấy NEI-F1 lại giảm và candidate không hơn matched control, nên cơ chế này không được xác nhận.

## 8. Định vị với 10 kết quả liên quan nhất

**Kết luận sau audit nguồn ngày 2026-09-14:** GraphCURE hiện có thể được mô tả là **P1 fixed-corpus/system-retrieved point-estimate SOTA trên official MOCHEG test** trong các kết quả đã xác minh. Đây không phải claim “SOTA trên mọi thiết lập MOCHEG”: gold evidence, claim–image domain generalization, filtered test và dynamic Web là các protocol khác.

Không có leaderboard MOCHEG duy nhất hoàn toàn đồng nhất. Paper khác nhau về official/filtered split, gold/system evidence, text-only/multimodal, retriever và cách gọi F1. Vì vậy bảng dưới là **audit 10 system rows liên quan**, không phải một leaderboard đồng nhất. Cột F1 giữ đúng tên metric của nguồn; chỉ những dòng ghi rõ Macro-F1 mới được dùng cho claim Macro-F1 trực tiếp.

“Rank” trong bảng dùng **CORE 2023** cho hội nghị (A* cao hơn A), và **JCR/SJR quartile** cho tạp chí. Workshop, preprint và công trình chưa nộp không thừa hưởng rank của hội nghị mẹ; chúng được ghi “không xếp hạng”.[^11] Vì hai hệ thống có thể xuất phát từ cùng một paper, đây là bảng **10 kết quả/system rows**, không phải 10 paper độc lập.

| Hạng tham khảo | Method | Năm | Hội nghị/tạp chí | Rank/uy tín venue | Accuracy | F1/Macro-F1 | Khả năng so trực tiếp |
|---:|---|---:|---|---|---:|---:|---|
| 1 | **GraphCURE-B18A heterogeneous ensemble** | 2026 | Chưa nộp; kết quả nghiên cứu nội bộ | Chưa xếp hạng | **0.57535** | **MF1 0.55507** | **P1 official n=2442; validation-selected composition** |
| 2 | AMuFC arXiv v2 | 2026 | arXiv preprint | Preprint, chưa peer review/xếp hạng | 0.546 | MF1 0.540 | P1 retrieved multimodal; đối thủ trực tiếp mạnh nhất đã xác minh[^6] |
| 3 | M-RAV, Qwen2.5-32B | 2026 | Information Processing & Management | **Q1** JCR/SJR[^12] | 0.5002 | MF1 0.5014 | System evidence nhưng test MOCHEG lọc còn n=2001; không xếp trực tiếp[^9] |
| 4 | MEVER | 2026 | EACL 2026, long paper | **CORE A** | 0.483 ± 0.021 | MF1 0.497 ± 0.012 | Retrieved evidence nhưng paper dùng preprocessing thống nhất riêng; gần P1, không đồng nhất tuyệt đối[^7] |
| 5 | MetaSumPerceiver | 2024 | ACL 2024, long paper | **CORE A\*** | — | F-score 0.486 | System text+image evidence; nguồn không gọi đây là Macro-F1[^2] |
| 6 | CMSA Top-15 | 2026 | Journal of Computer Applications (计算机应用) | Tạp chí Trung Quốc; không có CORE, chưa xác minh JCR/SJR | — | F1 0.4828 | Official n=2442, retrieved multimodal Top-15; nguồn không ghi rõ averaging[^8] |
| 7 | HGTMFC multimodal | 2025 | AAAI 2025 | **CORE A\*** | 0.4861 | F1 0.4678 | Official n=2442, retrieved text+image[^5] |
| 8 | LVLM4FV multimodal | 2024 | CIKM 2024 | **CORE A** | 0.451 | MF1 ≈0.450 | Paper gốc báo micro-F1 0.451; 0.450 được tính từ ba class-F1 và được AMuFC Table 3 ghi lại[^3] |
| 9 | MOCHEG paper gốc | 2023 | SIGIR 2023 | **CORE A\*** | — | F-score 0.4406 | System text+image evidence; paper gốc chỉ báo F-score trong Table 4[^1] |
| 10 | MOCHEG do HGTMFC tái chạy | 2025 | AAAI 2025 | **CORE A\*** | 0.4562 | F1 0.4384 | Không phải số do paper SIGIR gốc tự báo; là baseline trong HGTMFC Table 1[^5] |

GraphCURE strict robustness đạt `0.5690/0.5458` trên n=2434 nhưng không được xếp như một hàng cạnh tranh riêng, vì đó là robustness split nội bộ chứ không phải official n=2442.

### 8.1. Các con số mạnh nhưng không được trộn vào bảng P1

| Method | Reported result | Vì sao không so trực tiếp |
|---|---:|---|
| AMuFC gold evidence | Acc 0.612 / MF1 0.600 | Oracle evidence |
| Entailed Opinion/TBE-3 | MF1 0.57 | Claim–image/text domain-generalization setup, không dùng P1 system retrieval[^13] |
| HGTMFC gold evidence | 0.5405 / 0.5203 | Oracle evidence |
| LVLM4FV gold evidence | ~0.534 / ~0.535 | Oracle evidence |
| MetaSumPerceiver gold/summary setting | Acc 0.556 / system-evidence F-score 0.486 | Hai số thuộc hai bảng/setting khác nhau, không được ghép thành một cặp Acc/MF1 |
| DEFAME | Acc 0.592 | Dynamic open-web P2; không báo cùng MF1[^4] |
| Knowledge-transfer verifier | MOCHEG F1 tới khoảng 0.65 | Transfer/no-evidence setup và có thảo luận contamination; không phải P1 retrieved-evidence[^10] |

### 8.2. Kết luận định vị hiện tại

- **Accuracy:** GraphCURE official 0.5680 cao hơn AMuFC-v2 0.546 khoảng 2,20 điểm phần trăm.
- **Macro-F1:** GraphCURE 0.5453 cao hơn AMuFC-v2 0.540 khoảng 0,53 điểm. Đây là point estimate cao nhất trong các hàng P1 đã xác minh, không phải kiểm định statistical superiority.
- **So với MOCHEG do HGTMFC tái chạy:** +11,18 điểm Accuracy và +10,69 điểm F1 tuyệt đối. Không gọi `0.4562/0.4384` là cặp số từ paper SIGIR gốc.
- **B16:** development fold tăng +1,65 điểm nhưng confirmation chỉ đạt +0,47 điểm so anchor và **-0,10 điểm so matched control**; nhánh đã đóng và không được dùng để tuyên bố SOTA.

## 9. Kết quả tốt nhất hiện tại theo từng tầng bằng chứng

| Tầng | Hệ thống | Kết quả | Có thể dùng để claim gì? |
|---|---|---|---|
| Official test | B1 GraphCURE-Qwen3 ensemble | Acc 0.5680, MF1 0.5453 | Main P1 test result hiện tại |
| Strict test | B1 GraphCURE-Qwen3 ensemble | 0.5690, 0.5458 | Robustness sau dedup |
| Official validation | B6-A auxiliary ensemble | MF1 0.7080 | Development/ablation, không phải test SOTA |
| Train-only OOF aggregate | B10 crossfit calibrator | MF1 0.6705 | Negative/diagnostic result vì gate fail |
| Fresh train-only screen | B16 fold 0 | Acc 0.6738, MF1 0.6568 | Screen-only; sau đó confirmation fail |
| Oracle diagnostic | B5 expert selector | MF1 0.7492 | Trần complementarity, không deployable |

## 10. Trạng thái so với proposal gốc

| Thành phần proposal | Đã làm được | Còn thiếu |
|---|---|---|
| Multimodal constraint encoder | Text, image, metadata-like descriptors, sufficiency/polarity tasks, counterfactual absence | Entity/temporal parsers chưa thành expert mạnh end-to-end |
| Dependency-aware reasoning | Typed graph trên NewsCLIPpings; hierarchical sufficiency/polarity; evidence-set attention | Chưa có graph reasoning vượt flat/Qwen anchor ổn định |
| Conflict-aware uncertainty | Entropy/confidence/conflict, PCGrad, crossfit value gate, source/group audits | Gate utility AUROC còn thấp; chưa đủ cho production routing |
| Closed-corpus verdict | B18-A grounded explanation distillation + heterogeneous ensemble; official 0.57535/0.55507 | **Phase-B champion đã đóng băng; B1 là baseline, B2–B17 là ablation/failure analysis** |
| Open-web verification | Mới ở mức research/protocol definition | Chưa xây/freeze Phase-C expert |
| Cost-aware routing | Budget routers và selective routers đã thử | Chưa đo Pareto B-vs-C vì C chưa tồn tại |
| Explanation | B18-A dùng structured grounded teacher explanations trong training; direct-verdict inference không đổi | Cần human/explanation-quality evaluation nếu paper claim chất lượng explanation đầu ra |

## 11. Việc cần làm tiếp

### 11.1. B17 đã hoàn tất và chu kỳ B16/B17 đã được đóng

B16 đã fail independent confirmation. B17 không huấn luyện model và không chọn threshold; nó đọc duy nhất held predictions của folds 1–4 để so B16 với **matched control**. Báo cáo:

- transition nào tạo helpful/harmful cases so với matched control;
- lỗi tập trung theo fold, class, source, qrel/retrieval status hay confidence;
- thay đổi class-F1, đặc biệt NEI;
- kiểm toán exposure của counterfactual curriculum trên từng fold;
- các tương tác source×qrel, qrel×label và eligibility×retrieval.

Atlas cho thấy B16 thấp hơn matched control 0.00104 MF1, NEI-F1 giảm 0.00422,
và cả hai nguồn đều âm nhẹ. Harm tập trung ở retrieval confidence/margin thấp,
nhưng các chuyển đổi supported↔NEI gần đối xứng và không tạo ra một treatment
subgroup ổn định. Vì vậy không tune omission ratio/router trên cùng dữ liệu và
không đăng ký một B18 hậu nghiệm từ chính các subgroup của atlas đó.

### 11.2. B18 là một fresh hypothesis cycle độc lập

B18 không tiếp tục tune threshold/subgroup của B17. Nó kiểm tra một giả thuyết
mới: teacher-distilled structured explanations có thể cải thiện representation
cho direct verdict hay không. Ensemble membership được chọn trên validation;
main official `n=2442` chỉ được dùng sau khi policy đã khóa. Kết quả:

- seed 100: Accuracy `0.56962`, Macro-F1 `0.55128`;
- ensemble 5 B1 + top-3 B18-A: Accuracy `0.57535`, Macro-F1 `0.55507`;
- delta so B1: `+0.00737` Accuracy, `+0.00976` Macro-F1;
- paired bootstrap: CI `[+0.00097,+0.01850]`, `P(Δ>0)=0.9839`.

### 11.3. Đóng băng Phase B

Phase B được đóng băng tại B18-A heterogeneous ensemble: official
Acc `0.57535`/MF1 `0.55507`; strict Acc `0.57477`/MF1 `0.55383`. Đây là
point-estimate SOTA trong các hàng P1 đã xác minh. B1 được giữ làm frozen
baseline và B2–B17 làm ablation/failure analysis.

### 11.4. Sau Phase B

- **Phase C:** xây open-web expert với query decomposition, evidence provenance/time, source diversity, contradiction-aware evidence table và MLLM judge; freeze cost/time/token accounting.
- **Phase D:** route bằng expected value of information: lợi ích xác suất của open-web trừ latency/GPU/token/search cost. So Pareto curves ở fixed coverage/budget, không chỉ một threshold.
- **Paper:** main table chỉ official test; strict test là robustness; validation/train-only cho ablation; gold evidence là oracle; mỗi failed branch được rút gọn thành ablation/failure analysis thay vì kể như model variant ngang hàng.

## 12. Đóng góp có thể viết thành paper

1. **Protocol contribution:** phân tách P0/P1/P2/gold rõ ràng và deduplicated robustness track.
2. **Strong closed-corpus pipeline:** modern dense retrieval + reranking + grounded explanation distillation + heterogeneous verifier ensemble.
3. **Evidence-absence finding:** retrieval recall cao nhưng verifier thất bại chủ yếu ở sufficiency/NEI, không phải top-k coverage.
4. **Counterfactual verdict curriculum (negative finding):** biến constraint “evidence missing ⇒ NEI” thành supervision trong cùng verdict space và compute-neutral, nhưng independent confirmation cho thấy nó không hơn matched control; đây là bằng chứng để không tiếp tục tune omission ratio.
5. **Negative-results discipline:** visual fusion, graph constraints, GroupDRO, PCGrad, calibration và routing được kiểm tra bằng matched controls/fresh confirmations, làm rõ cái gì không tổng quát hóa.
6. **Cost-aware roadmap:** closed-corpus expert là nhánh rẻ; open-web chỉ dành cho high-risk samples sau khi Phase C/D hoàn tất.

## 13. Hạn chế

- B16 chỉ pass development fold và đã co về -0.00104 MF1 so matched control khi confirmation; đây là negative result đã đóng.
- Main B18-A vẫn là text-retrieved, chưa hiện thực đầy đủ multimodal promise của proposal.
- Một số paper dùng filtered splits hoặc evidence setup không đồng nhất; bảng 10 hệ thống là audit định vị, không phải leaderboard chính thức.
- Các VLM visual reranker có chi phí rất cao (ước tính khoảng 25 GPU-hours cho một full validation configuration) nhưng stance gain thấp.
- Official MOCHEG có thể chứa cross-split duplicate texts; cần báo song song official và strict, không chọn một track thuận lợi.
- Các bảng SOTA phải tiếp tục được đối chiếu bằng nguồn sơ cấp; hàng AMuFC workshop `0.5577/0.5560` đã bị xóa vì không có nguồn xác minh.

## 14. Kết luận

GraphCURE đã đi từ graph-feature models khoảng MF1 0.42–0.46 đến B1 official
MF1 `0.54531`, rồi đạt B18-A official MF1 `0.55507` và Accuracy `0.57535`.
B18-A cải thiện đặc biệt ở NEI/sufficiency nhờ grounded explanation
distillation, còn heterogeneous ensemble giữ lại thế mạnh bổ sung của B1.

Vì vậy trạng thái khoa học hiện tại là: **Phase B đã đóng băng tại B18-A và có
P1 official-test point estimate cao nhất trong các nguồn đã xác minh.** Có thể
claim paired superiority so với B1, nhưng không claim paired statistical
superiority với AMuFC khi chưa có prediction của AMuFC. B2–B17 được giữ như
ablation/negative results; bước chính tiếp theo là Phase C open-web và Phase D
cost-aware routing.

## Nguồn tham khảo

[^1]: B. M. Yao et al., “End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models,” SIGIR 2023. [arXiv:2205.12487](https://arxiv.org/abs/2205.12487).
[^2]: T.-C. Chen, C.-W. Tang, and C. Thomas, “MetaSumPerceiver: Multimodal Multi-Document Evidence Summarization for Fact-Checking,” ACL 2024. [ACL Anthology](https://aclanthology.org/2024.acl-long.474/).
[^3]: S. Zhang et al., “Multimodal Misinformation Detection using Large Vision-Language Models,” 2024. [arXiv:2407.14321](https://arxiv.org/abs/2407.14321).
[^4]: “DEFAME: Dynamic Evidence-based Fact-checking with Multimodal Experts,” open-web fact-checking paper. [OpenReview PDF](https://openreview.net/pdf/96cc682d9e23d510917c6871fb57481fa6e676e9.pdf).
[^5]: H. Pang et al., “Beyond Text: Fine-Grained Multi-Modal Fact Verification with Hypergraph Transformers,” AAAI 2025. [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/32684).
[^6]: “AMuFC: Adaptive Multimodal Fact-Checking,” 2026. [arXiv:2604.04692](https://arxiv.org/abs/2604.04692); [OpenReview](https://openreview.net/forum?id=IPGgVvGPwQ).
[^7]: “MEVER: Multi-Modal and Explainable Claim Verification with Graph-based Evidence Retrieval,” EACL 2026. [ACL Anthology](https://aclanthology.org/2026.eacl-long.242/).
[^8]: “Multimodal Fact Verification with Cross-modal Semantic Association,” Journal of Computer Applications, 2026, 46(4):1069–1076. [Journal page](https://www.joca.cn/EN/abstract/abstract27447.shtml).
[^9]: “M-RAV: Multimodal Retrieval-Augmented Verification,” Information Processing & Management, 2026. [DOI:10.1016/j.ipm.2026.104988](https://doi.org/10.1016/j.ipm.2026.104988).
[^10]: M. Singhal et al., “How to Train Your Fact Verifier: Knowledge Transfer with Multimodal Open Models,” Findings of EMNLP 2024. [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.764/).
[^11]: ICORE/CORE, “CORE 2023 Conference Rankings.” AAAI, ACL và SIGIR được xếp A*; CIKM và EACL được xếp A trong hệ quy chiếu sử dụng cho báo cáo này. [ICORE Conference Portal](https://portal.core.edu.au/conf-ranks/?by=all&page=1&search=&sort=arank&source=CORE2023).
[^12]: Information Processing & Management được ghi nhận ở Q1 theo cả JCR và SJR 2025. [Journal ranking record](https://www.iit.comillas.edu/publicacion/info_revista/en/659/Information_Processing_%26_Management).
[^13]: G. Kumar et al., “Entailed Opinion Matters: Improving the Fact-Checking Performance of Language Models by Relying on their Entailment Ability,” arXiv:2505.15050v5, Table 8. [arXiv PDF](https://arxiv.org/pdf/2505.15050).

## Nguồn nội bộ tái lập

- [Protocol definitions](PROTOCOLS.md)
- [Server result ledger](SERVER_RESULTS.md)
- [Full experiment registry](RESULTS.md)
- [Protocol-aware SOTA comparison](MOCHEG_SOTA_COMPARISON.md)
- [B16 frozen protocol](MOCHEG_PHASE_B16_COUNTERFACTUAL_VERDICT.md)
- [B17 B16-confirmation failure atlas](MOCHEG_PHASE_B17_B16_FAILURE_ATLAS.md)
