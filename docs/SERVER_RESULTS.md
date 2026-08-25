# Server results snapshot

Generated: 2026-08-05T15:23:27.157199+00:00

Git commit: `86616c3aa87237666212961e134ea8bf49bab642`

PyTorch: `2.13.0+cu130`; CUDA runtime: `13.0`

## Phase B-v4 atomic-evidence preregistration (2026-08-25)

- Source-only DRO was rejected: fold-0 overall Macro-F1 changed from `0.6371`
  to `0.6376`; Snopes improved only `+0.0086`, Politifact fell `-0.0132`, and
  evidence-absent claims fell from `0.5182` to `0.4821`.
- Evidence-availability GroupDRO was also rejected despite its large
  qrel-absent gain because overall Macro-F1 changed by `-0.0021` and performance
  moved away from evidence-available claims.
- The next registered branch is B-v4 claim-conditioned atomic evidence. It
  changes only evidence representation: frozen Qwen3 article candidates are
  expanded to official/synthetic sentence units, ranked with dense, lexical,
  and parent-rank views, diversity packed, then passed to the unchanged
  Qwen3-LoRA verifier.
- Validation/test gold injection is forbidden. The seed-42 promotion gate is
  `>=0.6735` validation Macro-F1 versus the prior `0.670471`; no new test access
  is permitted before configuration freeze. See
  `docs/MOCHEG_PHASE_B4_ATOMIC.md`.

## R2V-SV-CV-01 — GraphCURE-SV fold-0 screen (2026-08-23)

- Protocol: duplicate-family-safe fold `0`, constructed exclusively from the
  strict MOCHEG training split. Neither official validation nor test was used.
- Flat three-class control: accuracy `0.6639`, Macro-F1 `0.6371`.
  Per-class F1 was Supported `0.6492`, Refuted `0.8085`, and NEI `0.4536`.
- GraphCURE-SV (`lambda_s=0.5`, `lambda_p=0.25`, counterfactual ratio `0.5`,
  counterfactual weight `0.35`, square-root class balancing): accuracy
  `0.6506`, Macro-F1 `0.6398`. Per-class F1 was Supported `0.6082`, Refuted
  `0.7984`, and NEI `0.5127`.
- Delta versus the matched control: accuracy `-0.0133`, Macro-F1 `+0.0027`,
  Supported F1 `-0.0410`, Refuted F1 `-0.0101`, and NEI F1 `+0.0591`.
- Diagnosis: the sufficiency/counterfactual objective successfully addresses
  the NEI bottleneck but over-routes sufficient examples to NEI. Predicted NEI
  count increased from `486` to `795` for `656` gold NEI examples.
- Decision: the preregistered one-fold promotion threshold (`+0.01` Macro-F1)
  failed. Do not run five-fold SV or touch official test. Screen two
  train-only ablations next: hierarchical-only, then a light counterfactual
  variant without class balancing. This isolates the source of the NEI gain
  while reducing the Supported penalty.

### R2V-SV-CV-02 — hierarchical-only recovery

- The three training epochs completed, but the final fold evaluation was
  interrupted by a client power loss. Evaluation-only recovery loaded the
  atomically saved epoch-3 adapter; no optimizer step or retraining occurred.
- Configuration: sufficiency weight `0.25`, polarity weight `0.5`, no
  counterfactual rows, and no class balancing.
- Accuracy `0.657069`; Macro-F1 `0.641182` (`+0.0041` versus Flat and `+0.0014`
  versus the original SV screen).
- Per-class F1: Supported `0.613315`, Refuted `0.811706`, NEI `0.498525`.
  The prediction histogram contains `725` Supported, `902` Refuted, and `700`
  NEI predictions for gold counts `762 / 909 / 656`.
- Decision: hierarchical supervision contains useful NEI ranking signal, but
  still exceeds the sufficient-to-NEI error budget and fails the `+0.01`
  fold-0 gate. Before another training run, test whether the saved Flat and
  hierarchical probability distributions are complementary and whether a
  train-only NEI logit offset can recover Supported errors. Counterfactual
  training remains paused until that diagnostic passes.

### R2V-SV-CV-03 — fold-0 complementarity diagnostic

- Flat and hierarchical predictions agreed on `1391` correct and `644` wrong
  examples. Flat alone was correct on `154`; hierarchical alone on `138`; the
  models disagreed on `346` of `2327` held-out samples.
- A fold-0-selected probability interpolation with frozen hierarchical weight
  `0.63` reached accuracy `0.665664` and Macro-F1 `0.649213`, a `+0.012114`
  diagnostic delta versus Flat. Its prediction counts (`765 / 901 / 661`)
  closely matched gold counts (`762 / 909 / 656`).
- A hierarchical-only NEI logit bias reached only `0.645067` Macro-F1. The
  gain therefore comes from complementary rankings, not merely a class-prior
  correction.
- Decision: complementarity promotion potential passes on the development
  fold. Freeze interpolation weight `0.63`; train matched Flat and
  hierarchical models on untouched folds `1–4`; do not retune interpolation,
  loss weights, epochs, or decision thresholds. The official validation and
  test splits remain unused.

### R2V-SV-CV-04 — frozen folds 1–4 confirmation

- The fold-0-selected interpolation weight `0.63` was frozen and evaluated on
  untouched train-only folds `1–4`; no coefficient or threshold was retuned.
- Fold Macro-F1 (`Flat / Hierarchical / Frozen ensemble`, paired delta):
  fold 1 `0.6560 / 0.6667 / 0.6687` (`+0.0128`); fold 2
  `0.6617 / 0.6459 / 0.6563` (`-0.0054`); fold 3
  `0.6802 / 0.6740 / 0.6747` (`-0.0055`); fold 4
  `0.6670 / 0.6630 / 0.6675` (`+0.0006`).
- Paired ensemble-minus-Flat Macro-F1 delta: mean `+0.000601`, standard
  deviation `0.007445`; values `[+0.012776, -0.005433, -0.005489, +0.000551]`.
- The predeclared `+0.015` mean-delta gate failed; stability passed. Official
  validation and test remained unused.
- Decision: reject the hierarchical loss, counterfactual variant, and
  probability ensemble as primary Phase-B methods. The fold-0 gain was a
  development-fold selection effect. Retain the Flat verifier baseline and
  stop tuning class weights, NEI offsets, or interpolation on these folds.
  The next Phase-B investigation must model the train-to-evaluation domain
  shift with train-only source/topic/coverage groups before another verifier
  is trained.

### R2V-DG-CV-05 — train-only OOF domain audit

- The Flat verifier's five held-out fold predictions cover all `11631` strict
  training examples; validation and test were not read.
- Source gap: Politifact Macro-F1 `0.7044` versus Snopes `0.6231`.
- Evidence-availability gap: qrel-absent (`n=4294`) Macro-F1 `0.5460` versus
  qrel-available (`n=7337`) `0.6363`. Top-5 gold miss (`n=5030`) accuracy was
  `0.6022`, versus `0.7390` for a hit.
- Long claims were weaker (Q4 accuracy `0.6416`) than short claims (Q1
  `0.7152`), but their Macro-F1 difference was only `0.0387`.
- Retrieval confidence and score-margin quartiles had nearly flat Macro-F1
  (`0.6560–0.6624`) and therefore do not justify another scalar uncertainty
  threshold.
- Decision: Phase B-v3 targets source and evidence-availability robustness,
  not calibration. Screen inference-agnostic Evidence-Availability GroupDRO
  over train-only `source x qrel-availability` groups, using the unchanged
  Flat verdict loss. Qrel metadata defines training groups only and is never
  supplied to the verifier prompt or required at inference.

### R2V-DG-CV-06 — Evidence-Availability GroupDRO fold-0 screen

- Matched Flat fold-0 reference: accuracy `0.6639`, Macro-F1 `0.6371`.
  Evidence-Availability GroupDRO: accuracy `0.6450`, Macro-F1 `0.6350`
  (`-0.0021`).
- Qrel-absent Macro-F1 improved from `0.5182` to `0.6187` (`+0.1004`), and
  top-5 gold-miss Macro-F1 improved from `0.5994` to `0.6188` (`+0.0194`).
- The robustness transfer was not free: qrel-available Macro-F1 fell from
  `0.6243` to `0.5960` (`-0.0283`), gold-hit Macro-F1 fell from `0.6394` to
  `0.6123` (`-0.0271`), Politifact fell `-0.0101`, and Snopes fell `-0.0068`.
- The predeclared qrel-absent and overall non-inferiority conditions passed;
  the required `+0.01` Snopes improvement failed. No validation/test access.
- Decision: reject source-by-availability GroupDRO for promotion. It proves
  that availability robustness is learnable, but over-allocates capacity away
  from evidence-grounded examples and does not address the source gap. Run one
  source-only DRO ablation next; only a positive Snopes result justifies a
  dual-axis robust objective.

### R2V-DG-CV-07 — source-only GroupDRO fold-0 screen

- Group weights were unstable: epoch 1 assigned Snopes `0.9604`, epoch 2
  `0.7195`, then epoch 3 collapsed to Politifact `0.9994`.
- Overall Macro-F1 changed from `0.6371` to `0.6376` (`+0.0005`). Snopes
  improved from `0.6141` to `0.6227` (`+0.0086`), below the predeclared
  `+0.01` gate; Politifact decreased `-0.0132` but remained inside its
  non-inferiority bound.
- Qrel-absent Macro-F1 fell from `0.5182` to `0.4821`, while qrel-available
  increased from `0.6243` to `0.6309`.
- Decision: reject source-only and dual-axis GroupDRO. The near-zero overall
  gain and oscillating adversarial weights show that loss reweighting moves
  errors between domains instead of learning invariant evidence relations.
  Stop the robust-weighting branch. Phase B-v4 will replace raw article text
  with claim-conditioned atomic sentence evidence, using MOCHEG's
  sentence-level qrels for train-only supervision and natural retrieval for
  held-out evaluation.

## R2V-VIS-VAL-09 — global-embedding stance diagnostic (2026-08-20)

- Protocol: strict validation only; frozen text anchor and retrieval-ranked
  visual candidates; no validation gold injection and no test access.
- Text anchor: accuracy `0.564560`, Macro-F1 `0.549948`.
- Global-embedding visual expert: accuracy `0.494505`, Macro-F1 `0.459179`.
- Gold-candidate visual stance: accuracy `0.355401`, Macro-F1 `0.340672`.
- Visual sufficiency: AUROC `0.458078`, average precision `0.361864`.
- Frozen retrieval Select@1: `0.540070`; oracle router Macro-F1: `0.617875`.
- Decision: reject further scalar/global-embedding fusion and routing. Phase B
  continues with token-level claim-to-patch cross-attention; the text anchor
  and test split remain frozen.

### R2V-VIS-VAL-10 — token/patch cross-attention screen

- Best epoch: `3`; claim accuracy `0.381181`, Macro-F1 `0.345528`.
- Natural validation Select@1: `0.429561`.
- Gold-candidate stance: accuracy `0.219038`, Macro-F1 `0.184794`.
- Relevance: AUROC `0.380450`, average precision `0.130005`.
- Both preregistered gates failed. Increasing capacity is not justified: the
  image-qrel annotation identifies useful evidence, but does not guarantee
  that pixels alone express the claim-level supported/refuted/NEI stance.
- Decision: stop pixel-only verdict supervision. The next expert uses a VLM as
  a claim-conditioned constraint analyzer and leaves the final verdict to the
  evidence-set verifier.

### R2V-VIS-DATA-11 — claim-conditioned visual reports

- Analyzer: `Qwen/Qwen3-VL-2B-Instruct`; top `2` images per claim.
- Train: `11631/11631` complete; train-only positive injection is recorded in
  its metadata and never exposed in the analyzer prompt.
- Validation: `1456/1456` complete; no gold injection.
- Shared report signature:
  `0d43d871e6205a3728bab2fa9537a60c5e2c98593524021bdbcae91e931841c1`.
- Next frozen screen: encode these reports with the text verifier's
  `Qwen/Qwen3-Embedding-0.6B` encoder, train only the visual report expert, and
  keep routing and the test split disabled.

### R2V-VIS-VAL-11 — visual report expert

- Frozen text anchor: accuracy `0.564560`, Macro-F1 `0.549948`.
- Visual report expert: accuracy `0.509615`, Macro-F1 `0.480531`.
- Gold-candidate report stance: accuracy `0.444444`, Macro-F1 `0.411050`.
- Report Select@1: `0.947090`; oracle router Macro-F1: `0.618723`.
- Sufficiency AUROC `0.406769` and AP `0.211952` fail the sufficiency gate.
- Decision: preserve the highly discriminative report selector, remove the
  invalid qrel-coverage sufficiency objective, and screen a small safe residual
  fusion adapter anchored to the frozen text verifier. This remains phase-B
  fusion; cost-aware routing remains disabled.

### R2V-FUSE-VAL-12 — safe report logit fusion

- Predeclared acceptance delta: `+0.003` Macro-F1 over `0.549948`.
- Best candidate was epoch `1`: accuracy `0.563187`, Macro-F1 `0.543902`,
  gate mean `0.060809`, help rate `0.018544`, harm rate `0.019918`.
- The candidate underperformed the anchor by `-0.006046`; final mode correctly
  reverted to `text_anchor` with delta `0.0`.
- Decision: reject separate late fusion and stop visual gate tuning. Retrieval
  is already strong; replace frozen pooled embeddings with a jointly fine-tuned
  long-context claim/evidence verifier. Visual reports become an ablation input
  to that unified verifier rather than a separately routed verdict expert.

### R2V-TEXT-VAL-13 — ModernBERT long-context verifier

- Best epoch `3`: accuracy `0.4753`, Macro-F1 `0.442092`, ECE `0.0835`.
- Delta versus the frozen evidence-set anchor: `-0.107856` Macro-F1.
- Training loss decreased through epoch 5 while validation remained poor,
  ruling out simple under-training. The natural train retrieval has far lower
  gold coverage than validation, so a monolithic classifier is frequently
  trained to infer labels from contexts lacking relevant evidence.
- Decision: reject the random-head long-context classifier. Use an
  FEVER/NLI-pretrained pair verifier, inject one train-only gold article when
  available, preserve natural validation retrieval, and aggregate pair states
  into one claim verdict.

### R2V-TEXT-VAL-14 — FEVER/NLI pair-set verifier

- Backbone: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`;
  source label order `[entailment, neutral, contradiction]` was remapped to
  GraphCURE `[supported, refuted, NEI]` with source indices `[0, 2, 1]`.
- Train gold coverage after train-only injection: `0.630814`; natural
  validation coverage: `0.936813`.
- Best epoch `1`: accuracy `0.5103`, claim Macro-F1 `0.478109`, pair Macro-F1
  `0.4764`, Select@1 `0.6364`, ECE `0.2384`.
- Delta versus frozen anchor: `-0.071839` Macro-F1. Later epochs did not
  improve either pair or claim performance.
- Decision: reject both the transferred pair stance and its smooth-max
  aggregator. Do not tune temperature or learning rate. The next screen keeps
  the proven cached verifier frozen and learns only a zero-initialized,
  conflict-aware set-interaction residual with an automatic anchor fallback.

### R2V-TEXT-VAL-15 — anchor-preserving conflict residual

- Frozen anchor: accuracy `0.564560`, Macro-F1 `0.549948`.
- Best residual candidate: epoch `2`, Macro-F1 `0.550082`, delta
  `+0.000134`.
- The predeclared `+0.003` acceptance gate failed, so the deployed mode was
  correctly set to `anchor_fallback`; final accuracy and Macro-F1 remained
  `0.564560 / 0.549948`, with zero help and zero harm.
- The originally printed fallback gate mean (`0.587806`) belonged to the
  rejected candidate, not the deployed logits. Reporting is patched to emit a
  zero gate for fallback mode.
- Decision: reject ungrounded residual tuning. The next screen directly
  addresses the natural-train (`~0.58`) versus validation (`~0.94`) retrieved
  gold-coverage mismatch with paired natural/grounded train views. Validation
  remains natural and the test split remains locked.

### R2V-TEXT-VAL-16 — paired retrieval curriculum

- Natural train gold coverage: `0.578368`; grounded-view coverage: `0.630814`;
  only `610` previously uncovered claims could receive a valid train-only
  text-qrel insertion.
- Best candidate: epoch `5`, Macro-F1 `0.540604`, delta `-0.009344` versus the
  frozen anchor.
- The automatic fallback preserved accuracy `0.564560`, Macro-F1 `0.549948`,
  Select@1 `0.807860`, and ECE `0.048946`.
- Decision: reject retrieval curriculum. Missing train text qrels are an
  annotation/modal-evidence limitation, not a candidate-ranking issue. Stop
  tuning small heads on frozen embeddings; screen an instruction-tuned causal
  verifier with BF16 LoRA and deterministic one-token verdict scoring.

### R2V-TEXT-VAL-17 — Qwen3-4B LoRA verifier, seed 42

- Model: `Qwen/Qwen3-4B-Instruct-2507`; BF16 LoRA rank `16`; top-5
  reranked evidence; maximum context `2048`; deterministic `A/B/C` next-token
  verdict scoring.
- Best epoch `3`: accuracy `0.679945`, Macro-F1 `0.670471`, confusion matrix
  `[[318, 71, 102], [14, 434, 39], [98, 142, 238]]`.
- Delta versus frozen cached verifier: **`+0.120523` Macro-F1**. The model
  passed the predeclared validation gate without class collapse.
- ECE-10 is `0.219151`, so raw confidence is not yet suitable for routing.
- Decision: freeze architecture and hyperparameters; run seeds
  `13, 21, 87, 100`, then fit per-seed validation temperatures and evaluate a
  fixed five-seed ensemble. Test remains locked.

### R2V-TEXT-VAL-18 — frozen five-seed Qwen3 LoRA validation

- Per-seed accuracy: `0.684753 +/- 0.007855`; per-seed Macro-F1:
  `0.674751 +/- 0.008527`. All five runs passed independently.
- Raw ensemble: accuracy `0.700549`, Macro-F1 `0.692045`, ECE-10 `0.151147`.
- Validation-temperature-scaled ensemble: accuracy `0.699863`, Macro-F1
  `0.691260`, ECE-10 `0.037251`.
- Temperatures were stable (`3.0065` to `3.1932`); calibration greatly reduced
  ECE without a material verdict change.
- Stability gate passed. Freeze the raw ensemble as the primary test method;
  use the validation-temperature-scaled ensemble only for calibration and the
  later cost router. No parameter may be selected on test.

### R2V-TEXT-TEST-19 — one-shot frozen Qwen3 LoRA strict test

- Protocol: strict cross-split-deduplicated MOCHEG test (`n=2434`), system
  retrieval from the fixed MOCHEG corpus; five frozen seeds; zero parameters
  fitted on test.
- Per-seed accuracy: `0.566804 +/- 0.009794`; per-seed Macro-F1:
  `0.543815 +/- 0.011777`.
- Primary raw ensemble: accuracy **`0.569022`**, Macro-F1 **`0.545806`**,
  ECE-10 `0.290171`, NLL `1.777144`; confusion matrix
  `[[383, 284, 149], [12, 751, 62], [96, 446, 251]]`.
- Validation-temperature-scaled ensemble: accuracy `0.566146`, Macro-F1
  `0.541914`, ECE-10 `0.127515`, NLL `0.977286`. Calibration substantially
  improves confidence quality but not verdict quality, so it remains a
  secondary routing result rather than the primary test score.
- Raw-ensemble 95% bootstrap intervals (5000 resamples): accuracy
  `[0.549302, 0.587921]`; Macro-F1 `[0.525431, 0.564776]`.
- Inference latency: approximately `40.25 ms/sample/model` on the server RTX
  5090 (range `39.89` to `40.64` across seeds), before ensemble amortization.
- Conservative comparison with the strongest reported AMuFC retrieved point
  (`0.5577` Accuracy, `0.5560` Macro-F1): `+0.011322` Accuracy but
  `-0.010194` Macro-F1.
- Status: strong strict-split result, but **not Macro-F1 SOTA**.

### R2V-TEXT-TEST-20 — frozen Qwen3 LoRA official test

- Protocol: `P1_closed_corpus_retrieved_official_n2442`; official MOCHEG test
  (`n=2442`); the same five adapters, validation temperatures, prompt, top-k,
  and ensemble were retained; zero parameters fitted on test.
- Per-seed accuracy: `0.566503 +/- 0.009929`; per-seed Macro-F1:
  `0.543984 +/- 0.011925`.
- Primary raw ensemble: accuracy **`0.567977`**, Macro-F1 **`0.545309`**,
  ECE-10 `0.290937`, NLL `1.774854`; confusion matrix
  `[[384, 284, 149], [13, 750, 62], [96, 451, 253]]`.
- Validation-temperature-scaled ensemble: accuracy `0.565111`, Macro-F1
  `0.541436`, ECE-10 `0.128220`, NLL `0.976907`; confusion matrix
  `[[379, 287, 151], [13, 752, 60], [98, 453, 249]]`.
- Raw-ensemble 95% bootstrap intervals (5000 resamples): accuracy
  `[0.549140, 0.587633]`; Macro-F1 `[0.525794, 0.565170]`.
- Against the strongest reported AMuFC retrieved result (`0.5577` Accuracy,
  `0.5560` Macro-F1), GraphCURE is `+0.010277` Accuracy but `-0.010691`
  Macro-F1. Against the lower arXiv-v2 AMuFC row (`0.546/0.540`), GraphCURE
  is `+0.021977/+0.005309`; both versions must be disclosed rather than
  selecting the favorable comparison.
- Status: official fixed-corpus retrieved **accuracy improvement**, but not the
  strongest reported Macro-F1 SOTA. Phase B is frozen; no post-test tuning is
  allowed on this protocol.

## GraphCURE-R2V validation ledger (2026-08-17)

The entries in this section are validation-only development results. Official
test retrieval remains locked until architecture and hyperparameters freeze.

### R2V-RET-01 — Qwen3-Embedding-4B hybrid candidate retrieval

- Split: MOCHEG strict validation (`n=1456`), frozen local corpus (`4067` docs)
- Output: `outputs/retrieval_mocheg_qwen3_hybrid/summary.json`
- Recall@1: **0.774725**
- Recall@5: **0.896978**
- Recall@10: **0.927198**
- Recall@50: **0.953984**
- MRR: **0.829072**
- Matched dense-top50 reference: Recall@50 `0.943681`, MRR `0.817854`
- Delta versus matched reference: Recall@50 **+0.010302**, MRR **+0.011218**
- Decision: **candidate-retrieval gate passed**; proceed to validation-only
  Qwen3-Reranker-4B screening.

### R2V-RER-00 — invalid Qwen3 reranker implementation

- Observed Recall@10: `0.916896`; MRR: `0.289560`
- Status: **invalidated; exclude from scientific comparison**
- Cause: the first implementation truncated the fully formatted prompt. Long
  evidence could remove the required assistant suffix, so the final-token
  `yes`/`no` logits were not valid relevance scores.
- Resolution: truncate only query-document content, then append the fixed Qwen
  prefix/suffix exactly as specified by the official model card. Rerun as
  `R2V-RER-01` on validation only.

### R2V-RER-01 — corrected Qwen3-Reranker-4B

- Split: MOCHEG strict validation (`n=1456`), Qwen3 hybrid top-50 candidates
- Output: `outputs/retrieval_mocheg_qwen3_reranked/summary.json`
- Recall@10: **0.945742**
- MRR: **0.886368**
- Delta versus Qwen3 hybrid pre-reranking: Recall@10 **+0.018544**,
  MRR **+0.057296**
- Reachable-hit retention: `0.945742 / 0.953984 = 0.99136`; the reranker
  retains 99.1% of validation claims whose gold evidence exists in candidate
  top-50 while moving relevant evidence substantially earlier.
- Decision: **reranking gate passed**; materialize train retrieval/reranking,
  then train the claim-level evidence-set verifier. Official test remains locked.

### R2V-RET-TRAIN-01 — Qwen3 hybrid train materialization

- Split: MOCHEG strict train (`n=11631`), frozen corpus (`18989` docs)
- Claims with annotated text evidence: `7337`; raw retrieval ceiling when
  no-text-qrel claims count as misses: `7337/11631 = 0.630814`
- Raw Recall@1/5/10/50: `0.429628 / 0.534778 / 0.562032 / 0.592382`
- Raw MRR: `0.476523`
- Conditional on claims with annotated text evidence, Recall@1/5/10/50:
  `0.681069 / 0.847758 / 0.890964 / 0.939076`
- Conditional MRR: `0.755410`
- Matched dense-top50 train reference: raw Recall@50 `0.581549`, MRR
  `0.474593`
- Decision: **train candidate gate passed**. The raw train/validation gap is
  primarily annotation availability; both raw and coverage-conditional metrics
  must be reported.

### R2V-RER-TRAIN-01 — Qwen3-Reranker-4B train materialization

- Split: MOCHEG strict train (`n=11631`), Qwen3 hybrid top-50 candidates
- Raw Recall@1/5/10: `0.485599 / 0.567535 / 0.581377`
- Raw MRR: `0.521653`
- Conditional on the `7337` claims with annotated text evidence,
  Recall@1/5/10: `0.769797 / 0.899687 / 0.921630`
- Conditional MRR: `0.826952`
- Delta versus pre-reranking train candidates: Recall@1 `+0.055971`,
  Recall@5 `+0.032757`, Recall@10 `+0.019345`, MRR `+0.045130`
- Candidate-hit retention in top-10: `0.581377 / 0.592382 = 0.981422`
- Decision: **train reranking gate passed**; mine structured reasoning traps and
  train the evidence-utility/sufficiency verifier. Official test remains locked.

### R2V-NEG-01 — structured reasoning-negative audit

- Claims: `11631`; claims without annotated text positives: `4294`
- Mined non-qrel candidates: `102520` (`8.81` per claim on average)
- Semantic hard: `44299` (`43.21%`)
- Negation traps: `37272` (`36.36%`)
- Quantity traps: `19446` (`18.97%`)
- Lexical-overlap traps: `1503` (`1.47%`)
- Interpretation: the candidate pool contains substantial negation and numeric
  confounders rather than only generic dense neighbors. Because qrels may be
  incomplete, the verifier uses conservative auxiliary relevance weights
  (`1.25` for negation/quantity traps and `1.10` for lexical traps), while the
  claim-level verdict remains the primary objective.

### R2V-VER-VAL-01 — frozen cached evidence-set verifier

- Split: MOCHEG strict validation (`n=1456`); **test split not used**
- Seeds: `13, 21, 42, 87, 100`
- Frozen cache: Qwen3-Embedding-0.6B, top-8 reranked evidence, 1024 dimensions
- Accuracy: **0.5611 +/- 0.0103**
- Macro-F1: **0.5436 +/- 0.0104**
- Evidence selection hit@1 (conditional on a gold candidate):
  **0.8399 +/- 0.0173**
- ECE-10: `0.1768 +/- 0.1262`
- Retrieval gold coverage@8: `0.943681`
- Stability gate: **passed** (mean Macro-F1 >= `0.50`, standard deviation
  <= `0.02`)
- Interpretation: verdict quality is stable enough to freeze Phase B. ECE is
  seed-sensitive and is reserved for post-hoc calibration in the routing phase;
  it is not used to select a seed or revise the frozen Phase-B verifier.
- Frozen specification: `configs/mocheg_r2v_frozen.json`
- Test policy: materialize retrieval/cache once, then report the matched
  five-seed mean and standard deviation without selecting a test seed.

### R2V-RET-TEST-01 — frozen test retrieval and reranking

- Split: MOCHEG strict test (`n=2434`), fixed local corpus (`6268` docs)
- Hybrid Qwen3-Embedding-4B Recall@1/5/10/50:
  `0.742810 / 0.889071 / 0.925637 / 0.954807`
- Hybrid MRR: `0.807668`
- Qwen3-Reranker-4B Recall@1/5/10:
  `0.822104 / 0.929745 / 0.945357`
- Reranked MRR: `0.869985`
- Reranker delta: Recall@1 `+0.079293`, Recall@5 `+0.040674`,
  Recall@10 `+0.019721`, MRR `+0.062317`
- Top-10 retention of reachable top-50 hits: `0.945357 / 0.954807 = 0.99010`

### R2V-VER-TEST-01 — frozen five-seed test result

- Protocol: `P1_closed_corpus_retrieved_evidence`
- Seeds: `13, 21, 42, 87, 100`; no seed selected using test performance
- Accuracy: **0.4961 +/- 0.0108**
- Macro-F1: **0.4711 +/- 0.0118**
- Evidence selection hit@1: **0.8204 +/- 0.0307**
- ECE-10: `0.2481 +/- 0.1259`
- Numerical delta versus the preregistered HGTMFC milestone: Accuracy
  **+0.0100**, Macro-F1 **+0.0033**. This is not an exact protocol match because
  published baselines use the official `n=2442` test split while this result
  uses the cross-split-deduplicated `n=2434` strict split.
- Current matched AMuFC target: Accuracy `0.546`, Macro-F1 `0.540`; GraphCURE
  gaps: Accuracy `-0.0499`, Macro-F1 `-0.0689`
- Conclusion: the frozen text-retrieved R2V system is a strong reproducible
  strict-split control and numerically clears the HGTMFC milestone, but is not
  the current overall P1 SOTA. Preserve this strict result and later add an
  official-split comparability track. Proceed now to validation-only adaptive
  visual-evidence development (R2V-v2).
- Detailed comparison: `docs/MOCHEG_SOTA_COMPARISON.md`

### R2V-VIS-VAL-01 — Qwen3-VL direct visual retrieval

- Split: MOCHEG strict validation (`n=1456`); **test split not used**
- Image corpus: `12267`; claims with aligned gold image evidence: `905`
- Model: `Qwen/Qwen3-VL-Embedding-2B`
- Output: `outputs/retrieval_mocheg_qwen3vl_images/summary.json`
- Raw Recall@1/5/10/50: `0.204670 / 0.269231 / 0.296703 / 0.370879`
- Conditional Recall@1/5/10/50 on the 905 image-annotated claims:
  `0.329282 / 0.433149 / 0.477348 / 0.596685`
- Raw MRR: `0.235726`; conditional MRR: `0.379245`
- Signature: `d9fa11aba6eb63d53e4d99f620dff68dcdaa23c8796a24ab35e336e34b05fcdc`
- At deeper candidate depths, raw Recall@100/200 was `0.396291 / 0.433379`
  and conditional Recall@100/200 was `0.637569 / 0.697238`; conditional MRR
  was `0.380187`. Top-200 signature:
  `88ae2f4bcf11a60a647ebb6b1fd388e8770fcfd5f9b063891cb28c86256950c2`.
- Gate: conditional Recall@10 >= `0.40` **passed**; conditional Recall@50
  >= `0.65` **failed** by `0.053315`.
- Candidate-ceiling gate: conditional Recall@200 >= `0.75` **failed** by
  `0.052762`.
- Decision: **partial pass; do not evaluate test**. Add a leakage-safe,
  constraint-aligned multi-intent candidate ensemble before visual reranking.

### R2V-VIS-VAL-02 — constraint-aligned prompt ensemble

- Split: MOCHEG strict validation only; four semantic/entity/temporal-context
  query views over the frozen Qwen3-VL image embedding corpus
- Fused conditional Recall@1/5/10/50/100/200:
  `0.3193 / 0.4309 / 0.4718 / 0.5967 / 0.6376 / 0.6884`
- Fused conditional MRR: `0.3735`
- Best individual conditional Recall@200: semantic `0.6972`; entity `0.6950`;
  temporal/provenance `0.6895`; contextual/OCR `0.6464`
- Conditional Recall@300 by view: semantic `0.7370`; entity `0.7359`;
  temporal/provenance `0.7271`; contextual/OCR `0.6829`
- Decision: **failed**. Prompt-only views remain too correlated and fusion
  underperforms direct retrieval at top-200. Exclude the prompt ensemble from
  the frozen system. Next screen: claim-independent, pixel-derived caption/OCR
  descriptors fused with direct visual retrieval. Test remains locked.

### R2V-VIS-CAP-SMOKE-01 — invalidated pixel-descriptor generation

- Split: first 32 strict-validation corpus images; model
  `Qwen/Qwen3-VL-2B-Instruct`
- Signature: `273232bf1caed5413420645b8d9498f24812bb9393624a097017ddf8f13fbcd9`
- Positive observation: image content and OCR were generally relevant.
- Failure: four of the first five inspected outputs contained severe repeated
  phrases/sentences and several ended mid-phrase at the token limit.
- Decision: **invalidated before full materialization**. Preserve the 32-row
  v1 output as a negative smoke run. Rerun in a distinct v2 output root with a
  one-line structured descriptor, repetition penalty `1.15`, and no-repeat
  4-gram decoding. Test remains locked.

### R2V-VIS-CAP-SMOKE-02 — invalidated structured descriptor decoding

- Split: first 32 strict-validation images; signature
  `d0207071b0cc83697b7f8c8d1cf8b7b32d53c3f74583b5802f1b413518ab8917`
- Repeated 4-gram outputs: `0/32`; word count min/mean/max:
  `28 / 60.53 / 78`
- Structured four-field compliance: `15/32`, below the preregistered `24/32`
  smoke gate.
- Root cause confirmed in runtime warnings: decoder-only generation used right
  padding. Several outputs began mid-word or ended mid-field; the four-field
  prompt was also too long for stable 2B-model compliance.
- Decision: **invalidated; do not materialize full v2**. V3 uses left padding,
  processor-scoped padding arguments, a three-field 55-word prompt, one-line
  normalization, repetition penalty `1.05`, and no-repeat 4-grams.

### R2V-VIS-CAP-SMOKE-03 — compact left-padded descriptors

- Split: first 32 strict-validation images; model
  `Qwen/Qwen3-VL-2B-Instruct`
- Repeated 4-gram outputs: `0/32`; word count min/mean/max:
  `14 / 34.16 / 59`
- Manual inspection: sampled descriptions were concise, visually relevant,
  and contained useful named entities/OCR without the v1/v2 mid-token failure.
- The initial automated structured count was incorrectly `0/32` because Qwen
  emitted the full-width Unicode colon in `Type/Clues：`. This is a parser-level
  typography variant, not a generation failure.
- Resolution: canonicalize Unicode colons during descriptor write/read and
  stamp normalization version `unicode-colon-v1` in caption-fusion provenance.
  Re-audit the existing 32 descriptors; no pixel regeneration is required.

### R2V-VIS-CAP-VAL-01 — full validation descriptor materialization

- Split: strict validation image corpus
- Images/descriptors: `12267/12267`; status: **complete**
- Model: `Qwen/Qwen3-VL-2B-Instruct`
- Descriptor signature:
  `ff9e34ed57d758d6f31ca4b46c4928cf82d0a08ebab2576101fd5dc98f397072`
- Protocol: descriptors are claim-independent and derived from pixels only;
  claim text, qrels, labels, and filename topic IDs are not model inputs.
- Decision: materialization gate **passed**. Run validation-only caption
  dense+lexical retrieval, direct-visual fusion, and candidate-union analysis.

### R2V-VIS-CAP-VAL-02 — complementary caption candidate retrieval

- Split: MOCHEG strict validation (`n=1456`); **test split not used**
- Claims with aligned gold image evidence: `905`
- Direct/caption/union raw candidate recall:
  `0.433379 / 0.438874 / 0.484203`
- Direct/caption/union conditional candidate recall:
  `0.697238 / 0.706077 / 0.779006`
- Caption-only gold recoveries missed by direct retrieval: `74`
- RRF fused raw Recall@1/5/10/50/100/200:
  `0.1916 / 0.2747 / 0.3029 / 0.3777 / 0.4100 / 0.4444`
- RRF fused conditional Recall@1/5/10/50/100/200:
  `0.3083 / 0.4420 / 0.4873 / 0.6077 / 0.6597 / 0.7149`
- Per-view conditional Recall@200: direct `0.6972`, caption dense `0.6343`,
  caption lexical `0.5558`.
- Decision: the `0.75` candidate-union gate **passed**, proving genuine
  complementary evidence, but fixed RRF loses part of that gain. Freeze neither
  ranking nor test. Measure the top-400 fused ceiling, then apply a
  claim-conditioned Qwen3-VL cross-reranker over pixels plus descriptors.

### R2V-VIS-CAP-VAL-03 — top-400 reranker candidate pool

- Split: MOCHEG strict validation (`n=1456`); **test split not used**
- Fused raw Recall@1/5/10/50/100/200/300/400:
  `0.1916 / 0.2747 / 0.3029 / 0.3777 / 0.4100 / 0.4444 / 0.4567 / 0.4698`
- Fused conditional Recall@1/5/10/50/100/200/300/400:
  `0.3083 / 0.4420 / 0.4873 / 0.6077 / 0.6597 / 0.7149 / 0.7348 / 0.7558`
- Candidate union conditional ceiling: `0.779006`; top-400 retains
  approximately `97.02%` of the reachable union hits.
- Decision: top-300 fails the preregistered `0.75` ceiling gate; top-400
  **passes** by `0.0058` and is the smallest measured admissible pool. Run a
  two-claim Qwen3-VL reranker smoke benchmark before authorizing the full
  `1456 x 400 = 582400` validation pair pass.

### R2V-VIS-RERANK-SMOKE-01 — invalid truncation configuration

- Scope: two validation claims, ten candidates each; zero pairs completed
- Failure: `max_length=1024` truncated expanded visual tokens, producing a
  processor image-token count mismatch before the first score was written.
- Runtime before failure: `80.208` seconds, including model initialization.
- Classification: engineering/configuration failure; **not an experimental
  result** and excluded from comparisons.
- Resolution: use the Qwen official multimodal reranker cap `max_length=10240`
  with dynamic padding, then repeat the same smoke gate.

### R2V-VIS-RERANK-SMOKE-02 — corrected multimodal interface smoke

- Scope: first two strict-validation claims, ten fused image candidates each
- Completed claims/pairs: `2/2` and `20/20`; status: **pass**
- Runtime including cached model initialization: `27.917` seconds
- Smoke-only Recall@1/5/10 and MRR: `0.5 / 0.5 / 0.5 / 0.5`; these values
  are not comparative metrics because the two-claim sample is intentionally
  tiny.
- Model: `Qwen/Qwen3-VL-Reranker-2B`; `max_length=10240`; pixels and
  claim-independent descriptors both supplied to each candidate document.
- Retrieval SHA-256:
  `8913c081061d3187eb9bd9c5cfba45db95ac6a149bbbfc672d540d16384fc78a`
- Reranker signature:
  `b8790584396c2a2cc974fd70877ba4338fced0ad29cedfdb54545ad000f3a03f`
- Leakage audit: gold identifiers were used only after scoring; reported
  `gold_used_for_scoring=false`.
- Decision: functional gate **passed**. Measure steady-state throughput on 500
  pairs before choosing a full reranking or cascaded reranking design.

### R2V-VIS-RERANK-BENCH-01 — batch-4 throughput

- Scope: first ten strict-validation claims, 50 candidates per claim;
  `500/500` pairs completed
- Batch size: `4`; elapsed time: `78.073` seconds
- Throughput: `6.4043` claim-image pairs/second
- Projected top-400 validation runtime (`582400` pairs): `25.26` hours
- Benchmark-only Recall@1/5/10/50: `0.4 / 0.6 / 0.8 / 0.8`; excluded
  from comparative results because this is a non-random ten-claim timing slice.
- Output integrity: `10/10` claims, `complete=true`,
  `gold_used_for_scoring=false`.
- Decision: a resumable full run is operationally feasible, but benchmark
  batch size 8 before spending approximately 25 GPU-hours.

### R2V-VIS-RERANK-BENCH-02 — frozen batch-8 throughput

- Scope: the same ten claims and 500 candidate pairs as BENCH-01
- Batch size: `8`; elapsed time: `76.829` seconds
- Throughput: `6.5080` pairs/second; projected full runtime: `24.86` hours
- Speedup over batch 4: approximately `1.62%`; output ranking remained
  consistent apart from negligible floating-point score differences.
- Output integrity: `10/10` claims, `complete=true`,
  `gold_used_for_scoring=false`.
- Decision: freeze batch 8 for the full validation run. Because batch size is
  excluded from the semantic reranker signature, an interrupted or OOM run can
  safely resume at batch 4 without mixing model configurations.

### R2V-VIS-RERANK-VAL-01 — full Qwen3-VL visual reranking

- Split: MOCHEG strict validation (`n=1456`); **test split not used**
- Completed claims: `1456/1456`; claims with aligned gold images: `905`
- Model: `Qwen/Qwen3-VL-Reranker-2B`; candidate pool `400`; output `100`
- Raw Recall@1/5/10/50/100:
  `0.212912 / 0.309753 / 0.342033 / 0.412775 / 0.438187`
- Conditional Recall@1/5/10/50/100:
  `0.342541 / 0.498343 / 0.550276 / 0.664088 / 0.704972`
- Raw/conditional MRR: `0.258091 / 0.415227`
- Conditional improvements over fixed RRF at 1/5/10/50/100:
  approximately `+0.0342 / +0.0563 / +0.0630 / +0.0564 / +0.0453`.
- Acceptance gates: conditional Recall@10 greater than `0.4873` **passed**;
  conditional Recall@50 greater than `0.6077` **passed**.
- Retrieval SHA-256:
  `8913c081061d3187eb9bd9c5cfba45db95ac6a149bbbfc672d540d16384fc78a`
- Visual reranker signature:
  `ff96b33ef5c4403a8d0cdbfb627f1ecfe71695a9d3de2df14cd43f367ca95687`
- Leakage audit: `gold_used_for_scoring=false`.
- Decision: **freeze the validation visual retrieval/reranking stack**. Do not
  open test yet. Materialize train-split direct visual hard negatives, inject
  train-only qrel positives, and develop the multimodal verifier on
  train/validation.

### R2V-VIS-TRAIN-CACHE-01 — resumable aspect-ratio failure

- Train image embedding reached shard `284/305` (`93%`) before encountering
  an image with absolute aspect ratio `232.0`; Qwen requires a ratio `<200`.
- The accompanying libpng iCCP message is a non-fatal metadata warning.
- No completed work was lost: image embeddings are checkpointed per 256-image
  shard and the failing shard was not committed.
- Resolution: preprocessing now detects images with aspect ratio `>=200` and
  letterboxes them to ratio at most `190` without stretching or cropping.
  Decoder-incompatible formats and geometric normalization share a versioned
  RGB-PNG cache. Resume the identical train retrieval command.

### R2V-VIS-RET-TRAIN-01 — direct visual hard-negative materialization

- Split: MOCHEG strict train (`n=11631`); image corpus: `78031`
- Claims with aligned gold images: `6867`
- Model: `Qwen/Qwen3-VL-Embedding-2B`; output top-50
- Raw Recall@1/5/10/50:
  `0.137821 / 0.185883 / 0.202734 / 0.250537`
- Conditional Recall@1/5/10/50 among claims with aligned gold images:
  `0.233435 / 0.314839 / 0.343381 / 0.424348`
- Raw/conditional MRR: `0.160478 / 0.271811`
- Output integrity: `11631/11631` rows; visual retrieval signature
  `d9fa11aba6eb63d53e4d99f620dff68dcdaa23c8796a24ab35e336e34b05fcdc`.
- Interpretation: the much larger train image corpus makes direct retrieval
  harder than validation. Reranking all `11631 x 400` train pairs with the 2B
  teacher would be unnecessarily expensive. The frozen verifier protocol uses
  direct train top-50 as hard negatives and injects train qrel positives only.
  Validation uses the frozen reranked output and never injects qrels.
- Decision: **train visual cache gate passed**; assemble the leakage-audited
  text/visual feature cache and screen one validation-only verifier seed.

### R2V-MM-VER-VAL-01 — invalid candidate-count-biased fusion

- Split: MOCHEG strict validation (`n=1456`); **test split not used**
- Accuracy/Macro-F1: `0.567308 / 0.547740`
- Text/visual gold coverage: `0.943681 / 0.394231`
- Text/visual Select@1: `0.878457 / 0.048780`
- Mean visual modality mass: `0.691079`; ECE-10: `0.028160`
- Status: **architecture gate failed; exclude as the final multimodal model**.
- Diagnosis: a joint softmax normalized 8 text and 32 visual candidates
  together. Candidate count alone gives visual evidence an approximately
  `32/(32+8)=0.8` prior, explaining high visual mass despite nearly random
  visual selection (`1/32=0.03125`). Verdict Macro-F1 was also `0.00221` below
  the frozen seed-42 text verifier (`0.549948`).
- Resolution: normalize evidence attention separately within each modality;
  replace imbalanced candidate-wise BCE with positive-mass listwise selection;
  initialize from the frozen text verifier; and learn only a conservative,
  reliability-gated visual residual during the next screen.

### R2V-MM-VER-VAL-02 — frozen text-anchor residual screen

- Split: MOCHEG strict validation (`n=1456`); **test split not used**
- Combined and embedded text-only Accuracy: both `0.564560`
- Combined and embedded text-only Macro-F1: both `0.549948`
- Text/visual Select@1: `0.807860 / 0.073171`
- Mean visual gate: `0.100750`
- Gate with/without a gold visual candidate: `0.101389 / 0.100334`
- Visual help/harm rates: `0.0 / 0.0`; ECE-10: `0.048946`
- Decision: **non-inferiority safeguard passed, multimodal utility gate failed**.
  Epoch-zero anchoring exactly preserved the frozen text verifier, but the
  selected checkpoint used no effective visual correction. Visual Select@1
  remained below the preregistered `0.15` threshold and gate separation was
  negligible (`+0.001055`). Do not open test. Audit the epoch trajectory before
  deciding whether to stage selector pretraining or revise visual supervision.

### R2V-MM-VER-VAL-02A — epoch-trajectory diagnosis

- Visual Select@1 rose monotonically enough to pass the selector gate, from
  `0.073171` at epoch 0 to a maximum of `0.216028` at epoch 8.
- At epoch 8, combined Macro-F1 fell to `0.484189`, while the frozen text
  branch remained `0.549948`.
- At that epoch the visual branch corrected `9.13%` of examples but harmed
  `13.12%`; mean gate mass was `0.3813`.
- Interpretation: visual evidence selection is learnable. The remaining
  bottleneck is utility routing, not candidate retrieval or selector capacity.
  A perfect validation-only oracle over the observed corrections has material
  headroom, but gold labels must never be used by the inference router.
- Next experiment: staged selector, full-strength visual expert, then a gate
  supervised only on train by detached per-example cross-entropy reduction
  minus an explicit visual-use cost. Test remains locked.

### R2V-MM-VER-VAL-03 — invalid staged expert selection criterion

- Selector checkpoint: validation Select@1 `0.216028` (**passed**).
- Saved visual expert, oracle router, and learned router Macro-F1 were all
  `0.549948`, exactly equal to the frozen text anchor.
- Final visual gate was effectively zero (`2.06e-9`), with zero help and harm.
- Status: **engineering/model-selection failure; not a negative result for
  expert complementarity**.
- Cause: Stage 2 selected checkpoints by standalone visual-expert Macro-F1.
  This rejects a specialist that is weaker globally but corrects text-specific
  errors—the precise behavior that Stage 3 is designed to route.
- Resolution: select Stage-2 checkpoints by validation oracle-router Macro-F1,
  record standalone expert performance at that checkpoint, and persist
  `selector_best.pt` and `expert_best.pt` independently. The learned Stage-3
  router continues to use train-only cross-entropy-reduction targets.

### R2V-MM-VER-VAL-04 — complementary expert ceiling and router failure

- Text anchor Macro-F1: `0.549948`
- Final visual expert Macro-F1: `0.527767`
- Oracle-router Macro-F1: `0.647325` (**+0.097377 over text**)
- Visual Select@1: `0.216028`
- Learned router selected the text anchor exactly: gate `2.06e-9`, zero help,
  zero harm, and Macro-F1 `0.549948`.
- Interpretation: the visual specialist has substantial complementary signal;
  evidence retrieval, evidence selection, and expert diversity are no longer
  the limiting factors. The deployable utility estimator is the bottleneck.
- Two structural defects were identified: the old visual-expert computation
  depended on the gate it was meant to route, and the gate did not observe
  text/expert probabilities, margins, entropy, or prediction disagreement.
- Resolution: make the visual expert gate-independent; expose probability-level
  conflict/uncertainty features to the gate; and supervise train-time routing
  primarily on decisive pairs (visual corrects text versus visual harms text),
  with class balancing and a low-weight soft target for ambiguous pairs.

### R2V-MM-VER-VAL-05 — conflict-aware soft-router screen

- Text anchor Macro-F1: `0.549948`
- Gate-independent visual expert Macro-F1: `0.509576`
- Oracle-router Macro-F1: `0.647698` (**+0.097750 over text**)
- Visual Select@1: `0.205575` (**selector gate passed**)
- The learned best checkpoint remained the epoch-zero text anchor: gate
  `2.06e-9`, zero help/harm, Macro-F1 `0.549948`.
- Interpretation: decoupling expert computation did not remove the large
  complementarity ceiling, but a soft logit-interpolation router still failed
  the validation improvement gate. Final-best metrics alone cannot distinguish
  router-score failure from threshold/interpolation failure. Preserve the run
  and inspect all Stage-3 epochs before choosing between a hard calibrated
  selector and a new router representation. Test remains locked.

### R2V-MM-VER-VAL-05A — router distribution-shift diagnosis

- Across router epochs, train decisive pairs were fixed at `2198` helpful
  versus `402` harmful (help:harm `5.47:1`).
- Validation showed the opposite behavior: the best learned epoch helped
  `0.08654` but harmed `0.10508`, with mean visual gate `0.78088` and
  Macro-F1 `0.52586`.
- Cause: the expert/selector train cache injects qrel-positive images, while
  validation uses naturally retrieved candidates. This is appropriate for
  expert supervision but invalid as the router's deployment distribution.
- Resolution: retain the injected cache for Stages 1–2, assemble a second
  train cache from natural direct retrieval for Stage 3, balance decisive
  correction/harm labels, and calibrate a hard specialist-selection threshold
  on validation. Soft logit blending remains a reported ablation.

### R2V-MM-VER-VAL-06 — matched-cache hard-router screen

- Text anchor Macro-F1: `0.549948`; learned hard-router Macro-F1: `0.551186`
  (delta `+0.001238`).
- Accuracy: `0.565247`, exactly one additional correct prediction over the
  `0.564560` text anchor.
- Calibrated threshold: `0.55`; visual route rate: `0.072802` (`106/1456`).
- Visual help/harm: `0.008242 / 0.007555`, corresponding to `12` corrections
  and `11` regressions.
- Soft-router Macro-F1 remained lower at `0.537612`.
- Oracle-router Macro-F1 remained `0.647698`; Visual Select@1 was `0.205575`.
- Decision: **engineering direction passed, scientific improvement gate not
  passed**. The net gain is one example and is not yet a stable or meaningful
  result; do not launch five seeds or open test.
- The reported hard-router ECE `0.138624` is invalidated because the run paired
  hard expert choices with soft-blend probabilities. Exclude it until routed
  probabilities are recomputed consistently.
- Next audit: matched-train helpful/harmful counts and validation decisive-pair
  AUROC/AUPRC of the gate. If scalar ranking remains weak, use out-of-fold
  calibration on frozen conflict/uncertainty features rather than revisiting
  retrieval, selector, or expert training.

### R2V-MM-VER-VAL-06A — hard-router ranking and significance audit

- Hard router: Accuracy `0.565247`, Macro-F1 `0.551186`; deltas over the text
  anchor were only `+0.000687 / +0.001238`.
- It routed `106/1456` validation examples and produced `12` corrections versus
  `11` regressions. Exact McNemar `p=1.0`.
- Helpful-vs-all gate AUROC/AUPRC: `0.496248 / 0.097796`.
- Decisive helpful-vs-harmful AUROC/AUPRC: `0.467828 / 0.425329` over `332`
  decisive examples.
- Mean gate scores were `0.433124` for helpful, `0.440939` for harmful, and
  `0.433190` for neutral cases. The scalar gate therefore did not learn a
  useful ranking; harmful cases ranked slightly above helpful cases.
- Paired Macro-F1 bootstrap delta: mean `+0.001269`, 95% percentile interval
  `[-0.005034, +0.007690]`, probability of a positive delta `0.651`.
- Decision: **scientific gate failed**. Do not tune this score, run repeated
  seeds, or unlock test. Replace it with a cross-fitted set-level utility
  estimator. Select its threshold exclusively on out-of-fold train predictions
  before one frozen validation evaluation.

### R2V-MM-VER-VAL-07 — set-level utility router

- Natural-train utility labels: `786` harmful, `9283` neutral, and `1562`
  helpful.
- Train-OOF threshold `-0.523033` routed `11282/11631` examples (`96.999%`),
  yielding Accuracy `0.819534` and Macro-F1 `0.813366` with `1459` corrections
  versus `578` regressions.
- Frozen validation routed `1400/1456` examples (`96.154%`) but fell to
  Accuracy `0.535714` and Macro-F1 `0.513815`; it made `122` corrections versus
  `164` regressions.
- Helpful-vs-all validation AUROC/AUPRC improved to `0.624981 / 0.247338`, but
  decisive helpful-vs-harmful AUROC/AUPRC remained random at
  `0.496108 / 0.422203`.
- Mean utility scores were `0.254788` for helpful, `0.261918` for harmful, and
  `0.026109` for neutral cases. The model learned decisive-vs-neutral evidence
  disagreement, not the sign of its utility.
- Paired Macro-F1 delta bootstrap: mean `-0.036196`, 95% interval
  `[-0.060030, -0.011899]`, probability positive `0.0016`.
- Decision: **failed**. Cross-fitting only the meta-router is insufficient
  because every natural-train outcome still comes from a visual expert fitted
  on all train labels. The extreme train/validation route-rate mismatch is
  consistent with expert-outcome leakage/overfit. The next scientifically valid
  router screen must generate each train utility target using a fold expert that
  never observed that example.

### R2V-MM-VER-VAL-08 — fully cross-fitted utility router

- Honest fold outcomes contained `1376` harmful, `9072` neutral, and `1183`
  helpful examples. Unlike VAL-07, neither fold text nor fold visual expert had
  observed the example producing its router target.
- Inner train-OOF threshold `0.238257` routed `815/11631` examples (`7.007%`),
  with Accuracy `0.584301`, Macro-F1 `0.569682`, `391` corrections, and `278`
  regressions.
- Frozen validation route rate remained matched at `8.104%` (`118/1456`), so
  the previous gross route-rate shift was resolved. Nevertheless, validation
  fell to Accuracy `0.555632` and Macro-F1 `0.536688`, with `39` corrections
  versus `52` regressions.
- Helpful-vs-all AUROC/AUPRC: `0.515060 / 0.193457`; decisive
  helpful-vs-harmful AUROC/AUPRC: `0.519348 / 0.415944` over `332` examples.
- Paired Macro-F1 bootstrap delta: mean `-0.013405`, 95% interval
  `[-0.026876, -0.000347]`, probability positive `0.0234`.
- Decision: **router hypothesis failed for the current experts**. Cross-fitting
  fixed the protocol but confirmed that available uncertainty/conflict features
  do not predict the sign of visual utility. Freeze routing research and return
  to Stage B expert quality. The next control audits whether learned visual
  attention degrades the upstream Qwen3-VL reranker order.

### R2V-MM-SEL-VAL-01 — learned selector versus upstream order

- Gold-containing validation candidate sets: `574`.
- Upstream reranker order: Hit@1 `0.540070`, Hit@5 `0.785714`, Hit@10
  `0.867596`, MRR `0.652585`, mean first-gold rank `4.5174`.
- Learned attention: Hit@1 `0.205575`, Hit@5 `0.452962`, Hit@10 `0.639373`,
  MRR `0.337707`, mean first-gold rank `9.1289`.
- Upstream order was better on `381` examples, learned order on `86`, with
  `107` ties. Learned-minus-upstream deltas were `-0.334495` Hit@1 and
  `-0.314878` MRR.
- Decision: **selector is the current Stage-B bottleneck**. Replace unconstrained
  learned attention with an upstream reciprocal-rank prior plus a bounded
  learnable residual. Penalize KL divergence from the retrieval prior and
  compare against a retrieval-only control before revisiting the router.

### R2V-MM-SEL-VAL-02 — retrieval-anchored attention controls

| Selector | Select@1 | Visual-expert Macro-F1 | Oracle-router Macro-F1 | Text Macro-F1 |
|---|---:|---:|---:|---:|
| Old unconstrained learned | 0.2056 | 0.5096 | 0.6477 | 0.5499 |
| Retrieval only | **0.5401** | **0.5132** | **0.6485** | 0.5499 |
| Retrieval + KL residual | 0.5453 | 0.5032 | 0.6410 | 0.5499 |

- Retrieval-only attention recovered the strong upstream ordering and slightly
  improved both standalone expert and oracle complementarity over the old
  learned selector.
- The KL-constrained residual gained only `+0.0052` Select@1 over retrieval-only
  while losing `-0.0100` expert Macro-F1 and `-0.0075` oracle Macro-F1.
- Decision: freeze **retrieval-only** as the selector baseline and reject the
  learned residual. Selector ranking is no longer the primary bottleneck;
  diagnose visual-expert help/harm by retrieved-evidence sufficiency before
  changing the residual-verdict objective.

### R2V-MM-EXP-VAL-01 — retrieval-only expert utility strata

- Overall visual expert lost `-0.036791` Macro-F1 versus text, with `134` help
  and `188` harm (exact McNemar `p=0.00308`).
- With gold visual evidence in candidates, it lost `-0.063035` Macro-F1 and
  produced `46` help versus `91` harm (`p=0.000150`).
- Even when the selector ranked a gold image first, the expert lost
  `-0.067998` Macro-F1 with `27` help versus `56` harm (`p=0.00193`).
- Without gold candidates it lost `-0.014099` Macro-F1 (`88` help, `97` harm).
- A qrel-oracle sufficiency policy reached only `0.520129` Macro-F1, below the
  `0.549948` text anchor.
- Decision: **free-form residual verdict fusion is the bottleneck**. Neither
  better selection nor oracle evidence availability rescues it. Replace it
  with an evidence-grounded stance product in which visual evidence affects
  class logits only through supervised support/refute/NEI state and a
  supervised sufficiency gate.

Discovered test metric files: 50

| Result path | Architecture | Seed | Samples | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|
| `outputs/ablations/multi_fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.641520 | 0.641067 |
| `outputs/ablations/multi_independent/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.652946 | 0.652922 |
| `outputs/ablations/multi_typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.644824 | 0.644310 |
| `outputs/ablations/newsclippings_adaptive_screen/multi_adaptive_graph/seed_42/test_metrics.json` | multi_adaptive_graph | 42 | 7264 | 0.637252 | 0.637141 |
| `outputs/ablations/newsclippings_clip_screen/fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.634774 | 0.633860 |
| `outputs/ablations/newsclippings_clip_screen/independent/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.636839 | 0.636569 |
| `outputs/ablations/newsclippings_clip_screen/linear/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.535380 | 0.535328 |
| `outputs/ablations/newsclippings_clip_screen/mlp/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.532902 | 0.532379 |
| `outputs/ablations/newsclippings_clip_screen/typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.631195 | 0.630390 |
| `outputs/ablations/newsclippings_constraints_screen/fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.622384 | 0.621664 |
| `outputs/ablations/newsclippings_constraints_screen/independent/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.631195 | 0.630822 |
| `outputs/ablations/newsclippings_constraints_screen/typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.629268 | 0.629259 |
| `outputs/ablations/newsclippings_counterfactual_screen/multi_independent/seed_42/test_metrics.json` | multi_independent | 42 | 7264 | 0.646063 | 0.645786 |
| `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_100/test_metrics.json` | multi_independent | 100 | 7264 | 0.660242 | 0.660159 |
| `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_13/test_metrics.json` | multi_independent | 13 | 7264 | 0.647990 | 0.647492 |
| `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_21/test_metrics.json` | multi_independent | 21 | 7264 | 0.642483 | 0.642381 |
| `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_42/test_metrics.json` | multi_independent | 42 | 7264 | 0.653634 | 0.653397 |
| `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_87/test_metrics.json` | multi_independent | 87 | 7264 | 0.644273 | 0.644083 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/test_minimal_metrics.json` | multi_independent | 100 | 7264 | 0.652395 | 0.652393 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/val_metrics.json` | multi_independent | 100 | 7024 | 0.652904 | 0.652828 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/test_minimal_metrics.json` | multi_independent | 13 | 7264 | 0.649367 | 0.649227 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/val_metrics.json` | multi_independent | 13 | 7024 | 0.633685 | 0.633419 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/test_minimal_metrics.json` | multi_independent | 21 | 7264 | 0.636977 | 0.636972 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/val_metrics.json` | multi_independent | 21 | 7024 | 0.648918 | 0.648872 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/test_metrics.json` | multi_independent | 42 | 7264 | 0.649917 | 0.647456 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/test_minimal_metrics.json` | multi_independent | 42 | 7264 | 0.649917 | 0.647456 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/val_metrics.json` | multi_independent | 42 | 7024 | 0.646498 | 0.643473 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/test_minimal_metrics.json` | multi_independent | 87 | 7264 | 0.632434 | 0.632434 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/val_metrics.json` | multi_independent | 87 | 7024 | 0.650769 | 0.650733 |
| `outputs/ablations/newsclippings_minimal_reference/multi_independent/seed_42/test_metrics.json` | multi_independent | 42 | 7264 | 0.652946 | 0.652922 |
| `outputs/ablations/newsclippings_multiview_screen/multi_fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.641520 | 0.641067 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/test_metrics.json` | multi_independent | 100 | 7264 | 0.654598 | 0.654230 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/test_minimal_metrics.json` | multi_independent | 100 | 7264 | 0.654598 | 0.654230 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/val_metrics.json` | multi_independent | 100 | 7024 | 0.645074 | 0.644455 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/test_metrics.json` | multi_independent | 13 | 7264 | 0.645925 | 0.645921 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/test_minimal_metrics.json` | multi_independent | 13 | 7264 | 0.645925 | 0.645921 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/val_metrics.json` | multi_independent | 13 | 7024 | 0.644789 | 0.644642 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/test_metrics.json` | multi_independent | 21 | 7264 | 0.645787 | 0.645658 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/test_minimal_metrics.json` | multi_independent | 21 | 7264 | 0.645787 | 0.645658 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/val_metrics.json` | multi_independent | 21 | 7024 | 0.648491 | 0.648203 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_metrics.json` | multi_independent | 42 | 7264 | 0.652946 | 0.652922 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_minimal_metrics.json` | multi_independent | 42 | 7264 | 0.652946 | 0.652922 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/val_metrics.json` | multi_independent | 42 | 7024 | 0.643366 | 0.643283 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/test_metrics.json` | multi_independent | 87 | 7264 | 0.648403 | 0.648147 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/test_minimal_metrics.json` | multi_independent | 87 | 7264 | 0.648403 | 0.648147 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/val_metrics.json` | multi_independent | 87 | 7024 | 0.647637 | 0.647186 |
| `outputs/ablations/newsclippings_multiview_screen/multi_typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.644824 | 0.644310 |
| `outputs/newsclippings_embeddings/test_metrics.json` | legacy | NA | 7264 | 0.631195 | 0.630390 |
| `outputs/validation/r3_minimal/val_metrics.json` | multi_independent | 42 | 7024 | 0.643366 | 0.643283 |
| `outputs/validation/r7_minimal/val_metrics.json` | multi_independent | 42 | 7024 | 0.646498 | 0.643473 |

## `outputs/ablations/multi_fully_connected/seed_42/test_metrics.json`

- Confusion matrix: `[[2459, 1173], [1431, 2201]]`
- Class 0: precision=0.632134, recall=0.677037, F1=0.653815
- Class 1: precision=0.652341, recall=0.606002, F1=0.628319

## `outputs/ablations/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2402, 1230], [1291, 2341]]`
- Class 0: precision=0.650420, recall=0.661344, F1=0.655836
- Class 1: precision=0.655559, recall=0.644548, F1=0.650007

## `outputs/ablations/multi_typed_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2480, 1152], [1428, 2204]]`
- Class 0: precision=0.634596, recall=0.682819, F1=0.657825
- Class 1: precision=0.656734, recall=0.606828, F1=0.630796

## `outputs/ablations/newsclippings_adaptive_screen/multi_adaptive_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2251, 1381], [1254, 2378]]`
- Provenance: `{"architecture": "multi_adaptive_graph", "checkpoint": "outputs/ablations/newsclippings_adaptive_screen/multi_adaptive_graph/seed_42/best.pt", "config_file": "configs/newsclippings_adaptive.yaml", "config_sha256": "0912bf05ed5ce5a2c9dffc5faa38ec282441fb205253735d2fd839a47f1043df", "cuda_runtime": "13.0", "git_commit": "b653494dd131b982108874c8cc5b21d46bd34ef9", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Node mix gate mean `[semantic, entity, temporal, contextual]`: `[0.8219451308250427, 0.22429713606834412, 0.5543850064277649, 0.7950358390808105]`
- Node mix gate std: `[0.2824525535106659, 0.10313897579908371, 3.838471820927225e-05, 0.19432076811790466]`
- Class 0: precision=0.642225, recall=0.619769, F1=0.630797
- Class 1: precision=0.632615, recall=0.654736, F1=0.643485

## `outputs/ablations/newsclippings_clip_screen/fully_connected/seed_42/test_metrics.json`

- Confusion matrix: `[[2487, 1145], [1508, 2124]]`
- Class 0: precision=0.622528, recall=0.684747, F1=0.652157
- Class 1: precision=0.649740, recall=0.584802, F1=0.615563

## `outputs/ablations/newsclippings_clip_screen/independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2412, 1220], [1418, 2214]]`
- Class 0: precision=0.629765, recall=0.664097, F1=0.646475
- Class 1: precision=0.644729, recall=0.609581, F1=0.626663

## `outputs/ablations/newsclippings_clip_screen/linear/seed_42/test_metrics.json`

- Confusion matrix: `[[1983, 1649], [1726, 1906]]`
- Class 0: precision=0.534645, recall=0.545980, F1=0.540253
- Class 1: precision=0.536146, recall=0.524780, F1=0.530402

## `outputs/ablations/newsclippings_clip_screen/mlp/seed_42/test_metrics.json`

- Confusion matrix: `[[1814, 1818], [1575, 2057]]`
- Class 0: precision=0.535261, recall=0.499449, F1=0.516736
- Class 1: precision=0.530839, recall=0.566355, F1=0.548022

## `outputs/ablations/newsclippings_clip_screen/typed_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2462, 1170], [1509, 2123]]`
- Class 0: precision=0.619995, recall=0.677863, F1=0.647639
- Class 1: precision=0.644701, recall=0.584526, F1=0.613141

## `outputs/ablations/newsclippings_constraints_screen/fully_connected/seed_42/test_metrics.json`

- Confusion matrix: `[[2419, 1213], [1530, 2102]]`
- Class 0: precision=0.612560, recall=0.666024, F1=0.638174
- Class 1: precision=0.634087, recall=0.578744, F1=0.605153

## `outputs/ablations/newsclippings_constraints_screen/independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2177, 1455], [1224, 2408]]`
- Class 0: precision=0.640106, recall=0.599394, F1=0.619081
- Class 1: precision=0.623350, recall=0.662996, F1=0.642562

## `outputs/ablations/newsclippings_constraints_screen/typed_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2268, 1364], [1329, 2303]]`
- Class 0: precision=0.630525, recall=0.624449, F1=0.627473
- Class 1: precision=0.628034, recall=0.634086, F1=0.631045

## `outputs/ablations/newsclippings_counterfactual_screen/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2448, 1184], [1387, 2245]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_counterfactual_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_counterfactual.yaml", "config_sha256": "821facb3374a5607c01b16b4f8324efded28dd1465b650724451aae89bfd1182", "cuda_runtime": "13.0", "git_commit": "70c4b8cfa12bf9862648de4bf1adf887896c2caa", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6109581497797357, "counterfactual_verdict_consistency": 0.5008259911894273, "descendant_intervention_accuracy": 0.4821035242290749, "non_descendant_invariance": 0.7676211453744494, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.70181718061674}`
- Class 0: precision=0.638331, recall=0.674009, F1=0.655685
- Class 1: precision=0.654710, recall=0.618117, F1=0.635887

## `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_100/test_metrics.json`

- Confusion matrix: `[[2455, 1177], [1291, 2341]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_directional_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_directional.yaml", "config_sha256": "0edc4ee9fc8887206d055ee7dcc305e840e4ddac8b83f1497f99e7250db39c7a", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6327092511013216, "counterfactual_verdict_consistency": 0.513215859030837, "descendant_intervention_accuracy": 0.45787444933920707, "non_descendant_invariance": 0.7549559471365639, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7243942731277533}`
- Class 0: precision=0.655366, recall=0.675936, F1=0.665492
- Class 1: precision=0.665435, recall=0.644548, F1=0.654825

## `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_13/test_metrics.json`

- Confusion matrix: `[[2490, 1142], [1415, 2217]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_directional_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_directional.yaml", "config_sha256": "0edc4ee9fc8887206d055ee7dcc305e840e4ddac8b83f1497f99e7250db39c7a", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6178414096916299, "counterfactual_verdict_consistency": 0.49531938325991187, "descendant_intervention_accuracy": 0.4754955947136564, "non_descendant_invariance": 0.7421071953010279, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7213656387665198}`
- Class 0: precision=0.637644, recall=0.685573, F1=0.660740
- Class 1: precision=0.660018, recall=0.610407, F1=0.634244

## `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_21/test_metrics.json`

- Confusion matrix: `[[2272, 1360], [1237, 2395]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_directional_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_directional.yaml", "config_sha256": "0edc4ee9fc8887206d055ee7dcc305e840e4ddac8b83f1497f99e7250db39c7a", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6288546255506607, "counterfactual_verdict_consistency": 0.5024779735682819, "descendant_intervention_accuracy": 0.4834801762114537, "non_descendant_invariance": 0.7315528634361234, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7059471365638766}`
- Class 0: precision=0.647478, recall=0.625551, F1=0.636325
- Class 1: precision=0.637816, recall=0.659416, F1=0.648436

## `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2469, 1163], [1353, 2279]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_directional_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_directional.yaml", "config_sha256": "0edc4ee9fc8887206d055ee7dcc305e840e4ddac8b83f1497f99e7250db39c7a", "cuda_runtime": "13.0", "git_commit": "70c4b8cfa12bf9862648de4bf1adf887896c2caa", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6227973568281938, "counterfactual_verdict_consistency": 0.493942731277533, "descendant_intervention_accuracy": 0.4642070484581498, "non_descendant_invariance": 0.7510095447870778, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7268722466960352}`
- Class 0: precision=0.645997, recall=0.679791, F1=0.662463
- Class 1: precision=0.662115, recall=0.627478, F1=0.644331

## `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_87/test_metrics.json`

- Confusion matrix: `[[2424, 1208], [1376, 2256]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_directional_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_directional.yaml", "config_sha256": "0edc4ee9fc8887206d055ee7dcc305e840e4ddac8b83f1497f99e7250db39c7a", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6159140969162996, "counterfactual_verdict_consistency": 0.4944933920704846, "descendant_intervention_accuracy": 0.4688876651982379, "non_descendant_invariance": 0.7354074889867841, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7029185022026432}`
- Class 0: precision=0.637895, recall=0.667401, F1=0.652314
- Class 1: precision=0.651270, recall=0.621145, F1=0.635851

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/test_minimal_metrics.json`

- Confusion matrix: `[[2379, 1253], [1272, 2360]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.637114537444934, "counterfactual_verdict_consistency": 0.32544052863436124, "descendant_intervention_accuracy": 0.47384361233480177, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6422081497797357}`
- Class 0: precision=0.651602, recall=0.655011, F1=0.653302
- Class 1: precision=0.653197, recall=0.649780, F1=0.651484

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/val_metrics.json`

- Confusion matrix: `[[2345, 1167], [1271, 2241]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6341116173120729, "counterfactual_verdict_consistency": 0.31605922551252846, "descendant_intervention_accuracy": 0.46070615034168566, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6446469248291572}`
- Class 0: precision=0.648507, recall=0.667711, F1=0.657969
- Class 1: precision=0.657570, recall=0.638098, F1=0.647688

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/test_minimal_metrics.json`

- Confusion matrix: `[[2431, 1201], [1346, 2286]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6230726872246696, "counterfactual_verdict_consistency": 0.3354900881057269, "descendant_intervention_accuracy": 0.47879955947136565, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6270649779735683}`
- Class 0: precision=0.643633, recall=0.669328, F1=0.656229
- Class 1: precision=0.655578, recall=0.629405, F1=0.642225

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/val_metrics.json`

- Confusion matrix: `[[2320, 1192], [1381, 2131]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6272779043280182, "counterfactual_verdict_consistency": 0.33015375854214124, "descendant_intervention_accuracy": 0.47494305239179957, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6278473804100227}`
- Class 0: precision=0.626858, recall=0.660592, F1=0.643283
- Class 1: precision=0.641288, recall=0.606777, F1=0.623555

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/test_minimal_metrics.json`

- Confusion matrix: `[[2300, 1332], [1305, 2327]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6283039647577092, "counterfactual_verdict_consistency": 0.3310848017621145, "descendant_intervention_accuracy": 0.4887114537444934, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6269273127753304}`
- Class 0: precision=0.638003, recall=0.633260, F1=0.635622
- Class 1: precision=0.635966, recall=0.640694, F1=0.638321

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/val_metrics.json`

- Confusion matrix: `[[2239, 1273], [1193, 2319]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6261389521640092, "counterfactual_verdict_consistency": 0.3364179954441913, "descendant_intervention_accuracy": 0.48604783599088835, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6321184510250569}`
- Class 0: precision=0.652389, recall=0.637528, F1=0.644873
- Class 1: precision=0.645601, recall=0.660308, F1=0.652872

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2664, 968], [1575, 2057]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b337c801a2be8d0ae840222a293d489cc500663a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6349118942731278, "counterfactual_verdict_consistency": 0.31208700440528636, "descendant_intervention_accuracy": 0.4743942731277533, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6398678414096917}`
- Class 0: precision=0.628450, recall=0.733480, F1=0.676915
- Class 1: precision=0.680000, recall=0.566355, F1=0.617996

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/test_minimal_metrics.json`

- Confusion matrix: `[[2664, 968], [1575, 2057]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6349118942731278, "counterfactual_verdict_consistency": 0.31208700440528636, "descendant_intervention_accuracy": 0.4743942731277533, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6398678414096917}`
- Class 0: precision=0.628450, recall=0.733480, F1=0.676915
- Class 1: precision=0.680000, recall=0.566355, F1=0.617996

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/val_metrics.json`

- Confusion matrix: `[[2594, 918], [1565, 1947]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6264236902050114, "counterfactual_verdict_consistency": 0.2984054669703872, "descendant_intervention_accuracy": 0.4635535307517084, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6338268792710706}`
- Class 0: precision=0.623708, recall=0.738610, F1=0.676313
- Class 1: precision=0.679581, recall=0.554385, F1=0.610632

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/test_minimal_metrics.json`

- Confusion matrix: `[[2299, 1333], [1337, 2295]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6150881057268722, "counterfactual_verdict_consistency": 0.3323237885462555, "descendant_intervention_accuracy": 0.4818281938325991, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.621420704845815}`
- Class 0: precision=0.632288, recall=0.632985, F1=0.632636
- Class 1: precision=0.632580, recall=0.631883, F1=0.632231

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/val_metrics.json`

- Confusion matrix: `[[2321, 1191], [1262, 2250]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6227220956719818, "counterfactual_verdict_consistency": 0.33670273348519364, "descendant_intervention_accuracy": 0.47750569476082005, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6380979498861048}`
- Class 0: precision=0.647781, recall=0.660877, F1=0.654264
- Class 1: precision=0.653880, recall=0.640661, F1=0.647203

## `outputs/ablations/newsclippings_minimal_reference/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2402, 1230], [1291, 2341]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "f413e23ca05ea06b9dd0076bce39e2c55cab4842", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.2889592511013216, "descendant_intervention_accuracy": 0.4600770925110132, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6188050660792952}`
- Class 0: precision=0.650420, recall=0.661344, F1=0.655836
- Class 1: precision=0.655559, recall=0.644548, F1=0.650007

## `outputs/ablations/newsclippings_multiview_screen/multi_fully_connected/seed_42/test_metrics.json`

- Confusion matrix: `[[2459, 1173], [1431, 2201]]`
- Class 0: precision=0.632134, recall=0.677037, F1=0.653815
- Class 1: precision=0.652341, recall=0.606002, F1=0.628319

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/test_metrics.json`

- Confusion matrix: `[[2496, 1136], [1373, 2259]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.625, "counterfactual_verdict_consistency": 0.5008259911894273, "descendant_intervention_accuracy": 0.4785242290748899, "non_descendant_invariance": 0.7599118942731278, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7169603524229075}`
- Class 0: precision=0.645128, recall=0.687225, F1=0.665511
- Class 1: precision=0.665390, recall=0.621971, F1=0.642949

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/test_minimal_metrics.json`

- Confusion matrix: `[[2496, 1136], [1373, 2259]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.625, "counterfactual_verdict_consistency": 0.29033590308370044, "descendant_intervention_accuracy": 0.4785242290748899, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6284416299559471}`
- Class 0: precision=0.645128, recall=0.687225, F1=0.665511
- Class 1: precision=0.665390, recall=0.621971, F1=0.642949

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/val_metrics.json`

- Confusion matrix: `[[2412, 1100], [1393, 2119]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6255694760820045, "counterfactual_verdict_consistency": 0.2901480637813212, "descendant_intervention_accuracy": 0.46554669703872437, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6170273348519362}`
- Class 0: precision=0.633903, recall=0.686788, F1=0.659287
- Class 1: precision=0.658279, recall=0.603360, F1=0.629624

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/test_metrics.json`

- Confusion matrix: `[[2359, 1273], [1299, 2333]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6164647577092511, "counterfactual_verdict_consistency": 0.4966960352422907, "descendant_intervention_accuracy": 0.4686123348017621, "non_descendant_invariance": 0.7400881057268722, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7045704845814978}`
- Class 0: precision=0.644888, recall=0.649504, F1=0.647188
- Class 1: precision=0.646977, recall=0.642346, F1=0.644653

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/test_minimal_metrics.json`

- Confusion matrix: `[[2359, 1273], [1299, 2333]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6164647577092511, "counterfactual_verdict_consistency": 0.3025881057268722, "descendant_intervention_accuracy": 0.4686123348017621, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6065528634361234}`
- Class 0: precision=0.644888, recall=0.649504, F1=0.647188
- Class 1: precision=0.646977, recall=0.642346, F1=0.644653

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/val_metrics.json`

- Confusion matrix: `[[2336, 1176], [1319, 2193]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6153189066059226, "counterfactual_verdict_consistency": 0.3078018223234624, "descendant_intervention_accuracy": 0.4772209567198178, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6164578587699316}`
- Class 0: precision=0.639124, recall=0.665148, F1=0.651877
- Class 1: precision=0.650935, recall=0.624431, F1=0.637407

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/test_metrics.json`

- Confusion matrix: `[[2415, 1217], [1356, 2276]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.49807268722466963, "descendant_intervention_accuracy": 0.45759911894273125, "non_descendant_invariance": 0.7452276064610867, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7125550660792952}`
- Class 0: precision=0.640414, recall=0.664923, F1=0.652438
- Class 1: precision=0.651589, recall=0.626652, F1=0.638877

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/test_minimal_metrics.json`

- Confusion matrix: `[[2415, 1217], [1356, 2276]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.2906112334801762, "descendant_intervention_accuracy": 0.45759911894273125, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6095814977973568}`
- Class 0: precision=0.640414, recall=0.664923, F1=0.652438
- Class 1: precision=0.651589, recall=0.626652, F1=0.638877

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/val_metrics.json`

- Confusion matrix: `[[2378, 1134], [1335, 2177]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6326879271070615, "counterfactual_verdict_consistency": 0.29256833712984054, "descendant_intervention_accuracy": 0.4527334851936219, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6225797266514806}`
- Class 0: precision=0.640452, recall=0.677107, F1=0.658270
- Class 1: precision=0.657505, recall=0.619875, F1=0.638136

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2402, 1230], [1291, 2341]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "70c4b8cfa12bf9862648de4bf1adf887896c2caa", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.5063325991189427, "descendant_intervention_accuracy": 0.4600770925110132, "non_descendant_invariance": 0.7406387665198237, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7131057268722467}`
- Class 0: precision=0.650420, recall=0.661344, F1=0.655836
- Class 1: precision=0.655559, recall=0.644548, F1=0.650007

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_minimal_metrics.json`

- Confusion matrix: `[[2402, 1230], [1291, 2341]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.2889592511013216, "descendant_intervention_accuracy": 0.4600770925110132, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6188050660792952}`
- Class 0: precision=0.650420, recall=0.661344, F1=0.655836
- Class 1: precision=0.655559, recall=0.644548, F1=0.650007

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/val_metrics.json`

- Confusion matrix: `[[2313, 1199], [1306, 2206]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6244305239179955, "counterfactual_verdict_consistency": 0.27648063781321186, "descendant_intervention_accuracy": 0.4547266514806378, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.618878132118451}`
- Class 0: precision=0.639127, recall=0.658599, F1=0.648717
- Class 1: precision=0.647871, recall=0.628132, F1=0.637849

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/test_metrics.json`

- Confusion matrix: `[[2453, 1179], [1375, 2257]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.626101321585903, "counterfactual_verdict_consistency": 0.5033039647577092, "descendant_intervention_accuracy": 0.4743942731277533, "non_descendant_invariance": 0.7380690161527166, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7153083700440529}`
- Class 0: precision=0.640805, recall=0.675385, F1=0.657641
- Class 1: precision=0.656868, recall=0.621421, F1=0.638653

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/test_minimal_metrics.json`

- Confusion matrix: `[[2453, 1179], [1375, 2257]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.626101321585903, "counterfactual_verdict_consistency": 0.3032764317180617, "descendant_intervention_accuracy": 0.4743942731277533, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6265143171806168}`
- Class 0: precision=0.640805, recall=0.675385, F1=0.657641
- Class 1: precision=0.656868, recall=0.621421, F1=0.638653

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/val_metrics.json`

- Confusion matrix: `[[2400, 1112], [1363, 2149]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6161731207289294, "counterfactual_verdict_consistency": 0.295130979498861, "descendant_intervention_accuracy": 0.46697038724373574, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6126138952164009}`
- Class 0: precision=0.637789, recall=0.683371, F1=0.659794
- Class 1: precision=0.659000, recall=0.611902, F1=0.634578

## `outputs/ablations/newsclippings_multiview_screen/multi_typed_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2480, 1152], [1428, 2204]]`
- Class 0: precision=0.634596, recall=0.682819, F1=0.657825
- Class 1: precision=0.656734, recall=0.606828, F1=0.630796

## `outputs/newsclippings_embeddings/test_metrics.json`

- Confusion matrix: `[[2462, 1170], [1509, 2123]]`
- Class 0: precision=0.619995, recall=0.677863, F1=0.647639
- Class 1: precision=0.644701, recall=0.584526, F1=0.613141

## `outputs/validation/r3_minimal/val_metrics.json`

- Confusion matrix: `[[2313, 1199], [1306, 2206]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "9eee88950c48d789f0eaf7afbbe69c38d666ac2e", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6244305239179955, "counterfactual_verdict_consistency": 0.27648063781321186, "descendant_intervention_accuracy": 0.4547266514806378, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.618878132118451}`
- Class 0: precision=0.639127, recall=0.658599, F1=0.648717
- Class 1: precision=0.647871, recall=0.628132, F1=0.637849

## `outputs/validation/r7_minimal/val_metrics.json`

- Confusion matrix: `[[2594, 918], [1565, 1947]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "9eee88950c48d789f0eaf7afbbe69c38d666ac2e", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6264236902050114, "counterfactual_verdict_consistency": 0.2984054669703872, "descendant_intervention_accuracy": 0.4635535307517084, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6338268792710706}`
- Class 0: precision=0.623708, recall=0.738610, F1=0.676313
- Class 1: precision=0.679581, recall=0.554385, F1=0.610632
