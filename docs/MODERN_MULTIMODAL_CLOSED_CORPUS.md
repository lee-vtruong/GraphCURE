# GraphCURE-R2V-v2: adaptive multimodal closed-corpus verification

R2V-v1 established a frozen text-retrieved control. It exceeds the HGTMFC
milestone but remains below AMuFC, whose retrieved setting uses both text and
visual evidence with a dedicated visual-necessity Analyzer. R2V-v2 therefore
adds visual evidence without changing the P1 fixed-corpus protocol.

The development order is intentionally gated:

1. validation-only modern visual retrieval;
2. train materialization after the retrieval gate passes;
3. claim-conditioned modality/constraint utility analyzer;
4. multimodal verifier and matched ablations;
5. freeze, then one-shot test evaluation.

## Gate B2-RET-VIS-01

Use the current physical GPU as CUDA device zero. This command reads only the
strict validation manifest and validation image corpus.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_mocheg_visual_retrieval \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --model Qwen/Qwen3-VL-Embedding-2B \
  --output-root outputs/retrieval_mocheg_qwen3vl_images \
  --cache-root data/processed/retrieval_cache \
  --top-k 50 --batch-size 8 --query-batch-size 16 \
  --device cuda --splits val \
  2>&1 | tee outputs/mocheg-qwen3vl-image-retrieval-val.log

cat outputs/retrieval_mocheg_qwen3vl_images/summary.json
```

If GPU memory is insufficient, rerun with `--batch-size 4
--query-batch-size 8`. Cached image embeddings are reused after a completed
encoding pass.

Raw recall counts all claims, including claims without annotated visual
evidence, as misses. `conditional_recall@k` and `conditional_mrr` restrict the
denominator to claims whose gold image identifier exists in the split image
corpus. Both must be reported.

Do not request `--splits test` during this gate.

### Observed result: R2V-VIS-VAL-01

The validation run covered `905/1456` claims with aligned annotated image
evidence. Conditional Recall@1/5/10/50 was
`0.329282 / 0.433149 / 0.477348 / 0.596685`, and conditional MRR was
`0.379245`. Recall@10 passed its `0.40` gate, while Recall@50 missed its `0.65`
gate by `0.053315`. The direct visual retriever is therefore not frozen.

Before introducing a more expensive reranker, reuse the cached image and
query embeddings to measure the reachable candidate ceiling at depth 200:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_mocheg_visual_retrieval \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --model Qwen/Qwen3-VL-Embedding-2B \
  --output-root outputs/retrieval_mocheg_qwen3vl_images_top200 \
  --cache-root data/processed/retrieval_cache \
  --top-k 200 --batch-size 8 --image-shard-size 256 \
  --query-batch-size 16 --score-batch-size 64 \
  --device cuda --splits val \
  2>&1 | tee outputs/mocheg-qwen3vl-image-retrieval-top200-val.log
```

The observed conditional Recall@100/200 was `0.637569 / 0.697238`, so the
`0.75` candidate-ceiling gate failed. Do not rerank or evaluate test yet.

The next validation-only screen uses four leakage-safe query views aligned to
GraphCURE constraints (semantic, entity, temporal/provenance, contextual/OCR)
and reciprocal-rank fusion. It deliberately does not parse claim/topic IDs
from image filenames.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_mocheg_visual_ensemble \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --output-root outputs/retrieval_mocheg_qwen3vl_constraint_ensemble \
  --cache-root data/processed/retrieval_cache \
  --model Qwen/Qwen3-VL-Embedding-2B \
  --candidate-k 300 --output-k 200 --rrf-k 60 \
  --batch-size 8 --image-shard-size 256 --query-batch-size 16 \
  --score-batch-size 32 --device cuda --splits val \
  2>&1 | tee outputs/mocheg-qwen3vl-constraint-ensemble-val.log
```

The ensemble passes if conditional Recall@200 is at least `0.75`. If it does
not, generate pixel-derived captions/OCR and add them as a complementary
retrieval view; do not use filename-derived topic identifiers.

### Observed result: R2V-VIS-VAL-02

The four-view prompt ensemble failed: fused conditional Recall@200 was
`0.6884`, below both the `0.75` gate and the direct-retrieval value `0.6972`.
Prompt variation did not produce sufficiently diverse neighborhoods. The run
is retained as a negative result and excluded from the frozen system.

The next screen generates claim-independent descriptors directly from pixels
with `Qwen/Qwen3-VL-2B-Instruct`. Descriptors include visible entities, scenes,
OCR, numbers, dates, document/meme type, and reuse clues. Generation is JSONL
checkpointed after every batch and may be resumed.

Smoke-test 32 validation images, then resume the same output to completion:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.caption_mocheg_images \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --output-root data/processed/mocheg_visual_descriptors_v3 \
  --cache-root data/processed/retrieval_cache \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --batch-size 4 --max-new-tokens 80 --max-pixels 1003520 \
  --repetition-penalty 1.05 --no-repeat-ngram-size 4 \
  --device cuda --splits val --limit 32

CUDA_VISIBLE_DEVICES=0 python -m scripts.caption_mocheg_images \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --output-root data/processed/mocheg_visual_descriptors_v3 \
  --cache-root data/processed/retrieval_cache \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --batch-size 4 --max-new-tokens 80 --max-pixels 1003520 \
  --repetition-penalty 1.05 --no-repeat-ngram-size 4 \
  --device cuda --splits val
```

After all `12267` descriptors exist, run dense+lexical descriptor retrieval and
fuse it with the direct visual top-200 candidates:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_mocheg_caption_fusion \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --descriptor-root data/processed/mocheg_visual_descriptors_v3 \
  --direct-root outputs/retrieval_mocheg_qwen3vl_images_top200 \
  --output-root outputs/retrieval_mocheg_caption_fusion \
  --cache-root data/processed/retrieval_cache \
  --text-model Qwen/Qwen3-Embedding-4B \
  --candidate-k 300 --output-k 200 --rrf-k 60 \
  --fusion-weights 1.0 1.0 0.5 \
  --batch-size 8 --score-batch-size 32 --device cuda --splits val
```

### Observed result: R2V-VIS-CAP-VAL-02

Direct and caption candidates are complementary: conditional candidate recall
is `0.697238` for direct, `0.706077` for caption, and `0.779006` for their
union. The caption branch recovers 74 annotated gold images missed by direct
retrieval. Fixed RRF reaches only `0.7149` conditional Recall@200, so candidate
generation passes the `0.75` gate while candidate ordering remains the current
bottleneck.

Materialize a larger fused pool using the already cached embeddings:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_mocheg_caption_fusion \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --descriptor-root data/processed/mocheg_visual_descriptors_v3 \
  --direct-root outputs/retrieval_mocheg_qwen3vl_images_top200 \
  --output-root outputs/retrieval_mocheg_caption_fusion_top400 \
  --cache-root data/processed/retrieval_cache \
  --text-model Qwen/Qwen3-Embedding-4B \
  --candidate-k 300 --output-k 400 --rrf-k 60 \
  --fusion-weights 1.0 1.0 0.5 \
  --batch-size 8 --score-batch-size 32 --device cuda --splits val
```

Observed top-400 depth curve: conditional Recall@200/300/400 was
`0.7149 / 0.7348 / 0.7558`. Thus top-400 is the smallest measured pool that
passes the `0.75` candidate-ceiling gate. Its conditional recall retains about
`97.02%` of the `0.779006` full candidate-union ceiling. The next gate is a
two-claim runtime smoke benchmark, not a full validation launch.

Do not run the full 2B reranker before a two-claim smoke test. It scores both
the original pixels and their claim-independent descriptor, displays progress
in claim-image pairs, flushes every completed claim, and supports `--resume`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.rerank_mocheg_visual_qwen3 \
  --retrieval-root outputs/retrieval_mocheg_caption_fusion_top400 \
  --descriptor-root data/processed/mocheg_visual_descriptors_v3 \
  --output-root outputs/retrieval_mocheg_visual_reranker_smoke \
  --model Qwen/Qwen3-VL-Reranker-2B \
  --candidate-k 10 --output-k 10 --batch-size 2 \
  --score-chunk-size 4 --max-length 10240 \
  --limit-claims 2 --device cuda --splits val
```

If the smoke test completes, run validation with a resumable top-400 to top-100
reranking pass:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.rerank_mocheg_visual_qwen3 \
  --retrieval-root outputs/retrieval_mocheg_caption_fusion_top400 \
  --descriptor-root data/processed/mocheg_visual_descriptors_v3 \
  --output-root outputs/retrieval_mocheg_visual_reranked \
  --model Qwen/Qwen3-VL-Reranker-2B \
  --candidate-k 400 --output-k 100 --batch-size 2 \
  --score-chunk-size 16 --max-length 10240 \
  --device cuda --splits val --resume
```

Use a new output root when changing any reranker setting. The run is accepted
only if conditional Recall@10 improves over `0.4873` and conditional Recall@50
improves over `0.6077`, without using test results.

The first two-claim launch used `max_length=1024` and was invalidated before
scoring any pair: Transformers detected that truncation removed expanded image
tokens (`processed_claims=0`). This is an engineering configuration failure,
not an experimental result. The reranker now follows Qwen's official default
cap of `10240`; dynamic padding avoids allocating that full length for every
pair.

The corrected smoke completed two claims and 20 pairs in `27.917` seconds,
including cached model initialization. Interface, checkpointing, metrics, and
the post-scoring-only qrel audit passed. The two-claim retrieval values are not
used for comparison. A 500-pair runtime benchmark is required before the
582,400-pair top-400 validation pass can be authorized.

The 500-pair batch-4 benchmark completed in `78.073` seconds (`6.4043`
pairs/second), projecting `25.26` hours for full top-400 validation. The timing
slice's retrieval values are excluded from comparison. Batch 8 is screened
once before freezing the production throughput setting.

Batch 8 completed the same 500 pairs in `76.829` seconds (`6.5080`
pairs/second), projecting `24.86` hours. It is the frozen throughput setting;
batch 4 remains the OOM-safe resume fallback because batch size does not alter
the semantic experiment signature.

### Observed result: R2V-VIS-RERANK-VAL-01

Full validation reranking completed all `1456` claims. Conditional
Recall@1/5/10/50/100 was
`0.342541 / 0.498343 / 0.550276 / 0.664088 / 0.704972`; conditional MRR was
`0.415227`. Relative to fixed RRF, Recall@10 and Recall@50 improved by about
`0.0630` and `0.0564`, respectively, so both preregistered gates passed. The
reranker is frozen on validation; test remains locked until a multimodal
verifier is frozen.

The scalable training protocol does not rerank all 4.65 million train pairs
with the 2B teacher. Instead, retrieve direct top-50 train hard negatives and
inject qrel positives using training annotations only. This supplies positive
and difficult negative visual evidence for verifier training without using
validation or test labels.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_mocheg_visual_retrieval \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --model Qwen/Qwen3-VL-Embedding-2B \
  --output-root outputs/retrieval_mocheg_qwen3vl_images_train_top50 \
  --cache-root data/processed/retrieval_cache \
  --top-k 50 --batch-size 8 --image-shard-size 256 \
  --query-batch-size 16 --score-batch-size 32 \
  --device cuda --splits train
```

The initial train materialization stopped at shard `284/305` on an image with
aspect ratio `232.0`; the libpng iCCP warning was unrelated and harmless.
Preprocessing now letterboxes only images at or above Qwen's forbidden ratio
of 200 to a safe ratio of 190. Existing completed shards remain valid and are
reused when the identical command is resumed.

### Stage B4: multimodal evidence-set verifier

The completed train retrieval has raw Recall@1/5/10/50 of
`0.137821 / 0.185883 / 0.202734 / 0.250537` and conditional recall of
`0.233435 / 0.314839 / 0.343381 / 0.424348`. This is used as a source of hard
visual negatives. Gold visual candidates are injected and deterministically
shuffled for **training only**. Validation consumes the already frozen
Qwen3-VL reranker output; its qrels are used only to compute auxiliary metrics,
never to build the candidate set.

Assemble frozen text and image features on CPU. The command reads the existing
Qwen image memmaps and normally needs no GPU or model download:

```bash
python -m scripts.cache_mocheg_multimodal_features \
  --manifest-root data/processed/mocheg_manifest_strict \
  --text-cache-root data/processed/mocheg_reasoning_cache \
  --train-visual-retrieval \
    outputs/retrieval_mocheg_qwen3vl_images_train_top50/train.jsonl \
  --val-visual-retrieval \
    outputs/retrieval_mocheg_visual_reranked/val.jsonl \
  --image-cache-root data/processed/retrieval_cache \
  --output-root data/processed/mocheg_multimodal_cache \
  --visual-model Qwen/Qwen3-VL-Embedding-2B \
  --visual-top-k 32 \
  2>&1 | tee outputs/mocheg-multimodal-cache.log
```

Audit the protocol flags before training:

```bash
python - <<'PY'
import json
for split in ("train", "val"):
    path = f"data/processed/mocheg_multimodal_cache/{split}.metadata.json"
    meta = json.load(open(path))
    print(split, "samples=", meta["samples"])
    print("  train_gold_injection=", meta["train_gold_injection"])
    print("  validation_gold_injection=", meta["validation_gold_injection"])
    print("  visual_gold_coverage=", round(meta["visual_gold_coverage"], 6))
PY
```

Required audit: train injection is `True`, validation injection is `False`,
and validation visual coverage is determined by frozen retrieval rather than
qrels. Then run one validation-only screen:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_multimodal_verifier \
  --cache-root data/processed/mocheg_multimodal_cache \
  --output outputs/mocheg_multimodal_residual_seed42 \
  --text-checkpoint outputs/mocheg_cached_verifier_seed42/best.pt \
  --freeze-text-branch \
  --hidden-dim 384 --batch-size 128 \
  --relevance-weight 0.50 \
  --gate-weight 0.10 --visual-gate-target 0.25 \
  --epochs 60 --patience 10 --seed 42 --device cuda \
  2>&1 | tee outputs/mocheg-multimodal-residual-seed42.log
```

The first joint-softmax screen produced Macro-F1 `0.547740`, visual Select@1
`0.048780`, and visual mass `0.691079`. This failed the architecture gate:
candidate count, rather than reliability, dominated modality allocation.

The revised screen normalizes attention within each modality, uses listwise
positive-mass supervision for evidence selection, transfers the exact frozen
text verifier, and learns a visual residual behind a reliability gate. The
screen reports combined and text-only Macro-F1, text/visual Select@1, gate mass,
visual help/harm rates, conflict, and ECE. Continue to five seeds only if:

- combined validation Macro-F1 is at least `0.55` and no lower than its
  embedded text-only branch;
- visual Select@1 is at least `0.15` (well above random `1/32`);
- visual gate mass is at most `0.35`, and gate mass with a gold visual
  candidate is greater than without one; and
- visual help rate is greater than visual harm rate.

The test split remains locked throughout this screen.

### Stage B5: counterfactual visual-utility routing

The residual trajectory showed that the selector itself passes: validation
Visual Select@1 reached `0.216028`. Joint training nevertheless reduced
Macro-F1 because the gate routed more harmful than helpful corrections. The
next preregistered run therefore separates optimization into three stages:

1. train visual evidence selection with listwise qrel supervision;
2. freeze selection and train a full-strength visual expert; and
3. freeze both experts and train only the router. Its train-only soft target is
   the detached reduction in per-example cross-entropy delivered by the visual
   expert, minus a visual-use cost margin.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_staged_multimodal \
  --cache-root data/processed/mocheg_multimodal_router_cache \
  --val-cache-root data/processed/mocheg_multimodal_cache \
  --text-checkpoint outputs/mocheg_cached_verifier_seed42/best.pt \
  --output outputs/mocheg_staged_multimodal_seed42 \
  --hidden-dim 384 --batch-size 128 \
  --selector-epochs 12 --selector-patience 4 \
  --expert-epochs 15 --expert-patience 5 \
  --router-epochs 20 --router-patience 6 \
  --router-cost-margin 0.05 --router-temperature 0.25 \
  --router-target-weight 0.50 \
  --seed 42 --device cuda \
  2>&1 | tee outputs/mocheg-staged-multimodal-seed42.log
```

Model selection begins from a near-zero-gate copy of the frozen text anchor;
therefore a harmful router cannot replace the baseline checkpoint. Report the
visual expert and oracle-router ceilings as diagnostics, but compare only the
learned router as the method result. Proceed to five seeds only if learned
Macro-F1 exceeds the embedded text-only branch, visual help exceeds harm, and
the oracle ceiling confirms at least two Macro-F1 points of available headroom.

The first staged run preserved only the zero-residual expert because Stage 2
incorrectly selected by standalone expert Macro-F1. That criterion is now
replaced by oracle-router Macro-F1 on validation: complementarity, rather than
global standalone strength, selects `expert_best.pt`. This oracle is used only
for development checkpoint selection and ceiling analysis; the deployable gate
still learns exclusively from train losses. Rerun the same command into the new
output root `outputs/mocheg_staged_utility_seed42_v2`.

The corrected expert-selection run exposed a strong complementary ceiling:
text Macro-F1 was `0.549948`, visual-expert Macro-F1 was `0.527767`, but oracle
routing reached `0.647325`. The learned gate still collapsed to text. The next
router is therefore explicitly conflict-aware: the visual expert is computed
independently of the gate, while the gate observes both experts' probabilities,
confidence, entropy, margins, disagreement, evidence entropy, retrieval quality,
and constraint conflict. On train it learns from balanced decisive correction
versus harm pairs; ambiguous samples retain a low-weight loss-reduction target.
Use output root `outputs/mocheg_conflict_router_seed42_v3` and add
`--router-ambiguous-weight 0.10` to the staged command.

The router trajectory then exposed train/deployment shift: injected training
candidates produced `2198` helpful versus only `402` harmful pairs, while
validation harm exceeded help and the gate rose to `0.78`. Build a router-only
train cache that preserves natural retrieval; do not replace the injected
selector/expert cache:

```bash
python -m scripts.cache_mocheg_multimodal_features \
  --manifest-root data/processed/mocheg_manifest_strict \
  --text-cache-root data/processed/mocheg_reasoning_cache \
  --train-visual-retrieval \
    outputs/retrieval_mocheg_qwen3vl_images_train_top50/train.jsonl \
  --image-cache-root data/processed/retrieval_cache \
  --output-root data/processed/mocheg_multimodal_router_cache \
  --visual-model Qwen/Qwen3-VL-Embedding-2B \
  --visual-top-k 32 --train-candidate-policy retrieved --splits train \
  2>&1 | tee outputs/mocheg-multimodal-router-cache.log
```

Its metadata must report `train_gold_injection=false`. Run the staged trainer
with the original injected `--cache-root` and the new natural
`--router-cache-root`. Router model selection now evaluates both soft blending
and hard expert selection over a preregistered `0.00..1.00` threshold grid;
the threshold and routed fraction are persisted in the checkpoint.

The gate-independent expert from the conflict-router run can be reused because
the expert architecture and weights are unchanged. Pass
`--expert-checkpoint outputs/mocheg_conflict_router_seed42_v3/expert_best.pt`
with `--selector-epochs 0 --expert-epochs 0` to train only the matched router.

### Cross-fitted set-level utility router

The matched scalar gate failed its validation ranking audit (helpful-vs-all
AUROC `0.4962`; decisive help-vs-harm AUROC `0.4678`). Threshold tuning cannot
repair a score whose ordering is effectively random. The next screen therefore
freezes the text and visual experts and fits a three-state utility model:
`harmful`, `neutral`, or `helpful`. Features summarize both verdict
distributions, their disagreement, constraint conflict, evidence-attention
concentration, multimodal sufficiency, and per-set retrieval statistics.

The routing threshold is selected using five-fold out-of-fold **training**
predictions and is frozen before validation. This avoids the optimistic
same-validation threshold selection used by the scalar-gate diagnostic. Run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_set_router \
  --expert-checkpoint \
    outputs/mocheg_conflict_router_seed42_v3/expert_best.pt \
  --expert-cache-root data/processed/mocheg_multimodal_cache \
  --router-cache-root data/processed/mocheg_multimodal_router_cache \
  --output outputs/mocheg_set_router_seed42_v5 \
  --device cuda \
  --batch-size 256 \
  --folds 5 \
  --neutral-weight 0.25 \
  --harm-penalty 1.0 \
  --bootstrap-iterations 5000 \
  --seed 42 \
  2>&1 | tee outputs/mocheg-set-router-seed42-v5.log
```

This remains a validation-only screen. Proceed to repeated seeds only if its
frozen validation Macro-F1 improves the text anchor by at least `0.01`, the
bootstrap interval is directionally convincing, and decisive gate AUROC is at
least `0.60`. Test remains locked.

The first set-router screen failed because cross-fitting the meta-router did
not make the underlying expert outcomes honest: both experts had already seen
all training labels. Its OOF train Macro-F1 reached an implausible `0.8134` and
the selected policy routed `97.0%` of train versus `96.2%` of validation, where
Macro-F1 collapsed to `0.5138`. Decisive help-vs-harm AUROC was `0.4961`.

Build fold predictions in which neither the text anchor nor the visual expert
has observed the held-out example. Fixed epoch counts avoid using fold labels
for early stopping. The job writes each fold atomically and resumes completed
folds after interruption:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.build_mocheg_crossfit_router_features \
  --text-cache-root data/processed/mocheg_reasoning_cache \
  --full-expert-checkpoint \
    outputs/mocheg_conflict_router_seed42_v3/expert_best.pt \
  --expert-cache-root data/processed/mocheg_multimodal_cache \
  --router-cache-root data/processed/mocheg_multimodal_router_cache \
  --output-root data/processed/mocheg_crossfit_router_features_v6 \
  --folds 5 \
  --text-epochs 2 \
  --selector-epochs 8 \
  --expert-epochs 4 \
  --batch-size 128 \
  --device cuda \
  --seed 42 \
  2>&1 | tee outputs/mocheg-crossfit-router-features-v6.log
```

Then fit the utility router on the honest fold outcomes. Its inner OOF layer
selects the threshold using train only; validation remains a single frozen
evaluation:

```bash
python -m scripts.train_mocheg_set_router \
  --feature-cache-root data/processed/mocheg_crossfit_router_features_v6 \
  --output outputs/mocheg_crossfit_set_router_seed42_v6 \
  --device cpu \
  --folds 5 \
  --neutral-weight 0.25 \
  --harm-penalty 1.0 \
  --bootstrap-iterations 5000 \
  --seed 42 \
  2>&1 | tee outputs/mocheg-crossfit-set-router-seed42-v6.log
```

### Retrieval-anchored visual attention

The selector audit found that learned attention destroyed the upstream visual
reranker order: conditional Hit@1 fell from `0.5401` to `0.2056` and MRR from
`0.6526` to `0.3377`. The revised selector treats reciprocal rank as a prior:

`attention = softmax(log(reciprocal_rank) / temperature + scale * residual)`.

A KL trust region prevents the residual from repeating the unconstrained
ranking collapse. Screen a retrieval-only control and the proposed residual
variant on validation; keep routing disabled:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_staged_multimodal \
  --cache-root data/processed/mocheg_multimodal_router_cache \
  --val-cache-root data/processed/mocheg_multimodal_cache \
  --text-checkpoint outputs/mocheg_cached_verifier_seed42/best.pt \
  --output outputs/mocheg_retrieval_attention_control_seed42_v7 \
  --visual-attention-mode retrieval \
  --visual-prior-temperature 0.5 \
  --selector-epochs 4 --selector-patience 4 \
  --expert-epochs 12 --expert-patience 5 \
  --router-epochs 0 \
  --device cuda --batch-size 128 --seed 42 \
  2>&1 | tee outputs/mocheg-retrieval-attention-control-v7.log

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_staged_multimodal \
  --cache-root data/processed/mocheg_multimodal_cache \
  --text-checkpoint outputs/mocheg_cached_verifier_seed42/best.pt \
  --output outputs/mocheg_retrieval_residual_seed42_v7 \
  --visual-attention-mode retrieval_residual \
  --visual-prior-temperature 0.5 \
  --visual-residual-scale 0.10 \
  --selector-prior-kl-weight 1.0 \
  --selector-epochs 10 --selector-patience 4 \
  --expert-epochs 12 --expert-patience 5 \
  --router-epochs 0 \
  --device cuda --batch-size 128 --seed 42 \
  2>&1 | tee outputs/mocheg-retrieval-residual-v7.log
```

Compare `stages.selector.best_val_select_at_1`, visual-expert Macro-F1, and
oracle-router Macro-F1. Test remains locked.

The retrieval-only expert utility audit then showed that free-form residual
fusion remained harmful even when the top-ranked image was annotated gold
(`-0.0680` Macro-F1; `27` help versus `56` harm). Replace the verdict residual
with a stance-product expert. Its visual branch is explicitly supervised to
predict support/refute/NEI on relevant images; a balanced sufficiency head
controls whether centered visual log-probabilities may modify frozen text
logits. No-gold cases therefore have an architectural path back to text:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_staged_multimodal \
  --cache-root data/processed/mocheg_multimodal_router_cache \
  --val-cache-root data/processed/mocheg_multimodal_cache \
  --text-checkpoint outputs/mocheg_cached_verifier_seed42/best.pt \
  --output outputs/mocheg_stance_product_seed42_v8 \
  --visual-attention-mode retrieval \
  --visual-prior-temperature 0.5 \
  --visual-expert-mode stance_product \
  --visual-stance-scale 1.0 \
  --selector-stance-weight 1.0 \
  --selector-epochs 10 --selector-patience 10 \
  --expert-sufficiency-weight 1.0 \
  --expert-epochs 12 --expert-patience 5 \
  --residual-penalty 0 \
  --router-epochs 0 \
  --device cuda --batch-size 128 --seed 42 \
  2>&1 | tee outputs/mocheg-stance-product-v8.log
```

In addition to expert/oracle Macro-F1, report visual-stance Macro-F1 on
gold-containing candidate sets and sufficiency AUROC/AUPRC. If stance itself is
weak, global frozen image embeddings are inadequate for verdict relation
modeling and the next expert must use token-level claim-image cross-attention.

### Token-level claim--image cross-attention gate

The v8 screen triggered that gate: gold-candidate stance Macro-F1 was `0.3407`
and sufficiency AUROC was `0.4581`, despite retrieval Select@1 of `0.5401`.
Run the SigLIP2 token/patch interaction screen first on a smoke subset, then on
full validation. Gold is injected only into training; validation is untouched.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_token_visual_expert \
  --train-retrieval outputs/retrieval_mocheg_qwen3vl_images_train_top50/train.jsonl \
  --val-retrieval outputs/retrieval_mocheg_visual_reranked/val.jsonl \
  --output outputs/mocheg_token_visual_smoke_v9 \
  --top-k 4 --batch-size 2 --epochs 1 --num-workers 2 \
  --limit-train 64 --limit-val 64 --device cuda

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_token_visual_expert \
  --train-retrieval outputs/retrieval_mocheg_qwen3vl_images_train_top50/train.jsonl \
  --val-retrieval outputs/retrieval_mocheg_visual_reranked/val.jsonl \
  --output outputs/mocheg_token_visual_seed42_v9 \
  --top-k 4 --batch-size 4 --epochs 6 --patience 2 \
  --negative-weight 0.25 --claim-weight 1.0 \
  --num-workers 4 --device cuda --seed 42 \
  2>&1 | tee outputs/mocheg-token-visual-v9.log
```

Do not fuse this expert or unlock test unless validation gold-candidate stance
Macro-F1 reaches `0.50` and relevance AUROC reaches `0.65`. These are component
gates, not paper claims.

The v9 token/patch screen failed both gates (`0.1848` stance Macro-F1 and
`0.3804` relevance AUROC). Do not tune or unfreeze it. Generate resumable,
claim-conditioned visual constraint reports for the next analyzer--verifier
screen. The prompt receives neither labels nor qrel flags; train may inject one
annotated image while validation remains natural retrieval.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.analyze_mocheg_claim_images \
  --train-retrieval outputs/retrieval_mocheg_qwen3vl_images_train_top50/train.jsonl \
  --val-retrieval outputs/retrieval_mocheg_visual_reranked/val.jsonl \
  --output-root data/processed/mocheg_claim_visual_reports_v10 \
  --top-k 2 --batch-size 4 --max-pixels 501760 \
  --splits train val --device cuda \
  2>&1 | tee outputs/mocheg-claim-visual-reports-v10.log
```

After both report summaries are complete, encode them with the same frozen
Qwen3 embedding model used by the text verifier:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.cache_mocheg_visual_report_features \
  --text-cache-root data/processed/mocheg_reasoning_cache \
  --report-root data/processed/mocheg_claim_visual_reports_v10 \
  --output-root data/processed/mocheg_visual_report_cache_v10 \
  --encoder Qwen/Qwen3-Embedding-0.6B \
  --batch-size 32 --device cuda --splits train val \
  2>&1 | tee outputs/mocheg-visual-report-cache-v10.log
```

Screen the report expert with the frozen text teacher. The first reported
validation point must reproduce the text anchor; routing remains disabled:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_staged_multimodal \
  --cache-root data/processed/mocheg_visual_report_cache_v10 \
  --text-checkpoint outputs/mocheg_cached_verifier_seed42/best.pt \
  --output outputs/mocheg_visual_report_expert_seed42_v10 \
  --visual-attention-mode learned \
  --visual-expert-mode stance_product \
  --visual-stance-scale 1.0 \
  --selector-stance-weight 0.25 \
  --selector-epochs 8 --selector-patience 3 \
  --expert-sufficiency-weight 1.0 \
  --expert-epochs 12 --expert-patience 4 \
  --residual-penalty 0 --router-epochs 0 \
  --batch-size 128 --device cuda --seed 42 \
  2>&1 | tee outputs/mocheg-visual-report-expert-v10.log
```

If report selection is strong but qrel-coverage sufficiency fails, do not train
another sufficiency classifier. Freeze both experts and screen a conservative
logit adapter. It is initialized near the text anchor, penalizes report use,
and is accepted only above a predeclared `+0.003` validation Macro-F1 delta:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_report_fusion \
  --cache-root data/processed/mocheg_visual_report_cache_v10 \
  --expert-checkpoint outputs/mocheg_visual_report_expert_seed42_v10/best.pt \
  --output outputs/mocheg_report_fusion_seed42_v11 \
  --epochs 40 --patience 8 --batch-size 256 \
  --gate-cost 0.005 --anchor-kl 0.02 --minimum-delta 0.003 \
  --device cuda --seed 42 \
  2>&1 | tee outputs/mocheg-report-fusion-v11.log
```

The safe adapter reverted to text (`best delta 0.0`). Stop visual late-fusion
tuning. Phase B now targets the actual verifier bottleneck with a long-context
cross-encoder. Screen text-only first so any later report contribution is an
isolated ablation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_long_context_verifier \
  --retrieval-root outputs/retrieval_mocheg_qwen3_reranked \
  --output outputs/mocheg_modernbert_text_seed42_v12 \
  --model answerdotai/ModernBERT-large \
  --top-k 5 --max-length 2048 --max-evidence-chars 3000 \
  --batch-size 2 --gradient-accumulation 8 \
  --epochs 5 --patience 2 --device cuda --seed 42 \
  2>&1 | tee outputs/mocheg-modernbert-text-v12.log
```

The ModernBERT screen failed (`0.4421` Macro-F1). Replace its randomly
initialized classifier with an NLI/FEVER-pretrained pair verifier and correct
the train/validation evidence-coverage mismatch using train-only injection:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_nli_set_verifier \
  --retrieval-root outputs/retrieval_mocheg_qwen3_reranked \
  --output outputs/mocheg_nli_set_seed42_v13 \
  --top-k 5 --max-length 512 \
  --batch-size 2 --gradient-accumulation 8 \
  --epochs 4 --patience 2 --learning-rate 5e-6 \
  --pair-loss-weight 0.25 --weak-negative-weight 0.25 \
  --device cuda --seed 42 \
  2>&1 | tee outputs/mocheg-nli-set-v13.log
```

The NLI screen failed (`0.4781` Macro-F1; pair Macro-F1 `0.4764`). Its error is
already present at pair level, so replacing only the aggregator is not a valid
next step. Screen a bounded set-interaction residual on the established frozen
anchor instead. Epoch zero must reproduce `0.5499479` exactly; otherwise stop.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_selective_residual \
  --cache-root data/processed/mocheg_reasoning_cache \
  --anchor-checkpoint outputs/mocheg_cached_verifier_seed42/best.pt \
  --output outputs/mocheg_selective_residual_seed42_v14 \
  --hidden-dim 192 --layers 2 --heads 4 \
  --epochs 60 --patience 10 --batch-size 128 \
  --learning-rate 2e-4 --anchor-correct-kl 0.5 \
  --global-kl 0.05 --gate-cost 0.01 --anchor-error-weight 1.5 \
  --minimum-delta 0.003 --device cuda --seed 42 \
  2>&1 | tee outputs/mocheg-selective-residual-v14.log
```

Only `mode=selective_residual` with `accepted=true` advances. Otherwise the
script emits `mode=anchor_fallback`, preserving the frozen result and ending
this residual family without test access.

The residual screen correctly fell back: its best delta was only `+0.000134`.
Do not tune its gate. Build one aligned, train-only grounded cache instead.
Gold insertion uses a stable hash to avoid a fixed rank shortcut; validation
continues to use the untouched natural cache.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.cache_mocheg_reasoning_features \
  --manifest-root data/processed/mocheg_manifest_strict \
  --retrieval-root outputs/retrieval_mocheg_qwen3_reranked \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --output-root data/processed/mocheg_reasoning_cache_grounded_v15 \
  --encoder Qwen/Qwen3-Embedding-0.6B \
  --top-k 8 --batch-size 32 --max-length 256 \
  --device cuda --splits train --inject-train-gold \
  2>&1 | tee outputs/mocheg-reasoning-cache-grounded-v15.log
```

Fine-tune from the exact frozen anchor on paired natural and grounded views.
The natural-view KL is applied only where the teacher is correct, while the
grounded view supplies evidence selection, stance, and sufficiency targets.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_retrieval_curriculum \
  --natural-cache-root data/processed/mocheg_reasoning_cache \
  --grounded-cache-root data/processed/mocheg_reasoning_cache_grounded_v15 \
  --anchor-checkpoint outputs/mocheg_cached_verifier_seed42/best.pt \
  --output outputs/mocheg_retrieval_curriculum_seed42_v15 \
  --epochs 40 --patience 8 --batch-size 256 \
  --learning-rate 5e-5 --natural-verdict-weight 0.35 \
  --anchor-correct-kl 0.5 --relevance-weight 0.25 \
  --stance-weight 0.15 --sufficiency-weight 0.15 \
  --minimum-delta 0.003 --device cuda --seed 42 \
  2>&1 | tee outputs/mocheg-retrieval-curriculum-v15.log
```

Epoch zero must reproduce the anchor. Advance to multi-seed validation only if
`accepted=true`; otherwise keep the anchor and close this curriculum branch.

The curriculum candidate failed (`0.540604` Macro-F1). The 610 injected rows
cannot repair the split-level annotation mismatch. Stop training shallow heads
on frozen sentence embeddings and perform one controlled instruction-model
screen. Qwen3-4B-Instruct-2507 is run in non-thinking mode; LoRA targets its
attention and MLP projections. The verifier emits a single constrained token
(`A/B/C`), so evaluation uses exact next-token probabilities rather than
sampling or fragile JSON parsing.

Install the one new optional dependency:

```bash
python -m pip install "peft>=0.17" "accelerate>=1.0"
```

Run a small integration/OOM smoke test first:

Compatibility note: the verifier normalizes recent `tokenizers.Encoding`
chat-template outputs before collation. Runs that failed
with `Could not infer dtype of tokenizers.Encoding` performed no training and
must be rerun; downloaded model weights remain cached.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_lora_verifier \
  --output outputs/mocheg_qwen3_lora_smoke_v16 \
  --limit-train 128 --limit-val 64 \
  --top-k 5 --max-length 2048 --max-evidence-chars 1500 \
  --epochs 1 --batch-size 1 --gradient-accumulation 8 \
  --num-workers 0 --device cuda --seed 42 \
  2>&1 | tee outputs/mocheg-qwen3-lora-smoke-v16.log
```

If the smoke test completes without OOM and reports three distinct label token
IDs, run the locked validation screen:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_lora_verifier \
  --output outputs/mocheg_qwen3_lora_seed42_v16 \
  --top-k 5 --max-length 3072 --max-evidence-chars 2200 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --epochs 3 --patience 2 --batch-size 1 \
  --gradient-accumulation 16 --learning-rate 1e-4 \
  --num-workers 2 --device cuda --seed 42 \
  2>&1 | tee outputs/mocheg-qwen3-lora-v16.log
```

Do not run the test split. The validation gate remains `+0.003` Macro-F1 over
the frozen `0.549948` anchor before any multi-seed or multimodal extension.
