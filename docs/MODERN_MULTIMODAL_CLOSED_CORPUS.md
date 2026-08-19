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
  --cache-root data/processed/mocheg_multimodal_cache \
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
