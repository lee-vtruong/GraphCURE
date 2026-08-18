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
