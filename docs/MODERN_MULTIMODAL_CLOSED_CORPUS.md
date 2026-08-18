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

Proceed to visual reranking only if conditional Recall@200 is at least `0.75`.
Otherwise first add a complementary image-context text retriever and fuse its
candidates with direct text-to-image retrieval.
