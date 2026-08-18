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

