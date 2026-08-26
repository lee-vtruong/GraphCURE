# Phase B-v5: retrieval-anchored context packets

## Decision from B-v4

B-v4 atomic sentence retrieval passed its retrieval gate on validation
(`Recall@8=0.866071`, `MRR=0.680348`, no validation gold injection), but the
seed-42 verifier reached only `0.643463` Macro-F1. This is `-0.027007` versus
the matched seed-42 article verifier (`0.670471`) and fails the preregistered
`0.6735` promotion gate. The trainer's historical `accepted=true` flag used
the obsolete embedding anchor `0.549948`; it is not the B-v4 decision.

The class-level degradation is consistent with evidence fragmentation:
compared with the article verifier, correct Supported predictions fall from
`318` to `300`, Refuted from `434` to `433`, and NEI from `238` to `220`.
Train-only gold injection also increases from about `736` article claims to
`1861` atomic claims, increasing the natural/injected training mismatch.

Do not run additional B-v4 seeds and do not access test.

## B-v5 hypothesis

Preserve the strong claim-conditioned sentence rank, but replace each isolated
sentence with a retrieval-anchored packet containing the selected sentence and
one adjacent sentence on each side. The selected sentence is explicitly
marked; context never crosses an article boundary. Official qrel sentences use
the same window construction, reducing train-injection mismatch.

This changes evidence representation only. The article candidates, dense
weights, diversity policy, verifier, LoRA settings, context budget, seed, and
validation gate remain frozen. Radius `1` is preregistered; do not tune it on
validation.

## Prepare train and validation packets

```bash
python -m scripts.prepare_mocheg_atomic_evidence \
  --manifest-root data/processed/mocheg_manifest_strict \
  --article-retrieval-root outputs/retrieval_mocheg_qwen3_reranked \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --sentence-corpus data/raw/mocheg_dataset/extracted/mocheg/supplementary/Corpus3_sentence_level.csv \
  --output-manifest-root data/processed/mocheg_packet_manifest \
  --output-corpus-root data/processed/mocheg_packet_corpus \
  --output-candidates-root outputs/retrieval_mocheg_packet_candidates \
  --article-top-k 10 --max-units-per-article 32 --context-radius 1 \
  --splits train val \
  2>&1 | tee outputs/mocheg-packet-prepare.log
```

Run validation packet retrieval first. Use a separate cache namespace because
IDs are the same but their encoded text has changed:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_mocheg_atomic_retrieval \
  --manifest-root data/processed/mocheg_packet_manifest \
  --candidate-root outputs/retrieval_mocheg_packet_candidates \
  --corpus-root data/processed/mocheg_packet_corpus \
  --output-root outputs/retrieval_mocheg_packet_dense \
  --cache-root data/processed/retrieval_cache_packets_r1 \
  --model Qwen/Qwen3-Embedding-4B \
  --output-k 8 --max-per-parent 3 \
  --dense-weight 0.65 --lexical-weight 0.20 --parent-weight 0.15 \
  --rrf-k 20 --batch-size 16 --max-length 256 \
  --device cuda --splits val
```

Proceed only if Recall@8 remains at least `0.84`. Then run the identical
command with `--splits train`.

## Seed-42 verifier screen

Use the frozen seed-42 article result as the explicit anchor so the generated
summary cannot report a false promotion:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_lora_verifier \
  --manifest-root data/processed/mocheg_packet_manifest \
  --retrieval-root outputs/retrieval_mocheg_packet_dense \
  --raw-root data/processed/mocheg_packet_corpus \
  --output outputs/mocheg_packet_qwen3_seed42 \
  --top-k 5 --max-length 3072 --max-evidence-chars 2200 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --epochs 3 --patience 2 --batch-size 1 \
  --gradient-accumulation 16 --learning-rate 1e-4 \
  --num-workers 2 --device cuda --seed 42 \
  --anchor-macro-f1 0.670470583003719 --minimum-delta 0.003 \
  2>&1 | tee outputs/mocheg-packet-qwen3-seed42.log
```

Promote only if the summary says `accepted=true` and Macro-F1 is at least
`0.673471`. Otherwise archive B-v5 as another negative ablation.
