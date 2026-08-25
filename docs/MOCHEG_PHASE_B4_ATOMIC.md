# Phase B-v4: claim-conditioned atomic evidence

## Motivation and hypothesis

The frozen official GraphCURE verifier reaches `54.53` Macro-F1, below the
reported fixed-corpus target `55.60`.  Train-only audits show that scalar
reweighting is not the bottleneck: source-DRO changes overall OOF Macro-F1 by
only `+0.0005`, while source/evidence-availability GroupDRO trades one subgroup
for another.  This branch therefore changes evidence representation rather
than label weights.

Raw fact-check articles contain navigation, source boilerplate, and mutually
irrelevant passages.  B-v4 first retrieves/reranks articles using the already
frozen Qwen3 pipeline, expands only those articles into atomic sentence units,
and ranks the units conditioned on the claim.  The verifier receives a small,
diverse evidence set rather than truncated articles.

The evidence path is:

`Qwen3 article retrieval/rerank -> atomic expansion -> Qwen3 sentence rank ->
diversity pack -> unchanged Qwen3-LoRA verifier`.

This is a representation-only ablation.  Labels, verifier architecture, LoRA
hyperparameters, and official split definitions remain unchanged.

## Leakage controls

- Official sentence IDs come from `Corpus3_sentence_level.csv` and
  sentence-level qrels.
- Gold rows are placed in the atomic corpus so the existing **train-only**
  injection path can read them.
- Preparation and retrieval never inject gold into candidate lists.
- Validation metadata must say `validation_gold_injection: false`.
- Do not read or generate the test split during screening.
- Because the earlier official test result is already known, B-v4 is a new
  preregistered branch. Freeze it on train/validation before any new test run;
  report this history explicitly rather than calling the test untouched.

## Stage 1 — deterministic CPU preparation

```bash
python -m scripts.prepare_mocheg_atomic_evidence \
  --manifest-root data/processed/mocheg_manifest_strict \
  --article-retrieval-root outputs/retrieval_mocheg_qwen3_reranked \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --sentence-corpus data/raw/mocheg_dataset/extracted/mocheg/supplementary/Corpus3_sentence_level.csv \
  --output-manifest-root data/processed/mocheg_atomic_manifest \
  --output-corpus-root data/processed/mocheg_atomic_corpus \
  --output-candidates-root outputs/retrieval_mocheg_atomic_candidates \
  --article-top-k 10 --max-units-per-article 32 \
  --splits train val \
  2>&1 | tee outputs/mocheg-atomic-prepare.log
```

Abort if `missing_gold_sentence_rows` is nonzero.  Record natural coverage but
do not optimize a threshold against labels.

## Stage 2 — validation-only atomic rank screen

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_mocheg_atomic_retrieval \
  --manifest-root data/processed/mocheg_atomic_manifest \
  --candidate-root outputs/retrieval_mocheg_atomic_candidates \
  --corpus-root data/processed/mocheg_atomic_corpus \
  --output-root outputs/retrieval_mocheg_atomic_dense \
  --cache-root data/processed/retrieval_cache \
  --model Qwen/Qwen3-Embedding-4B \
  --output-k 8 --max-per-parent 3 \
  --dense-weight 0.65 --lexical-weight 0.20 --parent-weight 0.15 \
  --rrf-k 20 --batch-size 16 --max-length 256 \
  --device cuda --splits val \
  2>&1 | tee outputs/mocheg-atomic-dense-val.log
```

The screen is worth training only if validation sentence Recall@8 is at least
`0.55`; otherwise audit exact-match coverage and sentence segmentation before
changing the verifier.

## Stage 3 — train retrieval after the validation gate

The document cache is reused.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_mocheg_atomic_retrieval \
  --manifest-root data/processed/mocheg_atomic_manifest \
  --candidate-root outputs/retrieval_mocheg_atomic_candidates \
  --corpus-root data/processed/mocheg_atomic_corpus \
  --output-root outputs/retrieval_mocheg_atomic_dense \
  --cache-root data/processed/retrieval_cache \
  --model Qwen/Qwen3-Embedding-4B \
  --output-k 8 --max-per-parent 3 \
  --dense-weight 0.65 --lexical-weight 0.20 --parent-weight 0.15 \
  --rrf-k 20 --batch-size 16 --max-length 256 \
  --device cuda --splits train \
  2>&1 | tee outputs/mocheg-atomic-dense-train.log
```

## Stage 4 — smoke and one-seed verifier screen

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_lora_verifier \
  --manifest-root data/processed/mocheg_atomic_manifest \
  --retrieval-root outputs/retrieval_mocheg_atomic_dense \
  --raw-root data/processed/mocheg_atomic_corpus \
  --output outputs/mocheg_atomic_qwen3_smoke \
  --limit-train 128 --limit-val 64 \
  --top-k 5 --max-length 2048 --max-evidence-chars 900 \
  --epochs 1 --batch-size 1 --gradient-accumulation 8 \
  --num-workers 0 --device cuda --seed 42 \
  2>&1 | tee outputs/mocheg-atomic-qwen3-smoke.log
```

After a successful smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_lora_verifier \
  --manifest-root data/processed/mocheg_atomic_manifest \
  --retrieval-root outputs/retrieval_mocheg_atomic_dense \
  --raw-root data/processed/mocheg_atomic_corpus \
  --output outputs/mocheg_atomic_qwen3_seed42 \
  --top-k 5 --max-length 3072 --max-evidence-chars 2200 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --epochs 3 --patience 2 --batch-size 1 \
  --gradient-accumulation 16 --learning-rate 1e-4 \
  --num-workers 2 --device cuda --seed 42 \
  2>&1 | tee outputs/mocheg-atomic-qwen3-seed42.log
```

The full screen deliberately restores the frozen verifier context settings
(`3072/2200`). Atomic units are already capped during preparation, so changing
these values is unnecessary and would confound the representation-only
comparison. The smaller `2048/900` setting above is smoke-test-only.

The preregistered one-seed promotion gate is Macro-F1 `>= 0.6735`, i.e.
at least `+0.003` over the original seed-42 validation result `0.670471`.
Only then run seeds `13, 21, 87, 100`; otherwise stop and preserve the result
as a negative representation ablation.
