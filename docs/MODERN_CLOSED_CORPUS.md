# GraphCURE-R2V: modern fixed-corpus verification

This is the Phase-B/P1 expert: retrieval is allowed only from the frozen local
MOCHEG corpus. It is **closed-corpus**, not parametric closed-book and not live
open-web search. Gold qrels are used for train-time supervision and evaluation,
never as model input at validation/test time.

## Why the previous stack is now a baseline

The KES pipeline (dense ensemble, BGE-M3 reranking, dense hard negatives) is a
valid retrieval baseline, but it does not model whether a candidate is useful
for *reasoning*, whether multiple pieces of evidence are jointly sufficient, or
whether high similarity hides a contradiction. The replacement is:

1. **Reasoning-aware hybrid retrieval.** Qwen3-Embedding retrieves candidates
   with a fact-verification instruction; a lexical view protects exact names,
   dates, numbers, and negation. Reciprocal-rank fusion avoids mixing
   uncalibrated raw scores.
2. **Instruction-aware reranking.** Qwen3-Reranker scores usefulness for
   support, refutation, or resolution rather than topical similarity.
3. **Structured hard negatives.** High-ranked non-qrel candidates are tagged as
   negation, quantity, lexical-overlap, or semantic traps. They supervise
   evidence utility conservatively because MOCHEG qrels may be incomplete.
4. **Claim-level evidence-set verification.** A shared modern encoder feeds a
   claim-conditioned selector. The model jointly predicts evidence utility,
   stance, sufficiency, and one verdict per claim. It does not copy the claim
   label onto every retrieved paragraph.

This design is informed by the open Qwen3 embedding/reranking release,
ReasonIR's reasoning-focused queries and plausible-but-unhelpful negatives,
and recent fact-checking systems that emphasize claim decomposition,
cross-modal alignment, and evidence relevance:

- <https://qwenlm.github.io/blog/qwen3-embedding/>
- <https://arxiv.org/abs/2504.20595>
- <https://arxiv.org/abs/2601.04720>
- <https://proceedings.mlr.press/v267/braun25b.html>
- <https://ojs.aaai.org/index.php/AAAI/article/view/32684>

Using newer backbones is not itself the paper novelty. The candidate-level
reasoning traps, joint utility/sufficiency objective, and later cost-aware
routing are the ablations that must establish the contribution.

## Stage gates

Do not train the expensive verifier until the retrieval gates pass.

| Gate | Validation criterion | Test use |
|---|---:|---|
| candidate retrieval | validation Recall@50 and MRR exceed the frozen dense baseline | run test once after choices freeze |
| reranking | validation Recall@10 improves without a material MRR loss | run test once after choices freeze |
| P1 verifier | validation Macro-F1 improves over the old verifier | compare to retrieved-evidence papers |
| final target | exceed HGTMFC retrieved Acc 48.61 / Macro-F1 46.78 under the same protocol | five seeds |

The final target is a research target, not a guaranteed outcome.

## Server runbook

Activate the existing environment:

```bash
cd ~/whale/GraphCURE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv
```

Update the retrieval dependencies and optionally authenticate Hugging Face:

```bash
python -m pip install -U "transformers>=4.51" "sentence-transformers>=5.0" accelerate
export HF_TOKEN="YOUR_READ_TOKEN"
```

First run only validation on GPU 1. Document embeddings are cached. Do not use
the official test qrels for architecture or hyperparameter selection.

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.run_mocheg_reasoning_retrieval \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --dense-model Qwen/Qwen3-Embedding-4B \
  --output-root outputs/retrieval_mocheg_qwen3_hybrid \
  --candidate-k 200 --output-k 50 \
  --batch-size 8 --splits val \
  2>&1 | tee outputs/mocheg-qwen3-retrieval-screen.log

cat outputs/retrieval_mocheg_qwen3_hybrid/summary.json
```

If that gate passes, rerank the same two splits. Reduce `--batch-size` to 2 if
memory is tight; reduce `--max-length` to 1024 if throughput is too low.

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.rerank_mocheg_qwen3 \
  --retrieval-root outputs/retrieval_mocheg_qwen3_hybrid \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --model Qwen/Qwen3-Reranker-4B \
  --output-root outputs/retrieval_mocheg_qwen3_reranked \
  --candidate-k 50 --top-k 10 --batch-size 4 --max-length 1536 \
  --splits val \
  2>&1 | tee outputs/mocheg-qwen3-rerank-screen.log

cat outputs/retrieval_mocheg_qwen3_reranked/summary.json
```

Only after both validation screens pass, generate train outputs and structured
negatives:

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.run_mocheg_reasoning_retrieval \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --dense-model Qwen/Qwen3-Embedding-4B \
  --output-root outputs/retrieval_mocheg_qwen3_hybrid \
  --candidate-k 200 --output-k 50 --batch-size 8 --splits train

CUDA_VISIBLE_DEVICES=1 python -m scripts.rerank_mocheg_qwen3 \
  --retrieval-root outputs/retrieval_mocheg_qwen3_hybrid \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --model Qwen/Qwen3-Reranker-4B \
  --output-root outputs/retrieval_mocheg_qwen3_reranked \
  --candidate-k 50 --top-k 10 --batch-size 4 --max-length 1536 \
  --splits train

python -m scripts.mine_mocheg_reasoning_negatives \
  --retrieval outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
  --output data/processed/mocheg_reasoning_negatives/train.jsonl
```

Cache the frozen encoder once. This prevents the top-k evidence encoder from
being recomputed with backward passes in every epoch.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.cache_mocheg_reasoning_features \
  --manifest-root data/processed/mocheg_manifest_strict \
  --retrieval-root outputs/retrieval_mocheg_qwen3_reranked \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --encoder Qwen/Qwen3-Embedding-0.6B \
  --output-root data/processed/mocheg_reasoning_cache \
  --top-k 8 --batch-size 32 --max-length 256 \
  --splits train val \
  2>&1 | tee outputs/mocheg-reasoning-cache.log
```

Train only the claim-conditioned evidence-set head during model screening:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_cached_verifier \
  --cache-root data/processed/mocheg_reasoning_cache \
  --output outputs/mocheg_cached_verifier_seed42 \
  --batch-size 256 --epochs 60 --patience 10 --seed 42 \
  2>&1 | tee outputs/mocheg-cached-verifier-seed42.log

cat outputs/mocheg_cached_verifier_seed42/val_metrics.json
```

Once the seed-42 validation gate passes, confirm stability with the fixed seed
set. Existing completed seeds are reused, and both machine-readable JSON and a
paper-ready Markdown table are written. This runner is validation-only and has
no test evaluation option.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_mocheg_cached_validation \
  --cache-root data/processed/mocheg_reasoning_cache \
  --seeds 13 21 42 87 100 --skip-existing \
  2>&1 | tee outputs/mocheg-cached-verifier-multiseed.log

cat outputs/mocheg_cached_verifier_summary_val.md
```

The accepted Phase-B configuration is frozen in
`configs/mocheg_r2v_frozen.json`. Do not change its retrieval, cache, verifier,
or seed settings after generating test artifacts. The one-shot evaluator checks
the validation gate, frozen seeds, and test-cache architecture before running.

The end-to-end encoder trainer is retained as an experimental LoRA/fine-tuning
path, but must not be used for the initial screen: it repeatedly encodes nine
sequences per sample and is prohibitively slow on the current shared GPU.

The key verifier diagnostics are `retrieval_gold_coverage`,
`evidence_selection_hit_at_1`, and `ece_10`, in addition to Accuracy and
Macro-F1. These let the paper attribute a failure to retrieval, selection, or
classification instead of reporting only an end metric.

After all architecture and hyperparameter choices are frozen, generate the
test retrieval/reranking files once and unlock checkpoint-only evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_mocheg_reasoning_retrieval \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --dense-model Qwen/Qwen3-Embedding-4B \
  --output-root outputs/retrieval_mocheg_qwen3_hybrid \
  --candidate-k 200 --output-k 50 --batch-size 8 --splits test

CUDA_VISIBLE_DEVICES=0 python -m scripts.rerank_mocheg_qwen3 \
  --retrieval-root outputs/retrieval_mocheg_qwen3_hybrid \
  --manifest-root data/processed/mocheg_manifest_strict \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --model Qwen/Qwen3-Reranker-4B \
  --output-root outputs/retrieval_mocheg_qwen3_reranked \
  --candidate-k 50 --top-k 10 --batch-size 4 --max-length 1536 \
  --splits test

CUDA_VISIBLE_DEVICES=0 python -m scripts.cache_mocheg_reasoning_features \
  --manifest-root data/processed/mocheg_manifest_strict \
  --retrieval-root outputs/retrieval_mocheg_qwen3_reranked \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --encoder Qwen/Qwen3-Embedding-0.6B \
  --output-root data/processed/mocheg_reasoning_cache \
  --top-k 8 --batch-size 32 --max-length 256 --splits test

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_cached_verifier \
  --cache-root data/processed/mocheg_reasoning_cache \
  --output outputs/mocheg_cached_verifier_seed42 \
  --checkpoint outputs/mocheg_cached_verifier_seed42/best.pt \
  --evaluate-test

cat outputs/mocheg_cached_verifier_seed42/test_metrics.json
```

For the frozen five-seed report, use the guarded evaluator instead of selecting
a single checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_mocheg_frozen_r2v \
  --freeze configs/mocheg_r2v_frozen.json \
  --validation-summary outputs/mocheg_cached_verifier_summary_val.json \
  --cache-root data/processed/mocheg_reasoning_cache \
  --skip-existing \
  2>&1 | tee outputs/mocheg-r2v-frozen-test.log

cat outputs/mocheg_cached_verifier_summary_test.md
```
