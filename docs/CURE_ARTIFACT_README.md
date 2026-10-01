# CURE artifact and reproducibility guide

This document describes the runnable research artifact accompanying the CURE manuscript. It is a supplement/repository document, not material to attach to an ACL abstract.

## Scope

The artifact evaluates claim verification on MOCHEG under two P1 tracks:

- **Raw official P1:** the 2,442-claim MOCHEG test split.
- **Strict P1:** the 2,434-claim duplicate-safe subset. Its retrieval is independently inferred because ranked evidence differs from the raw track for some shared IDs.

All canonical test evaluations use frozen checkpoints and policies. Test labels are read only for final metrics and paired statistics, never for checkpoint, seed, top-K, or threshold selection.

## Environment

Run from the repository root on Linux with a CUDA-capable GPU. The reported efficiency benchmark used an NVIDIA GeForce RTX 5090. Create the project environment and install repository dependencies before running experiments. The scripts download or reuse the Hugging Face model IDs they name.

```bash
cd /path/to/GraphCURE
source .venv/bin/activate
```

Expected inputs include:

```text
data/processed/mocheg_manifest/{train,val,test}.jsonl
data/processed/mocheg_manifest_strict/{train,val,test}.jsonl
outputs/retrieval_mocheg_qwen3_reranked/{train,val,test}.jsonl
outputs/retrieval_mocheg_qwen3_reranked_official/test.jsonl
data/raw/mocheg_dataset/extracted/mocheg/{train,val,test}/Corpus2.csv
```

## Canonical AND test audit

The canonical audit uses five frozen direct experts (seeds 13, 21, 42, 87, 100), three frozen rationale-trained experts (seeds 42, 87, 100), and the validation-frozen policy K=5, tau=.49.

```bash
bash scripts/run_mocheg_b18b_canonical_test.sh \
  2>&1 | tee outputs/mocheg_b18b_canonical_test/run_console.log
```

Key outputs:

```text
outputs/mocheg_b18b_canonical_test/canonical_router.md
outputs/mocheg_b18b_canonical_test/canonical_router.json
outputs/mocheg_b18b_canonical_test/raw_and_predictions.jsonl
```

The JSON audit records manifest/retrieval/prediction hashes, paired bootstrap estimates, and raw/strict alignment checks.

## Controls, ablations, and efficiency

```bash
# Mechanism comparisons and direct self-deferral control.
bash scripts/run_mocheg_b18b_reviewer_controls.sh

# Matched headline-recipe direct-only control and raw/strict confirmation.
bash scripts/run_mocheg_b20_headline_recipe_control.sh
bash scripts/run_mocheg_b20_canonical_test.sh

# Qwen2.5-3B versus Qwen2.5-7B rationale-teacher capacity control.
bash scripts/run_mocheg_b21_teacher_capacity.sh

# Frozen-validation component and evidence-size ablations.
bash scripts/run_mocheg_b18b_ablation_a.sh
bash scripts/run_mocheg_b18b_evidence_k_screen.sh
bash scripts/run_mocheg_b18b_evidence_k_comparison.sh

# Sequential-member latency and peak allocated GPU memory.
bash scripts/run_mocheg_b18b_efficiency_benchmark.sh

# CPU-only reporting diagnostics: complementarity, reliability, rank cross-tab,
# paired-design power approximation, and compute--quality plots.
bash scripts/run_mocheg_b18b_additional_diagnostics.sh
```

Validation ablations are for mechanism and policy development, not test results. The `*_canonical_test.sh` scripts are the test-confirmation entry points. Internal output directory names are retained for reproducibility; the manuscript uses descriptive system names.

## Qualitative case cards

The exporter operates only on fixed predictions. It does not invoke a model, change the router, or use gold labels to select a model, policy, or case. Gold is read only to retrospectively place cases in predeclared transition strata.

```bash
python -m scripts.export_mocheg_b18b_qualitative_cases \
  --manifest data/processed/mocheg_manifest/test.jsonl \
  --retrieval outputs/retrieval_mocheg_qwen3_reranked_official/test.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/test/Corpus2.csv \
  --direct-predictions outputs/mocheg_b18b_canonical_test/raw_direct_ensemble_predictions.jsonl \
  --and-predictions outputs/mocheg_b18b_canonical_test/raw_and_predictions.jsonl \
  --output-dir outputs/mocheg_b18b_canonical_test/qualitative_cases_raw \
  --top-k 2 --per-pattern 2
```

The output includes input hashes, stratum counts, and lexicographically selected case IDs. Do **not** feed post-hoc generated explanations back into the model, use them to select cases, or present them as a model output: CURE outputs a verdict and routing decision, not a natural-language explanation.

## Statistics and release

Canonical paired comparisons use claim-level paired percentile bootstrap (10,000 resamples, fixed seed 2026 unless a run states otherwise) and exact McNemar tests. Confidence intervals and probability estimates are descriptive; raw and strict tracks must be reported separately.

Before a public camera-ready release, add the environment lockfile, model/retrieval licenses, MOCHEG access instructions, exact hardware/driver versions, and an archival DOI. Do not redistribute benchmark data or model weights unless their licenses permit it.
