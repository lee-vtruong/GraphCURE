# Phase C2: auditable open-web evidence constraints

## Frozen C1 result

The validation snapshot contains 1,456 claims and 13,377 evidence rows. It
used 1,481 search calls (1.017 per claim), expanded 25 claims, retained a
94.44% usable-evidence rate and a median of eight domains per claim. It used no
benchmark gold evidence and did not access test.

C1 is frozen before transformation. This freeze is intentionally scoped to
retrieval and **does not unlock test**. Phase C can unlock test only after the
complete verifier and all selection rules have been frozen on validation.

```bash
python -m scripts.freeze_mocheg_open_web_snapshot \
  --snapshot-root outputs/mocheg_c1_serper_adaptive100_2026_09_14 \
  --split val \
  --api-key-env SERPER_API_KEY
```

## C2a: observable constraint table

C2a is deterministic and performs no network or model inference. For every
claim/evidence pair it records:

- canonical URL, domain, HTTPS and coarse source family;
- query view and per-view rank;
- fetched-text status and observable text quality;
- heuristic proper-name candidates and claim/evidence entity overlap;
- temporal mentions and literal temporal overlap;
- explicit `unscored` stance and sufficiency fields.

The source family is descriptive, not a credibility label. Entity extraction
is a deterministic candidate extractor, not a trained NER system. Temporal
overlap is not temporal entailment. These conservative names prevent proxy
features from being reported as semantic judgments.

```bash
python -m scripts.build_mocheg_open_evidence_table \
  --snapshot-root outputs/mocheg_c1_serper_adaptive100_2026_09_14 \
  --output-root data/processed/mocheg_open_web_c2a \
  --split val \
  2>&1 | tee outputs/mocheg-c2a-build.log

cat data/processed/mocheg_open_web_c2a/summary.json
```

The builder verifies the frozen row hash before reading data, excludes labels
from its output, and refuses a development freeze containing test or gold
evidence.

## Gate before C2b

- C1 freeze has no failures and `unlocks_test=false`;
- C2a output has exactly 1,456 unique claim IDs;
- input row hash matches the C1 freeze;
- minimum usable evidence remains at least five per claim;
- source-family and overlap distributions are plausible under manual audit;
- stance and sufficiency remain unscored.

C2b will then run a frozen claim-evidence judge for stance, sufficiency,
entity consistency and temporal consistency. C2b must include a direct-verdict
control and will be selected only on validation; it must not silently treat
the C2a heuristics as ground-truth constraint labels.

## C2b frozen constraint scorer

C2b uses a frozen instruction model and next-token A/B/C probabilities. Each
claim/evidence pair is scored independently for stance, sufficiency, entity
consistency and temporal consistency. The output contains no benchmark label
or gold evidence. Run a 32-claim smoke test before full validation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.score_mocheg_open_constraints \
  --shortlist data/processed/mocheg_open_web_c2b_shortlist/val.jsonl \
  --audit outputs/mocheg_c2a_audit.json \
  --output-root outputs/mocheg_c2b_constraints_smoke \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tasks stance sufficiency entity temporal \
  --top-k 8 \
  --max-evidence-chars 1800 \
  --max-length 2304 \
  --batch-size 4 \
  --device cuda \
  --limit 32 \
  2>&1 | tee outputs/mocheg-c2b-constraints-smoke.log
```

Smoke promotion requires complete output, 32 claims, four balanced task
counts, finite three-way probabilities summing to one, no label/gold/test use,
and throughput measured before scheduling the full 46k pair-task workload.

## C2c matched-verifier causal screen

C2c freezes the Phase-B seed-42 LoRA verifier and evaluates two matched
inference conditions. The direct control sees the claim and the same top-eight
web evidence. The treatment additionally sees C2b stance, sufficiency, entity
and temporal distributions. Both conditions use the same model, adapter, raw
evidence budget, label tokens and decoding. The scorer never loads validation
labels. A separate analysis command loads labels only after both prediction
files are frozen.

The primary comparison is constraint-aware versus direct. A 50/50 probability
ensemble is declared in advance and reported as a diagnostic; its weight must
not be tuned on validation. Promotion requires Macro-F1 delta >= 0.003,
bootstrap probability of positive delta >= 0.95, more helpful than harmful
changes, no negative source group, and non-inferiority to the frozen closed
anchor when that anchor is supplied.

Run the 32-claim scorer smoke test with the frozen Phase-B seed-42 adapter:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.score_mocheg_open_verdicts \
  --shortlist data/processed/mocheg_open_web_c2b_shortlist/val.jsonl \
  --audit outputs/mocheg_c2a_audit.json \
  --constraint-root outputs/mocheg_c2b_constraints_val \
  --output-root outputs/mocheg_c2c_verdicts_smoke \
  --adapter outputs/mocheg_qwen3_lora_seed42_v16/best_adapter \
  --top-k 8 --max-evidence-chars 1000 --max-length 4096 \
  --batch-size 2 --device cuda --limit 32
```

Remove `--limit 32` and use output root `outputs/mocheg_c2c_verdicts_val`
for the frozen full run. Then evaluate both conditions:

```bash
python -m scripts.analyze_mocheg_open_verdicts \
  --manifest data/processed/mocheg_manifest_strict/val.jsonl \
  --verdict-root outputs/mocheg_c2c_verdicts_val \
  --anchor-predictions \
    outputs/mocheg_qwen3_lora_seed42_v16/val_predictions.jsonl \
  --output outputs/mocheg_c2c_analysis.json \
  --predictions-output outputs/mocheg_c2c_predictions.jsonl
```

If the treatment fails promotion, generate the post-hoc failure atlas before
defining C3. It measures oracle complementarity, help/harm transitions and
quartiles of observable evidence/constraint signals. It is diagnostic only;
it must not be used to claim a validation result or tune a router on the same
examples.

```bash
python -m scripts.analyze_mocheg_c2c_failure_atlas \
  --manifest data/processed/mocheg_manifest_strict/val.jsonl \
  --anchor-predictions \
    outputs/mocheg_qwen3_lora_seed42_v16/val_predictions.jsonl \
  --c2c-predictions outputs/mocheg_c2c_predictions.jsonl \
  --shortlist data/processed/mocheg_open_web_c2b_shortlist/val.jsonl \
  --constraint-scores \
    outputs/mocheg_c2b_constraints_val/constraint_scores.jsonl \
  --output outputs/mocheg_c2c_failure_atlas.json
```

## C3 preregistered hypothesis

The C2c atlas showed high anchor/open oracle complementarity but unsafe open
replacement, dominated by correct supported/refuted predictions moving to
NEI. C3 therefore tests evidence selection rather than validation-fitted
routing. It restores the Phase-B verifier's training format (top five items,
up to 2200 characters each) and removes inline constraint annotations. The
matched control keeps the original top five. The treatment uses a fixed,
label-free cascade: safe decisive, decisive relevant, decisive,
non-irrelevant, rank fallback, then duplicate relaxation. Within every stage,
original search rank is preserved.

Generate and audit both evidence sets before any verdict inference:

```bash
python -m scripts.prepare_mocheg_c3_evidence_selection \
  --shortlist data/processed/mocheg_open_web_c2b_shortlist/val.jsonl \
  --audit outputs/mocheg_c2a_audit.json \
  --constraint-root outputs/mocheg_c2b_constraints_val \
  --output-root data/processed/mocheg_open_web_c3_selection \
  --top-k 5
```

## C2a failure audit and C2b shortlist

Before judge inference, measure weak claims, social-source concentration,
conditional temporal coverage and duplicated text. Construct a top-8 shortlist
that preserves search rank while preferring usable text, limiting one result
per domain and at most two social results. The fallback stages are recorded;
no source is assigned a learned or hand-written credibility score.

```bash
python -m scripts.analyze_mocheg_open_evidence_table \
  --table-root data/processed/mocheg_open_web_c2a \
  --split val \
  --output outputs/mocheg_c2a_audit.json \
  --shortlist-output data/processed/mocheg_open_web_c2b_shortlist/val.jsonl \
  --top-k 8 \
  --minimum-text-chars 80 \
  --max-per-domain 1 \
  --max-social 2 \
  2>&1 | tee outputs/mocheg-c2a-audit.log
```
