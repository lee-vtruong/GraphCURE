# MOCHEG Phase B13: B12 failure atlas

B13 is a diagnostic-only phase. It does not train a model, tune a threshold,
or select a checkpoint. It uses only the held partition of B12's fresh
train-only fold 0 (`seed=2027`) and refuses official validation or test data.

## Questions

1. How much performance is lost merely by repeating verdict training to match
   the joint model's update count?
2. How much of that optimization loss is recovered by sufficiency, polarity,
   and evidence-ablation supervision?
3. Are the auxiliary heads themselves accurate on their labelled held-fold
   targets?
4. Which labels, sources, retrieval states, confidence bands, and claim-length
   strata account for harmful transitions?
5. What is the oracle complementarity ceiling between the anchor and joint
   candidate?

## Outputs

- `outputs/mocheg_b12_failure_atlas.json`: complete machine-readable audit;
- `outputs/mocheg_b12_failure_atlas.md`: concise report;
- `outputs/mocheg_b12_failure_cases.jsonl`: claim-level casebook, ordered with
  harmful transitions first.

The atlas reports observations, not a promoted B13 architecture. A subsequent
method may be registered only after the dominant measured bottleneck is mapped
to a mechanism supported by prior work. Folds 1--4, official validation, and
test remain locked.

## Run

```bash
python -m scripts.analyze_mocheg_b12_failure_atlas \
  --anchor outputs/mocheg_b12_fresh/fold_0/anchor \
  --control outputs/mocheg_b12_fresh/fold_0/direct_control \
  --joint outputs/mocheg_b12_fresh/fold_0/joint_constraints \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --retrieval outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
  --targets data/processed/mocheg_b6_targets_natural/train.jsonl \
  --fold-spec data/processed/mocheg_b12_folds.json \
  --fold 0
```
