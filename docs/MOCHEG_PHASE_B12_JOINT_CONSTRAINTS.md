# MOCHEG Phase B12: joint constraint supervision from base initialization

## Motivation

B6 showed that continuing a converged verdict adapter causes most of the
damage, while B9--B11 showed that seed averaging and calibration cannot provide
a stable SOTA-sized gain. B12 changes the training trajectory rather than the
decision threshold: verdict, evidence-sufficiency, polarity, and
evidence-ablation tasks jointly train a new LoRA from the base model.

The candidate is compared against two controls on a new duplicate-safe fold
assignment (`seed=2027`): the standard verdict anchor and a from-base
verdict-only model repeated to exactly the same number of training examples as
the joint candidate. All models use checkpoint epoch 3. Hierarchical inference
is disabled (`weight=0`), so any gain must be learned in the shared verifier,
not selected by a held-fold blend.

## Frozen gate

- joint Macro-F1 at least `+0.005` over the standard anchor;
- at least `+0.003` over the compute-matched direct control;
- bootstrap positive probability versus control at least `0.95`;
- accuracy and every source within `-0.002` of the standard anchor;
- more helpful than harmful changes versus control.

The new fold assignment is required because previous train folds informed the
hypothesis. Official validation and test remain locked.

## Prepare the fresh folds

```bash
python -m scripts.prepare_mocheg_sv_folds \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --folds 5 --seed 2027 \
  --output data/processed/mocheg_b12_folds.json
```

The full server commands are provided in the associated handoff. Run the
standard anchor, then the joint candidate, then its count-matched direct-only
control. Do not run folds 1--4 unless the fold-0 gate passes.
