# MOCHEG B14: five-fold direct-curriculum failure atlas

## Why this comes before another model

The frozen B13 hierarchical blend failed confirmation. Its direct verdict is
better than the anchor in some held folds and worse in others, so aggregate
confirmation-fold performance alone is not sufficient evidence for promotion.
B14 diagnoses this heterogeneity before defining another intervention.

## Locked protocol

- Inputs: held predictions from all five disjoint seed-2027 train-only folds.
- Fold 0: B12 anchor and B13 curriculum screen.
- Folds 1--4: completed B13 confirmation runs.
- Official validation: not read.
- Test: not read.
- Gold injection on held examples: forbidden.
- Hyperparameter or probability-weight search: none.
- Purpose: post-failure diagnosis, not model selection or a performance claim.

## Command

```bash
python -m scripts.analyze_mocheg_b14_direct_curriculum_atlas \
  --output outputs/mocheg_b14_direct_curriculum_atlas.json \
  --markdown outputs/mocheg_b14_direct_curriculum_atlas.md \
  --cases outputs/mocheg_b14_direct_curriculum_cases.jsonl \
  2>&1 | tee outputs/mocheg-b14-direct-curriculum-atlas.log
```

## Required decision rule

Do not retune B13 on these five folds. Use the atlas to formulate one bounded
B14 hypothesis tied to a repeated harmful slice. Then generate a new locked
train-only fold assignment before screening that hypothesis. Official
validation and test stay locked until the new method passes fresh-fold gates.

## Atlas-derived fixed policy diagnostic

The first bounded candidate is asymmetric NEI escape: retain every determinate
anchor prediction and use the curriculum expert only when the anchor predicts
NEI while the expert predicts supported or refuted. There is no confidence
threshold or interpolation weight. Because this rule was derived from the
five-fold atlas, its evaluation on those folds remains exploratory. It is
eligible for preregistration only if its fixed diagnostic gate passes; actual
confirmation requires a new fold assignment.

```bash
python -m scripts.analyze_mocheg_b14_nei_escape_policy \
  2>&1 | tee outputs/mocheg-b14-nei-escape-diagnostic.log
```

The fixed policy failed: accuracy increased by `+0.011263`, but Macro-F1
increased only `+0.002091`, NEI F1 fell `-0.028590`, Politifact regressed, and
bootstrap probability was only `0.7958`. Its effect depended strongly on
unobservable qrel availability. B15 therefore measures whether value of an
expert call can instead be predicted from observable signals under five-way
cross-fitting. The value threshold is frozen at `0.5`, and source/qrel/gold
features are prohibited.
