# MOCHEG Phase B13: compute-neutral constraint curriculum

## Evidence-based motivation

The B12 failure atlas rules out blind architecture search. Extra optimization
reduced Macro-F1 by `0.019145`, while auxiliary constraints recovered only
`0.005382` (28.1%) of that loss. Of 207 harmful anchor-to-joint changes, 150
(72.5%) changed a correct supported/refuted verdict into NEI. The sufficiency
head marked 282 of 1120 truly sufficient examples as insufficient (25.2%). The
polarity head was also asymmetric: 94 of 320 supported cases were predicted as
refuted, versus 43 of 800 refuted cases predicted as supported. Qrel-absent and
Politifact cases were the most damaged groups.

## Frozen B13 hypothesis

B13 keeps the exact optimizer-update budget of the verdict anchor. In epoch 1,
25% of verdict rows are replaced by labelled auxiliary rows; half are
sufficiency and half polarity. Sufficiency sampling is 70% sufficient to target
the measured false-insufficient failure, and polarity sampling is balanced
50/50 to target the supported/refuted skew. Epochs 2 and 3 are verdict-only
recovery. Missing auxiliary labels remain masked. Ablation examples are omitted
because they add more insufficient targets to the observed NEI collapse.

Inference remains a single direct verdict (`hierarchical weight = 0`). Thus a
gain cannot come from a threshold fitted on the held fold. Development uses the
already locked duplicate-safe seed-2027 fold 0; official validation and test
remain untouched.

## Frozen gate

- Macro-F1 at least `+0.005` over the standard anchor;
- accuracy no worse than `-0.002`;
- paired bootstrap probability of positive Macro-F1 delta at least `0.95`;
- helpful changes exceed harmful changes;
- each source remains within `-0.002` Macro-F1 of the anchor.

Only a passing fold-0 candidate may be confirmed on folds 1--4.
