# Publication-grade experiment protocol

## Primary claims

1. Minimal interventions improve descendant sensitivity while reducing
   non-descendant sensitivity.
2. Typed compatibility predicts annotated conflicts better than symmetric
   probability divergence.
3. Sequential EVI dominates binary confidence routing on the accuracy-cost
   Pareto frontier.

Each claim has a direct null baseline and must survive five seeds, bootstrap
confidence intervals, and Holm correction across the three primary tests.

## Required splits

- Official NewsCLIPpings splits for comparability.
- Event- and image-disjoint split for leakage analysis.
- MOCHEG official test set for evidence-grounded transfer.
- Counterfactual tuples grouped by source instance; tuple members never cross
  splits.

## Counterfactual metrics

- Descendant Intervention Accuracy (DIA): fraction of annotated descendants
  whose state changes in the required direction.
- Non-descendant Invariance (NDI): fraction of non-descendants whose predicted
  state remains unchanged.
- Counterfactual Verdict Consistency (CVC).

## Acquisition metrics

- Accuracy/Macro-F1 at fixed budgets.
- Area under the accuracy-cost Pareto frontier.
- Regret relative to an oracle with observed evidence outcomes.
- Mean action count, action histogram per manipulation type, stop accuracy.
- Token, monetary, and wall-clock cost reported separately.

## Reviewer-proof ablations

- Same cached backbone for every architecture.
- Parameter-matched MLP and cross-attention baselines.
- Remove each graph edge.
- Fixed versus learned compatibility tensors.
- JS divergence versus typed conflict.
- Counterfactual loss with random interventions as a negative control.
- EVI outcome model versus entropy-per-cost, contextual bandit, random, and
  oracle acquisition.
- Gold versus pseudo constraint labels.

## Failure criteria

Do not claim graph novelty if the parameter-matched cross-attention baseline is
statistically tied. Do not claim causal discovery: the method learns
intervention-faithful predictive dependencies under constructed interventions.
Do not claim cost reduction without quality at equal budget and latency.

