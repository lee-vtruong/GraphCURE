# Phase B-v2: GraphCURE-SV

Phase B remains open because the frozen official result (`56.80` accuracy,
`54.53` Macro-F1 for the raw ensemble) does not exceed the strongest reported
fixed-corpus Macro-F1 target (`55.60`).  No Phase C or D claim is made yet.

GraphCURE-SV targets the observed error structure rather than increasing model
size.  It decomposes verification into (1) evidence sufficiency: sufficient vs
NEI, and (2) conditional polarity: support vs refute.  Its training loss is

`L = L_verdict + lambda_s L_sufficiency + lambda_p L_polarity`.

For supported/refuted training claims, a weak counterfactual view removes all
annotated evidence from the retrieved set and assigns NEI with a conservative
sample weight.  These rows teach evidence dependence; they are not counted as
new independent claims.  Non-qrel documents can contain unlabeled relevant
evidence, so the counterfactual weight defaults to `0.35`.

## Leakage-control protocol

All B-v2 choices use five duplicate-family-safe folds made solely from the
strict training split.  Neither official validation nor test is read by the
fold builder.  Promote the method only if mean train-only CV Macro-F1 improves
by at least `0.015` and fold standard deviation is at most `0.02`.

Run a one-fold baseline (`lambda_s=lambda_p=counterfactual_ratio=0`, no class
balancing) and a one-fold GraphCURE-SV screen first.  If the challenger wins,
run all five folds.  Only after the five-fold gate passes may one freeze the
configuration and perform one external validation run.  Official test remains
locked during this development cycle.

The required commands are printed in the accompanying handoff; every run saves
`summary.json`, predictions, per-class metrics, loss components, settings, and
an explicit `test_split_used: false` marker.
