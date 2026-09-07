# MOCHEG Phase B11: provenance-conditioned disagreement calibration

B10 improved aggregate OOF Macro-F1 but harmed both source strata, suggesting
a mixture-prior effect. B11 tests a distinct, GraphCURE-aligned hypothesis:
provenance is a contextual constraint, so uncertainty calibration should be
conditioned on source rather than forcing one global decision boundary.

For each held fold 1--4, B11 trains one fixed calibrator per source using only
the other three folds. Features and logistic-regression settings are identical
to B10. Source is explicitly declared as an inference feature; qrels, gold
coverage, labels, official validation, and test are unavailable at inference.

## Frozen exploratory gate

- `+0.003` Macro-F1 over the unweighted ensemble;
- `+0.002` over the B10 global calibrator;
- positive ensemble delta on at least three of four folds;
- bootstrap positive probability versus ensemble at least `0.95`;
- accuracy within `-0.002` of the ensemble;
- every source must improve, with more helpful than harmful changes.

Because B11 was motivated after inspecting B10 on these folds, a pass is only
exploratory evidence and requires confirmation using a newly generated fold
assignment. It does not authorize official validation or test.

Run without GPU:

```bash
python -m scripts.analyze_mocheg_b11_provenance_calibrator \
  --output outputs/mocheg_b11_provenance_calibrator.json \
  --predictions outputs/mocheg_b11_provenance_predictions.jsonl \
  2>&1 | tee outputs/mocheg-b11-provenance-calibrator.log
```

## Frozen outcome

B11 failed every primary promotion criterion. Its Macro-F1 was `0.665340`,
which was `-0.001757` below the unweighted ensemble and `-0.005202` below the
global calibrator. Only two folds improved, accuracy fell by `-0.008061`, and
there were `418` helpful versus `493` harmful changes. Snopes also declined
by `-0.001492`. No fresh confirmation, official validation, or test was run.
The calibration/seed-stacking branch is closed.
