# MOCHEG Phase B10: cross-fitted seed-disagreement calibrator

## Hypothesis

B9 found a small, source-safe aggregate gain from seed diversity, but an
unweighted probability mean failed confirmation. B10 tests whether the
pattern of seed agreement and uncertainty contains a stable correction signal
that a fixed low-capacity calibrator can learn.

This is not a post-hoc weighting search. For each held fold among train folds
1--4, a standardized multinomial logistic regression is trained only on the
other three folds. Its fixed features are per-seed log probabilities, mean,
standard deviation, entropy, maximum confidence, and class vote fractions.
`C=1`, balanced class weights, and LBFGS are fixed before execution.

Every prediction is therefore out-of-fold for both the Qwen verifier and the
calibrator. Fold 0, official validation, and test are not read.

## Promotion gate

- at least `+0.005` Macro-F1 over seed 42;
- at least `+0.003` over the frozen unweighted ensemble;
- positive calibrator-versus-ensemble delta on at least three of four folds;
- bootstrap positive probability versus ensemble at least `0.95`;
- accuracy and every source within `-0.002` of the ensemble;
- more helpful than harmful changes versus the ensemble.

If this gate passes, fit one final calibrator on all folds 1--4 OOF rows and
freeze it before external validation. If it fails, close seed-level stacking;
do not tune `C`, features, or class weights on these predictions.

## Run

No GPU is required:

```bash
python -m scripts.analyze_mocheg_b10_crossfit_calibrator \
  --output outputs/mocheg_b10_crossfit_calibrator.json \
  --predictions outputs/mocheg_b10_crossfit_predictions.jsonl \
  2>&1 | tee outputs/mocheg-b10-crossfit-calibrator.log
```

Inspect:

```bash
python - <<'PY'
import json
s = json.load(open("outputs/mocheg_b10_crossfit_calibrator.json"))
print("per fold:", s["per_fold"])
print("aggregate:", json.dumps(s["aggregate"], indent=2))
print("gate:", json.dumps(s["promotion_gate"], indent=2))
print("fold 0 used:", s["fold0_used"])
print("official validation used:", s["official_validation_used"])
print("test used:", s["test_split_used"])
PY
```
