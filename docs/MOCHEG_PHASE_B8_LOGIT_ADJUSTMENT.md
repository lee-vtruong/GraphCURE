# MOCHEG Phase B8: prior-robust logit adjustment

## Why this experiment

B7 showed that sparse corrections exist, but its `+0.004604` Macro-F1 gain
missed the fixed `+0.005` gate and its bootstrap positive probability was
`0.9294`, below `0.95`. Continuing to tune B7 thresholds on the same fold
would be post-hoc overfitting.

B8 instead tests a separate hypothesis: part of the remaining error is a
stable class-prior/decision-boundary mismatch. The frozen anchor probabilities
receive two identifiable additive logit offsets, one for `supported` and one
for `NEI`; `refuted` remains zero. No model is trained and no expert is used.

## Preregistered fold-0 screen

- Supported and NEI offsets each use `[-0.30, -0.25, ..., 0.30]`.
- Accuracy may fall by at most `0.002`.
- Every source must remain within `-0.002` Macro-F1 of its anchor.
- Promotion requires Macro-F1 delta at least `+0.005`, bootstrap probability
  of a positive delta at least `0.95`, and more helpful than harmful changes.
- Source is a robustness diagnostic only, never an inference input.
- Official validation and test remain locked.

The grid and gates are fixed before the result is observed. Passing fold 0 is
not a paper result; the exact two offsets must then be confirmed unchanged on
duplicate-safe train folds 1--4.

## Run

```bash
cd ~/whale/GraphCURE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv

python -m scripts.analyze_mocheg_b8_logit_adjustment \
  --output outputs/mocheg_b8_fold0_logit_adjustment.json \
  2>&1 | tee outputs/mocheg-b8-fold0-logit-adjustment.log
```

Inspect:

```bash
python - <<'PY'
import json
s = json.load(open("outputs/mocheg_b8_fold0_logit_adjustment.json"))
print("anchor:", s["anchor"])
print("adjustment:", s["selected_adjustment"])
print("selected:", s["selected_metrics"])
print("comparison:", s["comparison_vs_anchor"])
print("sources:", s["source_diagnostics"])
print("gate:", s["promotion_gate"])
print("official validation used:", s["official_validation_used"])
print("test used:", s["test_split_used"])
PY
```

Do not run confirmation folds or official validation/test unless every gate
field is true.

## Frozen fold-0 outcome

B8 failed and is closed. The selected offsets were `+0.30` for supported,
`0` for refuted, and `+0.15` for NEI. They changed accuracy from `0.657069`
to `0.659648` and Macro-F1 from `0.641063` to `0.643223` (`+0.002160`).
There were `25` helpful and `19` harmful changes, but bootstrap positive
probability was only `0.7604` with interval `[-0.003905, 0.007970]`.

Both sources improved and the accuracy gate passed, but the Macro-F1 and
bootstrap gates failed. No confirmation fold, official validation, or test
was used. The grid is not widened after observing that the supported offset
reached its upper boundary.
