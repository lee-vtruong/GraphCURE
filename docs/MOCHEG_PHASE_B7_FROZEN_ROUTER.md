# MOCHEG Phase B7: frozen-anchor selective residual routing

## Motivation

B6-C separated two failure modes on train fold 0. Merely continuing the
fixed-epoch anchor reduced Macro-F1 by `0.016489`; standard auxiliary training
lost another `0.005615` relative to its compute-matched control. Soft conflict
projection recovered only part of the smaller auxiliary loss. B7 therefore
freezes the anchor and performs no additional model optimization.

B7 selects a sparse, deterministic correction policy over already-computed
expert probabilities. The default decision remains the anchor. A claim is
routed only when the expert disagrees and satisfies frozen thresholds on
anchor confidence, expert confidence, confidence advantage, and predicted
class. This is the cheapest causal screen for whether the available experts
contain identifiable, useful corrections.

## Development-only policy grid

The grid is fixed in code before inspecting its result:

- maximum anchor confidence: `0.45, 0.55, 0.65, 0.75, 0.85, 1.01`;
- minimum expert confidence: `0.00, 0.50, 0.60, 0.70, 0.80`;
- minimum expert-minus-anchor confidence: `-0.20, -0.10, 0, 0.10, 0.20`;
- class filters: any, each expert-predicted class, each anchor-predicted class;
- experts: compute control, standard auxiliary, soft-0.25, soft-0.50, and
  severity-adaptive B6-C.

Routes must cover between `1%` and `25%` of fold 0. The selected rule must
improve anchor Macro-F1 by at least `0.005`, have bootstrap probability of a
positive delta at least `0.95`, produce more helpful than harmful changes, and
remain within `-0.002` Macro-F1 of the anchor on every source group. Fold 0 is
development evidence only, so even a passing result requires the exact rule
to be frozen and confirmed on duplicate-safe train folds 1--4.

Official validation and test are never read. Qrels, gold coverage, labels, or
source identity are not routing features; source is used only for a reported
robustness gate after policy selection.

## Server command

This screen uses saved probabilities and requires no GPU:

```bash
cd ~/whale/GraphCURE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv

python -m scripts.analyze_mocheg_b7_frozen_router \
  --output outputs/mocheg_b7_fold0_router_screen.json \
  2>&1 | tee outputs/mocheg-b7-fold0-router-screen.log
```

Inspect the immutable result:

```bash
python - <<'PY'
import json
s = json.load(open("outputs/mocheg_b7_fold0_router_screen.json"))
print("anchor:", s["anchor"])
print("raw experts:", s["raw_experts"])
print("selected policy:", s["selected_policy"])
print("selected metrics:", s["selected_metrics"])
print("comparison:", s["comparison_vs_anchor"])
print("sources:", s["source_diagnostics"])
print("promotion gate:", s["promotion_gate"])
print("validation used:", s["official_validation_used"])
print("test used:", s["test_split_used"])
PY
```

Do not run fold confirmation or official validation/test unless every
promotion-gate field is true.
