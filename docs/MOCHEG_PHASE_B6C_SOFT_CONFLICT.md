# MOCHEG Phase B6-C: train-only soft conflict projection

## Why B6-C exists

B6-A showed a useful auxiliary signal but failed its frozen stability gate.
B6-B showed that full PCGrad was active, yet complete removal of every
conflicting auxiliary component was too aggressive. B6-C tests partial and
conflict-severity projection without reusing official validation for method
selection.

For a verdict gradient `g_v` and auxiliary gradient `g_a`, a conflicting
component is adjusted as

`g_a' = g_a - lambda * <g_a,g_v>/||g_v||^2 * g_v`.

Fixed soft projection uses a constant `lambda`. Severity-adaptive projection
uses `lambda * min(1, -cos(g_a,g_v)/tau)`, so near-zero conflicts receive only
a small correction. Full B6-B PCGrad is the special case `lambda=1` without a
temperature.

## Leakage controls

- Configuration selection is restricted to duplicate-family-safe train fold
  0. Official validation and test are not read.
- Fold anchors and B6-C continuations use a fixed checkpoint epoch. Held-fold
  Macro-F1 is never used for early stopping or checkpoint selection.
- Training targets may contain train-only gold injection. Held-fold targets
  come from a separate natural target root and must report zero injection.
- Hierarchical inference weight is fixed to zero. The experiment tests a
  training-time regularizer, not a held-fold blend search.
- If the fold-0 gate passes, the exact winning configuration is frozen before
  folds 1--4. Fold 0 is development evidence, not confirmation evidence.

## Fold-0 development screen

The preregistered candidates are:

- `soft_025`: fixed projection strength `0.25`;
- `soft_050`: fixed projection strength `0.50`;
- `severity_010`: maximum strength `1.0`, temperature `0.10`.

Promotion to train-only folds 1--4 requires the selected candidate to use
conflict projection, beat its fold anchor, improve over the compute-matched
direct control by at least `0.003` Macro-F1, and improve over standard
auxiliary training by at least `0.002`. These thresholds must not be changed
after seeing fold-0 output.

## Server runbook

Activate the environment and create natural held-fold targets once:

```bash
cd ~/whale/GraphCURE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv

python -m scripts.prepare_mocheg_sufficiency_targets \
  --manifest-root data/processed/mocheg_manifest_strict \
  --retrieval-root outputs/retrieval_mocheg_qwen3_reranked \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --output-root data/processed/mocheg_b6_targets_natural \
  --top-k 5 --splits train
```

Run the fixed-epoch fold-0 anchor in tmux:

```bash
tmux new -s mocheg-b6c

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_lora_verifier \
  --manifest-root data/processed/mocheg_manifest_strict \
  --retrieval-root outputs/retrieval_mocheg_qwen3_reranked \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --fold-spec data/processed/mocheg_sv_folds.json --fold-index 0 \
  --fixed-checkpoint-epoch 3 --epochs 3 \
  --output outputs/mocheg_b6c_oof/fold_0/anchor \
  --seed 42 --batch-size 1 --gradient-accumulation 16 \
  --learning-rate 1e-4 --max-length 3072 \
  2>&1 | tee outputs/mocheg-b6c-fold0-anchor.log
```

Run standard auxiliary first because its task count defines the
compute-matched control:

```bash
COMMON="--target-root data/processed/mocheg_b6_targets \
--held-target-root data/processed/mocheg_b6_targets_natural \
--raw-root data/raw/mocheg_dataset/extracted/mocheg \
--fold-spec data/processed/mocheg_sv_folds.json --fold-index 0 \
--fixed-checkpoint-epoch 3 --epochs 3 --patience 2 \
--initial-adapter outputs/mocheg_b6c_oof/fold_0/anchor/best_adapter \
--anchor-predictions outputs/mocheg_b6c_oof/fold_0/anchor/val_predictions.jsonl \
--seed 42 --batch-size 1 --gradient-accumulation 16 \
--learning-rate 3e-5 --max-length 3072 --hierarchical-weights 0"

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_hierarchical_lora \
  $COMMON --gradient-mode standard \
  --output outputs/mocheg_b6c_oof/fold_0/standard_auxiliary \
  2>&1 | tee outputs/mocheg-b6c-fold0-standard.log
```

Run the compute-matched verdict-only control:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_hierarchical_lora \
  $COMMON --gradient-mode standard \
  --sufficiency-loss-weight 0 --polarity-loss-weight 0 \
  --ablation-loss-weight 0 --ablation-ratio 0 \
  --match-training-examples-from \
    outputs/mocheg_b6c_oof/fold_0/standard_auxiliary/summary.json \
  --output outputs/mocheg_b6c_oof/fold_0/direct_control \
  2>&1 | tee outputs/mocheg-b6c-fold0-control.log
```

Run the three frozen soft-conflict candidates:

```bash
for spec in \
  "soft_025 0.25 none" \
  "soft_050 0.50 none" \
  "severity_010 1.00 0.10"
do
  set -- $spec
  name="$1"; strength="$2"; temperature="$3"
  EXTRA=""
  if [ "$temperature" != "none" ]; then
    EXTRA="--conflict-temperature $temperature"
  fi
  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_hierarchical_lora \
    $COMMON --gradient-mode pcgrad \
    --projection-strength "$strength" $EXTRA \
    --output "outputs/mocheg_b6c_oof/fold_0/$name" \
    2>&1 | tee "outputs/mocheg-b6c-fold0-$name.log"
done
```

Finally run the immutable fold-0 screen:

```bash
python -m scripts.analyze_mocheg_b6c_oof_screen \
  --output outputs/mocheg_b6c_fold0_screen.json \
  2>&1 | tee outputs/mocheg-b6c-fold0-screen.log
```

Do not run official validation/test. Do not start folds 1--4 until
`promotion_gate.passed` is true; the screen JSON supplies the configuration
that must be frozen for confirmation.

## Frozen fold-0 outcome

B6-C failed and is closed without folds 1--4 or official validation/test.
All runs used `n=2327` held-fold claims, checkpoint epoch 3, hierarchical
inference weight zero, and no held-fold gold injection.

| Run | Accuracy | Macro-F1 | Delta vs anchor |
|---|---:|---:|---:|
| Fixed-epoch anchor | 0.6571 | 0.6411 | -- |
| Compute-matched direct control | 0.6352 | 0.6246 | -0.0165 |
| Standard auxiliary | 0.6296 | 0.6190 | -0.0221 |
| Soft projection 0.25 | 0.6317 | 0.6213 | -0.0198 |
| Soft projection 0.50 | 0.6317 | 0.6215 | -0.0195 |
| Severity-adaptive, tau 0.10 | 0.6193 | 0.6085 | -0.0326 |

The selected development candidate was soft projection `0.50`. It improved
over standard auxiliary by `+0.002563` Macro-F1, but lost `-0.003051` to the
compute-matched direct control and `-0.019540` to the frozen fold anchor.
Against the anchor, the bootstrap interval was
`[-0.034777, -0.004056]`, probability of a positive delta was `0.006`, and
exact McNemar `p=0.000938`. Against the direct control, the bootstrap interval
was `[-0.016643, +0.010699]` and probability of a positive delta was `0.3388`.

The result separates two effects. Continuing the anchor for three matched
epochs already costs `-0.016489` Macro-F1, while standard auxiliary training
adds another `-0.005615` relative to that control. Partial projection recovers
only `+0.002563` of the auxiliary loss and cannot repair the dominant
continued-training degradation. Further projection-strength tuning is not
justified; the next branch must protect the frozen anchor itself and address
generalization/domain shift rather than gradient geometry alone.
