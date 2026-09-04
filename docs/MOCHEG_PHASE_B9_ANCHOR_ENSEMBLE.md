# MOCHEG Phase B9: fixed-seed anchor stabilization

## Purpose

The final GraphCURE-Qwen3 baseline is an ensemble, but the recent train-only
screens used a single seed-42 anchor. Before developing another GraphCURE
expert, B9 tests whether an identical-model, fixed three-seed ensemble is a
materially stronger and more stable train-only anchor. This is a baseline
strengthening step, not a novelty claim.

The seeds are preregistered as `13, 42, 87`. All use duplicate-safe train fold
0, checkpoint epoch 3, identical data and hyperparameters. Their probabilities
are combined by an unweighted arithmetic mean; no interpolation weight or
class bias is selected.

Promotion requires at least `+0.005` Macro-F1 over seed 42, bootstrap positive
probability at least `0.95`, accuracy and each source within `-0.002` of seed
42, more helpful than harmful changes, and no more than `0.002` Macro-F1 loss
against the strongest constituent. Official validation and test remain locked.

## Run the two missing constituents

Seed 42 is reused from `outputs/mocheg_b6c_oof/fold_0/anchor`. Run seeds 13
and 87 in separate tmux sessions, preferably on different GPUs:

```bash
cd ~/whale/GraphCURE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv

for seed in 13 87; do
  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_lora_verifier \
    --manifest-root data/processed/mocheg_manifest_strict \
    --retrieval-root outputs/retrieval_mocheg_qwen3_reranked \
    --raw-root data/raw/mocheg_dataset/extracted/mocheg \
    --fold-spec data/processed/mocheg_sv_folds.json --fold-index 0 \
    --fixed-checkpoint-epoch 3 --epochs 3 \
    --output "outputs/mocheg_b9_oof/fold_0/seed_${seed}" \
    --seed "$seed" --batch-size 1 --gradient-accumulation 16 \
    --learning-rate 1e-4 --max-length 3072 \
    2>&1 | tee "outputs/mocheg-b9-fold0-seed${seed}.log"
done
```

Do not run that loop twice concurrently on the same GPU. If both GPUs are
healthy, run seed 13 with `CUDA_VISIBLE_DEVICES=0` and seed 87 in another tmux
session with `CUDA_VISIBLE_DEVICES=1`.

## Analyze without GPU

```bash
python -m scripts.analyze_mocheg_b9_anchor_ensemble \
  --output outputs/mocheg_b9_fold0_anchor_ensemble.json \
  2>&1 | tee outputs/mocheg-b9-fold0-anchor-ensemble.log
```

Inspect:

```bash
python - <<'PY'
import json
s = json.load(open("outputs/mocheg_b9_fold0_anchor_ensemble.json"))
print("per seed:", s["per_seed"])
print("ensemble:", s["ensemble"])
print("strongest:", s["strongest_constituent"])
print("vs seed42:", s["comparison_vs_seed42"])
print("sources:", s["source_diagnostics_vs_seed42"])
print("gate:", s["promotion_gate"])
print("official validation used:", s["official_validation_used"])
print("test used:", s["test_split_used"])
PY
```

Do not run folds 1--4 or official validation/test unless the complete gate
passes.
