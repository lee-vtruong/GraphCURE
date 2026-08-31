# MOCHEG Phase B6-B: conflict-aware auxiliary training

## Scientific status

B6-A failed its frozen five-seed confirmation because auxiliary-minus-control
bootstrap probability was `0.9354`, below the preregistered `0.95`. That result
is immutable. B6-B is a new, post-hoc robustness branch designed specifically
to reduce seed-dependent auxiliary harm. Official test remains locked.

Inference is unchanged direct verdict prediction with hierarchical weight
zero. During each gradient-accumulation window, verdict and auxiliary LoRA
gradients are accumulated separately. If their dot product is negative, the
conflicting component of the auxiliary gradient is projected away before the
optimizer step. The verdict gradient is never projected.

## Frozen screen

Only seeds `42` and `87` are used initially. Seed 42 had the strongest causal
screen evidence; seed 87 was the B6-A failure case versus compute control.
This targeted choice is reported as post-hoc and cannot replace B6-A.

Promotion to five-seed validation confirmation requires:

- PCGrad conflict rate greater than zero in both seeds;
- PCGrad beats compute control in both seeds;
- mean PCGrad-minus-control Macro-F1 at least `+0.005`;
- seed-87 PCGrad-minus-control Macro-F1 at least `+0.003`;
- mean PCGrad-minus-standard-auxiliary Macro-F1 non-negative;
- two-seed raw-ensemble bootstrap probability of positive delta versus control
  at least `0.95`.

No threshold may be changed after the screen.

## Server runbook

```bash
cd ~/whale/GraphCURE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv
```

Run the frozen seed-42/87 screen in tmux:

```bash
tmux new -s mocheg-b6b
```

```bash
for spec in \
  "42 0.670470583003719" \
  "87 0.6875591947992383"
do
  set -- $spec
  seed="$1"
  anchor="$2"

  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_hierarchical_lora \
    --target-root data/processed/mocheg_b6_targets \
    --raw-root data/raw/mocheg_dataset/extracted/mocheg \
    --initial-adapter "outputs/mocheg_qwen3_lora_seed${seed}_v16/best_adapter" \
    --output "outputs/mocheg_b6b_pcgrad_seed${seed}" \
    --seed "$seed" \
    --anchor-macro-f1 "$anchor" \
    --epochs 3 --patience 2 --batch-size 1 --gradient-accumulation 16 \
    --learning-rate 3e-5 --max-length 3072 \
    --hierarchical-weights 0 \
    --gradient-mode pcgrad \
    --auxiliary-gradient-scale 1 \
    2>&1 | tee "outputs/mocheg-b6b-pcgrad-seed${seed}.log"
done
```

Inspect gradient diagnostics and run the frozen screen:

```bash
grep -E 'gradient_cosine|gradient_conflict_rate' \
  outputs/mocheg-b6b-pcgrad-seed*.log

python -m scripts.analyze_mocheg_b6b_pcgrad_screen \
  --output outputs/mocheg_b6b_pcgrad_screen.json \
  2>&1 | tee outputs/mocheg-b6b-pcgrad-screen.log
```

Do not run seeds `13/21/100` and do not access official test unless every
screen gate is true.
