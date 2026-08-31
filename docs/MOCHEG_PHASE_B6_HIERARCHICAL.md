# MOCHEG Phase B6: sufficiency-polarity Qwen3 verifier

## Status and scientific boundary

The frozen article ensemble remains the Phase-B anchor. Its official MOCHEG
result is `0.567977` Accuracy and `0.545309` Macro-F1. This exceeds the
strongest comparison Accuracy but remains `0.010691` below the strongest
reported Macro-F1. B6 is therefore a validation-only attempt to improve the
minority/NEI decision without touching official test.

The official test split must not be loaded by any B6 development script. The
trainer accepts only prepared `train` and `val` targets and always records
`test_split_used: false`.

The scientific anchor is loaded from the frozen `val_predictions.jsonl` next
to each article adapter. A fresh re-inference is retained only as a library
drift diagnostic; it is not allowed to lower the promotion threshold.

## Method

One continued Qwen3-4B LoRA answers three task prompts:

1. direct verdict: Supported / Refuted / NEI;
2. evidence sufficiency: sufficient or insufficient for a binary verdict;
3. polarity: Supported or Refuted, conditional on sufficient evidence.

The hierarchical probability is

```text
P(NEI)       = P(insufficient)
P(Supported) = P(sufficient) * P(support)
P(Refuted)   = P(sufficient) * P(refute)
```

It is blended with the direct verdict distribution. The blend weight may be
screened once on seed 42 and must then be frozen for all confirmation seeds.

Sufficiency targets are evidence-conditioned:

- every NEI claim is insufficient for a binary polarity decision;
- a Supported/Refuted claim is sufficient only when a labelled relevant
  article occurs in the supplied candidate set;
- a Supported/Refuted claim without qrels has no sufficiency target;
- removing labelled relevant evidence creates an auxiliary insufficient
  example, not a fabricated NEI verdict example;
- gold insertion is allowed for train only and is audited explicitly.

## Promotion gates

Seed-42 screen:

- Macro-F1 delta at least `+0.005`;
- NEI F1 delta at least `+0.020`;
- Supported F1 drop no worse than `-0.005`;
- Accuracy drop no worse than `-0.003`;
- selected hierarchical weight greater than zero.

Frozen five-seed confirmation:

- mean paired Macro-F1 delta at least `+0.003`;
- raw ensemble Macro-F1 delta at least `+0.003`;
- raw B6 ensemble Macro-F1 at least `0.695` on validation;
- raw ensemble NEI F1 delta at least `+0.020`;
- raw ensemble Supported F1 drop no worse than `-0.005`;
- at least four of five seeds improve.

Failure ends B6 without an official-test run.

## Seed-42 outcome and matched control

The first seed-42 screen selected hierarchical inference weight `0`, so the
registered B6 decomposition gate failed. Nevertheless, the direct verdict
output after auxiliary training reached approximately `0.686273` validation
Macro-F1, a `+0.015802` gain over the frozen seed-42 article anchor. This is a
new hypothesis, not a B6 success: decomposition may be useful as training-only
supervision even when its probability factorization is unsuitable at inference.

Before any more auxiliary seeds, run a matched direct-only continuation with
the same adapter, optimizer-update count, optimizer, schedule, and verdict
loss. Auxiliary task weights are zero, zero-weight rows are omitted, and the
verdict rows are repeated to the exact training-example count recorded by the
auxiliary summary. B6-A advances only if the auxiliary run beats both the
frozen anchor and this compute-matched control.

## Server runbook

Activate the environment:

```bash
cd ~/whale/GraphCURE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv
```

Prepare and audit train/validation targets:

```bash
python -m scripts.prepare_mocheg_sufficiency_targets \
  --manifest-root data/processed/mocheg_manifest_strict \
  --retrieval-root outputs/retrieval_mocheg_qwen3_reranked \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --output-root data/processed/mocheg_b6_targets \
  --top-k 5 \
  --inject-train-gold \
  2>&1 | tee outputs/mocheg-b6-targets.log

python - <<'PY'
import json
for split in ("train", "val"):
    path = f"data/processed/mocheg_b6_targets/{split}.summary.json"
    print(split.upper(), json.load(open(path)))
PY
```

Run a small smoke test first:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_hierarchical_lora \
  --target-root data/processed/mocheg_b6_targets \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --initial-adapter outputs/mocheg_qwen3_lora_seed42_v16/best_adapter \
  --output outputs/mocheg_b6_smoke \
  --seed 42 \
  --epochs 1 \
  --limit-train 128 \
  --limit-val 64 \
  --max-length 2048 \
  --gradient-accumulation 8 \
  --num-workers 0 \
  --skip-anchor-check \
  2>&1 | tee outputs/mocheg-b6-smoke.log
```

Run the seed-42 validation screen in tmux:

```bash
tmux new -s mocheg-b6

cd ~/whale/GraphCURE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_hierarchical_lora \
  --target-root data/processed/mocheg_b6_targets \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --initial-adapter outputs/mocheg_qwen3_lora_seed42_v16/best_adapter \
  --output outputs/mocheg_b6_hierarchical_seed42 \
  --seed 42 \
  --epochs 3 \
  --patience 2 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --learning-rate 3e-5 \
  --max-length 3072 \
  --hierarchical-weights 0,0.25,0.5,0.75,1 \
  2>&1 | tee outputs/mocheg-b6-seed42.log
```

Detach with `Ctrl-b`, then `d`. Monitor with:

```bash
tail -f outputs/mocheg-b6-seed42.log
```

If the process is interrupted after at least one improved epoch was saved,
repeat the same command with `--resume-from-best`. It reloads
`outputs/mocheg_b6_hierarchical_seed42/best_adapter`; optimizer state is reset,
so the resumed run must still be reported as a recovery run.

Inspect the frozen screen result:

```bash
python - <<'PY'
import json
s = json.load(open("outputs/mocheg_b6_hierarchical_seed42/summary.json"))
print("mode:", s["mode"])
print("best epoch:", s["best_epoch"])
print("weight:", s["selected_hierarchical_weight"])
print("anchor:", s["anchor"])
print("candidate:", s["final_candidate"])
print("deltas:", s["deltas"])
print("gate:", s["promotion_gate"])
print("test used:", s["test_split_used"])
PY
```

Do not run seeds `13/21/87/100` until seed 42 passes. If it passes, freeze the
reported hierarchical weight and use that one value for every confirmation
seed; do not repeat the weight grid per seed.

For the observed training-only auxiliary signal, run the seed-42 direct-only
control instead of the original B6 confirmation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_hierarchical_lora \
  --target-root data/processed/mocheg_b6_targets \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --initial-adapter outputs/mocheg_qwen3_lora_seed42_v16/best_adapter \
  --output outputs/mocheg_b6_direct_control_seed42 \
  --seed 42 \
  --epochs 3 \
  --patience 2 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --learning-rate 3e-5 \
  --max-length 3072 \
  --ablation-ratio 0 \
  --sufficiency-loss-weight 0 \
  --polarity-loss-weight 0 \
  --ablation-loss-weight 0 \
  --hierarchical-weights 0 \
  --match-training-examples-from outputs/mocheg_b6_hierarchical_seed42/summary.json \
  2>&1 | tee outputs/mocheg-b6-direct-control-seed42.log
```

Then run the preregistered, validation-only causal diagnostic:

```bash
python -m scripts.analyze_mocheg_b6_auxiliary_control \
  --anchor outputs/mocheg_qwen3_lora_seed42_v16/val_predictions.jsonl \
  --direct-control outputs/mocheg_b6_direct_control_seed42/val_predictions.jsonl \
  --auxiliary outputs/mocheg_b6_hierarchical_seed42/val_predictions.jsonl \
  --output outputs/mocheg_b6_auxiliary_control.json \
  2>&1 | tee outputs/mocheg-b6-auxiliary-control.log
```

The seed-42 causal screen passed: auxiliary training improved Macro-F1 by
`+0.015802` versus the frozen anchor and `+0.018452` versus the compute-matched
direct control. The latter bootstrap interval was `[+0.000610, +0.036260]`
with probability of positive delta `0.9788`. Exact McNemar `p=0.05045` is
borderline and is not treated as confirmation; five frozen seeds are required.

Run auxiliary and compute-matched direct-control arms for the four remaining
seeds. Hierarchical inference weight stays frozen at zero:

```bash
for spec in \
  "13 0.6742734332823367" \
  "21 0.6621083137026326" \
  "87 0.6875591947992383" \
  "100 0.6793433665313913"
do
  set -- $spec
  seed="$1"
  anchor="$2"

  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_hierarchical_lora \
    --target-root data/processed/mocheg_b6_targets \
    --raw-root data/raw/mocheg_dataset/extracted/mocheg \
    --initial-adapter "outputs/mocheg_qwen3_lora_seed${seed}_v16/best_adapter" \
    --output "outputs/mocheg_b6_auxiliary_seed${seed}" \
    --seed "$seed" \
    --anchor-macro-f1 "$anchor" \
    --epochs 3 --patience 2 --batch-size 1 --gradient-accumulation 16 \
    --learning-rate 3e-5 --max-length 3072 \
    --hierarchical-weights 0 \
    2>&1 | tee "outputs/mocheg-b6-auxiliary-seed${seed}.log"

  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_qwen3_hierarchical_lora \
    --target-root data/processed/mocheg_b6_targets \
    --raw-root data/raw/mocheg_dataset/extracted/mocheg \
    --initial-adapter "outputs/mocheg_qwen3_lora_seed${seed}_v16/best_adapter" \
    --output "outputs/mocheg_b6_direct_control_seed${seed}" \
    --seed "$seed" \
    --anchor-macro-f1 "$anchor" \
    --epochs 3 --patience 2 --batch-size 1 --gradient-accumulation 16 \
    --learning-rate 3e-5 --max-length 3072 \
    --ablation-ratio 0 --sufficiency-loss-weight 0 \
    --polarity-loss-weight 0 --ablation-loss-weight 0 \
    --hierarchical-weights 0 \
    --match-training-examples-from \
      "outputs/mocheg_b6_auxiliary_seed${seed}/summary.json" \
    2>&1 | tee "outputs/mocheg-b6-direct-control-seed${seed}.log"
done
```

Summarize all five frozen seeds:

```bash
python -m scripts.summarize_mocheg_b6a_confirmation \
  --output outputs/mocheg_b6a_validation_summary.json \
  --markdown outputs/mocheg_b6a_validation_summary.md
```

The confirmation gate requires mean auxiliary gains over both anchor and
compute-matched control, gains in both raw ensembles, at least four positive
seed-level control deltas, auxiliary ensemble Macro-F1 at least `0.695`, and
ensemble bootstrap probability of a positive control delta at least `0.95`.

## Frozen five-seed outcome

B6-A failed the preregistered confirmation gate and must not access official
test. The result is nevertheless a strong development signal:

- all five auxiliary runs beat their corresponding frozen article anchors;
- four of five beat their compute-matched direct controls;
- mean auxiliary-minus-anchor Macro-F1 was `+0.012826 +/- 0.007729`;
- mean auxiliary-minus-control Macro-F1 was `+0.008863 +/- 0.010806`;
- the raw auxiliary ensemble reached `0.708005` Macro-F1 and `0.714973`
  Accuracy, versus control ensemble `0.698093` and `0.705357`;
- ensemble auxiliary-minus-control was `+0.009912` Macro-F1, but its bootstrap
  interval crossed zero (`[-0.002870, +0.022677]`) and probability of positive
  delta was `0.9354`, below the frozen `0.95` gate;
- exact McNemar `p=0.16058` also does not establish a reliable causal gain.

The gate must not be weakened after observing this outcome. A future B6-B
branch, if pursued, is a new post-hoc robustness experiment and must preserve
this failed B6-A result. Its target is the seed-dependent auxiliary harm seen
at seed `87`, not a re-analysis of the current predictions.
