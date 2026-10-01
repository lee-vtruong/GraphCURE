# B22: Predeclared Eight-Member Direct-Only Size Control

## Purpose

This is the remaining ensemble-size control for the CURE paper.  It separates
an effect of adding three model members from an effect of combining differently
trained experts.

The experiment fixes three new direct-only seeds, `314`, `2718`, and `2026`,
before any test inference.  All eight direct members receive equal posterior
weight.  A seed is never removed based on validation or test performance.

The raw and strict reports make two paired comparisons on the same manifests:

1. **Direct-8 vs. legacy direct-5:** the benefit, if any, from three more
   direct-only members.
2. **Heterogeneous CURE-Ensemble (5 direct + 3 rationale-trained) vs.
   direct-8:** the residual benefit of heterogeneity at the same eight-member
   inference budget.

This is an ensemble control, not an AND-router experiment.  It does not tune
the AND threshold, select an ensemble, or modify the paper's locked routing
result.

## Frozen configuration

The new direct members intentionally reproduce the legacy direct-verdict
recipe:

| Setting | Value |
| --- | --- |
| Backbone | `Qwen/Qwen3-4B-Instruct-2507` |
| Objective | one-token A/B/C direct verdict |
| Evidence | top-5 retrieved passages, 2,200 chars/passage |
| Context | 3,072 tokens |
| LoRA | rank 16, alpha 32, dropout .05 |
| Optimizer | AdamW, lr `1e-4`, weight decay `.01`, cosine warmup `.05` |
| Training | 3 epochs, batch 1, accumulation 16, validation checkpoint selection |
| Train-only retrieval augmentation | identical legacy default (gold-candidate injection at train time only) |
| Added seeds | `314`, `2718`, `2026` |
| Test member selection/weighting | none |

The test protocol is P1 raw official ($n=2{,}442$) and strict duplicate-safe
P1 ($n=2{,}434$), with independent inference on their respective retrieval
manifests.

## Run

On the server:

```bash
cd ~/whale/GraphCURE
git pull --ff-only
source .venv/bin/activate

tmux new -s mocheg-b22-direct8
bash scripts/run_mocheg_b22_direct8_control.sh \
  2>&1 | tee outputs/mocheg_b22_direct8_control/runner.log
```

Detach with `Ctrl-b d`.  The script is resumable: completed seed summaries and
completed raw/strict inference reports are skipped.  It trains serially on the
GPU selected by `CUDA_VISIBLE_DEVICES` (default `0`).

## Monitoring

```bash
cd ~/whale/GraphCURE

for seed in 314 2718 2026; do
  path="outputs/mocheg_b22_direct8_control/direct_seed${seed}/summary.json"
  test -s "$path" && echo "TRAIN DONE: $seed" || echo "TRAIN RUNNING/PENDING: $seed"
done

test -s outputs/mocheg_b22_direct8_control/size_control.md \
  && echo "ALL DONE" || echo "TRAINING / INFERENCE / ANALYSIS PENDING"

tail -f outputs/mocheg_b22_direct8_control/direct_seed314/train.log
```

## Expected duration

The job consists of three full direct-verdict LoRA training runs plus six
single-adapter test inference passes (raw and strict for each added member).
On the RTX 5090, budget approximately **10--15 GPU-hours serially**.  The
precise duration depends on sequence truncation, data-loader throughput, and
Hugging Face cache state.  No new retrieval or teacher generation is required.

## Outputs

The final auditable artifact is:

```text
outputs/mocheg_b22_direct8_control/size_control.md
outputs/mocheg_b22_direct8_control/size_control.json
```

It stores the raw/strict manifest and retrieval hashes, all input prediction
hashes, paired bootstrap intervals, exact McNemar tests, and helpful/harmful
counts.  It must be inspected before adding any result to the manuscript.
