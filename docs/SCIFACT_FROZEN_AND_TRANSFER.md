# SciFact Zero-Shot Frozen AND Transfer

## Scope

This external evaluation transfers the already frozen MOCHEG policy unchanged:

- direct expert: five legacy direct Qwen3-4B LoRA members;
- rationale-trained expert: frozen members 42, 87, and 100;
- evidence count: $K=5$;
- AND threshold: $\tau=.49$;
- decision rule: replace a direct prediction only when the rationale-trained
  ensemble predicts NEI with probability at least $\tau$.

No SciFact label, claim, evidence annotation, seed selection, or threshold is
used to choose a model or policy.  SciFact development labels are used only in
the final metric calculation.  Retrieval is deterministic corpus-wide TF--IDF
over the official SciFact corpus; cited documents and gold rationale sentences
are not read by the retriever or model.

SciFact's public labelled split is development; its test labels are unavailable.
Accordingly, this is an **external zero-shot development-split audit**, not an
official SciFact-test result and not a SciFact leaderboard/SOTA comparison.

## Download and run

```bash
cd ~/whale/GraphCURE
git pull --ff-only
source .venv/bin/activate

tmux new -s scifact-cure-and
bash scripts/run_scifact_frozen_and_transfer.sh \
  2>&1 | tee outputs/scifact_cure_and_zero_shot/runner.log
```

The script downloads the official AllenAI SciFact release from
`https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz` only
when `data/external/scifact_raw/data/` is absent.  It then prepares the
retrieval artifact, runs eight frozen adapters, and evaluates AND once.

## Monitoring

```bash
cd ~/whale/GraphCURE

test -s outputs/scifact_cure_and_zero_shot/protocol/protocol.json \
  && echo "PREPARATION DONE" || echo "PREPARATION PENDING"
test -s outputs/scifact_cure_and_zero_shot/direct_inference/inference_scifact_direct.json \
  && echo "DIRECT DONE" || echo "DIRECT RUNNING/PENDING"
test -s outputs/scifact_cure_and_zero_shot/rationale_inference/inference_scifact_rationale.json \
  && echo "RATIONALE DONE" || echo "RATIONALE RUNNING/PENDING"
test -s outputs/scifact_cure_and_zero_shot/result/report.md \
  && echo "ALL DONE" || echo "PENDING"
```

## Expected cost and output

No training, teacher generation, or new neural retriever is run.  The TF--IDF
preparation is CPU-only over 5,183 abstracts.  The GPU workload is eight
adapter inference passes over the labelled SciFact development split; budget
roughly 10--25 minutes on an RTX 5090, depending on abstract lengths and
cache state.

The final frozen audit is:

```text
outputs/scifact_cure_and_zero_shot/result/report.md
outputs/scifact_cure_and_zero_shot/result/report.json
```

It records corpus/claim/retrieval and prediction hashes, class metrics,
route rate, paired bootstrap interval, and McNemar test.
