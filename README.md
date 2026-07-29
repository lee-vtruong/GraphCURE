# GraphCURE

Research code for intervention-faithful multimodal fact verification and
cost-aware sequential evidence acquisition.

The implementation deliberately separates frozen/cached backbone embeddings
from the GraphCURE reasoning layer. This makes architectural ablations fair,
reduces server training cost, and lets the same manifests work with CLIP,
SigLIP, DINOv2, or MLLM embeddings.

## Current status

- Typed four-node constraint graph: semantic, entity, temporal, contextual.
- Relation-specific compatibility tensors for typed conflict.
- Constraint-state and evidential uncertainty heads.
- Counterfactual invariance/sensitivity objective.
- One-step expected-value-of-information action selection.
- Reproducible JSONL manifest interface, training loop, smoke data, and tests.

The current code is a research baseline, not yet an A* result. The A* claim
must be earned by the experiment protocol in `docs/EXPERIMENTS.md`.

## 1. Server setup

Recommended: Ubuntu, Python 3.11, CUDA 12.x.

```bash
git clone <YOUR_REPOSITORY_URL> GraphCURE
cd GraphCURE

conda create -n graphcure python=3.11 -y
conda activate graphcure

# Choose the PyTorch command matching the server CUDA version:
# https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[train,retrieval,dev]"
pytest -q
```

Do not blindly reuse the `cu124` line if the server driver requires another
wheel. Verify with `nvidia-smi` and the PyTorch installation selector.

## 2. Smoke test before downloading datasets

```bash
python -m scripts.make_synthetic
python -m scripts.train --config configs/synthetic.yaml --device cuda
```

Expected artifacts:

```text
outputs/synthetic/best.pt
outputs/synthetic/metrics.json
```

## 3. Download official dataset repositories

```bash
bash scripts/download_data.sh data/raw
```

This clones only official code/metadata:

- NewsCLIPpings: <https://github.com/g-luo/news_clippings>
- MOCHEG: <https://github.com/VT-NLP/Mocheg>

NewsCLIPpings depends on VisualNews images and MOCHEG distributes data through
the link in its README. Their licenses/terms prevent this repository from
silently redistributing the assets. Follow each official README, then place:

```text
data/raw/
  news_clippings/
  visualnews/
  mocheg/
```

## 4. Unified manifest

Training consumes one JSON object per line:

```json
{
  "id": "sample-id",
  "label": 2,
  "text_embedding": "embeddings/text/sample-id.pt",
  "image_embedding": "embeddings/image/sample-id.pt",
  "metadata": [0.0, 0.0],
  "constraint_labels": [0, 0, 1, 1],
  "conflict_labels": [1, 0, 1, 0, 0],
  "counterfactual": {
    "text_embedding": "embeddings/text/sample-id-cf.pt",
    "changed_mask": [false, false, true, true]
  }
}
```

State mapping is `0=SATISFIED`, `1=VIOLATED`, `2=UNKNOWN`; use `-100` when a
constraint has no supervision. Final label mappings must be recorded per
dataset and normalized by the adapter.

## 5. Train

Edit or copy `configs/base.yaml`, then:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --config configs/base.yaml \
  --device cuda
```

For multi-GPU, first keep the single-GPU run as a reproducibility reference,
then launch the same module through Accelerate or `torchrun` after adding the
distributed wrapper.

## 6. Research order

Run experiments in this order:

1. Frozen CLIP/SigLIP embeddings + linear verdict classifier.
2. Independent constraint heads.
3. Fully connected constraint attention.
4. Prior graph, learned graph, prior+learned graph.
5. Typed conflict versus JS disagreement.
6. Observational graph versus counterfactual training.
7. Always-closed, always-open, entropy router, entropy-per-cost, EVI.
8. Cross-dataset and event/source-disjoint evaluation.

Do not tune only on the new counterfactual benchmark. Model selection must use
the validation split of the target benchmark, and the manipulation benchmark
should remain a held-out stress test.

## Reproducibility

- Keep raw data outside Git; manifests store stable IDs and relative paths.
- Commit every config used for a reported table.
- Save seed, package versions, GPU name, git commit, and dataset checksums.
- Report mean and standard deviation over at least five seeds for main claims.
- Report Macro-F1, hard-negative F1, ECE, AURC, accuracy-cost Pareto area,
  acquisition regret, evidence recall, token cost, and latency.

