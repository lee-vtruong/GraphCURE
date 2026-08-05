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

### NewsCLIPpings precomputed-embedding baseline

The official release includes CLIP image/text embedding dictionaries, so the
closed-book baseline does not require VisualNews images. Pack the dictionaries
into memory-mapped arrays:

```bash
python -m scripts.prepare_newsclippings \
  --raw-root data/raw/news_clippings/news_clippings \
  --output data/processed/newsclippings_clip

CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --config configs/newsclippings_embeddings.yaml \
  --device cuda

CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate \
  --config configs/newsclippings_embeddings.yaml \
  --checkpoint outputs/newsclippings_embeddings/best.pt \
  --device cuda
```

This is an observational binary-label baseline. Constraint, conflict, and
counterfactual losses are disabled until their supervision is generated.

### Matched architecture ablations (required next step)

Run a quick one-seed screen first. It trains and evaluates a linear head, MLP,
independent constraint nodes, fully connected graph, and the typed prior graph:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_ablations \
  --config configs/newsclippings_embeddings.yaml \
  --device cuda \
  --seeds 42 \
  --output-root outputs/ablations/newsclippings_clip_screen
```

After the screen succeeds, run the five-seed experiment (preferably in tmux):

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_ablations \
  --config configs/newsclippings_embeddings.yaml \
  --device cuda \
  --seeds 13 21 42 87 100 \
  --skip-existing \
  2>&1 | tee outputs/newsclippings-ablation.log
```

The aggregate table is written to
`outputs/ablations/newsclippings_clip/summary.json`; individual predictions and
metrics remain under each architecture/seed directory. `--skip-existing`
resumes completed runs safely. Do not interpret the graph as useful unless the
typed graph consistently beats both the MLP and fully connected graph.

### Weak constraint supervision

NewsCLIPpings records which generator produced each mismatch. Convert that
provenance into explicitly documented weak targets (semantic, entity, or
contextual/scene) without repacking the embedding arrays:

```bash
python -m scripts.annotate_newsclippings_constraints

CUDA_VISIBLE_DEVICES=0 python -m scripts.run_ablations \
  --config configs/newsclippings_constraints.yaml \
  --device cuda \
  --architectures independent fully_connected typed_graph \
  --seeds 42 \
  --output-root outputs/ablations/newsclippings_constraints_screen
```

These are pseudo-labels derived from dataset construction, not human-verified
constraint annotations. Temporal targets remain unknown because the official
benchmark has no dedicated temporal manipulation subset.

### Constraint-specific multi-view model

Add the official SBERT, FaceNet, and Places365 views to the already packed
samples. Face embeddings are mean pooled and every optional view carries an
availability mask:

```bash
python -m scripts.prepare_newsclippings_multiview

CUDA_VISIBLE_DEVICES=0 python -m scripts.run_ablations \
  --config configs/newsclippings_multiview.yaml \
  --device cuda \
  --architectures multi_independent multi_fully_connected multi_typed_graph \
  --seeds 42 \
  --output-root outputs/ablations/newsclippings_multiview_screen
```

The three variants use identical node inputs and heads. They differ only in
message passing. Multi-view graph layers use conservative learnable residual
gates initialized near 0.12 so a node can reject initially harmful messages.

If fixed graph propagation underperforms the independent model, screen the
adaptive typed graph. It preserves pre-graph nodes through a learnable skip,
uses per-sample/node mixing gates, and applies edge dropout during training:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_ablations \
  --config configs/newsclippings_adaptive.yaml \
  --device cuda \
  --architectures multi_independent multi_adaptive_graph \
  --seeds 42 \
  --output-root outputs/ablations/newsclippings_adaptive_screen
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
