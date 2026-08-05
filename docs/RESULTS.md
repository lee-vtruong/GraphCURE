# GraphCURE experiment ledger

This file is the curated decision log for paper development. All values below
are preliminary single-seed (`seed=42`) NewsCLIPpings Merged/Balanced results
unless stated otherwise. They must not be presented as final results before the
registered five-seed protocol is complete.

## Dataset and evaluation

- Official split sizes: train 71,072; validation 7,024; test 7,264.
- Test labels are balanced: 3,632 pristine and 3,632 falsified.
- Primary selection metric: validation Macro-F1.
- Reported test metrics: accuracy and Macro-F1 at the best validation checkpoint.
- Cached CLIP dimensions: text 512, image 512.
- Multi-view dimensions: SBERT 768, FaceNet 512, Places365 2,048.
- Multi-view availability on train: SBERT 71,072; FaceNet 50,144; Places 70,808.
- Constraint targets are weak labels derived from generation provenance, not
  human-verified ground truth. Temporal supervision is unavailable.

## R0: observational CLIP baseline

Config: `configs/newsclippings_embeddings.yaml`. Ten epochs, seed 42.

| Architecture | Accuracy | Macro-F1 |
|---|---:|---:|
| GraphCURE typed graph | 0.6312 | 0.6304 |

Confusion matrix: `[[2462, 1170], [1509, 2123]]`. Falsified recall (0.5845)
was materially below pristine recall (0.6779). This run established that the
pipeline was valid but did not test the proposed auxiliary objectives because
constraint, conflict, and counterfactual losses were disabled.

## R1: matched CLIP architecture screen

Config: `configs/newsclippings_embeddings.yaml`. Seed 42.

| Architecture | Accuracy | Macro-F1 |
|---|---:|---:|
| Linear | 0.5354 | 0.5353 |
| MLP | 0.5329 | 0.5324 |
| Independent constraint nodes | **0.6368** | **0.6366** |
| Fully connected graph | 0.6348 | 0.6339 |
| Typed graph | 0.6312 | 0.6304 |

Decision: do not run five seeds. The graph did not beat independent nodes.

## R2: weak constraint supervision screen

Config: `configs/newsclippings_constraints.yaml`. Seed 42.

| Architecture | Accuracy | Macro-F1 |
|---|---:|---:|
| Independent constraint nodes | **0.6312** | **0.6308** |
| Fully connected graph | 0.6224 | 0.6217 |
| Typed graph | 0.6293 | 0.6293 |

Decision: provenance-derived auxiliary labels alone did not improve verdict
prediction. Do not enable conflict loss or run five seeds for this design.

## R3: constraint-specific multi-view screen

Config: `configs/newsclippings_multiview.yaml`. Seed 42. Early stopping patience
5. Semantic uses CLIP; entity uses SBERT/FaceNet; contextual uses CLIP/Places.

| Architecture | Best val epoch | Best val Macro-F1 | Test accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|
| Multi-view independent | 8 | 0.6433 | **0.6529** | **0.6529** |
| Multi-view fully connected | 8 | **0.6457** | 0.6415 | 0.6411 |
| Multi-view typed graph | 9 | 0.6435 | 0.6448 | 0.6443 |

Test confusion matrices:

- Multi-view independent: `[[2402, 1230], [1291, 2341]]`.
- Multi-view fully connected: `[[2459, 1173], [1431, 2201]]`.
- Multi-view typed graph: `[[2480, 1152], [1428, 2204]]`.

Interpretation: specialized views improved the best test Macro-F1 by about 1.63
points over the R1 independent model. Fixed message passing still caused
negative transfer: typed graph was 0.86 points below multi-view independent.
Typed structure was 0.32 points above a fully connected graph, but neither
result supports a graph novelty claim.

Decision: screen an adaptive typed graph with a pre-graph skip, node-wise
mixing gates, and edge dropout. Do not run five seeds until it exceeds the R3
independent reference (0.6529 Macro-F1).

## R4: adaptive typed graph

Config: `configs/newsclippings_adaptive.yaml`. Seed 42.

| Architecture | Test accuracy | Test Macro-F1 |
|---|---:|---:|
| Multi-view adaptive typed graph | 0.6373 | 0.6371 |

Confusion matrix: `[[2251, 1381], [1254, 2378]]`.

Node mixing gates in `[semantic, entity, temporal, contextual]` order:

- Mean: `[0.8219, 0.2243, 0.5544, 0.7950]`.
- Standard deviation: `[0.2825, 0.1031, 0.00004, 0.1943]`.

Interpretation: adaptive routing underperformed the R3 independent reference by
1.58 Macro-F1 points. The temporal gate was almost constant despite temporal
evidence being unavailable, indicating a learned bias rather than
evidence-conditioned routing. High semantic/contextual mixing also failed to
prevent negative transfer.

Decision: reject this design and do not run five seeds. Stop iterating on
observational graph routing. Use the R3 multi-view independent model as the
closed-book reference and move to paired counterfactual intervention training,
where graph/constraint behavior has direct supervision.

## Reporting rules

- One-seed screens guide engineering only; they are not paper claims.
- Keep all failed and negative results in this ledger.
- Never select a checkpoint using the test set.
- For retained claims, run seeds 13, 21, 42, 87, and 100 and report mean ± SD.
- Record exact git commit, config, dataset split, GPU, package versions, and raw
  per-seed metrics with `python -m scripts.export_results`.

## R5: paired counterfactual intervention training

Status: pending. Official pairing was verified exactly for all splits: 35,536
train, 3,512 validation, and 3,632 test pairs had pristine-then-falsified
ordering, identical caption IDs, different image IDs, and identical generation
sources. The first screen compares the R3 multi-view independent reference
against the same model with counterfactual sensitivity/invariance loss. Report
DIA, NDI, CVC, accuracy, and Macro-F1; retain the method only if intervention
metrics improve without a material verdict-quality collapse.
