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

Official pairing was verified exactly for all splits: 35,536
train, 3,512 validation, and 3,632 test pairs had pristine-then-falsified
ordering, identical caption IDs, different image IDs, and identical generation
sources.

| Model | Accuracy | Macro-F1 | DIA | NDI | CVC |
|---|---:|---:|---:|---:|---:|
| R3 independent reference | **0.6529** | **0.6529** | 0.4601 | 0.7406 | **0.5063** |
| R5 + symmetric JS intervention | 0.6461 | 0.6458 | **0.4821** | **0.7676** | 0.5008 |

R5 confusion matrix: `[[2448, 1184], [1387, 2245]]`.

Interpretation: DIA improved by 2.20 points and NDI by 2.70 points, but test
Macro-F1 fell by 0.71 points and CVC fell by 0.55 points. Symmetric JS enforces
that a changed node differs, but not the known direction from SATISFIED/pristine
to VIOLATED/falsified.

Decision: do not run five seeds. R6 adds directional pair ranking for both the
changed constraint's VIOLATED probability and the falsified verdict score,
while reducing the symmetric JS weight.

## R6: directional intervention objective

Config: `configs/newsclippings_directional.yaml`. Seed 42.

| Accuracy | Macro-F1 | DIA | NDI | CVC |
|---:|---:|---:|---:|---:|
| 0.6536 | 0.6534 | 0.4642 | 0.7510 | 0.4939 |

Confusion matrix: `[[2469, 1163], [1353, 2279]]`.

Interpretation: R6 recovered verdict quality and slightly exceeded the R3
single-seed Macro-F1 by 0.05 points. NDI improved 1.04 points and DIA improved
0.41 points over R3, but thresholded CVC fell 1.24 points. Because the
directional objective optimizes pairwise score order rather than thresholded
label flips, final acceptance is pending a checkpoint-only re-evaluation with
verdict and changed-constraint pair-order accuracy.

Checkpoint-only pair-order re-evaluation:

| Model | Macro-F1 | DIA | NDI | CVC | Verdict order | Constraint order |
|---|---:|---:|---:|---:|---:|---:|
| R3 reference | 0.6529 | 0.4601 | 0.7406 | **0.5063** | 0.7131 | 0.6187 |
| R5 symmetric JS | 0.6458 | **0.4821** | **0.7676** | 0.5008 | 0.7018 | 0.6110 |
| R6 directional | **0.6534** | 0.4642 | 0.7510 | 0.4939 | **0.7269** | **0.6228** |

Relative to R3, R6 improved Macro-F1 by 0.05 points, DIA by 0.41,
NDI by 1.04, verdict pair order by 1.38, and constraint pair order by 0.41;
thresholded CVC decreased by 1.24 points. Pair-order metrics directly match the
training objective, whereas CVC additionally depends on the classification
threshold.

Decision: advance only R3 and R6 to the registered five-seed screen. Treat the
result as viable only if pair-order gains are stable and verdict quality is
non-inferior across seeds; continue to report the negative CVC result.

### R3 versus R6 five-seed confirmation

Seeds: 13, 21, 42, 87, 100.

| Metric | R3 mean ± SD | R6 mean ± SD | Paired delta mean ± SD |
|---|---:|---:|---:|
| Macro-F1 | 0.6494 ± 0.0040 | 0.6495 ± 0.0073 | +0.0001 ± 0.0040 |
| Accuracy | 0.6495 ± 0.0041 | 0.6497 ± 0.0073 | +0.0002 ± 0.0040 |
| DIA | 0.4678 ± 0.0090 | 0.4700 ± 0.0099 | +0.0021 ± 0.0171 |
| NDI | 0.7448 ± 0.0089 | 0.7430 ± 0.0100 | -0.0018 ± 0.0089 |
| CVC | 0.5010 ± 0.0039 | 0.4999 ± 0.0082 | -0.0012 ± 0.0100 |
| Verdict pair order | 0.7125 ± 0.0048 | 0.7163 ± 0.0111 | +0.0038 ± 0.0128 |
| Constraint pair order | 0.6210 ± 0.0043 | 0.6236 ± 0.0071 | +0.0026 ± 0.0079 |

Decision: reject R6 as a primary contribution. Effects are small relative to
seed variation and paired deltas change sign. The underlying intervention is
not minimal: replacing an image changes CLIP-image, FaceNet, and Places inputs
simultaneously, while the generation-source mask marks only one constraint.
Thus the invariance objective can penalize nodes whose actual evidence changed.
R7 must construct feature-level minimal interventions that replace only the
target node's views and preserve all non-target inputs exactly.

## R7: feature-level minimal interventions

Config: `configs/newsclippings_minimal_intervention.yaml`. Seed 42.
Semantic interventions replace only the semantic node's CLIP image; entity
interventions replace only FaceNet; contextual interventions replace only the
contextual CLIP image and Places view. Every other node input is copied bitwise
from the factual member. This is a constructed predictive intervention and must
not be described as causal discovery.

| Accuracy | Macro-F1 | DIA | NDI | CVC | Verdict order | Constraint order |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6499 | 0.6475 | 0.4744 | 1.0000 | 0.3121 | 0.6399 | 0.6349 |

Confusion matrix: `[[2664, 968], [1575, 2057]]`.

Interpretation: NDI reaches 1.0 by construction for an independent-node model:
non-target inputs are identical and there is no message passing. It is therefore
a sanity check, not evidence of superiority. Thresholded CVC drops sharply and
Macro-F1 is 0.55 points below the R3 seed-42 reference. Constraint pair order is
higher than R6 seed 42, but acceptance is pending re-evaluation of the frozen R3
checkpoint under the identical minimal-intervention protocol.
