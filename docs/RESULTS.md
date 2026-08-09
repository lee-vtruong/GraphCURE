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

Minimal-protocol comparison using the same seed-42 test pairs:

| Model | Macro-F1 | DIA | NDI | CVC | Verdict order | Constraint order |
|---|---:|---:|---:|---:|---:|---:|
| Frozen R3 reference | **0.6529** | 0.4601 | 1.0000 | 0.2890 | 0.6188 | 0.6187 |
| R7 minimal trained | 0.6475 | **0.4744** | 1.0000 | **0.3121** | **0.6399** | **0.6349** |

R7 improves DIA by 1.43 points, CVC by 2.31, verdict pair order by
2.11, and constraint pair order by 1.62, at a 0.55-point Macro-F1 cost.
This is the first intervention variant to improve every targeted metric under a
matched protocol, but the verdict trade-off remains material.

Protocol decision: the official test set has now been inspected during several
engineering screens. Freeze it immediately. R8 loss balancing and all further
model selection must use validation metrics only; test evaluation is permitted
again only after the configuration and seeds are preregistered in this ledger.

Validation confirmation under the same minimal protocol:

| Model | Macro-F1 | DIA | NDI | CVC | Verdict order | Constraint order |
|---|---:|---:|---:|---:|---:|---:|
| Frozen R3 reference | 0.6433 | 0.4547 | 1.0000 | 0.2765 | 0.6189 | 0.6244 |
| R7 minimal trained | **0.6435** | **0.4636** | 1.0000 | **0.2984** | **0.6338** | **0.6264** |

Validation confirms the intervention signal without a verdict-quality loss:
DIA +0.89 points, CVC +2.19, verdict order +1.49, constraint order +0.20,
and Macro-F1 +0.02. Advance R3 and R7 to five-seed validation confirmation;
do not evaluate new checkpoints on test during this stage.

### R7 five-seed validation confirmation and model lock

Seeds: 13, 21, 42, 87, 100.

| Metric | R3 mean ± SD | R7 mean ± SD | Paired delta mean ± SD |
|---|---:|---:|---:|
| Macro-F1 | 0.6456 ± 0.0021 | 0.6459 ± 0.0078 | +0.0003 ± 0.0072 |
| Accuracy | 0.6459 ± 0.0021 | 0.6466 ± 0.0076 | +0.0007 ± 0.0071 |
| DIA | 0.4634 ± 0.0100 | 0.4726 ± 0.0104 | +0.0091 ± 0.0151 |
| NDI | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| CVC | 0.2924 ± 0.0112 | 0.3235 ± 0.0164 | +0.0311 ± 0.0107 |
| Verdict pair order | 0.6175 ± 0.0036 | 0.6353 ± 0.0064 | +0.0178 ± 0.0083 |
| Constraint pair order | 0.6228 ± 0.0072 | 0.6273 ± 0.0042 | +0.0045 ± 0.0072 |

CVC and verdict pair order improved for all five paired seeds. Mean verdict
quality was unchanged. DIA improved in three of five seeds and constraint order
in four of five, so both remain secondary outcomes. NDI=1.0 is structural for
independent nodes under minimal interventions and is only a sanity check.

The R7 configuration is now locked at commit/config provenance recorded by the
server metrics. Final test acceptance criteria, declared before evaluating the
remaining R7 test checkpoints, are: (1) positive mean paired delta for CVC and
verdict pair order; (2) Macro-F1 non-inferiority margin of -0.005 absolute;
(3) report DIA and constraint order as secondary regardless of direction. No
additional loss, architecture, seed, or checkpoint tuning is permitted before
the final comparison.

### R7 locked final test result

| Metric | R3 mean ± SD | R7 mean ± SD | Paired delta mean ± SD |
|---|---:|---:|---:|
| Macro-F1 | 0.6494 ± 0.0040 | 0.6437 ± 0.0086 | -0.0057 ± 0.0072 |
| Accuracy | 0.6495 ± 0.0041 | 0.6442 ± 0.0089 | -0.0053 ± 0.0074 |
| DIA | 0.4678 ± 0.0090 | 0.4795 ± 0.0061 | +0.0117 ± 0.0130 |
| NDI | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| CVC | 0.2952 ± 0.0071 | 0.3273 ± 0.0092 | +0.0321 ± 0.0065 |
| Verdict pair order | 0.6180 ± 0.0098 | 0.6315 ± 0.0090 | +0.0135 ± 0.0108 |
| Constraint pair order | 0.6210 ± 0.0043 | 0.6277 ± 0.0090 | +0.0067 ± 0.0105 |

The locked test met the two intervention criteria: CVC improved in all five
seeds and mean verdict pair order improved. It failed the preregistered verdict
non-inferiority criterion: Macro-F1 delta was -0.0057, below the fixed -0.005
margin. Therefore **R7 is rejected** as the primary method. The margin must not
be changed after observing this result. R7 remains a transparent Pareto result:
stronger intervention behavior at a small but criterion-exceeding verdict cost.

NewsCLIPpings test is now fully consumed and must not be used for further model
selection or acceptance decisions. Any R8 optimization uses NewsCLIPpings
validation only and requires an untouched external dataset for final
confirmation.

## R8: primary-protected intervention gradients

Status: validation-only development. Config:
`configs/newsclippings_minimal_pcgrad.yaml`. The primary gradient comprises
verdict and weak constraint classification. The auxiliary gradient comprises
minimal-intervention JS and directional ranking. When their global dot product
is negative, the conflicting component of the auxiliary gradient is projected
out before summation. Epoch logs record gradient cosine and conflict rate.

R8 is selected exclusively on NewsCLIPpings validation. It must improve the R7
intervention metrics while keeping validation Macro-F1 non-inferior; no
NewsCLIPpings test evaluation is authorized during R8 development.

Seed-42 validation screen:

| Model | Macro-F1 | DIA | CVC | Verdict order | Constraint order |
|---|---:|---:|---:|---:|---:|
| R3 reference | 0.6433 | 0.4547 | 0.2765 | 0.6189 | 0.6244 |
| R7 minimal | **0.6435** | **0.4636** | 0.2984 | 0.6338 | 0.6264 |
| R8 projected | 0.6434 | 0.4624 | **0.3166** | **0.6402** | **0.6307** |

R8 preserves verdict quality while improving CVC by 4.01 points, verdict pair
order by 2.14, constraint order by 0.63, and DIA by 0.77 relative to R3. Relative
to R7, it further improves CVC and both order metrics with essentially unchanged
Macro-F1, while DIA decreases by 0.12 points. This is provisional until gradient
conflict diagnostics are inspected and five-seed validation confirms stability.

Gradient diagnostics reject the proposed mechanism: mean primary/auxiliary
gradient cosine was +0.2311 and mean conflict rate only 0.0017 across 13 epochs.
Projection activated in only a few early/late batches. Therefore R8 must not be
described as a gradient-conflict solution and does not advance to five seeds.
The remaining implementation difference from R7 is FP32 versus AMP. R8-C
(`configs/newsclippings_minimal_fp32.yaml`) is a seed-42 validation control with
identical losses and FP32 training but no projection.

FP32 negative control:

| Model | Macro-F1 | DIA | CVC | Verdict order | Constraint order |
|---|---:|---:|---:|---:|---:|
| R7 AMP | **0.6435** | **0.4636** | 0.2984 | 0.6338 | 0.6264 |
| R8 projection FP32 | 0.6434 | 0.4624 | **0.3166** | **0.6402** | 0.6307 |
| R8-C FP32 no projection | 0.6425 | 0.4428 | **0.3166** | 0.6388 | **0.6347** |

R8-C exactly matches R8 CVC, nearly matches verdict order, and exceeds its
constraint order. Projection retains higher DIA than the FP32 control, but its
observed conflict rate is only 0.17% and this is a single-seed comparison.
Decision: reject gradient surgery as a contribution and do not run more R8
seeds. Treat the apparent R8 gains as an FP32/training control effect unless a
future independent benchmark provides contrary preregistered evidence.

NewsCLIPpings development is closed. The next research stage is external
evidence-grounded transfer and sequential acquisition on MOCHEG; no further
NewsCLIPpings test-guided optimization is allowed.

## MOCHEG external transfer audit

MOCHEG was downloaded from the official release, extracted, and validated with
magic-byte checks for all split images. The audit passed with 78,031 train,
12,267 validation, and 22,777 test images; no zero-byte or invalid payloads
were found (one valid PSD file was stored with a `.jpg` suffix). Claim-level
manifests contain 11,669/1,490/2,442 official train/val/test claims. A strict
protocol removed 38/34/8 duplicate claim texts from train/val/test, leaving
11,631/1,456/2,434 claims. The strict leakage audit then reported zero overlap
for claim IDs and normalized claim text in every split pair.

### MOCHEG modality and architecture controls

All results below use the strict split, seed 42 unless stated otherwise, and
three labels (`supported=0`, `refuted=1`, `NEI=2`).

| Model/control | Accuracy | Macro-F1 | Decision |
|---|---:|---:|---|
| MiniLM claim-only | 0.4499 | 0.4237 | baseline control |
| MiniLM claim + evidence | 0.4523 | 0.4053 | evidence pooling rejected |
| MiniLM evidence-only | 0.4507 | 0.3960 | rejected |
| MPNet claim-only | 0.4606 | 0.4350 | best single-seed model |
| MPNet claim + evidence | 0.4437 | 0.3974 | negative transfer |
| MPNet class-weighted | 0.4445 | 0.3850 | negative control |

MPNet claim-only five-seed results were Accuracy `0.4607 +/- 0.0072` and
Macro-F1 `0.4382 +/- 0.0083`; per-seed Macro-F1 values were
`[0.4373, 0.4266, 0.4350, 0.4439, 0.4483]`.

The MPNet claim-only architecture screen was:

| Architecture | Accuracy | Macro-F1 |
|---|---:|---:|
| independent | 0.4606 | 0.4350 |
| typed_graph | 0.4528 | 0.4269 |
| fully_connected | 0.4478 | 0.4185 |
| mlp | 0.4454 | 0.4201 |
| linear | 0.3981 | 0.3519 |

MOCHEG therefore does not provide evidence that the current typed graph or
multimodal fusion improves over an independent claim representation. We retain
these results as a negative-transfer cross-dataset diagnostic, not as a claim
of a graph improvement. The primary GraphCURE contribution remains the
intervention-faithful acquisition protocol established on NewsCLIPpings;
MOCHEG supplies a strict external transfer check and exposes evidence-pooling
failure modes.

### Official versus strict MOCHEG protocol

The best MPNet claim-only model was also evaluated on the untouched official
split (2,442 test claims) using the same seed and architecture. Results are:

| Protocol | Samples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Strict deduplicated | 2,434 | 0.4606 | 0.4350 |
| Official | 2,442 | 0.4566 | 0.4387 |

Official test per-class recall was 0.3733 (supported), 0.7079 (refuted), and
0.2825 (NEI). The close agreement between protocols indicates that duplicate
claim wording is not driving the observed transfer result. The remaining
failure mode is class confusion, especially the low recall of NEI and
supported claims, rather than split leakage.

### Adaptive open/closed router

Using TF-IDF retrieval confidence, a threshold was selected on validation and
then locked for test. The router used the closed MPNet claim-only checkpoint
below the threshold and the open retrieved-evidence checkpoint above it.

| Protocol | Macro-F1 | Accuracy | Open coverage |
|---|---:|---:|---:|
| Closed claim-only | 0.4350 | 0.4606 | 0.0% |
| Open fixed | 0.4045 | 0.4084 | 100.0% |
| Adaptive router | 0.4302 | 0.4515 | 13.1% |

The validation-selected threshold was `0.42` (validation Macro-F1 `0.4640`,
open coverage `10.4%`). On test, the router opened evidence for only `13.1%`
of claims and stayed within `0.0048` Macro-F1 of the closed baseline. It did
not exceed the closed baseline, so this is reported as a cost-sensitive
near-non-inferiority result rather than a definitive accuracy improvement.

### Dense top-1 adaptive router

Replacing TF-IDF with dense MPNet retrieval gave test Recall@1 `0.7436`. Using
the dense top-1 evidence checkpoint and expanding threshold selection to the
full cosine-score range `[0, 1]`, validation selected threshold `0.66`:

| Protocol | Macro-F1 | Accuracy | Open coverage |
|---|---:|---:|---:|
| Closed MPNet claim-only | 0.4350 | 0.4606 | 0.0% |
| Dense top-1 open fixed | 0.4209 | 0.4339 | 100.0% |
| Dense top-1 adaptive | **0.4422** | **0.4630** | 57.1% |

The adaptive router improved Macro-F1 by `+0.0072` over the closed baseline
while opening evidence for 57.1% of test claims. This is the first positive
open/closed routing result and must be confirmed with multiple seeds before it
is treated as the primary claim.

Five-seed confirmation (13, 21, 42, 87, 100) gave Accuracy `0.4610 +/- 0.0067`
and Macro-F1 `0.4413 +/- 0.0056`; the per-seed Macro-F1 values were
`[0.4395, 0.4400, 0.4422, 0.4349, 0.4501]`. Relative to the seed-42 closed
baseline Macro-F1 `0.4350`, the mean gain was `+0.0063`. Open coverage averaged
`0.5154 +/- 0.2660` with values `[0.4326, 0.4326, 0.5711, 0.9310, 0.2095]`.
The accuracy gain is promising, but this coverage variance requires calibrated
budget control before making a strong efficiency claim.

### Fixed-budget routing

To remove seed-dependent coverage variance, thresholds were selected on
validation quantiles for fixed open budgets. The dense top-1 router produced:

| Target budget | Val coverage | Test coverage | Test Accuracy | Test Macro-F1 |
|---:|---:|---:|---:|---:|
| 10% | 10.0% | 10.5% | **0.4688** | **0.4445** |
| 25% | 25.0% | 25.8% | 0.4659 | 0.4409 |
| 50% | 50.0% | 54.8% | 0.4651 | 0.4435 |

The 10% budget is the strongest cost-sensitive result: it improves Macro-F1
by `+0.0095` over closed claim-only (`0.4350`) while opening evidence for only
10.5% of test claims. This fixed-budget protocol is preferred over unconstrained
per-seed threshold selection for the main comparison.
