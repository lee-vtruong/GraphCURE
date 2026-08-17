# Server results snapshot

Generated: 2026-08-05T15:23:27.157199+00:00

Git commit: `86616c3aa87237666212961e134ea8bf49bab642`

PyTorch: `2.13.0+cu130`; CUDA runtime: `13.0`

## GraphCURE-R2V validation ledger (2026-08-17)

The entries in this section are validation-only development results. Official
test retrieval remains locked until architecture and hyperparameters freeze.

### R2V-RET-01 — Qwen3-Embedding-4B hybrid candidate retrieval

- Split: MOCHEG strict validation (`n=1456`), frozen local corpus (`4067` docs)
- Output: `outputs/retrieval_mocheg_qwen3_hybrid/summary.json`
- Recall@1: **0.774725**
- Recall@5: **0.896978**
- Recall@10: **0.927198**
- Recall@50: **0.953984**
- MRR: **0.829072**
- Matched dense-top50 reference: Recall@50 `0.943681`, MRR `0.817854`
- Delta versus matched reference: Recall@50 **+0.010302**, MRR **+0.011218**
- Decision: **candidate-retrieval gate passed**; proceed to validation-only
  Qwen3-Reranker-4B screening.

### R2V-RER-00 — invalid Qwen3 reranker implementation

- Observed Recall@10: `0.916896`; MRR: `0.289560`
- Status: **invalidated; exclude from scientific comparison**
- Cause: the first implementation truncated the fully formatted prompt. Long
  evidence could remove the required assistant suffix, so the final-token
  `yes`/`no` logits were not valid relevance scores.
- Resolution: truncate only query-document content, then append the fixed Qwen
  prefix/suffix exactly as specified by the official model card. Rerun as
  `R2V-RER-01` on validation only.

### R2V-RER-01 — corrected Qwen3-Reranker-4B

- Split: MOCHEG strict validation (`n=1456`), Qwen3 hybrid top-50 candidates
- Output: `outputs/retrieval_mocheg_qwen3_reranked/summary.json`
- Recall@10: **0.945742**
- MRR: **0.886368**
- Delta versus Qwen3 hybrid pre-reranking: Recall@10 **+0.018544**,
  MRR **+0.057296**
- Reachable-hit retention: `0.945742 / 0.953984 = 0.99136`; the reranker
  retains 99.1% of validation claims whose gold evidence exists in candidate
  top-50 while moving relevant evidence substantially earlier.
- Decision: **reranking gate passed**; materialize train retrieval/reranking,
  then train the claim-level evidence-set verifier. Official test remains locked.

### R2V-RET-TRAIN-01 — Qwen3 hybrid train materialization

- Split: MOCHEG strict train (`n=11631`), frozen corpus (`18989` docs)
- Claims with annotated text evidence: `7337`; raw retrieval ceiling when
  no-text-qrel claims count as misses: `7337/11631 = 0.630814`
- Raw Recall@1/5/10/50: `0.429628 / 0.534778 / 0.562032 / 0.592382`
- Raw MRR: `0.476523`
- Conditional on claims with annotated text evidence, Recall@1/5/10/50:
  `0.681069 / 0.847758 / 0.890964 / 0.939076`
- Conditional MRR: `0.755410`
- Matched dense-top50 train reference: raw Recall@50 `0.581549`, MRR
  `0.474593`
- Decision: **train candidate gate passed**. The raw train/validation gap is
  primarily annotation availability; both raw and coverage-conditional metrics
  must be reported.

### R2V-RER-TRAIN-01 — Qwen3-Reranker-4B train materialization

- Split: MOCHEG strict train (`n=11631`), Qwen3 hybrid top-50 candidates
- Raw Recall@1/5/10: `0.485599 / 0.567535 / 0.581377`
- Raw MRR: `0.521653`
- Conditional on the `7337` claims with annotated text evidence,
  Recall@1/5/10: `0.769797 / 0.899687 / 0.921630`
- Conditional MRR: `0.826952`
- Delta versus pre-reranking train candidates: Recall@1 `+0.055971`,
  Recall@5 `+0.032757`, Recall@10 `+0.019345`, MRR `+0.045130`
- Candidate-hit retention in top-10: `0.581377 / 0.592382 = 0.981422`
- Decision: **train reranking gate passed**; mine structured reasoning traps and
  train the evidence-utility/sufficiency verifier. Official test remains locked.

Discovered test metric files: 50

| Result path | Architecture | Seed | Samples | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|
| `outputs/ablations/multi_fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.641520 | 0.641067 |
| `outputs/ablations/multi_independent/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.652946 | 0.652922 |
| `outputs/ablations/multi_typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.644824 | 0.644310 |
| `outputs/ablations/newsclippings_adaptive_screen/multi_adaptive_graph/seed_42/test_metrics.json` | multi_adaptive_graph | 42 | 7264 | 0.637252 | 0.637141 |
| `outputs/ablations/newsclippings_clip_screen/fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.634774 | 0.633860 |
| `outputs/ablations/newsclippings_clip_screen/independent/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.636839 | 0.636569 |
| `outputs/ablations/newsclippings_clip_screen/linear/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.535380 | 0.535328 |
| `outputs/ablations/newsclippings_clip_screen/mlp/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.532902 | 0.532379 |
| `outputs/ablations/newsclippings_clip_screen/typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.631195 | 0.630390 |
| `outputs/ablations/newsclippings_constraints_screen/fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.622384 | 0.621664 |
| `outputs/ablations/newsclippings_constraints_screen/independent/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.631195 | 0.630822 |
| `outputs/ablations/newsclippings_constraints_screen/typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.629268 | 0.629259 |
| `outputs/ablations/newsclippings_counterfactual_screen/multi_independent/seed_42/test_metrics.json` | multi_independent | 42 | 7264 | 0.646063 | 0.645786 |
| `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_100/test_metrics.json` | multi_independent | 100 | 7264 | 0.660242 | 0.660159 |
| `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_13/test_metrics.json` | multi_independent | 13 | 7264 | 0.647990 | 0.647492 |
| `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_21/test_metrics.json` | multi_independent | 21 | 7264 | 0.642483 | 0.642381 |
| `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_42/test_metrics.json` | multi_independent | 42 | 7264 | 0.653634 | 0.653397 |
| `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_87/test_metrics.json` | multi_independent | 87 | 7264 | 0.644273 | 0.644083 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/test_minimal_metrics.json` | multi_independent | 100 | 7264 | 0.652395 | 0.652393 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/val_metrics.json` | multi_independent | 100 | 7024 | 0.652904 | 0.652828 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/test_minimal_metrics.json` | multi_independent | 13 | 7264 | 0.649367 | 0.649227 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/val_metrics.json` | multi_independent | 13 | 7024 | 0.633685 | 0.633419 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/test_minimal_metrics.json` | multi_independent | 21 | 7264 | 0.636977 | 0.636972 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/val_metrics.json` | multi_independent | 21 | 7024 | 0.648918 | 0.648872 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/test_metrics.json` | multi_independent | 42 | 7264 | 0.649917 | 0.647456 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/test_minimal_metrics.json` | multi_independent | 42 | 7264 | 0.649917 | 0.647456 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/val_metrics.json` | multi_independent | 42 | 7024 | 0.646498 | 0.643473 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/test_minimal_metrics.json` | multi_independent | 87 | 7264 | 0.632434 | 0.632434 |
| `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/val_metrics.json` | multi_independent | 87 | 7024 | 0.650769 | 0.650733 |
| `outputs/ablations/newsclippings_minimal_reference/multi_independent/seed_42/test_metrics.json` | multi_independent | 42 | 7264 | 0.652946 | 0.652922 |
| `outputs/ablations/newsclippings_multiview_screen/multi_fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.641520 | 0.641067 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/test_metrics.json` | multi_independent | 100 | 7264 | 0.654598 | 0.654230 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/test_minimal_metrics.json` | multi_independent | 100 | 7264 | 0.654598 | 0.654230 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/val_metrics.json` | multi_independent | 100 | 7024 | 0.645074 | 0.644455 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/test_metrics.json` | multi_independent | 13 | 7264 | 0.645925 | 0.645921 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/test_minimal_metrics.json` | multi_independent | 13 | 7264 | 0.645925 | 0.645921 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/val_metrics.json` | multi_independent | 13 | 7024 | 0.644789 | 0.644642 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/test_metrics.json` | multi_independent | 21 | 7264 | 0.645787 | 0.645658 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/test_minimal_metrics.json` | multi_independent | 21 | 7264 | 0.645787 | 0.645658 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/val_metrics.json` | multi_independent | 21 | 7024 | 0.648491 | 0.648203 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_metrics.json` | multi_independent | 42 | 7264 | 0.652946 | 0.652922 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_minimal_metrics.json` | multi_independent | 42 | 7264 | 0.652946 | 0.652922 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/val_metrics.json` | multi_independent | 42 | 7024 | 0.643366 | 0.643283 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/test_metrics.json` | multi_independent | 87 | 7264 | 0.648403 | 0.648147 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/test_minimal_metrics.json` | multi_independent | 87 | 7264 | 0.648403 | 0.648147 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/val_metrics.json` | multi_independent | 87 | 7024 | 0.647637 | 0.647186 |
| `outputs/ablations/newsclippings_multiview_screen/multi_typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.644824 | 0.644310 |
| `outputs/newsclippings_embeddings/test_metrics.json` | legacy | NA | 7264 | 0.631195 | 0.630390 |
| `outputs/validation/r3_minimal/val_metrics.json` | multi_independent | 42 | 7024 | 0.643366 | 0.643283 |
| `outputs/validation/r7_minimal/val_metrics.json` | multi_independent | 42 | 7024 | 0.646498 | 0.643473 |

## `outputs/ablations/multi_fully_connected/seed_42/test_metrics.json`

- Confusion matrix: `[[2459, 1173], [1431, 2201]]`
- Class 0: precision=0.632134, recall=0.677037, F1=0.653815
- Class 1: precision=0.652341, recall=0.606002, F1=0.628319

## `outputs/ablations/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2402, 1230], [1291, 2341]]`
- Class 0: precision=0.650420, recall=0.661344, F1=0.655836
- Class 1: precision=0.655559, recall=0.644548, F1=0.650007

## `outputs/ablations/multi_typed_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2480, 1152], [1428, 2204]]`
- Class 0: precision=0.634596, recall=0.682819, F1=0.657825
- Class 1: precision=0.656734, recall=0.606828, F1=0.630796

## `outputs/ablations/newsclippings_adaptive_screen/multi_adaptive_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2251, 1381], [1254, 2378]]`
- Provenance: `{"architecture": "multi_adaptive_graph", "checkpoint": "outputs/ablations/newsclippings_adaptive_screen/multi_adaptive_graph/seed_42/best.pt", "config_file": "configs/newsclippings_adaptive.yaml", "config_sha256": "0912bf05ed5ce5a2c9dffc5faa38ec282441fb205253735d2fd839a47f1043df", "cuda_runtime": "13.0", "git_commit": "b653494dd131b982108874c8cc5b21d46bd34ef9", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Node mix gate mean `[semantic, entity, temporal, contextual]`: `[0.8219451308250427, 0.22429713606834412, 0.5543850064277649, 0.7950358390808105]`
- Node mix gate std: `[0.2824525535106659, 0.10313897579908371, 3.838471820927225e-05, 0.19432076811790466]`
- Class 0: precision=0.642225, recall=0.619769, F1=0.630797
- Class 1: precision=0.632615, recall=0.654736, F1=0.643485

## `outputs/ablations/newsclippings_clip_screen/fully_connected/seed_42/test_metrics.json`

- Confusion matrix: `[[2487, 1145], [1508, 2124]]`
- Class 0: precision=0.622528, recall=0.684747, F1=0.652157
- Class 1: precision=0.649740, recall=0.584802, F1=0.615563

## `outputs/ablations/newsclippings_clip_screen/independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2412, 1220], [1418, 2214]]`
- Class 0: precision=0.629765, recall=0.664097, F1=0.646475
- Class 1: precision=0.644729, recall=0.609581, F1=0.626663

## `outputs/ablations/newsclippings_clip_screen/linear/seed_42/test_metrics.json`

- Confusion matrix: `[[1983, 1649], [1726, 1906]]`
- Class 0: precision=0.534645, recall=0.545980, F1=0.540253
- Class 1: precision=0.536146, recall=0.524780, F1=0.530402

## `outputs/ablations/newsclippings_clip_screen/mlp/seed_42/test_metrics.json`

- Confusion matrix: `[[1814, 1818], [1575, 2057]]`
- Class 0: precision=0.535261, recall=0.499449, F1=0.516736
- Class 1: precision=0.530839, recall=0.566355, F1=0.548022

## `outputs/ablations/newsclippings_clip_screen/typed_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2462, 1170], [1509, 2123]]`
- Class 0: precision=0.619995, recall=0.677863, F1=0.647639
- Class 1: precision=0.644701, recall=0.584526, F1=0.613141

## `outputs/ablations/newsclippings_constraints_screen/fully_connected/seed_42/test_metrics.json`

- Confusion matrix: `[[2419, 1213], [1530, 2102]]`
- Class 0: precision=0.612560, recall=0.666024, F1=0.638174
- Class 1: precision=0.634087, recall=0.578744, F1=0.605153

## `outputs/ablations/newsclippings_constraints_screen/independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2177, 1455], [1224, 2408]]`
- Class 0: precision=0.640106, recall=0.599394, F1=0.619081
- Class 1: precision=0.623350, recall=0.662996, F1=0.642562

## `outputs/ablations/newsclippings_constraints_screen/typed_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2268, 1364], [1329, 2303]]`
- Class 0: precision=0.630525, recall=0.624449, F1=0.627473
- Class 1: precision=0.628034, recall=0.634086, F1=0.631045

## `outputs/ablations/newsclippings_counterfactual_screen/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2448, 1184], [1387, 2245]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_counterfactual_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_counterfactual.yaml", "config_sha256": "821facb3374a5607c01b16b4f8324efded28dd1465b650724451aae89bfd1182", "cuda_runtime": "13.0", "git_commit": "70c4b8cfa12bf9862648de4bf1adf887896c2caa", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6109581497797357, "counterfactual_verdict_consistency": 0.5008259911894273, "descendant_intervention_accuracy": 0.4821035242290749, "non_descendant_invariance": 0.7676211453744494, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.70181718061674}`
- Class 0: precision=0.638331, recall=0.674009, F1=0.655685
- Class 1: precision=0.654710, recall=0.618117, F1=0.635887

## `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_100/test_metrics.json`

- Confusion matrix: `[[2455, 1177], [1291, 2341]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_directional_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_directional.yaml", "config_sha256": "0edc4ee9fc8887206d055ee7dcc305e840e4ddac8b83f1497f99e7250db39c7a", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6327092511013216, "counterfactual_verdict_consistency": 0.513215859030837, "descendant_intervention_accuracy": 0.45787444933920707, "non_descendant_invariance": 0.7549559471365639, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7243942731277533}`
- Class 0: precision=0.655366, recall=0.675936, F1=0.665492
- Class 1: precision=0.665435, recall=0.644548, F1=0.654825

## `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_13/test_metrics.json`

- Confusion matrix: `[[2490, 1142], [1415, 2217]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_directional_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_directional.yaml", "config_sha256": "0edc4ee9fc8887206d055ee7dcc305e840e4ddac8b83f1497f99e7250db39c7a", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6178414096916299, "counterfactual_verdict_consistency": 0.49531938325991187, "descendant_intervention_accuracy": 0.4754955947136564, "non_descendant_invariance": 0.7421071953010279, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7213656387665198}`
- Class 0: precision=0.637644, recall=0.685573, F1=0.660740
- Class 1: precision=0.660018, recall=0.610407, F1=0.634244

## `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_21/test_metrics.json`

- Confusion matrix: `[[2272, 1360], [1237, 2395]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_directional_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_directional.yaml", "config_sha256": "0edc4ee9fc8887206d055ee7dcc305e840e4ddac8b83f1497f99e7250db39c7a", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6288546255506607, "counterfactual_verdict_consistency": 0.5024779735682819, "descendant_intervention_accuracy": 0.4834801762114537, "non_descendant_invariance": 0.7315528634361234, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7059471365638766}`
- Class 0: precision=0.647478, recall=0.625551, F1=0.636325
- Class 1: precision=0.637816, recall=0.659416, F1=0.648436

## `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2469, 1163], [1353, 2279]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_directional_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_directional.yaml", "config_sha256": "0edc4ee9fc8887206d055ee7dcc305e840e4ddac8b83f1497f99e7250db39c7a", "cuda_runtime": "13.0", "git_commit": "70c4b8cfa12bf9862648de4bf1adf887896c2caa", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6227973568281938, "counterfactual_verdict_consistency": 0.493942731277533, "descendant_intervention_accuracy": 0.4642070484581498, "non_descendant_invariance": 0.7510095447870778, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7268722466960352}`
- Class 0: precision=0.645997, recall=0.679791, F1=0.662463
- Class 1: precision=0.662115, recall=0.627478, F1=0.644331

## `outputs/ablations/newsclippings_directional_screen/multi_independent/seed_87/test_metrics.json`

- Confusion matrix: `[[2424, 1208], [1376, 2256]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_directional_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_directional.yaml", "config_sha256": "0edc4ee9fc8887206d055ee7dcc305e840e4ddac8b83f1497f99e7250db39c7a", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6159140969162996, "counterfactual_verdict_consistency": 0.4944933920704846, "descendant_intervention_accuracy": 0.4688876651982379, "non_descendant_invariance": 0.7354074889867841, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7029185022026432}`
- Class 0: precision=0.637895, recall=0.667401, F1=0.652314
- Class 1: precision=0.651270, recall=0.621145, F1=0.635851

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/test_minimal_metrics.json`

- Confusion matrix: `[[2379, 1253], [1272, 2360]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.637114537444934, "counterfactual_verdict_consistency": 0.32544052863436124, "descendant_intervention_accuracy": 0.47384361233480177, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6422081497797357}`
- Class 0: precision=0.651602, recall=0.655011, F1=0.653302
- Class 1: precision=0.653197, recall=0.649780, F1=0.651484

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/val_metrics.json`

- Confusion matrix: `[[2345, 1167], [1271, 2241]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6341116173120729, "counterfactual_verdict_consistency": 0.31605922551252846, "descendant_intervention_accuracy": 0.46070615034168566, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6446469248291572}`
- Class 0: precision=0.648507, recall=0.667711, F1=0.657969
- Class 1: precision=0.657570, recall=0.638098, F1=0.647688

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/test_minimal_metrics.json`

- Confusion matrix: `[[2431, 1201], [1346, 2286]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6230726872246696, "counterfactual_verdict_consistency": 0.3354900881057269, "descendant_intervention_accuracy": 0.47879955947136565, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6270649779735683}`
- Class 0: precision=0.643633, recall=0.669328, F1=0.656229
- Class 1: precision=0.655578, recall=0.629405, F1=0.642225

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/val_metrics.json`

- Confusion matrix: `[[2320, 1192], [1381, 2131]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6272779043280182, "counterfactual_verdict_consistency": 0.33015375854214124, "descendant_intervention_accuracy": 0.47494305239179957, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6278473804100227}`
- Class 0: precision=0.626858, recall=0.660592, F1=0.643283
- Class 1: precision=0.641288, recall=0.606777, F1=0.623555

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/test_minimal_metrics.json`

- Confusion matrix: `[[2300, 1332], [1305, 2327]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6283039647577092, "counterfactual_verdict_consistency": 0.3310848017621145, "descendant_intervention_accuracy": 0.4887114537444934, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6269273127753304}`
- Class 0: precision=0.638003, recall=0.633260, F1=0.635622
- Class 1: precision=0.635966, recall=0.640694, F1=0.638321

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/val_metrics.json`

- Confusion matrix: `[[2239, 1273], [1193, 2319]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6261389521640092, "counterfactual_verdict_consistency": 0.3364179954441913, "descendant_intervention_accuracy": 0.48604783599088835, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6321184510250569}`
- Class 0: precision=0.652389, recall=0.637528, F1=0.644873
- Class 1: precision=0.645601, recall=0.660308, F1=0.652872

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2664, 968], [1575, 2057]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b337c801a2be8d0ae840222a293d489cc500663a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6349118942731278, "counterfactual_verdict_consistency": 0.31208700440528636, "descendant_intervention_accuracy": 0.4743942731277533, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6398678414096917}`
- Class 0: precision=0.628450, recall=0.733480, F1=0.676915
- Class 1: precision=0.680000, recall=0.566355, F1=0.617996

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/test_minimal_metrics.json`

- Confusion matrix: `[[2664, 968], [1575, 2057]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6349118942731278, "counterfactual_verdict_consistency": 0.31208700440528636, "descendant_intervention_accuracy": 0.4743942731277533, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6398678414096917}`
- Class 0: precision=0.628450, recall=0.733480, F1=0.676915
- Class 1: precision=0.680000, recall=0.566355, F1=0.617996

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/val_metrics.json`

- Confusion matrix: `[[2594, 918], [1565, 1947]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6264236902050114, "counterfactual_verdict_consistency": 0.2984054669703872, "descendant_intervention_accuracy": 0.4635535307517084, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6338268792710706}`
- Class 0: precision=0.623708, recall=0.738610, F1=0.676313
- Class 1: precision=0.679581, recall=0.554385, F1=0.610632

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/test_minimal_metrics.json`

- Confusion matrix: `[[2299, 1333], [1337, 2295]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6150881057268722, "counterfactual_verdict_consistency": 0.3323237885462555, "descendant_intervention_accuracy": 0.4818281938325991, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.621420704845815}`
- Class 0: precision=0.632288, recall=0.632985, F1=0.632636
- Class 1: precision=0.632580, recall=0.631883, F1=0.632231

## `outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/val_metrics.json`

- Confusion matrix: `[[2321, 1191], [1262, 2250]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6227220956719818, "counterfactual_verdict_consistency": 0.33670273348519364, "descendant_intervention_accuracy": 0.47750569476082005, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6380979498861048}`
- Class 0: precision=0.647781, recall=0.660877, F1=0.654264
- Class 1: precision=0.653880, recall=0.640661, F1=0.647203

## `outputs/ablations/newsclippings_minimal_reference/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2402, 1230], [1291, 2341]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "f413e23ca05ea06b9dd0076bce39e2c55cab4842", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.2889592511013216, "descendant_intervention_accuracy": 0.4600770925110132, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6188050660792952}`
- Class 0: precision=0.650420, recall=0.661344, F1=0.655836
- Class 1: precision=0.655559, recall=0.644548, F1=0.650007

## `outputs/ablations/newsclippings_multiview_screen/multi_fully_connected/seed_42/test_metrics.json`

- Confusion matrix: `[[2459, 1173], [1431, 2201]]`
- Class 0: precision=0.632134, recall=0.677037, F1=0.653815
- Class 1: precision=0.652341, recall=0.606002, F1=0.628319

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/test_metrics.json`

- Confusion matrix: `[[2496, 1136], [1373, 2259]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.625, "counterfactual_verdict_consistency": 0.5008259911894273, "descendant_intervention_accuracy": 0.4785242290748899, "non_descendant_invariance": 0.7599118942731278, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7169603524229075}`
- Class 0: precision=0.645128, recall=0.687225, F1=0.665511
- Class 1: precision=0.665390, recall=0.621971, F1=0.642949

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/test_minimal_metrics.json`

- Confusion matrix: `[[2496, 1136], [1373, 2259]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.625, "counterfactual_verdict_consistency": 0.29033590308370044, "descendant_intervention_accuracy": 0.4785242290748899, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6284416299559471}`
- Class 0: precision=0.645128, recall=0.687225, F1=0.665511
- Class 1: precision=0.665390, recall=0.621971, F1=0.642949

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/val_metrics.json`

- Confusion matrix: `[[2412, 1100], [1393, 2119]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 100, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6255694760820045, "counterfactual_verdict_consistency": 0.2901480637813212, "descendant_intervention_accuracy": 0.46554669703872437, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6170273348519362}`
- Class 0: precision=0.633903, recall=0.686788, F1=0.659287
- Class 1: precision=0.658279, recall=0.603360, F1=0.629624

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/test_metrics.json`

- Confusion matrix: `[[2359, 1273], [1299, 2333]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6164647577092511, "counterfactual_verdict_consistency": 0.4966960352422907, "descendant_intervention_accuracy": 0.4686123348017621, "non_descendant_invariance": 0.7400881057268722, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7045704845814978}`
- Class 0: precision=0.644888, recall=0.649504, F1=0.647188
- Class 1: precision=0.646977, recall=0.642346, F1=0.644653

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/test_minimal_metrics.json`

- Confusion matrix: `[[2359, 1273], [1299, 2333]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6164647577092511, "counterfactual_verdict_consistency": 0.3025881057268722, "descendant_intervention_accuracy": 0.4686123348017621, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6065528634361234}`
- Class 0: precision=0.644888, recall=0.649504, F1=0.647188
- Class 1: precision=0.646977, recall=0.642346, F1=0.644653

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/val_metrics.json`

- Confusion matrix: `[[2336, 1176], [1319, 2193]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6153189066059226, "counterfactual_verdict_consistency": 0.3078018223234624, "descendant_intervention_accuracy": 0.4772209567198178, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6164578587699316}`
- Class 0: precision=0.639124, recall=0.665148, F1=0.651877
- Class 1: precision=0.650935, recall=0.624431, F1=0.637407

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/test_metrics.json`

- Confusion matrix: `[[2415, 1217], [1356, 2276]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.49807268722466963, "descendant_intervention_accuracy": 0.45759911894273125, "non_descendant_invariance": 0.7452276064610867, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7125550660792952}`
- Class 0: precision=0.640414, recall=0.664923, F1=0.652438
- Class 1: precision=0.651589, recall=0.626652, F1=0.638877

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/test_minimal_metrics.json`

- Confusion matrix: `[[2415, 1217], [1356, 2276]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.2906112334801762, "descendant_intervention_accuracy": 0.45759911894273125, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6095814977973568}`
- Class 0: precision=0.640414, recall=0.664923, F1=0.652438
- Class 1: precision=0.651589, recall=0.626652, F1=0.638877

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/val_metrics.json`

- Confusion matrix: `[[2378, 1134], [1335, 2177]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6326879271070615, "counterfactual_verdict_consistency": 0.29256833712984054, "descendant_intervention_accuracy": 0.4527334851936219, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6225797266514806}`
- Class 0: precision=0.640452, recall=0.677107, F1=0.658270
- Class 1: precision=0.657505, recall=0.619875, F1=0.638136

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2402, 1230], [1291, 2341]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "70c4b8cfa12bf9862648de4bf1adf887896c2caa", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.5063325991189427, "descendant_intervention_accuracy": 0.4600770925110132, "non_descendant_invariance": 0.7406387665198237, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7131057268722467}`
- Class 0: precision=0.650420, recall=0.661344, F1=0.655836
- Class 1: precision=0.655559, recall=0.644548, F1=0.650007

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_minimal_metrics.json`

- Confusion matrix: `[[2402, 1230], [1291, 2341]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.2889592511013216, "descendant_intervention_accuracy": 0.4600770925110132, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6188050660792952}`
- Class 0: precision=0.650420, recall=0.661344, F1=0.655836
- Class 1: precision=0.655559, recall=0.644548, F1=0.650007

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/val_metrics.json`

- Confusion matrix: `[[2313, 1199], [1306, 2206]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6244305239179955, "counterfactual_verdict_consistency": 0.27648063781321186, "descendant_intervention_accuracy": 0.4547266514806378, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.618878132118451}`
- Class 0: precision=0.639127, recall=0.658599, F1=0.648717
- Class 1: precision=0.647871, recall=0.628132, F1=0.637849

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/test_metrics.json`

- Confusion matrix: `[[2453, 1179], [1375, 2257]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.626101321585903, "counterfactual_verdict_consistency": 0.5033039647577092, "descendant_intervention_accuracy": 0.4743942731277533, "non_descendant_invariance": 0.7380690161527166, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7153083700440529}`
- Class 0: precision=0.640805, recall=0.675385, F1=0.657641
- Class 1: precision=0.656868, recall=0.621421, F1=0.638653

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/test_minimal_metrics.json`

- Confusion matrix: `[[2453, 1179], [1375, 2257]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "b979e4e4d4f186c84938b91aa99bd6f9d026747a", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "split": "test", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.626101321585903, "counterfactual_verdict_consistency": 0.3032764317180617, "descendant_intervention_accuracy": 0.4743942731277533, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6265143171806168}`
- Class 0: precision=0.640805, recall=0.675385, F1=0.657641
- Class 1: precision=0.656868, recall=0.621421, F1=0.638653

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/val_metrics.json`

- Confusion matrix: `[[2400, 1112], [1363, 2149]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "5bf7afe3c642d628246a09b3e21d37bc3853912f", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6161731207289294, "counterfactual_verdict_consistency": 0.295130979498861, "descendant_intervention_accuracy": 0.46697038724373574, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6126138952164009}`
- Class 0: precision=0.637789, recall=0.683371, F1=0.659794
- Class 1: precision=0.659000, recall=0.611902, F1=0.634578

## `outputs/ablations/newsclippings_multiview_screen/multi_typed_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2480, 1152], [1428, 2204]]`
- Class 0: precision=0.634596, recall=0.682819, F1=0.657825
- Class 1: precision=0.656734, recall=0.606828, F1=0.630796

## `outputs/newsclippings_embeddings/test_metrics.json`

- Confusion matrix: `[[2462, 1170], [1509, 2123]]`
- Class 0: precision=0.619995, recall=0.677863, F1=0.647639
- Class 1: precision=0.644701, recall=0.584526, F1=0.613141

## `outputs/validation/r3_minimal/val_metrics.json`

- Confusion matrix: `[[2313, 1199], [1306, 2206]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "9eee88950c48d789f0eaf7afbbe69c38d666ac2e", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6244305239179955, "counterfactual_verdict_consistency": 0.27648063781321186, "descendant_intervention_accuracy": 0.4547266514806378, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.618878132118451}`
- Class 0: precision=0.639127, recall=0.658599, F1=0.648717
- Class 1: precision=0.647871, recall=0.628132, F1=0.637849

## `outputs/validation/r7_minimal/val_metrics.json`

- Confusion matrix: `[[2594, 918], [1565, 1947]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_minimal_intervention_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_minimal_intervention.yaml", "config_sha256": "c969735e3aa7c6766ed0ef3c5ab6e15bfd69d3933f2bb9e5dceea33d0c97dabb", "cuda_runtime": "13.0", "git_commit": "9eee88950c48d789f0eaf7afbbe69c38d666ac2e", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "split": "val", "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6264236902050114, "counterfactual_verdict_consistency": 0.2984054669703872, "descendant_intervention_accuracy": 0.4635535307517084, "non_descendant_invariance": 1.0, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.6338268792710706}`
- Class 0: precision=0.623708, recall=0.738610, F1=0.676313
- Class 1: precision=0.679581, recall=0.554385, F1=0.610632
