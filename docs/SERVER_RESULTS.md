# Server results snapshot

Generated: 2026-08-05T13:29:08.303934+00:00

Git commit: `ae34c7d61a29617dbe0ce3c09089dd49467a5a7d`

PyTorch: `2.13.0+cu130`; CUDA runtime: `13.0`

Discovered test metric files: 26

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
| `outputs/ablations/newsclippings_multiview_screen/multi_fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.641520 | 0.641067 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_100/test_metrics.json` | multi_independent | 100 | 7264 | 0.654598 | 0.654230 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/test_metrics.json` | multi_independent | 13 | 7264 | 0.645925 | 0.645921 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/test_metrics.json` | multi_independent | 21 | 7264 | 0.645787 | 0.645658 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_metrics.json` | multi_independent | 42 | 7264 | 0.652946 | 0.652922 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/test_metrics.json` | multi_independent | 87 | 7264 | 0.648403 | 0.648147 |
| `outputs/ablations/newsclippings_multiview_screen/multi_typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.644824 | 0.644310 |
| `outputs/newsclippings_embeddings/test_metrics.json` | legacy | NA | 7264 | 0.631195 | 0.630390 |

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

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/test_metrics.json`

- Confusion matrix: `[[2359, 1273], [1299, 2333]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_13/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 13, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6164647577092511, "counterfactual_verdict_consistency": 0.4966960352422907, "descendant_intervention_accuracy": 0.4686123348017621, "non_descendant_invariance": 0.7400881057268722, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7045704845814978}`
- Class 0: precision=0.644888, recall=0.649504, F1=0.647188
- Class 1: precision=0.646977, recall=0.642346, F1=0.644653

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/test_metrics.json`

- Confusion matrix: `[[2415, 1217], [1356, 2276]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_21/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 21, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.49807268722466963, "descendant_intervention_accuracy": 0.45759911894273125, "non_descendant_invariance": 0.7452276064610867, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7125550660792952}`
- Class 0: precision=0.640414, recall=0.664923, F1=0.652438
- Class 1: precision=0.651589, recall=0.626652, F1=0.638877

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2402, 1230], [1291, 2341]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "70c4b8cfa12bf9862648de4bf1adf887896c2caa", "gpu": "NVIDIA GeForce RTX 5090", "seed": 42, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.6186674008810573, "counterfactual_verdict_consistency": 0.5063325991189427, "descendant_intervention_accuracy": 0.4600770925110132, "non_descendant_invariance": 0.7406387665198237, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7131057268722467}`
- Class 0: precision=0.650420, recall=0.661344, F1=0.655836
- Class 1: precision=0.655559, recall=0.644548, F1=0.650007

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/test_metrics.json`

- Confusion matrix: `[[2453, 1179], [1375, 2257]]`
- Provenance: `{"architecture": "multi_independent", "checkpoint": "outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_87/best.pt", "config_file": "configs/newsclippings_multiview.yaml", "config_sha256": "a79bf3c59a9a4e09ffe6542d5093c00591e5d6381ab7d5ed73d555786af9fda1", "cuda_runtime": "13.0", "git_commit": "ae34c7d61a29617dbe0ce3c09089dd49467a5a7d", "gpu": "NVIDIA GeForce RTX 5090", "seed": 87, "torch": "2.13.0+cu130"}`
- Counterfactual metrics: `{"constraint_pair_order_accuracy": 0.626101321585903, "counterfactual_verdict_consistency": 0.5033039647577092, "descendant_intervention_accuracy": 0.4743942731277533, "non_descendant_invariance": 0.7380690161527166, "note": "paired rows are evaluated bidirectionally; CVC expects verdict flip", "verdict_pair_order_accuracy": 0.7153083700440529}`
- Class 0: precision=0.640805, recall=0.675385, F1=0.657641
- Class 1: precision=0.656868, recall=0.621421, F1=0.638653

## `outputs/ablations/newsclippings_multiview_screen/multi_typed_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2480, 1152], [1428, 2204]]`
- Class 0: precision=0.634596, recall=0.682819, F1=0.657825
- Class 1: precision=0.656734, recall=0.606828, F1=0.630796

## `outputs/newsclippings_embeddings/test_metrics.json`

- Confusion matrix: `[[2462, 1170], [1509, 2123]]`
- Class 0: precision=0.619995, recall=0.677863, F1=0.647639
- Class 1: precision=0.644701, recall=0.584526, F1=0.613141
