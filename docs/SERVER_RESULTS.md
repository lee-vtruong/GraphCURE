# Server results snapshot

Generated: 2026-08-05T10:19:42.028753+00:00

Git commit: `a202d8f7053d7332a6022c03f04574f51c516cad`

PyTorch: `2.13.0+cu130`; CUDA runtime: `13.0`

Discovered test metric files: 15

| Result path | Architecture | Seed | Samples | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|
| `outputs/ablations/multi_fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.641520 | 0.641067 |
| `outputs/ablations/multi_independent/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.652946 | 0.652922 |
| `outputs/ablations/multi_typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.644824 | 0.644310 |
| `outputs/ablations/newsclippings_clip_screen/fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.634774 | 0.633860 |
| `outputs/ablations/newsclippings_clip_screen/independent/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.636839 | 0.636569 |
| `outputs/ablations/newsclippings_clip_screen/linear/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.535380 | 0.535328 |
| `outputs/ablations/newsclippings_clip_screen/mlp/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.532902 | 0.532379 |
| `outputs/ablations/newsclippings_clip_screen/typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.631195 | 0.630390 |
| `outputs/ablations/newsclippings_constraints_screen/fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.622384 | 0.621664 |
| `outputs/ablations/newsclippings_constraints_screen/independent/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.631195 | 0.630822 |
| `outputs/ablations/newsclippings_constraints_screen/typed_graph/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.629268 | 0.629259 |
| `outputs/ablations/newsclippings_multiview_screen/multi_fully_connected/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.641520 | 0.641067 |
| `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_metrics.json` | legacy | NA | 7264 | 0.652946 | 0.652922 |
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

## `outputs/ablations/newsclippings_multiview_screen/multi_fully_connected/seed_42/test_metrics.json`

- Confusion matrix: `[[2459, 1173], [1431, 2201]]`
- Class 0: precision=0.632134, recall=0.677037, F1=0.653815
- Class 1: precision=0.652341, recall=0.606002, F1=0.628319

## `outputs/ablations/newsclippings_multiview_screen/multi_independent/seed_42/test_metrics.json`

- Confusion matrix: `[[2402, 1230], [1291, 2341]]`
- Class 0: precision=0.650420, recall=0.661344, F1=0.655836
- Class 1: precision=0.655559, recall=0.644548, F1=0.650007

## `outputs/ablations/newsclippings_multiview_screen/multi_typed_graph/seed_42/test_metrics.json`

- Confusion matrix: `[[2480, 1152], [1428, 2204]]`
- Class 0: precision=0.634596, recall=0.682819, F1=0.657825
- Class 1: precision=0.656734, recall=0.606828, F1=0.630796

## `outputs/newsclippings_embeddings/test_metrics.json`

- Confusion matrix: `[[2462, 1170], [1509, 2123]]`
- Class 0: precision=0.619995, recall=0.677863, F1=0.647639
- Class 1: precision=0.644701, recall=0.584526, F1=0.613141
