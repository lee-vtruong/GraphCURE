# MOCHEG protocol-aware comparison

Last checked: 2026-08-22.

Results must be separated by evidence protocol. `P1` uses a fixed MOCHEG
knowledge corpus and system-retrieved evidence; it is not open-web retrieval.
Gold-evidence and open-web results are not directly comparable with P1.

## P1 fixed-corpus retrieved evidence

| Method | Accuracy | Macro-F1 | Evidence/modalities | Status |
|---|---:|---:|---|---|
| MOCHEG | 0.456 | 0.438 | retrieved text + image | published |
| LVLM4FV | 0.451 | 0.450 | retrieved text + image | published |
| HGTMFC | 0.486 | 0.468 | retrieved text + image | published |
| MetaSumPerceiver | 0.486 | not reported reproducibly | retrieved text + image | published |
| AMuFC (arXiv v2) | 0.546 | 0.540 | retrieved text + image; Analyzer + VLM Verifier | published preprint |
| AMuFC (stronger workshop report) | 0.5577 | **0.5560** | retrieved text + image; Analyzer + VLM Verifier | conservative target |
| GraphCURE-R2V | 0.4961 +/- 0.0108 | 0.4711 +/- 0.0118 | retrieved text only; five seeds | strict deduplicated split |
| GraphCURE-Qwen3 (raw ensemble) | 0.5690 | 0.5458 | fixed-corpus retrieved text; five frozen LoRA seeds | strict deduplicated split |
| GraphCURE-Qwen3 (raw ensemble) | **0.5680** | 0.5453 | fixed-corpus retrieved text; five frozen LoRA seeds | official split |

Primary comparison source for the common table: AMuFC, Table 3,
<https://arxiv.org/abs/2604.04692> and
<https://openreview.net/pdf?id=IPGgVvGPwQ>.

GraphCURE-R2V numerically exceeds the preregistered HGTMFC milestone by
`+0.0100` Accuracy and `+0.0033` Macro-F1 using the exact target values stored
in the freeze manifest. This is not yet an exact paper-to-paper comparison:
GraphCURE uses the leakage-controlled, cross-split-deduplicated test set
(`n=2434`), whereas published MOCHEG tables use the official test set
(`n=2442`). The final paper must report both an official-split comparability
track and this strict robustness track.

The original cached GraphCURE-R2V verifier does **not** exceed AMuFC. The
subsequent frozen Qwen3 raw ensemble reaches `0.569022/0.545806` on the strict
split and `0.567977/0.545309` on the official split. Relative to the arXiv-v2
AMuFC row above, the official point-estimate gains are `+0.021977` Accuracy and
`+0.005309` Macro-F1. A stronger AMuFC workshop version reports
`0.5577/0.5560`; against that version, GraphCURE gains `+0.010277` Accuracy but
loses `-0.010691` Macro-F1. The defensible conclusion is therefore an official
accuracy improvement, but **not a new strongest-reported Macro-F1 SOTA**.

## Diagnostic interpretation

- Hybrid Qwen3 retrieval reaches test Recall@50 `0.954807` and MRR `0.807668`.
- Qwen3 reranking raises Recall@1 from `0.742810` to `0.822104`, Recall@10
  from `0.925637` to `0.945357`, and MRR to `0.869985`.
- The verifier selects a gold text candidate at rank 1 with probability
  `0.8204 +/- 0.0307`, conditional on gold evidence being present.
- Text retrieval is therefore no longer the main bottleneck. The next matched
  development stage must add system-retrieved visual evidence and learn when it
  is useful, while keeping all model selection on validation.

## Non-comparable protocols

- Gold evidence is an oracle diagnostic and must be placed in a separate table.
- DEFAME performs dynamic open-web search and belongs to the P2/open-web table.
- M-RAV reports stronger LLM results under its own gold/system-evidence setup;
  those numbers should be quoted with its exact evidence and metric definitions,
  not merged blindly into the P1 table.
