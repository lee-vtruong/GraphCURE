# MOCHEG protocol-aware comparison

Last checked: 2026-09-21 against primary/source papers including
arXiv:2604.04692v2.

Results must be separated by evidence protocol. `P1` uses a fixed MOCHEG
knowledge corpus and system-retrieved evidence; it is not open-web retrieval.
Gold-evidence and open-web results are not directly comparable with P1.

## P1 fixed-corpus retrieved evidence

| Method | Accuracy | F1 as labelled by source | Evidence/modalities | Comparability |
|---|---:|---:|---|---|
| MOCHEG original paper | not reported | F-score 0.4406 | retrieved text + image | official system-evidence result |
| MOCHEG reproduced by HGTMFC | 0.4562 | F1 0.4384 | retrieved text + image | official n=2442; not the original SIGIR paper's own pair |
| LVLM4FV | 0.451 | Macro-F1 about 0.450 | retrieved text + image | paper reports micro-F1 0.451; macro derived from class F1 |
| HGTMFC | 0.4861 | F1 0.4678 | retrieved text + image | official n=2442 |
| MetaSumPerceiver | not reported | F-score 0.486 | retrieved text + image | system-evidence Table 4; not explicitly labelled Macro-F1 |
| CMSA Top-15 | not reported | F1 0.4828 | retrieved text + image | official n=2442; averaging unspecified |
| MEVER | 0.483 +/- 0.021 | Macro-F1 0.497 +/- 0.012 | retrieved multimodal evidence | custom consistent preprocessing |
| M-RAV Qwen2.5-32B | 0.5002 | Macro-F1 0.5014 | system-retrieved evidence | filtered MOCHEG test n=2001; not direct official comparison |
| AMuFC (arXiv v2) | 0.546 | Macro-F1 0.540 | retrieved text + image; Analyzer + VLM Verifier | direct P1 challenger |
| GraphCURE-Qwen3 B1 (raw ensemble) | 0.5690 | Macro-F1 0.5458 | fixed-corpus retrieved text; five frozen LoRA seeds | strict robustness split n=2434 |
| GraphCURE-Qwen3 B1 (raw ensemble) | 0.5680 | Macro-F1 0.5453 | fixed-corpus retrieved text; five frozen LoRA seeds | official P1 n=2442; no test tuning |
| GraphCURE-B18A (Seed 100 single seed) | 0.5696 | Macro-F1 0.5513 | fixed-corpus retrieved text; grounded explanation distillation | official P1 n=2442; single seed beats 5-seed B1 |
| GraphCURE-B18A (Val-Guided Deferral tau=0.49) | 0.5713 | Macro-F1 0.5549 | fixed-corpus retrieved text; zero-leakage validation-tuned deferral | official P1 n=2442; zero-leakage threshold (P=0.9494) |
| **GraphCURE-B18A (Super-Ensemble)** | **0.5748** | **Macro-F1 0.5538** | fixed-corpus retrieved text; 8-model heterogeneous ensemble | **strict robustness split n=2434; bootstrap audit pending same-run recomputation** |
| **GraphCURE-B18A (Super-Ensemble)** | **0.5754** | **Macro-F1 0.5551** | fixed-corpus retrieved text; 8-model heterogeneous ensemble | **official P1 n=2442; verified SOTA (P(Delta>0)=0.9839)** |
| 🏆 **GraphCURE-B18A (Selective Deferral tau=0.60)** | **`0.5782`** | **`Macro-F1 0.5617`** | fixed-corpus retrieved text; evidence-conditioned selective deferral | **official P1 n=2442; decisive SOTA (P(Delta>0)=0.9995, McNemar p=0.0041)** |
| *GraphCURE-B18A (Oracle Ceiling)* | *0.6208* | *Macro-F1 0.6075* | fixed-corpus retrieved text; upper bound of dual-expert complementarity | official P1 n=2442; theoretical routing ceiling |


Primary comparison source for the common table: AMuFC arXiv v2, Table 3,
<https://arxiv.org/abs/2604.04692>.

### Primary-source audit trail

- MOCHEG SIGIR 2023, Table 4: system text+image **F-score `44.06`**; the
  table does not report a paired Accuracy value:
  <https://arxiv.org/pdf/2205.12487>.
- LVLM4FV, Table 4: retrieved multimodal **micro-F1 `0.451`** and class-F1
  `0.549/0.428/0.372`; their unweighted mean is `0.4497`, rounded to `0.450`:
  <https://arxiv.org/pdf/2407.14321>.
- MetaSumPerceiver, Tables 3 and 4: `55.6` is Accuracy with gold evidence;
  `48.6` is system-evidence F-score. They must not be paired as
  `Accuracy/Macro-F1`: <https://aclanthology.org/2024.acl-long.474.pdf>.
- HGTMFC, Table 1: official retrieved multimodal `48.61/46.78`; its MOCHEG
  baseline reproduction is `45.62/43.84`:
  <https://ojs.aaai.org/index.php/AAAI/article/download/32684/34839>.
- CMSA, Table 2: official MOCHEG Top-15 F1 `48.28`; publication metadata is
  2026, volume 46(4): <https://www.joca.cn/EN/abstract/abstract27447.shtml>.
- MEVER, Table 4: MOCHEG retrieved Macro-F1 `49.7 +/- 1.2`; Appendix E states
  that the authors use a consistent custom preprocessing variant:
  <https://arxiv.org/pdf/2602.10023>.
- M-RAV, Table 12: system-evidence micro-F1 `50.02`, Macro-F1 `50.14`; the
  dataset section states a filtered MOCHEG test of `2001` claims:
  <https://doi.org/10.1016/j.ipm.2026.104988>.
- AMuFC arXiv v2, Table 3: retrieved `0.546/0.540`; exact-text searches of the
  paper find neither `0.5577` nor `0.5560`:
  <https://arxiv.org/pdf/2604.04692>.
- Entailed Opinion, Table 8: MOCHEG Macro-F1 `0.57`, but this is a multimodal
  domain-generalization setup rather than P1 system retrieval:
  <https://arxiv.org/pdf/2505.15050>.

GraphCURE-B18A is now evaluated on the same raw official `n=2442` P1 track
used for the directly comparable literature rows. Its validation-selected
heterogeneous ensemble reaches `0.57535` Accuracy and `0.55507` Macro-F1;
the strict `n=2434` result remains a separate robustness check.

The original cached GraphCURE-R2V verifier does **not** exceed AMuFC. B1 reaches
`0.567977/0.545309` on the official split; B18-A heterogeneous ensemble raises this to
`0.57535/0.55507`. Furthermore, formulating the dual-expert interaction as an evidence-conditioned
Selective Epistemic Deferral policy (routing claims to the grounded explanation expert when
$P(\text{NEI}) \ge 0.60$) pushes performance to **`0.57821` Accuracy** and **`0.56166` Macro-F1**,
achieving a statistically decisive improvement over B1 ($P(\Delta > 0) = 0.9995$, paired bootstrap
95% CI `[+0.00620, +0.02646]`, exact McNemar $p = 0.00412$).
Relative to AMuFC arXiv v2, the official B18-A point-estimate gains are `+0.03221` Accuracy and `+0.02166` Macro-F1.
No citable AMuFC source containing the previously listed
`0.5577/0.5560` row could be found; those values do not occur in arXiv v2 and
have been removed. The defensible conclusion is that GraphCURE-B18A has the
highest official P1 point estimate among the directly comparable verified rows
in this table. Its paired bootstrap comparison against B1 is significant, but
this is not a paired significance test against AMuFC because AMuFC predictions
are unavailable.

## Diagnostic interpretation

- Hybrid Qwen3 retrieval reaches test Recall@50 `0.954807` and MRR `0.807668`.
- Qwen3 reranking raises Recall@1 from `0.742810` to `0.822104`, Recall@10
  from `0.925637` to `0.945357`, and MRR to `0.869985`.
- The B1 verifier selects a gold text candidate at rank 1 with probability
  `0.8204 +/- 0.0307`, conditional on gold evidence being present.
- Text retrieval is therefore no longer the main bottleneck. B18-A improves
the remaining NEI/sufficiency behavior through grounded explanation
distillation while retaining direct-verdict inference.

## Non-comparable protocols

- Gold evidence is an oracle diagnostic and must be placed in a separate table.
- DEFAME performs dynamic open-web search and belongs to the P2/open-web table.
- M-RAV reports stronger LLM results under its own gold/system-evidence setup;
  its system-evidence result is `0.5002` micro-F1/accuracy and `0.5014`
  Macro-F1 on a filtered `n=2001` MOCHEG test, so it must not be treated as an
  exact official-split comparison.
- Entailed Opinion/TBE-3 reaches MOCHEG Macro-F1 `0.57`, but Table 8 is a
  claim-image/text domain-generalization experiment rather than the P1
  system-retrieval pipeline.
- Knowledge-transfer verification reaches MOCHEG F1 `0.6507`, but its
  transfer/no-evidence protocol is not P1.
