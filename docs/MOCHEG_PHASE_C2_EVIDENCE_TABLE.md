# Phase C2: auditable open-web evidence constraints

## Frozen C1 result

The validation snapshot contains 1,456 claims and 13,377 evidence rows. It
used 1,481 search calls (1.017 per claim), expanded 25 claims, retained a
94.44% usable-evidence rate and a median of eight domains per claim. It used no
benchmark gold evidence and did not access test.

C1 is frozen before transformation. This freeze is intentionally scoped to
retrieval and **does not unlock test**. Phase C can unlock test only after the
complete verifier and all selection rules have been frozen on validation.

```bash
python -m scripts.freeze_mocheg_open_web_snapshot \
  --snapshot-root outputs/mocheg_c1_serper_adaptive100_2026_09_14 \
  --split val \
  --api-key-env SERPER_API_KEY
```

## C2a: observable constraint table

C2a is deterministic and performs no network or model inference. For every
claim/evidence pair it records:

- canonical URL, domain, HTTPS and coarse source family;
- query view and per-view rank;
- fetched-text status and observable text quality;
- heuristic proper-name candidates and claim/evidence entity overlap;
- temporal mentions and literal temporal overlap;
- explicit `unscored` stance and sufficiency fields.

The source family is descriptive, not a credibility label. Entity extraction
is a deterministic candidate extractor, not a trained NER system. Temporal
overlap is not temporal entailment. These conservative names prevent proxy
features from being reported as semantic judgments.

```bash
python -m scripts.build_mocheg_open_evidence_table \
  --snapshot-root outputs/mocheg_c1_serper_adaptive100_2026_09_14 \
  --output-root data/processed/mocheg_open_web_c2a \
  --split val \
  2>&1 | tee outputs/mocheg-c2a-build.log

cat data/processed/mocheg_open_web_c2a/summary.json
```

The builder verifies the frozen row hash before reading data, excludes labels
from its output, and refuses a development freeze containing test or gold
evidence.

## Gate before C2b

- C1 freeze has no failures and `unlocks_test=false`;
- C2a output has exactly 1,456 unique claim IDs;
- input row hash matches the C1 freeze;
- minimum usable evidence remains at least five per claim;
- source-family and overlap distributions are plausible under manual audit;
- stance and sufficiency remain unscored.

C2b will then run a frozen claim-evidence judge for stance, sufficiency,
entity consistency and temporal consistency. C2b must include a direct-verdict
control and will be selected only on validation; it must not silently treat
the C2a heuristics as ground-truth constraint labels.
