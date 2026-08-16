# GraphCURE evaluation protocols

This file freezes the four-stage research program.

1. **A — Protocol and leakage audit.** Materialize ID-aligned close,
   retrieved-evidence, and gold-evidence manifests. Gold evidence is never
   called closed-book.
2. **B — Closed expert.** Train without external/qrel evidence and evaluate on
   NewsCLIPpings plus a real-world, modality-balanced external benchmark.
3. **C — Open expert.** Train and evaluate with system-retrieved evidence;
   report gold evidence only as an oracle diagnostic.
4. **D — Cost-aware router.** Freeze the two experts and route by expected
   verification gain minus measured retrieval/inference cost.

## MOCHEG protocol definitions

| Protocol | Model-visible inputs | Valid use |
|---|---|---|
| `close` | claim and a claim-owned image, if the release identifies one | closed baseline |
| `open_retrieved` | close inputs plus system-retrieved evidence | deployable open system |
| `open_gold_oracle` | close inputs plus qrel text/image evidence | oracle diagnostic only |

The current strict MOCHEG manifest stores qrel-derived `image_paths`. Those
paths are evidence images, not automatically claim-owned images. Therefore the
protocol builder removes them from `close`. If `close_claim_images` is zero in
the audit, MOCHEG close-book is text-only under the currently available schema.

Generate and audit the frozen manifests:

```bash
python -m scripts.prepare_mocheg_protocols \
  --manifest-root data/processed/mocheg_manifest_strict \
  --retrieval-root outputs/retrieval_mocheg_dense_top50 \
  --raw-root data/raw/mocheg_dataset/extracted/mocheg \
  --output-root data/processed/mocheg_protocols
```

The command exits non-zero on ID, label, retrieval, or closed-evidence leakage.
Its authoritative report is
`data/processed/mocheg_protocols/protocol_audit.json`.
