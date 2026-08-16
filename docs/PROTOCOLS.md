# GraphCURE evaluation protocols

This file freezes the four-stage research program.

1. **A — Protocol and leakage audit.** Materialize ID-aligned close,
   retrieved-evidence, and gold-evidence manifests. Gold evidence is never
   called closed-book.
2. **B — Closed-corpus expert.** Retrieval from the frozen benchmark corpus is
   allowed, but live Internet/search-engine access is forbidden. A no-retrieval
   model remains a P0 diagnostic rather than the main Phase-B system.
3. **C — Open-web expert.** Live search, iterative query reformulation, and
   newly retrieved external evidence are allowed; gold benchmark evidence is
   still only an oracle diagnostic.
4. **D — Cost-aware router.** Freeze the two experts and route by expected
   verification gain minus measured retrieval/inference cost.

## MOCHEG protocol definitions

| Protocol | Model-visible inputs | Valid use |
|---|---|---|
| `close` | claim and a claim-owned image, if the release identifies one | P0 no-retrieval diagnostic |
| `open_retrieved` | close inputs plus system-retrieved evidence from the frozen local corpus | P1 closed-corpus expert (legacy artifact name) |
| `open_gold_oracle` | close inputs plus qrel text/image evidence | oracle diagnostic only |

The current strict MOCHEG manifest stores qrel-derived `image_paths`. Those
paths are evidence images, not automatically claim-owned images. Therefore the
protocol builder removes them from `close`. If `close_claim_images` is zero in
the audit, the P0 diagnostic is text-only under the currently available schema.
It must not be compared directly with papers that use system-retrieved evidence.

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

## Matched protocol embeddings

Use the same MPNet text backbone in all three protocols. Close and
text-retrieved settings skip CLIP entirely because the audited schema exposes
no claim-owned image; gold-oracle alone may encode qrel images.

```bash
python -m scripts.embed_mocheg \
  --manifest-root data/processed/mocheg_protocols/close \
  --output-root data/processed/mocheg_protocol_embeddings/close \
  --text-model sentence-transformers/all-mpnet-base-v2 \
  --skip-images --device cuda --batch-size 64

python -m scripts.embed_mocheg \
  --manifest-root data/processed/mocheg_protocols/open_retrieved \
  --output-root data/processed/mocheg_protocol_embeddings/open_retrieved \
  --text-model sentence-transformers/all-mpnet-base-v2 \
  --skip-images --device cuda --batch-size 64

python -m scripts.embed_mocheg \
  --manifest-root data/processed/mocheg_protocols/open_gold_oracle \
  --output-root data/processed/mocheg_protocol_embeddings/open_gold_oracle \
  --text-model sentence-transformers/all-mpnet-base-v2 \
  --device cuda --batch-size 64
```

Verify alignment before any matched experiment:

```bash
python -m scripts.verify_mocheg_protocol_embeddings \
  --close-root data/processed/mocheg_protocol_embeddings/close \
  --retrieved-root data/processed/mocheg_protocol_embeddings/open_retrieved \
  --gold-root data/processed/mocheg_protocol_embeddings/open_gold_oracle
```
