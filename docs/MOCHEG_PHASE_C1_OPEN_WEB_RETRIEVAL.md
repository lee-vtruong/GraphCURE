# Phase C1: immutable open-web evidence snapshots

Phase C uses protocol `P2_open_web`: live search is allowed, while benchmark
gold evidence remains forbidden as model input.  C1 creates a reproducible
snapshot before any open-web verifier is trained or selected.

For each claim, the deterministic policy creates semantic, entity, temporal
(when present), and contextual fact-check queries. Results are fused by RRF,
canonical URLs are deduplicated, and optional page extraction is cached. The
API key is read from an environment variable and is never serialized.

Every snapshot is immutable. Re-running the same command resumes missing
claims; changing provider or retrieval settings requires a new output root.
Private, loopback, link-local and non-HTTP(S) fetch targets are rejected.
Official test is locked unless a separate `P2_open_web` freeze manifest has
status `frozen_after_validation`.

## Smoke test with Brave Search

```bash
cd ~/whale/GraphCURE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv
git pull --ff-only origin main

export BRAVE_SEARCH_API_KEY="YOUR_KEY"

python -m scripts.run_mocheg_open_web_retrieval \
  --manifest-root data/processed/mocheg_manifest_strict \
  --output-root outputs/mocheg_c1_brave_2026_09_14_smoke \
  --provider brave \
  --splits val \
  --results-per-query 10 \
  --output-k 20 \
  --query-budget 2 \
  --fetch-pages \
  --fetch-top-k 5 \
  --fetch-workers 5 \
  --limit 20 \
  2>&1 | tee outputs/mocheg-c1-brave-smoke.log

cat outputs/mocheg_c1_brave_2026_09_14_smoke/summary.json
```

For Serper, use `--provider serper --api-key-env SERPER_API_KEY`. Do not put a
secret directly on the command line or in a tracked file.

## Inspect evidence quality

```bash
python - <<'PY'
import json

path = "outputs/mocheg_c1_brave_2026_09_14_smoke/val.jsonl"
for line in open(path):
    row = json.loads(line)
    print("\nCLAIM:", row["claim"])
    print("QUERIES:", [(x["constraint"], x["query"]) for x in row["queries"]])
    for evidence in row["evidence"][:3]:
        print("-", evidence["domain"], evidence["title"])
        print(" ", evidence["fetch_status"], evidence["text"][:300])
PY
```

## Promotion gate before a full validation snapshot

- all requested claims complete with no duplicate IDs;
- API keys absent from every cached artifact;
- at least 90% usable evidence (full text or a snippet of 80+ characters);
- page-fetch success and 401/403/429/timeout counts are reported separately;
- median at least three unique domains per claim;
- manual audit of 20 claims finds no gold/qrel leakage;
- snapshot ID/date, provider, query policy and hashes are recorded.

After this gate, C2 will assign provenance, temporal, entity, stance and
sufficiency fields to the frozen evidence rather than performing new searches.
