# MOCHEG B17: B16 confirmation failure atlas

## Why B17 is diagnostic-only

B16 passed fresh fold 0 but failed independent folds 1--4. Its aggregate
Macro-F1 was `0.665174`, versus `0.660505` for the standard anchor and
`0.666210` for the compute-matched direct control. The candidate therefore
gained `+0.004669` over the anchor but lost `-0.001036` to its causal control.
NEI F1 also fell `-0.002017` relative to the anchor. This does not confirm the
registered evidence-omission mechanism.

B17 fits no model, selects no threshold and changes no hyperparameter. It
audits the frozen held predictions from confirmation folds 1--4 with the
matched control as the primary baseline. Fold 0, official validation and test
remain excluded.

## Questions

1. Which label transitions account for B16 harm relative to the matched
   control?
2. Is harm concentrated by fold, source, qrel availability, retrieval rank,
   confidence or counterfactual eligibility?
3. Did every fold receive the locked `0.15` exposure and fixed epoch budget?
4. Is any apparent gain a counterfactual effect, or only a longer/fresh
   training-trajectory effect already captured by the matched control?

## Run

```bash
python -m scripts.analyze_mocheg_b17_b16_failure_atlas \
  --root outputs/mocheg_b16_fresh \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --retrieval outputs/retrieval_mocheg_qwen3_reranked/train.jsonl \
  --targets data/processed/mocheg_b6_targets_natural/train.jsonl \
  --fold-spec data/processed/mocheg_b16_folds.json \
  --output outputs/mocheg_b17_b16_failure_atlas.json \
  --markdown outputs/mocheg_b17_b16_failure_atlas.md \
  --cases outputs/mocheg_b17_b16_failure_cases.jsonl \
  2>&1 | tee outputs/mocheg-b17-atlas.log
```

Inspect the primary result and worst slices:

```bash
python - <<'PY'
import json

s = json.load(open("outputs/mocheg_b17_b16_failure_atlas.json"))
print("METRICS")
print(json.dumps(s["metrics"], indent=2))
print("\nB16 VS MATCHED CONTROL")
print(json.dumps(s["comparisons"]["candidate_vs_matched_control"], indent=2))
print("\nCLASS DELTAS VS CONTROL")
print(json.dumps(s["class_f1_delta_candidate_vs_control"], indent=2))
print("\nPREDICTION SHIFT")
print(json.dumps(s["prediction_shift"], indent=2))
print("\nTOP TRANSITIONS")
for row in s["transitions_vs_control"][:20]:
    print(row)
print("\nWORST GROUPS")
for row in s["worst_candidate_vs_control_groups"][:20]:
    print(row)
print("\nHIGHEST HARM EXCESS")
for row in s["highest_harm_excess_groups"][:20]:
    print(row)
print("\nDIAGNOSIS")
print(json.dumps(s["diagnosis"], indent=2))
print("\nVALIDATION USED:", s["official_validation_used"])
print("TEST USED:", s["test_split_used"])
PY
```

Inspect the most harmful claim-level cases:

```bash
python - <<'PY'
import json

shown = 0
for line in open("outputs/mocheg_b17_b16_failure_cases.jsonl"):
    row = json.loads(line)
    if row["candidate_vs_control"] != "harmful":
        continue
    print("\nID:", row["id"], "fold:", row["fold"], "source:", row["source"])
    print("gold:", row["gold"], "predictions:", row["predictions"])
    print("eligible:", row["counterfactual_eligible"],
          "retrieval:", row["retrieval_status"])
    print("claim:", row["claim"])
    shown += 1
    if shown == 20:
        break
PY
```

## Decision rule

- B16 remains closed regardless of the atlas.
- Do not search omission fractions or thresholds on folds 1--4.
- Any next intervention must state a mechanism supported by B17, use the
  matched control as the main baseline, and receive a new duplicate-safe fold
  assignment.
- Official validation and test remain locked.

## Frozen outcome (2026-09-14)

B17 confirms that the B16 gain over the weaker standard anchor was not a
counterfactual-treatment effect.  Across confirmation folds 1--4, the B16
candidate reached Macro-F1 `0.665174`, below the compute-matched direct
control at `0.666210` (`-0.001036`).  It made `550` helpful and `561` harmful
corrections.  Class-F1 changed by `+0.001403` for supported, `-0.000287` for
refuted and `-0.004224` for NEI relative to the matched control.

The largest stable harm slices were low retrieval confidence/margin q1
(`-0.011932` Macro-F1), control confidence `0.70--0.90` (`-0.010598`) and
gold evidence ranked 2--5 (`-0.009919`).  Counterfactual-ineligible examples
also regressed (`-0.002519`).  The effect was slightly negative for both
Politifact (`-0.000567`) and Snopes (`-0.000462`), so there is no
source-specific recovery that could justify promotion.

The transition audit exposes a nearly symmetric and unstable boundary shift:
`176` supported examples moved helpfully from NEI to supported, but `167`
correct supported examples moved harmfully to NEI; similarly, `142` NEI
examples were recovered from supported while `152` correct NEI examples were
lost to supported.  This is not evidence that the model learned a reliable
evidence-sufficiency intervention.

**Decision:** B16 and B17 are closed.  No omission-ratio, threshold or router
search is permitted on these folds.  B17 does not support a B18 training
hypothesis strong enough to justify more Phase-B compute.  The frozen Phase-B
expert remains the official Qwen3 five-seed P1 ensemble; work moves to the
separately evaluated Phase-C open-web expert.
