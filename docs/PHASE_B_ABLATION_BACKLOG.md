# Frozen Phase-B ablation backlog

Phase B is frozen at `configs/mocheg_phase_b_qwen3_frozen.json`.  The items
below complete reporting; they cannot replace the main system using an already
observed MOCHEG test result.

## Required before paper submission

- Dense-only, lexical-only, hybrid-RRF and hybrid-plus-reranker retrieval,
  reporting Recall@1/5/10/50, MRR, latency and index size.
- Verifier evidence sensitivity at top-1, top-3, frozen top-5 and top-10.
- Zero-shot, single LoRA, five-seed raw ensemble and calibrated ensemble.
- Train-time gold-candidate injection on/off with at least three fixed seeds.
- Inference latency, peak VRAM, tokens per claim and total training GPU-hours.
- Compact negative-ablation table covering visual fusion, hierarchical heads,
  auxiliary constraints, PCGrad, routing and B16 counterfactual omission.

## Generalization after Phase C/D is functional

- One size-comparable open-weight verifier backbone; proprietary models are
  baselines, not component ablations.
- MOCHEG-to-FIN-FACT and MOCHEG-to-WebFC zero-shot transfer where licensing and
  label mapping permit it.
- Keep AVerImaTeC, VERITE, ClaimReview2024+ and VeriTaS in the Phase-C/P2 table;
  do not mix their live-web protocol with the Phase-B/P1 leaderboard.

## Guardrails

- Register the exact matrix before running any new test inference.
- Select variants on train-only OOF or validation, never on test.
- Report official and strict MOCHEG tracks together.
- Gold evidence is oracle/train supervision and is never system evidence at
  validation or test.
