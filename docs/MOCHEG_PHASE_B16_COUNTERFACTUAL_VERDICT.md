# MOCHEG B16: counterfactual evidence-omission verdict curriculum

## Evidence-based hypothesis

B14 showed that the B13 expert improves supported/refuted but damages NEI.
B15 showed that this damage cannot be routed reliably from current confidence
features (utility AUROC `0.5660`). The largest failure is evidence absence, but
qrel availability is not observable at inference. B16 therefore changes the
verifier rather than the router.

For a supported/refuted training claim with injected gold evidence, B16 creates
a paired prompt after removing labelled gold documents. This counterfactual
prompt is trained directly on the ordinary verdict token `C` (NEI). It does not
use a separate sufficiency head at inference. In epoch 1, 15% of ordinary
verdict rows are replaced with these omission rows. Epochs 2 and 3 are
verdict-only recovery. Every epoch has exactly the same example and optimizer
update budget as the matched direct control.

## Locked protocol

- New duplicate-safe train-only folds: seed `2039`.
- Development: fold 0 only.
- Initialization: Qwen3 base plus fresh LoRA.
- Fixed checkpoint: epoch 3.
- Counterfactual fraction: `0.15`, epoch 1 only.
- Counterfactual target: direct verdict `C` (NEI).
- Hierarchical inference weight: `0`.
- Controls: standard article anchor and same-script matched direct control.
- Official validation and test: forbidden.

## Promotion gate

- at least `+0.005` Macro-F1 over the standard anchor;
- at least `+0.003` Macro-F1 over the matched direct control;
- NEI F1 gain at least `+0.010` over the anchor;
- supported F1 no worse than `-0.005`;
- accuracy no worse than `-0.002`;
- paired-bootstrap positive probability at least `0.95` versus both controls;
- helpful changes exceed harmful changes;
- every source remains within `-0.002` Macro-F1 of the anchor.

Only a passing fold-0 run may be confirmed on folds 1--4.
