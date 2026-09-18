# MOCHEG Phase B18-A: Evidence-Grounded Explanation Distillation

## 1. Research Motivation & Hypothesis

B6-A previously revealed that auxiliary multi-task reasoning signals could provide meaningful positive transfer (+0.0099 Macro-F1 over a compute-matched direct control), but binary sufficiency and conditional polarity were too coarse to prevent NEI collapse. Furthermore, B16 demonstrated that modifying inference format or token decoding introduces confounding generation noise.

**Phase B18-A tests a central scientific question:**
> *Does supervising the student representations with teacher-distilled, structured, grounded explanations during training improve direct verdict accuracy and Macro-F1 beyond matched compute alone?*

### Key Design Principles:
1. **Unchanged B1 Retrieval & Inference Architecture:** No new sentence selectors or span attention layers are added yet. Model inputs and inference latency remain identical to B1.
2. **Dual-Prompt Separation:**
   - **Prompt A (Verdict Token):** `(Claim, Evidence) -> A, B, or C` (used for training & inference).
   - **Prompt B (Explanation Generation):** `(Claim, Evidence) -> Structured JSON` (used **strictly during training**).
3. **No Gold Leakage to Teacher:** The teacher model is ONLY allowed to inspect P1 system-retrieved evidence. Gold qrel injection is forbidden.
4. **Groundedness Safeguard:** If the retrieved evidence is insufficient to justify the claim's true verdict, the teacher outputs `grounded: false`. In this case, **no explanation loss is applied**, preventing the model from hallucinating rationales for ungrounded claims.

---

## 2. Structured Explanation Schema

The teacher outputs a strictly validated JSON object:

```json
{
  "grounded": true,
  "verdict": "SUPPORTED",
  "key_evidence_ids": [1, 4],
  "reason": "Evidence [1] establishes X, while Evidence [4] confirms Y.",
  "missing_information": null
}
```

For Not Enough Information (NEI):
```json
{
  "grounded": true,
  "verdict": "NEI",
  "key_evidence_ids": [2],
  "reason": "Evidence [2] mentions the event but does not confirm that the subject participated.",
  "missing_information": "Official participant roster or primary confirmation."
}
```

For Ungrounded / Insufficient Retrieved Evidence:
```json
{
  "grounded": false,
  "verdict": "NEI",
  "key_evidence_ids": [],
  "reason": "Retrieved evidence is completely off-topic.",
  "missing_information": "Relevant primary documentation."
}
```

### Automated Validation Criteria:
* `verdict` must be one of `{"SUPPORTED", "REFUTED", "NEI"}`.
* `key_evidence_ids` must be valid integers referencing 1-based evidence indices present in the prompt.
* If `grounded == true` and `verdict in {"SUPPORTED", "REFUTED"}`: $\ge 1$ key evidence ID required.
* If `verdict == "NEI"` and `grounded == true`: `missing_information` must be a non-empty string.
* Quoted text in `reason` must not hallucinate phrases absent from the cited passages.

---

## 3. Matched Compute Control Design

To isolate the causal effect of explanation supervision from additional training updates:

$$\text{Primary Comparison} = \mathbf{\text{Explanation Candidate}} - \mathbf{\text{Matched Direct Control}}$$

| Model Condition | Initialization | Tasks Trained | Loss Objective | Optimizer Updates |
|---|---|---|---|---|
| **Frozen Anchor** | Qwen3-4B base + B1 LoRA | None | None | 0 |
| **Matched Direct Control** | Qwen3-4B base | Prompt A only | $L_{\text{verdict}}$ | $N$ steps |
| **Explanation Candidate** | Qwen3-4B base | Prompt A + Prompt B | $L_{\text{verdict}} + 0.25 \cdot L_{\text{explanation}}$ | $N$ steps |

---

## 4. Locked Protocol & Promotion Gate

* **Data Splits:** Train-only folds using seed `2040` (`data/processed/mocheg_b18_folds.json`). Fold 0 is used for the screening stage. Official validation and test splits remain locked.
* **Inference:** Direct verdict token ($A, B, C$) evaluation only.

### Preregistered Fold-0 Promotion Gate:
1. $\Delta \text{Macro-F1} \ge +0.005$ over Matched Direct Control.
2. Paired bootstrap probability $P(\Delta > 0) \ge 0.95$ (10,000 resamples).
3. Supported F1 $\ge -0.005$ relative to control.
4. Accuracy $\ge -0.002$ relative to control.
5. Helpful corrections strictly exceed harmful regressions ($\text{Helpful} > \text{Harmful}$).

*Only if Fold 0 clears all gate criteria will the model be promoted to 5-fold confirmation.*

---

## 5. Roadmap Progression (If B18-A Passes)

```
B18-A: Grounded explanation distillation (Multitask representation learning)
  │
  ▼ (Pass gate)
B18-B: Explanation-derived adaptive top-k sentence selector
  │
  ▼
B18-C: Fine-grained sentence -> phrase/span rationale grounding
  │
  ▼
B18-D: Error-aware rationale refinement
```

---

## 6. GPU Server Runbook

Execute the following commands on the GPU server:

```bash
cd ~/whale/GraphCURE
git fetch origin
git checkout feature/mocheg-phase-b18-explanation-distillation
git pull --ff-only origin feature/mocheg-phase-b18-explanation-distillation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/whale/GraphCURE/.venv

export HF_HOME=~/whale/cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

set -euo pipefail

# 1. Build fresh B18 duplicate-safe train-only folds
python -m scripts.prepare_mocheg_b18_folds \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --output data/processed/mocheg_b18_folds.json \
  --seed 2040

# 2. Generate teacher explanations for Fold 0 train split
mkdir -p outputs/mocheg_b18_explanations
CUDA_VISIBLE_DEVICES=0 python -m scripts.generate_mocheg_b18_teacher_explanations \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --retrieval outputs/retrieval_mocheg_dense_top50/train.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/evidence.csv \
  --folds data/processed/mocheg_b18_folds.json \
  --fold 0 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output data/processed/mocheg_b18_explanations/train_fold0_explanations.jsonl \
  --summary outputs/mocheg_b18_explanations/summary_fold0.json \
  --device cuda \
  2>&1 | tee outputs/mocheg_b18_explanations/generate_fold0.log

# 3. Train Matched Direct Control on Fold 0
mkdir -p outputs/mocheg_b18/control_fold0
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_b18_explanation_verifier \
  --mode matched_control \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --retrieval outputs/retrieval_mocheg_dense_top50/train.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/evidence.csv \
  --folds data/processed/mocheg_b18_folds.json \
  --fold 0 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output outputs/mocheg_b18/control_fold0 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4 \
  --device cuda \
  2>&1 | tee outputs/mocheg_b18/control_fold0/train.log

# 4. Train Explanation Candidate on Fold 0
mkdir -p outputs/mocheg_b18/candidate_fold0
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_b18_explanation_verifier \
  --mode explanation_candidate \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --retrieval outputs/retrieval_mocheg_dense_top50/train.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/evidence.csv \
  --folds data/processed/mocheg_b18_folds.json \
  --fold 0 \
  --explanations data/processed/mocheg_b18_explanations/train_fold0_explanations.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output outputs/mocheg_b18/candidate_fold0 \
  --lambda-exp 0.25 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4 \
  --device cuda \
  2>&1 | tee outputs/mocheg_b18/candidate_fold0/train.log

# 5. Run Paired Statistical Analysis & Gate Audit
python -m scripts.analyze_mocheg_b18_explanation_verifier \
  --candidate outputs/mocheg_b18/candidate_fold0/val_predictions.jsonl \
  --control outputs/mocheg_b18/control_fold0/val_predictions.jsonl \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --output outputs/mocheg_b18/analysis_fold0.json \
  --markdown outputs/mocheg_b18/analysis_fold0.md \
  2>&1 | tee outputs/mocheg_b18/analysis_fold0.log
```
