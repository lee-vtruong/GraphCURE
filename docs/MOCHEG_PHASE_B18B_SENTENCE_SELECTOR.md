# MOCHEG Phase B18-B: Explanation-Derived Adaptive Sentence & Passage Selector

## 1. Research Motivation & Scientific Rationale

### Empirical Diagnosis from Phase B18-A
In Phase B18-A (Grounded Explanation Distillation), we evaluated multi-task training where student models generated both structured rationales and fact verdicts:
- **Single-Seed Success:**
  - Seed 42: $+0.00878$ Macro-F1 over matched control ($0.65231$ vs $0.64353$).
  - Seed 87: $+0.00319$ Macro-F1 over matched control ($0.65497$ vs $0.65178$).
  - Mean Single-Seed Delta: $\mathbf{+0.00599 \pm 0.00280}$ (2/2 positive seeds).
- **Ensemble Diversity Collapse:**
  - In the matched control, errors across random seeds were largely independent, yielding a strong ensemble gain ($+2.3\% \to \mathbf{0.67081}$).
  - In candidate models, multi-task explanation token generation strongly coupled both seeds to identical teacher token sequences, collapsing inter-seed variance. As a result, the candidate ensemble reached only $0.65525$ ($\Delta = -0.01555$).
  - Furthermore, dense retrieval (top-5) routinely passed irrelevant distractor passages into the prompt, diluting verification precision (especially on Snopes claims).

### Phase B18-B Solution: Architectural Decoupling
To retain the benefits of teacher rationale distillation while preserving 100% of inter-seed ensemble diversity, Phase B18-B **decouples evidence selection from verdict classification**:
1. **Teacher Rationale Mining:** The 2,594 `grounded_true` explanations from B18-A provide fine-grained evidence attribution (`key_evidence_ids`). These serve as high-precision pseudo-labels:
   - Passages cited in `key_evidence_ids` $\to$ **Positive (Relevance = 1.0)**.
   - Unselected passages in retrieved top-5 $\to$ **Hard Negatives (Relevance = 0.0)**.
   - Passages from ungrounded claims $\to$ **Irrelevant / Distractor Negatives (Relevance = 0.0)**.
2. **Adaptive Passage Selector:** Train a lightweight Cross-Encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to score $(c, e)$ pairs.
3. **Adaptive Pruning Policy:** Filter the noisy top-5 retrieved passages down to 1–3 dense, clean passages based on score threshold and relative margin.
4. **Diversity-Preserving Direct Verdict Verifier:** Train the downstream Qwen3-4B LoRA verifier **solely** on direct verdict tokens ($A, B, C$) over the purified evidence. This eliminates distractors while preserving full random-seed ensemble diversity.

---

## 2. Mathematical Formulation

### 2.1 Selector Dataset Construction
Given a set of teacher explanations $\mathcal{D}_{\text{exp}}$ generated on Fold 0 training split:
- For grounded claim $(c_i, y_i)$ with retrieved evidence candidates $\mathcal{E}_i = \{e_{i,1}, \dots, e_{i,K}\}$ and key evidence index set $\mathcal{K}_i \subseteq \{1, \dots, K\}$:
  $$\forall j \in \mathcal{K}_i: \quad (c_i, e_{i,j}, 1.0)$$
  $$\forall j \in \{1, \dots, K\} \setminus \mathcal{K}_i: \quad (c_i, e_{i,j}, 0.0) \quad (\text{Hard Negatives})$$
- For ungrounded claims where teacher deemed retrieval insufficient ($\text{grounded} = \text{false}$):
  $$\forall j \in \{1, \dots, \min(K, 2)\}: \quad (c_i, e_{i,j}, 0.0)$$

### 2.2 Selector Training Objective
The selector $f_\theta(c, e) \in \mathbb{R}$ is trained with binary cross-entropy with logits:
$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{(c, e, y)} \left[ y \log \sigma(f_\theta(c, e)) + (1 - y) \log (1 - \sigma(f_\theta(c, e))) \right]$$

### 2.3 Adaptive Evidence Selection Policy
Given top candidates $\mathcal{E} = \{e_1, \dots, e_K\}$ sorted descending by score $s_1 \ge s_2 \ge \dots \ge s_K$:
1. Always retain rank 1: $\mathcal{E}^* = \{e_1\}$.
2. For rank $k \in \{2, \dots, K_{\max}\}$:
   Include $e_k$ in $\mathcal{E}^*$ if and only if:
   $$s_k \ge \tau \quad \text{AND} \quad (s_1 - s_k) \le \delta$$
   where $\tau$ is the absolute score threshold and $\delta$ is the maximum relative margin from the top passage.
3. Enforce $K_{\min} \le |\mathcal{E}^*| \le K_{\max}$ (default $K_{\min}=1, K_{\max}=3$).

---

## 3. Preregistered Promotion Gate (Fold 0 Held-Out Validation, $n=2326$)

To qualify for promotion and 5-seed confirmation:
1. **Ensemble Benchmark:** 2-Seed Candidate Ensemble Macro-F1 $\ge 0.67081$ (must match or exceed the 2-Seed Matched Control benchmark).
2. **Single-Seed Delta:** Mean $\Delta \text{Macro-F1} > 0$ over Matched Direct Control across seeds.
3. **Harm Mitigation:** Ensemble helpful corrections strictly exceed harmful regressions ($\text{Helpful} > \text{Harmful}$).
4. **Class Balance:** $\text{F1}_{\text{Supported}} \ge 0.60$ and $\text{F1}_{\text{Refuted}} \ge 0.82$.

---

## 4. GPU Server Runbook

Execute the following commands in sequence on the GPU server:

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

# 1. Train Adaptive Sentence/Passage Selector on Fold 0 Teacher Rationales
mkdir -p outputs/mocheg_b18b_selector
python -m scripts.train_mocheg_b18b_sentence_selector \
  --explanations data/processed/mocheg_b18_explanations/train_fold0_explanations.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  --output outputs/mocheg_b18b_selector \
  --base-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --epochs 3 \
  --batch-size 32 \
  --device cuda \
  2>&1 | tee outputs/mocheg_b18b_selector/train.log

# 2. Filter Retrieval Candidates using the Trained Selector
mkdir -p outputs/mocheg_b18b_filtered_retrieval
python -m scripts.prepare_mocheg_b18b_selected_evidence \
  --selector outputs/mocheg_b18b_selector \
  --retrieval outputs/retrieval_mocheg_dense_top50/train.jsonl \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  --output outputs/mocheg_b18b_filtered_retrieval/train.jsonl \
  --summary outputs/mocheg_b18b_filtered_retrieval/summary.json \
  --min-k 1 \
  --max-k 3 \
  --adaptive-margin 1.5 \
  --device cuda \
  2>&1 | tee outputs/mocheg_b18b_filtered_retrieval/filter.log

# 3. Train Candidate Verifier on Filtered Evidence (Seed 42)
mkdir -p outputs/mocheg_b18b/candidate_seed42
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_b18_explanation_verifier \
  --mode matched_control \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --retrieval outputs/mocheg_b18b_filtered_retrieval/train.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  --folds data/processed/mocheg_b18_folds.json \
  --fold 0 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output outputs/mocheg_b18b/candidate_seed42 \
  --seed 42 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4 \
  --device cuda \
  2>&1 | tee outputs/mocheg_b18b/candidate_seed42/train.log

# 4. Train Candidate Verifier on Filtered Evidence (Seed 87)
mkdir -p outputs/mocheg_b18b/candidate_seed87
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mocheg_b18_explanation_verifier \
  --mode matched_control \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --retrieval outputs/mocheg_b18b_filtered_retrieval/train.jsonl \
  --corpus data/raw/mocheg_dataset/extracted/mocheg/train/Corpus2.csv \
  --folds data/processed/mocheg_b18_folds.json \
  --fold 0 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output outputs/mocheg_b18b/candidate_seed87 \
  --seed 87 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4 \
  --device cuda \
  2>&1 | tee outputs/mocheg_b18b/candidate_seed87/train.log

# 5. Multi-Seed Ensemble & Statistical Significance Evaluation
python -m scripts.summarize_mocheg_b18_seeds \
  --candidate-roots outputs/mocheg_b18b/candidate_seed42 outputs/mocheg_b18b/candidate_seed87 \
  --control-roots outputs/mocheg_b18/control_seed42 outputs/mocheg_b18/control_seed87 \
  --manifest data/processed/mocheg_manifest_strict/train.jsonl \
  --output outputs/mocheg_b18b/summary_2seeds.json \
  --markdown outputs/mocheg_b18b/summary_2seeds.md \
  2>&1 | tee outputs/mocheg_b18b/summary_2seeds.log
```
