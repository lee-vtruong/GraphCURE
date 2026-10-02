# B23: Rationale-Teacher Family Ablation

## Question

Does the rationale-trained student's behavior depend on the Qwen2.5 teacher
family, rather than only on teacher capacity?  B23 holds the student,
explanation-loss recipe, data, retrieved passages, teacher prompt, seeds, and
raw/strict evaluation protocols fixed.  It changes only the teacher family:

- candidate: `mistralai/Mistral-7B-Instruct-v0.3`;
- reference: frozen `Qwen/Qwen2.5-7B-Instruct` B18A artifacts;
- student: `Qwen/Qwen3-4B-Instruct-2507` with `lambda_exp=.25`, 3 epochs,
  batch size 2, gradient accumulation 4, learning rate `2e-4`;
- seeds: `42, 87, 100` with equal probability averaging.

The experiment neither selects a teacher, seed, checkpoint, ensemble member,
nor policy using raw/strict test labels.  It is a teacher-family ablation of
the rationale-trained branch, not an AND-routing retune.

## Run

```bash
cd ~/whale/GraphCURE
git pull --ff-only
source .venv/bin/activate

tmux new-session -d -s mocheg-b23 \
  'cd ~/whale/GraphCURE && source .venv/bin/activate && \
   bash scripts/run_mocheg_b23_teacher_family.sh \
   2>&1 | tee outputs/mocheg_b23_teacher_family/run.log'
```

The default Mistral teacher has a Hugging Face chat template and is a
different model family from Qwen.  If the server receives a Hugging Face access
error, authenticate first with an account that has accepted the model's terms:

```bash
huggingface-cli login
```

To substitute a different instruction teacher deliberately, set all three
identifiers before starting a fresh run directory.  Do not overwrite the
default artifacts:

```bash
MOCHEG_ALT_TEACHER_MODEL='meta-llama/Llama-3.1-8B-Instruct' \
MOCHEG_ALT_TEACHER_NAME='Llama-3.1-8B-Instruct' \
MOCHEG_ALT_TEACHER_TAG='llama31_8b' \
bash scripts/run_mocheg_b23_teacher_family.sh
```

## Runtime and monitoring

The job has one sequential GPU stream: generate 9,304 training rationales,
train three students, infer each student on raw P1 and strict P1, and finally
run CPU-only paired analysis.  Teacher generation is typically the dominant
stage; the runner is restart-safe and skips a completed artifact.

```bash
tail -f outputs/mocheg_b23_teacher_family/generate_mistral7b.log

for seed in 42 87 100; do
  test -s "outputs/mocheg_b23_teacher_family/candidate_mistral7b_seed${seed}/summary.json" \
    && echo "TRAIN DONE: $seed" || echo "TRAIN RUNNING/PENDING: $seed"
done

test -s outputs/mocheg_b23_teacher_family/canonical_comparison.md \
  && cat outputs/mocheg_b23_teacher_family/canonical_comparison.md \
  || echo 'Comparison pending'
```

## Interpretation

Report raw and strict tracks separately.  The decisive comparisons are the
three-seed Mistral-teacher rationale-trained ensemble minus the frozen
three-seed Qwen2.5-7B-teacher rationale-trained ensemble, with paired
bootstrap interval, directional bootstrap probability, and exact McNemar
test.  A positive point estimate alone does not establish a model-family
effect; intervals and both tracks must be reported.
