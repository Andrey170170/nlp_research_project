# Jayden — Direction 3 Task Brief: Principled Early Sparsification

## What I am asking you to do

Your task is to improve **early sparsification** for our exact chunked tracing pipeline.

Concretely, I want you to:

1. **design a better early-pruning method** than the current coarse top-`k` approach,
2. **implement it mainly in the forked library** at `../circuit-tracer_chunked`,
3. **evaluate it against exact-reference traces** that you generate yourself,
4. recommend a sparsification configuration that we can trust for the final project runs.

The goal is **not** just to make tracing faster. The goal is to make it faster **without changing the resulting trace too much**.

---

## Why this matters

Right now, Phases 0–2 of tracing finish relatively quickly even on long prompts, but they produce **millions of active features**. Phases 3–4 then have to process those candidates, and that is one of the main reasons exact tracing is so expensive.

We already have a prototype early-sparsification method in the fork, but it is too coarse. In particular:

- on long-prompt stress runs, it heavily favors **later layers**,
- the retained explanatory mass is still low,
- and it looks like the issue is **not** that earlier layers lack active features,
- but rather that the current scoring rule is a poor proxy for which features actually matter to the final exact trace.

So the real problem is:

> how do we decide, early in the pipeline, which active features are genuinely worth keeping for exact tracing later on?

---

## Short background / definitions

You do **not** need to understand every internal detail of the project before starting, but the following definitions matter for this task.

### What is “early sparsification” here?

In our pipeline, the model forward pass and sparse feature extraction happen before the expensive exact attribution phases.

For this task, “early sparsification” means:

> pruning the active feature set **before** the expensive later attribution stages, so that Phases 3–4 have fewer candidates to process.

### What are the tracing phases, at a high level?

- **Phase 0**: forward pass / setup / sparse encoding
- **Phases 1–2**: reconstruction-related setup and attribution preparation
- **Phase 3**: logit attribution
- **Phase 4**: feature attribution / exact graph expansion

The expensive part is mostly **Phases 3–4**.

### Why is this hard?

If you prune a feature early, it never gets a chance to appear in the final exact graph. So a bad pruning rule can make the trace cheaper, but also wrong.

---

## What I want you to optimize for

Please do **not** optimize only for:

- runtime,
- memory,
- retained activation mass,
- or raw candidate count.

Those matter, but they are not enough.

What I actually want is:

> a pruning method that removes many candidates early **while preserving the exact trace as faithfully as possible**.

So the task is to find a **better importance surrogate** than the current coarse top-`k` rule.

---

## Recommended technical direction

I want you to start from this idea:

### Primary direction

Use a **decoder-aware importance score** instead of ranking features only by a local activation-like value.

The intuition is:

- a feature is important not just because it is active,
- but because it can write a meaningful contribution downstream,
- and because removing it changes the reconstruction / later attribution state.

So a better score should include both:

- feature activation,
- and some measure of decoder contribution strength.

One natural starting form is something like:

\[
s_i = a_i^2 \lVert D_i \rVert_F^2
\]

where:

- `a_i` = activation of feature `i`
- `D_i` = decoder contribution block / downstream write tensor for feature `i`

This is not sacred; it is a starting point. The important thing is that the score should be more principled than “large local activation = important.”

### Budget allocation

Please also avoid a blind global collapse where almost all budget goes to late layers.

That does **not** mean forcing uniform retention across layers. It means using a better allocation rule so that the retained set does not become pathological just because the scoring signal is poorly calibrated.

A layer-aware allocation rule is a good starting point.

### Stretch goal

If the decoder-aware score is still not good enough, a stronger option is:

- use a good prefilter,
- then run a greedy refinement step that tries to preserve reconstruction quality.

That is a stretch goal, not the baseline requirement.

---

## Scope: what you should own

Please own these three things:

1. **A better scoring rule** for early pruning.
2. **A better budget-allocation rule** than the current coarse collapse.
3. **An evaluation loop** that compares sparse runs to exact-reference runs.

### What is out of scope

To keep this bounded, you do **not** need to own:

- general Phase 4 scheduler optimization,
- prefix caching,
- the overall benchmark orchestration for the whole project,
- late-prefix stress fixtures,
- broad generalization outside the exact chunked CLT path.

Please focus on **normal prompt traces**, not late-prefix traces.

---

## Where to work

### Main implementation: the fork

Most of your code changes should go in:

`../circuit-tracer_chunked`

The most relevant files are:

- `circuit_tracer/attribution/sparsification.py`
- `circuit_tracer/transcoder/cross_layer_transcoder.py`
- `circuit_tracer/attribution/attribute_nnsight.py`
- `circuit_tracer/replacement_model/replacement_model_nnsight.py`
- `tests/test_double_pass_sparsification.py`

You may also look at:

- `circuit_tracer/transcoder/single_layer_transcoder.py`

but exact chunked CLT is the main target.

### Project-side scripts you will use

In this repo (`nlp_research_project`), the most relevant files are:

- `trace_pipeline_chunked.py`
- `experiments/run_sparsification_experiment.py`
- `experiments/analyze_sparsification_experiment.py`
- `scripts/trace_exact_smoke.ascend.sbatch`
- `scripts/trace_exact_reference_overnight.ascend.sbatch`
- `scripts/trace_sparsification_experiment.ascend.sbatch`

---

## Deliverables

By the end of this task, I want the following from you.

### 1. Library implementation

In the fork, add:

- a better early-sparsification scoring path,
- a clean config / API path,
- useful diagnostics,
- tests.

### 2. Exact-reference evaluation subset

Create your own small exact-reference dataset consisting of a handful of prompts that you will use as the gold standard.

### 3. Sparse-vs-exact comparison results

For those prompts, compare the new method against exact runs and summarize:

- how much runtime / memory improved,
- how similar the sparse graphs are to exact,
- how the retained set behaves by layer,
- whether the new method is clearly better than the current baseline.

### 4. Final recommendation

At the end, I want a short note from you saying:

- what method you implemented,
- what configuration you recommend,
- why we should trust it,
- and where it still fails.

---

## Evaluation plan I want you to follow

### Step 1: build a small exact-reference set first

Please generate your **own exact-reference runs** and use those as the gold standard.

My recommendation:

- use about **5 prompts**,
- choose them to cover a range of difficulty / prompt length / feature count,
- use only **normal prompts**,
- do **not** use late-prefix variants for this task.

### Important runtime clarification

On Ascend, you should assume roughly:

- about **1.5 hours per traced step** in a typical exact run,
- and **2 hours per traced step** as the safe planning budget for harder prompts.

So when I say “5 prompts,” I mean something like:

- **1 traced step per prompt** to start,
- not a long multi-step completion trace.

Start small. You can always add more steps later if the method looks promising.

### Suggested initial exact-reference setup

For each of the ~5 prompts:

- `temperature=0.0`
- `completions=1`
- trace **1 step** at first
- save raw graph outputs

That gives you a manageable exact-reference subset for comparison.

---

## Save both exact and sparse graph outputs

Please save **both**:

1. the exact-reference raw graphs,
2. the corresponding sparse-run raw graphs.

Use **your own scratch storage**, not local repo storage, because scratch is:

- large enough for these artifacts,
- visible to the rest of the project team,
- and therefore useful to other teammates too.

Recommended scratch layout in **your** scratch folder:

```text
/fs/scratch/PAS3272/<your-osc-username>/jayden_direction3/
  exact_reference/
  sparse_eval/
```

Please keep the directory names clean and stable, and send the path to the team once you start writing results there so the rest of us know where your artifacts live.

---

## Important caveat about raw graphs

Saving raw `.pt` graphs is absolutely worth doing, but it is **not enough by itself**.

Why?

Because the saved raw graph is still the graph produced by the attribution procedure after its usual selection / expansion logic. It is not automatically the complete pre-Phase-4 candidate universe.

So for this task, I want you to preserve **both**:

### A. Final graph outputs

These are for exact-vs-sparse graph comparison.

Useful metrics here include:

- feature overlap / Jaccard,
- edge Jaccard,
- weighted edge Jaccard,
- overlap in final selected features.

### B. Pre-Phase-4 diagnostics

These are for understanding how good the pruning rule is *before* final graph selection.

Useful diagnostics include:

- candidate count before pruning,
- candidate count after pruning,
- per-layer candidate counts,
- per-layer retained counts,
- retained score mass,
- retained activation mass,
- reconstruction error relative to the unpruned candidate set,
- any decoder-aware retention metric you add.

This is important because a sparse run can look okay at the end while still having a poor pruning rule internally.

---

## What “good” looks like

I will consider this successful if your method achieves **both** of the following.

### Systems improvement

- significantly fewer candidates entering Phases 3–4,
- reduced runtime and/or memory,
- no catastrophic instability on normal long prompts.

### Fidelity improvement

- clearly better exact-vs-sparse similarity than the current coarse top-`k` baseline,
- less pathological collapse into later layers,
- better reconstruction / retention behavior,
- enough evidence that we can trust it for the final project runs.

The target is **not** perfect identity with exact. The target is a sparse method that remains faithful enough to be a real optimization, not a different algorithm.

---

## Suggested milestones

### Milestone 1 — understand the current baseline

Before changing anything, make sure you can clearly answer:

- what the current scoring rule is,
- what the current budget rule is,
- how the retained set distributes by layer,
- how current sparse runs compare to exact on a tiny subset.

### Milestone 2 — implement a better scorer

Add a better importance score in the fork.

Minimum expectation:

- better motivated than raw activation top-`k`,
- configurable,
- instrumented with useful diagnostics.

### Milestone 3 — implement a better allocation rule

Add a budget-allocation rule that avoids pathological late-layer collapse.

### Milestone 4 — build the exact-reference subset

Run the ~5-prompt exact-reference set and save raw outputs to your scratch folder.

### Milestone 5 — compare sparse vs exact

Run the same prompts with your sparsification method and compare them to the exact references.

### Milestone 6 — recommend a default configuration

Decide what we should actually use for final project tracing.

---

## How to use this repo

### Environment rules

Please follow these rules while working:

- use `uv run ...` for Python,
- do **not** run GPU tracing on login nodes,
- launch tracing through **SLURM jobs**,
- keep large outputs on `/fs/scratch/PAS3272/<your-osc-username>/...`.

### Repo structure

- main project repo: `nlp_research_project`
- forked tracing library: `../circuit-tracer_chunked`

You will mostly **edit code in the fork**, then **run experiments from the main repo**.

---

## What to run locally (safe on login node)

### In the main repo

```bash
uv run ruff check .
uv run ty check .
```

### In the fork repo

After changing sparsification code, run targeted tests such as:

```bash
uv run pytest tests/test_double_pass_sparsification.py
```

If you add new targeted tests, run those too.

---

## What to run for tracing

### Preferred tracing path

Use the **fork-native exact chunked** pipeline:

- `trace_pipeline_chunked.py`

Do **not** use the legacy monkey-patch path unless you specifically need it as a comparison baseline.

### Single-prompt command pattern

This is the basic command pattern for one exact run:

```bash
uv run python trace_pipeline_chunked.py \
  --prompts 1 \
  --gsm8k-indices <PROMPT_ID> \
  --completions 1 \
  --temperature 0.0 \
  --max-steps 1 \
  --output-dir <SCRATCH_DIR> \
  --attribution-batch-size <BATCH> \
  --decoder-chunk-size <CHUNK> \
  --cross-batch-decoder-cache-bytes <CACHE_BYTES> \
  --save-raw
```

This should be run inside a SLURM job, not on a login node.

### Sparse-run command pattern

For sparse runs, add the sparsification flags:

```bash
  --sparsify-per-layer-position-topk <TOPK> \
  --sparsify-global-cap <GLOBAL_CAP>
```

If you change the sparsification API in the fork, update the CLI path as needed.

---

## Existing SLURM entrypoints you can reuse

These scripts already exist and are good starting points:

- `scripts/trace_exact_smoke.ascend.sbatch`
- `scripts/trace_exact_reference_overnight.ascend.sbatch`
- `scripts/trace_sparsification_experiment.ascend.sbatch`

The experiment runner that these scripts use is:

- `experiments/run_sparsification_experiment.py`

You can either:

- reuse those scripts directly, or
- make a small scenario file for your own exact-reference and sparse-eval runs.

---

## Recommended working loop

Please use this loop rather than starting with a large sweep:

1. modify code in `../circuit-tracer_chunked`,
2. run lightweight local tests,
3. run one tiny exact smoke trace,
4. run one tiny sparse trace on the same prompt,
5. inspect the diagnostics and graph outputs,
6. only then scale to the ~5-prompt exact-reference subset.

---

## Minimum diagnostics I want you to add or preserve

Please make sure your implementation reports enough information to debug the method.

At minimum, I want diagnostics such as:

- `candidate_count_before`
- `candidate_count_after`
- `retained_activation_mass`
- a decoder-aware retained-mass metric if you add one
- `per_layer_candidate_counts`
- `per_layer_retained_counts`
- reconstruction error relative to the unpruned set
- sparsification wall-clock time

---

## Final expectation

At the end of this task, I want you to be able to say something like:

> I implemented a more principled early-sparsification method in `../circuit-tracer_chunked`, evaluated it against a small exact-reference set saved in my scratch folder, and here is the configuration we should use because it reduces Phase 3–4 cost while staying much closer to exact tracing than the current baseline.
