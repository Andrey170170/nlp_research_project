# Jay — Direction 3 Task Brief: Repeated-Prefix / Temporal-Tracing Caching

## What I am asking you to do

Your task is to improve **repeated-prefix caching** for our temporal-tracing workflow.

Concretely, I want you to:

1. **build a baseline forward-pass prefix cache** so shared-prefix activations are reused instead of recomputed at every generation step,
2. **investigate whether some backward-pass information can also be safely reused** when the prefix and attributed features match,
3. **evaluate caching fidelity against uncached exact traces**, and
4. recommend the safest caching mode we should trust for the project runs.

The goal is **not** just to make tracing faster. The goal is to make it faster **without changing the traced result in ways that matter**.

---

## Why this matters

In our temporal-tracing use case, consecutive generation steps often share a **long common prefix**.

That means we are repeatedly paying for the same work:

- forward-pass activations over the shared prefix,
- and potentially parts of the backward / attribution path that depend only on that same prefix.

If we can reuse that work safely, we may get a major speedup on repeated-step traces. But cache reuse is only useful if it preserves trace fidelity. A fast cache that silently changes the graph is not acceptable.

---

## Short background / definitions

You do **not** need to solve every caching problem at once. The task has two layers.

### Baseline target: forward-pass prefix caching

For consecutive temporal-tracing steps:

- the input prefix is the same,
- only the newly generated token(s) change,
- so the prefix activations should be reusable.

At minimum, I want a path that caches and reuses the shared-prefix forward state instead of recomputing it every step.

### Stretch target: partial backward-pass reuse

Beyond that, investigate whether some **backward-pass intermediates** can also be reused when the trace structure is stable enough.

A plausible direction is:

- identify features on the shared prefix,
- check that they are attributed similarly across consecutive steps,
- fingerprint or hash intermediate backward results,
- and skip redundant recomputation only when the cached state is provably equivalent.

This should be treated as an **exploratory reuse path**, not the baseline requirement.

---

## What I want you to optimize for

Please do **not** optimize only for:

- runtime,
- memory,
- or cache hit rate.

Those matter, but they are not enough.

What I actually want is:

> a cache strategy that removes redundant repeated-prefix work **while preserving exact traced results as faithfully as possible**.

So the priority order is:

1. correctness / fidelity,
2. then measurable speedup,
3. then implementation simplicity.

---

## Scope: what you should own

Please own this slice of the problem:

### In scope

- repeated-prefix cache design for temporal traces,
- forward-pass cache reuse over shared prefixes,
- cache keying / invalidation rules,
- diagnostic instrumentation for cache hits and misses,
- exploratory backward-pass reuse guarded by fingerprints / hashes,
- comparison against uncached exact baseline runs.

### Out of scope

To keep this bounded, please do **not** take on:

- unrelated sparsification work,
- general model optimization outside repeated-prefix temporal tracing,
- broad cross-prompt memoization,
- training changes,
- or large-scale benchmark orchestration for the whole project.

Focus on **consecutive-step traces with a long shared prefix**.

---

## Where to work

### Main implementation: the fork

Most of your code changes should go in:

[circuit-tracer_chunked](https://github.com/Andrey170170/circuit-tracer_chunked)

The likely touchpoints are the tracing / caching / attribution paths in the fork, especially wherever the prefix state, decoder cache, or step-to-step reuse logic lives. I am intentionally not naming exact function names here because the right entry points may shift as you inspect the code.

### Project-side scripts you will likely use

In this repo (`nlp_research_project`), the relevant files are likely:

- `trace_pipeline.py`
- `trace_pipeline_chunked.py`
- `circuit_utils.py`
- `scripts/*.sbatch`

You may also want to add or update a small experiment driver / analysis note in `docs/` if that helps keep the evaluation organized.

### Artifact location

Please keep outputs on scratch, not in the repo.

Recommended layout:

```text
/fs/scratch/PAS3272/<your-osc-username>/temporal_prefix_cache/
  baseline_exact/
  forward_cache/
  backward_reuse/
  analysis/
```

Keep the directory names stable so the team can find and compare results later.

---

## Deliverables

By the end of this task, I want the following from you.

### 1. Forward-pass prefix cache baseline

A working repeated-prefix cache path that:

- reuses shared-prefix forward activations,
- has clear invalidation rules,
- and can be toggled on/off for comparison.

### 2. Diagnostic instrumentation

Add enough logging / metadata to answer:

- what was cached,
- what was reused,
- what was recomputed,
- and why cache reuse was rejected when it was.

### 3. Exploratory backward-pass reuse path

A conservative prototype or study for reusing some backward-pass information, ideally with:

- fingerprints / hashes,
- strict equivalence checks,
- and a clear explanation of when reuse is safe.

### 4. Small exact-reference evaluation set

Create a small repeated-prefix evaluation set of your own, using a handful of stepwise traces that share a prefix.

### 5. Recommendation

Write a short final note saying:

- what you implemented,
- whether forward cache reuse is trustworthy,
- whether backward reuse is trustworthy,
- what configuration you recommend,
- and where it still fails.

---

## Evaluation plan I want you to follow

### Step 1: start with a tiny exact baseline

First build a small set of exact uncached traces that you will treat as the gold standard.

Suggested minimum:

- 3–5 scenarios,
- each with a clear repeated-prefix structure,
- deterministic decoding where possible,
- and enough steps to see reuse across consecutive generations.

### Step 2: compare three modes

For the same scenarios, compare:

1. **uncached exact**
2. **forward-prefix cache only**
3. **forward-prefix cache + backward reuse prototype**

### Step 3: measure fidelity first, speed second

For each mode, record:

- final trace equivalence / overlap,
- graph similarity metrics,
- cache hit rate,
- runtime,
- peak VRAM,
- host RAM if available,
- and invalidation / fallback counts.

### Step 4: test the risky case explicitly

Please include at least one case where:

- the prefix is long,
- the generation steps are close together,
- and backward reuse is most likely to look tempting but could be wrong.

That is where hidden cache bugs usually show up.

---

## What to save

Please save both:

1. the **uncached exact baseline artifacts**,
2. the **cached artifacts** for each cache mode,
3. and a small comparison table / summary.

Useful artifacts include:

- per-step manifests,
- trace graphs,
- cache metadata,
- reuse decisions,
- and any hash/fingerprint records used for backward reuse.

If you save raw graph outputs, that is good, but it is not enough by itself. I also want the cache decision history so we can tell whether a result is genuinely reused or just re-derived.

---

## What “good” looks like

I will consider this successful if your work achieves both of the following.

### Systems improvement

- shared-prefix forward work is clearly reused,
- runtime is meaningfully lower on repeated-prefix traces,
- cache overhead does not dominate the win.

### Fidelity improvement

- cached runs match the uncached exact baseline closely,
- any reuse rule is conservative and well-justified,
- backward reuse is only used when equivalence is strong enough to trust,
- and we have enough evidence to know when not to use it.

The target is **not** “maximum cache hit rate.” The target is “safe reuse that preserves the trace.”

---

## Suggested milestones

### Milestone 1 — baseline understanding

Before changing anything, make sure you can clearly answer:

- where the repeated-prefix work is currently recomputed,
- what state is already available for reuse,
- and what the exact baseline behavior looks like.

### Milestone 2 — forward cache path

Implement the forward-pass prefix cache and verify it against the uncached baseline.

### Milestone 3 — diagnostics

Add cache-hit / miss / invalidation diagnostics and make them easy to inspect.

### Milestone 4 — backward reuse study

Prototype a conservative backward-pass reuse mechanism using fingerprints or hashes.

### Milestone 5 — evaluation

Run the small repeated-prefix set and compare all modes.

### Milestone 6 — recommendation

Decide what reuse mode we should actually trust for the final project workflow.

---

## How to work in this repo

### Environment rules

Please follow these rules while working:

- use `uv run ...` for Python,
- do **not** run GPU tracing on login nodes,
- launch GPU work through **SLURM jobs**,
- keep large outputs on `/fs/scratch/PAS3272/<your-osc-username>/...`.

### Validation

For local validation, keep it lightweight:

- `uv run ruff check .`
- `uv run ty check .`

Use SLURM for any GPU-dependent tracing or caching experiments.

---

## Risks and open questions

### Main risks

- cache invalidation may be harder than the reuse win,
- backward-pass reuse may turn out to be too brittle,
- floating-point sensitivity may make “same prefix” less identical than it looks,
- and cache bookkeeping may eat part of the runtime savings.

### Open questions

- What is the safest cache key for a repeated-prefix temporal trace?
- Which backward intermediates, if any, are stable enough to fingerprint?
- Where is the right cutoff between safe reuse and overly aggressive reuse?
- Is the best result a full reuse path, or just a conservative forward-only cache?

Please treat these as part of the task, not as blockers.

---

## Bottom line

I want you to make repeated-prefix temporal tracing cheaper **without compromising the trace**.

Start with the forward-prefix cache. Then, if the evidence supports it, explore conservative backward-pass reuse.
