# Sparsification experiment spec

## Problem statement
Exact chunked tracing is now instrumented in the local fork, but full-completion runs are still too slow for routine science. Profiling shows the bottleneck is not sparse encoding; it is reconstruction/component computation and phase 3 chunked attribution over a very large active set (~3.9M features on a representative prompt).

We need a controlled experiment to decide whether double-pass sparsification is a scientifically acceptable middle ground between:
- the exact fork baseline,
- the old early-cap monkeypatch in `trace_pipeline.py`, and
- a proposed double-pass sparsification design that reduces work before expensive exact phases.

## Scope
- Compare three methods:
  1. exact fork baseline on a tiny scope,
  2. old early-cap/top-K patch baseline,
  3. proposed double-pass sparsification.
- Use GSM8K with Gemma-3-1B-IT + GemmaScope-2 CLTs.
- Keep tracing single-GPU per run.
- Preserve `attribute(...)` compatibility and existing `Graph` schema/node ordering.

## Non-goals
- Full overnight exact tracing of long completions.
- Multi-GPU single-trace execution.
- Final graph-only pruning as the only sparsification mechanism.

## Proposed approach

### 1) Calibration first
Run a small calibration set before the main comparison.
- Candidate budgets: `16k`, `32k`, `64k`, `128k`.
- Optional secondary sweep: decoder chunk size (only if budget results are inconclusive).
- Measure the effect on phase 0, reconstruction, and phase 3 timing separately.
- Select the smallest budget that keeps fidelity within acceptable bounds on the calibration prompts.

### 2) Main comparison slice
Use ~10 GSM8K prompts, stratified by prompt length and/or active feature count if feasible.
- Default to one deterministic completion per prompt.
- Optionally add a second sampled completion on a smaller subset for robustness.
- Trace only a few early steps per completion; do not target full completion coverage.

### 3) Baselines
- **Exact baseline:** only on a tiny scope (2-3 steps, 1 deterministic completion). Use it as the ground truth reference, not as the overnight workload.
- **Old early-cap patch:** keep as a speed reference, but treat its semantics as intentionally altered.
- **Double-pass sparsification:** primary candidate method.

### 4) Metrics
Record per method, per prompt, and per step:
- runtime and peak memory by phase,
- node/edge overlap vs exact,
- attribution mass retained,
- rank correlation and overlap@K,
- temporal stability metric correlation vs exact,
- whether correct-vs-incorrect separation trends survive.

### 5) Orchestration
For overnight runs, schedule one trace per GPU.
- Prefer SLURM job arrays or multiple independent jobs.
- Do not introduce multi-GPU single-trace work.
- Keep each job independent so failures do not block the full batch.

## Acceptance criteria
Go only if the proposed method:
- reduces runtime materially versus exact on the main slice,
- preserves exact-like rankings and overlap on calibration cases,
- retains the direction of the correct-vs-incorrect separation signal,
- and stays memory-safe without special-case manual pruning.

Go/no-go gate:
- **Go** if fidelity on the calibration set is stable across budgets and the main slice shows a clear runtime win without reversing the scientific signal.
- **No-go** if the method only works by collapsing to the old semantics or if the temporal stability signal is not qualitatively preserved.

## Validation plan
1. Run exact baseline on 2-3 steps of one deterministic completion.
2. Calibrate budgets on the same tiny scope.
3. Compare all three methods on ~10 prompts with the same tracing settings.
4. Inspect failures where sparsified traces diverge most from exact.
5. If needed, rerun a small robustness subset with a second completion.

## Risks / open questions
- The active set may still be too large for some prompts even after calibration.
- The best budget may differ by layer depth or prompt length.
- Old top-K semantics may look fast but remain scientifically misaligned.
- Need a clear rule for selecting the calibration prompts so the budget choice is not biased.

## Assumptions
- Exact tracing remains the reference only for tiny slices.
- Sparse candidate restriction must happen before or during reconstruction/attribution, not after graph materialization.
