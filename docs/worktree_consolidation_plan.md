# Worktree consolidation plan

Last reviewed: 2026-04-21

## Decisions locked after review

- prefer **main worktree behavior** when there is a conceptual overlap, because it is more comprehensive and exposes more control
- still adopt the opt branch's public dtype knob naming: **`exact_trace_internal_dtype`**
- keep the new main-side cross-cluster debug / debug-telemetry artifact flow
- preserve the newer opt-worktree findings doc/spec updates
- do **not** merge generated artifacts; regenerate them after consolidation if needed

## Current state

- main repo worktree: `exact-trace-bench-harness` @ `b432c4e`
- sibling worktree: `../worktrees_opt/nlp_research_project` on `exact-trace-bench-opt` @ `50752cd`
- branch relationship: `exact-trace-bench-harness` is the merge-base of `exact-trace-bench-opt`
- practical meaning: the opt worktree is **ahead by 2 commits**, while the main repo has additional **uncommitted local changes**

So the git history itself is not badly diverged, but the working trees now overlap in a few important files.

## What exists only in the opt worktree branch

Committed on `exact-trace-bench-opt`:

- `EXPERIMENTS.md`
- `docs/phase4_refresh_optimization_spec.md`
- `trace_pipeline.py`
- `experiments/exact_trace_bench/config.py`
- `experiments/exact_trace_bench/scenarios.py`
- `experiments/run_sparsification_experiment.py`
- smaller dtype-plumbing edits in:
  - `trace_pipeline_chunked.py`
  - `experiments/exact_trace_bench/extract.py`
  - `experiments/extract_benchmark_index.py`
  - `PLAN.md`

Additionally, the opt worktree has **uncommitted** doc updates in:

- `EXPERIMENTS.md`
- `docs/phase4_refresh_optimization_spec.md`

Those local doc edits add the newest true-fp32/fp64 findings and should be preserved.

## What exists only in the main worktree

Uncommitted on `exact-trace-bench-harness`:

- `trace_pipeline_chunked.py`
- `experiments/exact_trace_bench/extract.py`
- `experiments/extract_benchmark_index.py`
- `PLAN.md`
- `midpoint_checkin_draft.md`
- `tests/test_cross_cluster_debug_artifacts.py` (new)

Main also has untracked generated artifacts that should not be treated as source-of-truth.

## Merge buckets

### Safe / mostly mechanical to bring over

These do not currently collide with main local edits and can be taken largely as-is:

- `EXPERIMENTS.md`
- `docs/phase4_refresh_optimization_spec.md`
- `trace_pipeline.py`
- `experiments/exact_trace_bench/config.py`
- `experiments/exact_trace_bench/scenarios.py`
- `experiments/run_sparsification_experiment.py`

### Manual merge hotspots

These need explicit reconciliation because both sides now touch the same conceptual area:

- `trace_pipeline_chunked.py`
- `experiments/exact_trace_bench/extract.py`
- `experiments/extract_benchmark_index.py`
- `PLAN.md`

## Expected conflict themes

### 1. Dtype knob naming + artifact schema

Opt branch direction:

- standardizes on `exact_trace_internal_dtype`
- propagates that field through scenarios and pipelines

Main local direction:

- uses older `internal_precision` / `internal_precision_requested`
- already adds extraction fields for resolved dtype maps

Consolidation recommendation:

- keep **`exact_trace_internal_dtype`** as the public runtime knob name
- preserve main's richer manifest/extraction reporting for:
  - resolved dtype map
  - requested dtype
  - downstream benchmark columns
- treat this as a rename + schema merge, not an either/or choice

### 2. Cross-cluster debug plumbing

Main local changes add:

- `cross_cluster_debug_*` manifest/artifact capture
- extraction support for those artifacts
- `tests/test_cross_cluster_debug_artifacts.py`

Opt branch dtype work was made on an older base and effectively drops that plumbing from the touched files.

Consolidation recommendation:

- keep the cross-cluster debug work
- reapply the dtype-plumbing changes on top of that version of the files
- keep the test file and update it only if helper names/signatures change

### 2.5 Launcher wiring gap (must be fixed during consolidation)

There is a real wiring bug in the current main-side launcher path: the pipeline
supports the new debug telemetry / debug-artifact knobs, but the experiment
launcher path does not propagate them consistently.

Current gap in main:

- `trace_pipeline_chunked.py` exposes:
  - `--internal-precision` (to be renamed during consolidation)
  - `--cross-cluster-debug`
  - `--telemetry-max-events`
- but `experiments/run_sparsification_experiment.py` currently only forwards:
  - `--phase4-anomaly-debug`
- and does **not** forward:
  - `cross_cluster_debug`
  - `telemetry_max_events`
  - the dtype knob (`internal_precision` today, `exact_trace_internal_dtype` after consolidation)

Related schema/config gaps in main:

- `experiments/exact_trace_bench/config.py` does not yet include these fields in `base_trace_defaults()`
- `experiments/exact_trace_bench/scenarios.py` does not yet list them in `EXACT_MODE_KNOB_KEYS`

Consolidation recommendation:

- fix launcher/config propagation as part of the merge, not as a later cleanup
- final launcher/config surface should use:
  - `exact_trace_internal_dtype`
  - `cross_cluster_debug`
  - `telemetry_max_events`
- benchmark scenario generation, scenario JSON, and runtime command construction should all round-trip these fields cleanly

This is important because otherwise the consolidated branch would keep the code
paths but still fail to launch them from the benchmark harness.

### 3. Planning-doc focus drift

Main `PLAN.md` is now a broad roadmap covering:

- precision contract cleanup
- cross-cluster diagnostics
- RSS / memory stabilization

Opt `PLAN.md` is a narrow implementation plan for:

- Direction A / Phase 4 refresh chunk reuse

Consolidation recommendation:

- keep `PLAN.md` as the **single active roadmap**
- do **not** overwrite main `PLAN.md` with the opt version
- fold the opt plan's tactical refresh work into:
  - a refreshed section of `PLAN.md`, or
  - the detailed spec in `docs/phase4_refresh_optimization_spec.md`

## Docs consolidation recommendation

### Keep as canonical

- `PLAN.md` = active roadmap / decisions
- `EXPERIMENTS.md` = living experiment inventory and conclusions
- `docs/phase4_refresh_optimization_spec.md` = detailed refresh-optimization spec

### Keep, but treat as secondary / archival

- `midpoint_checkin_draft.md` = report narrative, not operational source-of-truth
- `experiments/exact_trace_bench/README.md` = how-to / workflow doc
- `plan.md` and `prefix_caching/plan.md` = archive unless those directions are still active

### Important content to preserve from the opt docs

- the true `fp32` vs `fp64` runtime-knob findings
- the conclusion that healthy-prompt speedups were better explained by dtype behavior than by the first refresh-cache attempt
- the warning that `fp64` is the current default, but not necessarily the final long-trace numerical-stability solution

### Important content to preserve from the main docs

- broader cross-cluster debug goals
- resolved dtype-map / precision-contract framing
- RSS-spike and host-memory investigation framing
- midpoint narrative context

## Generated or non-source artifacts to ignore / regenerate

Do not merge these as source files:

- `experiments/generated/cross_cluster_phase1_ascend_scenarios.json`
- `experiments/generated/cross_cluster_phase1_cardinal_scenarios.json`
- `experiments/generated/exact_trace_bench/`
- `experiments/generated/weekend_exact_chunked_fixtures_matched_debug/`
- `midpoint_report.pdf`

These should be regenerated from the consolidated code/docs state when needed.

## Recommended consolidation order

1. **Checkpoint the main worktree first**
   - stash or commit a temporary WIP snapshot before attempting any merge/cherry-pick flow
   - the main worktree currently contains the cross-cluster debug work that should not be lost

2. **Bring over the opt-only committed files**
   - especially `EXPERIMENTS.md`, `docs/phase4_refresh_optimization_spec.md`, `trace_pipeline.py`, and the scenario/config plumbing
   - preserve the newer spec/findings additions from the opt worktree's local docs too

3. **Manually reconcile the overlapping Python files**
   - start with `trace_pipeline_chunked.py`
   - then `experiments/exact_trace_bench/extract.py`
   - then `experiments/extract_benchmark_index.py`
   - target final state: opt's `exact_trace_internal_dtype` naming + main's cross-cluster debug and richer extraction fields

4. **Fix launcher/config propagation immediately after the file merges**
   - update `experiments/exact_trace_bench/config.py`
   - update `experiments/exact_trace_bench/scenarios.py`
   - update `experiments/run_sparsification_experiment.py`
   - ensure benchmark harness scenarios can actually launch:
     - `exact_trace_internal_dtype`
     - `cross_cluster_debug`
     - `telemetry_max_events`

5. **Reconcile docs intentionally**
   - keep `PLAN.md` broad
   - keep `docs/phase4_refresh_optimization_spec.md` detailed
   - keep `EXPERIMENTS.md` as the factual experiment log
   - optionally refresh `midpoint_checkin_draft.md` wording if it will be reused

6. **Drop or regenerate generated artifacts last**
   - do not use generated directories to decide the merge

## Recommended final target state

- one branch/worktree again
- public knob name: `exact_trace_internal_dtype`
- cross-cluster debug support retained
- launcher/config path correctly forwards debug + telemetry knobs
- extraction retains dtype-map and debug-artifact columns
- `PLAN.md`, `EXPERIMENTS.md`, and `docs/phase4_refresh_optimization_spec.md` each have a clear role with minimal duplication

## Short version

This is **not** a scary history merge. The opt branch is simply ahead of the current main commit. The real work is a **manual content consolidation** of four overlapping files, mainly to combine:

- opt's dtype-knob cleanup and new docs
- main's cross-cluster debug plumbing and broader roadmap

If we do that carefully, the rest can come over mostly cleanly.
