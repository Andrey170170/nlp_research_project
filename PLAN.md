# Current Implementation Plan — Worktree Consolidation + Launcher Wiring

## Problem statement

We now have two worktrees that partially overlap:

- main repo worktree: `exact-trace-bench-harness`
- sibling worktree: `../worktrees_opt/nlp_research_project` on `exact-trace-bench-opt`

The opt branch is ahead in git history, but the main worktree has newer local
changes in the files that now matter most:

- `trace_pipeline_chunked.py`
- `experiments/exact_trace_bench/extract.py`
- `experiments/extract_benchmark_index.py`
- `PLAN.md`

The main risk is no longer branch divergence by itself; it is **semantic drift**
between two versions of the same control surface:

1. the opt worktree introduces the newer public dtype knob naming
   (`exact_trace_internal_dtype`) and newer findings/docs,
2. the main worktree has the more comprehensive debug/control surface,
   especially cross-cluster debug artifact capture and richer extraction,
3. the benchmark harness currently does **not** wire all of the main-side debug
   knobs through to the launcher path.

We should consolidate now before more code accumulates on both sides.

Supporting investigation notes live in:

- `docs/worktree_consolidation_plan.md`

This `PLAN.md` is the active execution plan for the consolidation itself.

## Scope

This phase covers:

- merging the worktree changes back into the main repo state,
- defining one canonical public runtime knob surface,
- preserving the richer main-side debug and extraction behavior,
- fixing benchmark-harness propagation for the retained knobs,
- reconciling `PLAN.md`, `EXPERIMENTS.md`, and
  `docs/phase4_refresh_optimization_spec.md`.

## Non-goals

- no generated artifact merge,
- no heavy tracing runs on login nodes,
- no new algorithmic Phase-4 optimization work during the merge itself,
- no attempt to solve the deeper numerical-stability problem in the same change,
- no broad cleanup of all historical docs unless it is required for consistency,
- do not touch `midpoint_checkin_draft.md` in this phase.

## Locked decisions

### 1. Prefer main behavior on conceptual overlap

When main and opt express different versions of the same idea, prefer the main
worktree behavior if it is more comprehensive and exposes more control.

This especially applies to:

- cross-cluster debug artifacts,
- debug telemetry capture,
- richer extraction/manifests,
- broader roadmap framing.

### 2. Adopt the opt branch's public dtype knob name

The canonical public name should be:

- `exact_trace_internal_dtype`

We should rename directly and stop treating `internal_precision` as the public
contract. Backward compatibility is not required for this consolidation.

### 3. Keep main-side cross-cluster debug and debug telemetry

Do not drop the newer main-side artifact flow during the rename/plumbing merge.

Required preserved behavior:

- `cross_cluster_debug_summary.json`
- `cross_cluster_debug_checkpoints.jsonl`
- `cross_cluster_debug_batches.jsonl`
- manifest fields that describe those artifacts
- extraction fields that surface those artifacts downstream

### 4. Preserve opt-side findings docs

The opt worktree contains newer findings that should survive the merge,
especially:

- true `fp32` vs `fp64` findings in `EXPERIMENTS.md`,
- the newer findings added to
  `docs/phase4_refresh_optimization_spec.md`.

### 5. Ignore generated artifacts during source consolidation

Do not merge generated directories or PDFs as source-of-truth.

Generated artifacts should be regenerated later from the consolidated code.

## Why this matters for the research roadmap

After consolidation, the repo should be in a state where future work can happen
in one place again.

The downstream technical goals remain:

- explicit internal dtype control,
- broad cross-cluster diagnostics,
- better visibility into memory spikes,
- later refresh-path optimization work.

But we should not continue those efforts on split worktrees.

## Proposed approach

## Workstream A — Establish one canonical merge target

### Goal

Make the main repo the only active target again and treat the worktree as an
input source, not as an independently evolving branch.

### Tasks

1. checkpoint current main local work before any content merge,
2. intake opt-only committed files first,
3. manually reconcile the overlapping Python files,
4. update docs so the final repo explains the merged state clearly.

### Files in this bucket

Mostly safe to intake from opt:

- `EXPERIMENTS.md`
- `docs/phase4_refresh_optimization_spec.md`
- `trace_pipeline.py`
- `experiments/exact_trace_bench/config.py`
- `experiments/exact_trace_bench/scenarios.py`
- `experiments/run_sparsification_experiment.py`

Manual-merge files:

- `trace_pipeline_chunked.py`
- `experiments/exact_trace_bench/extract.py`
- `experiments/extract_benchmark_index.py`
- `PLAN.md`

## Workstream B — Control-surface consolidation

### Goal

End up with one consistent runtime/control surface that combines:

- opt's dtype naming,
- main's richer debug capture,
- main's richer extraction schema.

### Final API direction

Canonical public field / CLI name:

- `exact_trace_internal_dtype`

Canonical debug fields to keep exposed:

- `phase4_anomaly_debug`
- `cross_cluster_debug`
- `telemetry_max_events`

Canonical manifest/extraction information to preserve:

- requested dtype value,
- resolved dtype map,
- cross-cluster debug artifact presence/status/counts,
- telemetry presence/status/counts.

### Detailed reconciliation plan by file

#### `trace_pipeline_chunked.py`

Target state:

- retain main's cross-cluster debug capture helpers,
- retain main's manifest/debug-artifact writing,
- rename the public dtype control to `exact_trace_internal_dtype`,
- keep step/manifests rich enough for downstream extraction,
- preserve safety checks around unsupported combinations.

Implementation intent:

- do not accept the opt version wholesale,
- instead apply the dtype naming/plumbing changes onto the main version.

#### `experiments/exact_trace_bench/extract.py`

Target state:

- keep main's cross-cluster debug extraction fields,
- keep main's resolved dtype map extraction,
- update naming/columns so the canonical dtype field is aligned with
  `exact_trace_internal_dtype`.

#### `experiments/extract_benchmark_index.py`

Target state:

- same principle as the main extractor above,
- preserve cross-cluster debug visibility,
- preserve dtype-map visibility,
- standardize the benchmark-row schema around the final public knob naming.

## Workstream C — Fix launcher/config propagation

### Goal

Make sure the benchmark harness can actually launch the merged control surface.

### Current bug

Today, the runtime pipeline exposes more debug controls than the launcher path
forwards.

Current mismatch in main:

- `trace_pipeline_chunked.py` exposes:
  - dtype control,
  - `--cross-cluster-debug`,
  - `--telemetry-max-events`
- but `experiments/run_sparsification_experiment.py` currently only forwards:
  - `--phase4-anomaly-debug`

This means the harness cannot fully exercise the newer debug path even though
the pipeline supports it.

### Required config/launcher changes

#### `experiments/exact_trace_bench/config.py`

Add these to `base_trace_defaults()`:

- `exact_trace_internal_dtype`
- `cross_cluster_debug`
- `telemetry_max_events`

#### `experiments/exact_trace_bench/scenarios.py`

Ensure `EXACT_MODE_KNOB_KEYS` includes:

- `exact_trace_internal_dtype`
- `cross_cluster_debug`
- `telemetry_max_events`

This matters so scenario generation and scenario serialization preserve those
fields.

#### `experiments/run_sparsification_experiment.py`

Update command construction so it forwards, when present:

- `--exact-trace-internal-dtype`
- `--cross-cluster-debug`
- `--telemetry-max-events`

### Acceptance for this workstream

- scenario JSON can store these fields,
- generated commands include these flags when requested,
- runtime manifests reflect the launched values.

## Workstream D — Docs reconciliation

### Goal

Bring the documentation back to one coherent story.

### Canonical doc roles after consolidation

- `PLAN.md` = active roadmap / execution plan
- `EXPERIMENTS.md` = living experiment inventory and findings
- `docs/phase4_refresh_optimization_spec.md` = detailed refresh and stability
  spec

### Doc-specific intent

#### `PLAN.md`

This file should track the active merge/consolidation execution plan now, then
later return to the post-consolidation implementation roadmap.

#### `EXPERIMENTS.md`

Preserve:

- the true `fp32` / `fp64` comparison matrix,
- the conclusion that healthy-prompt speedups were more about dtype behavior
  than the first refresh-cache attempt,
- the current understanding that `fp64` is the right immediate default.

#### `docs/phase4_refresh_optimization_spec.md`

Preserve:

- refresh-path optimization context,
- newer normalization-precision findings,
- the warning that fp64 is an immediate default, not the final permanent
  numerical-stability solution.

#### `midpoint_checkin_draft.md`

Keep as secondary narrative/report material only.

Do not edit it during this phase.

## Workstream E — Safe validation

### Goal

Validate the consolidation without violating the repo's HPC constraints.

### Allowed validation in this phase

- lightweight Python checks via `uv run`
- lightweight unit-style checks
- lint/type checks if touched files need them

### Recommended validation set

1. `uv run python tests/test_cross_cluster_debug_artifacts.py`
2. targeted sanity check that scenario command construction includes the new
   flags when requested
3. `uv run ruff check` on touched files or repo-wide if cheap enough

### Explicitly disallowed here

- no GPU tracing on login nodes,
- no heavy extraction over scratch unless intentionally submitted through the
  proper job path.

## Sequencing

### Phase 1 — Plan and checkpoint

1. finish this `PLAN.md`
2. preserve main local state before any merge/cherry-pick attempt
3. use `docs/worktree_consolidation_plan.md` as the detailed reference during
   execution

### Phase 2 — Intake opt-only source/docs

Bring over the opt worktree content that is mostly non-conflicting:

- `EXPERIMENTS.md`
- `docs/phase4_refresh_optimization_spec.md`
- `trace_pipeline.py`
- launcher/config/scenario updates from the opt branch as inputs

Important:

- preserve the newer local doc edits from the opt worktree too, not just the two
  committed branch commits

### Phase 3 — Manually merge overlapping files

Order:

1. `trace_pipeline_chunked.py`
2. `experiments/exact_trace_bench/extract.py`
3. `experiments/extract_benchmark_index.py`

Guiding rule:

- use main as the behavioral base,
- reapply dtype naming/plumbing from opt on top of that base.

### Phase 4 — Fix launcher propagation

Immediately after the overlapping file merge:

1. update `config.py`
2. update `scenarios.py`
3. update `run_sparsification_experiment.py`
4. verify end-to-end flag propagation in the generated command path

### Phase 5 — Validate the consolidated state first

Before starting any new investigation or optimization work:

1. run the lightweight validation set,
2. confirm launcher/config propagation is working,
3. confirm manifests/extraction still surface the expected debug + dtype fields,
4. fix any regressions immediately before opening new workstreams.

This phase exists specifically to avoid continuing on top of a broken merge.

### Phase 6 — Reconcile final docs

1. keep `PLAN.md` as the active roadmap,
2. keep `EXPERIMENTS.md` factual,
3. keep the spec detailed,
4. leave generated artifacts out of the merge.

### Phase 7 — Split into parallel post-merge tracks

Only after Phase 5 passes cleanly:

#### Track A — Cross-cluster drift investigation

Use the consolidated branch as the canonical correctness/debug branch.

Immediate focus:

- paired Ascend/Cardinal reruns,
- earliest-divergence localization,
- validating whether drift begins before Phase 4,
- using the retained cross-cluster debug artifacts as the main evidence source.

Constraint:

- avoid mixing in new optimization changes that would muddy the debug signal.

#### Track B — Optimization work

Run optimization work in a separate branch/worktree off the validated
consolidated state.

Immediate focus:

- refresh-path optimization,
- memory-pressure reduction,
- later numerical-stability improvements.

Constraint:

- periodically rebase/merge from the validated consolidated branch,
- do not let optimization experiments redefine the canonical debug schema.

#### Coordination rule

Track A owns the canonical interpretation/debug surface.

Track B may optimize implementation details, but if it changes:

- dtype semantics,
- debug artifact schema,
- launch/config shape,

then those changes need explicit review before becoming the new baseline.

## Acceptance criteria

The consolidation phase is successful when all of the following are true:

1. there is one active repo/worktree again for normal development,
2. the public dtype knob name is `exact_trace_internal_dtype`,
3. main's cross-cluster debug artifact flow is preserved,
4. extraction still exposes dtype-map and debug-artifact information,
5. benchmark harness config/scenario/launcher code can actually propagate:
   - `exact_trace_internal_dtype`
   - `cross_cluster_debug`
   - `telemetry_max_events`
6. `EXPERIMENTS.md` and the spec preserve the newer worktree findings,
7. generated artifacts are excluded from the source merge.

## Risks

### 1. Silent schema drift

Risk:

- the CLI name changes, but manifests/extraction/benchmark tables still use the
  old field names inconsistently.

Mitigation:

- treat this as an explicit schema merge,
- review runtime args, manifests, extraction, and scenario generation together.

### 2. Launcher appears fixed but still drops knobs

Risk:

- config defaults are updated but command construction still forgets a flag.

Mitigation:

- do a direct command-generation sanity check as part of validation.

### 3. Docs become duplicated instead of clarified

Risk:

- findings get copied into multiple docs without clear ownership.

Mitigation:

- keep strict roles: plan vs experiment log vs detailed spec.

### 4. Generated artifacts contaminate the merge

Risk:

- generated directories distract from the true source merge or create noisy diffs.

Mitigation:

- explicitly exclude them from the source merge and regenerate later if needed.

## Open questions

- after consolidation, should the next active implementation plan return first
  to cross-cluster diagnostics or to numerical-stability cleanup?
