# Current Implementation Plan — Phase 4 Hybrid Row-Store Fix

## 1. Problem statement

The permanent overflow fix remains validated and fp32 remains the correct default
for exact compact tracing.

The latest Phase 4 investigation changed the immediate priority again.

Current best-supported conclusion:

- the main recurring scheduler-relevant Phase 4 growth is still concentrated
  **between** `compute_batch(...)` calls,
- the strongest current suspect remains the **row append / writeback path**, not
  refresh ranking itself,
- however, the broad row-store rewrite that replaced the full read/write path
  with explicit file I/O was the wrong fix:
  - refresh became far slower,
  - Phase 5 also became much slower,
  - RSS behavior did not improve enough to justify the regression.

Operational conclusion:

- the current problem is still a **row-store memory behavior** problem,
- but the safe fix is **not** a full row-store file-I/O replacement,
- we need to preserve the cheap read path for refresh and Phase 5,
- and target only the **append/writeback residency behavior**.

## 2. Chosen approach

Chosen implementation direction:

1. **keep/reintroduce the cheap mmap-style read path** for:
   - `read_feature_rows(...)`
   - `materialize_dense_feature_slice(...)`
2. **change only the append/writeback path** so Phase 4 no longer keeps a
   long-lived writable full-file mapping hot across batches,
3. validate that this reduces the batch-aligned file-backed RSS growth without
   regressing refresh or Phase 5,
4. only after that, revisit smaller staging/cache cleanup.

Explicitly rejected as the current path:

- full row-store file-I/O replacement for both writes and reads,
- refresh/scheduler redesign as the immediate fix,
- cheaper frontier indexes/scores as the immediate fix,
- two-tier storage as the immediate fix.

## 3. Scope / non-goals

### In scope

- `_FileBackedFeatureRowStore.append_rows(...)`,
- row-store writeback mechanics and append-side residency behavior,
- preserving or restoring the cheap existing read/materialization path,
- focused tests for exact row content, ordering, and denominator semantics,
- lightweight follow-up staging/cache hygiene only after the main hybrid fix is
  validated.

Primary files likely in scope:

- `../worktrees_opt/circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
- `../worktrees_opt/circuit-tracer_chunked/tests/test_attribute_nnsight_telemetry.py`
- `../worktrees_opt/circuit-tracer_chunked/tests/test_partial_influences.py`
- `../worktrees_opt/circuit-tracer_chunked/tests/test_chunked_decoder_optimizations.py`

### Non-goals

- no frontier-selection changes,
- no approximate scheduler/index redesign,
- no full read-path replacement,
- no major `compute_batch(...)` redesign in this phase,
- no GPU launches from login nodes.

## 4. Root-cause model to act on

Current working model:

1. `ctx.compute_batch(...)` produces a large row block,
2. Phase 4 staging prepares CPU rows,
3. append/writeback makes more of the backing row store hot/resident,
4. that batch-aligned growth drives the bad file-backed/cgroup pattern.

New lesson from the rejected experiment:

- replacing **reads** with explicit file-I/O copies destroys the cheap demand-
  paged path used heavily by refresh and Phase 5,
- so the next fix must stay narrow:
  **fix writeback residency while preserving read efficiency**.

## 5. Main implementation workstreams

### Workstream A — hybrid row-store fix

Goal: keep the row-store read path cheap while changing append/writeback behavior
to reduce batch-aligned file-backed RSS growth.

Target design:

- use a **hybrid** row-store,
- preserve mmap-style cheap reads/materialization,
- replace only the append-side writable full-file behavior with a narrower write
  mechanism, such as:
  - explicit offset writes, or
  - a short-lived append-window mapping that is flushed/unmapped immediately,
- if needed, reopen/invalidate the read view after append so semantics remain
  simple and exact.

Implementation constraints:

- exact same row content,
- exact same `row_to_node_index` alignment,
- exact same denominator semantics,
- no silent approximation,
- no change to frontier or queue semantics.

### Workstream B — validate the hybrid cut before extras

Goal: show that the writeback-targeted fix improves memory behavior without the
large read-path regression seen in the rejected full file-I/O attempt.

Questions to answer:

1. does batch-aligned file-backed RSS growth shrink materially,
2. does refresh wall time stay near the earlier probe baseline,
3. does Phase 5 wall time stay sane,
4. do outputs remain unchanged.

### Workstream C — follow-up staging/cache hygiene after validation

This stays **second**, not first.

Candidate follow-up cleanup:

- reduce or disable row-store read cache only if it still contributes after the
  hybrid fix,
- shrink/reset reusable CPU staging buffers when safe,
- remove obviously unnecessary staging retention.

Decision rule:

- do this only after the hybrid writeback fix is validated, so its effect stays
  measurable.

## 6. Implementation order

Recommended sequence:

1. back out the rejected broad read/write file-I/O replacement if needed,
2. implement the **append-side-only** hybrid writeback change,
3. preserve or restore cheap mmap-style reads/materialization,
4. update focused tests for row-store correctness and telemetry invariants,
5. run lightweight local validation,
6. rerun the small Ascend Phase 4 memory probe on:
   - `828_base`
   - `361_base`
7. compare:
   - batch-to-batch file RSS growth,
   - cgroup/Grafana spike shape,
   - refresh wall time,
   - Phase 5 wall time,
   - exact artifact/correctness invariants,
8. only then add the smaller staging/cache hygiene cleanup.

## 7. Acceptance criteria

This plan is successful when:

1. the batch-aligned file-backed RSS growth is materially reduced on the canonical
   fast prompts,
2. process file RSS no longer tracks Phase 4 batch-buffer size nearly 1:1,
3. refresh wall time does not suffer a major regression relative to the earlier
   Phase 4 memory probe baseline,
4. Phase 5 wall time does not blow up,
5. fp32 compact outputs remain stable,
6. retained edges remain `20000`,
7. decoder-load counts remain stable.

## 8. Risks and open questions

- append-side-only changes may still leave too much residency pressure,
- keeping cheap reads may preserve some of the original file-backed growth,
- reopening/invalidation logic must stay simple enough to avoid row-store bugs,
- staging/cache hygiene may still be needed, but it should remain secondary,
- broader scheduler ideas may still be valuable later, but they should not be
  mixed into the current safe fix.

## 9. Later scheduler-redesign considerations

These are **not** standalone work items for the current phase.

Do not pursue them now as separate optimization tracks. Only reconsider them if
they become useful during a later **scheduler/frontier redesign** and only in
that broader context where we can reason about graph exploration behavior and the
memory operations implied by the new scheduler shape.

Examples to keep in mind later, not now:

- two-tier hot/cold row storage,
- cheaper scheduler indexes/summaries,
- incremental refresh state,
- approximate ranking or prefilter schemes,
- larger storage/layout redesigns coordinated with scheduler work.
