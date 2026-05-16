# Phase 4 Memory Stabilization and Refresh Optimization Spec

Status: Current optimization guidance
Last updated: 2026-05-16

## 1. Goal

Document the current post-fix exact-trace memory situation and define the
durable strategy for the **current safe fix**.

Primary targets:

- reduce scheduler-relevant **Grafana/cgroup RSS spikes**,
- preserve the exact compact trace semantics,
- avoid large throughput regressions,
- keep future scheduler/storage redesign ideas explicitly deferred until after
  the current memory issue is safely contained.

Main optimization fixtures remain:

- `828_base`
- `361_base`

`94_base` remains the anomaly/watch gate.

## 2. Current validated state

### 2.1 Normalization / correctness status

The permanent overflow fix remains validated on the canonical prompt set.

Validated facts:

- the earlier healthy-prompt fp32 collapse findings applied to the **pre-fix**
  code path,
- on the fixed code path, fp32/fp64 compact artifacts matched exactly on
  `828_base`, `361_base`, and `94_base`,
- retained edges stayed at `20000`,
- the old healthy-prompt fp32 collapse signature is gone on the fixed path,
- fp32 is now the default runtime dtype for exact compact tracing.

Runtime policy:

- optimize on top of the post-fix fp32 default,
- keep fp64 available for parity spot-checks and targeted diagnostics.

### 2.2 What the latest memory probes established

The targeted Phase 4 probe established that:

- the dominant recurring Phase 4 growth is **not** mainly the net footprint of
  `phase4.refresh`,
- it is also **not** mainly the net start→end footprint of
  `context.compute_batch(...)`,
- the strongest signal is still the **batch-aligned row append / writeback path**,
- process file RSS previously tracked the Phase 4 batch buffer size closely,
  which implicated append-side row-store residency.

## 3. Rejected direction (important)

### 3.1 What was tried

A broad row-store rewrite replaced the persistent writable row-store memmap with
explicit offset-based file I/O for **both** writes and reads/materialization.

### 3.2 What happened

That direction was rejected because it was too broad:

- refresh slowed down dramatically,
- Phase 5 also slowed down badly,
- RSS behavior did not improve enough to justify the regression.

### 3.3 Interpretation

The failed attempt showed that the cheap existing read path matters a lot.

Key lesson:

> the current safe fix must target **append/writeback residency**, while
> **preserving the cheap read/materialization path** used by refresh and Phase 5.

## 4. Current problem framing

### 4.1 What this spec now assumes

The current best explanation is:

1. Phase 4 batch expansion produces a large row block,
2. append/writeback makes more of the row-store backing hot/resident,
3. that batch-aligned growth drives the bad file-backed/cgroup pattern,
4. broad replacement of the read path is too expensive.

### 4.2 What this changes strategically

Earlier versions of the plan over-rotated toward either refresh-first work or a
broad storage rewrite.

New priority order:

1. fix **append/writeback residency** safely,
2. preserve the cheap read path,
3. only after the current memory issue is contained, return to broader scheduler
   and storage redesign ideas.

## 5. Chosen current approach

Choose a **hybrid row-store** fix.

Decision summary:

- keep or restore the cheap mmap-style read/materialization path,
- change only the append/writeback path so Phase 4 no longer relies on a
  long-lived writable full-file mapping staying hot across batches,
- keep the change narrow and reversible,
- defer scheduler/frontier redesign ideas.

Rejected for the current phase:

- full read/write file-I/O replacement,
- two-tier storage as the immediate fix,
- cheaper scheduler indexes/scores as the immediate fix,
- incremental refresh-state redesign as the immediate fix.

## 6. Design principles and invariants

### 6.1 Preserve exact semantics

- no approximation,
- no frontier-policy change,
- no queue-semantics change,
- no row-to-node mapping drift,
- no normalization-regression path back toward the old fp32 collapse behavior.

### 6.2 Optimize the proven bad path only

- target the append/writeback residency problem directly,
- do not destroy the cheap read path while fixing writes,
- keep secondary staging/cache cleanup separate enough that the main effect stays
  measurable.

### 6.3 Optimize for operational success

- lower cgroup-visible spikes,
- preserve throughput as much as possible,
- prefer the narrowest reversible change that attacks the proven bad path.

### 6.4 Preserve validation expectations

- fp32/fp64 compact artifacts remain identical on `828_base`, `361_base`, and
  `94_base`,
- retained edges remain `20000`,
- decoder-load counts remain stable,
- `94_base` remains the anomaly/watch gate.

## 7. Code map for the current fix

### 7.1 Primary current target

- `../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
  - `_FileBackedFeatureRowStore.append_rows(...)`
  - `_FileBackedFeatureRowStore` writeback/invalidation mechanics
  - surrounding Phase 4 post-`compute_batch` row staging / append flow

### 7.2 Paths to preserve

- `../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
  - `read_feature_rows(...)`
  - `materialize_dense_feature_slice(...)`

Those read/materialization paths are explicitly part of the **do not regress**
surface.

### 7.3 Validation/test files

- `../../circuit-tracer_chunked/tests/test_attribute_nnsight_telemetry.py`
- `../../circuit-tracer_chunked/tests/test_partial_influences.py`
- `../../circuit-tracer_chunked/tests/test_chunked_decoder_optimizations.py`

## 8. Implementation workstreams

### Workstream 1 — hybrid append/writeback fix

Objective:

Reduce append-side file-backed residency growth without replacing the read path.

Target shape:

- append/writeback uses a narrower mechanism than a long-lived writable full-file
  mmap,
- reads/materialization remain cheap and mmap-style,
- if needed, the read view is reopened/invalidated after append to keep the model
  simple and exact.

### Workstream 2 — validate before secondary cleanup

Objective:

Prove the hybrid cut improves memory behavior without the broad read-path
regression seen in the rejected attempt.

Questions to answer:

1. does batch-aligned file RSS growth shrink materially,
2. does refresh wall time stay near the earlier probe baseline,
3. does Phase 5 wall time stay sane,
4. do outputs remain exact.

### Workstream 3 — secondary cleanup after validation

Deferred until after the main hybrid fix is validated:

- row-store read-cache tuning,
- CPU staging buffer shrink/reset,
- smaller post-batch staging hygiene.

## 9. Later scheduler-redesign considerations

These are **not** planned standalone follow-up work for the current memory fix.

Do not pursue them now as separate optimization tracks. Only revisit them if a
later **scheduler/frontier redesign** makes them useful, and only in that wider
context where we are explicitly reasoning about graph exploration behavior and
the memory operations induced by the scheduler.

Ideas to keep in mind later, not now:

- two-tier hot/cold row storage,
- cheaper scheduler indexes or metadata-only summaries,
- incremental refresh state,
- approximate ranking/prefilter schemes,
- larger storage-layout redesigns coordinated with scheduler work.

These ideas remain context-dependent and are **not** the current safe fix.

## 10. Recommended near-term sequence

1. keep the telemetry improvements,
2. implement the append-side-only hybrid writeback change,
3. preserve/restore cheap reads/materialization,
4. run lightweight local tests,
5. rerun the small Ascend Phase 4 probe on `828_base` and `361_base`,
6. compare:
   - batch-aligned file RSS growth,
   - cgroup/Grafana spike shape,
   - refresh wall time,
   - Phase 5 wall time,
   - exact output invariants,
7. only then decide whether smaller cleanup is needed.

## 11. Acceptance criteria

This strategy is successful when:

1. the repeating Phase 4 batch-aligned file-backed growth is materially reduced,
2. refresh wall time does not suffer a major regression,
3. Phase 5 wall time does not blow up,
4. fp32/fp64 compact artifacts remain identical on the canonical set,
5. retained edges remain `20000`,
6. decoder-load counts remain stable.

## 12. Current recommendation

The next implementation phase should target **the append/writeback path only**,
while preserving the cheap read path.

Best current move:

- fix writeback residency,
- do not broaden the change into read-path replacement,
- push two-tier/scheduler/index ideas into the later optimization pile.
