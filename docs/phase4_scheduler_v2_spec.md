# Phase 4 Frontier Planner / Scheduler V2 Spec

Status: Proposed/deferred optimization design
Last updated: 2026-05-16

## 1. Problem statement

Phase 4 is still the dominant exact-trace cost after the validated overflow fix,
hybrid row-store fix, and first locality / batch-shaping pass. The current
locality pass improves execution order inside a fixed frontier, but it does not
redesign the frontier scheduler itself.

The next optimization track is a real **Phase 4 scheduler v2** execution path,
not shadow mode. The first implementation step should be **Planner V1**: a
membership-preserving frontier planner that turns the current ad hoc
rank/reorder/slice helpers into a coherent planning surface. This is both a
possible small optimization and the foundation for later membership-aware or
adaptive scheduler policies.

This direct live-path approach is acceptable because this work is experimental,
baselines already exist, and git rollback is cheap. The implementation must,
however, emit enough structured telemetry to make regressions diagnosable after
one small validation run rather than after a long debugging cycle.

## 2. Scope / non-goals

### In scope

- A feature-gated live scheduler path for Phase 4.
- Scheduler changes that affect frontier ordering, grouping, refresh cadence, and
  batch planning.
- Structured metrics that explain scheduler decisions and their runtime effects.
- Validation against existing canonical baselines (`828_base`, `361_base`, and
  targeted parity checks when needed).

### Non-goals for Planner V1

- No silent approximation on the exact compact path.
- No required change to selected frontier membership within a refresh.
- No broad row-store read-path rewrite.
- No cross-trace / prefix reuse work.
- No default-on rollout before artifact and runtime validation.
- No large parameter sweep as the first step.

## 3. Planner V1 foundation

Planner V1 is the first scheduler-v2 implementation step.

Core purpose:

> Replace scattered Phase 4 frontier scheduling logic with one explicit planner
> that owns frontier ordering, batch construction, boundary reasons, invariants,
> and telemetry.

Planner V1 should preserve the current selected frontier membership for a given
refresh. It may change execution order and batch boundaries. Later planner
versions may change selection policy, but Planner V1 should make that future work
localized and measurable.

### 3.1 Current logic to consolidate

Current Phase 4 scheduling is distributed across:

- influence ranking inside the Phase 4 refresh loop,
- `_reorder_pending_for_phase4_locality(...)`,
- `_compute_phase4_locality_shaped_frontier_size(...)`,
- `_compute_phase4_locality_shaped_batch_end(...)`,
- the inner `pending_offset` batch loop,
- refresh / batch telemetry emitted around that loop.

Planner V1 should centralize these responsibilities behind a small planning API.

### 3.2 Planner inputs

The planner should receive only the state needed to build the next executable
frontier plan:

- unvisited feature rank or candidate indices,
- candidate influence scores or score lookup,
- `visited` mask or equivalent selected-node state,
- feature metadata:
  - source layer,
  - token position,
  - feature id,
  - decoder chunk id when available,
- `feature_batch_size`,
- `update_interval`,
- `actual_max_feature_nodes`,
- `n_visited`,
- scheduler mode/config,
- exact/chunked decoder metadata.

### 3.3 Planner outputs

Planner V1 should return a structured plan, not just a tensor:

- selected frontier tensor,
- planned feature batches,
- selected-node membership hash,
- selected-node order hash,
- locality / fragmentation summary,
- batch boundary reasons,
- invariant/debug summary,
- telemetry payload that can be attached to refresh and batch events.

The Phase 4 execution loop should then execute the returned planned batches
directly.

### 3.4 Planner V1 policy

Planner V1 policy should be conservative:

1. choose the same top unvisited influence candidates the current path would
   choose for the refresh,
2. order/group those candidates for source-layer and decoder-chunk locality,
3. build shaped batches with explicit boundary reasons,
4. enforce non-advancing-boundary and duplicate/missing-node checks,
5. record the plan summary before executing batches.

This creates a clean foundation for cache-aware execution without yet changing
the ranking objective.

### 3.5 Later planner stages enabled by V1

After Planner V1 validates, later policies can be implemented by swapping only
the planner policy:

- **Planner V2 — bounded membership-aware selection**:
  choose from a slightly wider high-score candidate window to improve
  layer/chunk locality while tracking selected-node overlap and score loss.
- **Adaptive refresh planner**:
  vary refresh cadence based on cutoff margin, frontier fragmentation, useful
  chunk runs, or memory/runtime signals.
- **Hard-prompt planner**:
  choose safer frontier/batch behavior for long-prefix or high-active-feature
  cases such as `361_late`.

## 4. Required flags and metadata

The scheduler must be selectable without patching code between runs.

Required public flag:

- `phase4_scheduler_mode`
  - allowed values should include at least:
    - `legacy` — pre-locality fixed-frontier behavior if still practical to keep,
    - `locality` — current locality-shaped fixed-frontier behavior,
    - `planner_v1` — membership-preserving planner execution path,
    - `planner_v2` or `adaptive` — reserved names for later selection-policy
      experiments,
  - default should remain the currently validated behavior until Planner V1 clears
    the canonical checks.

Required debug / telemetry controls:

- `phase4_scheduler_debug` or equivalent structured-debug enablement.
- `phase4_scheduler_telemetry_detail`, if needed, with values such as `summary`,
  `normal`, and `debug`.
- compatibility with existing `profile`, `compact_output`,
  `cross_cluster_debug`, and `telemetry_max_events` handling.
- a bounded event policy so debug metrics cannot explode artifact size on long
  prompts.

Every run must record these scheduler-identifying fields in telemetry / compact
metadata:

- `phase4_scheduler_mode`,
- `phase4_scheduler_version`,
- `phase4_scheduler_policy` or short policy name,
- `phase4_scheduler_telemetry_detail`,
- `feature_batch_size`,
- `update_interval`,
- `actual_max_feature_nodes`,
- `total_active_features`,
- `exact_chunked_decoder`,
- `decoder_chunk_size`,
- `cross_batch_decoder_cache_bytes`,
- `exact_trace_internal_dtype`,
- whether membership-preservation checks were enabled.

## 5. Required telemetry

### 4.1 Phase-level summary

Record once per Phase 4 run:

- total selected features,
- total Phase 4 wall time,
- total refresh wall time,
- total feature-batch wall time,
- total row append/writeback wall time if available,
- refresh count,
- feature-batch count,
- final selected-node hash / order hash,
- retained-edge count when available from the outer pipeline,
- peak RSS / CUDA reserved snapshots already available in the existing memory
  helpers.

### 4.2 Refresh-level metrics

Record for each refresh, subject to event caps:

- refresh index,
- stored row count (`st`),
- visited feature count,
- unvisited candidate count,
- frontier size before and after scheduler selection,
- selected frontier hash and small sample,
- influence score stats for candidates,
- cutoff rank / score / next-score margin,
- scheduler decision elapsed time,
- influence recomputation elapsed time,
- row-store read calls / rows / cache hits / misses,
- streaming chunk-cache requests / hits / misses / stores,
- memory before / after.

### 4.3 Frontier locality / fragmentation metrics

For each planned frontier, record enough structure to explain whether scheduler
v2 improved locality:

- layer transition count,
- decoder-chunk transition count,
- number of distinct `(layer, chunk)` groups,
- min / p50 / max group length,
- first and last few group keys,
- planned batch count for the frontier,
- planned batch-size distribution,
- batch boundary reasons, for example:
  - max batch size,
  - layer boundary,
  - decoder-chunk boundary,
  - refresh/update boundary,
  - final frontier tail.

### 4.4 Batch-level metrics

Record for each feature batch, subject to event caps:

- global Phase 4 batch index,
- refresh index that produced the batch,
- batch row count,
- batch node hash and small sample,
- min / max / distinct source layers,
- min / max / distinct decoder chunks,
- whether chunk order is monotonic,
- `context.compute_batch` elapsed time,
- row copy / denominator / append elapsed time if available,
- decoder-load/cache counters from transcoder diagnostics if available,
- memory before / after.

### 4.5 Failure/debug invariants

When debug is enabled, the run should expose enough information to quickly tell
whether a bad result came from scheduling or from downstream attribution work:

- selected-node membership hash before and after v2 scheduling,
- selected-node order hash,
- visited mask consistency checks,
- duplicate selected-node count,
- missing selected-node count relative to requested `actual_max_feature_nodes`,
- non-advancing frontier / batch boundary detection,
- nonfinite influence score count,
- nonfinite row-denominator count,
- explicit scheduler error reason if v2 falls back or aborts.

## 6. Acceptance criteria

Planner V1 is acceptable for experimental validation when:

1. it is selectable by flag and easy to A/B against the current locality path,
2. local focused tests cover planner output, scheduler ordering, batch boundary
   generation, membership preservation, and non-advancing-boundary failure cases,
3. telemetry identifies the scheduler mode/version and records the required
   phase, refresh, frontier, and batch metrics,
4. canonical validation can compare compact artifacts against existing baselines,
5. a regression can be attributed quickly to one of:
   - candidate scoring / frontier selection,
   - frontier grouping / batch shaping,
   - refresh cadence,
   - decoder-load churn,
   - row-store read/write behavior,
   - memory pressure.

## 7. Risks and open questions

- Changing frontier membership is correctness-sensitive. The first v2 pass should
  prefer changing ordering/grouping/cadence before changing the influence-ranking
  objective itself.
- More debug telemetry can increase artifact size; event caps and summary metrics
  are required.
- Better locality can trade off against too many small batches. Batch-boundary
  reason metrics are required so this is visible immediately.
- Scheduler v2 may improve `361_base` but regress `828_base`; canonical fast
  validation must include both prompts before accepting the direction.
