# Current Implementation Plan — Phase 4 Frontier Planner V1

## 1. Problem statement

The current Phase 4 locality / batch-shaping work is useful, but it is still an
ad hoc layer on top of the old scheduler loop. While the v2 locality validation
is queued, the next implementation item is to build a real **Phase 4 Frontier
Planner V1**.

Planner V1 is not intended to be the final scheduler optimization. It is the
foundation for later cache-aware, membership-aware, or adaptive policies.

Immediate goal:

> consolidate Phase 4 frontier selection, ordering, batch construction,
> boundary reasons, invariants, flags, and telemetry into one explicit planner
> path while preserving selected-node membership for each refresh.

## 2. Working assumptions

- fp32 exact compact tracing remains the default.
- The hybrid row-store append/writeback fix remains the accepted memory baseline.
- The current locality-shaped path remains the validated fallback until Planner V1
  clears canonical validation.
- This work is within-trace optimization, not cross-trace or prefix reuse.
- This is a live experimental scheduler path, not shadow mode.
- Git rollback is acceptable, but cluster time is not free; therefore diagnostics
  must be rich enough to localize regressions quickly.
- Planner V1 should preserve frontier membership within a refresh; later planner
  versions can relax this deliberately.

## 3. Scope

### In scope

- Add a feature-gated Phase 4 planner mode.
- Consolidate current rank → locality reorder → shaped slicing logic into a
  structured planner result.
- Emit structured planner telemetry and stable scheduler metadata.
- Add invariants for selected-node membership, duplicate/missing nodes, and
  non-advancing batch boundaries.
- Add focused local tests for planner behavior and telemetry shape.
- Prepare canonical fast validation against existing baselines.

Primary library files likely in scope:

- `../worktrees_opt/circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
- `../worktrees_opt/circuit-tracer_chunked/tests/test_chunked_decoder_optimizations.py`
- `../worktrees_opt/circuit-tracer_chunked/tests/test_attribute_nnsight_telemetry.py`

Project files likely in scope if flag plumbing is needed:

- `trace_pipeline_chunked.py`
- scenario-generation helpers under `experiments/`

### Non-goals for Planner V1

- Do not change selected frontier membership as a policy goal.
- Do not introduce approximation on the exact compact path.
- Do not reopen broad row-store read-path rewrite.
- Do not start a large decoder-cache or hidden-knob sweep.
- Do not make Planner V1 the default until canonical validation passes.

## 4. Required flags / metadata

Planner V1 must be selectable without code patches between runs.

Required scheduler mode flag:

- `phase4_scheduler_mode`
  - `locality` = current validated locality-shaped fixed-frontier path,
  - `planner_v1` = new membership-preserving planner path,
  - `legacy` can exist if cheap to preserve, but is not required if it complicates
    the implementation,
  - later reserved modes: `planner_v2`, `adaptive`.

Required debug / telemetry controls:

- `phase4_scheduler_debug` or equivalent,
- optional `phase4_scheduler_telemetry_detail=summary|normal|debug` if the event
  volume needs finer control,
- compatibility with existing `profile`, `compact_output`, `cross_cluster_debug`,
  and `telemetry_max_events`.

Every run should record:

- `phase4_scheduler_mode`,
- `phase4_scheduler_version`,
- `phase4_scheduler_policy`,
- `phase4_scheduler_debug`,
- `phase4_scheduler_telemetry_detail` if present,
- `feature_batch_size`,
- `update_interval`,
- `actual_max_feature_nodes`,
- `total_active_features`,
- `exact_chunked_decoder`,
- `decoder_chunk_size`,
- `cross_batch_decoder_cache_bytes`,
- `exact_trace_internal_dtype`,
- whether membership-preservation checks were enabled.

## 5. Planner V1 design target

Create a small planner surface that takes refresh-time state and returns a plan.

Conceptual inputs:

- unvisited feature rank / candidate indices,
- candidate scores or score lookup,
- visited mask,
- feature metadata: layer, position, feature id, decoder chunk,
- feature batch size,
- update interval,
- selected-feature target and current `n_visited`,
- scheduler mode/config.

Conceptual output:

- selected frontier tensor,
- list of planned feature batches,
- selected membership hash,
- selected order hash,
- locality / fragmentation summary,
- batch-size distribution,
- batch boundary reasons,
- invariant/debug summary,
- telemetry payloads for refresh and batch events.

Planner V1 policy:

1. select the same top unvisited influence candidates as the current locality path
   would select for that refresh,
2. order/group the selected frontier for source-layer and decoder-chunk locality,
3. build shaped batches with explicit boundary reasons,
4. execute planned batches directly in the Phase 4 loop,
5. record plan-level and batch-level telemetry.

## 6. Required telemetry

The durable telemetry requirements live in
`docs/phase4_scheduler_v2_spec.md`. For this implementation pass, make sure the
following minimum set lands:

### Phase summary

- scheduler mode/version/policy,
- selected feature count,
- Phase 4 wall time,
- refresh count,
- feature-batch count,
- total refresh elapsed time,
- selected-node membership/order hashes,
- peak memory snapshots already supported by existing helpers.

### Refresh/plan summary

- refresh index,
- stored row count,
- visited count,
- unvisited candidate count,
- pre-plan frontier size,
- planned frontier size,
- frontier hash and sample,
- candidate score stats / cutoff margin,
- planner decision elapsed time,
- locality fragmentation summary,
- planned batch count and batch-size distribution,
- boundary reason counts.

### Batch summary

- global Phase 4 batch index,
- originating refresh index,
- batch row count,
- batch node hash/sample,
- distinct source layers,
- distinct decoder chunks,
- monotonic chunk-order flag,
- compute-batch elapsed time,
- row append/writeback timing if cheap to expose,
- memory before/after.

### Invariants

- membership-preservation hash check against the current locality-selected
  frontier,
- duplicate selected-node count,
- missing selected-node count,
- non-advancing boundary detection,
- nonfinite score count,
- explicit fallback/abort reason if Planner V1 cannot form a valid plan.

## 7. Implementation sequence

1. Add scheduler mode parsing / validation in the library attribution path.
2. Add a small internal planner result structure and helper functions.
3. Move current locality reorder + shaped batch slicing into Planner V1 while
   preserving current membership semantics.
4. Route the Phase 4 loop through the planner when
   `phase4_scheduler_mode=planner_v1`.
5. Keep the current locality path available as the A/B fallback.
6. Add scheduler metadata to run/phase telemetry.
7. Add refresh-level plan telemetry.
8. Add batch-level plan telemetry.
9. Add invariant checks and clear error/fallback reporting.
10. Add focused tests.
11. Run lightweight local validation only.
12. Prepare small Ascend validation scenarios for `828_base` and `361_base`.

## 8. Local validation

Run only lightweight checks locally:

- `uv run pytest tests/test_chunked_decoder_optimizations.py`
- `uv run pytest tests/test_attribute_nnsight_telemetry.py`
- `uv run ruff check circuit_tracer/attribution/attribute_nnsight.py tests/test_chunked_decoder_optimizations.py tests/test_attribute_nnsight_telemetry.py`

No GPU/model execution outside SLURM.

## 9. Cluster validation after implementation

First validation should compare Planner V1 against current locality/hybrid
baselines on canonical fast prompts:

- `828_base`
- `361_base`

Primary questions:

1. did selected-node membership remain stable by design,
2. did compact outputs / retained edges remain acceptable,
3. did Planner V1 avoid obvious runtime regression,
4. did telemetry explain batch counts, refresh counts, and locality shape,
5. did `828_base` avoid the over-splitting behavior seen in the first locality
   pass,
6. did `361_base` preserve a meaningful Phase 4 improvement.

Do not use `361_late` as the first Planner V1 benchmark; it still looks partly
like a parameter-feasibility / VRAM-planning problem.

## 10. Acceptance criteria

Planner V1 is successful as a foundation when:

1. it runs behind an explicit mode flag,
2. it preserves current selected-frontier membership within a refresh,
3. it emits the required scheduler/plan telemetry,
4. focused local tests pass,
5. canonical fast validation is easy to compare against current baselines,
6. any runtime regression can be localized to candidate scoring, frontier
   grouping, refresh cadence, decoder-load churn, row-store behavior, or memory
   pressure.

Planner V1 does not need to reach the final ~3 minute end-to-end trace target by
itself. Its main job is to make later scheduler policies cheap and safe to test.

## 11. Later work unlocked by Planner V1

After Planner V1 validates, choose the next policy experiment:

1. **bounded membership-aware Planner V2** — choose from a wider high-score window
   to improve cache locality while measuring overlap/score loss,
2. **adaptive refresh planner** — adjust refresh cadence based on cutoff margin,
   fragmentation, useful chunk runs, or memory signals,
3. **hard-prompt planner** — safer long-prefix behavior for cases like
   `361_late`,
4. **decoder-cache tuning** — use Planner V1 telemetry to choose cache sizes more
   intelligently rather than sweeping blindly.

## 12. Current risks / open questions

- Planner V1 may mostly improve code structure and telemetry rather than runtime.
- Rich telemetry may need caps to avoid large artifacts.
- Membership-preserving scheduling may not be enough for the ~3 minute target;
  Planner V2 likely needs bounded membership changes.
- The current locality v2 validation result may slightly change the fallback path
  we compare against, but it should not block Planner V1 implementation.
