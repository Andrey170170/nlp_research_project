# Current Implementation Plan — Phase 4 Execution V1

## 1. Problem statement

Planner V1 is the current best Phase 4 scheduler baseline. Planner V2 validated
the instrumentation path for bounded membership-aware selection, but it is not a
runtime win:

- `828_base` Planner V2 vs Planner V1:
  - Phase 4 `+57.47 s` (`+18.3%`),
  - refresh elapsed `173.89 s -> 247.55 s`,
  - weighted edge Jaccard vs V1 `0.955868`.
- `361_base` Planner V2 vs Planner V1:
  - Phase 4 `+129.78 s` (`+22.4%`),
  - refresh elapsed `266.20 s -> 380.78 s`,
  - weighted edge Jaccard vs V1 `0.907005`.

Conclusion: do **not** promote membership-changing Planner V2. Keep it as an
experimental mode, but move the next optimization effort away from frontier
membership changes and toward Phase 4 execution mechanics.

Immediate goal:

> implement and validate two independent execution-side optimization tracks:
> **refresh optimization** and **streaming row execution**, then test each alone
> and together against the Planner V1 baseline.

This effort should be named **Phase 4 Execution V1**, not Planner V3. Planner V1
remains the scheduler; the new work changes refresh and row-execution internals
under that scheduler.

## 2. Current baseline facts

Use Planner V1 as the direct comparison baseline.

### `828_base` Planner V1

- scenario duration: `603.81 s`
- completion duration: `563.91 s`
- attribution time: `562.51 s`
- Phase 0 wall-clock: `94.49 s`
- Phase 3 wall-clock: `142.45 s`
- Phase 4 wall-clock: `314.78 s`
- Phase 4 refresh elapsed: `173.89 s`
- Phase 4 feature-batch elapsed sum: `140.67 s`
- refreshes / batches: `9 / 33`
- active features: `2993540`
- retained edges: `20000`

### `361_base` Planner V1

- scenario duration: `998.25 s`
- completion duration: `957.78 s`
- attribution time: `956.02 s`
- Phase 0 wall-clock: `135.67 s`
- Phase 3 wall-clock: `223.61 s`
- Phase 4 wall-clock: `580.01 s`
- Phase 4 refresh elapsed: `266.20 s`
- Phase 4 feature-batch elapsed sum: `313.59 s`
- refreshes / batches: `9 / 34`
- active features: `5223267`
- retained edges: `20000`

Implications:

- Phase 4 is still the biggest cost.
- Phase 3 is also large enough to matter.
- Inside Phase 4, refresh/ranking is a first-class target, not just batch compute.
- Streaming execution may help both Phase 3 and Phase 4 if it reduces dense block
  materialization, CPU/GPU synchronization, and row-writeback stalls.

## 3. Working assumptions

- fp32 exact compact tracing remains the default.
- The hybrid row-store append/writeback fix remains the accepted memory baseline.
- Planner V1 remains the scheduler baseline for this work.
- Planner V2 remains available but is not used for baseline validation.
- This work is within-trace optimization, not cross-trace or prefix reuse.
- No silent approximation on the exact compact path.
- No GPU/model execution outside SLURM.

## 4. Scope

### In scope

Two independent flags / implementation tracks:

1. **Refresh optimization V1**
   - reduce Phase 4 refresh/ranking elapsed time,
   - preserve selected frontier semantics initially,
   - improve or instrument row-store read / partial influence computation,
   - produce refresh-specific telemetry explaining wins/regressions.

2. **Streaming row executor V1**
   - investigate and implement a row-execution path that streams attribution row
     chunks through compute → normalization/sparsification → CPU/writeback more
     continuously than the current batch-block flow,
   - target Phase 4 first, but keep Phase 3 compatibility in mind because Phase 3
     uses similar row-producing attribution mechanics,
   - preserve exact compact outputs unless a deliberate experimental mode says
     otherwise.

Validation should test:

1. refresh-only,
2. streaming-only,
3. refresh + streaming together.

### Non-goals

- Do not call this Planner V3.
- Do not change frontier membership/ranking as part of this plan.
- Do not promote Planner V2.
- Do not start with `361_late`; use base prompts first.
- Do not run a broad hidden-knob or decoder-cache sweep as part of the first pass.
- Do not rewrite the entire row-store read/write subsystem unless a narrow
  streaming interface proves it is necessary.

## 5. Proposed flags and run naming

Keep the scheduler mode separate from execution flags.

Scheduler baseline:

- `phase4_scheduler_mode=planner_v1`

New execution flags:

- `phase4_refresh_optimization=off|v1`
- `phase4_row_executor=batched|streaming_v1`

Test matrix:

| Variant | Scheduler | Refresh opt | Row executor |
|---|---|---|---|
| baseline | `planner_v1` | `off` | `batched` |
| refresh-only | `planner_v1` | `v1` | `batched` |
| streaming-only | `planner_v1` | `off` | `streaming_v1` |
| combined | `planner_v1` | `v1` | `streaming_v1` |

Run / scenario family names:

- `phase4_refresh_opt_v1`
- `phase4_streaming_executor_v1`
- `phase4_refresh_streaming_v1`

Example scenario names:

- `ascend_fast_828_base_phase4_refresh_opt_v1_fp32_b256_c4096_cache0`
- `ascend_fast_828_base_phase4_streaming_executor_v1_fp32_b256_c4096_cache0`
- `ascend_fast_828_base_phase4_refresh_streaming_v1_fp32_b256_c4096_cache0`
- same for `361_base`.

## 6. Track A — Refresh optimization V1

### 6.1 Problem

Planner V1 refresh time is large:

- `828_base`: `173.89 s`
- `361_base`: `266.20 s`

Planner V2 made refresh roughly `42–43%` slower, which confirms refresh work is
sensitive and can dominate any batch-execution gain.

### 6.2 Likely code surfaces

Library repo:

- `circuit_tracer/attribution/attribute_nnsight.py`
  - Phase 4 refresh loop,
  - planner/refresh telemetry,
  - row-store read and influence-ranking call sites.
- `circuit_tracer/graph.py`
  - `compute_partial_feature_influences_streaming(...)`,
  - chunk-cache / row-reader behavior if present.

### 6.3 Initial implementation targets

Start with measurement and a narrow exact optimization, not a policy change.

Candidate refresh work:

1. expose refresh substage telemetry:
   - row-store read elapsed,
   - partial influence compute elapsed,
   - rank/top-k elapsed,
   - normalization/denominator read elapsed if relevant,
   - number of rows and chunks touched,
   - chunk-cache hit/miss/store counters.
2. enable or improve solver-local row-chunk reuse during refresh if existing
   `compute_partial_feature_influences_streaming(...)` support allows it.
3. avoid redundant row-store reads within the same refresh / adjacent refresh when
   exact immutable chunks are reused.
4. keep selected-frontier membership identical to Planner V1 for the first
   refresh-opt implementation.

### 6.4 Acceptance criteria for refresh-only

- Compact outputs match Planner V1 exactly or near-exactly:
  - feature Jaccard `1.0`,
  - edge Jaccard ideally `1.0`,
  - weighted edge Jaccard ideally `1.0`.
- Refresh elapsed improves measurably:
  - target first-pass win: at least `10%` refresh-time reduction on one prompt,
  - no more than `5%` refresh-time regression on the other.
- Phase 4 total improves or stays roughly neutral.
- Memory / CUDA peak does not materially regress.
- Telemetry explains which refresh substage changed.

## 7. Track B — Streaming row executor V1

### 7.1 Problem

Current Phase 3 / Phase 4 attribution works in row batches. A batch can
conceptually produce a dense block over all active features. For `361_base`, a
`256 x 5.2M` fp32 block is about `5.3 GiB` before autograd state, decoder chunks,
normalization, staging, and writeback. Batching is required, but the current
batch-block lifecycle may force avoidable materialization and synchronization.

### 7.2 Design direction

Explore a streaming executor path that breaks attribution row production into
smaller chunks:

```text
for planned feature batch/group:
    compute attribution row chunk on GPU
    normalize / sparsify chunk or partial row exactly
    asynchronously stage/copy/write chunk to CPU row store
    continue without waiting for the whole dense batch block when safe
```

This is not a guarantee that everything can be one continuous GPU stream. Autograd
and replay constraints may still impose batch boundaries. The goal is to reduce
peak dense intermediate size and CPU/GPU stalls where the current code already
chunks internally.

### 7.3 Likely code surfaces

Library repo:

- `circuit_tracer/attribution/context_nnsight.py`
  - `compute_batch(...)`,
  - `_compute_chunked_feature_attributions_from_grads(...)`,
  - chunked feature replay and decoder chunk loops.
- `circuit_tracer/attribution/attribute_nnsight.py`
  - Phase 3 / Phase 4 batch execution,
  - row normalization / denominator computation,
  - append/writeback integration.
- `circuit_tracer/transcoder/cross_layer_transcoder.py`
  - decoder chunk access / caching interactions.

### 7.4 Initial implementation targets

Start with a narrow executor mode, not a full rewrite.

Candidate streaming work:

1. identify where `compute_batch(...)` already has internal chunked outputs or
   can expose row chunks without changing math,
2. add a `streaming_v1` path that streams chunk outputs into existing exact row
   normalization/writeback helpers,
3. keep the existing `batched` path untouched as fallback,
4. emit executor telemetry:
   - chunk count,
   - chunk rows/features,
   - GPU compute elapsed,
   - CPU staging/copy elapsed,
   - denominator / normalization elapsed,
   - row append/writeback elapsed,
   - peak memory before/after,
   - sync/wait time if measurable.

### 7.5 Acceptance criteria for streaming-only

- Compact outputs remain acceptable vs Planner V1 baseline:
  - target exact equality if the math is unchanged,
  - any drift must be explained before further use.
- Phase 4 feature-batch/executor elapsed improves measurably on at least one
  prompt without increasing refresh time.
- Phase 3 timing is not worsened; if the executor is also used in Phase 3, record
  Phase 3-specific effects.
- Peak CUDA memory should improve or remain neutral.
- Existing `batched` mode remains available and tested.

## 8. Combined validation

After refresh-only and streaming-only are individually validated, run the combined
variant:

- `phase4_refresh_optimization=v1`
- `phase4_row_executor=streaming_v1`
- `phase4_scheduler_mode=planner_v1`

The combined run must be evaluated separately because the optimizations can
interact:

- refresh optimization may change row-read/cache behavior that streaming also
  touches,
- streaming may change memory pressure and alter refresh/runtime side effects,
- the combined win may be less than the sum of individual wins.

Acceptance criteria for combined:

- no compact-output regression beyond individually accepted behavior,
- Phase 4 total should beat Planner V1 and both single-feature variants, or at
  least explain why one component suppresses the other,
- memory behavior must not regress materially.

## 9. Project plumbing and metadata

Project repo changes likely needed:

- `trace_pipeline_chunked.py`
  - CLI flags for `phase4_refresh_optimization` and `phase4_row_executor`,
  - run config / completion manifest metadata,
  - preserve requested/effective execution-mode fields if the library reports
    fallback.
- `experiments/exact_trace_bench/config.py`
  - defaults: refresh `off`, row executor `batched`.
- `experiments/exact_trace_bench/scenarios.py`
  - include new exact-mode keys.
- `experiments/run_sparsification_experiment.py`
  - pass flags to `trace_pipeline_chunked.py`.
- `experiments/exact_trace_bench/extract.py`
  - extract refresh/executor fields and timing summaries.
- `experiments/extract_benchmark_index.py`
  - legacy index metadata support if still needed.
- `tests/test_cross_cluster_debug_artifacts.py`
  - round-trip new execution metadata.

Library metadata should record:

- `phase4_scheduler_mode`, still usually `planner_v1`,
- `phase4_refresh_optimization_requested`,
- `phase4_refresh_optimization_effective`,
- `phase4_row_executor_requested`,
- `phase4_row_executor_effective`,
- fallback reason if either execution mode falls back,
- refresh/executor telemetry detail level.

## 10. Local validation

Run only lightweight checks locally.

Library examples:

- `uv run pytest tests/test_chunked_decoder_optimizations.py -k 'phase4 or refresh or executor or scheduler'`
- `uv run pytest tests/test_attribute_nnsight_telemetry.py -k 'phase4 or telemetry'`
- `uv run ruff check circuit_tracer/attribution/attribute_nnsight.py circuit_tracer/attribution/context_nnsight.py circuit_tracer/graph.py tests/test_chunked_decoder_optimizations.py tests/test_attribute_nnsight_telemetry.py`

Project examples:

- `uv run python tests/test_cross_cluster_debug_artifacts.py`
- `uv run ruff check trace_pipeline_chunked.py experiments tests`

No model loading or GPU work outside SLURM.

## 11. Cluster validation plan

### First validation matrix

Run on Ascend fast tier:

- `828_base`
- `361_base`

Use the same canonical settings as Planner V1:

- `phase4_scheduler_mode=planner_v1`
- `exact_trace_internal_dtype=fp32`
- `feature_batch_size=256`
- `decoder_chunk_size=4096`
- `cross_batch_decoder_cache_bytes=0`
- `max_feature_nodes=8192`
- `max_edges=20000`
- `attribution_update_interval=4`

Variants:

1. refresh-only:
   - `phase4_refresh_optimization=v1`
   - `phase4_row_executor=batched`
2. streaming-only:
   - `phase4_refresh_optimization=off`
   - `phase4_row_executor=streaming_v1`
3. combined:
   - `phase4_refresh_optimization=v1`
   - `phase4_row_executor=streaming_v1`

Compare against existing Planner V1 outputs:

- `ascend_fast_828_base_phase4_planner_v1_fp32_b256_c4096_cache0`
- `ascend_fast_361_base_phase4_planner_v1_fp32_b256_c4096_cache0`

### Validation metrics

For each variant/prompt:

- completion text,
- active feature count,
- retained edge count,
- compact feature Jaccard,
- compact edge Jaccard,
- weighted edge Jaccard,
- Phase 0 / Phase 3 / Phase 4 wall-clock,
- Phase 4 refresh elapsed,
- Phase 4 feature-batch/executor elapsed,
- row-store read/write counts,
- decoder load/cache counters,
- peak RSS / CUDA memory,
- fallback counts/reasons for refresh and executor modes.

## 12. Acceptance criteria

Phase 4 Execution V1 should be considered successful if:

1. execution flags are explicit and default to current behavior,
2. Planner V1 baseline behavior remains available and unchanged,
3. refresh-only and streaming-only can be evaluated independently,
4. compact outputs remain exact or any drift is clearly explained,
5. at least one variant gives a clear Phase 4 win on `828_base` or `361_base`,
6. no variant introduces unacceptable memory regression,
7. telemetry makes wins/regressions attributable to refresh, executor, row-store,
   decoder-load, or synchronization behavior.

Promotion bar:

- Do not make either execution flag default until both canonical prompts pass and
  the combined behavior is understood.
- A single-prompt win is enough to continue iteration, but not enough for default
  rollout.

## 13. Risks and open questions

- **Refresh optimization ceiling:** refresh may be dominated by unavoidable dense
  row-store scans; telemetry must prove where time goes before deeper refactors.
- **Streaming executor complexity:** autograd/replay may not allow fully continuous
  GPU streaming; a chunked executor may still need batch boundaries.
- **Output exactness:** streaming normalization/writeback must preserve exact row
  semantics. Any change in normalization order or sparsification can create drift.
- **Interaction effects:** refresh caching and streaming may compete for memory or
  alter row-read patterns.
- **Phase 3 coupling:** streaming changes may help or hurt Phase 3 differently from
  Phase 4; keep phase-specific telemetry.
- **Scope creep:** do not let this become a broad storage-layout rewrite unless the
  narrow streaming interface clearly requires it.

## 14. Immediate next steps

1. Implement project/library plumbing for the two execution flags with defaults
   preserving current Planner V1 behavior.
2. Add refresh substage telemetry before changing refresh logic.
3. Implement refresh-only optimization V1 and validate locally with unit tests.
4. Design the narrow streaming row executor interface from current
   `compute_batch(...)`/chunked attribution internals.
5. Implement streaming-only behind `phase4_row_executor=streaming_v1`.
6. Generate Ascend fast scenario files for refresh-only, streaming-only, and
   combined validation.
