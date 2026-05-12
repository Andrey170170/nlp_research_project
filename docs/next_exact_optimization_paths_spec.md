# Next Exact Optimization Paths Spec

## 1. Problem statement

The current exact compact path is validated and operational, but it is still too
slow for the intended workload. The working target is roughly **~3 minutes per
token** for practical exact tracing, which means we likely need cumulative wins
across the main expensive stages rather than one isolated fix.

Current durable baseline assumptions:

- the overflow fix is validated,
- fp32 is the correct default exact-trace internal dtype,
- the Phase 4 hybrid row-store append/writeback fix is **good enough for now**,
- no silent approximation should be introduced on the main exact path,
- Phase 4 frontier membership/ranking changes remain correctness-sensitive,
- `94_base` should no longer be treated as a special anomaly prompt from the old
  row-sum bug story,
- instead, cross-cluster **parity** should be treated as the remaining watch
  condition across the canonical prompt set,
- this spec is for **within-trace** optimization, not cross-trace reuse.

This spec ranks the next plausible optimization paths using three axes:

1. **implementation complexity**,
2. **correctness risk**,
3. **expected speed gain**.

Scale used below:

- **Low** = bounded local work / easy rollback
- **Medium** = multi-file or nontrivial validation burden
- **High** = invasive design work or large correctness surface

## 2. Scope / non-goals

### In scope

- exact-only optimization paths for the current chunked tracing stack,
- improvements aimed mainly at **Phase 0**, **Phase 3**, and **Phase 4**,
- smaller end-to-end cleanup if it removes meaningful non-attribution tail time,
- scheduler/frontier redesign as a **possible optimization track**, provided it
  is feature-gated, heavily instrumented, and evaluated with stronger
  parity/correctness gates before default rollout,
- options already suggested by existing docs/specs/TODOs and supported by the
  current code structure.

### Non-goals

- no silent approximation on the primary exact path,
- no cross-trace / cross-step prefix reuse work in this spec,
- no reopening of the rejected broad row-store read/write rewrite,
- no blind default rollout of the scheduler/frontier without flags, rich
  telemetry, and a parity-oriented evaluation plan,
- no broad artifact-schema changes,
- no GPU execution outside SLURM.

## 3. Current code map for the next optimization phase

Primary code surfaces implicated by the current options:

### Project repo

- `trace_pipeline_chunked.py`
  - multistep orchestration
  - exact-mode knob plumbing
  - per-step repeated Phase 0/3/4 invocation

### Library repo

- `circuit_tracer/attribution/attribute.py`
  - `attribute_phase0_stats(...)`
- `circuit_tracer/attribution/attribute_nnsight.py`
  - `_reorder_pending_for_phase4_locality(...)`
  - `_plan_phase4_feature_batch_size_preflight(...)`
  - main Phase 3 / Phase 4 orchestration
  - file-backed compact row-store integration
- `circuit_tracer/attribution/context_nnsight.py`
  - `compute_batch(...)`
  - `_compute_chunked_feature_attributions_from_grads(...)`
  - `materialize_encoder_vectors(...)`
  - `_prepare_error_vector_window(...)`
  - `get_error_vectors_for_layer(...)`
- `circuit_tracer/transcoder/cross_layer_transcoder.py`
  - `create_decoder_block_cache(...)`
  - `get_decoder_chunk(...)`
  - `compute_reconstruction_chunked(...)`
- `circuit_tracer/graph.py`
  - `compute_partial_feature_influences_streaming(...)`

## 4. Ranked candidate optimization paths

### 4.1 Summary ranking table

| Priority | Path | Main phases | Main files | Complexity | Correctness risk | Expected gain | Why it matters |
|---|---|---|---|---|---|---|---|
| 1 | **Phase 4 locality / batch-shaping v2** | 4 | `attribute_nnsight.py`, `context_nnsight.py` | Medium | Medium | Medium-High | Better source-layer × decoder-chunk locality can convert existing replay/cache machinery into real wall-clock savings without changing frontier membership. |
| 2 | **Scheduler/frontier redesign (feature-gated live path)** | 4 | `attribute_nnsight.py`, `graph.py`, Phase 4 refresh/frontier logic | High | High | High | This is a real possible optimization path now; run it as an explicit experimental mode with rich telemetry and parity/correctness gates before default rollout. |
| 3 | **Refresh / solver-local chunk reuse** | 4 refresh | `graph.py`, `attribute_nnsight.py` | Medium | Low-Medium | Medium | `compute_partial_feature_influences_streaming(...)` already has optional chunk reuse machinery; current exact path appears to leave this mostly untapped. |
| 4 | **Phase 0 within-trace materialization / reconstruction cleanup** | 0 | `cross_layer_transcoder.py`, `attribute.py`, setup path | Medium | Low-Medium | Medium | If Phase 0 still matters, the relevant wins are within-trace load/materialization/reconstruction improvements, not cross-trace reuse. |
| 5 | **CPU staging / denominator-pass cleanup** | 3, 4 | `attribute_nnsight.py`, `context_nnsight.py` | Medium | Low | Low-Medium | Per-batch host copies and denominator work are on the hot path; not the whole answer, but a useful additive cleanup. |
| 6 | **Decoder-cache tuning** | 4, partial 0 | `cross_layer_transcoder.py`, `context_nnsight.py`, scenario plumbing | Low | Low | Medium | Current canonical runs often use `cross_batch_decoder_cache_bytes=0`; decoder loads remain an obvious reuse opportunity, but the sweep is resource-expensive and better used later for fine-tuning. |
| 7 | **Exact hidden-knob sweep + fixed policy** | 3, 4 | `trace_pipeline_chunked.py`, `attribute_nnsight.py`, `context_nnsight.py` | Low | Low | Medium | Useful for final tuning, but better postponed until a few concrete implementation wins land and cluster pressure is lower. |
| 8 | **Outer pipeline / finalization tail cleanup** | end-to-end | `trace_pipeline_chunked.py`, save/cleanup path | Low-Medium | Low | Low-Medium | Some recent runs show a non-attribution tail; worth trimming after core wins, not before. |
| D | **Still-deferred research tracks** | mostly 4 | storage layout, approximation, incremental scheduler state | High | High | High but uncertain | Potentially large upside, but too invasive or too approximation-heavy for the current exact path. |

### 4.2 Candidate details

#### Path 1 — Phase 4 locality / batch-shaping v2

**Problem:** `_reorder_pending_for_phase4_locality(...)` is already a first pass,
but the main replay loop in `context_nnsight.py` still walks nested
source-layer → decoder-chunk → output-layer → row-subchunk structure. Better
batch shaping may increase decoder reuse and reduce wasted replay churn.

**Proposed direction:**

- preserve frontier membership exactly,
- keep ranking semantics unchanged,
- improve execution grouping by source layer and decoder chunk,
- consider whether pending frontier slices should be shaped to align better with
  decoder chunk boundaries and row-subchunks,
- benchmark only after the knob and decoder-cache sweeps establish a baseline.

**Why ranked first now:** this is a direct implementation exploration path with
real upside and does not require an expensive cluster-wide sweep up front.

#### Path 2 — Scheduler/frontier redesign (feature-gated live path)

**Problem:** the current Phase 4 refresh/frontier process may leave performance
on the table, and broader cross-cluster parity drift suggests we may want more
visibility into how frontier construction itself affects both runtime and overlap.

**Proposed direction:**

- treat scheduler/frontier redesign as a **real optimization candidate**, not as
  something permanently out of scope,
- implement a live experimental scheduler path behind an explicit mode flag,
- make the path richly instrumented enough to diagnose regressions after small
  validation runs,
- compare candidate frontier policies using:
  - frontier overlap,
  - selected-node overlap,
  - influence-rank correlation,
  - Ascend/Cardinal parity metrics,
  - wall-clock and refresh-time impact,
- only promote a redesign into the default production path if it clears explicit
  correctness and parity gates.

**Why ranked second:** this may be one of the larger remaining speed levers, but
it has the highest correctness surface of the non-approximate options.

#### Path 3 — Refresh / solver-local chunk reuse

**Problem:** `compute_partial_feature_influences_streaming(...)` already exposes
`chunk_cache_max_bytes`, but current call sites do not appear to treat solver-local
chunk reuse as an active optimization track.

**Proposed direction:**

- enable and tune solver-local row-chunk reuse for refresh / seed ranking,
- align row-reader chunk boundaries with reuse windows where possible,
- measure read-call reductions and refresh wall-time improvements,
- keep this strictly exact: cache only immutable row chunks with clear invalidation.

**Why ranked third:** it is an implementation-first optimization with plausible
benefit and lower cluster cost than broad tuning sweeps.

#### Path 4 — Phase 0 within-trace materialization / reconstruction cleanup

**Problem:** if Phase 0 is still a meaningful contributor, the relevant next work
for this spec is not cross-trace reuse, but rather within-trace setup cost:
encoder loading, decoder chunk access, reconstruction chunking, and unnecessary
materialization.

**Relevant surfaces:**

- `attribute_phase0_stats(...)`
- `compute_reconstruction_chunked(...)`
- lazy encoder materialization and setup-time staging paths

**Proposed direction:**

- keep the focus on exact within-trace setup cost,
- reduce unnecessary materialization during sparse setup and reconstruction,
- tune or restructure Phase 0 loading/reconstruction only where semantics remain
  unchanged,
- keep this separate from cross-step prefix reuse work.

**Why mid priority:** this fits the within-trace scope and may matter, but the
strongest current evidence still points to Phase 4 as the dominant lever.

#### Path 5 — CPU staging / denominator-pass cleanup

**Problem:** the Phase 3 / Phase 4 path still does substantial CPU staging and
row-denominator handling around appended row blocks.

**Relevant surfaces:**

- `_copy_rows_to_cpu_staging(...)`
- `_compute_row_denominator_scaled_l1(...)`
- append/writeback flow around the file-backed compact row store

**Proposed direction:**

- reduce redundant host copies,
- shrink/reset reusable staging buffers more aggressively when safe,
- avoid extra denominator passes if the exact same result can be produced during
  staging or append.

**Why here:** likely useful, but still more of an additive cleanup than a primary
speed lever.

Follow-up emphasis after the hidden-knobs matrix:

- A more ambitious version of this path is **GPU-side denominator / row-summary
  computation**:
  - compute scaled row-L1 denominators while rows are still on GPU,
  - optionally compute small exact row summaries needed for diagnostics or future
    refresh planning,
  - copy rows plus denominator summaries to CPU once.
- This is attractive because Phase 3/4 currently produce rows on GPU, copy them to
  CPU, and then scan very wide row slices for denominator statistics.
- Correctness risk is not approximation, but **floating-point reduction order**:
  GPU reductions can perturb refresh scores enough to change frontier ties or
  retained edges. First implementation should therefore support a shadow/parity
  mode that computes both CPU and GPU denominators and records max/mean absolute
  differences, frontier overlap, and compact-output equality before any promotion.
- This path is explicitly desired for a later optimization pass; it is not the
  selected candidate for the immediate combined array because the next array
  should minimize live-path correctness risk.

#### Path 6 — Decoder-cache tuning

**Problem:** replay repeatedly loads decoder chunks via
`get_decoder_chunk(...)`; cache support exists, but current validation probes
often intentionally ran with `cross_batch_decoder_cache_bytes=0`.

**Proposed direction:**

- sweep nonzero `cross_batch_decoder_cache_bytes`,
- measure decoder load counts, cache hit/miss/eviction behavior, and wall time,
- pair this with locality-aware ordering rather than treating cache tuning in
  isolation.

**Why later:** it is a low-risk, code-supported optimization path, but the sweep is
resource-expensive and better used for fine-tuning after implementation-first
changes land.

Current update:

- The immediate next combined array will include a **single bounded re-test** of
  this path on top of `active_encoder_cpu + rowstore_fadvise`.
- Historical context: earlier cross-batch decoder-cache experiments suggested
  `361_base` may require a large budget (`~8-12 GiB`) and did not obviously help
  enough. Those results predated the current post-overflow-fix fp32 path and the
  active-encoder/rowstore improvements, so a limited retest is justified.
- Keep the retest explicit and conservative:
  - use `cross_batch_decoder_cache_bytes=<fixed budget>`, not a global default,
  - prefer one budget in the next shared array rather than a sweep,
  - treat high eviction churn, OOM, or <~5% wall-clock gain as falsification for
    this path as the next default-combo candidate.
- Required telemetry remains decoder cache hits/misses/evictions/skips/resident
  bytes, decoder load count/time, CUDA peak reserved, sacct MaxRSS, and compact
  parity vs the combo baseline.

#### Path 7 — Exact hidden-knob sweep, then fixed policy

**Problem:** the exact path already exposes multiple replay/staging knobs, but
they are not yet treated as a disciplined tuning surface.

**Relevant existing knobs:**

- `chunked_feature_replay_window`
- `error_vector_prefetch_lookahead`
- `stage_encoder_vecs_on_cpu`
- `stage_error_vectors_on_cpu`
- `row_subchunk_size`

These are already plumbed through `trace_pipeline_chunked.py`,
`attribute_nnsight.py`, and `context_nnsight.py`.

**Proposed direction:**

- run a small exact benchmark matrix on `828_base` and `361_base`,
- keep `94_base` in the canonical set,
- use cross-cluster overlap/parity metrics as the main watch condition,
- choose one default policy for the fast inner loop instead of retuning ad hoc.

**Why later:** this is still valuable, but the sweep is resource-expensive and is
best used as a final tuning pass after implementation-first changes produce a
better baseline.

#### Path 8 — Outer pipeline / finalization tail cleanup

**Problem:** some recent telemetry suggests part of the remaining wall time may be
outside the instrumented attribution phases.

**Proposed direction:**

- profile the post-attribution tail in `trace_pipeline_chunked.py`,
- separate artifact-save, cleanup, and filesystem tail costs,
- trim only if the tail proves stable and nontrivial.

**Why late:** this is worthwhile housekeeping, but the biggest missing wins are
still in Phase 0 / Phase 4.

### 4.3 Still-deferred research tracks

These should remain explicitly **deferred** unless promoted into a separate
research branch/spec:

- incremental refresh state,
- cheaper scheduler indexes / metadata-only ranking summaries,
- two-tier hot/cold row storage,
- approximate ranking / prefilter schemes,
- larger storage-layout redesigns coordinated with scheduler work.

These ideas may eventually matter, but they are still too correctness-sensitive,
too invasive, or too approximation-heavy for the near-term exact path.

## 5. Recommended execution order

### 5.0 Current execution addendum — Ascend-first algorithmic path

The earlier ordering below is still useful as a map of plausible exact
optimization tracks, but the current execution focus should be narrower:

- prioritize **Ascend/A100** work first; defer Cardinal/H100 capacity pushing
  until the algorithmic and Phase 0/handoff work has landed, because Cardinal job
  availability is currently a bottleneck,
- use `361_base` as the hard normal prompt and `828_base` as the nicer normal
  prompt,
- treat late fixtures as feasibility/stress tests, not the fast inner-loop:
  - `828_late` completes but is near RAM limit,
  - `361_late` currently CUDA-OOMs in Phase 1,
- remember that `361_base` Phase 3/4 already sits near the 40 GB A100 VRAM limit
  (about `38 GB`), while `828_base` has more headroom (about `32 GB`),
- avoid persistent large GPU caches through Phase 3/4 on A100 unless they replace
  existing allocations,
- row-store memmap/page-cache growth can hammer the RAM/cgroup limit and cause
  large slowdowns, so stability against cache pressure is part of performance.

Current preferred execution order:

1. **Phase 1 trace-batch decoupling** for late feasibility.
   - Investigate whether Phase 1 unnecessarily expands to the attribution batch
     size during the forward/cache pass.
   - Goal: make `361_late` pass Phase 1 without changing exact graph semantics.
2. **Incremental refresh shadow mode** for algorithmic Phase 4 reduction.
   - This is the highest-upside Ascend-relevant algorithmic path, but must start
     as shadow comparison against Planner V1 refresh outputs/frontiers.
3. **Row-store/page-cache pressure controls** for late-profile stability.
   - Add telemetry and bounded/advisory controls without repeating the rejected
     broad row-store read/write rewrite.
4. **Phase 0 / handoff aggressive temporary or pinned residency.**
   - Phase 0 appears memory-light, so spend memory there where it is freed before
     Phase 3/4, or use pinned CPU residency to avoid repeated safetensor reads.
5. **Only after those:** denominator/GPU row-summary work, layer-depth planner
   refinements, and then Cardinal/H100 capacity tuning.

This addendum does not remove the older paths; it defines the current ordering for
the next execution cycle.

Recommended order for the next optimization cycle:

1. **Hold the current hybrid row-store baseline fixed** while current validation
   jobs finish.
2. **Implement locality / batch-shaping v2** first as the main implementation
   exploration track.
3. **Design scheduler/frontier alternatives in shadow mode first** and compare
   overlap/parity/runtime before any production rollout.
4. **Try solver-local chunk reuse for refresh** and measure refresh-specific wall
   time before and after.
5. **Do Phase 0 within-trace setup cleanup** if Phase 0 still shows meaningful
   share in the updated timings.
6. **Do CPU staging / denominator cleanup** as an additive hygiene pass.
7. **After a few concrete implementation wins land, run a nonzero decoder-cache
   sweep** to fine-tune the new baseline.
8. **Run the exact hidden-knob sweep last** as a final tuning pass once cluster
   pressure is lower and the implementation direction is more stable.
9. **Only then reconsider still-deferred research tracks**, and only under a separate
   spec with stronger correctness gates.

## 6. Acceptance criteria

This optimization-path spec is successful if it gives a clear, defensible
ordering for the next work and keeps correctness boundaries explicit.

An individual path should be considered successful only if all are true:

1. exact compact outputs remain unchanged on canonical checks,
2. cross-cluster parity remains acceptable on the canonical prompt set for Phase
   4 touching changes,
3. timing improves measurably on `828_base` and/or `361_base`,
4. memory behavior does not regress materially,
5. the change is accompanied by focused validation or tests appropriate to its
   risk level.

Project-level success target:

- cumulative accepted changes move exact tracing materially closer to
  **~3 min/token** without relying on approximation.

## 7. Risks and open questions

- **Prompt dependence:** which knob choices generalize beyond a small prompt set?
- **Locality ceiling:** how much speedup remains after the current hybrid
  row-store fix and existing locality reorder?
- **Scheduler redesign proof burden:** what exact overlap/parity criteria should a
  candidate frontier redesign meet before production rollout?
- **Refresh reuse ceiling:** will solver-local chunk reuse pay off enough to
  justify the added complexity?
- **Phase 0 relevance:** after the latest fixes, how much of the remaining wall
  time is still actually in Phase 0 on canonical runs?
- **Tail cost relevance:** is the observed non-attribution tail real and stable,
  or just a run-specific artifact?
- **Deferral discipline:** if near-term paths underperform, when is it justified
  to reopen scheduler/frontier redesign rather than continue incremental tuning?
