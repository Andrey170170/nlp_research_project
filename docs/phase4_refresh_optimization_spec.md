# Phase 4 Refresh Optimization Spec

## 1. Goal

Document the current understanding of exact-trace optimization status and define
the next optimization directions, with primary focus on **Phase 4 refresh** for
the main fast-loop fixtures:

- `828_base`
- `361_base`

`94_base` remains anomaly/watch-only context and is not the main optimization
target for this spec.

## 2. Current optimization status

### Already landed in the fork

The current optimization branch in `../../circuit-tracer_chunked/` already has:

- lazy encoder materialization
  - [`attribute_nnsight.py`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)
  - [`cross_layer_transcoder.py`](../../circuit-tracer_chunked/circuit_tracer/transcoder/cross_layer_transcoder.py)
- compact exact edge storage / file-backed exact rows
  - [`_FileBackedFeatureRowStore`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)
- fixed-frontier Phase 4 locality reordering
  - [`_reorder_pending_for_phase4_locality`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)
- structured telemetry
  - [`TelemetryRecorder`](../../circuit-tracer_chunked/circuit_tracer/utils/telemetry.py)
  - [`telemetry_gathering.py`](../experiments/telemetry_gathering.py)
- preflight Phase 4 feature-batch planner
  - [`_plan_phase4_feature_batch_size_preflight`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)

### Current strategic stance

- keep the **preflight planner** path
- do **not** return to the old live autoscale path
- treat prompt 94 as a gate/watchpoint, not as the main tuning prompt
- use the newer post-fix telemetry runs, not the earlier truncated telemetry run,
  when reasoning about current hot paths

## 3. Benchmark-backed findings

### 3.1 Locality reorder helped, but only modestly

Ascend fast benchmarks show:

- `828_base`: about `3182s -> 3142s` Phase 4
- `361_base`: about `4836s -> 4765s` Phase 4

Interpretation:

- fixed-frontier locality reordering is worth keeping
- but it is not the next major lever

### 3.2 Preflight planner is the right batch-size path

Planner runs improved throughput substantially:

- `828_base`: planner chose `352`, Phase 4 fell to about `2579s`
- `361_base`: planner chose `270`, Phase 4 fell to about `4300s`

This is consistent with feature batch size remaining an important throughput knob.

### 3.3 Newer telemetry points at refresh as the next bottleneck

From the newer post-fix prompt-828 deep-debug telemetry runs:

#### Ascend

- `phase4.refresh`: about `629.9s`
- `phase4.feature_batch`: about `163.0s`

#### Cardinal

- `phase4.refresh`: about `295.1s`
- `phase4.feature_batch`: about `147.9s`

Interpretation:

- the dominant remaining Phase 4 cost is now the **refresh / rescoring step**
- not the already-executing feature batches themselves

### 3.4 Refresh cost grows with stored rows

In the newer Ascend `828_base` debug run, refresh cost grew with the stored row
prefix (`st`):

| refresh | stored rows | read rows | refresh sec |
|---|---:|---:|---:|
| 0 | 1 | 1 | 0.855 |
| 1 | 513 | 2052 | 7.299 |
| 2 | 1025 | 4100 | 12.040 |
| 4 | 2049 | 10245 | 23.393 |
| 8 | 4097 | 20481 | 43.921 |
| 12 | 6145 | 28676 | 60.152 |
| 15 | 7681 | 34820 | 71.979 |

Cardinal shows the same pattern with smaller absolute times.

Interpretation:

- refresh work scales with the growing discovered prefix
- the current compact exact refresh path repeatedly rescans stored rows
- this matches the modest win from locality reorder: reorder helps replay locality,
  but it does not fix refresh recomputation

## 4. Code map for the refresh path

### 4.1 Entry from the main repo

- [`trace_pipeline_chunked.py`](../trace_pipeline_chunked.py)
  calls exact compact tracing through
  [`attribute_nnsight.attribute`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)

### 4.2 Compact exact Phase 3 row production

Phase 3 writes compact feature rows into the file-backed store:

- [`_FileBackedFeatureRowStore.append_rows`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)
- used from Phase 3 and Phase 4 row production in
  [`attribute_nnsight.py`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)

### 4.3 Phase 4 refresh loop

Main hot loop:

- [`attribute_nnsight.py`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)

Relevant operations inside refresh:

1. read stored compact rows through
   [`read_feature_rows`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)
2. recompute influences with
   [`compute_partial_feature_influences_streaming`](../../circuit-tracer_chunked/circuit_tracer/graph.py)
3. rank all features with `torch.argsort(...)`
4. mask visited features
5. select queue-sized frontier
6. reorder that frontier with
   [`_reorder_pending_for_phase4_locality`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)

### 4.4 Phase 4 batch execution path

After refresh picks `pending`, the actual feature batches run via:

- [`AttributionContext.compute_batch`](../../circuit-tracer_chunked/circuit_tracer/attribution/context_nnsight.py)
- chunked replay inside
  [`_compute_chunked_feature_attributions_from_grads`](../../circuit-tracer_chunked/circuit_tracer/attribution/context_nnsight.py)

This remains important, but current evidence suggests refresh is now the bigger
remaining optimization target.

## 5. Why refresh is expensive today

### 5.1 Streaming influence computation is recomputed from scratch every refresh

The compact path uses:

- [`compute_partial_feature_influences_streaming`](../../circuit-tracer_chunked/circuit_tracer/graph.py)

Current behavior:

- repeated chunked scans over stored rows
- repeated row reads through the memmap-backed row store
- iterative propagation until convergence / fixed point
- full recomputation each time the frontier is refreshed

This means refresh cost grows as `st` grows.

### 5.2 Full ranking is redone every refresh

After influences are recomputed, the code does:

- full `torch.argsort(feature_influences, descending=True)`
- visited filtering
- queue truncation

That means refresh is paying both:

- expensive influence recomputation
- expensive full-feature ranking

### 5.3 Hot-path telemetry/debug stats add some CPU work too

Per-refresh work also includes:

- vector stats on candidate scores
- normalization stats
- row-store snapshot diffs

These are not the main bottleneck, but they are part of the hot refresh path.

## 6. Candidate next optimization directions

Directions are ordered roughly by safety / expected usefulness.

### Direction A — exact row-chunk reuse inside refresh computation

**Summary:** avoid repeatedly rereading and re-materializing the same row chunks
within a refresh, and possibly across append-only refreshes.

**Why it is promising:**

- directly targets the observed growth in `feature_row_store.read_rows`
- preserves exact math if the same chunks are reused without approximation
- fits the current compact-row architecture

**Main code locations:**

- [`compute_partial_feature_influences_streaming`](../../circuit-tracer_chunked/circuit_tracer/graph.py)
- [`read_feature_rows`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)
- refresh loop in
  [`attribute_nnsight.py`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)

**Risk:** low to moderate

**Expected payoff:** high

### Direction B — exact masked top-k / frontier extraction without full argsort

**Summary:** replace full ranking with an exact selection path that still returns
the same frontier membership and deterministic ordering.

Possible shape:

- exact masked top-k on unvisited features
- then deterministic tie resolution
- then the existing locality reorder inside the fixed frontier

**Why it is promising:**

- full argsort over all active features is repeated every refresh
- frontier size is much smaller than total active features

**Main code locations:**

- refresh ranking / masking in
  [`attribute_nnsight.py`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)
- locality reorder helper
  [`_reorder_pending_for_phase4_locality`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)

**Risk:** moderate

Main trap:

- tie handling must preserve frontier membership semantics and deterministic
  ordering expectations

**Expected payoff:** medium to high

### Direction C — slim refresh-path telemetry / debug work outside debug mode

**Summary:** reduce unnecessary per-refresh scalar-stat work on normal benchmark
runs.

**Why it is promising:**

- easy / safe cleanup
- may reduce CPU overhead in the refresh loop

**Main code locations:**

- refresh stats and anomaly-debug helpers in
  [`attribute_nnsight.py`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)

**Risk:** low

**Expected payoff:** low to medium

### Direction D — tune row chunk sizing / prefetch strategy for streamed influence computation

**Summary:** reduce Python / IO overhead in streaming influence computation by
changing chunk sizing or caching behavior.

**Main code locations:**

- [`compute_partial_feature_influences_streaming`](../../circuit-tracer_chunked/circuit_tracer/graph.py)
- [`read_feature_rows`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)

**Risk:** low

**Expected payoff:** medium

### Direction E — incremental refresh state across frontier refreshes

**Summary:** avoid recomputing the full influence fixed point from scratch after
every refresh by carrying exact incremental state forward.

**Why it is interesting:**

- likely the biggest theoretical payoff
- directly attacks repeated whole-prefix recomputation

**Main code locations:**

- refresh loop in
  [`attribute_nnsight.py`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)
- solver in
  [`compute_partial_feature_influences_streaming`](../../circuit-tracer_chunked/circuit_tracer/graph.py)

**Risk:** high

Main traps:

- easy to accidentally change frontier membership semantics
- harder to reason about exact equivalence and convergence behavior

**Expected payoff:** high

### Direction F — replay-path work after refresh work is addressed

Replay is still worth revisiting later, especially in:

- [`compute_batch`](../../circuit-tracer_chunked/circuit_tracer/attribution/context_nnsight.py)
- [`_compute_chunked_feature_attributions_from_grads`](../../circuit-tracer_chunked/circuit_tracer/attribution/context_nnsight.py)

But current evidence suggests refresh should be optimized first.

## 7. Correctness invariants to preserve

### 7.1 Frontier membership safety first

For the next optimization step, prefer changes that preserve frontier membership.

Safe class:

- same influences
- same selected frontier
- different execution order / lower repeated overhead

Higher-risk class:

- changes to ranking, refresh policy, or queue semantics that can select a
  different node set

### 7.2 Exact normalization must stay exact

The compact solver relies on exact full-row absolute sums.

Relevant code:

- [`compute_partial_feature_influences_streaming`](../../circuit-tracer_chunked/circuit_tracer/graph.py)
- row-abs-sum handling in
  [`_FileBackedFeatureRowStore`](../../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py)

### 7.3 Row-to-node mapping must stay aligned

The compact path assumes strict alignment between:

- stored rows
- `row_to_node_index`
- active feature columns

### 7.4 Prompt 94 remains a gate

Prompt 94 is not the main optimization target here, but it remains a blocker
check after nontrivial Phase 4 changes.

## 8. Recommended near-term order

1. formalize refresh bottleneck findings in this spec
2. prototype **Direction A** first (exact row-chunk reuse / caching)
3. benchmark on `828_base` and `361_base`
4. if needed, follow with **Direction B** (exact masked top-k)
5. keep `94_base` as the watch/gate check after meaningful Phase 4 changes

## 9. Current recommendation

The next implementation step should target **Phase 4 refresh recomputation**,
starting with the lowest-risk exact optimization:

- reuse / cache row chunks in the streaming influence computation path

This is the best current candidate because it directly targets the measured
growth in refresh time while staying compatible with the current planner-based,
frontier-preserving Phase 4 flow.

## 9.5 Normalization precision findings

Follow-up matched runs established that normalization precision is a first-class
systems/correctness concern, separate from the refresh-cache experiment.

### Main result

Using the new external runtime knob `exact_trace_internal_dtype`, true fp32 vs
fp64 runs showed:

- `828_base` and `361_base` **collapse under fp32**
- `94_base` does **not** collapse via the same mechanism
- fp64 materially improves runtime on healthy prompts by preventing the fp32
  collapse mode

### Observed fp32 collapse pattern on `828_base` and `361_base`

- Phase-3 logit-row absolute sums overflow to `inf`
- later refresh ranking signal becomes effectively all-zero
- debug summaries show:
  - `rank_signal_effectively_all_zero_refresh_count = 16`
  - deterministic-shadow overlap collapses to `0.0`
  - float64-shadow overlap remains `1.0`
- telemetry shifts from a healthy regime to a degenerate one:
  - tiny `phase4.refresh`
  - huge `phase4.feature_batch` / `context.compute_batch`
  - large decoder-load counts
- compact output still keeps the same selected feature set, but fp32 runs on the
  healthy prompts retain only `8192` top edges rather than the full intended
  `20000`

### Observed fp64 behavior

- fp64 eliminates the healthy-prompt collapse mode
- `361_base` clean baseline improves by roughly:
  - `~16.7%` end-to-end
  - `~18.3%` in Phase 4
- decoder-load counts drop drastically under fp64 on `828_base` / `361_base`
- host RSS increases substantially

### Interpretation

The earlier large speedups on healthy prompts are best explained by the
float64-normalization path, not by the first refresh-cache attempt.

## 9.6 Long-trace scaling risk

fp64 is the right default for the current regime, but it is not a permanent
proof against overflow for arbitrarily long or unstable traces.

### Why fp64 may still fail eventually

The current exact path still forms raw L1 normalization sums over attribution
rows. If row magnitudes keep growing with trace depth / token count, then even
fp64 can eventually overflow (`~1e308`), just much later than fp32.

So the current fp64 default should be treated as:

- the correct immediate default,
- but not the final numerical-stability solution for very long autoregressive
  traces.

### Likely permanent-solution direction

The robust long-term fix is to stop representing the normalization denominator as
an unscaled raw sum that can overflow.

Promising designs:

1. **Scaled row-L1 computation (recommended first)**
   - For each row, compute:
     - `row_max_abs = max(abs(row))`
     - `scaled_l1 = sum(abs(row / row_max_abs))`
     - `row_l1 = row_max_abs * scaled_l1`
   - Keep the normalization in a scaled representation rather than immediately
     collapsing to one raw float that can overflow.
   - This preserves semantics while extending the safe numeric range.

2. **Exponent/mantissa normalization representation**
   - Store row normalization as something like `(mantissa, exponent)` or an
     equivalent `frexp`-style decomposition.
   - Use that representation directly inside the influence solver instead of a
     single floating-point scalar.

3. **Log-domain / log-scale normalization metadata**
   - More invasive, but can make very large dynamic ranges tractable.
   - Higher risk because solver math becomes less direct.

### Practical takeaway

For now:

- keep fp64 as the default runtime dtype
- do not revert to fp32 for exact compact tracing on healthy prompts

For longer-token tracing / scaling work:

- plan a dedicated numerical-stability refactor for row normalization
- treat “raw row-abs-sum overflow avoidance” as its own workstream, separate
  from replay scheduling and refresh caching

## 10. Direction A implementation plan: exact row-chunk reuse / caching

### 10.1 Objective

Reduce Phase 4 refresh time by reusing exact row chunks inside influence
recomputation, without changing selected frontier membership or numeric results.

### 10.2 Non-goals

- no approximation / sketching
- no change to refresh semantics, frontier selection, or planner behavior
- no Cardinal validation in this step; defer until after Ascend proof points
- no replay-path redesign

### 10.3 Likely design shape

Keep the current exact solver, but add a chunk-level reuse layer so repeated
refreshes avoid re-reading / re-materializing the same stored row segments.
Likely shape:

- cache or reuse chunk materialization keyed by stored-row span + refresh state
- preserve exact row ordering and row-to-node alignment
- invalidate/rebuild only when the underlying row store grows
- keep the existing streaming solver and ranking logic intact

### 10.4 Affected code locations in `circuit-tracer_chunked`

- `circuit_tracer/graph.py`
  - `compute_partial_feature_influences_streaming`
- `circuit_tracer/attribution/attribute_nnsight.py`
  - `read_feature_rows`
  - `_FileBackedFeatureRowStore`
  - Phase 4 refresh loop / call site
- possibly small helper wiring near the row-store read path if chunk cache state
  needs to live beside the file-backed store

### 10.5 Invariants / correctness checks

- identical frontier membership before and after the change
- identical exact influence values for the same refresh inputs
- preserved row-to-node alignment and stored-row ordering
- no change to planner-selected batch size or queue semantics
- no regressions in `94_base` watch checks after the change lands

### 10.6 Implementation steps

1. instrument the current refresh path to confirm where repeated chunk reads happen
2. add exact chunk reuse to the streaming influence path, scoped to the file-backed
   row store and refresh loop
3. keep invalidation minimal and explicit when new rows are appended
4. preserve all existing ranking / masking / locality-reorder behavior
5. add targeted telemetry for chunk reuse hit rate and refresh-time deltas
6. run Ascend validation on `828_base` and `361_base`
7. only after Ascend results are clean, decide whether Cardinal should be checked
   later as a separate follow-up

### 10.7 Ascend-only benchmark / validation plan

Primary validation fixtures:

- `828_base`
- `361_base`

Benchmark questions:

- does Phase 4 refresh wall time decrease?
- does total Phase 4 time improve or stay flat while refresh drops?
- are outputs unchanged at the compact exact-trace level?

Validation checks:

- compare refresh time and total Phase 4 time against the current baseline
- confirm chunk reuse telemetry shows expected hits / reduced rereads
- confirm no change in selected frontier or downstream trace artifacts
- keep Cardinal out of the fast iteration loop for now

### 10.8 Follow-up telemetry checks

After a successful run, verify:

- `phase4.refresh` drops materially on both Ascend fixtures
- `feature_row_store.read_rows` or equivalent read amplification decreases
- chunk reuse hit rate is nontrivial and stable across refreshes
- any refresh-time reduction is not offset by new overhead in telemetry/debug code

### 10.9 Deferred validation

Cardinal remains deferred until Ascend proves the optimization is both correct
and measurably faster. If the Ascend change lands cleanly, schedule Cardinal as a
later confirmation pass rather than part of the initial implementation loop.
