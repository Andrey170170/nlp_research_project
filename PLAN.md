# Current Implementation Plan — Phase 4 Planner V2

## 1. Problem statement

Planner V1 is now validated as a safe scheduler foundation:

- it is selectable with `phase4_scheduler_mode=planner_v1`,
- it preserves locality-v2 compact outputs exactly on canonical fast prompts,
- it emits useful plan / batch / invariant telemetry,
- it improved `828_base` substantially vs locality v2 and was effectively neutral
  on `361_base`.

The remaining limitation is that Planner V1 is **membership-preserving**. It can
reorder and batch the selected frontier, but it cannot choose a frontier that is
more locality-friendly than the canonical top-ranked set. The next experiment is
therefore **Planner V2**: a bounded membership-aware scheduler that may choose
from a slightly wider high-score candidate window to improve source-layer and
decoder-chunk locality while measuring score loss, selected-node overlap, and
compact-artifact drift.

Immediate goal:

> implement a feature-gated `planner_v2` path that uses Planner V1 as the
> reference plan, allows tightly bounded frontier membership changes, and emits
> enough telemetry to decide after one small validation run whether the direction
> is promising.

## 2. Current baseline facts

Use these as the comparison anchor for Planner V2.

### Validated baselines

- Hybrid row-store baseline remains the accepted memory baseline.
- Locality v2 is the current conservative fallback path.
- Planner V1 is the current best instrumented scheduler path and should be the
  direct A/B comparator for Planner V2.
- fp32 exact compact tracing remains the default.

### Recent fast-run results

`828_base`:

- hybrid Phase 4: `434.24 s`
- locality v2 Phase 4: `392.82 s`
- Planner V1 Phase 4: `314.78 s`
- Planner V1 vs locality v2: `-78.04 s` (`-19.9%`)
- Planner V1 refreshes / batches: `9 / 33`
- Planner V1 compact outputs matched locality v2 exactly.

`361_base`:

- hybrid Phase 4: `674.39 s`
- locality v1 Phase 4: `504.12 s` but with unsafe/generalization concern because
  it regressed `828_base`
- locality v2 Phase 4: `587.04 s`
- Planner V1 Phase 4: `580.01 s`
- Planner V1 vs locality v2: `-7.03 s` (`-1.2%`)
- Planner V1 refreshes / batches: `9 / 34`
- Planner V1 compact outputs matched locality v2 exactly.

Interpretation for V2:

- `828_base` already benefits from Planner V1; V2 must not give back that win.
- `361_base` likely has more locality upside; V2 should try to recover some of
  the v1-aggressive locality benefit without the `828_base` regression.
- Any V2 membership drift must be explicit, measured, and easy to roll back.

## 3. Working assumptions

- This is still within-trace optimization, not cross-trace or prefix reuse.
- Planner V2 is a live experimental path, not a default rollout.
- `phase4_scheduler_mode=locality` and `planner_v1` must remain available.
- V2 may change selected frontier membership **only inside bounded, logged
  constraints**.
- V2 must compute or reconstruct the Planner V1 reference frontier per refresh so
  telemetry can report overlap and score-loss metrics.
- If V2 cannot satisfy invariants, it should fail closed to Planner V1 for that
  refresh with an explicit fallback reason, or raise in debug mode if the failure
  indicates a bug.
- No GPU/model execution outside SLURM.

## 4. Scope

### In scope

- Add `phase4_scheduler_mode=planner_v2` in the library and project plumbing.
- Implement a bounded membership-aware frontier selection policy.
- Keep Planner V1 reference-plan generation for comparison and fallback.
- Add V2-specific telemetry for membership drift, score loss, locality gain, and
  fallback reasons.
- Add local unit tests for selection bounds, fallback behavior, and telemetry.
- Prepare canonical fast Ascend validation against Planner V1 and locality v2.

Primary library files likely in scope:

- `../worktrees_opt/circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
- `../worktrees_opt/circuit-tracer_chunked/circuit_tracer/attribution/attribute.py`
- `../worktrees_opt/circuit-tracer_chunked/tests/test_chunked_decoder_optimizations.py`
- `../worktrees_opt/circuit-tracer_chunked/tests/test_attribute_nnsight_telemetry.py`

Project files likely in scope:

- `trace_pipeline_chunked.py`
- `experiments/exact_trace_bench/config.py`
- `experiments/exact_trace_bench/scenarios.py`
- `experiments/run_sparsification_experiment.py`
- `experiments/exact_trace_bench/extract.py`
- `experiments/extract_benchmark_index.py`
- `tests/test_cross_cluster_debug_artifacts.py`

### Non-goals

- Do not make Planner V2 the default.
- Do not introduce silent approximation on the default exact compact path.
- Do not reopen broad row-store read-path rewrite.
- Do not run a large cache/hidden-knob sweep as part of the first V2 pass.
- Do not start with `361_late`; validate base prompts first.
- Do not change refresh cadence in the first V2 implementation unless needed for
  fallback safety. Adaptive refresh remains a later policy.

## 5. Planner V2 policy design

### 5.1 Reference frontier

For each Phase 4 refresh, first form the same canonical selected frontier that
Planner V1 would execute:

- same rank inputs,
- same visited mask,
- same target frontier size,
- same locality ordering and batch shaping as Planner V1.

Call this the **reference frontier**. It is used for:

- fallback execution,
- membership overlap metrics,
- score-loss metrics,
- compact-artifact drift interpretation.

### 5.2 Candidate window

Planner V2 may select from a wider candidate window than the reference frontier.

Initial conservative policy:

- `reference_frontier_size = F`
- `candidate_window_size = min(unvisited_count, max(F, F * window_multiplier))`
- default `window_multiplier`: `2`
- cap candidate window to avoid pathological memory/CPU overhead; start with a
  hard internal cap if plumbing another public knob is not worth it yet.

The window should be score-ordered by the same influence ranking used today. V2
must not consider low-score arbitrary candidates outside this window.

### 5.3 Locked high-score prefix

To avoid large semantic drift, the highest-ranked candidates should be locked into
the V2 frontier.

Initial conservative policy:

- lock the top `50%` of the reference frontier by rank,
- allow substitutions only in the lower half of the selected frontier,
- preserve final selected count exactly unless fewer unvisited candidates exist.

This gives the policy enough freedom to improve locality while keeping the most
important frontier mass stable.

### 5.4 Membership-change bound

V2 must enforce an explicit maximum membership delta relative to the reference
frontier.

Initial conservative policy:

- maximum replaced fraction: `25%` of the reference frontier,
- hard invariant: `selected_count == reference_selected_count`,
- hard invariant: no duplicates,
- hard invariant: no already-visited features.

Telemetry must report:

- retained reference count,
- replaced count,
- added count,
- removed count,
- selected-node Jaccard vs reference,
- locked-prefix violation count.

### 5.5 Score-loss bound

V2 should reject plans that buy locality by dropping too much influence score.

Initial conservative policy:

- compute reference selected score sum,
- compute V2 selected score sum,
- require V2 score sum to remain above a configured ratio of reference,
- default minimum score-sum ratio: `0.995`,
- also record min / p50 / max rank displacement for added nodes.

If score values are unstable or not directly comparable for a refresh, V2 should
fall back to Planner V1 and record `fallback_reason=score_metrics_unavailable`.

### 5.6 Locality objective

V2 should optimize locality inside the allowed replacement budget.

Primary locality keys:

1. source layer,
2. decoder chunk id,
3. token position if useful and cheap,
4. original score rank as a final tie-breaker.

Initial objective:

- preserve locked prefix membership,
- choose optional replacements that extend useful `(source_layer, decoder_chunk)`
  runs already present in the reference frontier,
- prefer candidates that reduce planned batch fragmentation,
- avoid creating extra tiny batches,
- keep final execution ordering compatible with Planner V1 batch shaping.

Implementation can be simple for the first pass:

1. build candidate groups by `(source_layer, decoder_chunk)`,
2. score each possible group by:
   - candidate influence score,
   - group run length / batch-fill benefit,
   - rank displacement penalty,
3. fill the replaceable portion of the frontier from the best locality groups
   until the selected count is reached,
4. run the existing Planner V1 ordering/batch-shaping logic on the selected V2
   membership.

Avoid overengineering a global optimizer in the first pass. The goal is a
diagnosable policy, not an optimal scheduler.

### 5.7 Fallback behavior

Fallback to Planner V1 for the refresh if any of these happen:

- candidate window is unavailable or malformed,
- score-loss metrics are unavailable,
- selected count differs from reference selected count,
- duplicate or visited feature appears,
- locked-prefix preservation fails,
- score-sum ratio is below threshold,
- membership delta exceeds threshold,
- batch boundaries do not advance,
- telemetry serialization would fail.

Fallback must be visible in telemetry:

- `phase4_scheduler_mode=planner_v2`,
- `phase4_scheduler_effective_policy=planner_v1_fallback` or equivalent,
- `planner_v2_fallback_reason=<reason>`,
- reference hashes and candidate-window summary where available.

## 6. Required flags / metadata

Keep the public surface small for the first V2 pass.

Required mode:

- `phase4_scheduler_mode=planner_v2`

Reuse existing controls:

- `phase4_scheduler_debug`
- `phase4_scheduler_telemetry_detail=summary|normal|debug`
- `telemetry_max_events`

Optional V2 controls if they are cheap and low-risk to plumb:

- `phase4_planner_v2_window_multiplier`
- `phase4_planner_v2_locked_prefix_fraction`
- `phase4_planner_v2_max_replacement_fraction`
- `phase4_planner_v2_min_score_ratio`

If adding all V2 knobs would slow implementation, hard-code conservative defaults
in the library and expose only the scheduler mode first. Do not create a broad
tuning surface before the policy shows promise.

Every completion / step record should include:

- requested scheduler mode,
- effective scheduler mode / policy,
- V2 policy version,
- V2 defaults or explicit knob values,
- V2 fallback count,
- V2 fallback reasons observed,
- V2 membership-overlap summary,
- V2 score-loss summary.

## 7. Required telemetry

### Phase-level summary

- scheduler mode/version/policy,
- total selected features,
- Phase 4 wall time,
- refresh count,
- feature-batch count,
- total refresh elapsed time,
- total feature-batch elapsed time,
- Planner V2 refresh count,
- Planner V2 fallback count,
- aggregate membership Jaccard vs reference,
- aggregate score-sum ratio vs reference,
- compact retained-edge count when available.

### Refresh-level V2 summary

For each refresh, record:

- refresh index,
- reference frontier size,
- candidate window size,
- selected V2 frontier size,
- reference membership hash,
- V2 membership hash,
- reference order hash,
- V2 order hash,
- retained / added / removed / replaced counts,
- selected-node Jaccard vs reference,
- locked-prefix size and violation count,
- reference score sum,
- V2 score sum,
- score-sum ratio,
- cutoff score and replacement score range if cheap,
- rank displacement min / p50 / max for additions,
- locality fragmentation before / after,
- planned batch count before / after,
- boundary reason counts,
- fallback reason if V2 did not execute.

### Batch-level V2 summary

Reuse Planner V1 batch telemetry, and add if cheap:

- whether the batch contains any V2-added nodes,
- count of reference-retained vs V2-added nodes in the batch,
- batch membership hash,
- distinct source layers,
- distinct decoder chunks,
- monotonic chunk-order flag.

### Compact comparison outputs after validation

The analysis pass should compare V2 artifacts against both Planner V1 and
locality v2:

- feature Jaccard,
- edge Jaccard,
- weighted edge Jaccard,
- retained edge count,
- completion text,
- active feature count,
- Phase 4 wall time,
- attribution wall time,
- scenario duration,
- refresh / batch count,
- memory snapshots.

## 8. Implementation sequence

### Step 1 — library policy skeleton

1. Add `planner_v2` to scheduler-mode validation.
2. Add a V2 policy/version identifier.
3. Make the V2 path call the existing Planner V1 plan builder to create the
   reference plan.
4. Initially execute the reference plan unchanged while emitting V2 identity
   telemetry; this is the plumbing sanity checkpoint.

### Step 2 — bounded candidate window

1. Build the score-ordered candidate window from unvisited candidates.
2. Add candidate-window telemetry.
3. Add tests for window size, empty windows, and short-frontier edge cases.

### Step 3 — membership-aware selection

1. Lock the top reference prefix.
2. Build replacement candidates from the candidate window outside the locked set.
3. Select replacements using the simple locality objective.
4. Enforce selected count, duplicate, visited, replacement-fraction, and score-loss
   invariants.
5. Fall back to Planner V1 on invariant failure.

### Step 4 — batch construction and telemetry

1. Reuse Planner V1 ordering and shaped-batch construction on V2-selected
   membership.
2. Emit reference-vs-V2 membership and score metrics on refresh events.
3. Emit V2-added node counts on batch events if cheap.
4. Add phase-level aggregate summaries.

### Step 5 — project plumbing

1. Ensure `trace_pipeline_chunked.py` accepts `planner_v2`.
2. Ensure scenario config and run launcher pass the mode through.
3. Ensure manifests, run configs, benchmark extraction, and index rows preserve
   V2 mode/effective-policy fields.
4. Add / update project round-trip tests.

### Step 6 — focused local validation

Run only lightweight checks:

- library:
  - `uv run pytest tests/test_chunked_decoder_optimizations.py -k 'phase4 or scheduler or planner'`
  - `uv run pytest tests/test_attribute_nnsight_telemetry.py`
  - `uv run ruff check circuit_tracer/attribution/attribute_nnsight.py circuit_tracer/attribution/attribute.py tests/test_chunked_decoder_optimizations.py tests/test_attribute_nnsight_telemetry.py`
- project:
  - `uv run python tests/test_cross_cluster_debug_artifacts.py`
  - `uv run ruff check trace_pipeline_chunked.py experiments tests`

No model loading or GPU work outside SLURM.

## 9. Cluster validation plan

### First validation run

Run on Ascend fast tier only:

- `828_base`
- `361_base`

Use the same canonical settings as Planner V1:

- `exact_trace_internal_dtype=fp32`
- `feature_batch_size=256`
- `decoder_chunk_size=4096`
- `cross_batch_decoder_cache_bytes=0`
- `max_feature_nodes=8192`
- `max_edges=20000`
- `attribution_update_interval=4`
- `phase4_scheduler_mode=planner_v2`
- `phase4_scheduler_debug=true`
- `phase4_scheduler_telemetry_detail=debug`

Expected scenario names:

- `ascend_fast_828_base_phase4_planner_v2_fp32_b256_c4096_cache0`
- `ascend_fast_361_base_phase4_planner_v2_fp32_b256_c4096_cache0`

Compare against:

1. Planner V1:
   - `ascend_fast_828_base_phase4_planner_v1_fp32_b256_c4096_cache0`
   - `ascend_fast_361_base_phase4_planner_v1_fp32_b256_c4096_cache0`
2. locality v2:
   - `ascend_fast_828_base_phase4_locality_validation_v2_fp32_b256_c4096_cache0`
   - `ascend_fast_361_base_phase4_locality_validation_v2_fp32_b256_c4096_cache0`
3. hybrid row-store baseline for broader historical comparison.

### First validation questions

1. Did V2 execute or mostly fall back to V1?
2. If V2 executed, how much membership changed per refresh?
3. What was the score-sum ratio vs V1 reference?
4. Did locality fragmentation improve?
5. Did Phase 4 batch elapsed time improve?
6. Did refresh elapsed time regress?
7. Did compact edges drift mildly or catastrophically?
8. Did `828_base` keep the Planner V1 win?
9. Did `361_base` move toward the aggressive locality v1 result without `828_base`
   regression?

### Optional second validation

Only if the first validation is promising:

- rerun `828_base` and `361_base` with a slightly more permissive V2 policy, or
- run one prompt-diversity / parity-oriented check before touching `361_late`.

Do not use `361_late` until base-prompt behavior is understood.

## 10. Acceptance criteria

Planner V2 is successful enough to keep iterating if all are true:

1. it is fully feature-gated behind `phase4_scheduler_mode=planner_v2`,
2. Planner V1 and locality fallback modes still work,
3. local tests pass,
4. V2 telemetry clearly reports membership drift and score loss,
5. no invariant failures are silent,
6. canonical fast runs complete successfully,
7. completion text and active feature counts are unchanged,
8. retained edges remain operationally sane,
9. compact edge drift vs Planner V1 is explainable and not catastrophic,
10. runtime is not materially worse than Planner V1 on `828_base`,
11. `361_base` shows either a Phase 4 improvement or telemetry that clearly points
    to the next policy adjustment.

Suggested first-pass quantitative gates:

- fallback count should be low enough to evaluate the policy; if fallback is high,
  the run is a plumbing/policy-bound failure rather than a speed result,
- average selected-node Jaccard vs V1 reference should stay high; start by treating
  `< 0.85` as too aggressive,
- score-sum ratio should stay `>= 0.995` unless there is a deliberate follow-up
  experiment,
- `828_base` Phase 4 should not regress by more than `5%` vs Planner V1,
- `361_base` should ideally improve Phase 4 by at least `5%` vs Planner V1 to
  justify further membership-changing work.

## 11. Risks and open questions

- **Correctness / interpretability drift:** V2 intentionally changes selected
  frontier membership, so exact compact outputs may differ. The goal is bounded,
  measured drift, not exact Jaccard `1.0` vs V1.
- **Score-loss metric quality:** influence score sums may not fully capture graph
  quality. Compact edge comparison remains required.
- **Prompt dependence:** `828_base` and `361_base` may prefer different locality
  aggressiveness.
- **Overfitting to batch locality:** reducing fragmentation can still lose if it
  increases refresh cost or chooses lower-value nodes.
- **Too many knobs:** expose only what is needed for controlled validation.
- **Fallback ambiguity:** high fallback rates need to be distinguishable from a
  successful conservative policy.

## 12. Likely next step after Planner V2

If V2 gives useful telemetry but mixed performance, choose one narrow follow-up:

1. tune only the replacement budget / locked prefix,
2. add adaptive refresh cadence using V2 fragmentation and score-margin metrics,
3. test a nonzero decoder cache setting guided by V2 batch locality telemetry,
4. run a small cross-cluster parity spot-check if compact drift looks meaningful.

Avoid broad hidden-knob sweeps until Planner V2 either proves useful or is rolled
back to Planner V1 as the stable scheduler foundation.
