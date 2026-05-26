# Exact-trace knob/API taxonomy map

Status: Phase 3 mapping draft
Last updated: 2026-05-16

This document maps the exact-trace knobs that currently exist across the project
repo and sibling `../circuit-tracer_chunked` library. It is intentionally detailed:
the goal is to make the existing surface explicit before we clean, stabilize,
deprecate, archive, or test anything.

Scope of this pass:

- project CLI/scenario surfaces in `nlp_research_project`,
- sibling library public/NNSight attribution entrypoints,
- generated scenario files already present in the repo,
- debug/replay artifact surfaces that look like knobs but are actually schemas.

This pass is a map, not yet a code change. The next Phase 3 step should convert
the decisions in the `Recommended cleanup actions` section into small tests and
mechanical changes.

## Intended taxonomy

The cleanup target is **not** to hide most internal controls. This project is
still in an active measurement/optimization phase, and we expect to run sweeps
over many internal influences to understand defaults and stability. The goal is
to make knob ownership explicit:

- which knobs are stable enough for ordinary users,
- which knobs are advanced but intentionally surfaced for precise internal
  adjustments,
- which knobs are research/debug tools with artifact/provenance burden,
- which knobs are compatibility shims or true implementation details.

| Bucket | Meaning | Desired user visibility |
|---|---|---|
| Fully public stable | Ordinary documented controls for normal exact-bench/library users | visible in README/harness docs and tested as stable defaults |
| Advanced public / research-tuning | Internal controls intentionally exposed for sweeps, optimization, and precise adjustment | documented in advanced docs; default-preserving; expected to be used by project researchers |
| Debug/replay public | Track-A capture/replay/fingerprint tooling and artifact controls | documented as opt-in validation/debug workflow with provenance requirements |
| Private/internal implementation detail | Backend mode switches, helper representation choices, or values with no useful tuning value | not exposed in normal scenarios; may appear in code only |
| Deprecated/compatibility | Historical name or alias kept to avoid breaking old code | documented as compat only with preferred replacement |
| Artifact schema field | Emitted diagnostic/provenance field, not a user knob | documented with artifacts, not scenario defaults |

## Desired public surface shape

The normal exact-bench defaults should be small and stable, but the advanced
surface should remain available for sweeps.

### Fully public stable knobs

These are the knobs a normal user should expect to see first:

1. `exact_trace_internal_dtype`
   - canonical precision contract,
   - default `fp32`,
   - `fp64` retained for targeted parity/diagnostic spot checks.
2. `decoder_chunk_size`
   - resource/performance knob passed to model/transcoder loading,
   - changing it can change compact outputs relative to the current `c2048`
     reference, so comparisons must hold it fixed unless testing chunk behavior.
3. `cross_batch_decoder_cache_bytes`
   - resource/performance knob for Phase-4 decoder-cache budget,
   - validated cache-size changes were exact within a fixed chunk size.

### Advanced public / research-tuning knobs

These should not be hidden. They should be surfaced in an advanced section and
protected by default-preserving tests:

- Phase-1 trace-batch sizing:
  - `phase1_trace_batch_policy`,
  - `phase1_trace_batch_size_max`.
- Phase-4 execution/planning:
  - `phase4_scheduler_mode`,
  - `phase4_refresh_policy`,
  - `phase4_refresh_interval_multiplier`,
  - `phase4_ranker`,
  - `phase4_refresh_optimization`,
  - `phase4_row_executor`.
- Memory/placement/cache behavior:
  - `row_store_cache_control`,
  - `exact_encoder_residency`,
  - `chunked_feature_replay_window`,
  - `error_vector_prefetch_lookahead`,
  - `stage_encoder_vecs_on_cpu`,
  - `stage_error_vectors_on_cpu`,
  - `row_subchunk_size`.
- Planner/resource search controls:
  - `plan_feature_batch_size`,
  - `feature_batch_size_max`,
  - `feature_batch_target_reserved_fraction`,
  - `feature_batch_min_free_fraction`,
  - `feature_batch_probe_batches`.

### Debug/replay public knobs

These are also legitimate user-facing knobs for this research project, but they
should be grouped separately because they create artifacts, replay donor state, or
single-step semantics:

- `cross_cluster_debug`,
- `phase0_activation_threshold_compare_mode`,
- Phase-0 donor capture/replay controls,
- Phase-3 seed/gradient/row capture and replay controls,
- semantic descriptor capture controls,
- telemetry caps and verbose debug telemetry.

### Private or compatibility candidates

Initial candidates for private/compat treatment:

- `internal_precision`: compatibility plumbing behind
  `exact_trace_internal_dtype`, not a second first-class precision contract.
- `compact_output`: backend execution-mode switch chosen by project raw-vs-compact
  orchestration, not a normal scenario knob.
- environment-variable debug overrides such as `PHASE4_ANOMALY_DEBUG`: remove
  rather than preserve. They bypass scenario provenance, are easy to forget in
  SLURM environments, and duplicate explicit CLI/scenario knobs.
- fixed internal constants that meaningfully affect behavior, such as hard-coded
  planner-v2 policy constants: either expose them as named advanced public knobs
  for sweeps, or delete the dead path if they do not provide tuning value. Avoid
  preserving hidden behavioral constants indefinitely.

## Project-side surfaces

### `trace_pipeline_chunked.py`

This is the main project CLI and runtime guard layer for exact/chunked tracing.

Relevant parser helpers and accepted values:

| Parser/helper | Accepted values / aliases | Current default | Notes |
|---|---|---|---|
| `parse_exact_trace_internal_dtype` | `fp32`, `float32`, `torch.float32`, `fp64`, `float64`, `torch.float64` | `fp32` | Maps to `resolve_internal_precision(...)` for backend compat. |
| `parse_phase0_activation_threshold_compare_mode` | `baseline`, `default`, `bf16`, `bfloat16`, `fp32`, `float32`, `fp64`, `float64` | `baseline` | Track-A diagnostic; changes Phase-0 compare semantics. |
| `parse_phase0_replay_mode` | `disabled`, `donor_phase0` | `disabled` | Requires `--phase0-donor-bundle` when enabled. |
| `parse_phase0_donor_context_policy` | `strict`, `warn` | `strict` | Replay validation behavior. |
| `parse_phase3_replay_mode` | `disabled`, `donor` | `disabled` | Used separately for gradient and row replay. |
| `parse_phase3_replay_validation_policy` | `strict` only | `strict` | Currently a hard strict gate. |
| `parse_phase1_trace_batch_policy` | `legacy`, `cap_effective_batches` | `legacy` | Scenario-only optimization/resource control. |
| `parse_phase4_refresh_policy` | `standard`, `deferred_v1` | `standard` | Scenario-only optimization; changes refresh cadence. |
| `parse_phase4_ranker` | `argsort`, `topk_v1` | `argsort` | Experimental; may affect equal-score frontier membership. |
| `parse_row_store_cache_control` | `off`, `fadvise_dontneed_after_append_v1` | `off` | Platform/cache-control optimization. |
| `parse_exact_encoder_residency` | `lazy`, `active_cpu`, `active_pinned_cpu` | `lazy` | Exact-path memory/placement knob. |
| `parse_phase4_scheduler_mode` | `locality`, `planner_v1`, `planner_v2`, `legacy` alias | `locality` | Experimental scheduler surface. |
| `parse_phase4_scheduler_telemetry_detail` | `summary`, `normal`, `debug`, `compact` alias, `full` alias | `normal` | Telemetry verbosity only. |
| `parse_phase4_refresh_optimization` | `off`, `v1` | `off` | Experimental refresh optimization. |
| `parse_phase4_row_executor` | `batched`, `streaming_v1` | `batched` | Experimental row executor path. |

CLI flags that are present directly in `trace_pipeline_chunked.py`:

| Flag | Default | Current classification | Notes |
|---|---:|---|---|
| `--exact-trace-internal-dtype` | `fp32` | Canonical default / public precision knob | Compact exact-chunked only for non-default values. |
| `--decoder-chunk-size` | `256` at raw CLI level | Public resource/perf knob | Canonical exact-bench scenarios override to cluster/tier values. |
| `--cross-batch-decoder-cache-bytes` | `None` at raw CLI level | Public resource/perf knob | Canonical scenarios set `0`, long-eval cache probes set `8 GiB`. |
| `--phase0-activation-threshold-compare-mode` | `baseline` | Debug/replay diagnostic | Historical Track-A compare-mode experiments used `fp32`/`fp64`. |
| `--cross-cluster-debug` | false | Debug artifact knob | Writes scalar checkpoint/batch summaries. |
| `--capture-phase0-donor-bundle` | false | Debug artifact knob | Writes Phase-0 donor bundle `.npz`. |
| `--phase0-donor-bundle` | `None` | Debug replay input | Must pair with `--phase0-replay-mode donor_phase0`. |
| `--phase0-replay-mode` | `disabled` | Debug replay knob | Single-step intended semantics are recorded in `run_config.json`. |
| `--phase0-donor-context-policy` | `strict` | Debug replay validation knob | `warn` exists but normal path should not use it. |
| `--capture-phase3-seed-bundle` | false | Debug artifact knob | Writes Phase-3 seed bundle `.npz`. |
| `--capture-phase3-gradient-bundle` | false | Debug artifact knob | Writes Phase-3 gradient bundle `.npz`. |
| `--capture-phase3-row-bundle` | false | Debug artifact knob | Writes Phase-3 row bundle `.npz`. |
| `--phase3-gradient-donor-bundle` | `None` | Debug replay input | Must pair with gradient replay mode. |
| `--phase3-gradient-replay-mode` | `disabled` | Debug replay knob | Accepted enabled value: `donor`. |
| `--phase3-row-donor-bundle` | `None` | Debug replay input | Must pair with row replay mode. |
| `--phase3-row-replay-mode` | `disabled` | Debug replay knob | Accepted enabled value: `donor`. |
| `--phase3-replay-validation-policy` | `strict` | Debug replay validation knob | Strict only today. |
| `--capture-feature-semantic-descriptors` | false | Debug artifact knob | Writes bounded semantic descriptor `.npz`. |
| `--semantic-descriptor-top-k` | `2048` | Debug artifact shape knob | Only meaningful when descriptor capture is enabled. |
| `--semantic-descriptor-dim` | `64` | Debug artifact shape knob | Only meaningful when descriptor capture is enabled. |
| `--phase1-trace-batch-policy` | `legacy` | Scenario-only optimization/resource knob | Paired with `phase1_trace_batch_size_max`. |
| `--phase1-trace-batch-size-max` | `None` | Scenario-only optimization/resource knob | Positive integer or omitted. |
| `--phase4-refresh-policy` | `standard` | Scenario-only optimization knob | `deferred_v1` expands refresh cadence. |
| `--phase4-refresh-interval-multiplier` | `1` | Scenario-only optimization knob | Positive integer. |
| `--phase4-ranker` | `argsort` | Scenario-only/experimental correctness-sensitive knob | `topk_v1` may differ on ties. |
| `--row-store-cache-control` | `off` | Scenario-only platform/resource knob | `fadvise...` is cache-control telemetry/platform behavior. |
| `--exact-encoder-residency` | `lazy` | Scenario-only memory-placement knob | Active modes are exact-path-only. |
| `--phase4-scheduler-mode` | `locality` | Scenario-only experimental scheduler knob | `legacy` aliases to `locality`; `planner_v1/v2` are explicit experiments. |
| `--phase4-scheduler-debug` | false | Debug/telemetry knob | Should not appear in canonical scenarios. |
| `--phase4-scheduler-telemetry-detail` | `normal` | Debug/telemetry knob | `summary`, `normal`, `debug`. |
| `--phase4-refresh-optimization` | `off` | Scenario-only optimization knob | `v1` is opt-in. |
| `--phase4-row-executor` | `batched` | Scenario-only optimization knob | `streaming_v1` is opt-in. |
| `--phase4-anomaly-debug` | false | Historical debug knob | Compact exact-chunked only; old anomaly scaffolding. |
| `--chunked-feature-replay-window` | `4` | Scenario-only backend tuning | Currently in base defaults and forwarded. |
| `--error-vector-prefetch-lookahead` | `2` | Scenario-only backend tuning | Currently in base defaults and forwarded. |
| `--stage-encoder-vecs-on-cpu` | `None` | Scenario-only placement override | Optional bool. |
| `--stage-error-vectors-on-cpu` | `None` | Scenario-only placement override | Optional bool. |
| `--row-subchunk-size` | `None` | Scenario-only backend tuning | Defaults to decoder chunk behavior. |
| `--plan-feature-batch-size` | false | Scenario-only planner/resource knob | Compact exact only. |
| `--auto-scale-feature-batch-size` | false | Deprecated/compat alias | Legacy alias for planner behavior. |
| `--feature-batch-size-max` | `None` | Scenario-only planner/resource knob | Upper bound for planned Phase-4 feature microbatch size. |
| `--feature-batch-target-reserved-fraction` | `0.9` | Scenario-only planner/resource knob | Planner target for CUDA reserved-memory utilization. |
| `--feature-batch-min-free-fraction` | `0.05` | Scenario-only planner/resource knob | Planner safety headroom. |
| `--feature-batch-probe-batches` | `1` | Scenario-only planner/resource knob | Number of preflight probe batches. |
| `--telemetry-max-events` | `None` | Debug/telemetry knob | Caps in-memory telemetry event storage. |
| `--diagnostic-feature-cap` | `None` | Debug/profiling semantic-changing knob | Early cap on active features; not exact canonical behavior. |
| `--sparsify-per-layer-position-topk` | `None` | Experimental approximation/sparsification knob | Screens candidates before exact attribution. |
| `--sparsify-global-cap` | `None` | Experimental approximation/sparsification knob | Global candidate cap after per-bucket sparsification. |

Runtime guard notes:

- Debug/replay/capture features reject `--save-raw`; they are compact
  exact-chunked only.
- Phase-0 replay requires `phase0_donor_bundle` and errors if a donor path is
  supplied while replay is disabled.
- Phase-3 gradient/row replay each require their corresponding donor bundle and
  error if a donor path is supplied while replay is disabled.
- Non-default `exact_trace_internal_dtype` is rejected for `--save-raw`; the
  explicit precision contract is compact exact-chunked only.
- Non-default Phase-1/Phase-4 execution controls are rejected for `--save-raw`.

### `experiments/exact_trace_bench/config.py`

`base_trace_defaults()` currently includes many values beyond the desired normal
public surface:

| Default key | Current value | Classification |
|---|---:|---|
| `exact_trace_internal_dtype` | `fp32` | Canonical default |
| `phase0_activation_threshold_compare_mode` | `baseline` | Debug/replay public default; should stay defaulted in normal scenarios unless an explicit compare-mode experiment requests non-default behavior. |
| `chunked_feature_replay_window` | `None` | Scenario-only/backend tuning default. |
| `error_vector_prefetch_lookahead` | `None` | Scenario-only/backend tuning default. |
| `stage_encoder_vecs_on_cpu` | `None` | Scenario-only/backend tuning default. |
| `stage_error_vectors_on_cpu` | `None` | Scenario-only/backend tuning default. |
| `row_subchunk_size` | `None` | Scenario-only/backend tuning default. |
| `plan_feature_batch_size` | `False` | Scenario-only planner/resource default. |
| `auto_scale_feature_batch_size` | `False` | Deprecated/compat planner alias. |
| `feature_batch_size_max` | `None` | Scenario-only planner/resource default. |
| `feature_batch_target_reserved_fraction` | `0.9` | Scenario-only planner/resource default. |
| `feature_batch_min_free_fraction` | `0.05` | Scenario-only planner/resource default. |
| `feature_batch_probe_batches` | `1` | Scenario-only planner/resource default. |
| `phase4_anomaly_debug` | `False` | Historical debug default. |
| `phase4_scheduler_mode` | `locality` | Scenario-only experimental scheduler default. |
| `phase4_scheduler_debug` | `False` | Debug/telemetry default. |
| `phase4_scheduler_telemetry_detail` | `normal` | Debug/telemetry default. |
| `phase4_refresh_optimization` | `off` | Scenario-only optimization default. |
| `phase4_row_executor` | `batched` | Scenario-only optimization default. |
| `cross_cluster_debug` | `False` | Debug artifact default. |
| `telemetry_max_events` | `None` | Debug/telemetry default. |

Cleanup implication: `base_trace_defaults()` is currently a mixed bag. It acts as
a scenario default block, not as a clean public API. Phase 3 should either split
it into `NORMAL_TRACE_DEFAULTS`, `EXPERIMENTAL_TRACE_DEFAULTS`, and
`DEBUG_TRACE_DEFAULTS`, or add tests that canonical generated scenarios do not
materialize debug/replay fields unless intentionally requested.

### `experiments/exact_trace_bench/scenarios.py`

`EXACT_MODE_KNOB_KEYS` is the widest project-side scenario allowlist. It includes
normal, optimization, debug, replay, and telemetry knobs in one tuple:

```text
chunked_feature_replay_window
error_vector_prefetch_lookahead
stage_encoder_vecs_on_cpu
stage_error_vectors_on_cpu
row_subchunk_size
exact_trace_internal_dtype
phase0_activation_threshold_compare_mode
phase1_trace_batch_policy
phase1_trace_batch_size_max
plan_feature_batch_size
auto_scale_feature_batch_size
feature_batch_size_max
feature_batch_target_reserved_fraction
feature_batch_min_free_fraction
feature_batch_probe_batches
phase4_anomaly_debug
phase4_refresh_policy
phase4_refresh_interval_multiplier
phase4_ranker
row_store_cache_control
exact_encoder_residency
phase4_scheduler_mode
phase4_scheduler_debug
phase4_scheduler_telemetry_detail
phase4_refresh_optimization
phase4_row_executor
cross_cluster_debug
capture_phase0_donor_bundle
phase0_donor_bundle
phase0_replay_mode
phase0_donor_context_policy
phase3_gradient_donor_bundle
phase3_gradient_replay_mode
phase3_row_donor_bundle
phase3_row_replay_mode
phase3_replay_validation_policy
capture_phase3_seed_bundle
capture_phase3_gradient_bundle
capture_phase3_row_bundle
capture_feature_semantic_descriptors
semantic_descriptor_top_k
semantic_descriptor_dim
telemetry_max_events
```

Current canonical builder behavior is safer than that list looks when reading the
raw scenario rows:

- `CLUSTER_SETTINGS` for `fast`, `anomaly`, and `long_eval` only specify batch,
  chunk, cache, labels, and fixture choices.
- Canonical generated scenario rows therefore include `decoder_chunk_size` and
  `cross_batch_decoder_cache_bytes`, while debug/replay knobs only appear in the
  `defaults` block or historical one-off scenario files.

Important caveat: `run_sparsification_experiment.py` merges `defaults` into each
scenario before building the command. Several default-valued advanced/debug fields
therefore can still be materialized as CLI flags even if they are absent from raw
scenario rows. Guard tests must inspect the effective merged scenario or the
dry-run command, not only the raw `scenarios[]` entries.

Cleanup implication: split this allowlist or add a guard test. A single
`EXACT_MODE_KNOB_KEYS` tuple makes it too easy for ordinary scenario generation to
inherit replay/debug knobs later.

### `experiments/run_sparsification_experiment.py`

This runner is the scenario JSON → `trace_pipeline_chunked.py` CLI bridge. It
forwards any recognized scenario fields into command-line flags.

It forwards all major categories:

- canonical/public: `exact_trace_internal_dtype`, `decoder_chunk_size`,
  `cross_batch_decoder_cache_bytes`,
- advanced public optimization/resource controls: Phase-1 trace-batch policy,
  feature-batch planner controls, Phase-4 scheduler/refresh/ranker/executor/
  cache-control/residency controls,
- debug/replay controls: Phase-0 replay/capture, Phase-3 gradient/row replay,
  semantic descriptors, cross-cluster debug, telemetry caps,
- semantic-changing experiment controls: `diagnostic_feature_cap`, sparsification
  caps,
- deprecated/compat: `auto_scale_feature_batch_size`.

Cleanup implication: this bridge should stay wide because it executes explicit
scenario files. The safer place to enforce default behavior and category labeling
is scenario generation and tests, not this low-level runner.

## Sibling library surfaces

### `circuit_tracer/attribution/attribute.py::attribute(...)`

This public wrapper exposes a smaller but still mixed exact-trace surface:

| Parameter | Default | Classification | Notes |
|---|---:|---|---|
| `exact_trace_internal_dtype` | `fp32` | Canonical default / public precision knob | NNSight path only; documented as post-fix default. |
| `chunked_feature_replay_window` | `4` | Scenario-only backend tuning | Passed to NNSight backend. |
| `error_vector_prefetch_lookahead` | `2` | Scenario-only backend tuning | Passed to NNSight backend. |
| `stage_encoder_vecs_on_cpu` | `None` | Scenario-only placement override | Passed to NNSight backend. |
| `stage_error_vectors_on_cpu` | `None` | Scenario-only placement override | Passed to NNSight backend. |
| `row_subchunk_size` | `None` | Scenario-only backend tuning | Passed to NNSight backend. |
| `plan_feature_batch_size` | `False` | Scenario-only planner/resource knob | Public wrapper rejects planner-enabled use and directs users to NNSight compact output. |
| `auto_scale_feature_batch_size` | `False` | Deprecated/compat alias | Same wrapper rejection as planner. |
| `feature_batch_size_max` | `None` | Scenario-only planner/resource knob | Present but not part of the `phase4_overrides_requested` rejection subset by itself. |
| `feature_batch_target_reserved_fraction` | `0.9` | Scenario-only planner/resource knob | Meaningful with planner-enabled paths. |
| `feature_batch_min_free_fraction` | `0.05` | Scenario-only planner/resource knob | Meaningful with planner-enabled paths. |
| `feature_batch_probe_batches` | `1` | Scenario-only planner/resource knob | Meaningful with planner-enabled paths. |
| `diagnostic_feature_cap` | `None` | Debug/profiling semantic-changing knob | Exposed on public wrapper and backend. |
| `sparsification` | `None` | Experimental approximation/sparsification config | Exposed on public wrapper and backend. |
| `phase4_scheduler_mode` | `locality` | Scenario-only experimental scheduler knob | Only supported for NNSight backend. |
| `phase4_scheduler_debug` | `False` | Debug/telemetry knob | Only NNSight. |
| `phase4_scheduler_telemetry_detail` | `normal` | Debug/telemetry knob | Only NNSight. |
| `phase4_refresh_optimization` | `off` | Scenario-only optimization knob | Only NNSight. |
| `phase4_row_executor` | `batched` | Scenario-only optimization knob | Only NNSight. |
| `phase1_trace_batch_policy` | `legacy` | Scenario-only resource knob | Only NNSight. |
| `phase1_trace_batch_size_max` | `None` | Scenario-only resource knob | Only NNSight. |
| `phase4_refresh_policy` | `standard` | Scenario-only optimization knob | Only NNSight. |
| `phase4_refresh_interval_multiplier` | `1` | Scenario-only optimization knob | Only NNSight. |
| `phase4_ranker` | `argsort` | Scenario-only correctness-sensitive experimental knob | Only NNSight. |
| `row_store_cache_control` | `off` | Scenario-only platform/resource knob | Only NNSight. |
| `exact_encoder_residency` | `lazy` | Scenario-only memory-placement knob | Only NNSight. |

Important behavior:

- The wrapper forwards these settings only to NNSight.
- For non-NNSight backends, only the `phase4_overrides_requested` subset raises a
  `ValueError` today. Several exact/NNSight-only settings are silently ignored on
  the TransformerLens path, including `exact_trace_internal_dtype`, chunked replay
  window/prefetch/staging knobs, `row_subchunk_size`, planner sizing subknobs,
  `diagnostic_feature_cap`, and `sparsification` handling is delegated separately.
  This is a cleanup target: either document the ignored behavior explicitly in the
  library or add validation guards.
- The wrapper does **not** expose Track-A replay/capture parameters. Those exist
  on the NNSight backend and project CLI, not on the general public wrapper.

### `circuit_tracer/attribution/attribute_nnsight.py::attribute(...)`

This is the broadest library API and still contains both the old precision knob
and the new canonical precision knob.

Precision-related parameters:

| Parameter | Default | Classification | Current problem |
|---|---:|---|---|
| `exact_trace_internal_dtype` | `fp32` | Canonical precision knob | Good. This is the desired public precision contract. |
| `internal_precision` | `float64` | Deprecated/compatibility | Still public in the NNSight signature and docstring. The project compact wrapper derives and passes this from `exact_trace_internal_dtype`, but sibling `attribute.py::attribute(...)` currently does **not** pass it when forwarding to NNSight, so direct library use can still see old `float64` behavior. |

Precision ownership caveat:

- Project `trace_pipeline_chunked.py` derives `internal_precision` from
  `exact_trace_internal_dtype` before calling the library compact path.
- Sibling public `attribute.py::attribute(...)` forwards
  `exact_trace_internal_dtype` but does not derive/pass `internal_precision`; the
  NNSight backend therefore receives its own `internal_precision="float64"`
  default unless called through the project compact wrapper or direct callers set
  it.
- Phase 3 should decide whether sibling `attribute.py` should also derive/pass the
  compatibility value, or whether direct NNSight callers keep the old default with
  an explicit deprecation note.

Resolver constants and aliases:

- `_EXACT_TRACE_INTERNAL_DTYPE_BY_NAME` accepts `fp32`, `float32`,
  `torch.float32`, `fp64`, `float64`, `torch.float64`.
- `_resolve_exact_trace_internal_dtype(...)` accepts strings or `torch.dtype`
  values, but only resolves to `torch.float32` or `torch.float64`.

Replay/debug resolver constants:

| Resolver map | Accepted values | Classification |
|---|---|---|
| `_PHASE0_ACTIVATION_THRESHOLD_COMPARE_MODE_BY_NAME` | `baseline/default`, `bf16/bfloat16`, `fp32/float32/torch.float32`, `fp64/float64/torch.float64` | Track-A diagnostic |
| `_PHASE0_REPLAY_MODE_BY_NAME` | `disabled`, `donor_phase0` | Track-A replay |
| `_PHASE0_DONOR_CONTEXT_POLICY_BY_NAME` | `strict`, `warn` | Replay validation |
| `_PHASE3_REPLAY_MODE_BY_NAME` | `disabled`, `donor` | Track-A replay |
| `_PHASE3_REPLAY_VALIDATION_POLICY_BY_NAME` | `strict` | Replay validation |

Phase-4 / optimization resolver constants:

| Resolver map / default | Accepted values | Classification |
|---|---|---|
| `_PHASE4_SCHEDULER_MODE_ALIAS` | `legacy -> locality` | Deprecated alias |
| `_PHASE4_SCHEDULER_VERSION_BY_MODE` | `locality`, `planner_v1`, `planner_v2` | Scenario-only scheduler experiment |
| `_PHASE4_SCHEDULER_TELEMETRY_DETAIL_ALIAS` | `compact -> summary`, `full -> debug` | Deprecated/compat telemetry aliases |
| `_PHASE4_REFRESH_OPTIMIZATION_VERSION_BY_MODE` | `off`, `v1` | Scenario-only optimization |
| `_PHASE4_ROW_EXECUTOR_VERSION_BY_MODE` | `batched`, `streaming_v1` | Scenario-only optimization |
| `_PHASE1_TRACE_BATCH_POLICY_DEFAULT` | `legacy` | Scenario-only resource default |
| `_PHASE4_REFRESH_POLICY_DEFAULT` | `standard` | Scenario-only optimization default |
| `_PHASE4_RANKER_DEFAULT` | `argsort` | Scenario-only correctness-sensitive default |
| `_ROW_STORE_CACHE_CONTROL_DEFAULT` | `off` | Scenario-only platform/resource default |
| `_EXACT_ENCODER_RESIDENCY_DEFAULT` | `lazy` | Scenario-only memory-placement default |

Debug/replay parameters on the NNSight entrypoint:

| Parameter | Default | Classification |
|---|---:|---|
| `cross_cluster_debug` | `False` | Debug artifact knob |
| `capture_phase0_donor_bundle` | `False` | Debug artifact knob |
| `capture_phase3_seed_bundle` | `False` | Debug artifact knob |
| `capture_phase3_gradient_bundle` | `False` | Debug artifact knob |
| `capture_phase3_row_bundle` | `False` | Debug artifact knob |
| `capture_feature_semantic_descriptors` | `False` | Debug artifact knob |
| `semantic_descriptor_top_k` | `2048` | Debug artifact shape knob |
| `semantic_descriptor_dim` | `64` | Debug artifact shape knob |
| `phase0_donor_bundle` | `None` | Debug replay input |
| `phase0_replay_mode` | `disabled` | Debug replay knob |
| `phase0_donor_context_policy` | `strict` | Debug replay validation knob |
| `phase3_gradient_donor_bundle` | `None` | Debug replay input |
| `phase3_gradient_replay_mode` | `disabled` | Debug replay knob |
| `phase3_row_donor_bundle` | `None` | Debug replay input |
| `phase3_row_replay_mode` | `disabled` | Debug replay knob |
| `phase3_replay_validation_policy` | `strict` | Debug replay validation knob |
| `telemetry_max_events` | `None` | Debug/telemetry knob |
| `phase4_anomaly_debug` | `False` | Historical debug knob |
| `compact_output` | `False` | Backend execution-mode switch |

`compact_output` is not part of the project user-facing scenario taxonomy because
`trace_pipeline_chunked.py` chooses compact vs raw with `--save-raw`. In the
library backend it is a major mode switch: many exact-path optimization/debug
features are only meaningful when `compact_output=True`.

Environment-variable surfaces to remove/deprecate:

| Environment variable | Role | Classification |
|---|---|---|
| `PHASE4_ANOMALY_DEBUG` | Can activate Phase-4 anomaly debug scaffolding in the backend | Remove/deprecate; use explicit CLI/scenario `phase4_anomaly_debug` instead. |
| `CIRCUIT_TRACER_TELEMETRY_MAX_EVENTS` | Backend default for telemetry event cap when not passed explicitly | Remove/deprecate; use explicit CLI/scenario `telemetry_max_events` instead. |

Policy: environment variables should not be part of the steady-state debug/tuning
surface. They are not visible in scenario JSON or run configs unless copied
manually, so they are a provenance hazard on OSC/SLURM runs.

### Sibling tests already covering taxonomy assumptions

Relevant coverage in `../circuit-tracer_chunked/tests/test_attribute_nnsight_telemetry.py`:

- `test_exact_trace_internal_dtype_resolution_supports_fp32_and_fp64`,
- `test_exact_trace_internal_dtype_default_is_fp32_on_public_entrypoints`,
- `test_exact_trace_internal_dtype_resolution_rejects_unknown_value`,
- `test_phase4_scheduler_defaults_match_between_public_entrypoints`,
- `test_phase4_execution_flag_type_hints_include_new_modes`,
- row-store cache-control behavior tests,
- Phase-3 row effective-state tests added during Phase 1.

Relevant replay validation coverage:

- `../circuit-tracer_chunked/tests/test_phase3_replay_validation.py`,
- project-side replay/artifact tests under `tests/test_cross_cluster_debug_artifacts.py`,
  `tests/test_phase0_replay_matrix_compare.py`,
  `tests/test_phase3_seed_bundle_compare.py`, and
  `tests/test_semantic_feature_compare.py`.

Coverage gap: no project-side test currently asserts that canonical exact-bench
scenario generation excludes replay/debug keys from ordinary `fast`, `anomaly`,
and `long_eval` scenarios.

## Artifact schemas that are not knobs

Some names look like knobs in docs/tests but are emitted artifact fields.

| Name | Current role | Where seen | Classification |
|---|---|---|---|
| `phase0_boundary_fingerprints` | Cross-cluster debug payload block | `tests/test_cross_cluster_debug_artifacts.py`, historical Phase-0 boundary spec | Artifact schema field, not a scenario/API knob |
| `boundary_fingerprint` | Historical shorthand | historical docs/specs | Not a live knob found in current code |
| `compare_mode` | Generic shorthand | old wording/reports | Use exact knob name `phase0_activation_threshold_compare_mode` |
| `replay_validation` | Generic shorthand | old wording/reports | Use exact knob name `phase3_replay_validation_policy` |
| `capture_phase` | Generic shorthand | old wording/reports | Real knobs are per-artifact capture flags |

## Generated scenario inventory

The repo currently contains many generated JSON files, not all of which are
canonical current templates.

Inventory command used for this pass:

```bash
uv run python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

key_files = defaultdict(list)
for path in Path('experiments/generated').rglob('*.json'):
    data = json.loads(path.read_text())
    scenarios = data.get('scenarios') if isinstance(data, dict) else data
    if isinstance(data, dict) and isinstance(data.get('defaults'), dict):
        for key in data['defaults']:
            key_files[f'defaults.{key}'].append(str(path))
    if isinstance(scenarios, list):
        for scenario in scenarios:
            if isinstance(scenario, dict):
                for key in scenario:
                    key_files[key].append(str(path))
PY
```

High-level counts from the current tree:

- 87 generated JSON files were inspected.
- `decoder_chunk_size` appears in 62 generated files.
- `cross_batch_decoder_cache_bytes` appears in 61 generated files.
- scenario-level `exact_trace_internal_dtype` appears in 19 generated files.
- `defaults.exact_trace_internal_dtype` appears in 36 generated files.
- replay inputs appear only in a small Track-A subset:
  - `phase0_donor_bundle`: 5 files,
  - `phase3_gradient_donor_bundle`: 2 files,
  - `phase3_row_donor_bundle`: 3 files.
- Phase-1/Phase-4 optimization fields appear primarily in explicit optimization
  scenario files:
  - `phase1_trace_batch_policy`: 4 files,
  - `phase4_refresh_policy`: 4 files,
  - `phase4_ranker`: 4 files,
  - `row_store_cache_control`: 4 files,
  - `exact_encoder_residency`: 4 files,
  - `phase4_scheduler_mode`: 8 files,
  - `phase4_refresh_optimization`: 6 files,
  - `phase4_row_executor`: 6 files.

### Current canonical exact-bench generated files

These match the current package-style harness and should remain the ordinary
template set:

- `experiments/generated/exact_trace_bench/exact_trace_bench_fast_ascend_scenarios.json`
- `experiments/generated/exact_trace_bench/exact_trace_bench_fast_cardinal_scenarios.json`
- `experiments/generated/exact_trace_bench/exact_trace_bench_anomaly_ascend_scenarios.json`
- `experiments/generated/exact_trace_bench/exact_trace_bench_anomaly_cardinal_scenarios.json`
- `experiments/generated/exact_trace_bench/exact_trace_bench_long_eval_ascend_scenarios.json`
- `experiments/generated/exact_trace_bench/exact_trace_bench_long_eval_cardinal_scenarios.json`

Expected ordinary scenario-level knobs in these files:

- source/fixture metadata,
- batch sizes,
- `decoder_chunk_size`,
- `cross_batch_decoder_cache_bytes`,
- no Track-A replay/capture donor paths,
- no semantic descriptor capture,
- no non-default Phase-4 scheduler/refresh/ranker/executor experiments.

### Historical / one-off exact-bench generated files

These should be treated as historical/provenance or explicit debug/optimization
campaign inputs, not normal templates:

| File family | Why not normal |
|---|---|
| `matched_cross_cluster_*` | Track-A matched debug/parity campaign inputs. |
| `phase0_donor_capture_94_base_*` | Track-A donor capture. |
| `phase0_replay_matrix_94_base_*` | Track-A Phase-0 replay matrix. |
| `phase3_gradient_donor_capture_94_base_*` | Track-A Phase-3 donor capture. |
| `phase3_replay_matrix_94_base_*` | Track-A gradient/row replay matrix. |
| `phase3_row_replay_smoke_94_base_ascend.json` | Phase-1 row replay validation smoke. Keep as validation fixture, not ordinary template. |
| `quick_cross_cluster_fp64_828_*` | Historical quick fp64 debug. |
| `exact_trace_cache8g_interaction_ascend_scenarios.json` | Explicit cache/Phase-1/Phase-4 interaction experiment. |
| `exact_trace_hidden_knobs_planner_v1_ascend_scenarios.json` | Explicit hidden-knob planner experiment. |
| `exact_trace_wave1_cache_chunk_resweep_ascend_scenarios.json` | Explicit cache/chunk/optimization sweep. |
| root `experiments/generated/exact_trace_phase4_*` | Older optimization scenario outputs, outside current exact-bench package. |
| `weekend_exact_chunked_*` | Pre-consolidation/weekend harness generation. |
| `weekend_exact_chunked_fixtures_matched_debug/` | Historical matched-debug fixture directory. |

Cleanup implication: generated debug/optimization configs need either an archive
location or a `historical` marker so users do not copy them as current defaults.

## Current classification matrix

| Knob / field | Project surfaces | Library surfaces | Current default | Classification | Phase 3 action |
|---|---|---|---|---|---|
| `exact_trace_internal_dtype` | scenario defaults, CLI, run config | public wrapper + NNSight backend | `fp32` | Canonical default / public precision | Keep; add project scenario/default tests. |
| `internal_precision` | derived only by project compact wrapper | NNSight backend public param; sibling public wrapper does not pass it today | `float64` in backend signature | Deprecated/compatibility | Hide/deprecate direct public use; decide whether sibling public wrapper should derive/pass it. |
| `decoder_chunk_size` | canonical scenarios, CLI, model load | model/transcoder loading outside attribution API | CLI `256`, canonical `2048/4096` | Public resource/perf | Keep; document that changes can affect compact references. |
| `cross_batch_decoder_cache_bytes` | canonical scenarios, CLI, model load | model/transcoder loading outside attribution API | CLI `None`, canonical `0`, cache probes `8 GiB` | Public resource/perf | Keep; document exactness within fixed chunk size. |
| `phase0_activation_threshold_compare_mode` | CLI/scenario bridge/run config | NNSight backend | `baseline` | Debug/replay public | Keep surfaced; ensure canonical defaults stay `baseline` unless a compare-mode sweep opts in. |
| `cross_cluster_debug` | defaults, CLI, scenario bridge | NNSight backend | `False` | Debug/replay public | Keep surfaced; canonical scenarios must not enable by accident. |
| `capture_phase0_donor_bundle` | CLI/scenario bridge | NNSight backend | `False` | Debug/replay public | Keep surfaced for validation/debug campaigns. |
| `phase0_donor_bundle` | CLI/scenario bridge | NNSight backend | `None` | Debug/replay public input | Keep surfaced; require explicit replay mode. |
| `phase0_replay_mode` | CLI/scenario bridge | NNSight backend | `disabled` | Debug/replay public | Keep surfaced. |
| `phase0_donor_context_policy` | CLI/scenario bridge | NNSight backend | `strict` | Debug/replay public validation | Keep surfaced; default remains strict. |
| `capture_phase3_seed_bundle` | CLI/scenario bridge | NNSight backend | `False` | Debug/replay public artifact | Keep surfaced. |
| `capture_phase3_gradient_bundle` | CLI/scenario bridge | NNSight backend | `False` | Debug/replay public artifact | Keep surfaced. |
| `capture_phase3_row_bundle` | CLI/scenario bridge | NNSight backend | `False` | Debug/replay public artifact | Keep surfaced. |
| `phase3_gradient_donor_bundle` | CLI/scenario bridge | NNSight backend | `None` | Debug/replay public input | Keep surfaced; require replay mode. |
| `phase3_gradient_replay_mode` | CLI/scenario bridge | NNSight backend | `disabled` | Debug/replay public | Keep surfaced. |
| `phase3_row_donor_bundle` | CLI/scenario bridge | NNSight backend | `None` | Debug/replay public input | Keep surfaced; require replay mode. |
| `phase3_row_replay_mode` | CLI/scenario bridge | NNSight backend | `disabled` | Debug/replay public | Keep surfaced. |
| `phase3_replay_validation_policy` | CLI/scenario bridge | NNSight backend | `strict` | Debug/replay public validation | Keep surfaced; strict remains default. |
| `capture_feature_semantic_descriptors` | CLI/scenario bridge | NNSight backend | `False` | Debug/replay public artifact | Keep surfaced. |
| `semantic_descriptor_top_k` | CLI/scenario bridge | NNSight backend | `2048` | Debug artifact shape | Only meaningful with descriptor capture. |
| `semantic_descriptor_dim` | CLI/scenario bridge | NNSight backend | `64` | Debug artifact shape | Only meaningful with descriptor capture. |
| `phase1_trace_batch_policy` | scenario bridge, CLI | public wrapper + NNSight backend | `legacy` | Advanced public resource optimization | Keep surfaced; default-preserving tests. |
| `phase1_trace_batch_size_max` | scenario bridge, CLI | public wrapper + NNSight backend | `None` | Advanced public resource optimization | Keep surfaced. |
| `feature_batch_size_max` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `None` | Advanced public planner/resource | Keep surfaced. |
| `feature_batch_target_reserved_fraction` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `0.9` | Advanced public planner/resource | Keep surfaced. |
| `feature_batch_min_free_fraction` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `0.05` | Advanced public planner/resource | Keep surfaced. |
| `feature_batch_probe_batches` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `1` | Advanced public planner/resource | Keep surfaced. |
| `phase4_refresh_policy` | scenario bridge, CLI | public wrapper + NNSight backend | `standard` | Advanced public optimization | Keep surfaced. |
| `phase4_refresh_interval_multiplier` | scenario bridge, CLI | public wrapper + NNSight backend | `1` | Advanced public optimization | Keep surfaced. |
| `phase4_ranker` | scenario bridge, CLI | public wrapper + NNSight backend | `argsort` | Advanced public correctness-sensitive experiment | Keep surfaced; non-defaults need validation. |
| `row_store_cache_control` | scenario bridge, CLI | public wrapper + NNSight backend | `off` | Advanced public platform/resource optimization | Keep surfaced. |
| `exact_encoder_residency` | scenario bridge, CLI | public wrapper + NNSight backend | `lazy` | Advanced public memory placement | Keep surfaced. |
| `phase4_scheduler_mode` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `locality` | Advanced public scheduler experiment | Keep surfaced; avoid accidental default drift. |
| `phase4_scheduler_debug` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `False` | Advanced debug/telemetry public | Keep surfaced. |
| `phase4_scheduler_telemetry_detail` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `normal` | Advanced debug/telemetry public | Keep surfaced; default remains normal. |
| `phase4_refresh_optimization` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `off` | Advanced public optimization | Keep surfaced. |
| `phase4_row_executor` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `batched` | Advanced public optimization | Keep surfaced. |
| `phase4_anomaly_debug` | defaults, CLI, scenario bridge | NNSight backend | `False` | Debug/replay public legacy | Keep surfaced if still useful; document historical status. |
| `chunked_feature_replay_window` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | project defaults `None`, CLI/backend `4` | Advanced public backend tuning | Keep surfaced; document interaction with defaults. |
| `error_vector_prefetch_lookahead` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | project defaults `None`, CLI/backend `2` | Advanced public backend tuning | Keep surfaced; document interaction with defaults. |
| `stage_encoder_vecs_on_cpu` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `None` | Advanced public placement override | Keep surfaced. |
| `stage_error_vectors_on_cpu` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `None` | Advanced public placement override | Keep surfaced. |
| `row_subchunk_size` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `None` | Advanced public backend tuning | Keep surfaced. |
| `plan_feature_batch_size` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `False` | Advanced public resource planner | Keep surfaced; public wrapper currently rejects planner-enabled use. |
| `auto_scale_feature_batch_size` | defaults, CLI, scenario bridge | public wrapper + NNSight backend | `False` | Deprecated/compat alias | Deprecate in docs/tests; prefer `plan_feature_batch_size`. |
| `telemetry_max_events` | defaults, CLI, scenario bridge | NNSight backend | `None` | Advanced debug/telemetry public | Keep surfaced. |
| `diagnostic_feature_cap` | CLI/scenario bridge | public wrapper + NNSight backend | `None` | Advanced debug/profiling semantic-changing | Keep surfaced with clear warning; never canonical default. |
| `sparsify_per_layer_position_topk`, `sparsify_global_cap` | older sparsification scenario bridge | sparsification config | `None` | Advanced experimental approximation/sparsification | Keep surfaced separately from canonical exact defaults. |
| `compact_output` | selected by project `--save-raw` path, not a scenario field | NNSight backend | `False` | Backend execution-mode switch | Document as internal/backend mode; many exact/debug knobs require compact output. |
| `PHASE4_ANOMALY_DEBUG` | environment only | NNSight backend | unset | Deprecated env override | Remove/deprecate; use explicit CLI/scenario `phase4_anomaly_debug`. |
| `CIRCUIT_TRACER_TELEMETRY_MAX_EVENTS` | environment only | NNSight backend | unset/backend policy | Deprecated env override | Remove/deprecate; use explicit CLI/scenario `telemetry_max_events`. |

## Recommended cleanup actions

### P3.1 Add project-side guard tests first

Add lightweight tests for `experiments.exact_trace_bench` scenario generation and
command construction:

1. canonical `fast`, `anomaly`, and `long_eval` builders preserve the expected
   defaults for fully public and advanced-public knobs,
2. no canonical scenario accidentally enables Track-A replay/capture behavior;
   debug/replay knobs may be present as explicit false/disabled defaults,
3. defaults contain `exact_trace_internal_dtype=fp32`,
4. `decoder_chunk_size` and `cross_batch_decoder_cache_bytes` are present in
   scenario rows,
5. long-eval cache probes are the only canonical generated scenarios with
   `cross_batch_decoder_cache_bytes > 0`,
6. after merging `defaults | scenario`, `build_command(...)` for canonical
   scenarios does not include donor paths, enabled capture flags, semantic
   descriptor capture, or non-default advanced optimization experiments unless an
   explicit sweep/debug scenario requested them.

This should be login-node safe.

### P3.2 Split scenario allowlists

Replace or supplement `EXACT_MODE_KNOB_KEYS` with category-specific names. This
is for documentation, validation, and scenario generation hygiene — not for
hiding the advanced knobs.

Example categories:

- `STABLE_PUBLIC_SCENARIO_KEYS`,
- `ADVANCED_PUBLIC_TUNING_KEYS`,
- `DEBUG_REPLAY_PUBLIC_KEYS`,
- `TELEMETRY_KEYS`,
- `DEPRECATED_COMPAT_KEYS`,
- `PRIVATE_INTERNAL_KEYS`.

Do not remove the wide bridge in `run_sparsification_experiment.py`; that runner
should remain capable of executing explicit advanced sweep/debug scenario files.
The risk is accidental default drift or unlabeled historical configs, not the
existence of advanced public knobs.

### P3.3 Deprecate direct `internal_precision`

Sibling library cleanup should make `exact_trace_internal_dtype` the public
precision contract and push `internal_precision` down to compatibility plumbing.

Implementation order:

1. Update docstrings to say `internal_precision` is compatibility-only and, where
   possible, derived from `exact_trace_internal_dtype` for normal callers.
2. Add tests asserting public `attribute(...)` default is `fp32` and project
   wrapper passes derived backend precision.
3. Deprecate direct NNSight `internal_precision` immediately. Preferred behavior:
   direct callers should use `exact_trace_internal_dtype`; `internal_precision`
   should either be derived, warn on direct use, or become private compatibility
   plumbing during the library cleanup.

### P3.4 Remove environment debug overrides and classify hidden constants

Environment-variable debug/tuning overrides should not remain part of the steady
state API because they are invisible in scenario JSON unless copied manually.

Initial targets:

1. Replace `PHASE4_ANOMALY_DEBUG` usage with explicit CLI/scenario
   `phase4_anomaly_debug`.
2. Replace `CIRCUIT_TRACER_TELEMETRY_MAX_EVENTS` usage with explicit CLI/scenario
   `telemetry_max_events` or a normal in-code default.
3. Audit hard-coded internal policy constants, especially planner-v2 constants.
   If a constant affects behavior and is useful for sweeps, expose it as an
   advanced public knob with defaults/tests; otherwise keep it as a true private
   implementation detail only if it is not expected to vary, or delete the dead
   path.

### P3.5 Mark generated historical configs

Options, from least to most invasive:

1. Add an index file under `experiments/generated/README.md` stating which files
   are canonical and which are historical.
2. Optionally move old exact-bench debug scenario files to
   `experiments/generated/history/` or `docs/history/generated_scenarios/` after
   tests confirm no active workflow depends on their current paths.
3. Regenerate canonical exact-bench configs into a clean directory and archive
   everything else only if the README/index marker proves insufficient.

Recommended first step: add an index/README and tests before moving files.

### P3.6 Update docs after tests

After guard tests exist, update:

- `docs/harness.md` to link this taxonomy and separate stable public,
  advanced-public, and debug/replay-public knobs,
- `experiments/exact_trace_bench/README.md` to distinguish ordinary launch
  defaults from advanced sweep/debug scenarios,
- root `EXPERIMENTS.md` only if the baseline interpretation changes.

## Open questions before implementation

Resolved / clarified decisions from review:

1. Inline docs are not a blocker for our own near-term sweeps. They mean concise
   parameter-level help/docstrings so future users can tell whether a knob is
   expected to affect semantics, resource use, telemetry only, or artifact output.
   We can add those opportunistically while cleaning code.
2. `phase0_activation_threshold_compare_mode` is low priority. It literally
   changes the Phase-0 comparison path, but prior experiments did not improve or
   explain the current divergence; the current interpretation places the important
   issue in Phase-3 gradient drift and later amplification. Keep it only as a
   historical/diagnostic knob for now, not as a knob we expect to tune defaults
   around.
3. Historical generated scenario files can be handled during repo cleanup. A
   README/index marker is enough initially; physical moves are optional later.
4. Direct NNSight `internal_precision` should be deprecated immediately in favor
   of `exact_trace_internal_dtype`.

Remaining open question:

1. Which currently hard-coded planner/scheduler constants are worth exposing for
   sweeps, and which should be deleted or treated as true implementation details?
