# Current exact-bench harness

Status: Current harness overview
Last updated: 2026-08-08

The current exact-trace benchmark harness is centered on the package-style module:

- `src/nlp_research_project/exact_trace_bench/`

The detailed CLI reference remains in:

- `src/nlp_research_project/exact_trace_bench/README.md`

This page records the current operating model and boundaries so old one-off
scripts do not look like the canonical path.

## Canonical workflow

1. Build or select scenarios under `experiments/generated/exact_trace_bench/`.
2. Render or submit a launch plan through `nlp_research_project.exact_trace_bench`.
3. Run GPU/model-loading work only inside SLURM jobs.
4. Let the CLI select templates from `slurm/exact_trace_bench/`, create or reuse
   an immutable read-only project+sibling snapshot, and execute only from the
   resolved snapshot paths.
5. Write artifacts to `/scratch/general/vast/$USER/nlp_research_project/exact_trace_bench/`
   unless `EXACT_TRACE_BENCH_SCRATCH_ROOT` is set.
6. Extract and compare compact outputs with the exact-bench extraction/comparison
   helpers.
7. Record baseline-changing results in root `EXPERIMENTS.md` and append structured
   records under `experiments/logs/`.

## Target runtime boundary

The project remains the experiment harness. It owns fixtures and scenarios,
campaign construction, SLURM/CHPC policy, workspace snapshots, two-repository
provenance, experiment directory layouts, extraction, comparison, and scientific
interpretation.

Reusable tracing runtime behavior should move behind first-class sibling-library
APIs in `../circuit-tracer_chunked`:

- `trace_one(...)` for one independent request;
- `trace_batch(...)` for multiple independent requests sharing one loaded
  model/provider runtime;
- `open_session(...)` for sequenced tracing and explicit full-sequence/window
  reuse.

The sibling target also owns typed request/semantics/resource/plan/result
contracts, model/transcoder provider loading, the memory governor, execution
mechanisms, and streaming telemetry sinks. Phase C2 atomically migrates this
harness to the canonical sibling runtime and deletes `attribute_nnsight`, flat
`attribute(...)`, and legacy translators; compatibility wrappers are not a
requirement. Experiment-specific token selection, sharding, SLURM submission,
artifact layout, and analysis stay here. See
`docs/tracing_runtime_rewrite_spec.md`.

This boundary is a Python package API, not a Git submodule boundary. Continue to
use the editable sibling dependency during development. Every SLURM launch that
executes project code uses an immutable two-repo snapshot by default, including
smokes and analysis jobs.

Direct `sbatch` submission must not rely on a template's
`WORKSPACE_ROOT=$SLURM_SUBMIT_DIR` fallback. Supply verified snapshot
`WORKSPACE_ROOT` and `LIB_WORKSPACE_ROOT` values explicitly. A live-workspace
override is exceptional and must be labeled and recorded with a rationale and
both dirty states; neither live checkout may be edited while such a job runs.

## Current Job Classes

The code still has legacy scenario tiers named `fast`, `anomaly`, and
`long_eval`. For new CHPC work, use operational job classes when planning and
naming runs:

| Job class | Purpose | Typical pool |
|---|---|---|
| `setup_prefetch` | Download model/transcoder weights and build caches | CPU partition |
| `smoke` | Minimal model/load/trace sanity checks | lab A100 or guest GPU |
| `baseline` | Rebuild canonical validated traces | H200/H200NVL or A100 80GB |
| `sweep` | Parameter/resource sweeps over the harness | H200/H200NVL |
| `long_trace` | Large-context or high-memory traces | H200/H200NVL with 1T+ host RAM |
| `full_answer` | Multi-token/full-answer trace campaigns | H200/H200NVL with 1T+ host RAM |
| `analysis` | Extraction, comparison, plotting | CPU or high-memory CPU |

Keep the legacy tier field in scenario JSON when the current generator needs it,
but do not let `fast/anomaly/full` be the conceptual split for new campaigns.

## Current fixture convention

| Fixture | Tier | Purpose |
|---|---|---|
| `828_base` | `fast` | normal quick validation/debug |
| `361_base` | `fast` | normal quick validation/debug |
| `94_base` | `fast` | normal base fixture for new work; historical anomaly paths remain historical provenance |
| late fixtures | `long_eval` | longer exact-bench evaluation tier |

New scratch output placement is organized by cluster and operational class:

- `granite/setup_prefetch`
- `granite/smoke`
- `granite/baseline`
- `granite/sweep`
- `granite/long_trace`
- `granite/full_answer`
- `granite/analysis`

The validated 12B setup baseline is under
`granite/baseline/chpc-baseline-gemma3-12b-20260710-04`. At b64/c4096 its
fixtures required 5.8--6.4 hours. Keep the 600G host request for comparable
12B strict baselines until repeated runs explain the observed RSS spread.

Legacy scenario tiers and historical OSC provenance may still contain:

- `granite/fast`
- `granite/anomaly`
- `granite/long_eval`

- `ascend/fast`
- `ascend/anomaly`
- `ascend/long_eval`
- `cardinal/fast`
- `cardinal/anomaly`
- `cardinal/long_eval`

Use scenario names, `run_id`, `run_name`, `run_description`, and `run_goal` to
distinguish campaigns within an operational class. Do not introduce new
ordinary buckets such as `matched_debug`.

## Canonical exact-trace knobs

Detailed mapping: `docs/knob_api_taxonomy.md`.

Stable public/resource surface:

- `exact_trace_internal_dtype`
- `decoder_chunk_size`
- `cross_batch_decoder_cache_bytes`

Advanced public/research-tuning knobs remain intentionally available for sweeps,
including Phase-1 trace-batch sizing, Phase-4 scheduler/refresh/ranker/executor
controls, row-store/cache/residency controls, and feature-batch planner controls.
They should be treated as explicit sweep dimensions, not hidden defaults.

Backward execution has one canonical resolver. `backward_engine_mode` retains
the named presets `duplicated_lanes`, `single_forward_serial_vjp`, and
`single_forward_batched_vjp`; its absent/default behavior remains
`duplicated_lanes`. Diagnostic callers may instead provide both
`forward_graph_mode` (`logical_capacity | single_lane`) and `vjp_kernel_mode`
(`nnsight_injected | autograd_serial | autograd_batched`). A preset and explicit
pair are mutually exclusive, partial pairs and unsupported combinations fail
closed, and inherited defaults are removed before an explicit pair is frozen.

Launch and result artifacts record the canonical preset, forward-graph mode,
VJP kernel, and physical forward-lane count. Post-run validation checks all four
against the effective execution descriptor, in addition to the required
feature-row mechanism. Source-layer grouping, autograd calls/timing, cotangent
peak bytes, and failure stage remain engine telemetry. Do not infer selection
from the requested scenario alone, and do not combine either autograd VJP mode
with Phase-3 gradient-donor replay.

Phase-3 localization flags are available directly from the full-answer spec
CLI: `--capture-phase3-gradient-bundle`, `--capture-phase3-row-bundle`, and
`--capture-phase3-seed-bundle`. The frozen three-arm 4B/256 campaign is
[`ls4_4b_vjp_decomposition_diagnostic_v1.json`](../experiments/performance_campaigns/ls4_4b_vjp_decomposition_diagnostic_v1.json).

Track-A replay/debug controls are also public for validation work, but they must
stay opt-in and provenance-heavy: donor paths, capture flags, semantic descriptor
capture, and verbose telemetry should appear only in explicit debug/replay
scenario files.

Deprecated/compatibility surfaces:

- direct NNSight `internal_precision` is compatibility-only; use
  `exact_trace_internal_dtype`,
- `auto_scale_feature_batch_size` is a legacy alias for planner behavior,
- environment-variable debug overrides are not part of the supported provenance
  surface; use explicit CLI/scenario knobs.

## Local vs SLURM boundary

Safe on a login node:

- `uv run ruff check .`
- `uv run ty check .`
- lightweight unit tests that do not load models or download weights
- scenario-count/inspection commands

SLURM-only:

- Gemma/GemmaScope model loading,
- nnsight tracing,
- exact attribution runs,
- heavy extraction/analysis over large scratch trees.

Before any serious run, record:

1. project repo branch/commit/dirty files,
2. sibling `../circuit-tracer_chunked` branch/commit/dirty files,
3. snapshot manifest and project/library snapshot roots,
4. output root and SLURM job IDs.

The sibling library is part of the experiment definition because SLURM launches
import its snapshot through the resolved editable-source layout.

## Historical harness artifacts

Old generated scenarios and configs containing `matched_debug`, weekend benchmark
names, or pre-consolidation branch assumptions are provenance artifacts. They may
remain useful for reconstructing old runs, but should not be copied as templates
for new scenarios without checking current defaults.

Old executable entrypoints live under `scripts/archive/`. Current exact-bench
templates live under `slurm/exact_trace_bench/` and should normally be reached
through `uv run exact-trace-bench ...`, not direct root scripts.

See `experiments/generated/README.md` for the current generated-config index.
