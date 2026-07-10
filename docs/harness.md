# Current exact-bench harness

Status: Current harness overview
Last updated: 2026-07-09

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
mechanisms, and streaming telemetry sinks. Existing `attribute(...)` remains a
compatibility facade during migration. Experiment-specific token selection,
sharding, SLURM submission, artifact layout, and analysis stay here.

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

Scratch output placement should stay organized by cluster and tier only:

- `granite/fast`
- `granite/anomaly`
- `granite/long_eval`

Historical OSC provenance may still contain:

- `ascend/fast`
- `ascend/anomaly`
- `ascend/long_eval`
- `cardinal/fast`
- `cardinal/anomaly`
- `cardinal/long_eval`

Use scenario names, `run_id`, `run_name`, `run_description`, and `run_goal` to
distinguish debug campaigns. Do not introduce new ordinary buckets such as
`matched_debug`.

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
