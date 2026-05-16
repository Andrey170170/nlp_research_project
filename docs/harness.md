# Current exact-bench harness

Status: Current harness overview
Last updated: 2026-05-16

The current exact-trace benchmark harness is centered on the package-style module:

- `experiments/exact_trace_bench/`

The detailed CLI reference remains in:

- `experiments/exact_trace_bench/README.md`

This page records the current operating model and boundaries so old one-off
scripts do not look like the canonical path.

## Canonical workflow

1. Build or select scenarios under `experiments/generated/exact_trace_bench/`.
2. Render or submit a launch plan through `experiments.exact_trace_bench`.
3. Run GPU/model-loading work only inside SLURM jobs.
4. Write artifacts to `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/`.
5. Extract and compare compact outputs with the exact-bench extraction/comparison
   helpers.
6. Record baseline-changing results in root `EXPERIMENTS.md` and append structured
   records under `experiments/logs/`.

## Current fixture/tier convention

| Fixture | Tier | Purpose |
|---|---|---|
| `828_base` | `fast` | normal quick validation/debug |
| `361_base` | `fast` | normal quick validation/debug |
| `94_base` | `anomaly` | anomaly/parity watch gate |
| late fixtures | `long_eval` | longer exact-bench evaluation tier |

Scratch output placement should stay organized by cluster and tier only:

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

Normal public/resource surface:

- `exact_trace_internal_dtype`
- `decoder_chunk_size`
- `cross_batch_decoder_cache_bytes`

Track-A replay/debug controls should be explicit scenario/debug settings, not
ordinary benchmark defaults. Phase 3 of the cleanup plan will make this taxonomy
stricter in code and tests.

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

Before any serious run, record both repository states:

1. project repo branch/commit/dirty files,
2. sibling `../circuit-tracer_chunked` branch/commit/dirty files.

The sibling library is part of the experiment definition because SLURM launches
import the editable checkout at that relative path.

## Historical harness artifacts

Old generated scenarios and configs containing `matched_debug`, weekend benchmark
names, or pre-consolidation branch assumptions are provenance artifacts. They may
remain useful for reconstructing old runs, but should not be copied as templates
for new scenarios without checking current defaults.
