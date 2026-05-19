# exact_trace_bench

Practical scaffold for exact-trace benchmark setup, extraction, graph comparison,
and immutable workspace snapshots.

## What is included

- Canonical fixture tiers:
  - base: `828_base`, `361_base`
  - anomaly: `94_base`
  - late: `828_late`, `361_late`, `94_late`
- Scenario generation for three tiers: `fast`, `anomaly`, `long_eval`
- Scratch-root defaults under:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench`
- Extraction/aggregation helpers for `result.json`, `run.log`, and SLURM logs
- Compact graph comparison helpers for prompt94-style analysis (`step_*.npz`)
- Immutable workspace snapshot helper for launch safety
- Exact-mode knob taxonomy documented in `docs/knob_api_taxonomy.md`, with
  stable public, advanced sweep, debug/replay, telemetry, and deprecated/compat
  categories.

Immutable workspace snapshots copy:

- this project repo, and
- the sibling editable `../circuit-tracer_chunked` library declared in `pyproject.toml`

so the benchmark can run against frozen project + library code together.

## CLI usage

Run via module entrypoint (recommended):

```bash
uv run python -m experiments.exact_trace_bench --help
```

### 1) Generate scenario configs

```bash
uv run python -m experiments.exact_trace_bench build-scenarios --all-tiers --all-clusters
```

Writes JSON configs to:

- `experiments/generated/exact_trace_bench/`

The generated-config index at `experiments/generated/README.md` distinguishes
current canonical templates from historical one-off debug/sweep configs.

### Scenario knob categories

Ordinary exact-bench templates keep the visible scenario rows small: fixture
metadata, batch sizes, `decoder_chunk_size`, and
`cross_batch_decoder_cache_bytes`. The defaults block pins the canonical precision
contract with `exact_trace_internal_dtype=fp32`.

Advanced public knobs remain available for explicit sweep configs:

- Phase-1 trace-batch sizing,
- Phase-4 scheduler/refresh/ranker/executor controls,
- row-store/cache/residency controls,
- chunked replay, prefetch, staging, row-subchunk, and feature-batch planner
  controls.

Debug/replay knobs are opt-in validation tooling. Use them only in explicit
debug/replay scenario files and record donor/capture provenance. This includes
Phase-0 and Phase-3 donor capture/replay, semantic descriptor capture,
cross-cluster debug summaries, and telemetry caps.

Deprecated/compatibility knobs are kept only to avoid breaking old scenario files;
prefer the canonical names in new configs.

### 1.5) Render a scratch-backed launch plan

```bash
uv run python -m experiments.exact_trace_bench launch-plan \
  --cluster ascend \
  --scenarios-file experiments/generated/exact_trace_bench/exact_trace_bench_fast_ascend_scenarios.json \
  --immutable-workspace
```

This prints an `sbatch` command that uses:

- a scratch output root under `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/...`
  with an auto-generated run folder (`<base>/<timestamp>_<run_slug>`), so
  launches no longer collide when reusing the same tier/cluster base
- and, optionally, an immutable workspace snapshot for `nnsight` safety.

You can attach run metadata for later extraction/disambiguation:

```bash
uv run python -m experiments.exact_trace_bench launch-plan \
  --cluster ascend \
  --scenarios-file experiments/generated/exact_trace_bench/exact_trace_bench_fast_ascend_scenarios.json \
  --run-name "ascend fast sanity" \
  --run-description "post-change smoke run" \
  --run-goal "confirm no launch collisions"
```

Supported metadata flags on both `launch-plan` and `submit-preset`:

- `--run-id` (optional explicit run folder id)
- `--run-name`
- `--run-description`
- `--run-goal`

`launch-plan` also selects the sbatch script automatically from the scenario
JSON `resource_profile` metadata:

- `standard` → normal fast/anomaly job profile
- `long_eval_high_mem` → high-memory long-eval profile

This matters in particular for:

- Ascend long evals, which should use the quad / high-memory script
- Cardinal long evals, which should use a higher-memory script than the fast baseline

### 1.6) Use preset submit helpers

For the common workflows, you can skip the long commands and use either the CLI
presets or the wrapper scripts in `scripts/`.

Preset meanings:

- `fast-*` = submit `fast` + `anomaly`
- `full-*` = submit `fast` + `anomaly` + `long_eval`

CLI form:

```bash
uv run python -m experiments.exact_trace_bench submit-preset --preset fast-ascend
uv run python -m experiments.exact_trace_bench submit-preset --preset full-all
```

Preset submissions also populate sensible default run metadata (name,
description, goal) even if you do not pass explicit metadata flags.

Wrapper scripts:

```bash
scripts/exact_trace_bench_fast_ascend.sh
scripts/exact_trace_bench_fast_cardinal.sh
scripts/exact_trace_bench_full_ascend.sh
scripts/exact_trace_bench_full_cardinal.sh
scripts/exact_trace_bench_fast_all.sh
scripts/exact_trace_bench_full_all.sh
```

All preset submitters default to immutable workspace snapshots. Add
`--no-immutable-workspace` if you intentionally want to run against the live tree,
or `--print-only` to inspect the generated plans without calling `sbatch`.

### 2) Extract benchmark tables

```bash
uv run python -m experiments.exact_trace_bench extract \
  --input-root /fs/scratch/PAS3272/kopanev.1/exact_trace_bench \
  --output-dir experiments/extracted/exact_trace_bench \
  --logs-dir /path/to/benchmark/slurm/logs
```

Outputs:

- `benchmark_index.csv/.jsonl`
- `runlog_summary.csv`
- `slurm_err_summary.csv` (unless `--skip-slurm`)
- `benchmark_enriched.csv/.jsonl`

If `--logs-dir` is omitted, SLURM log parsing is skipped to avoid mixing unrelated historical logs.

### 3) Compare compact outputs (prompt94-style)

```bash
uv run python -m experiments.exact_trace_bench compare-compact \
  /path/to/ascend/artifacts \
  /path/to/cardinal/artifacts \
  --output-json experiments/extracted/exact_trace_bench/prompt94_compare.json
```

### 4) Create immutable workspace snapshot

```bash
uv run python -m experiments.exact_trace_bench snapshot-workspace --print-path-only
```

This copies the project workspace plus the sibling `circuit-tracer_chunked`
library to scratch and marks them read-only by default so long-running jobs do
not observe mid-run source edits.

To keep snapshotting fast, the project copy excludes large local artifact dirs
such as:

- `experiments/explore/`
- `experiments/traces/`
- `experiments/extracted/`
- `experiments/figures/`

### 5) Verify import resolution

```bash
uv run python -m experiments.exact_trace_bench verify-imports \
  --workspace-root /fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/<id>/nlp_research_project
```

This prints the resolved import paths for:

- `trace_pipeline_chunked`
- `circuit_tracer`

and is useful for checking that immutable runs will import the snapped library
copy rather than the live editable checkout.
