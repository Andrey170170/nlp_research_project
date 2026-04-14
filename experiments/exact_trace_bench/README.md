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

### 1.5) Render a scratch-backed launch plan

```bash
uv run python -m experiments.exact_trace_bench launch-plan \
  --cluster ascend \
  --scenarios-file experiments/generated/exact_trace_bench/exact_trace_bench_fast_ascend_scenarios.json \
  --immutable-workspace
```

This prints an `sbatch` command that uses:

- a scratch output root under `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/...`
- and, optionally, an immutable workspace snapshot for `nnsight` safety.

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

This copies the workspace to scratch and marks it read-only by default so
long-running jobs do not observe mid-run source edits.
