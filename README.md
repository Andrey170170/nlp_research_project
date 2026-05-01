# Temporal Circuit Stability for LLM Reliability

This repository contains the course/research code for investigating whether
temporal stability of attribution circuits can be used as a reliability signal for
math reasoning. In practice, much of the project became measurement and systems
work around exact attribution-graph tracing: chunked tracing, numerical stability,
cross-cluster parity diagnostics, and optimization harnesses.

The codebase is intentionally experimental. Many scripts are one-off harnesses
used to reproduce a specific bug, benchmark, or cluster run. Treat this README as
orientation rather than a stable package API.

## Repository layout

Important entry points:

- `trace_pipeline.py` — original multi-prompt tracing pipeline.
- `trace_pipeline_chunked.py` — exact/chunked tracing pipeline used by most later
  experiments.
- `evaluate.py` / `analyze.py` — earlier evaluation and analysis scripts.
- `circuit_utils.py` — shared compact-graph and metric utilities.
- `experiments/` — scenario builders, one-off analysis scripts, extraction
  scripts, plotting helpers, and benchmark utilities.
- `scripts/` — SLURM submission scripts for OSC clusters.
- `prefix_caching/` — cross-run prefix-caching prototype direction.
- `tests/` — lightweight correctness/unit checks.

Most GPU work was run on OSC via SLURM. Do **not** launch model-loading or tracing
jobs on a login node.

## Required sibling checkout

This project depends on a local editable fork of `circuit-tracer`. The checkout
must be a sibling directory named exactly `circuit-tracer_chunked`:

```text
parent_directory/
├── nlp_research_project/
└── circuit-tracer_chunked/
```

The path is encoded in `pyproject.toml`:

```toml
[tool.uv.sources]
circuit-tracer = { path = "../circuit-tracer_chunked", editable = true }
```

If the sibling fork is missing or has the wrong directory name, imports and
environment setup will fail.

## Branches and provenance

The `main` branch is not a complete record of every experiment described in the
final report. Much of the work remains on separate project and sibling-library
branches.

Project branches used during the investigation include, for example:

- `exact-trace-bench-harness`
- `exact-trace-bench-opt`
- `exact-trace-bench-optimization`

Sibling `circuit-tracer_chunked` branches include, for example:

- `exact-trace-hidden-knobs`
- `exact-trace-hidden-knobs-opt`

For exact reproduction, use the branch/commit pair recorded in the relevant
experiment log or workspace snapshot. A result depends on **both** this repository
and the sibling `circuit-tracer_chunked` checkout.

## Environment setup

Python requirement: `>=3.12,<3.13`.

Preferred setup uses `uv`:

```bash
# from nlp_research_project/
uv sync
```

Run Python through `uv`:

```bash
uv run python --version
uv run ruff check .
```

If you are not using `uv`, create a Python 3.12 environment and install the
project in editable mode after placing the sibling tracer checkout correctly:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ../circuit-tracer_chunked
pip install -e .
```

The explicit sibling install is needed for non-`uv` environments because
`tool.uv.sources` is only interpreted by `uv`.

`uv` is what was used for project validation, so prefer it when possible.

## Running experiments

There is no single canonical experiment command. Most experiments follow this
pattern:

1. Generate or select a scenario/config file with a script under `experiments/`.
2. Submit an appropriate SLURM script from `scripts/`.
3. Write outputs to scratch.
4. Run an extraction/analysis script over the output directory.

Example: generate one family of exact/chunked benchmark scenarios:

```bash
uv run python experiments/build_weekend_exact_chunked_benchmark_configs.py
```

Example: count scenarios before launching an array job:

```bash
uv run python experiments/print_scenario_count.py \
  --scenarios-file experiments/generated/weekend_exact_chunked_wave1_ascend_scenarios.json
```

Example: submit an Ascend exact/chunked benchmark array:

```bash
sbatch --time=02:30:00 \
  --array=0-$(($(uv run python experiments/print_scenario_count.py \
    --scenarios-file experiments/generated/weekend_exact_chunked_wave1_ascend_scenarios.json)-1)) \
  --export=ALL,SCENARIOS_FILE=experiments/generated/weekend_exact_chunked_wave1_ascend_scenarios.json,OUTPUT_ROOT=/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked/ascend/wave1 \
  scripts/trace_weekend_exact_chunked.ascend.sbatch
```

Example: submit the newer decoder-aware comparison harness on Ascend:

```bash
uv run python experiments/build_decoder_aware_comparison_configs.py
sbatch scripts/decoder_aware_comparison.ascend.sbatch
```

Always inspect the scenario JSON and SLURM script before launching. Many scripts
were created for a specific run family and may hard-code scratch paths, cluster
assumptions, walltimes, or branch-specific options.

## Common script families

Some useful script groups:

- Exact/chunked tracing:
  - `trace_pipeline_chunked.py`
  - `scripts/trace_pipeline_chunked.sbatch`
  - `scripts/trace_weekend_exact_chunked.ascend.sbatch`
  - `scripts/trace_weekend_exact_chunked.cardinal.sbatch`
- Exact smoke/reference runs:
  - `scripts/trace_exact_smoke.ascend.sbatch`
  - `scripts/trace_exact_reference_overnight.ascend.sbatch`
- Feature-distribution/scaling:
  - `experiments/build_feature_distribution_analysis_configs.py`
  - `experiments/run_feature_distribution_analysis.py`
  - `scripts/feature_distribution_analysis.ascend.sbatch`
- Sparsification prototypes:
  - `experiments/build_sparsification_experiment_configs.py`
  - `experiments/run_sparsification_experiment.py`
  - `scripts/trace_sparsification_experiment.ascend.sbatch`
- Prefix caching prototype:
  - `prefix_caching/`
  - `scripts/trace_prefix_cache_bench.ascend.sbatch`
- Decoder-aware comparison work on `main`:
  - `experiments/build_decoder_aware_comparison_configs.py`
  - `experiments/analyze_decoder_aware_comparison.py`
  - `scripts/decoder_aware_comparison.ascend.sbatch`

## Output locations

Most large outputs were written outside the repository on OSC scratch, commonly
under paths like:

```text
/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/
/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked/
```

Do not commit generated trace artifacts, model outputs, large `.pt` graphs, or
scratch extraction directories unless a small derived artifact is intentionally
being tracked.

## Local validation

Safe login-node commands are limited to lightweight checks:

```bash
uv run ruff check .
uv run pytest tests -q
```

Avoid commands that load Gemma, download model weights, run nnsight tracing, or
perform heavy CPU/GPU work outside a SLURM job.

## Notes for future users

- Expect rough edges. The project evolved as an investigation rather than as a
  stable library.
- Prefer immutable workspace snapshots or explicit branch/commit notes for serious
  runs.
- Check both repositories: this project and `../circuit-tracer_chunked`.
- If a command appears to depend on an unmerged option, switch to the branch where
  that option was developed.
- For final-report reproduction, consult the experiment notes and the relevant
  branch-specific logs rather than assuming `main` contains every result.
