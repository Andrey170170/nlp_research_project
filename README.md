# Temporal Circuit Stability for LLM Reliability

This repository investigates whether temporal stability of internal attribution
circuits during autoregressive generation can predict answer correctness in math
reasoning. The current codebase is also the project harness for exact/chunked
attribution tracing, cross-cluster parity diagnostics, and optimization work.

The repo is experimental, but `main` is now the consolidated working baseline.
Use this README for orientation, `AGENTS.md` for durable operating rules, and
`EXPERIMENTS.md` for current experiment provenance and interpretation.

## Current baseline

| Item | Current value |
|---|---|
| Project repo | `nlp_research_project` on local `main` |
| Sibling library | `../circuit-tracer_chunked` on local `main` |
| Editable dependency | `circuit-tracer = { path = "../circuit-tracer_chunked", editable = true }` |
| Model stack | Gemma-3-1B-IT + GemmaScope-2 cross-layer transcoders |
| Canonical exact-trace dtype | `exact_trace_internal_dtype=fp32` |
| Current benchmark harness | `src/nlp_research_project/exact_trace_bench/` |
| Scratch root | `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/` |

Every serious run depends on **both** this repository and the sibling
`../circuit-tracer_chunked` checkout. Record both branch/commit states before
launching SLURM jobs.

## OSC safety boundary

This project runs on Ohio Supercomputer Center systems. Do **not** load models,
download weights, run nnsight tracing, or launch heavy CPU/GPU work on login
nodes.

Safe on login nodes:

```bash
uv run ruff check .
uv run ty check .
uv run pytest tests -q
```

Only run lightweight tests locally. If a test or script loads Gemma/GemmaScope,
uses GPUs, runs exact tracing, or scans large scratch trees, submit it through
SLURM instead.

## Required layout

The sibling circuit-tracer fork must be adjacent to this repo:

```text
parent_directory/
├── nlp_research_project/
└── circuit-tracer_chunked/
```

The path is encoded in `pyproject.toml`, so missing or misnamed sibling checkouts
will break imports and SLURM runs.

## Environment

Python requirement: `>=3.12,<3.13`.

Use `uv`:

```bash
uv sync
uv run python --version
```

All Python invocations in this repo should go through `uv run` or the `.venv`
created by uv.

## Current harness workflow

The canonical exact-bench harness lives at:

- `src/nlp_research_project/exact_trace_bench/`
- `docs/harness.md`

Typical flow:

```bash
# Inspect CLI options.
uv run exact-trace-bench --help

# Generate current scenario configs.
uv run exact-trace-bench build-scenarios --all-tiers --all-clusters

# Render a launch plan before submitting.
uv run exact-trace-bench launch-plan \
  --cluster ascend \
  --scenarios-file experiments/generated/exact_trace_bench/exact_trace_bench_fast_ascend_scenarios.json \
  --immutable-workspace
```

Submit GPU/model-loading work only via SLURM. Preset helpers and wrapper scripts
are documented in `src/nlp_research_project/exact_trace_bench/README.md`.

Run placement convention:

- cluster: `ascend` / `cardinal`
- tier: `fast` / `anomaly` / `long_eval`

Use `run_id`, `run_name`, `run_description`, `run_goal`, and scenario names to
distinguish campaigns. Do not introduce new ordinary scratch buckets like
`matched_debug`; those are historical provenance only.

## Documentation map

- `AGENTS.md` — durable repo policy and workflow conventions.
- `EXPERIMENTS.md` — compact current baseline, run-family meanings, and current
  interpretation.
- `experiments/logs/` — append-only structured experiment records.
- `docs/README.md` — documentation index.
- `docs/harness.md` — current exact-bench harness overview.
- `docs/current_project_roadmap.md` — current scratch roadmap for active cleanup.
- `docs/history/` — archived/superseded plans and long-form investigation logs.

## Repository layout

Important entry points:

- `trace_pipeline_chunked.py` — fork-native exact/chunked tracing entrypoint.
- `trace_pipeline.py` — older multi-prompt tracing pipeline.
- `evaluate.py` / `analyze.py` — earlier correctness evaluation and analysis.
- `circuit_utils.py` — compact graph metrics and `.npz` utilities.
- `src/nlp_research_project/exact_trace_bench/` — current benchmark setup/extraction/compare
  harness.
- `slurm/exact_trace_bench/` — canonical exact-bench SLURM templates used by
  the CLI.
- `scripts/archive/` — historical scripts/wrappers; not current launch templates.
- `tests/` — lightweight local checks where possible; GPU/model tests must be
  marked or run through SLURM.

Large generated artifacts, raw `.pt` graphs, model outputs, and scratch
extractions should stay out of git unless a small derived artifact is explicitly
chosen as a fixture.
