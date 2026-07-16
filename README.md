# Temporal Circuit Stability for LLM Reliability

This repository investigates whether temporal stability of internal attribution
circuits during autoregressive generation can predict answer correctness in math
reasoning. The current codebase is also the project harness for exact/chunked
attribution tracing, cross-cluster parity diagnostics, and optimization work.

The repo is experimental. Use this README for orientation, `AGENTS.md` for
durable operating rules, `CHPC.md` for practical GPU/RAM/walltime routing, and
`EXPERIMENTS.md` for the exact project and sibling commits that define the
current baseline. Do not infer experiment provenance from whichever local
branches happen to be checked out.

## Current baseline

| Item | Current value |
|---|---|
| Project repo | `nlp_research_project`; exact branch/commit recorded per run |
| Sibling library | `../circuit-tracer_chunked`; exact branch/commit recorded per run |
| Editable dependency | `circuit-tracer = { path = "../circuit-tracer_chunked", editable = true }` |
| Current calibration stack | Gemma 3 1B/4B/12B + GemmaScope-2 CLT/PLT providers |
| Canonical exact-trace dtype | `exact_trace_internal_dtype=fp32` |
| Governor status | Phase D and C2 gates passed; the first Phase E gate exposed one-shot resolver and recompute-cost defects, so staged constrained optimizer correction is active |
| Current benchmark harness | `src/nlp_research_project/exact_trace_bench/` |
| Scratch root | `/scratch/general/vast/$USER/nlp_research_project/exact_trace_bench/` |

Every serious run depends on **both** this repository and the sibling
`../circuit-tracer_chunked` checkout. Record both branch/commit states before
launching SLURM jobs.

## CHPC safety boundary

This project now runs primarily on Utah CHPC Granite. Do **not** load models,
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
UV_CACHE_DIR=.uv-cache uv sync
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

# Generate legacy-tier scenario configs.
uv run exact-trace-bench build-scenarios --all-tiers --all-clusters

# Render a launch plan before submitting.
uv run exact-trace-bench launch-plan \
  --cluster granite \
  --scenarios-file experiments/generated/exact_trace_bench/exact_trace_bench_fast_granite_scenarios.json \
  --immutable-workspace
```

Submit GPU/model-loading work only via SLURM. Preset helpers and wrapper scripts
are documented in `src/nlp_research_project/exact_trace_bench/README.md`.

Run placement convention for new work:

- cluster: `granite`
- operational class: `setup_prefetch`, `smoke`, `baseline`, `sweep`,
  `long_trace`, `full_answer`, or `analysis`

The scenario schema still contains legacy `fast` / `anomaly` / `long_eval`
tiers. Treat those as fixture metadata, not as the project-level split for new
scratch roots or campaign planning.

Use `run_id`, `run_name`, `run_description`, `run_goal`, and scenario names to
distinguish campaigns. Do not introduce new ordinary scratch buckets like
`matched_debug`; those are historical provenance only.

## Documentation map

- `AGENTS.md` — durable repo policy and workflow conventions.
- `CHPC.md` — practical Utah CHPC GPU, memory, walltime, and launch guidance.
- `EXPERIMENTS.md` — compact current baseline, run-family meanings, and current
  interpretation.
- `experiments/logs/` — append-only structured experiment records.
- `docs/README.md` — documentation index.
- `docs/harness.md` — current exact-bench harness overview.
- `docs/current_project_roadmap.md` — current gated Phase B-D, C2, E-F execution roadmap.
- `docs/history/` — archived/superseded plans and long-form investigation logs.

## Repository layout

Important entry points:

- `src/nlp_research_project/exact_trace_bench/trace_runtime/` — canonical
  project-to-sibling tracing requests, scenario execution, and project-owned
  artifacts.
- `src/nlp_research_project/exact_trace_bench/compact_io.py` — compact graph
  metrics and `.npz` helpers.
- `evaluate.py` / `analyze.py` — earlier correctness evaluation and analysis.
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
