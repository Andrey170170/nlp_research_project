# Workspace map

Status: Current-state map
Last updated: 2026-07-01

## Two-repo layout

| Repo | Role |
|---|---|
| `nlp_research_project/` | Research repo, docs, harness, experiments, and launch orchestration |
| `../circuit-tracer_chunked/` | Editable sibling tracing library used at runtime and in validation |

## Integration boundaries

| Boundary | What crosses it |
|---|---|
| Package import boundary | Project imports `circuit_tracer` from the sibling checkout; project-local runtime modules such as `trace_pipeline.py`, `trace_pipeline_chunked.py`, and `circuit_utils.py` stay in this repo |
| Workspace verification | Project workspace checks for both repo roots and importability before launch-sensitive work |
| SLURM launch boundary | Jobs receive workspace paths through environment variables such as `WORKSPACE_ROOT` and `LIB_WORKSPACE_ROOT` |
| Artifact boundary | Project writes benchmark outputs under the configured scratch root; sibling code supplies tracing/model internals |

## Current workspace assumptions

- Python `>=3.12,<3.13`
- Editable sibling checkout at `../circuit-tracer_chunked`
- Scratch root: `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench`
- Exact-trace GPU/model-loading work expects SLURM context on OSC systems

## Current-state pointers

- Project-side harness and CLI maps live in `project_code_map.md`.
- Sibling package/API maps live in `sibling_circuit_tracer_chunked_code_map.md`.
