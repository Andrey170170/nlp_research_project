# Workspace map

Status: Current-state map
Last updated: 2026-07-13

## Two-repo layout

| Repo | Role |
|---|---|
| `nlp_research_project/` | Research repo, docs, harness, experiments, and launch orchestration |
| `../circuit-tracer_chunked/` | Editable sibling tracing library used at runtime and in validation |

## Integration boundaries

| Boundary | What crosses it |
|---|---|
| Package import boundary | Project builds typed requests and imports the canonical `circuit_tracer` tracing API from the sibling checkout; project-specific request construction and artifacts stay under `exact_trace_bench/trace_runtime/` |
| Workspace verification | Project workspace checks for both repo roots and importability before launch-sensitive work |
| SLURM launch boundary | Jobs receive workspace paths through environment variables such as `WORKSPACE_ROOT` and `LIB_WORKSPACE_ROOT` |
| Artifact boundary | Project writes benchmark outputs under the configured scratch root; sibling results supply graphs, fingerprints, admission evidence, and telemetry |

## Current workspace assumptions

- Python `>=3.12,<3.13`
- Editable sibling checkout at `../circuit-tracer_chunked`
- Scratch root: `/scratch/general/vast/$USER/nlp_research_project/exact_trace_bench`
- New exact-trace GPU/model-loading work uses immutable two-repo snapshots on Utah CHPC Granite

## Current-state pointers

- Project-side harness and CLI maps live in `project_code_map.md`.
- Sibling package/API maps live in `sibling_circuit_tracer_chunked_code_map.md`.
