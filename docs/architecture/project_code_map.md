# Project code map

Status: Current implementation map; Phase C2 Granite validation pending
Last updated: 2026-07-13

Unless noted, paths are relative to the project repository root.

## Package and CLI

| Surface | Location |
|---|---|
| Console script `exact-trace-bench` | `pyproject.toml` -> `nlp_research_project.exact_trace_bench.cli:main` |
| Harness package | `src/nlp_research_project/exact_trace_bench/` |
| Canonical trace boundary | `src/nlp_research_project/exact_trace_bench/trace_runtime/` |
| Compact graph I/O | `src/nlp_research_project/exact_trace_bench/compact_io.py` |

## Ownership map

| Area | Owner |
|---|---|
| Scenario and request construction | `trace_runtime/request.py`, `trace_runtime/campaign.py`, `scenarios/*` |
| Model/provider loading | `trace_runtime/provider.py` |
| Completion and trace orchestration | `trace_runtime/generation.py`, `trace_runtime/tracing.py` |
| Project artifact layout | `trace_runtime/artifacts.py`, `completion_workspace.py`, `step_artifacts.py` |
| Incremental sibling telemetry consumption | `trace_runtime/observations.py`, `trace_runtime/telemetry.py` |
| Compact graph conversion | `trace_runtime/compact_graph.py`, `compact_io.py` |
| Full-answer selection/session orchestration | `full_answer/*` |
| Workspace snapshots and import verification | `workspace.py` |
| SLURM launch plans and templates | `jobs.py`, `presets.py`, `slurm/exact_trace_bench/*` |
| Extraction and comparison | `extract.py`, `graph_compare.py`, comparison helpers |

The project builds typed sibling `TraceRequest` objects and calls
`trace_one` or `open_session`. It does not own an attribution algorithm or a
flat scenario-to-library knob relay. The sibling returns graph, fingerprint,
admission, and telemetry evidence; this package owns experiment-specific files
and campaign provenance.

## Runtime flow

```text
scenario JSON
  -> trace_runtime request/provider construction
  -> circuit_tracer trace_one/open_session
  -> project compact/artifact writers
  -> summary, extraction, and comparison
```

Full-answer tracing uses the same sibling session API. Sequence/window reuse is
explicit in the request/session objects rather than implemented by a second
project trace pipeline.

## CLI areas

| Area | Commands | Main implementation |
|---|---|---|
| Scenarios and fixtures | `build-scenarios`, wave builders, `describe-fixtures` | `scenarios/*`, `fixtures.py` |
| Launch and workspace safety | `launch-plan`, submit commands, `snapshot-workspace`, `verify-imports` | `jobs.py`, `presets.py`, `workspace.py` |
| Extraction and comparison | `extract`, `compare-compact`, replay/semantic comparisons | `extract.py`, `graph_compare.py`, comparison helpers |
| Full-answer | build/run/aggregate/audit/trajectory commands | `full_answer/*`, `jobs.py` |
| Analysis and calibration | temporal, role, decoder-signature, metric-calibration commands | analysis/calibration modules |

All commands remain registered through `exact_trace_bench/cli.py`; command
registration is routing only, while domain behavior belongs to the modules
listed above.

## Artifact flow

The project owns `scenario.json`, `run.log`, `result.json`,
`summary.json`, scenario metrics, run metadata, compact NPZ artifacts,
full-answer shards, and cross-run analysis. Sibling telemetry is streamed and
consumed without reconstructing or resequencing its event schema.

## Validation

Login-safe checks cover request construction, artifacts, full-answer sessions,
stale-path architecture rules, launch/snapshot behavior, and failure
propagation. GPU/model loading and the immutable 1B CLT/PLT C2 gate remain
SLURM-only.

Useful commands:

- `uv run ruff check .`
- `uv run pytest -q`
- `uv run exact-trace-bench snapshot-workspace --print-path-only`
- `uv run exact-trace-bench verify-imports --workspace-root <snapshot-project-root>`

## Historical surfaces

Generated historical scenarios and archived scripts may describe older runtime
variants, but they are not executable integration guidance. Phase C2 removes
obsolete root trace pipelines rather than preserving compatibility with the
deleted flat sibling API.
