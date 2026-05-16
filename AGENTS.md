# Agent operating instructions

Status: Durable repo policy
Last updated: 2026-05-16

This file is the source of truth for agents working in this repository. Keep
`CLAUDE.md` as a pointer only.

## Project context

- Research project: temporal circuit stability for LLM reliability, with most
  current work centered on exact/chunked attribution tracing and validation.
- Model stack: Gemma-3-1B-IT with GemmaScope-2 cross-layer transcoders.
- Local editable library dependency: sibling checkout `../circuit-tracer_chunked`.
- Environment manager: `uv`; run Python as `uv run ...` unless already inside the
  uv-managed `.venv`.

Required workspace layout:

```text
parent_directory/
├── nlp_research_project/
└── circuit-tracer_chunked/
```

The project repo alone is not enough for provenance. Exact-trace results depend
on both this repo and the sibling library checkout.

## OSC / login-node safety

This repo runs on Ohio Supercomputer Center systems. GPU/model work must happen
inside SLURM jobs, never on login nodes.

Safe on login nodes:

- `uv run ruff check .`
- `uv run ty check .`
- lightweight unit tests that do not load models or download weights
- scenario inspection/counting commands

SLURM-only:

- Gemma/GemmaScope model loading,
- nnsight tracing,
- exact attribution runs,
- heavy extraction/analysis over large scratch trees,
- anything that may trigger GPU allocation or model downloads.

When in doubt, do not run it locally; prepare or inspect the SLURM command.

## Documentation roles

- `README.md` — contributor orientation and safe workflow summary.
- `AGENTS.md` — durable operating policy for agents.
- `CLAUDE.md` — pointer to `AGENTS.md`; do not duplicate policy there.
- `docs/README.md` — documentation index.
- `docs/harness.md` — current exact-bench harness overview.
- `docs/current_project_roadmap.md` — current scratch roadmap for active work.
- `docs/post_consolidation_cleanup_spec.md` — durable cleanup strategy.
- `EXPERIMENTS.md` — compact current baseline, run-family meanings, and current
  interpretation.
- `experiments/logs/YYYY-MM.jsonl` — append-only structured experiment records.
- `docs/history/` — archived/superseded plans and long-form historical logs.
- `PLAN.md` — local scratch only; keep it out of git history.

Update rules:

- Put durable workflow rules here.
- Put active task steps in `docs/current_project_roadmap.md`.
- Keep root `EXPERIMENTS.md` compact; only record baseline-changing decisions and
  current interpretation there.
- Put verbose structured experiment provenance in `experiments/logs/YYYY-MM.jsonl`.
- Move superseded docs to `docs/history/` rather than leaving stale current-looking
  docs in top-level `docs/`.

## Current exact-trace baseline

- Canonical exact-trace dtype: `exact_trace_internal_dtype=fp32`.
- Stable row-L1 denominator behavior is part of the validated baseline.
- Canonical prompt gates:
  - `828_base` and `361_base` in `fast`,
  - `94_base` in `anomaly`.
- Track-A interpretation: Ascend/Cardinal divergence is mainly Phase-3 gradient
  drift, with later stages amplifying that drift into compact graph differences.
- Keep Track-A replay/debug machinery as internal validation infrastructure, but
  do not expose it as the ordinary workflow.

## Harness and run placement

Current harness:

- module: `experiments/exact_trace_bench/`
- overview: `docs/harness.md`
- CLI help: `uv run python -m experiments.exact_trace_bench --help`

Scratch outputs should be organized by cluster and tier only:

- cluster: `ascend` / `cardinal`
- tier: `fast` / `anomaly` / `long_eval`

Use `run_id`, `run_name`, `run_description`, `run_goal`, and scenario names to
distinguish debug campaigns. Do not introduce ordinary scratch buckets like
`matched_debug`; those are historical provenance only.

Before any serious run, record:

1. project repo branch, commit, and dirty files,
2. sibling `../circuit-tracer_chunked` branch, commit, and dirty files,
3. whether the launch uses a live workspace or immutable workspace snapshot,
4. scratch output root and SLURM job IDs.

## Git hygiene

- Do not commit generated trace artifacts, model outputs, large `.pt` graphs, or
  scratch extraction directories unless a small derived artifact is intentionally
  promoted as a fixture.
- Keep upstream PR work small and separate from local-fork cleanup.
- Do not commit local scratch planning files such as `PLAN.md`.
- Review both project and sibling-library diffs before committing validation or
  exact-trace changes.
