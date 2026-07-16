# Agent operating instructions

Status: Durable repo policy
Last updated: 2026-07-10

This file is the source of truth for agents working in this repository. Keep
`CLAUDE.md` as a pointer only.

## Project context

- Research project: temporal circuit stability for LLM reliability, with most
  current work centered on a large-scale circuit-tracing harness: scenario
  generation, batching, SLURM orchestration, provenance, extraction, validation,
  and cross-run analysis.
- Current calibration stack: Gemma 3 1B/4B/12B with GemmaScope-2 CLT and PLT
  providers. The governor/runtime contract must remain provider-agnostic.
- Local editable library dependency: sibling checkout `../circuit-tracer_chunked`;
  that library owns the in-round tracing implementation while this repo owns the
  experiment harness around it.
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

Possible future direction: reusable harness pieces such as batched tracing
orchestration may move into `../circuit-tracer_chunked` so the library can run
more independently. Until that is an explicit task, keep this repo as the
orchestration/provenance layer and avoid opportunistic cross-repo merges.

The Phase B pure governor resolver remains advisory through Phase D
explicit-mechanism work and Phase C2 ("cleanup strikes again"). Phase E is the
first phase allowed to consume plans at runtime, after both the Phase D and C2
immutable Granite gates pass. Do not treat a resolved plan as an executed
configuration or promote its outputs to launch defaults before then.

Tracing cleanup phases must extract logging and telemetry mechanics into deep
sibling modules. Tracing algorithms should emit typed domain events or
lifecycle spans; schema construction, sequencing, resource sampling,
incremental sinks/flushing, and human log rendering belong to the observability
subsystem.

Phase C2 replaces the complete project-to-sibling tracing path before Phase E.
Legacy Python API compatibility is not a requirement. Build one canonical
typed runtime around meaningful domain objects with subsystem-owned invariants;
do not replace flat argument lists with generic context/config/input bags.
Migrate sibling and project callers atomically, then delete
`attribute_nnsight.py`, flat `attribute(...)` routing, legacy translators and
kwargs, private compatibility re-exports, and obsolete project trace pipelines.
There must be no dual runtime. Moving large methods behind imports is not a
cleanup result unless the new modules have coherent ownership and the top-level
trace flow is readable. The normative design is
`docs/tracing_runtime_rewrite_spec.md`.

## CHPC / login-node safety

This repo now runs primarily on Utah CHPC Granite. Historical Ascend/Cardinal
OSC paths remain in the codebase for provenance, but new launches should use the
`granite` cluster profile unless intentionally reconstructing old OSC work.
GPU/model work must happen inside SLURM jobs, never on login nodes.

Filesystem search safety on HPC:

- Never run broad filesystem searches rooted at `/` or other HPC-wide roots.
- Restrict file discovery and content searches to the repo, the sibling checkout,
  exact known scratch/log roots, or exact temp paths already identified.
- Prefer explicit paths over globbing large shared filesystems; ask before
  expanding an unknown search scope.

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
- `CHPC.md` — practical CHPC GPU/RAM/walltime routing and launch checklist.
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
  - `828_base`, `361_base`, and `94_base` in `fast` for new work.
- Track-A interpretation: Ascend/Cardinal divergence is mainly Phase-3 gradient
  drift, with later stages amplifying that drift into compact graph differences.
- Keep Track-A replay/debug machinery as internal validation infrastructure, but
  do not expose it as the ordinary workflow.

## Harness and run placement

Current harness:

- module: `src/nlp_research_project/exact_trace_bench/`
- overview: `docs/harness.md`
- CLI help: `uv run exact-trace-bench --help`

Scratch outputs should be organized by cluster and operational class:

- cluster: `granite` for new CHPC work; `ascend` / `cardinal` only for historical OSC provenance
- operational class: legacy scenario tiers still exist in code as `fast` / `anomaly` /
  `long_eval`, but new planning should classify work by operational resource
  class instead: `setup_prefetch`, `smoke`, `baseline`, `sweep`,
  `long_trace`, `full_answer`, and `analysis`.

Use `run_id`, `run_name`, `run_description`, `run_goal`, and scenario names to
distinguish campaigns. Do not introduce ordinary scratch buckets like
`matched_debug`; those are historical provenance only.

Practical GPU/RAM/walltime routing is documented in `CHPC.md`; detailed pool
inventory is in `docs/chpc_resource_pools.md`. In short: use Granite
`rai-gpu-grn` H200 nodes for scientific baselines and large traces, the
SOC-associated RTX PRO 6000 Blackwell route when its class policy permits 1B
functional testing, and Notchpeak `marasovic-gpu-np` A100 nodes for routine lab
smokes that do not need 1T+ host RAM. Treat guest GPUs as preemptable.

Workspace immutability is the default for every SLURM launch that executes
project code, including smoke, baseline, sweep, long-trace, full-answer, and
analysis jobs:

- create one immutable read-only snapshot containing both the project and the
  sibling library, and reuse it across a campaign when the code state is shared;
- packaged launch commands must snapshot automatically unless an existing
  verified snapshot is supplied;
- direct `sbatch` use must set `WORKSPACE_ROOT` and `LIB_WORKSPACE_ROOT` to
  the verified snapshot paths instead of relying on `SLURM_SUBMIT_DIR`;
- environments, secrets, model caches, and output roots remain external to the
  snapshot;
- a live-workspace launch is an explicit exceptional override only. Record the
  reason and both dirty states, label the run as live, and do not edit either
  runtime checkout until the job terminates.

Before any serious run, record:

1. project repo branch, commit, and dirty files,
2. sibling `../circuit-tracer_chunked` branch, commit, and dirty files,
3. immutable snapshot container/project/library roots and manifest,
4. any explicit live-workspace override and its rationale,
5. scratch output root and SLURM job IDs.

## Git hygiene

- Do not commit generated trace artifacts, model outputs, large `.pt` graphs, or
  scratch extraction directories unless a small derived artifact is intentionally
  promoted as a fixture.
- Keep upstream PR work small and separate from local-fork cleanup.
- Do not commit local scratch planning files such as `PLAN.md`.
- Review both project and sibling-library diffs before committing validation or
  exact-trace changes.
