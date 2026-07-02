# Current state overview

Status: Current-state inventory
Last updated: 2026-07-01

This document is a descriptive workspace map, not a target architecture, rule
set, or debt plan.

## Workspace roles

| Area | Role |
|---|---|
| `nlp_research_project/` | Main research repo: exact-trace benchmark harness, docs, experiments, and current workflows |
| `../circuit-tracer_chunked/` | Editable sibling library: tracing/attribution runtime used by the project |

The current exact-trace workflow depends on both checkouts being present.

## Documentation status taxonomy

| Status | Meaning |
|---|---|
| Authoritative | Current source-of-truth docs for contributor guidance, harness usage, and baseline interpretation |
| Active specs | Live scratch/spec documents that describe ongoing work or near-term design choices |
| Reports | Generated or scouting outputs; useful for inspection, not binding decisions |
| Historical | Archived/superseded material kept for provenance only |

## Authoritative docs

| File | Role |
|---|---|
| `README.md` | Contributor orientation and safe workflow summary |
| `AGENTS.md` | Durable repo policy and operating conventions |
| `CLAUDE.md` | Pointer back to `AGENTS.md` |
| `EXPERIMENTS.md` | Compact baseline/index and current interpretation |
| `docs/README.md` | Documentation index |
| `docs/harness.md` | Current exact-bench harness overview |
| `docs/metric_calibration.md` | Metric calibration workflow |
| `scripts/README.md` | Script entry-point notes |
| `experiments/logs/README.md` | Structured experiment log guidance |
| `src/nlp_research_project/exact_trace_bench/README.md` | Package-level harness notes |

## Active specs and scratch docs

| File | Role |
|---|---|
| `docs/current_project_roadmap.md` | Active scratch roadmap |
| `docs/post_consolidation_cleanup_spec.md` | Cleanup strategy |
| `docs/knob_api_taxonomy.md` | Knob/API taxonomy |
| `docs/full_answer_harness_spec.md` | Full-answer harness design |
| `docs/exact_trace_sweep_campaign_spec.md` | Sweep campaign plan |
| `docs/tracing_profiling_spec.md` | Profiling/telemetry design |
| `docs/phase4_scheduler_v2_spec.md` | Deferred scheduler-v2 proposal |
| `docs/phase4_refresh_optimization_spec.md` | Phase-4 refresh optimization notes |
| `docs/next_exact_optimization_paths_spec.md` | Optimization option map |
| `docs/plt_clt_optimization_parity_spec.md` | PLT/CLT parity notes |

## Reports and historical material

| Location | Role |
|---|---|
| `reports/**` | Generated/scouting outputs; inspect as evidence, not decisions |
| `docs/history/**` | Archived/superseded docs for provenance |

## Current workflow boundaries

- The current fast fixtures are `828_base`, `361_base`, and `94_base`; `94_base`
  also carries historical anomaly provenance.
- Descriptive architecture details now live under `docs/architecture/`.
- Current baseline decisions stay in `EXPERIMENTS.md`; active work stays in the owning spec.
- Reports are not treated as source-of-truth unless a current doc explicitly cites them.

## Pointers

- Start with `docs/README.md` for the current doc map.
- Use `docs/architecture/README.md` for descriptive code/workspace maps.
- Use `docs/harness.md` and `src/nlp_research_project/exact_trace_bench/README.md` for current benchmark entry points.
