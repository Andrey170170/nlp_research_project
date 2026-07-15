# Current state overview

Status: Current-state inventory
Last updated: 2026-07-15

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
| Active plans/specs | Binding documents for ongoing work and near-term design choices |
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
| `docs/memory_governor_rearchitecture_spec.md` | Current sibling governor/runtime target contract |
| `docs/tracing_runtime_rewrite_spec.md` | Normative Phase C2 tracing-runtime target |
| `plans/2026-07-03_governor_rearch.md` | Active gated Phase B-D, C2, E-F plan |
| `docs/metric_calibration.md` | Metric calibration workflow |
| `scripts/README.md` | Script entry-point notes |
| `experiments/logs/README.md` | Structured experiment log guidance |
| `src/nlp_research_project/exact_trace_bench/README.md` | Package-level harness notes |

## Active plan and specs

| File | Role |
|---|---|
| `docs/current_project_roadmap.md` | Active scratch roadmap |
| `docs/memory_governor_rearchitecture_spec.md` | Governor/runtime target contract |
| `docs/tracing_runtime_rewrite_spec.md` | Phase C2 target architecture and work packages |
| `docs/knob_api_taxonomy.md` | Knob/API taxonomy |
| `plans/2026-07-03_governor_rearch.md` | Active gated Phase B-D, C2, E-F execution plan |

## Reference and deferred designs

| File | Role |
|---|---|
| `docs/post_consolidation_cleanup_spec.md` | Consolidation rationale; current roadmap supersedes its sequence |
| `docs/tracing_profiling_spec.md` | Implemented telemetry schema/tooling reference |
| `docs/phase4_refresh_optimization_spec.md` | Historical Phase-4 constraints and evidence |
| `docs/phase4_scheduler_v2_spec.md` | Deferred scheduler design |

## Reports and historical material

| Location | Role |
|---|---|
| `reports/**` | Generated/scouting outputs; inspect as evidence, not decisions |
| `docs/history/**` | Archived/completed/superseded docs for provenance |

## Current workflow boundaries

- The canonical base fixtures are `828_base`, `361_base`, and `94_base`; `94_base`
  also carries historical anomaly provenance. New CHPC run placement uses
  operational classes rather than those legacy scenario tiers.
- Phase A is closed for implementation purposes. Phase B governor contracts and
  the pure resolver are complete at sibling `phase-b-governor-contract@0ce3f96`.
  Phase C1 structural runtime/observability extraction and its Granite gate are
  complete. Phase D mechanism validation and the Phase C2 canonical runtime
  rewrite are complete. Phase E governor v0.3 implementation and CPU gates are
  complete: safety/support contracts, bounded independent search, staged phase
  cost models, measured-unit refinement, and replay-cache ownership are wired.
  The implementation is sibling `phase-b-governor-contract@07bfefb`. Its
  two-run immutable Granite gate precedes the multi-model calibration sweep and
  Phase F governed harness consolidation.
- Descriptive architecture details now live under `docs/architecture/`.
- Current baseline decisions stay in `EXPERIMENTS.md`; active work stays in the owning spec.
- Reports are not treated as source-of-truth unless a current doc explicitly cites them.

## Pointers

- Start with `docs/README.md` for the current doc map.
- Use `docs/architecture/README.md` for descriptive code/workspace maps.
- Use `docs/harness.md` and `src/nlp_research_project/exact_trace_bench/README.md` for current benchmark entry points.
