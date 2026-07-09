# Documentation index

Status: Current docs map
Last updated: 2026-07-01

This directory mixes current-state references, active specs, and historical
material. Use the sections below to separate descriptive maps from working
plans.

## Entry points

| File | Role |
|---|---|
| `current_state.md` | Workspace-level current-state overview |
| `architecture/README.md` | Descriptive code/workspace map index |

## Current authoritative docs

| File | Role |
|---|---|
| `../README.md` | Contributor orientation and safe CHPC workflow |
| `../AGENTS.md` | Durable repo policy and operating conventions |
| `../CLAUDE.md` | Pointer back to `AGENTS.md` |
| `../EXPERIMENTS.md` | Compact experiment baseline/index and current interpretation |
| `harness.md` | Current exact-bench harness overview |
| `chpc_setup.md` | CHPC Granite setup, caches, SLURM profiles, and migration checklist |
| `chpc_resource_pools.md` | CHPC GPU pool candidates and job-class routing |
| `metric_calibration.md` | Phase-0/Phase-1 decoder-cache and metric calibration workflow |
| `../scripts/README.md` | Script entry-point notes |
| `../experiments/logs/README.md` | Structured experiment log guidance |
| `../src/nlp_research_project/exact_trace_bench/README.md` | Package-level harness notes |

## Active specs and scratch docs

| File | Role |
|---|---|
| `current_project_roadmap.md` | Current scratch roadmap for the active cleanup phase |
| `memory_governor_rearchitecture_spec.md` | Target design: budget-driven memory governor + library rearchitecture, with execution plan (PLT merge first, then probe campaign) |
| `knob_api_taxonomy.md` | Exact-trace knob/API taxonomy and Phase-3 cleanup map |
| `full_answer_harness_spec.md` | Track-2 full-answer / multi-token tracing harness design |
| `exact_trace_sweep_campaign_spec.md` | Next exact-trace sweep campaign plan and wave structure |
| `post_consolidation_cleanup_spec.md` | Durable cleanup strategy after Track-0B consolidation |
| `plt_clt_optimization_parity_spec.md` | Current PLT/CLT optimization parity plan and validation strategy |
| `phase4_refresh_optimization_spec.md` | Current Phase-4 normalization/RSS optimization guidance |
| `next_exact_optimization_paths_spec.md` | Current optimization option map |
| `tracing_profiling_spec.md` | Current profiling/telemetry design |
| `phase4_scheduler_v2_spec.md` | Proposed/deferred Phase-4 scheduler-v2 design |

## Reports and scouting outputs

| Location | Role |
|---|---|
| `../reports/**` | Generated/scouting outputs; useful for inspection, not binding decisions |

## Historical docs

Historical plans, old proposals, superseded specs, and duplicate inventories
live in `history/`.

Use them for provenance only. Do not treat them as current workflow unless a
current doc explicitly points to a section there.

## Update rules

- Keep baseline/interpretation changes in `../EXPERIMENTS.md`.
- Keep current execution steps in `current_project_roadmap.md`.
- Keep implementation tradeoffs in the active spec that owns that area.
- Treat reports as evidence unless a current doc explicitly promotes them.
- Prefer the new current-state and architecture pages for code/workspace maps.
