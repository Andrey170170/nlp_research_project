# Documentation index

Status: Current docs map
Last updated: 2026-08-23

This directory mixes current-state references, active specs, and historical
material. Use the sections below to separate descriptive maps from working
plans.

## Entry points

| File | Role |
|---|---|
| `current_state.md` | Workspace-level current-state overview |
| `architecture/README.md` | Descriptive code/workspace map index; C2 implementation state with validation status called out explicitly |

## Current authoritative docs

| File | Role |
|---|---|
| `../README.md` | Contributor orientation and safe CHPC workflow |
| `../AGENTS.md` | Durable repo policy and operating conventions |
| `../CHPC.md` | Practical CHPC GPU routing, request sizing, walltime, and launch checklist |
| `../CLAUDE.md` | Pointer back to `AGENTS.md` |
| `../EXPERIMENTS.md` | Compact experiment baseline/index and current interpretation |
| `harness.md` | Current exact-bench harness overview |
| `performance_optimization_loop.md` | Isolated H200 performance and graph-parity loop on `perf/exact-trace-loop` |
| `exact_trace_optimization_registry.md` | Single triage ledger for immediate, candidate, deferred, and rejected performance/code-quality ideas |
| `chpc_setup.md` | CHPC Granite setup, caches, SLURM profiles, and migration checklist |
| `chpc_resource_pools.md` | CHPC GPU pool candidates and job-class routing |
| `metric_calibration.md` | Phase-0/Phase-1 decoder-cache and metric calibration workflow |
| `../scripts/README.md` | Script entry-point notes |
| `../experiments/logs/README.md` | Structured experiment log guidance |
| `../src/nlp_research_project/exact_trace_bench/README.md` | Package-level harness notes |

## Active plan and specs

| File | Role |
|---|---|
| `current_project_roadmap.md` | Current gated Phase B-D, C2, E-F roadmap |
| `../plans/2026-08-21_exact_trace_optimization_and_refactoring.md` | Active survivability-first correctness, optimization, scaling, selector, and refactoring sequence for the isolated performance worktree |
| `../plans/2026-07-29_exact_trace_performance_optimization_loop.md` | Completed short-prefix promotion and prefix/prompt/model scaling plan through LS5; retain as execution provenance |
| `memory_governor_rearchitecture_spec.md` | Implementation-ready target design for the budget-driven governor and sibling runtime |
| `governor_calibration_matrix.md` | Staged observation campaign for resource, runtime, and fidelity response models |
| `tracing_runtime_rewrite_spec.md` | Normative Phase C2 end-to-end tracing architecture and atomic migration plan |
| `knob_api_taxonomy.md` | Phase B requirements document for knob ownership, caste, validation, and cost formulas |

## Reference and deferred designs

| File | Role |
|---|---|
| `post_consolidation_cleanup_spec.md` | Consolidation rationale and cleanup constraints; current roadmap supersedes its sequence |
| `phase4_refresh_optimization_spec.md` | Historical Phase-4 memory/refresh evidence and constraints |
| `tracing_profiling_spec.md` | Implemented telemetry schema/tooling reference; governor spec owns extensions |
| `phase4_scheduler_v2_spec.md` | Deferred scheduler-v2 design; not active Phase B work |

## Reports and scouting outputs

| Location | Role |
|---|---|
| `../reports/**` | Generated/scouting outputs; useful for inspection, not binding decisions |
| `../experiments/performance_campaigns/**` | Validated short-prefix mechanism ledgers and typed scaling campaign manifests |

## Historical docs

Historical plans, completed implementation specs, old proposals, and duplicate
inventories live in `history/`. This includes the completed full-answer and
PLT/CLT parity plans, the old sweep campaign, and the pre-governor optimization
ordering.

Use them for provenance only. Do not treat them as current workflow unless a
current doc explicitly points to a section there.

## Update rules

- Keep baseline/interpretation changes in `../EXPERIMENTS.md`.
- Keep current execution steps in `current_project_roadmap.md`.
- Keep implementation tradeoffs in the active spec that owns that area.
- Treat reports as evidence unless a current doc explicitly promotes them.
- Prefer the new current-state and architecture pages for code/workspace maps.
