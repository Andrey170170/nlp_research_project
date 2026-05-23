# Documentation index

Status: Current docs map
Last updated: 2026-05-23

This directory is split between current working documentation and archived
historical plans. Prefer linking to this index instead of relying on old file
names as source-of-truth signals.

## Current source-of-truth docs

| File | Role |
|---|---|
| `../README.md` | Contributor orientation and safe OSC workflow |
| `../AGENTS.md` | Durable repo policy and operating conventions |
| `../EXPERIMENTS.md` | Compact experiment baseline/index and current interpretation |
| `harness.md` | Current exact-bench harness overview |
| `knob_api_taxonomy.md` | Exact-trace knob/API taxonomy and Phase-3 cleanup map |
| `current_project_roadmap.md` | Current scratch roadmap for the active cleanup phase |
| `full_answer_harness_spec.md` | Track-2 full-answer / multi-token tracing harness design |
| `exact_trace_sweep_campaign_spec.md` | Next exact-trace sweep campaign plan and wave structure |
| `post_consolidation_cleanup_spec.md` | Durable cleanup strategy after Track-0B consolidation |
| `phase4_refresh_optimization_spec.md` | Current Phase-4 normalization/RSS optimization guidance |
| `next_exact_optimization_paths_spec.md` | Current optimization option map |
| `tracing_profiling_spec.md` | Current profiling/telemetry design |
| `phase4_scheduler_v2_spec.md` | Proposed/deferred Phase-4 scheduler-v2 design |

## Historical docs

Historical plans, old proposals, superseded specs, and duplicate inventories live
in `history/`.

Use them for provenance only. Do not treat them as current workflow unless a
current doc explicitly points to a section there.

Notable archived material:

- old Track-A localization plans and Prompt-94 debug specs,
- old consolidation/worktree notes,
- pre-consolidation exact-trace optimization specs,
- matched-debug/weekend benchmark plans,
- the pre-split long-form experiment log.

## Update rules

- Put durable policy in `AGENTS.md`.
- Put compact current baseline/interpretation in root `EXPERIMENTS.md`.
- Put structured event records in `experiments/logs/YYYY-MM.jsonl`.
- Put current execution steps in `current_project_roadmap.md`.
- Put implementation tradeoffs in the active spec that owns that area.
- Move superseded docs to `history/` instead of leaving stale current-looking
  files in the top-level `docs/` directory.
