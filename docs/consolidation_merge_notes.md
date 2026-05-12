# Consolidation Merge Notes

This is a working note for Track 0B branch consolidation. It records transient
merge state, branch provenance, dry-run conflicts, and the current merge plan so
`docs/current_project_roadmap.md` can stay focused on the higher-level execution
plan.

## 2026-05-12 start state

Optimization worktree state has been locked non-destructively before merges.

### Optimization project worktree

- Path: `/users/PAS2119/andreykopanev/worktrees_opt/nlp_research_project`
- Branch: `exact-trace-bench-opt`
- Commit: `c56be53` (`new traces`)
- Tracking: `origin/exact-trace-bench-opt`
- Local backup branch:
  `backup/exact-trace-bench-opt-pre-consolidation-20260512_184908`
- Dirty state preserved in stash:
  `backup/exact-trace-bench-opt-pre-consolidation-20260512_184908`
- The stash was re-applied, leaving the optimization worktree unchanged.
- Dirty files:
  - modified `EXPERIMENTS.md`,
  - untracked `.tmp_exact_trace_extract_hybrid/` containing local extraction
    scratch CSV / JSONL summaries.

### Optimization library worktree

- Path: `/users/PAS2119/andreykopanev/worktrees_opt/circuit-tracer_chunked`
- Branch: `exact-trace-hidden-knobs-opt`
- Commit: `493760d` (`decouple phase1 trace batch cap`)
- Tracking: `origin/exact-trace-hidden-knobs-opt`
- Local backup branch:
  `backup/exact-trace-hidden-knobs-opt-pre-consolidation-20260512_184908`
- Worktree: clean.

### Main-side states

- Main project workspace:
  - path: `/users/PAS2119/andreykopanev/nlp_research_project`,
  - branch: `exact-trace-bench-harness`,
  - commit: `9574fc5` (`Record consolidation roadmap`),
  - status: ahead of origin by one local commit.
- Main sibling library:
  - path: `/users/PAS2119/andreykopanev/circuit-tracer_chunked`,
  - branch: `exact-trace-hidden-knobs`,
  - commit: `f3add59` (`Validate Phase-3 gradients by trace batch width`),
  - status: clean.

### Integration branch pointers

- Project integration branch:
  - branch: `integrate/exact-trace-baseline-20260512`,
  - base: fork project `main`,
  - commit: `4d34483`.
- Library integration branch:
  - branch: `integrate/exact-trace-baseline-20260512`,
  - base: fork library `main`,
  - commit: `dcfd730`.

## Dry-run merge findings

### Project

- Branch merge base:
  `94e4283a9e2b0c04d63409890b5936c07c17108c`.
- Direct merge conflicts expected in:
  - `PLAN.md` (`delete/modify`; keep root plan deleted and keep current roadmap
    under `docs/current_project_roadmap.md`),
  - `TODO.md`,
  - `experiments/exact_trace_bench/config.py`,
  - `experiments/exact_trace_bench/scenarios.py`,
  - `tests/test_cross_cluster_debug_artifacts.py`,
  - `trace_pipeline_chunked.py`.
- Files auto-merging but requiring review:
  - `CLAUDE.md`,
  - `EXPERIMENTS.md`,
  - `experiments/exact_trace_bench/extract.py`,
  - `experiments/extract_benchmark_index.py`,
  - `experiments/run_sparsification_experiment.py`.

### Library

- Branch merge base:
  `d1f3df3e456fdb5430b5462a0a6834bdaf7fa716`.
- Direct merge conflicts expected in:
  - `circuit_tracer/attribution/attribute_nnsight.py`,
  - `circuit_tracer/attribution/context_nnsight.py`,
  - `tests/test_chunked_decoder_optimizations.py`.
- Auto-merge requiring manual review:
  - `circuit_tracer/replacement_model/replacement_model_nnsight.py`, because both
    tracks touch model / transcoder execution behavior.

## Recommended consolidation shape

1. Keep the optimization worktree dirty state intact until the integration branch
   exists; then carry the meaningful `EXPERIMENTS.md` notes into the integration
   branch deliberately.
2. Leave `.tmp_exact_trace_extract_hybrid/` out unless the summaries are needed as
   source data.
3. Create the project integration branch from `main`, then merge
   `exact-trace-bench-harness` before `exact-trace-bench-opt` so roadmap / debug
   provenance is established first.
4. Create the paired library integration branch from the current fork `main`, then
   merge `exact-trace-hidden-knobs` before layering optimization work.
5. For the library, consider cherry-picking the stable normalization/default-fp32
   commits (`3d9d01a`, `ac368bb`, `f432449`) before the larger Phase-4
   optimization chain if direct merging proves too entangled.
6. Resolve conflicts toward a runnable research baseline:
   - preserve Phase-3 / Phase-0 debug and replay hooks,
   - preserve stable row-L1 normalization and fp32 exact-trace defaults,
   - keep optimization knobs explicit rather than promoting new defaults during
     consolidation.
