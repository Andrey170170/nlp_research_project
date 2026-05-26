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

## 2026-05-12 merge progress

### Project integration

- Branch: `integrate/exact-trace-baseline-20260512`
- Merged `exact-trace-bench-harness` first:
  - merge commit: `8cca590`
- Merged `exact-trace-bench-opt` second:
  - merge commit: `9a6b873`
- Conflict-resolution policy applied:
  - root `PLAN.md` kept deleted,
  - `TODO.md` resolved additively,
  - cross-cluster debug/replay plumbing preserved,
  - optimization scenario/config plumbing preserved,
  - meaningful optimization-worktree `EXPERIMENTS.md` additions carried forward,
  - `.tmp_exact_trace_extract_hybrid/` left out.
- Lightweight validation:
  - `uv run ruff check experiments/exact_trace_bench/config.py experiments/exact_trace_bench/scenarios.py tests/test_cross_cluster_debug_artifacts.py trace_pipeline_chunked.py`
  - `uv run ruff check experiments/exact_trace_bench trace_pipeline_chunked.py tests/test_cross_cluster_debug_artifacts.py`
  - both passed.

### Library integration

- Branch: `integrate/exact-trace-baseline-20260512`
- Merged `exact-trace-hidden-knobs` first into the library integration branch.
- Merged `exact-trace-hidden-knobs-opt` second:
  - merge commit: `c8999d8`
- Conflict-resolution policy applied:
  - kept Phase-0 / Phase-3 donor capture, replay, validation, semantic
    descriptors, and boundary/debug hooks,
  - kept stable scaled row-L1 denominator representation and fp32 exact-trace
    default,
  - kept Phase-4 planner/refresh/ranker/row-store/executor/encoder-residency
    knobs explicit,
  - combined `context_nnsight.py` Phase-3 gradient replay/capture with opt memory
    telemetry,
  - reviewed `replacement_model_nnsight.py` via targeted lint as an auto-merged
    execution-path file.
- Lightweight validation:
  - `uv run python -m py_compile circuit_tracer/attribution/attribute_nnsight.py circuit_tracer/attribution/context_nnsight.py tests/test_chunked_decoder_optimizations.py`
  - `uv run ruff check circuit_tracer/attribution/attribute_nnsight.py circuit_tracer/attribution/context_nnsight.py tests/test_chunked_decoder_optimizations.py`
  - `uv run ruff check circuit_tracer/replacement_model/replacement_model_nnsight.py circuit_tracer/attribution/attribute.py circuit_tracer/attribution/sparsification.py circuit_tracer/graph.py circuit_tracer/utils/telemetry.py tests/test_attribute_nnsight_telemetry.py tests/test_double_pass_sparsification.py tests/test_partial_influences.py tests/utils/test_telemetry.py`
  - `git diff --check`
  - `uv run python -m py_compile circuit_tracer/attribution/attribute_nnsight.py circuit_tracer/attribution/context_nnsight.py circuit_tracer/replacement_model/replacement_model_nnsight.py tests/test_chunked_decoder_optimizations.py`
  - all passed; `ruff` emitted only the repository's existing top-level linter
    settings deprecation warning.
