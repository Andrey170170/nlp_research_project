# Current Execution Plan — Post-Consolidation Cleanup

Status: Current scratch roadmap
Last updated: 2026-05-16

## Problem statement

Track-0B consolidation is complete enough to treat local `main` in the project
repo and sibling `../circuit-tracer_chunked` repo as the current working
baseline. The next phase is cleanup and simplification without losing the ability
to validate exact-trace behavior or reproduce the cross-cluster investigation.

The cross-cluster Track-A investigation is now considered **mostly complete**:
the current interpretation is that the observed Ascend/Cardinal divergence is
driven by Phase-3 gradient differences, with later stages amplifying those
differences enough to affect compact circuit outputs. We should keep the
debug/replay machinery for future verification and regression tests, but it no
longer needs to dominate the public workflow.

## Current baseline to preserve

- Project repo: local `main` after Track-0B consolidation.
- Sibling library: `../circuit-tracer_chunked` local `main` after Track-0B
  consolidation.
- Current exact-trace default: `exact_trace_internal_dtype=fp32`.
- Stable row-L1 denominator behavior is part of the validated baseline.
- `828_base` and `361_base` matched same-generation Ascend baselines exactly.
- `94_base` matched the pre-consolidation optimization-control output exactly;
  mismatch against older Apr-20/21 references predates consolidation.
- Changing `decoder_chunk_size` can cause small compact-output drift relative to
  the current `c2048` reference; cache-size changes are exact within a fixed chunk
  size.

## Scope / non-goals

### In scope

1. Fix correctness risks found by cleanup reviewers.
2. Rewrite stale source-of-truth docs for the consolidated baseline.
3. Classify knobs into default, public resource, scenario-only optimization,
   debug/replay, and deprecated buckets.
4. Keep Track-A debug/replay machinery, but make it cleaner, documented, and less
   intrusive in normal workflows.
5. Add tests and validation hooks for the “gradient difference + amplification”
   interpretation and for future attempts to reduce amplification severity.
6. Split and simplify the project harness around `experiments/exact_trace_bench`.
7. Separate login-safe tests from GPU/model-loading/SLURM validation.

### Non-goals for this phase

1. Do not remove all debug/replay machinery just because Track A is mostly done.
2. Do not make broad runtime-default changes without SLURM validation.
3. Do not rewrite the whole harness before fixing known correctness risks and
   adding lightweight tests.
4. Do not optimize for upstream PR cleanliness in the local fork cleanup; upstream
   PRs should stay small and separate.

## Proposed approach

### Phase 0 — Synthesize cleanup reports

Inputs:

- `REVIEW_TASKS.md`
- `reports/library_core_exact_trace_cleanup.md`
- `reports/library_api_knob_taxonomy.md`
- `reports/project_harness_restructure_audit.md`
- `reports/tests_fixtures_cleanup.md`
- `reports/debug_replay_track_a_preservation.md`
- `reports/docs_roadmap_consistency.md`

Output:

- A single prioritized cleanup backlog, bucketed as:
  1. P0 correctness fixes,
  2. safe docs cleanup,
  3. safe deletions,
  4. test hygiene,
  5. knob/API taxonomy,
  6. Track-A machinery to keep but clean/document,
  7. amplification-reduction experiments,
  8. requires SLURM validation,
  9. major harness refactor.

### Phase 1 — P0 correctness and validation hooks

1. Investigate/fix the Phase-3 row capture/replay issue identified in
   `reports/library_core_exact_trace_cleanup.md`:
   - capture without donor replay may reference `row_abs_sums_cpu` before
     assignment,
   - donor replay may append stale host rows/denominators to the compact row
     store.
2. Add lightweight tests where possible for donor row capture/replay helper logic.
3. Add or update tests/docs that encode the Track-A conclusion:
   - primary divergence source: Phase-3 gradient differences,
   - later exact-trace stages can amplify those differences,
   - future work should measure and reduce amplification severity.
4. After local checks, run SLURM validation for canonical scenarios and any
   donor/replay-specific smoke needed.

Acceptance criteria:

- Phase-3 row capture/replay behavior is internally consistent.
- Existing compact exact behavior for `828_base`, `361_base`, and same-generation
  `94_base` remains preserved after validation.
- We have at least one documented path for testing future amplification-reduction
  changes.

### Phase 2 — Source-of-truth docs cleanup

1. Add a current-baseline table near the top of `EXPERIMENTS.md`.
2. Rewrite `README.md` for consolidated `main` and safe OSC workflow.
3. Keep `AGENTS.md` authoritative; reduce `CLAUDE.md` to a pointer or otherwise
   prevent drift.
4. Add status headers to durable specs under `docs/`.
5. Archive stale duplicate docs and plans that still describe pre-consolidation
   branch state or `matched_debug` as current.

Acceptance criteria:

- A new contributor can identify the current baseline and safe validation path
  from README + AGENTS + EXPERIMENTS.
- `docs/current_project_roadmap.md` describes current work, not completed
  consolidation.

### Phase 3 — Knob/API taxonomy

Mapping document: `docs/knob_api_taxonomy.md`.

1. Make `exact_trace_internal_dtype` the canonical precision knob.
2. Deprecate or hide direct public use of `internal_precision` unless explicitly
   needed as debug/compatibility plumbing.
3. Keep normal public surface small:
   - `exact_trace_internal_dtype`,
   - `decoder_chunk_size`,
   - `cross_batch_decoder_cache_bytes`.
4. Move debug/replay/experimental knobs out of normal scenario generation and
   into explicit debug/scenario-only paths.
5. Archive or mark historical generated scenario configs that encode old one-off
   debug runs.

Acceptance criteria:

- Normal exact-bench scenarios cannot accidentally inherit Track-A replay/debug
  settings.
- Canonical defaults are stated in code, docs, and tests consistently.

### Phase 4 — Track-A machinery cleanup and amplification-reduction support

Track A is mostly done as an investigation, but the machinery remains valuable as
test infrastructure. The cleanup goal is to make it reliable and documented, not
to delete it.

Keep and clean:

- cross-cluster debug summaries/checkpoints/batches,
- Phase-0 donor capture/replay,
- Phase-3 seed/gradient/row capture/replay,
- semantic descriptor artifacts,
- boundary fingerprint artifacts,
- graph/replay/semantic comparison tools.

Improvements:

1. Document artifact schemas for debug/replay bundles and manifests.
2. Add schema-version/backward-compatible loader tests where missing.
3. Consolidate duplicate NPZ serialization and metadata/hash helpers.
4. Add clear docs around single-step replay semantics.
5. Add comparison/summary hooks that help answer the next question:
   **how much does Phase-3 gradient drift get amplified later, and can we make
   that amplification less severe?**

Candidate amplification-reduction follow-ups:

- Compare ranking/frontier sensitivity to small Phase-3 gradient perturbations.
- Test more stable tie-breaking or tolerance-aware frontier/ranker behavior.
- Quantify amplification from Phase-3 seed differences through Phase-4 selection
  and final compact edges.
- Evaluate whether fp64 spot checks, deterministic ranking, or stable thresholding
  reduce output divergence without unacceptable cost.

Acceptance criteria:

- Debug/replay machinery is clearly documented as internal validation tooling.
- Future amplification-reduction experiments can be launched without re-learning
  the old Track-A artifact schema.

### Phase 5 — Test hygiene

1. Add pytest markers for login-safe vs GPU/model-loading/SLURM-only tests.
2. Move project SLURM matrix runner out of `tests/` or mark it explicitly.
3. Merge duplicate transcoder tests in the sibling library.
4. Split large grab-bag tests mechanically.
5. Add lightweight tests for project defaults, scenario generation, parser
   defaults, fixture catalog consistency, and row-L1 normalization boundaries.

Acceptance criteria:

- Login-node validation command cannot accidentally load models or require GPUs.
- Cleanup refactors have enough lightweight tests to catch config/default drift.

### Phase 6 — Library cleanup/refactor

1. Delete safe unused symbols after tests:
   - `filter_chunked_decoder_state`,
   - `_row_denominator_to_row_abs_sums`,
   - `_PHASE4_REFRESH_OPTIMIZATION_EFFECTIVE_MODE_BY_MODE`.
2. Consolidate stable row-L1 denominator helpers into one representation/module.
3. Split large functions mechanically:
   - `_run_attribution`,
   - `compute_partial_feature_influences_streaming`,
   - `AttributionContext.compute_batch`.
4. Keep Track-A artifact schemas compatible unless intentionally versioned.

Acceptance criteria:

- Refactors preserve exact compact outputs under validation scenarios.
- Stable row-L1 overflow behavior remains protected by tests.

### Phase 7 — Project harness restructure

1. Keep root CLIs as compatibility wrappers initially.
2. Move implementation toward package-style modules around
   `experiments/exact_trace_bench` or a dedicated harness package.
3. Rename/rehome `experiments/run_sparsification_experiment.py` as the exact-bench
   runner.
4. Move canonical fixtures out of `experiments/generated/` once tests cover the
   fixture catalog.
5. Archive old exploratory/prototype code after provenance checks.

Acceptance criteria:

- Current exact-bench launch/compare workflow remains runnable.
- Old scripts are either wrappers, archived with provenance, or deleted with clear
  justification.

## Risks and open questions

1. **Phase-3 replay bug risk:** if the reviewer finding is correct, some replay
   artifacts may not represent the intended donor-effective state. Fix first.
2. **Schema compatibility risk:** existing scratch artifacts may rely on current
   field names. Prefer additive schema/version changes and compatibility loaders.
3. **Amplification mitigation risk:** attempts to reduce divergence may change
   compact outputs; treat them as experiments until validated.
4. **Harness restructure risk:** moving entrypoints can break SLURM launch plans
   and immutable workspace imports. Keep wrappers and validate via SLURM.
5. **Open question:** which historical generated scenario files are immutable
   provenance versus disposable build products? Archive first, delete later.
