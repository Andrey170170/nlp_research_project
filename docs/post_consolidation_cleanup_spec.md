# Post-Consolidation Cleanup Spec

Status: Current cleanup strategy  
Date: 2026-05-15  
Applies to: project `main` + sibling `../circuit-tracer_chunked` `main` after
Track-0B consolidation

## 1. Problem statement

Track-0B produced a runnable, correctness-preserving project/library baseline.
The repo now needs a cleanup pass that reduces complexity without invalidating
the validated exact-trace behavior or losing the cross-cluster diagnostic tools.

The Track-A cross-cluster investigation is considered mostly complete as an
investigation. Current interpretation:

- Ascend/Cardinal divergence is primarily caused by Phase-3 gradient differences.
- Later exact-trace stages amplify those differences enough to affect compact
  circuit outputs.
- Future work should focus less on localizing the divergence and more on
  measuring and reducing the amplification severity.

Therefore the cleanup must preserve debug/replay machinery as internal validation
infrastructure, but make it cleaner, better documented, and less prominent in
normal user-facing workflows.

## 2. Scope / non-goals

### In scope

1. Fix correctness risks found during cleanup review.
2. Clean source-of-truth docs after consolidation.
3. Define a clear knob/API taxonomy.
4. Preserve Track-A diagnostic machinery while making it cleaner and documented.
5. Add hooks for future amplification-reduction experiments.
6. Separate login-safe tests from GPU/model-loading/SLURM-only tests.
7. Refactor library exact-trace internals after correctness risks are addressed.
8. Restructure the project harness around the current exact-bench workflow.

### Non-goals

1. Do not remove all Track-A debug/replay code merely because the investigation is
   mostly complete.
2. Do not promote new runtime defaults without SLURM validation.
3. Do not make the local fork cleanup serve as an upstream PR directly; upstream
   PRs should remain small, independent, and based on upstream `main`.
4. Do not rewrite the whole project harness before fixing known correctness risks
   and adding lightweight guard tests.

## 3. Baseline to preserve

- Current project workspace: local `main` after Track-0B consolidation.
- Current sibling library: `../circuit-tracer_chunked` local `main` after Track-0B
  consolidation.
- Current exact-trace default: `exact_trace_internal_dtype=fp32`.
- Stable row-L1 denominator behavior is part of the baseline.
- `828_base` and `361_base` matched same-generation Ascend baselines exactly.
- `94_base` matched the pre-consolidation optimization-control output exactly;
  mismatch against older Apr-20/21 references predates consolidation.
- Within a fixed `decoder_chunk_size`, cache-size changes were exact; changing
  `decoder_chunk_size` can cause small compact-output drift relative to the
  current `c2048` reference.

## 4. Proposed approach

### Phase 0 — Cleanup backlog synthesis

Create one prioritized backlog from the reviewer reports:

- `reports/library_core_exact_trace_cleanup.md`
- `reports/library_api_knob_taxonomy.md`
- `reports/project_harness_restructure_audit.md`
- `reports/tests_fixtures_cleanup.md`
- `reports/debug_replay_track_a_preservation.md`
- `reports/docs_roadmap_consistency.md`

Backlog buckets:

1. P0 correctness fixes
2. safe docs cleanup
3. safe deletions
4. test hygiene
5. knob/API taxonomy
6. Track-A machinery to keep but clean/document
7. amplification-reduction experiments
8. requires SLURM validation
9. major harness refactor

Acceptance criteria:

- Every reviewer finding is either represented in the backlog or explicitly
  rejected with rationale.
- P0 correctness and SLURM-gated work are clearly separated from safe docs/test
  cleanup.

### Phase 1 — P0 correctness and validation hooks

Primary target: Phase-3 row capture/replay correctness in the sibling library.

Reviewer-identified risks:

- `capture_phase3_row_bundle=True` may reference `row_abs_sums_cpu` before
  assignment when no row donor replay is active.
- Row donor replay may load donor rows/denominators but still append stale host
  `feature_row_slice` and stale host denominator components into the compact row
  store.

Required work:

1. Confirm the control flow and reproduce with lightweight helper/unit tests if
   possible.
2. Fix effective row/denominator handling so capture payloads and compact row
   storage use the same donor-effective state.
3. Add lightweight regression coverage for capture-without-replay and
   replay-with-row-store paths.
4. Update docs/tests to encode the Track-A conclusion:
   - Phase-3 gradient differences are the likely divergence source,
   - later stages amplify those differences,
   - future work should measure and reduce amplification.
5. Run SLURM validation after local checks.

Acceptance criteria:

- Phase-3 row capture/replay no longer has stale effective-state ambiguity.
- Regression tests cover the fixed helper/control-flow behavior without loading
  models.
- Canonical compact exact scenarios remain preserved after SLURM validation.

### Phase 2 — Source-of-truth docs cleanup

Required work:

1. Add a current-baseline table near the top of `EXPERIMENTS.md`.
2. Rewrite `README.md` for consolidated `main` and OSC-safe workflow.
3. Make `AGENTS.md` authoritative and prevent `CLAUDE.md` drift.
4. Add status headers to durable specs under `docs/`.
5. Archive stale plans/docs that still describe old branch state or
   `matched_debug` as current.

Acceptance criteria:

- README + AGENTS + EXPERIMENTS identify the current baseline, safe validation
  commands, and SLURM-only boundary.
- Roadmap reflects current cleanup, not completed consolidation.

### Phase 3 — Knob/API taxonomy

Canonical normal public/resource surface:

- `exact_trace_internal_dtype`
- `decoder_chunk_size`
- `cross_batch_decoder_cache_bytes`

Required work:

1. Make `exact_trace_internal_dtype=fp32` the canonical precision contract.
2. Deprecate or hide public `internal_precision`; keep only as debug/compatibility
   plumbing if needed.
3. Move debug/replay/experimental knobs out of normal scenario generation.
4. Archive or mark one-off generated debug scenarios as historical.
5. Add parser/scenario tests so normal exact-bench scenarios cannot accidentally
   inherit debug settings.

Acceptance criteria:

- Canonical defaults are consistent across code, docs, generated scenarios, and
  tests.
- Track-A replay/debug knobs are explicit debug/scenario choices, not ordinary
  public workflow defaults.

### Phase 4 — Track-A machinery cleanup and amplification-reduction support

Track A is mostly done, but its machinery remains useful as internal validation
infrastructure.

Keep and clean:

- cross-cluster debug summaries/checkpoints/batches,
- Phase-0 donor capture/replay,
- Phase-3 seed/gradient/row capture/replay,
- semantic descriptor artifacts,
- boundary fingerprint artifacts,
- graph/replay/semantic comparison tools.

Required work:

1. Document artifact schemas for debug/replay bundles and manifests.
2. Add schema-version or backward-compatible loader tests where missing.
3. Consolidate duplicate NPZ serialization and metadata/hash helpers.
4. Document single-step replay semantics.
5. Add comparison/summary hooks to quantify amplification from Phase-3 gradient
   drift through Phase-4 selection and final compact edges.

Candidate amplification-reduction experiments:

- Compare ranking/frontier sensitivity to small Phase-3 gradient perturbations.
- Test stable tie-breaking or tolerance-aware frontier/ranker behavior.
- Quantify amplification from Phase-3 seed differences through Phase-4 selection
  and final compact edges.
- Evaluate whether fp64 spot checks, deterministic ranking, or stable thresholding
  reduce output divergence without unacceptable cost.

Acceptance criteria:

- Debug/replay machinery is documented as internal validation tooling.
- Future amplification-reduction experiments can run without rediscovering the
  old Track-A artifact schema.

### Phase 5 — Test hygiene

Required work:

1. Add markers separating login-safe, GPU, model-download, SLURM-only, and
   artifact-check tests.
2. Move or explicitly mark project SLURM matrix runners.
3. Merge duplicate sibling-library transcoder tests.
4. Split large grab-bag tests mechanically.
5. Add lightweight tests for project defaults, scenario generation, parser
   defaults, fixture catalog consistency, and row-L1 normalization boundaries.

Acceptance criteria:

- Default login-node test command cannot load models or require GPUs.
- Cleanup refactors have lightweight tests covering defaults and scenario
  plumbing.

### Phase 6 — Library cleanup/refactor

Safe deletion candidates after tests:

- `filter_chunked_decoder_state`
- `_row_denominator_to_row_abs_sums`
- `_PHASE4_REFRESH_OPTIMIZATION_EFFECTIVE_MODE_BY_MODE`

Refactors:

1. Consolidate stable row-L1 denominator helpers into one representation/module.
2. Split large functions mechanically:
   - `_run_attribution`,
   - `compute_partial_feature_influences_streaming`,
   - `AttributionContext.compute_batch`.
3. Keep Track-A artifact schemas compatible unless intentionally versioned.

Acceptance criteria:

- Exact compact outputs remain stable under canonical validation scenarios.
- Stable row-L1 overflow behavior remains covered by tests.

### Phase 7 — Project harness restructure

Required work:

1. Keep root CLIs as compatibility wrappers initially.
2. Move implementation toward package-style modules around
   `experiments/exact_trace_bench` or a dedicated harness package.
3. Rename/rehome `experiments/run_sparsification_experiment.py` as the exact-bench
   runner.
4. Move canonical fixtures out of `experiments/generated/` once fixture catalog
   tests exist.
5. Archive old exploratory/prototype code after provenance checks.

Acceptance criteria:

- Current exact-bench launch/compare workflow remains runnable.
- Old scripts are wrappers, archived with provenance, or deleted with rationale.

## 5. Validation strategy

### Login-node safe

- `uv run ruff check ...`
- login-safe pytest subset only, after markers are added
- lightweight parser/config/scenario/fixture tests
- pure helper tests for row denominators and Phase-3 row replay/capture logic

### SLURM required

- Any model loading, tracing, or exact compact validation.
- Runtime changes to Phase-3 replay/capture.
- Runtime/default changes to exact-trace precision, scheduler, ranker, row store,
  encoder residency, or decoder chunk/cache behavior.
- Major harness restructure affecting launch plans or immutable workspace imports.

Canonical SLURM gates after behavior-affecting work:

- `828_base` fast
- `361_base` fast
- same-generation `94_base` anomaly
- donor/replay-specific smoke if Phase-0/Phase-3 machinery changed

## 6. Risks and open questions

1. **Phase-3 replay bug risk:** if the reviewer finding is correct, some row donor
   replay artifacts may not represent donor-effective compact row state. Fix
   before broad refactor.
2. **Schema compatibility risk:** existing scratch artifacts may rely on current
   field names and status strings. Prefer additive schema changes and compatible
   loaders.
3. **Amplification mitigation risk:** reducing divergence may change compact
   outputs. Treat as experiment until validated.
4. **Harness restructure risk:** moving entrypoints can break SLURM scripts and
   immutable snapshots. Keep wrappers and validate.
5. **Generated scenario provenance:** some old generated JSON files may be useful
   provenance. Archive first, delete later.
