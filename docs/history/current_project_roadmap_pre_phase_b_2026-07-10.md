# Current Project Roadmap

Status: Current scratch roadmap
Last updated: 2026-07-09

## Active priority — governor and reusable tracing runtime

The active execution plan is
`plans/2026-07-03_governor_rearch.md`, with the target contract in
`docs/memory_governor_rearchitecture_spec.md`. The immediate sequence is:

1. finish and record the Granite 1B/4B/12B baseline readout as CHPC
   resource/cost calibration;
2. promote the completed Cardinal A4 comparisons into the knob taxonomy;
   delayed Ascend job `6260319` is optional environment-reproducibility evidence
   and does not block implementation;
3. introduce sibling-library `TraceRequest`, `TraceSemantics`,
   `ResourceEnvelope`, `TracePlan`, and `TraceResult` contracts;
4. provide first-class `trace_one`, `trace_batch`, and `open_session` execution
   paths while preserving `attribute()` compatibility;
5. split logical decoder reduction/refresh semantics from physical
   fetch/cache/microbatch execution before allowing the governor to tune them;
6. implement the pure governor directly in the sibling library, with `strict`
   default fidelity and explicit validated-relaxed/research policies;
7. replace project-side raw attribution-kwarg forwarding with the sibling runtime
   adapter while retaining scenarios, SLURM, snapshots, artifacts, and analysis
   in this repository.

Acceptance gates for the first implementation slice:

- login-safe plan resolution with synthetic provider profiles;
- stable semantic and execution fingerprints in every resolved plan;
- compatibility tests proving legacy `attribute()` requests resolve to the same
  strict plan;
- fake-runtime tests for independent batching and sequenced/window sessions;
- streaming telemetry tests, including partial output on failure;
- Granite SLURM parity against canonical strict scenarios before defaults move.

### Immediate CHPC 12B rerun and workspace gate

The `chpc-baseline-gemma-stack-20260709-03` arrays were submitted directly from
the live project and sibling checkouts, not immutable workspace snapshots.
Their logs resolve `WORKSPACE_ROOT` and `LIB_WORKSPACE_ROOT` to the active
checkouts. Documentation-only edits are safe, but do not edit tracing/runtime
Python or the sibling library until all remaining `1607770` tasks terminate.

Before starting runtime implementation:

1. let the current 12B tasks reach terminal state and identify only the failed
   fixtures;
2. create an immutable read-only snapshot containing both repositories and
   verify that `circuit_tracer` and project imports resolve inside it;
3. resubmit failed strict baselines from that snapshot with
   `batch=64`, `decoder_chunk_size=4096`, at least `600G` host RAM, an
   eight-hour harness timeout, and an 8.5-hour Slurm limit so result
   serialization has a separate grace window;
4. enable `verbose_attribution=true`, `profile_attribution=true`, and
   `profile_log_interval=1`; keep `PYTHONUNBUFFERED=1`;
5. record both repository SHAs/dirty state, snapshot root, scenario file,
   output root, and replacement job IDs before runtime code changes begin.

After the current live-workspace tasks terminate, make snapshot enforcement the
first launcher safety patch:

1. keep packaged launch commands snapshot-by-default and allow reuse of an
   existing verified campaign snapshot;
2. add one shared snapshot validator for manifest presence, project/sibling
   roots, import resolution, and read-only state;
3. make Granite trace/full-answer/baseline templates fail closed when snapshot
   roots are absent or invalid instead of silently falling back to
   `SLURM_SUBMIT_DIR`;
4. retain an explicit live-workspace override only for exceptional debugging,
   and emit its rationale plus repository dirty states into launch metadata;
5. add login-safe launch-plan/template tests proving ordinary submissions cannot
   resolve to the active checkout.

### Incremental telemetry task

The current sibling `TelemetryRecorder` retains events in memory and exports
them only when attribution returns, so a timeout leaves an empty project-side
`telemetry.jsonl`. Make incremental telemetry the first sibling runtime task:

1. add an optional versioned event-sink protocol to `TelemetryRecorder`;
2. emit each sanitized event with a monotonic sequence number while preserving
   the bounded in-memory event list and final `export()` compatibility;
3. provide a project JSONL sink that appends during execution and flushes at a
   bounded interval, plus terminal flushes for success, failure, cancellation,
   and timeout;
4. make sink health, dropped events, sequence gaps, and truncation visible in
   the final telemetry summary;
5. add login-safe tests proving partial telemetry survives an interrupted trace
   and that enabling a sink does not change attribution outputs.

The older post-consolidation cleanup material below remains a secondary backlog;
where it conflicts with the governor plan or current CHPC policy, the newer
documents are authoritative.

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

## Active Track-2 plan — full-answer graph save policy refresh

Status: typed-bucketed reruns and collapsed-topology follow-up complete as of
2026-05-30; result interpretation is current.

The current full-answer traces proved the all-token harness works, but the compact
graph save policy is now known to be too lossy for feature-to-feature topology:
the current global `max_edges=16384` cap is dominated by the single traced logit
row and preserves only about 31--62% of selected feature-to-feature mass on a
5-token raw calibration from wrong `828_base` temp0.8 seed1002.

Raw calibration provenance:

- run root:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/fast/full-answer-raw-payload-calibration-20260527/828_base_temp08_seed1002/raw_graph_5tokens_20260527/828_base_temp08_seed1002_raw_graph_5tokens_20260527`
- analysis dir:
  `.../raw_graph_calibration_analysis/`
- sampled generated indices: `0, 20, 60, 100, 130`
- raw `Graph.to_pt` payload size: ~0.49--0.98 GB/token for prefix lengths
  73--203.

### 1. Replace global top-K compact save with typed edge buckets

Use typed buckets named by matrix direction `target <- source` and preserve raw
bucket mass metadata. Downstream analyses should report bucket-specific metrics
instead of mixing logit, feature, error, and token edges into one global top-K.

Initial save policy from the raw calibration:

| Bucket | Policy | Calibration note |
| --- | --- | --- |
| `logit<-feature` | save all | exactly 8192 nonzero edges in current one-target traces |
| `logit<-error` | save all | median ~3226 nonzero edges |
| `logit<-token` | save all | prefix-length scale, median ~133 nonzero edges |
| `feature<-error` | top-p 0.99, cap ~16k | 16k preserves >99.6% mass in sampled traces |
| `feature<-token` | top-p 0.95, cap ~250k | or top-p 0.99 with cap ~500k |
| `feature<-feature` | top-p 0.95, cap ~1M | or top-p 0.99 with cap ~2M |

Implementation requirements:

1. Save each bucket with:
   - source/target bucket names,
   - raw weights, not only globally renormalized weights,
   - retained raw bucket mass,
   - total raw bucket mass before truncation,
   - retained mass fraction,
   - rank/cumulative-mass metadata or enough information to reconstruct it.
2. Keep fixed-K derived views available for comparability (`128`, `512`, `1024`,
   `4096`, etc.), but derive them from bucketed outputs rather than retracing.
3. Update temporal analysis so exact edge churn can be computed separately for at
   least `feature<-feature`, `feature<-token`, and logit buckets.
4. Keep `save_raw_graph` as a calibration/debug mode, not the default workflow.

### 2. Merge optimization work back into Track-2 harness branch

The optimization branch has already merged this Track-2 branch and reportedly
works. Next action is the reverse integration into the current Track-2 working
branch using the cleanest git operation for the actual branch topology
(`merge`, `rebase`, or fast-forward where possible), while preserving provenance
for both repositories.

Before integration:

1. Record project repo branch/commit/dirty files.
2. Record sibling `../circuit-tracer_chunked` branch/commit/dirty files.
3. Inspect diffs in both repositories, especially exact-trace defaults and hidden
   optimization knobs.
4. Ensure login-safe checks still pass for touched code (`uv run ruff check ...`,
   `uv run ty check ...`, targeted tests).

After integration:

1. Run a small SLURM smoke on full-answer tracing with immutable workspace
   snapshots.
2. Confirm canonical exact-trace defaults remain `exact_trace_internal_dtype=fp32`
   and stable row-L1 denominator behavior.
3. Log the merge and validation in `experiments/logs/YYYY-MM.jsonl`.

### 3. Rerun current comparison traces with the new bucketed save format

Once bucketed output and optimization merge are in place, rerun the current
comparison set rather than over-interpreting the existing 16k-global-edge traces:

1. greedy-correct `828_base` full answer,
2. wrong `828_base` temp0.8 seed1002 full answer,
3. wrong `361_base` temp0.8 seed1002 if the attrbatch128 recovery path is clean.

For each rerun:

1. build all-token trace specs from the frozen trajectory,
2. use immutable workspace snapshots and record both repo SHAs,
3. save bucketed compact graphs using the policy above,
4. run temporal analysis/plots with bucket-specific metrics,
5. compare old 16k-global outputs against new bucketed outputs to quantify how
   much previous edge churn was a save-policy artifact.

The immediate scientific question for the rerun is whether the high apparent
feature-to-feature edge churn persists after preserving ~95% of feature-to-feature
mass, or whether churn mostly reflects the old global edge cap.

Current result from the typed-bucketed all-token reruns:

- all four comparison traces completed and aggregated under
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/fast/typed-bucketed-full-reruns-20260527`,
- cross-run summaries/plots live in
  `.../deep_temporal_comparison_v1/`,
- the three `828_base` trajectories remain close by adjacent/lag/phase metrics,
  with no clean sampled correct-vs-wrong split,
- the wrong `361_base` temp0.8 trace is more locally stable, especially in
  middle/late phase summaries,
- exact `feature<-feature` edge identity remains highly churny despite about 95%
  retained feature-feature mass,
- collapsed `feature<-feature` topology is much more stable: adjacent layer-flow
  weighted Jaccard is about `0.57`--`0.65`, and adjacent positionless feature-flow
  weighted Jaccard is about `0.41`--`0.49`,
- the wrong `361_base` temp0.8 trace remains the most stable; the sampled wrong
  `828_base` trace is somewhat more stable than sampled correct `828_base` under
  collapsed feature-feature metrics, but the current set is too small for a broad
  correct-vs-wrong claim.

Next Track-2 decision point:

1. Treat typed-bucketed + collapsed topology as the current analysis baseline.
2. Do not retrace these four trajectories unless a new metric needs data missing
   from the bucketed artifacts.
3. Decide whether the next scientific step is more trajectories for statistical
   support, richer token/phase annotation for the existing four trajectories, or a
   focused write-up of the current topology-stability finding.

## Current two-track plan after Wave 4

Wave 0--4 established a stable exact-trace baseline and promoted
`plan_feature_batch_size=true` as the safest broad finalist. Full-answer exact
tracing is still too expensive without better orchestration and deeper Phase 3/4
optimization, so the next work splits into two tracks.

### Track 1 — within-trace optimization

Primary workspace:

- project worktree: `../worktrees_opt/nlp_research_project_optimization`
  on branch `harness-cleanup-fullanswer`
- library worktree: `../worktrees_opt/circuit-tracer_chunked`
  on branch `opt-phase34-gpu-residency`

Most implementation should happen in the sibling library. This track can begin
before harness cleanup is finished because it primarily targets trace internals:

- Phase 3/4 transfer and synchronization profiling,
- GPU-resident row reductions / row-L1 computations,
- Phase 3 to Phase 4 handoff residency,
- planner v2 building on `plan_feature_batch_size=true`,
- eventual opt-in knobs for SLURM validation through this project harness.

### Track 2 — full-answer / multi-token tracing harness

This should wait for the Phase-0 harness cleanup below. The intended shape is:

1. freeze a generated answer trajectory,
2. build token-position trace specs against that frozen trajectory,
3. pack token positions into cost-balanced shards,
4. have each SLURM task load model/CLTs once and trace several assigned tokens,
5. aggregate per-token compact graphs, target probabilities/ranks, and timings.

The first useful mode should support selected-token tracing before all-token
full-answer tracing:

- final answer tokens,
- numeric tokens,
- high-surprisal tokens,
- uniform every-k tokens,
- explicit token index list.

### Phase 0 cleanup requirement before Track 2

Do a broader harness cleanup, not just a small scenario-file split:

1. Inventory root `scripts/` and classify each as current launcher, current
   wrapper, historical provenance, or removable stale script.
2. Keep the canonical exact-bench launchers explicit:
   `trace_weekend_exact_chunked*.sbatch`, fixture prep launchers, and the
   `exact_trace_bench_*` wrappers.
3. Move or archive stale one-off launchers/comparison scripts so they do not look
   like current templates.
4. Split `experiments/exact_trace_bench/scenarios.py` into wave/domain modules
   while preserving generated scenario semantics.
5. Add contract tests around scenario JSON schema, baseline key coverage, launch
   env propagation, immutable workspace behavior, and run metadata propagation.
6. Clarify generated artifact policy in `experiments/generated/README.md` and the
   harness docs.

## Current baseline to preserve

- Project repo: local `main` after Track-0B consolidation.
- Sibling library: `../circuit-tracer_chunked` local `main` after Track-0B
  consolidation.
- Current exact-trace default: `exact_trace_internal_dtype=fp32`.
- Stable row-L1 denominator behavior is part of the validated baseline.
- Current canonical base fixtures for new work are `828_base`, `361_base`, and
  `94_base` in the `fast` tier; historical `94_base` anomaly-tier paths remain
  provenance only.
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
- Existing compact exact behavior for `828_base`, `361_base`, and `94_base`
  remains preserved after validation.
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
Implementation status: mostly complete as of 2026-05-18; cleanup follow-up is
nearly done, and the next active phase after cleanup is the sweep campaign in
`docs/exact_trace_sweep_campaign_spec.md`.

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
- Generated scenario ownership is indexed in `experiments/generated/README.md`.

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
