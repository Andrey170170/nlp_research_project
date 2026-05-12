# Current Execution Plan — Consolidation and Next Project Tracks

## Goal

Stop the current exploratory phase at a good checkpoint, consolidate the project
and sibling library into a runnable/correctness-preserving baseline, and then
split follow-up work into clean, mostly independent tracks.

This plan is intentionally operational. Durable workflow rules still belong in
`AGENTS.md`; launch/results interpretation belongs in `EXPERIMENTS.md`; durable
implementation tradeoffs belong in the relevant `docs/` spec.

## Current baseline to preserve

The current work spans paired project/library branches:

- main project workspace: `exact-trace-bench-harness`
- main sibling library: `exact-trace-hidden-knobs`
- optimization project worktree: `exact-trace-bench-opt`
- optimization sibling library: `exact-trace-hidden-knobs-opt`

Important current findings to carry forward:

- Cross-cluster `94_base` drift is now causally localized to Phase-3
  gradient/row construction, likely reflecting A100/H100 bf16 model/transcoder
  forward/backward numerical drift.
- The stable row-L1 denominator / overflow fix removed the old fp32 compact
  collapse and made fp32 a validated exact-trace runtime default for the current
  setup.
- `active_encoder_cpu + cache8g` is the best current exact optimization
  candidate; `rowstore_fadvise + cache8g` is a memory-sensitive fallback.
- Changing `decoder_chunk_size` produces small compact-output drift relative to
  the current `c2048` reference, while cache-size changes are exact within a fixed
  chunk size.

## Track 0 — Parallel foundation work

Track 0 has two parallel workstreams. They should both finish before starting the
main follow-up implementation tracks.

### Track 0A — Original upstream audit for future PRs

#### Objective

Investigate `https://github.com/decoderesearch/circuit-tracer` directly so future
upstream PRs are based on the original library, not only on our fork.

This is an audit/planning track, not a broad implementation track.

#### Scope

1. Compare original upstream against our fork and identify relevant divergence.
2. Locate all normalization / row-denominator / row-normalized scoring paths.
3. Determine whether the row-L1 overflow bug exists upstream as an active or
   latent bug.
4. Identify backend/path coverage needed for an upstream-quality fix.
5. Inspect the upstream test suite, especially CUDA tests and fixture teardown.
6. Document whether the test suite leaks state or retains CUDA memory across
   tests, and whether a small hygiene PR is justified.
7. Produce a short PR candidate list with proposed order and scope.

#### Questions to answer

- Which upstream files/functions correspond to our fixed row-L1 denominator path?
- Are there multiple backends with duplicated denominator logic?
- Can the overflow fix be proposed without chunking?
- Which targeted tests can run reliably without the full GPU suite?
- Is CUDA memory cleanup a separate small PR, or too entangled for now?
- What hardware assumptions does the current upstream test suite appear to make?

#### Acceptance criteria

- We know whether the row-L1 fix is directly applicable upstream.
- We know the minimum clean test set for an overflow PR.
- We know whether to split out a test-suite hygiene PR before the numerical fix.
- Findings are recorded well enough to guide Track 1 PR work.

### Track 0B — Consolidate fork work to a runnable baseline

#### Objective

Merge the current exploratory and optimization work into a baseline that is
workable, runnable, and correctness-preserving.

Do **not** spend this track trying to perfectly decide which code is elegant,
upstream-ready, or permanently desirable. Cleanup and API pruning belong to later
tracks. The immediate goal is a coherent project/library pair that can reproduce
the validated behavior and remain usable for new work.

#### Scope

1. Preserve important experiment provenance.
2. Merge project branches into an integration branch from `main`.
3. Merge sibling library branches into a paired integration branch.
4. Resolve conflicts toward runnable/correct behavior, not minimalism.
5. Run only safe local validation on login nodes.
6. If needed, run small SLURM smoke jobs from the consolidated pair.

#### Dirty-file policy

- Keep and commit meaningful `EXPERIMENTS.md` updates.
- Everything else currently untracked in the project workspace can be discarded
  unless manually identified as needed before cleanup:
  - report/presentation drafts,
  - temporary extraction scratch files,
  - generated local leftovers not needed for launch definitions.

#### Consolidation checklist

- [x] Record current project + sibling library git states for both workspaces.
- [x] Commit or otherwise preserve important `EXPERIMENTS.md` updates.
- [ ] Discard irrelevant untracked drafts/scratch files.
- [x] Create project integration branch from `main`.
- [x] Bring in `exact-trace-bench-harness` and `exact-trace-bench-opt`.
- [x] Create paired library integration branch from library `main` / current fork
      baseline.
- [x] Bring in `exact-trace-hidden-knobs` and `exact-trace-hidden-knobs-opt`.
- [x] Resolve conflicts preserving correctness and runnable paths.
- [x] Run lightweight validation with `uv run` only.
- [ ] If appropriate, launch minimal SLURM smoke checks for canonical fast/anomaly
      scenarios.

Merge-state details and conflict triage for this track live in
`docs/consolidation_merge_notes.md`.

#### Acceptance criteria

- The consolidated project branch and sibling library branch are paired and their
  provenance is recorded.
- Existing exact-trace launch/scenario plumbing remains runnable.
- Stable normalization/overflow behavior is preserved.
- Cross-cluster debug/replay artifacts remain available.
- Optimization knobs remain explicit; no broad new default is promoted merely by
  consolidation.

## Track 1 — Small-scope PRs and cleanup

Track 1 starts only after Track 0A and Track 0B are complete enough to provide a
stable source of truth.

It has two related goals:

1. prepare small upstream PRs against original `decoderesearch/circuit-tracer`,
2. clean up our fork/consolidated baseline so later sweeps and optimization work
   are easier to reason about.

### Track 1A — Upstream PR sequence

#### Objective

Turn the Track 0A audit into one or more small upstream-quality PRs.

Possible PRs, depending on the audit:

1. CUDA/test-suite hygiene, if the memory/state cleanup is small and separable.
2. Stable row-L1 denominator / overflow fix.
3. Other small correctness or testability fixes discovered by the audit.

Keep each PR narrowly scoped. Do not bundle research harness code, chunking,
optimization knobs, or GSM8K-specific machinery into upstream PRs.

#### Overflow-fix branch strategy

Start from original upstream, not from the full fork branch:

```text
decoderesearch/circuit-tracer:main
  -> upstream/stable-row-l1-denominators
```

Use the consolidated fork only as a reference/cherry-pick source for the minimal
fix logic.

#### Overflow-fix validation strategy

The overflow PR should be valuable outside this GSM8K project. The canonical
`828_base`, `361_base`, and `94_base` runs are useful regression/reproduction
cases, but they are not broadly representative enough by themselves.

Required local/unit coverage:

- synthetic rows where raw fp32 row-abs sums overflow,
- safe-range rows matching the old/intended computation,
- zero and near-zero rows,
- nonfinite handling,
- dtype coverage for fp32/fp64 internal denominator paths,
- all backend/code paths that compute row denominators or row-normalized scores,
- short non-chunked traces where feasible, to show the fix itself does not require
  chunking.

Broader validation targets, as resources allow:

- more than one model,
- more than one CLT/transcoder family or size,
- at least one non-GSM8K-style prompt/task family if easy to add,
- both chunked large-prompt demonstrations and short non-chunked sanity checks.

Current view on task expansion:

- Expanding model and CLT coverage is more important than expanding GSM8K prompt
  count.
- Task expansion is useful but secondary; the overflow bug is a numerical
  normalization bug, so coverage over model/CLT scale and row magnitude regimes is
  more informative than many same-distribution math prompts.
- Keep GSM8K examples as compelling reproductions, not as the only evidence.

One practical issue: the real overflowing prompts currently require chunking to
run at useful scale, but chunking itself should not be part of the overflow PR.
Therefore:

- keep the PR branch focused on the overflow fix,
- test the fix directly with short/synthetic non-chunked cases,
- for large before/after demonstrations, use a separate validation branch or
  local/uncommitted chunking machinery,
- clearly separate “fix code under review” from “large-scale reproduction
  harness.”

#### Acceptance criteria

- Each PR diff is narrow and reviewable.
- Overflowing synthetic cases stay finite and match the mathematically intended
  scaled computation.
- Safe cases match prior behavior within expected tolerance.
- Relevant row-normalization backends/paths are covered.
- Large chunked before/after evidence is documented separately from the clean PR
  diff if chunking is not included in the PR.

### Track 1B — Fork cleanup and default taxonomy

#### Objective

Clean the consolidated research/library code after Track 0 and use broader sweeps
to decide which knobs should be retained, removed, hidden, or promoted.

This is the cleanup track that Track 0 intentionally avoids.

#### Cleanup questions

Classify each major knob/path as:

1. default candidate,
2. explicit optimization option,
3. diagnostic/debug only,
4. temporary experiment to remove,
5. rejected/deprecated.

Initial likely classifications:

- keep `exact_trace_internal_dtype=fp32` as current post-overflow-fix default,
- keep `decoder_chunk_size=2048` as the current canonical exactness reference,
- keep `cross_batch_decoder_cache_bytes` explicit until broader sweeps finish,
- treat `active_encoder_cpu + cache8g` as the leading optimization candidate,
- treat `rowstore_fadvise + cache8g` as a memory-sensitive fallback,
- do not promote refresh optimization V1,
- do not promote full `active_encoder_cpu + rowstore_fadvise + cache8g`,
- keep `phase1_cap64` as a feasibility/fallback knob, not a speed optimization.

#### Acceptance criteria

- Extra exploratory code is either removed, hidden, or documented as diagnostic.
- Defaults/knobs are organized enough that broad sweeps can be configured without
  accidental interactions.
- Rejected paths are explicitly disabled or marked as experimental.

## Track 2 — Broad sweeps, default selection, and numerical sensitivity

### Objective

After Track 1 cleanup, run broader validation sweeps and investigate why the
exact-trace outputs are highly sensitive to small numerical/hardware differences.

This track should decide what defaults are defensible and identify whether the
method itself needs additional stability mechanisms.

### Broad sweep plan

Use sweeps to collect telemetry before promoting defaults.

Priorities:

1. chunk-size drift investigation,
2. cache budget and active encoder residency interactions,
3. model/CLT generality,
4. backend/path parity,
5. prompt/task generality if resources permit,
6. numerical sensitivity and robustness probes.

Chunking question:

```text
Why does changing decoder_chunk_size slightly change compact outputs?
```

Required comparisons:

- `decoder_chunk_size in {1024, 2048, 4096}` at fixed cache settings,
- cache changes within fixed chunk size,
- fp32 vs fp64 internal math spot checks where feasible,
- retained edge membership/weights,
- pre-threshold influence/ranking differences,
- row denominator and row-normalization summaries.

Do not treat chunk size as a pure speed knob until this is understood well enough
to defend.

### Backend/model expansion

Track 2 should also identify whether the overflow fix and any retained defaults
need backend-specific work.

Questions:

- Are there non-nnsight or non-chunked paths with separate denominator logic?
- Are CPU and CUDA paths both covered?
- Which models/CLTs can be tested without excessive new engineering?
- Which differences are true backend/math differences versus expected numerical
  threshold sensitivity?

### Numerical sensitivity questions

The current cross-cluster results suggest that small A100/H100 bf16
forward/backward differences can be amplified into different Phase-3 gradients,
rows, frontiers, and retained compact graphs. Even if bf16 is much higher
precision than int8-style quantization, the tracing method may still be sensitive
because it repeatedly applies ranking, thresholding, frontier selection, and
top-k/retained-edge cutoffs to near-tied attribution scores.

Questions to investigate:

- Are graph differences concentrated near decision boundaries / cutoff ties?
- How much of the drift is score-scale drift versus rank-order drift?
- Which phases amplify small perturbations most: Phase 0 feature activation,
  Phase 3 gradients, row normalization, frontier refresh, or final sparsification?
- Would deterministic/tie-stable ranking reduce output instability?
- Would higher-precision islands around gradients, row construction, or ranking
  materially improve stability?
- Would reporting uncertainty bands / stability scores be more honest than forcing
  a single brittle compact graph?
- Can retained graphs be made less sensitive by adding hysteresis, margins,
  consensus-over-perturbations, or threshold buffers without changing the method's
  meaning too much?

Candidate robustness experiments:

- compare score/rank margins around the retained-edge cutoff,
- measure top-k overlap as a function of retained edge count,
- perturb rows/gradients with controlled noise and observe graph stability,
- run fp32/bf16/fp64 islands where feasible,
- compare deterministic/tie-stable rankers against current rankers,
- capture per-phase sensitivity curves rather than only final graph Jaccard.

### Acceptance criteria

- Defaults are chosen from evidence, not from the most recent successful run.
- Chunk-size drift has a working explanation or a clearly documented risk status.
- Retained knobs have telemetry showing their correctness and performance impact.
- Numerical sensitivity is either reduced by a justified method change or
  documented as an expected limitation with appropriate stability diagnostics.

## Track 3 — Larger optimization directions

### Objective

Pursue larger architectural speedups after consolidation and cleanup produce a
cleaner base to work from.

This track should come last unless deadlines force a narrower prototype earlier.
Cluster load is currently better, but broad tests can still be expensive in queue
time and allocation time.

### Direction A — reduce GPU/CPU data movement

This is likely the first major optimization direction after cleanup.

Before implementation, add or verify telemetry for:

- GPU→CPU bytes and elapsed time by phase,
- CPU→GPU bytes and elapsed time by phase,
- synchronization points,
- pinned vs pageable transfer behavior,
- row-store append/read time,
- denominator/ranking/top-k time,
- Phase 3 versus Phase 4 transfer breakdown.

Candidate computations to move or keep on GPU only after profiling:

- row denominator computation,
- partial influence computation,
- top-k/ranking,
- frontier scoring,
- selected refresh operations.

Guardrail: reducing transfers must not simply move the bottleneck into VRAM
pressure or reintroduce numerical instability.

### Direction B — multi-GPU exploration

Treat multi-GPU as a later architecture/design project, not immediate cleanup.

Possible partition axes:

- prompt/completion/job-level parallelism,
- feature-row batches,
- decoder layer/chunk groups,
- Phase 4 frontier batches.

Risks:

- nnsight/model execution may assume a single-device flow,
- cross-GPU communication may erase gains,
- determinism and provenance become harder,
- SLURM launch/debug complexity increases.

Prefer exhausting job-level and transfer-reduction wins before implementing
intra-trace multi-GPU.

## Global guardrails

- Do not run GPU/model-loading code outside SLURM allocations.
- Use `uv run` for all Python validation.
- Before serious launches, record both project and sibling library git state.
- Treat the project checkout and sibling library checkout as one experiment
  definition.
- Keep ordinary scratch runs organized only by cluster and tier:
  `ascend|cardinal` × `fast|anomaly|long_eval`.
- Record experiment launches/results in `EXPERIMENTS.md`.

## Near-term next actions

1. Start Track 0A upstream audit against `decoderesearch/circuit-tracer`.
2. Commit/preserve important `EXPERIMENTS.md` updates in both active project
   workspaces.
3. Discard irrelevant untracked report/scratch files.
4. Create project and library consolidation branches.
5. Merge current exploratory and optimization branches into runnable paired
   baselines.
6. Run lightweight local validation.
7. Decide whether to run a minimal SLURM smoke check before merging the
   consolidated baseline to `main`.
8. After Track 0A/0B, choose the first small upstream PR and start fork cleanup.
