# circuit-tracer Narrow Fork Spec (Temporal Circuit Stability)

## Problem statement

This repo studies whether **temporal stability of attribution circuits** predicts GSM8K correctness, using:

- Gemma-3-1B-IT
- GemmaScope-2 CLT (width_262k_l0_medium_affine)
- `circuit-tracer` attribution graphs over autoregressive steps

Current tracing is blocked by memory failures in upstream `circuit-tracer` when used with GemmaScope-2 CLTs on single GPUs.

Observed local state (repo today):

- `trace_pipeline.py` uses an **approximate early feature-cap monkey patch** (top-K activation pruning before decoder expansion).
- `trace_pipeline_chunked.py` prototypes an **exact chunked decoder** path via monkey patching.

Observed bottlenecks/failures:

1. **CLT decoder expansion OOM in setup**
   - Fails in `setup_attribution -> compute_attribution_components -> select_decoder_vectors`
   - Evidence: `logs/slurm-trace-pipeline-7210704.err`, `logs/slurm-trace-pipeline-4366354.err`
2. **Phase 4 OOM from `compute_partial_influences` copying dense matrix to CUDA**
   - `torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)` attempts ~90 GB allocation
   - Evidence: `logs/slurm-trace-pipeline-chunked-7215504.err`
3. **Host RAM pressure during loading and likely later packaging**
   - Current loader stacks all per-layer safetensors on CPU before transfer; peak RAM is high
   - Evidence: comments in `trace_pipeline.py` loader + OOM-kill events (e.g. `slurm-trace-pipeline-chunked-7214755.err`)
4. **Likely additional bottlenecks in dense packaging / sparsification path**
   - Current Phase 5 builds multiple dense matrices (`edge_matrix` then `full_edge_matrix`)

Scientific issue: the early top-K patch keeps runs alive but **changes attribution semantics early**, which can bias temporal-stability signals. The fork must recover exactness (relative to intended circuit-tracer semantics) while remaining memory-bounded on one GPU.

---

## Scope

Build a **narrow, research-focused fork** of `circuit-tracer` for **single-GPU, memory-bounded, exact tracing** with GemmaScope-2 CLTs.

In-scope outcomes:

- Move current monkey-patched behavior into first-class fork code paths.
- Implement exact chunked decoder handling end-to-end (setup + backward scoring).
- Rework Phase 4 partial influence computation to avoid giant CUDA copies.
- Reduce dense copies in Phase 5 graph packaging.
- Improve loader/offload policy for CPU RAM vs GPU VRAM tradeoffs.
- Add observability/profiling hooks for memory/runtime.
- Keep repo-facing API/artifacts stable where possible.

Compatibility targets:

- Keep `attribute(...)` callable from this repo with minimal/no call-site changes.
- Keep `Graph` schema usable by existing `graph_to_step_data` and `circuit_utils`.
- Keep current analysis pipeline (`analyze.py`) usable with minimal repo changes.

---

## Non-goals

- **Multi-GPU/model-parallel tracing is explicitly out of scope for this fork.**
- No redesign of attribution algorithm beyond memory-safe execution strategy.
- No new analysis feature engineering in this phase.
- No artifact format migration unless absolutely required for correctness.

Cluster note: Cardinal H100 access is scarce and Ascend offers 2×A100 40GB, but this fork still targets **single-GPU operation only**.

---

## Proposed approach

### 1) Replace monkey patches with first-class fork implementations

Promote logic currently in:

- `trace_pipeline.py::install_feature_cap_patch` (temporary, approximate)
- `trace_pipeline_chunked.py::install_chunked_decoder_patch` (prototype exact)

into fork internals with explicit modes/config flags, so repo scripts stop monkey-patching runtime classes.

### 2) Exact chunked decoder path (end-to-end)

Implement an exact path that:

- avoids full decoder materialization in setup,
- computes reconstruction in bounded chunks,
- defers source-feature -> destination-layer decoder expansion to attribution-time,
- preserves full active feature set until normal Phase-4 selection (`max_feature_nodes`).

This should be the default for GemmaScope-2 CLT usage in this repo.

### 3) Phase 4 partial influences on CPU/streaming with explicit device ownership

Current failure comes from copying a huge dense matrix to CUDA inside `compute_partial_influences`.
Fork behavior should:

- keep `edge_matrix` ownership explicit (CPU unless caller opts in),
- run normalization + power iteration on CPU (or chunked streaming) by default,
- never perform implicit full-matrix CUDA copies.

### 4) Reduce dense copies in Phase 5 packaging

Avoid unnecessary allocations/copies when building final adjacency:

- minimize intermediate dense matrix duplication,
- construct final matrix once where possible,
- keep row/column reindexing deterministic and schema-compatible.

### 5) Loader/offload policy improvements

Address memory pressure from CLT loading and attribution offload policy:

- provide loader modes with bounded CPU peak memory,
- support lazy decoder loading where beneficial,
- document defaults for H100 vs A100-40GB single-GPU runs,
- keep offload behavior explicit and observable.

### 6) Observability/profiling hooks

Add phase-level telemetry:

- per-phase wall time,
- peak GPU memory (allocated/reserved),
- process RSS / host RAM,
- matrix shapes and dtypes at key points.

These hooks should be lightweight and optional (debug/verbose mode).

---

## Architecture / module boundaries

Proposed fork touchpoints (upstream-style paths):

1. `circuit_tracer/transcoder/cross_layer_transcoder.py`
   - First-class exact chunked decoder/reconstruction path
   - No eager all-feature decoder expansion for CLT attribution setup

2. `circuit_tracer/replacement_model/replacement_model_nnsight.py`
   - Keep `setup_attribution(...)` API stable
   - Wire new attribution component contract without repo monkey patches

3. `circuit_tracer/attribution/context_nnsight.py`
   - Feature attribution scoring via chunked decoder slices
   - Strict device ownership checks (CPU/GPU index/device consistency)

4. `circuit_tracer/attribution/attribute_nnsight.py`
   - Phase orchestration updates for CPU/streaming partial influences
   - Phase 5 packaging copy-reduction

5. `circuit_tracer/graph.py`
   - Memory-safe `compute_partial_influences` implementation
   - No implicit full-matrix move to CUDA

6. `circuit_tracer/utils/*` (new or existing)
   - Loader policy utilities
   - Memory/runtime telemetry utilities

Repo boundary expectations:

- `trace_pipeline.py` and `trace_pipeline_chunked.py` should become thin consumers of fork behavior.
- Keep output artifacts (`step_*.npz`, optional `step_*.pt`, `completion.json`, `run_config.json`) stable.

---

## Phased implementation plan

### Phase 0 — Baseline capture + invariants

Deliverables:

- Reproduce baseline failures with current logs documented in fork issue tracker.
- Define invariants:
  - API: `attribute(...)` signature compatibility
  - `Graph` fields expected by this repo
  - no multi-GPU assumptions

**Gate 0:** baseline failure modes and compatibility invariants are documented and testable.

### Phase 1 — Exact chunked decoder as first-class path

Deliverables:

- Implement exact chunked reconstruction + deferred decoder scoring inside fork.
- Remove need for runtime monkey patch from repo scripts.
- Keep semantics: no early feature top-K before Phase 4.

**Gate 1:** setup attribution no longer OOMs in `select_decoder_vectors` for GemmaScope-2 CLT on single H100 for step 0; repo can call `attribute(...)` without monkey patching.

### Phase 2 — Phase 4 CPU/streaming influences

Deliverables:

- Replace implicit GPU copy path in `compute_partial_influences`.
- Add explicit device parameter flow and defaults.
- Validate ranking equivalence on small synthetic/tiny cases.

**Gate 2:** chunked exact run passes Phase 4 step-0 on H100 without `compute_partial_influences` CUDA OOM.

### Phase 3 — Phase 5 packaging memory reduction

Deliverables:

- Remove redundant dense copies during row/col selection and final adjacency assembly.
- Preserve deterministic node ordering and schema.

**Gate 3:** peak memory in Phase 5 is measurably reduced vs current fork baseline; `Graph` remains analysis-compatible.

### Phase 4 — Loader/offload policy hardening

Deliverables:

- Bounded-peak CLT load path (CPU RAM-aware).
- Documented/default policy profiles for single H100 and single A100-40GB.

**Gate 4:** model+transcoder load succeeds reliably under configured memory budgets without OOM-kill.

### Phase 5 — Observability + repo integration cleanup

Deliverables:

- Telemetry hooks integrated and optional.
- Repo scripts updated to use fork-native options (no monkey patches).

**Gate 5:** end-to-end single-completion smoke run emits phase telemetry and produces current artifact schema.

---

## Acceptance criteria

1. **Exactness goal:** no early activation top-K pruning before Phase 4 feature selection.
2. **Memory goal (single GPU):**
   - setup path avoids decoder expansion OOM,
   - Phase 4 avoids full dense CUDA copy OOM,
   - Phase 5 packaging avoids avoidable duplicate dense buffers.
3. **Compatibility goal:**
   - `attribute(...)` remains usable by repo scripts with minimal changes,
   - `Graph` schema remains compatible with `graph_to_step_data` + `circuit_utils`,
   - analysis pipeline continues to run on produced artifacts.
4. **Operational goal:** phase-level memory/runtime stats are available for SLURM debugging.
5. **Scientific goal:** attribution fidelity is preserved better than the early top-K patch (full active-feature set maintained until intended Phase-4 pruning).

---

## Validation plan

### A. Unit / component checks (fork)

- Chunked decoder correctness vs non-chunked reference on tiny synthetic CLT tensors.
- Device-ownership tests for decoder indexing and scoring paths.
- Partial-influence CPU/streaming numerical checks on toy matrices.

### B. Integration checks (repo)

- SLURM smoke: 1 prompt × 1 completion × small max steps on single H100.
- SLURM smoke on single A100-40GB profile (conservative batch/chunk settings).
- Confirm outputs: `completion.json`, `step_*.npz`, optional `step_*.pt` load in existing analysis/evaluation scripts.

### C. Fidelity checks (research-facing)

- Compare early-top-K pipeline vs exact-fork pipeline on identical prompts:
  - active feature counts over steps,
  - temporal overlap/churn metrics,
  - qualitative circuit differences.
- Expect exact-fork traces to retain richer feature sets and avoid early pruning bias.

### D. Profiling checks

- Capture per-phase memory/time telemetry in SLURM logs.
- Track regressions across phases before enabling larger prompt batches.

---

## Risks / open questions

1. **CPU influence computation may become runtime bottleneck.**
   - Mitigation: chunked matmul, configurable precision, phase telemetry.
2. **Lazy decoder I/O could trade memory for substantial latency.**
   - Mitigation: tunable chunk sizes, optional decoder cache.
3. **Dense adjacency remains structurally expensive.**
   - Open question: should medium-term plan introduce sparse/intermediate graph format while keeping external schema stable?
4. **Loader behavior vs OSC node variability.**
   - Need explicit tested presets per node class (H100 vs A100-40GB single-GPU).
5. **Upstream drift risk.**
   - Keep fork narrow and focused; periodically rebase only selected modules.

---

## Recommended implementation order (opinionated)

1. **Phase 1 first** (exact chunked path, no monkey patching).
2. **Immediately Phase 2** (remove Phase 4 CUDA-copy OOM risk).
3. **Then Phase 3** (packaging memory reduction).
4. **Then Phase 4 + 5** (loader policy + observability + cleanup).

This order gives the fastest path to scientifically useful, exact single-GPU traces while keeping compatibility with the current repo pipeline.
