# Current Implementation Plan

_This file duplicates the root `PLAN.md` so the active plan also lives under
`docs/`._

## Problem statement

We need the next investigation phase to do three things at once:

1. **Unify internal dtype handling** so exact tracing does not silently run with
   different precision behavior across modes, debug paths, or helper kernels.
2. **Reduce RSS spikes / unsafe upscaling behavior** without changing tracing
   semantics.
3. **Expand cross-cluster diagnostics far earlier than Phase 4** so one paired
   rerun captures enough evidence to localize the first meaningful divergence,
   instead of forcing a slow one-signal-at-a-time loop.

Recent evidence to design around:

- matched Ascend/Cardinal runs are still materially different,
- Cardinal is about `1.5x-1.6x` faster overall on the matched debug matrix,
- Cardinal is about `1.86x` faster in `phase4.refresh`,
- Ascend runs under discussion used **nextgen** nodes (`A100 + AMD EPYC 7H12`),
  not quad,
- Cardinal GPU nodes are `H100 + Intel Xeon Platinum 8470`.

## Non-goals for this phase

- no Phase-4 queue-policy changes intended to improve correctness/runtime by
  changing selection semantics,
- no approximate tracing,
- no narrow Phase-4-only debug pass,
- no repeated reruns just to inspect one additional scalar.

## Core decisions to lock first

### 1. Precision contract

Introduce **one public internal precision control** for exact tracing.

Recommended API shape:

- CLI / config field: `--internal-precision {float32,float64}`
- persisted run field: `internal_precision_requested`

But also persist a **resolved dtype map** so storage and compute never rely on
 implicit behavior:

- `feature_row_storage_dtype`
- `row_abs_sum_dtype`
- `influence_compute_dtype`
- `planner_compute_dtype`
- `shadow_debug_compute_dtype`

The point is not “everything must be stored in the same dtype”; the point is
 “all dtype choices must come from one explicit contract and be recorded.”

### 2. Broad debug mode

Keep `--phase4-anomaly-debug` for the old targeted mode, but add a broader
 cross-cluster package for paired reruns.

Recommended new flag:

- `--cross-cluster-debug`

Behavior:

- scalar-only / hash-heavy diagnostics,
- end-to-end checkpoints from Phase 0 through Phase 4,
- enough coverage to compare both clusters from one rerun pair.

### 3. Memory work must stay semantics-preserving

Memory/upscaling changes in this phase must not alter frontier selection policy.
They may change:

- temporary allocation patterns,
- planner safety checks,
- host-aware limits,
- staging / storage behavior,

but not the mathematical ranking policy itself.

Operational note:

- file-backed memmap / temp-file RSS spikes are a secondary concern as long as
  they remain reclaimable and do not crash the job,
- anonymous/transient CPU tensor spikes are the primary RSS target,
- host-aware planner logic is a safeguard, not the main investigation axis.

## Workstream A — Precision contract cleanup

### Goal

Remove accidental mixed-precision behavior and make exact-trace precision fully
 auditable.

### Target files

- `circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
- `circuit-tracer_chunked/circuit_tracer/graph.py`
- `nlp_research_project/trace_pipeline_chunked.py`
- `nlp_research_project/experiments/exact_trace_bench/extract.py`
- `nlp_research_project/experiments/extract_benchmark_index.py`

### Tasks

1. Add one resolved precision config path from CLI to fork internals.
2. Replace ad hoc internal `float32` / `float64` literals in the exact compact
   path, except for explicitly named shadow-debug comparisons.
3. Make `compute_partial_feature_influences_streaming(...)` use an explicit
   compute dtype rather than inheriting from `row_abs_sums.dtype` implicitly.
4. Ensure planner probes, runtime refreshes, and debug shadows cannot diverge in
   dtype unless that divergence is explicitly requested and persisted.
5. Surface the resolved dtype map in run artifacts and extracted tables.

### Acceptance criteria

- one requested precision value drives all non-shadow exact-trace math choices,
- runtime manifests show the full resolved dtype map,
- there is no silent “fp64 row sums caused fp64 whole refresh compute” path.

## Workstream B — Broad cross-cluster debug coverage

### Goal

Capture the earliest material Ascend/Cardinal divergence from a single paired
 rerun campaign.

### Diagnostic principle

We should assume Phase 4 is an amplifier, not the only source of divergence.
Therefore we need checkpoints at:

- Phase 0 sparse state,
- forward/logit state,
- feature ordering state,
- Phase 3 seed ranking state,
- Phase 4 refresh/batch evolution.

### Highest-priority checkpoints

#### Phase 0 checkpoint — sparse state fingerprint

Record after setup / sparse activation extraction:

- active feature count,
- per-layer retained counts,
- activation value summary stats,
- compact hashes/signatures of active feature indices,
- resolved dtype map,
- staging mode flags,
- environment fingerprint.

#### Phase 1 checkpoint — target logits fingerprint

Record after forward / target selection:

- chosen target token ids,
- top-k logits/probabilities,
- logits vector stats,
- compact hash/signature of the target-logit state.

#### Phase 2 checkpoint — feature ordering fingerprint

Record before Phase 3 replay:

- hashes of `feat_layers`, `feat_pos`, `feat_ids`,
- chunk/order metadata relevant to locality ordering,
- row-store mode and expected storage bytes.

#### Phase 3 checkpoint — seed ranking fingerprint

This is the most important new early checkpoint.

Record after Phase 3 logit attribution and before Phase 4 starts:

- feature influence vector summary/hash,
- top-K frontier before locality reorder,
- top-K frontier after locality reorder,
- deterministic shadow compare,
- float64 shadow compare,
- row-store read/write summary,
- normalization input stats.

If clusters already differ here, Phase 4 is mostly downstream amplification.

#### Phase 4 checkpoint set — full evolution package

At every refresh and feature batch, record:

- pre-locality and post-locality frontier fingerprints,
- cutoff margin / near-tie stats,
- deterministic shadow overlap,
- float64 shadow overlap,
- frozen-first-frontier overlap,
- layer/chunk composition of selected batches,
- row-store read counts / rows / cache stats,
- refresh latency and feature-batch latency.

### Cross-cutting memory checkpoints

At minimum record host/GPU memory at:

- post-Phase-0 setup,
- post-Phase-3 completion,
- first Phase-4 refresh,
- largest observed Phase-4 refresh,
- first Phase-4 feature batch,
- post-packaging,
- post-save.

### Target files

- `circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
- `circuit-tracer_chunked/circuit_tracer/attribution/context_nnsight.py`
- `circuit-tracer_chunked/circuit_tracer/transcoder/cross_layer_transcoder.py`
- `nlp_research_project/trace_pipeline_chunked.py`
- extraction / analysis helpers under `experiments/`

### Output artifacts

Per completion:

- `cross_cluster_debug_summary.json`
- `cross_cluster_debug_checkpoints.jsonl`
- `cross_cluster_debug_batches.jsonl`

Offline paired outputs for one Ascend/Cardinal rerun pair:

- `pair_checkpoint_diff.csv`
- `pair_batch_diff.csv`
- `pair_refresh_diff.csv`
- `pair_summary.json`

### Acceptance criteria

- one paired rerun can answer where the first meaningful divergence appears,
- debug artifacts are directly joinable across clusters by stage/checkpoint,
- we are no longer blocked on adding one more scalar and rerunning later.

## Workstream C — RSS spike reduction and safer upscaling

### Goal

Reduce large transient host-memory spikes, with priority on anonymous CPU
 temporaries rather than reclaimable file-backed page cache.

### Main hypotheses to test

1. row-abs-sum computation creates avoidable fp64 temporaries on CPU,
2. streamed refresh chunks are being upcast/materialized too aggressively,
3. packaging/materialization may create large late-stage host spikes,
4. memmap/temp-file placement may still matter, but only if it contributes to
   crashes or sustained slowdown,
5. current planner/sizing logic is mostly a GPU-memory concern and only
   secondarily a host-memory safeguard.

### Tasks

1. Audit all explicit upcasts / `.to(device="cpu", dtype=torch.float64)` paths
   in the exact compact runtime.
2. Separate “precision of normalization accumulator” from “dtype of whole
   refresh compute.”
3. Add host-memory telemetry for peak transient allocations per major step.
4. Add packaging/materialization memory checkpoints so late-stage spikes are
   distinguishable from refresh spikes.
5. Add temp-storage/environment fingerprinting:
   - temp dir path,
   - storage backend / fs type if available,
   - row-store file size / expected bytes.
6. Design a host-aware planner guardrail:
   - planner may stay fixed-size,
   - but skip/growth decisions should consider host RSS headroom only as a
     conservative safety check, not as the primary optimization target.
7. If needed, add an opt-in cap/guardrail for refresh compute dtype or chunk
   staging size that preserves semantics but reduces transient allocations.

### Target files

- `circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
- `circuit-tracer_chunked/circuit_tracer/graph.py`
- `circuit-tracer_chunked/circuit_tracer/utils/telemetry.py`
- `nlp_research_project/trace_pipeline_chunked.py`

### Acceptance criteria

- we can attribute RSS spikes to concrete checkpoints / operations,
- refresh compute no longer upcasts more than intended,
- memmap/page-cache behavior is measured but deprioritized unless it threatens
  job stability,
- planner safety logic can reason about host-memory risk, but planner changes do
  not dominate this phase.

## Sequencing / rerun strategy

### Phase 1 — precision cleanup + broad debug only

Implement Workstream A and Workstream B first.

Important: **do not change upscaling/memory behavior yet except where required
 to make dtype behavior explicit and auditable.**

Rationale:

- the first paired rerun must tell us where divergence starts,
- if we change memory behavior first, the debug evidence becomes harder to
  interpret.

### Phase 2 — first broad paired rerun campaign

Run one frozen-workspace Ascend/Cardinal pair with:

- matched config,
- `--cross-cluster-debug`,
- explicit internal precision setting,
- identical prompt fixture(s),
- identical extraction pipeline.

Recommended prompt priority:

1. `94_base`
2. one fast control prompt from the 5-prompt matched matrix (`828_base` or
   `1046_base`)

### Phase 3 — analyze first divergence + memory spikes

Use paired diff artifacts to decide:

- whether divergence starts before Phase 4,
- whether Phase 3 seed ranking already differs,
- which memory checkpoints correspond to the biggest RSS spikes.

### Phase 4 — host-aware memory stabilization

Only after the first paired debug pass:

- implement Workstream C,
- rerun the smallest paired validation set,
- confirm semantics/debug fingerprints are still interpretable.

## Success criteria for the whole plan

- dtype behavior is explicit, unified, and persisted,
- broad debug coverage exists from Phase 0 onward,
- one paired rerun gives enough evidence to localize first divergence,
- RSS spikes are tied to concrete operations and become actionable,
- future cluster-comparison work no longer depends on many blind reruns.

## Open questions

- exact public name for the precision control (`internal_precision` vs
  `internal_dtype`),
- whether broad debug should be a new flag or an expansion of the existing
  anomaly-debug mode,
- whether temp-storage diagnostics need a small startup microbenchmark in
  addition to telemetry.
