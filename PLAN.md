# Current Implementation Plan — Upcast / RSS Redesign

## 1. Problem statement

The permanent overflow fix is now validated on the optimization project+library
pair.

Current validated state:

- the earlier healthy-prompt fp32 collapse findings applied to the **pre-fix**
  code path,
- on the fixed code path, fp32/fp64 compact artifacts matched exactly on
  `828_base`, `361_base`, and `94_base`,
- fp32 is now the default runtime dtype for exact compact tracing,
- the next optimization target is no longer normalization correctness; it is
  **memory/transient allocation behavior**, especially host RSS spikes.

The next workstream is therefore:

- reduce transient RSS spikes,
- decouple stable normalization semantics from expensive eager CPU upcasts,
- preserve the newly validated post-fix semantics.

The main suspicion is that the current runtime still pays avoidable host-memory
costs from CPU staging, dense slice materialization, denominator handling,
packaging, and other eager conversions that are no longer required for the
permanent overflow fix itself.

## 2. Scope / non-goals

### In scope

- compact exact-trace CPU staging and row movement,
- denominator storage and any associated dtype/upcast choices,
- file-backed row-store read/materialization behavior,
- refresh-path and packaging-path transient allocations,
- telemetry additions needed to attribute RSS spikes to concrete operations,
- small project-side/runtime-default adjustments only if required to expose or
  validate the redesign.

Primary files likely in scope:

- `../worktrees_opt/circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
- `../worktrees_opt/circuit-tracer_chunked/circuit_tracer/attribution/context_nnsight.py`
- `../worktrees_opt/circuit-tracer_chunked/circuit_tracer/graph.py`
- `../worktrees_opt/circuit-tracer_chunked/circuit_tracer/utils/telemetry.py`
- `../worktrees_opt/circuit-tracer_chunked/tests/test_attribute_nnsight_telemetry.py`
- `../worktrees_opt/circuit-tracer_chunked/tests/test_chunked_decoder_optimizations.py`
- `../worktrees_opt/circuit-tracer_chunked/tests/test_partial_influences.py`
- `trace_pipeline_chunked.py` only if manifests/telemetry wiring needs updating

### Non-goals

- no replay-path redesign yet,
- no frontier/ranking policy changes,
- no approximate solver changes,
- no broad refresh-path speedups unless they fall directly out of RSS work,
- no new cross-cluster investigation work,
- no GPU launches from login nodes.

## 3. Proposed approach

### Workstream A — pinpoint the actual RSS hotspots

Goal: replace vague “RSS is high” reasoning with operation-level attribution.

Immediate audit targets:

1. `rows.cpu()` and related host transfers in Phase 3 / Phase 4 row production
2. `_stage_tensor_on_cpu(...)` in `context_nnsight.py`
3. `_FileBackedFeatureRowStore.materialize_dense_feature_slice(...)`
4. denominator/diagnostic materialization that reconstructs larger CPU tensors
5. packaging/save-time materialization after attribution is already complete

Tasks:

- identify every `.cpu()`, `.to(device="cpu", ...)`, pinned-memory stage, and
  dense temporary in the compact exact path,
- classify each as:
  - semantically required,
  - implementation convenience,
  - debug/telemetry only,
  - packaging only,
- add enough telemetry/checkpoints to measure peak RSS before/after each major
  phase boundary.

### Workstream B — separate stable semantics from storage/upcast policy

Goal: make stable normalization representation cheap to keep around.

Design direction:

- stable scaled row-L1 semantics are already validated,
- those semantics should not force eager fp64-style host behavior or oversized
  CPU temporaries,
- denominator storage, compute dtype, and debug materialization should be
  treated as separate concerns.

Tasks:

1. audit whether `row_abs_max` / `row_l1_scaled` storage can stay in a cheaper
   representation without changing solver semantics,
2. remove any leftover “stable normalization implies eager wider CPU tensor”
   assumption,
3. ensure debug/stat code does not reconstruct expensive CPU views in hot paths
   unless explicitly needed.

### Workstream C — reduce avoidable CPU staging and dense materialization

Goal: keep the compact path compact in practice, not only in theory.

Likely targets:

- avoid staging more columns than the current consumer actually needs,
- avoid creating dense CPU tensors solely for telemetry/debug summaries,
- delay or chunk packaging materialization where possible,
- make row-store reads and packaging paths respect the minimum necessary dtype.

Specific questions to answer:

1. can row production append directly in the needed host/storage dtype without an
   extra transient copy,
2. can refresh-path readers avoid materializing full dense slices when only a
   narrow view is needed,
3. can packaging/stat summaries reuse existing buffers or operate on chunked
   reductions instead of full reconstructions,
4. can CPU staging paths avoid unconditional pinning/contiguity copies when they
   are not providing measurable value.

### Workstream D — preserve semantics with explicit regression checks

Goal: ensure RSS work does not accidentally reintroduce correctness drift.

Required safeguards:

- fp32/fp64 compact artifact parity remains intact on the validated prompt set,
- retained edge counts stay stable,
- decoder-load counts do not regress toward the old collapse signature,
- no row-to-node alignment or frontier-selection changes.

Testing approach:

- add/extend lightweight unit tests for any refactored staging/materialization
  helper,
- keep focused regression coverage around denominator handling,
- use telemetry-oriented tests where new counters/checkpoints are introduced.

### Workstream E — sequence the redesign conservatively

Recommended sequence:

1. add attribution-grade telemetry for RSS/staging hotspots,
2. land the smallest no-semantic-change reduction in eager CPU staging,
3. validate targeted tests locally,
4. only then make deeper row-store/materialization changes,
5. once local checks are stable, prepare the next small Ascend validation pass.

Validation guidance for the later HPC pass:

- keep the same canonical validation prompts:
  - `828_base`
  - `361_base`
  - `94_base`
- first compare post-redesign fp32 against current validated fp32 baseline,
- use fp64 only as a parity spot-check, not as the main optimization target.

## 4. Acceptance criteria

This plan is successful when:

1. the main transient RSS hotspots are identified and tied to concrete runtime
   operations,
2. stable normalization semantics remain validated while denominator/storage
   handling becomes cheaper in practice,
3. at least one meaningful source of avoidable CPU upcast/materialization is
   removed or reduced,
4. no correctness regression appears in focused local tests,
5. the next Ascend validation run can answer whether RSS/transient allocation
   behavior materially improved without changing compact outputs.

## 5. Risks and open questions

- some observed RSS may still be dominated by packaging or other late-stage work
  rather than refresh itself,
- host RSS seen in logs may mix reclaimable file-backed behavior with genuinely
  dangerous anonymous allocations,
- removing one copy may expose a different hidden copy later in the pipeline,
- some staging paths may be required for throughput even if they are expensive in
  memory,
- telemetry itself can perturb memory behavior if added too aggressively,
- we still need to decide how much denominator/debug metadata should be kept
  materializable by default versus only on demand.
