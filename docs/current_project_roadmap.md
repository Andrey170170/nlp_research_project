# Current Project Roadmap

Status: Current scratch roadmap
Last updated: 2026-07-10

## Active Priority

The active work is the memory-governor and reusable tracing-runtime program in
`plans/2026-07-03_governor_rearch.md`, targeting the contract in
`docs/memory_governor_rearchitecture_spec.md`.

Phase A is closed for implementation purposes:

- Cardinal A4 supplied the scoped semantics-caste and session evidence needed
  to freeze the API. The delayed Ascend reproduction is optional and does not
  block implementation.
- Granite H200 strict baselines completed for Gemma 3 1B CLT, 1B PLT, 4B PLT,
  and 12B PLT across `828_base`, `361_base`, and `94_base`.
- The 12B rerun completed from an immutable project+sibling snapshot with live
  telemetry. Its three traces took 5.8--6.4 hours at b64/c4096. Retain the 600G
  host request until the 38--244 GiB fixture-dependent MaxRSS spread is repeated
  and explained.
- Snapshot-by-default launch enforcement and incremental telemetry are durable
  infrastructure, not pending roadmap tasks.

The pre-Phase-B roadmap was archived at
`docs/history/current_project_roadmap_pre_phase_b_2026-07-10.md`.

## Phase B - Complete

Phase B completed as login-safe contract and arithmetic work. It does not load
models or run attribution.

### B1. Freeze the taxonomy and schemas

1. Complete `docs/knob_api_taxonomy.md` with tier, demand class, bytes-cost
   formula, caste, validation scope, and ownership for every governor-relevant
   field.
2. Keep semantic choices scenario/provider-owned in `strict`; memory pressure
   may select only separately represented physical mechanisms.
3. Record the coupled NNSight trace-capacity family explicitly:
   `max(source_batch_size, feature_batch_size, logit_batch_size)`.
4. Version provider calibration profiles separately from semantic-relaxation
   evidence. Granite data calibrates cost/resources; Cardinal A4 remains the
   only current validated-relaxed evidence scope.

### B2. Land governor v0 in the sibling

Implement in `../circuit-tracer_chunked`:

- immutable `TraceSemantics`, `ResourceEnvelope`, provider-profile,
  `TracePlan`, demand-estimate, and admission-report value objects;
- stable semantic, profile, envelope, and execution fingerprints;
- SLURM/cgroup host-budget discovery with explicit provenance;
- a pure provider-agnostic resolver with no model loading or model-family
  branches;
- rigid VRAM/host admission, elastic file-cache planning, spill selection,
  walltime reporting, trace-capacity binding diagnostics, and actionable
  refusal reasons;
- synthetic cross-layer, same-layer, and approximate/top-k fixtures plus
  versioned Granite 1B/4B/12B calibration fixtures.

Completion record (2026-07-10):

- sibling branch/commit: `phase-b-governor-contract` / `0ce3f96`;
- B1 taxonomy frozen in `docs/knob_api_taxonomy.md`;
- B2 contracts, trusted-evidence registry, nested cgroup/Slurm discovery,
  provider profiles, pure resolver, fingerprints, and admission reports landed;
- 38 focused governor tests and 51 existing telemetry/provider regressions pass;
- Ruff and uv-based Pyright pass.

Resolver outputs remain advisory through Phase C validation and Phase D mechanism
work. Phase E is the first phase allowed to consume plans at runtime. The
trusted validation-evidence registry is intentionally empty until transferred
A4 artifacts are reproduced and reviewed.

## Phase C - Structural Implementation Complete; Granite Gate Pending

Completed at sibling `phase-b-governor-contract@0d65fba`:

1. `attribute_nnsight.py` is the public compatibility and lifecycle-orchestration
   layer at roughly 2k lines; typed phase 0-5 execution lives under
   `attribution/nnsight/phases/`;
2. policies, replay, row-store, prefix, numerics, and phase support are deep
   NNSight modules;
3. observability owns lifecycle, recorder, resources, exception export, and
   human-log rendering;
4. transcoder loading, decoder cache, diagnostics, and fingerprints are
   separate modules;
5. public defaults and artifacts remain unchanged, and no governor plan is
   consumed at runtime.

**C gate passed:** login-safe validation and immutable `361_base` 1B CLT/PLT
Granite H200 runs preserve exact compact artifacts and peak VRAM, with complete
terminal live JSONL and required lifecycle/schema coverage.

Initial immutable jobs `1613072` (1B CLT) and `1613073` (1B PLT) completed with
exact compact parity, unchanged peak VRAM, and 8.2%/8.6% lower trace walltime.
Their manually created scenarios omitted `incremental_telemetry_jsonl`, so the
complete terminal JSONL was written only after tracing and does not satisfy the
live-stream portion of the gate. Corrected project commit `fed889d` ran
from read-only snapshot `workspace_20260710_232642_phase_c_gate_live_telemetry_rerun`
as jobs `1613108` (CLT) and `1613109` (PLT). Both compact NPZs are byte-identical
to baseline; live/final event counts match at 1,720/1,720 and 11,109/11,109;
both sinks closed with zero errors and ended at `attribute.done`. PLT remained
within the timing gate. CLT reached essentially all of its `32G` request and
slowed to 240.72s; this is recorded as a memory-headroom allocation outlier by
explicit adjudication. Future 1B CLT validation jobs must request at least
`64G`. Phase D may begin.

Non-blocking lifecycle debt retained after review: isolate row-store/context/
sink cleanup failures so one cleanup cannot mask the primary exception, and
move observer creation after pure preflight validation so rejected requests do
not leave partially initialized telemetry resources. Once a lifecycle starts,
emit terminal telemetry whenever the sink remains usable. These are reliability
changes, not structural-parity acceptance criteria.

## Phase D - Explicit Controls and Mechanisms

After C passes:

1. harden the lifecycle boundary before adding mechanisms: run row-store,
   context, terminal-event, and sink cleanup independently; preserve the primary
   tracing exception; report cleanup failures without masking it; raise an
   `ExceptionGroup` only when cleanup is the sole failure; move pure request
   validation before observer/sink creation;
2. split logical decoder reduction and frontier-refresh semantics from physical
   fetch/cache and microbatch controls, with deterministic legacy translation;
3. as Phase-4 mechanisms are touched, organize the loop around cohesive
   operations such as initialize frontier, plan refresh, plan batch, execute
   batch, commit rows, update frontier, and finalize. Keep one explicit runtime
   state object and avoid classes or wrappers that add ceremony without owning
   an invariant;
4. implement explicit Phase-1 trace-capacity peak-reduction mechanisms;
5. implement full/tiled/recompute row-store paths and bounded Phase-3/4 dense
   operators so large active universes avoid mandatory full materialization;
6. expose mechanisms directly through the sibling runtime API while the
   governor remains advisory.

Typing is not a standalone deliverable. Strengthen protocols and concrete types
only at boundaries already touched by mechanism or ownership work.

**D gate:** immutable `361_base` 1B CLT/PLT runs must prove the default path
still matches C; explicit selectors must force each implemented mechanism;
Phase 1 must measurably lower peak allocated/reserved VRAM or survive a cap the
reference cannot; at least one bounded Phase-3/4 path must avoid its full dense
allocation while matching output; and only the execution fingerprint changes.
Injected failures must prove all cleanup attempts occur, primary exception
identity is preserved, cleanup-only failures are grouped, and terminal
telemetry closes whenever possible.
Before E, also validate `trace_one`, mixed-shape `trace_batch`, and
`open_session` sequence/reuse/cleanup/cancellation/failure behavior.

## Phase E - Staged Governor Integration

Only after D mechanisms pass parity:

1. apply Phase B pre-load admission;
2. measure permanent model VRAM and representative encoder/decoder costs after
   load, then re-plan headroom;
3. re-plan after Phase 0 using actual active-feature counts and distributions;
4. grant and release phase/transient working sets at phase boundaries;
5. compare predicted and actual resources and walltime for recalibration.

**E gate:** `361_base` 1B CLT/PLT automatic plans must match equivalent explicit
D runs, constrained envelopes must select expected rungs, all planning epochs
must be recorded, and strict compact outputs/semantic fingerprints must match C.

## Phase F - Project Harness Migration

After E passes, adapt the project once to the stable governed sibling API:

- map scenarios and CHPC allocations into sibling requests/envelopes;
- use `trace_one`, `trace_batch`, and `open_session` while preserving explicit
  sequence/window-reuse semantics;
- configure project-owned artifact paths and consume streamed telemetry while
  the sibling sink remains the sole serializer/sequencer/flusher;
- migrate project/tests from private `attribute_nnsight` helpers to stable
  sibling modules, then reduce explicit compatibility re-exports without a
  dynamic catch-all shim;
- consolidate launches through the packaged CLI without weakening snapshots;
- keep fixtures, campaigns, SLURM policy, artifacts, extraction, comparison,
  and scientific interpretation project-side.

**Final F gate:** run a `361_base` smoke for 1B CLT, 1B PLT, 4B PLT, and 12B
PLT, followed by the canonical `828_base`/`361_base`/`94_base` matrix when
promoting the runtime.
Require graph/artifact parity, complete telemetry, plan-versus-actual reporting,
batch/session correctness, and immutable two-repo provenance.

## Validation and Provenance

- Login nodes: pure resolver tests, lightweight unit tests, `ruff`, and `ty`
  only. Model/provider loading and attribution remain SLURM-only.
- Every SLURM run executes from an immutable read-only project+sibling snapshot
  unless an explicitly justified live-workspace exception is recorded.
- Every governed run records both repository states, snapshot manifest, output
  root, allocation, profile/evidence versions, semantic fingerprint, execution
  fingerprint, and job IDs.
- Baseline-changing decisions go to `EXPERIMENTS.md`; verbose events go to
  `experiments/logs/YYYY-MM.jsonl`.

## Deferred, Non-Blocking Follow-ups

- transfer the Cardinal A4 source artifacts from OSC and regenerate the derived
  report before shipping any validated-relaxed profile as a runtime default;
- optionally reproduce A4 on Ascend for historical environment comparison;
- repeat Granite calibration where needed to explain resource variance and
  tighten conservative envelopes;
- do not enable post-load or post-Phase-0 spending until the Phase D mechanisms
  pass fixed-semantics parity.
