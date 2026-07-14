# Current Project Roadmap

Status: Current scratch roadmap
Last updated: 2026-07-12

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

Resolver outputs remain advisory through Phase D mechanism validation and the
Phase C2 runtime rewrite. Phase E is the first phase allowed to consume plans at
runtime. The
trusted validation-evidence registry is intentionally empty until transferred
A4 artifacts are reproduced and reviewed.

## Phase C1 - First Structural Pass Complete

Completed at sibling `phase-b-governor-contract@0d65fba`:

1. `attribute_nnsight.py` remained a public compatibility and
   lifecycle-orchestration layer at roughly 2k lines; typed phase 0-5 execution lives under
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
`64G`. This gate proves the first extraction was behavior-preserving; it does
not certify the resulting architecture as readable or complete.

Non-blocking lifecycle debt retained after review: isolate row-store/context/
sink cleanup failures so one cleanup cannot mask the primary exception, and
move observer creation after pure preflight validation so rejected requests do
not leave partially initialized telemetry resources. Once a lifecycle starts,
emit terminal telemetry whenever the sink remains usable. These are reliability
changes, not structural-parity acceptance criteria.

## Phase D - Explicit Controls and Mechanisms

After C passes:

Phase D is the final mechanism/knob pass before the governor. It must solve two
specific scaling failures: the Phase-1 NNSight session peak must be decoupled
from logical later-phase throughput, and the extreme-case Phase-3/4 path must
not retain or materialize a full `K x N` dense structure. `O(N)` influence,
visited, and ranking vectors remain an explicit lower bound.

1. harden the lifecycle boundary before adding mechanisms: run row-store,
   context, terminal-event, and sink cleanup independently; preserve the primary
   tracing exception; report cleanup failures without masking it; raise an
   `ExceptionGroup` only when cleanup is the sole failure; move pure request
   validation before observer/sink creation;
2. split logical decoder reduction and frontier-refresh semantics from physical
   fetch/cache and microbatch controls. Temporary legacy translation exists only
   to validate Phase D and is deleted in C2;
3. as Phase-4 mechanisms are touched, organize the loop around cohesive
   operations such as initialize frontier, plan refresh, plan batch, execute
   batch, commit rows, update frontier, and finalize. Keep one explicit runtime
   state object and avoid classes or wrappers that add ceremony without owning
   an invariant;
4. introduce direct NNSight session-capacity, Phase-3 physical-microbatch, and
   Phase-4 physical-microbatch controls. Subdivide logical work without changing
   row order or refresh checkpoints, and size temporary buffers from active
   lanes rather than cached session width;
5. keep the current file-backed `K x N` store as the full-retention reference;
   add canonical column-tiled row production, a two-dimensional influence
   solver, and a `RowRecipeLedger` no-retention replay path that projects final
   selected rows without creating a `K x N` tensor or file;
6. expose mechanisms directly through the sibling runtime API while the
   governor remains advisory.

Typing is not a standalone deliverable. Strengthen protocols and concrete types
only at boundaries already touched by mechanism or ownership work.

**D gate:** immutable `361_base` 1B CLT/PLT runs cover legacy reference,
explicit-equal controls, reduced session plus split Phase 3/4, 2D tiled full
retention, and no-retention replay. The reduced session must lower allocated and
reserved Phase-1 peaks or survive a predeclared cap at least 10% below a
repeated reference peak. Tiled execution must prove bounded transient shapes;
replay must prove no `K x N` tensor or file. Both preserve logical checkpoints
and compact output under the accepted parity criterion. Synthetic large-shape
tests establish allocation scaling independently of the small 1B fixture.
Injected failures must prove all cleanup attempts occur, primary exception
identity is preserved, cleanup-only failures are grouped, and terminal
telemetry closes whenever possible.
Before E, also validate `trace_one`, mixed-shape `trace_batch`, and
`open_session` sequence/reuse/cleanup/cancellation/failure behavior.

Current gate state: A--C completed for both 1B CLT and PLT. CLT D completed
with exact feature/edge support and weighted-edge Jaccard `0.9999999756` versus
A. CLT E completed from sibling `37c2a0f` in 1,339 seconds with exact
feature/edge support, weighted-edge Jaccard `0.9999999795`, matching 68,025-event
live/final telemetry, terminal `attribute.done`, and no retained `K x N` store.
PLT D completed as `1621467_3` in 5,380 seconds with matching 30,698-event
live/final telemetry and terminal `attribute.done`. The paired five-hour E
attempt was healthy but projected just beyond its immutable application and
Slurm limits, so it was cancelled at 3:58:39 after CHPC rejected an in-place
extension. Final PLT E replacement `1621649_4` completed successfully in
19,820 seconds (`05:30:36` Slurm elapsed) from read-only snapshot
`workspace_20260712_215821_phase-d-plt-e-overnight-20260712`, project
`976c51a`, sibling `76954ce`, one H200, 100G host RAM, and matching seven-hour
application/Slurm limits. It used the same 16,384-column production tiles,
current-microbatch denominator tiles, and 4 GiB overlap-aware replay cache.
The run produced one compact graph with 8,192 features and 20,000 edges,
terminal `attribute.done`, a closed zero-error 198,106-event incremental sink,
`no_retained_full_feature_matrix=true`, `no_kxn_file=true`, enforced 4,096-row
by 2,048-column request bounds, and a 17,985,024-byte observed feature transient
peak. Versus PLT A, feature and edge Jaccard are exactly `1.0`, weighted-edge
Jaccard is `0.9999999791`, and top-k overlap through 1,024 is exactly `1.0`.

The Phase D mechanism gate is closed and C2 may begin. Carry two non-blocking
observability/tooling fixes into C2 ownership work: the in-memory telemetry copy
caps at 120,000 events while the incremental sink remains complete, and
`compare-compact` currently needs the project root added to `PYTHONPATH` to
import `circuit_utils` from an editable console-script invocation.

## Phase C2 - Cleanup Strikes Again

Phase C2 is complete before Phase E. The canonical sibling API,
project request/artifact runtime, phase-operation decomposition, observability
ownership, and stale-surface deletion have landed. Broad login-safe validation
and the immutable Granite CLT/PLT A/D/E gates are closed against the frozen
pre-C2 contract. Its
normative architecture and acceptance criteria live in
`docs/tracing_runtime_rewrite_spec.md`.

This is an atomic cross-repo tracing-path replacement:

1. define coherent domain objects for attribution intent, semantics, resources,
   resolved plans, active features, forward sessions, row storage, seed
   attributions, frontier expansion, graph components, and results;
2. build one canonical `trace_one` / `trace_batch` / `open_session` runtime for
   all backends;
3. separate planning/admission, execution, mechanisms, lifecycle,
   observability, and project artifact ownership with enforceable dependency
   direction;
4. rewrite the top-level runner so its visible body is the tracing algorithm,
   not hundreds of lines of resolution, validation, telemetry assembly, and
   cleanup;
5. audit the entire path, including `context_nnsight`, replacement-model setup,
   transcoder attribution methods, project `trace_pipeline_chunked.py`, the
   scenario command builder, and trace CLI registration. Moving code behind an
   import is not sufficient;
6. migrate sibling and project callers/tests atomically, then delete
   `attribute_nnsight.py`, generic flat `attribute(...)` routing,
   `_attribute_impl`, legacy translators/kwargs, private re-export namespaces,
   and obsolete project tracing paths. No compatibility window is required.

Meaningful objects must explain from their name why their fields travel
together, enforce useful invariants, and be owned by a subsystem. A renamed bag
of old arguments does not satisfy C2. Functions may take several direct
arguments when those arguments are honest domain concepts or capabilities.

**Current C2 gate state:** closed. Sibling lint,
architecture, failure, runtime, and CPU-safe tests pass (`543 passed`, `6
skipped`; three additional cache tests require Hugging Face network access).
Project Ruff and all `181` login-safe tests also pass. Stale imports of removed
paths are absent and the project invokes the canonical sibling API directly.
The immutable `361_base` 1B CLT A/D/E runs pass scheduler, harness, telemetry,
fingerprint, bounded-storage, and compact-parity checks. PLT A/D/E jobs
`1623224_0`, `1623226_3`, and `1623229_4` pass the same criteria. Phase D and C2
are both closed, so Phase E may begin. Repository-wide `ty` remains advisory and
currently reports 85 pre-existing/dynamic-boundary diagnostics; typing is not a
C2 closure gate.

## Phase E - Staged Governor Integration

Only after D mechanisms and the C2 canonical runtime pass their gates:

1. apply Phase B pre-load admission;
2. measure permanent model VRAM and representative encoder/decoder costs after
   load, then re-plan headroom;
3. re-plan after Phase 0 using actual active-feature counts and distributions;
4. grant and release phase/transient working sets at phase boundaries;
5. compare predicted and actual resources and walltime for recalibration.

**E gate:** `361_base` 1B CLT/PLT automatic plans must match equivalent explicit
D runs through the C2 runtime, constrained envelopes must select expected rungs,
all planning epochs must be recorded, and strict compact outputs/semantic
fingerprints must match the pinned reference.

## Phase F - Governed Harness Consolidation

Phase C2 already migrates project trace execution to the canonical sibling API.
After E passes, add governed envelopes and consolidate the remaining harness:

- map scenarios and CHPC allocations into sibling requests/envelopes;
- preserve explicit sequence/window-reuse semantics through the canonical APIs;
- configure project-owned artifact paths and consume streamed telemetry while
  the sibling sink remains the sole serializer/sequencer/flusher;
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
