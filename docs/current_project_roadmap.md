# Current Project Roadmap

Status: Current scratch roadmap
Last updated: 2026-08-28

## Active Priority

The active work is Phase E calibration-model integration and then Phase F
harness consolidation, targeting the contract in
`docs/memory_governor_rearchitecture_spec.md`. The original execution checklist
in `plans/2026-07-03_governor_rearch.md` is retained as implementation history.

The isolated exact-trace performance worktree has a separate active execution
plan in
`plans/2026-08-21_exact_trace_optimization_and_refactoring.md`. It succeeds the
completed July 29 short-prefix/scaling plan in
`plans/2026-07-29_exact_trace_performance_optimization_loop.md`. That plan's two
ordered campaigns formally finished and packaged the short-prefix mechanisms,
then measured and improved prefix/prompt/model scaling with 1B as the fast
development case
before 4B/12B transfer. SP0-SP5 and LS0-LS3 are complete. LS4 closed through
the frozen 4B/1,024 development rung and 828/512 holdout under the
single-forward batched-VJP regime. LS5 closed through cross-node repeat
qualification of the frozen 12B/1,024 selected circuit and resource envelope.
The successor completed evidence hardening and the frozen profile rerun. Its
2026-08-23 revision corrects the primary objective to memory survivability, with
speed optimized opportunistically inside the admitted set. Step 1a's atomic
code migration now makes the six versioned typed edge buckets the only new graph
artifact. Frozen 12B/1,024 GPU qualification job `1850118` completed and
strictly reopened the schema-v2 `typed_compact_graph_v2` artifact, closing
Step 1a persistence and packaging qualification. Step 1b's code and frozen
calibration are now locally validated and bound through immutable
prepared-workload knobs. It adds separate structural, alias-aware
numerical-stability, and behavioral-faithfulness verdicts plus a bounded
same-process intervention probe; held-out scientific qualification remains
open. Later work separates
row retention/access/reduction, selects full -> windowed -> tiled -> recompute,
and acquires source-agnostic 2K/5K/10K stress. Governor response-model promotion,
fidelity-scope authorization, baseline-registry changes, and launch-default
promotion remain
separate reviewed actions.

**Isolated performance-worktree status (2026-08-18):** LS2 and LS3 are closed.
LS4 has admitted the frozen one-lane batched-VJP 4B development curve through
1,024 tokens as one-run feasibility, after independent repeat qualification
through 512. The earlier fail-closed full-H200 `cuda_windowed` candidate was
`strict_exact` at 4B/512 and reduced
trace wall from 346.773 to 168.067 seconds; retain it as a selectable measured
4B mode, not an arbitrary-prompt or global default. One H200 plus 400 GiB is a
demonstrated safe allocation for this workload, but not a measured minimum
because control and candidate cgroups owned different amounts of shared page
cache. Keep the formal 400 GiB request until controlled ownership evidence
supports downsizing.

The selection and resource-control prerequisite is now implemented: one
fingerprinted selected-config/launch record, runtime-level
`preferred | required` mechanism enforcement, typed
`off | measure_only | enforce` resource policy, and machine-readable
requested-to-effective resolution deltas. Admission, allocation, late
copy/append, fallback, propagation, and artifact behavior have focused tests.

The legacy duplicated-lane 4B/1,024 development probe passed wrapper
validation, preheat, model load, and Phase 0, then filled one H200 to
142,553/143,771 MiB and failed in Phase 1 before `cuda_windowed` could resolve.
The first general opt-in response, `single_forward_batched_vjp`, retains one
physical forward lane, groups
cotangents by source layer, uses batched autograd VJPs, restores canonical
row/tape order, and preserves the legacy default behind a typed engine boundary.
Primary-error formatting, partial cleanup, topology identity, and fail-closed
mode/capacity/lane checks are hardened.

The original wide-versus-narrow 4B/256 full pair proved the memory mechanism
(37,003 to 14,597 MiB sampled HBM) but showed that the two execution regimes
are not numerically interchangeable (feature Jaccard 0.97827, edge Jaccard
0.95417, normalized L1 0.04683). The corrected four-arm Phase-3 diagnostic
(`1791882`--`1791885`) completed with required `cuda_windowed`, exact prepared
mechanisms, all requested captures valid, and no Phase 4. Moving injected
execution from 128 lanes to one preserved Phase-0 and Phase-1 target state but
first diverged in every captured Phase-3 layer gradient: gradient symmetric L1
was 2.63 percent, feature-row L1 2.43 percent, and pre-locality Top-1,024
overlap was 1,013/1,024. The one-lane arms used 14,597--14,661 MiB versus
34,443 MiB wide, confirming a 57.4--57.6 percent HBM reduction at this boundary.

At singleton width, injected versus serial and serial versus batched produced
bit-exact gradients and active-feature rows. Their first persisted difference
is the nonfeature error-column L1 contribution; row-denominator and seed drift
is only 40--54 ppm and both frontier arrays remain exactly ordered. Therefore
batched autograd is not the source of the earlier large bounded failure at one
lane, but this does not validate multi-vector batched VJP. The wide/narrow
contrast remains composite across graph/session/Phase-1 width and must not be
attributed to one kernel without further evidence. The comparator checkpoint
order has been corrected to test error-column and row-denominator arrays before
seed influence; focused validation passes, with integrated campaign validation
required before the next launch. See the
[factorial result](../reports/2026-08-12_vjp_early_localization_results.md).

The separate frozen one-lane batched-VJP regime is now qualified through
4B/512. Independent full jobs `1807387` and `1807388` on `grn027` and `grn031`
held the single-lane, batched-autograd, logical-capacity-128 identity fixed.
The canonical selected feature circuit and all four non-error typed buckets
were `strict_exact` across nodes. Auxiliary feature-from-error edges remained
bounded at 0.9946394 support Jaccard, 0.9942986 weighted Jaccard, and 0.0057175
signed L1; logit-from-error edges were bounded at 1.0, 0.9958178, and 0.0041973
respectively. Sampled peaks were 17,177/17,175 MiB HBM and process RSS was
34.86/34.69 GiB. The 168.44/200.44-second traces are descriptive because the
independent nodes had 27/266-second preheats and different cache states.

Job `1818091` completed the full 4B/1,024 development point with that frozen
regime unchanged. Trace time was 288.93 seconds, sampled peak HBM was 25,693
MiB, and process peak RSS was 60.89 GiB. Strict artifact reopen validated the
complete 8,192-feature/20,000-edge graph, while required `cuda_windowed`, the
single forward lane, batched-autograd VJP, and all capacity pins resolved
exactly. This closes one-run resource and artifact feasibility only; it does not
establish 1,024-token repeatability, arbitrary-prompt validity, or cross-regime
equivalence. Backward selection remains decomposed into forward-graph and
VJP-kernel axes, with named presets and the legacy default intact.

The frozen 4B 828/512 holdout completed in job `1818336`. The dependent 12B
ladder job `1818349` then completed the ordered 129-to-256-to-512 sequence in
one allocation and one preheat. Every rung preserved required `cuda_windowed`,
single-forward batched VJP, shared checkpoint scope, and the native 12B
session/backward/Phase-1/Phase-3/Phase-4 capacities of 64. Strict reopen and
telemetry gates passed. Sampled HBM was 32,227/32,457/33,079 MiB and process RSS
was 29.59/38.28/56.07 GiB. This establishes single-run feasibility and valid
compact artifacts only, not repeatability, cross-regime equivalence, or native
12B continuation behavior.

Job `1819287` attempted the full 12B/1,024 endpoint and timed out after two
hours with Phase 4 at 702/8,192 rows and no graph. The reusable live-HBM repair
then passed in job `1832086`: exact 8,691,901,440-byte active-row demand was
admitted after a 16 GiB safety margin, Phase 4 performed zero decoder-page
loads, and the full trace completed in 517.46s. Peak HBM was 48,861 MiB and
process RSS 93.41 GiB. Strict independent audit reopened the 8,192-feature,
20,000-edge typed graph and found no future-position violations. This closes
single-run feasibility and artifact qualification in the frozen v2 regime.
The cancelled eight-hour job `1740811` remains superseded preparation evidence.

Cross-node repeat job `1834396` completed on `grn028` from the same read-only
snapshot with a byte-identical trace spec. The selected 8,192-feature,
20,000-edge compact circuit was exact across jobs `1832086` and `1834396`, as
were all non-error typed buckets apart from one equal-weight cutoff tie in the
million-edge feature-to-feature bucket. Required dynamic residency admitted the
same 8,691,901,440-byte row set, peak HBM repeated at 48,861 MiB, and process
RSS repeated at about 93.4 GiB. Error buckets remain bounded rather than exact:
feature-from-error weighted Jaccard was 0.97398 with signed L1 0.02614, while
logit-from-error weighted Jaccard was 0.99418 with signed L1 0.00585. This
qualifies repeatability of the selected compact circuit and resource envelope
inside the frozen v2 regime; it does not make the whole typed NPZ exact, prove
cross-regime equivalence, change defaults, or establish native-12B continuation
behavior.

Evidence-hardening job `1843641` then completed the same frozen trace on
`grn032`. Slurm marked the job `FAILED` only because the frozen post-run gate
compared equivalent Torch UUID `b2c4...` and `nvidia-smi` UUID `GPU-b2c4...`
spellings literally. The bounded validator repair canonicalizes the optional
prefix, and replaying the full gate accepts the immutable output. The selected
8,192-feature/20,000-edge compact circuit is exact against both reference jobs;
all six typed buckets reopen, with bounded drift confined to the previously
scoped error buckets plus one equal-weight feature cutoff tie against job
`1832086`. The canonical trace took 617.90s and Phase 4 took 405.18s, so this is
profile evidence rather than a new speed baseline.

Typed-only qualification job `1850118` subsequently completed with Slurm state
`COMPLETED`, exit `0:0`, and 14m33s allocation wall. Preheat took 253.02s and
the trace took 508.31s. Strict reopen accepted schema 2
`typed_compact_graph_v2` with 8,192 features and 1,323,157 edges across the six
canonical buckets: 1,000,000 `feature<-feature`, 16,000 `feature<-error`,
250,000 `feature<-token`, 8,192 `logit<-feature`, 47,941 `logit<-error`, and
1,024 `logit<-token`. Required mechanisms completed with no fallback, Phase 4
performed zero decoder page loads, peak HBM was 48,861 MiB, and mechanism
validation completed. This closes Step 1a only: it supplies no Step 1b
repeat/reference numerical-stability or behavioral-faithfulness verdict.

Step 1b development jobs r4 `1861310`, r5 `1862192`, and r6 `1862432` form the
declared calibration corpus and cannot also serve as held-out qualification.
All three contribute immutable numerical receipts; r4 and r5 were behavioral
debug iterations and are not successful faithfulness evidence. R6 completed
the raw four-variant probe in 94.78 seconds and supplies the sole complete
behavioral calibration observation, but its live verdict correctly remained
`unknown` because no frozen calibration was yet declared. The implementation
now verifies the versioned calibration, nested numerical-reference manifest,
and transitive receipt hashes before required-mode execution. Its v1 required
scopes are repeat and canonical typed-graph stability plus repeat-frontier
stability; canonical-frontier stability is not claimed because the Step 1a
reference has no frontier sidecar. Held-out r7 job `1864164` completed its
12B/1,024 trace: structural conformance passed and all three required numerical
scopes were bounded within the frozen calibration. The gate then correctly
failed closed before behavioral execution because the project adapter attempted
direct NumPy conversion of a live BF16 compact tensor. R7 therefore supplies
held-out numerical evidence but does not qualify Step 1b. The next evidence
gate is an otherwise unchanged immutable r8 after the bounded adapter repair.
That repair is committed at `13d69ac`: the project adapter now casts only live
BF16 tensors to FP32 at the Torch-to-NumPy evidence boundary, and an end-to-end
preparation-seam regression covers BF16 compact activation and edge tensors.
The focused integrated correctness suite passed 162 tests, with Ruff, focused
typing, and strict review also clean. R8 job `1864886` then ran from the clean
read-only paired snapshot rooted at
`workspace_20260828_171112_cq-step1b-12b1024-required-r8-20260828-01`. It
completed the unchanged 12B/1,024 trace in 559.17 seconds and spent a further
101.70 seconds in correctness evaluation. Structural conformance and all three
required numerical scopes remained supported, while the optional canonical
frontier scope remained explicitly unknown. Required behavioral execution
failed closed before its baseline forward with `ordering_unqualified`; no
variant ran and the result is neither behavioral support nor contradiction.

The next evidence gate is therefore a qualification-only H200 job, not another
full trace. It reuses r8's immutable production artifacts to select exactly one
leading no-op, the earliest eligible one-source direct-frozen case with strictly
later observations, and its matched propagated-frozen-attention case. A sibling-
owned eager PyTorch-hook oracle and the production selective NNSight runtime are
compared twice in-process and again across two fresh processes. Qualification
requires no-op recovery, BF16-aware agreement, material post-intervention
downstream response for both semantics, cleanup, stable runtime identity, and
tamper-evident recomputable receipts. This gate makes only the narrow capture-
ordering claim; it does not independently qualify all intervention arithmetic,
promote a class-wide capability, or close Step 1b. A passing receipt must first
be bound to the exact model/provider/runtime scope before an unchanged r9 can
exercise required behavioral mode. Qualification job `1875143` was submitted
from project commit `2ae5cdc`, sibling commit `ab40054`, and read-only snapshot
`workspace_20260831_134248_cq-ordering-qualification-12b1024-r8-20260831-01`.
It requests one H200, 12 CPUs, 200 GB host RAM, and two hours on the short QoS;
it completed in 159 seconds but rejected the ordering claim. The run was
deterministic within each runtime and passed 193 of 195 comparisons. The two
failures were propagated downstream features from the layer-34 source: layer 35
feature 30246 differed by seven BF16 ULPs and layer 37 feature 12103 by three.
The source itself was 816 in the eager oracle and 812 in NNSight. Native
zeroing therefore applied unequal physical deltas (`-816` versus `-812`), while
the direct arm used the same graph-pinned delta in both runtimes. This makes the
qualification rejection valid but causally ambiguous: it may reflect source
computation-form drift rather than capture ordering.

A separate non-promoting propagation diagnostic now compares that native arm
with a common graph-pinned delta while preserving the same NNSight source
prepass, two-party barrier, attention-freeze path, and intervention invoke. It
captures source pre/write/contribution/post identity plus layer-by-layer
feature-input response deltas, persists only scalar summaries and hashes, and
reports both any difference and material divergence under the unchanged
two-BF16-ULP tolerance. Project commit `f93854b`, sibling commit `08d485b`, and
read-only snapshot
`workspace_20260831_162406_cq-ordering-propagation-diagnostic-12b1024-r8-20260831-01`
bind diagnostic job `1875372`. It requests one H200, 12 CPUs, 200 GB host RAM,
and two hours on the short QoS and was pending for resources when recorded.
The receipt cannot qualify or promote the runtime; r9 remains unauthorized.

Bounded parallel preheat is now implemented behind a reusable module and all
three preheated wrappers. Deterministic file discovery and inode deduplication
are preserved; a bounded file-level worker pool uses positional reads, exact
byte accounting, and fail-closed short-read/error handling. The allocation-
derived default is `min(8, max(1, CPUs / 4))`, so current 32-CPU jobs use eight
readers, while `PREHEAT_WORKERS=1..8` remains an explicit override and the
manifest records requested/admitted/effective concurrency. Unit, wrapper,
lint, typing, and shell-syntax gates pass. Job `1832086` read the exact 411 GB
set in 14.09s with 7.57x summed-file-time overlap, proving the eight-worker
implementation and cache-hot path. It does not establish cold-storage
throughput or shared-filesystem safety; retain the single-worker fallback and
run controlled warm 1-vs-8 plus cold-client qualification before treating the
27.8 GiB/s observation as general.

The immediate post-repeat work splits into two parallel lanes:

1. **CIE handoff and Llama 8B verification.** The qualified sibling library is
   merged into local `main` at `0a5a384`; the clean integration point is
   `ca92ea1`, tagged `cie-scorer-baseline-2026-08-21`. Let CIE create and compile its own worktree
   from that ref. This is a library/CIE integration point, not a claim that the
   older project `main` is compatible with the removed legacy library entry
   points. Validate directly against the
   original upstream library on Llama 8B at frozen prefixes through 500 tokens
   (start with 64/128/256/500). Hold model revision, token IDs, target, provider,
   hooks, thresholds, and graph budgets fixed. Compare the selected circuit,
   typed buckets, and the CIE-consumed score/output; add an intermediate
   unoptimized-fork arm only if upstream versus current diverges materially.
2. **Canonical artifact, correctness, and opportunistic 12B closure.** Step 1a
   has replaced the legacy/dual graph interface with typed-bucket-only persistence
   and job `1850118` completed the frozen 12B/1,024 GPU qualification:
   both writers use one versioned six-bucket policy, generic `max_edges` and
   current `all_edge_*` promotion metrics disappear, and legacy reading becomes
   explicitly historical. Step 1b now implements independent structural/algorithmic,
   repeat/reference numerical, and behavioral-faithfulness reports. Numerical
   stability retains exact-ID evidence but adds deterministic one-to-one
   decoder-soft matching for the selected/near-cutoff frontier; the bounded
   same-process probe adds direct-effect closure, three propagated-necessity
   samples versus matched controls, and conditional sampled causal alias
   substitution under a ten-variant/120-second ceiling. Full-circuit
   sufficiency remains unknown in version one. R7 and r8 preserved the frozen
   structural/numerical result but exposed implementation and ordering-admission
   blockers before behavioral evidence. First qualify and bind the targeted
   intervened-forward ordering receipt, then rerun the unchanged required-mode
   workload as r9; only after that run `cuda_full` directly at
   frozen 12B/1,024, paired with its windowed control in one allocation when
   practical. Queue latency makes a separately scheduled 4B/512 prerequisite
   low-value; retain 4B/512 only for concrete debugging.
3. **Survivability lane.** Separate row retention/ownership, influence
   access/residency, and reduction execution. Qualify canonical column-tiled
   production and then no-retention deterministic replay before extending staged
   selection across full -> windowed -> tiled -> recompute. Memory is a hard
   constraint; time is optimized inside the feasible set.
4. **Long-prefix/model stress.** Freeze source-agnostic nested 2K/5K/10K Gemma
   token prefixes from provenance-permitted long natural text, including eligible
   Qwen/BonaFide outputs, and trace Gemma's own endpoint target. Keep native long
   Gemma generation as a separate optional acquisition lane. Stop the 12B ladder
   on the first architecture failure. Inspect 27B assets and permanent-state
   demand after the survival rungs qualify; do not let this interrupt CIE.

The hardened Phase-4 profile measures 405.18s total. Its sampled current-stream
interval estimates are feature compute 157.36s, refresh row-store read 116.02s,
encoder materialization 44.26s, ordered row-store write 37.04s, normalization
9.58s, and CPU staging 4.69s. These intervals include host enqueue gaps and
stream waits; they are not GPU-active/kernel-only time. The complete wall census
shows normalization at 81.25s and CPU staging at 40.20s, locating much of their
cost on the host/launch/wait side. Peak framebuffer use remained 48,861 MiB of
143,771 MiB. This selects `cuda_full` as the next tactical GPU arm because it
directly targets the row-window interval and has measured HBM headroom. It is an
opportunistic top rung, not the scaling strategy. The ranked execution order now
puts the correctness contract, tiled/recompute survival paths, staged selector,
and 2K/5K/10K stress ahead of normalization and feature-kernel speed work. The
typed-bucket-only artifact migration is the immediate prerequisite for all new
qualification evidence; the registry owns the detailed order.

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
2. Keep semantic choices scenario/provider-owned in exact mode; memory pressure
   may select only separately represented physical mechanisms.
3. Record the coupled NNSight trace-capacity family explicitly:
   `max(source_batch_size, feature_batch_size, logit_batch_size)`.
4. Version provider calibration profiles separately from fidelity evidence.
   Granite data calibrates cost/resources; Cardinal A4 remains a narrow
   historical fidelity observation scope.

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
are both closed, so Phase E implementation proceeded. Repository-wide `ty` remains advisory and
currently reports 85 pre-existing/dynamic-boundary diagnostics; typing is not a
C2 closure gate.

## Phase E - Staged Constrained Governor Integration

The runtime integration and governor-v0.3 correction gate are complete. The
immutable 2026-07-15 1B CLT/PLT traces passed compact parity against the
original corrected-hook Granite artifacts and exposed the next limitation:
historical logical batches and fetch ceilings still prevented a meaningful
upward optimization search.

The governor is now treated as a staged constrained mixed-variable optimizer,
not a strict-rung lookup table. At every safe epoch it minimizes predicted
remaining runtime under hardware/resource limits, frozen state, hard user pins,
and one immutable fidelity budget. Its four policies are `exact`, `bounded`,
`best_effort`, and `research`. Calibration observations fit compact statistical
resource, runtime, and fidelity response models; no neural model is implied or
required. Campaign names never determine eligibility.

The v0.3 contract separates loose implementation safety limits from calibration
support. A fitting value outside observed support is legal but explicitly
extrapolated; a value beyond the safety limit is invalid. Session capacity,
Phase-1 scheduling, Phase-3/4 microbatches, decoder cache, replay-tile cache,
row policy, residency, and placement are independent constrained variables.
Source microbatch remains explicit/derived only because no sequenced source
executor consumes it.

Memory admission uses per-phase concurrent peaks. Walltime is an additive phase
projection: observed completed time plus predicted remaining work. Storage and
cache costs apply only to the phases they affect; no row-policy multiplier may
scale the whole trace. Loaded and phase-boundary observations replace matching
estimates and never refit the active run. The v0.3 implementation, CPU gates,
and immutable CLT/PLT gate completed on 2026-07-15. Phase E now enters staged
calibration rather than treating the two gate observations as a fitted model.

1. enumerate provisional candidates and enforce hard user requirements during
   pre-execution admission (true pre-load admission still needs a typed loader
   boundary);
2. measure model/provider costs after load and re-optimize controls not yet
   frozen by Phase 0 state;
3. re-optimize after Phase 0 using the actual feature universe;
4. refine still-free phase-local controls at Phase-3/4 entry using prior actuals;
5. grant/release resources and record candidates, constraints, objective scores,
   predictions, actuals, and prediction error at every epoch.

**E calibration:** execute the staged campaign in
`docs/governor_calibration_matrix.md`. Wave A is a 36-row 1B upward search: 17
CLT and 19 PLT rows spanning larger logical batches, decoder fetch/cache
controls, coupled high corners, isolated opt-in semantic axes, refresh cadence,
and exact repeats. Wave B observes 4B/12B transfer. Wave C fits phase-local
mechanisms around useful model-local knees and supplies held-out checks. Ingest
all scientifically valid rows immediately as observations, but separately
review response-model bundles, fidelity-scope authorization, and launch-default
changes.

**Wave A analyzed 2026-07-19:** both provider references match the original
corrected-hook Granite baseline. CLT is bounded by successful `b1500` at 132.98
GiB reserved and OOM at `b2000`; its useful low-risk gain is `c10080` at 1.43x.
PLT's primary measured range is `b256` through exploratory `b512`, with
`c16384/c32768` as opt-in fetch candidates. The measured `b512/c8192` corner is
5.08x but misses the Top-256 floor at `0.9617`; `c32768` is the fastest measured
row above the campaign's illustrative four floors at 4.01x. `b768+` and the
12-16x high corners establish the high-loss region. These rows are not discarded:
they train feasibility/runtime/fidelity response surfaces. Whether a candidate
is usable is decided later by its request's fidelity budget.

**Phase-4 feedback diagnostic:** before Wave B, isolate the mechanism behind
PLT batch drift with five 1B rows: canonical reference, session-capacity
control at logical `b256`, logical `b256` at a constant 512-feature refresh
window with session/physical `b128`, physical `b256` at that same window, and
logical `b256`/physical `b128` with the stale 1024-feature window. Hold
source/logit semantics and decoder configuration fixed. Use paired compact
comparisons to decide whether the
governor should cap a frontier-window budget independently while still scaling
physical microbatches aggressively.

**Diagnostic closed 2026-07-19:** jobs `1639225` and `1639431` show that
logical feature grouping is semantic even at a fixed 512-feature aggregate
frontier window (`Feature J=0.99367`, `Top-64 J=0.93939`), and doubling the
frontier window adds independent tail-graph drift (`Edge J=0.99203`). Session
capacity is graph-exact, and Phase-4 physical microbatch `128 -> 256` is
effectively exact while improving runtime by 1.39x. Treat logical grouping and
frontier window as fidelity-budgeted inputs; let the solver optimize session
capacity and physical microbatch from available resources.

**Static coalescing correction closed 2026-07-20:** the diagnostic also exposed an
avoidable implementation coupling. At a nominal 512-row window, changing
`128x4` to `256x2` changed the locality-shaped frontier from, for example, 509
rows to 512 before execution. The next frontier therefore started from a
different committed graph. Preserve the canonical semantic planner and refresh
checkpoints, then coalesce consecutive semantic batches only in the physical
executor. The canonical control is `phase4_execution_batch_max_rows`; it may be
larger than `feature_batch_size` but cannot cross the prepared refresh frontier.
For the current `128x4` baseline, 512 is therefore the useful static ceiling.

The first immutable gate completed as a strict 1B PLT matrix with execution/session caps
`128`, `256`, and `512`, fixed semantic batch 128, refresh stride 4, and all
other controls pinned to the corrected-hook baseline. Require equal refresh
counts, semantic batch counts, prepared-frontier membership/order hashes, exact
compact graph topology, and weighted-edge Jaccard of at least `0.999999`.
Record score-ranked pre-locality order as advisory because physical floating-point
differences may reorder equal-membership candidates before canonical locality
scheduling. The gate passed: semantic batches stayed at 65, refreshes at 17,
and execution calls fell from 65 to 33 to 17. Total runtime improved by
`1.434x` at 256 and `1.455x` at 512, but 512 nearly doubled reserved CUDA memory
over 256 for only `1.4%` additional total speedup. Treat 256 as the current 1B
PLT knee and 512 as valid headroom. Adaptive refresh remains deferred.

**Wave B corrected transfer matrix:** do not rerun the complete 36-row Wave A
campaign. Its old coupled PLT batch rows remain semantic/stress evidence, while
the immutable static-coalescing gate supplies exact 1B Phase-4
`128/256/512` evidence. Wave B fixes canonical 4B/12B source, feature, logit,
Phase-1, Phase-3, and refresh semantics; tests coupled session/Phase-4 execution
envelopes while deferring their isolated cost fit to Wave C; keeps
`c16384/c32768` opt-in; and includes one constant-frontier semantic-transfer row
per model. Run seven bounded rows per model through the immutable packaged
launcher on regular RAI H200 QOS: 4B at `400G/2h`, 12B at `600G/8h`. The
existing generated 4B/12B session/tiled-policy files are historical gates, not
Wave B inputs.

**Wave B closed 2026-07-21:** immutable Granite arrays `1642077` (4B,
`400G/2h`) and `1642078` (12B, `600G/8h`) completed all 14 rows successfully.
Execution coalescing transferred substantial speed (`1.52-1.60x` at 4B and
`1.70-2.14x` at 12B), but no larger-model arm preserved the full prepared
frontier contract. The 4B 256 arm retained compact topology and
weighted-edge Jaccard `0.99999999`, while its frontier membership still differed
at 2/17 refreshes. The 12B 128 arm reached only `0.98946` edge and `0.99009`
weighted-edge Jaccard; the 256 arm was closer at `0.99950`/`0.99953`, but also
diverged in frontier membership. This is not a failure of the governor design:
it establishes that execution grouping is numerically sensitive and gives the
fidelity model high-value observations at useful speed points. Exact mode must
remain model/scope-certified; bounded and best-effort modes may use these rows
when their predicted lower bounds or penalties permit it. Wave C may fit around
these measured knees without first turning them into exact defaults.

**Current implementation task:** execute the Wave-C prelaunch gate and ten-row
local-mechanism matrix defined in `docs/governor_calibration_matrix.md`. The
sibling governor owns the extensible response-model registry, fitting,
uncertainty, immutable bundles, and runtime evaluation. The project owns
artifact normalization, explicit fit/held-out splits, historical backfill,
Slurm finalization, campaign generation, and publication orchestration. The
initial estimators are intentionally small, but their family registry and bundle
contract must support later replacements without changing ingestion.
The prelaunch implementation must emit separate 4B/12B launcher files, backfill
historical rows through an explicit observation manifest, publish and reload a
preliminary bundle, and keep bundle activation opt-in. The dependent finalizer
maps each array task to one scenario root, preserves baseline provenance, joins
step accounting/GPU sidecars, and publishes the post-campaign bundle.

Hard requirements constrain variables rather than replacing optimization. The
runtime re-solves at pre-execution, loaded-state, post-Phase-0, and safe phase
entries, inheriting frozen decisions and replacing estimates with observations.
Session capacity, source/Phase-1 scheduling, Phase-3 microbatch, and Phase-4
microbatch require independent bounds and formulas. Checkpoint page-cache policy
remains an admission input until planning moves ahead of model loading.

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
