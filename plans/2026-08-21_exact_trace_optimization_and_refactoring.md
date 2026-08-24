# Exact-trace survivability, correctness, optimization, and refactoring plan

Status: active; revised to survivability-first execution

Date: 2026-08-21

Major revision: 2026-08-23

Scope: exact-trace memory survivability, correctness evidence, opportunistic
performance, staged physical-plan selection, long-prefix/model stress, and the
refactoring needed to keep one canonical typed runtime. This plan does not by
itself authorize a fidelity reclassification, baseline/default promotion, or
governor response-bundle promotion.

Predecessor:
`plans/2026-07-29_exact_trace_performance_optimization_loop.md`, complete through
the repeat-qualified LS5 12B/1,024 selected circuit.

Registry: `docs/exact_trace_optimization_registry.md`

## 1. Baseline and claim boundary

The committed repeat-qualified evidence is:

- job `1832086`: 517.46-second trace, including 399.97 seconds in Phase 4;
- job `1834396`: 540.81-second repeat, including 410.41 seconds in Phase 4;
- exact selected 8,192 features and historical 20,000-edge global projection
  across the repeat, with the already-recorded bounded interpretation for the
  six typed buckets. Step 1a makes only those typed buckets canonical for new
  evidence.

A newer 12B/1,024 observation completed in under eight minutes. It remains a
separate provisional operational result and is not replaced by the instrumented
rerun below.

Evidence-hardening job `1843641` completed the trace and graph on `grn032`, but
Slurm recorded `FAILED` because the frozen post-run validator treated Torch UUID
`b2c4...` and `nvidia-smi` UUID `GPU-b2c4...` as different devices. The repaired
validator canonicalizes that optional prefix, and replaying the complete gate
against the immutable output passes. Preserve the scheduler failure as
operational provenance; accept the trace as scientific evidence after that
bounded validator repair. Its canonical trace duration was 617.90 seconds and
Phase 4 was 405.18 seconds, so it does not replace the faster reference pair as a
speed baseline.

The corrected primary objective is **survivability**: complete a scientifically
valid trace whenever the irreducible model/provider state, minimum phase working
set, unavoidable `O(N)` influence/ranking state, and compact output fit within a
reasonable supplied envelope. Memory limits are hard constraints. Runtime is a
soft objective inside the feasible set, bounded by the caller's walltime.

Speed mechanisms remain useful, but only as opportunistic upper rungs. In
particular, `cuda_full` should be selected when spare HBM admits it; it is not the
general scaling strategy for 27B or 2K--10K prefixes. The general strategy is a
staged full-resident -> windowed -> tiled -> no-retention/recompute ladder.

## 2. Execution invariants

1. Every GPU/model run uses the project launcher and an immutable snapshot of
   both this repository and `../circuit-tracer_chunked`.
2. Logical trace semantics are fixed before resource admission. Runtime memory
   pressure may select only an authorized physical rung; it never silently
   changes dtype, hooks, caps, reduction order, refresh checkpoints, or provider
   approximation.
3. Each later optimization arm introduces one physical or numerical mechanism.
4. Every new scientific artifact uses one typed-bucket-only graph format. The
   legacy global `max_edges` projection may be read only through an explicitly
   historical adapter; it is never written, accepted, compared, or promoted by
   the canonical runtime.
5. Acceptance uses one schema-owned strict compact-graph validator. Analysis
   loaders must not silently accept a weaker graph than post-run acceptance.
6. CUDA-event current-stream interval-latency measurements are reported
   alongside a complete wall census and reconciled at one explicit lifecycle
   synchronization boundary. The sampled intervals may include host enqueue
   gaps and stream waits; they are not labeled GPU-active or kernel-only time.
7. Device samples identify the actual CUDA device. Batched-VJP metadata states
   what kernel path was requested, what was invoked, and whether fallback is
   observable rather than inferring success from the absence of a warning.
8. A workload-specific batch reduction, larger allocation, or longer walltime
   may localize a boundary but does not close a scaling rung.
9. Correctness is never represented by one comparison label. Structural
   conformance, numerical stability, and behavioral faithfulness receive
   separate verdicts and claim boundaries.
10. A stable graph can still be behaviorally unfaithful. Serious qualification
   runs therefore carry a bounded automatic intervention-closure probe before
   model/runtime teardown once that probe is implemented.
11. The selector first finds an admissible memory plan, then chooses the fastest
    sufficiently supported plan. It must retain rejected candidates, binding
    resources, predicted/actual demand, and the effective rung.

## 3. Ordered implementation sequence

| Priority | Target | Smallest implementation/experiment | Exit gate |
|---:|---|---|---|
| 0 | Evidence hardening and frozen profile rerun — complete | Strict graph validation, per-device identity, event timing, structured VJP evidence, and job `1843641` classification | Preserve this evidence contract in every later arm |
| 1a | Typed-bucket-only canonical artifact | Remove the legacy global-top-K edge payload and `max_edges` launch control from all new writers; migrate validation, comparison, temporal analysis, and promotion atomically to the six typed buckets | New artifacts contain no legacy edge arrays; both full-answer and multi-step writers round-trip the versioned bucket policy; new-run acceptance refuses legacy-only graphs |
| 1b | Three-axis, alias-aware correctness contract and automatic behavioral probe | Specify separate structural, stability, and faithfulness reports; add exact plus decoder-soft frontier comparison and a bounded same-process intervention probe | CPU-safe contract/calibration tests plus one frozen GPU run producing all three explicit verdicts, frontier-equivalence evidence, and bounded intervention evidence within the admitted two-minute probe budget |
| 2 | `OPT-CUDA-FULL-01` opportunistic closure | In one scheduled 12B/1,024 allocation, run the frozen `cuda_windowed` control and an independently admitted `cuda_full` candidate when practical | No 4B prerequisite; combined HBM admission, no fallback, all correctness reports, resource/timing comparison, and no default promotion |
| 3 | Separate retention, access, and reduction planning axes | Replace the mixed feature-row mode with subsystem-owned row retention, influence access/residency, and reduction-execution policies | Residency can vary without implicitly changing FP grouping; every axis has owner, fidelity class, freeze epoch, requirements, and provenance |
| 4 | Qualify the survivability mechanisms | First column-tiled bounded production/full retention, then no-retention deterministic recompute/replay | Complete artifacts with bounded materialization; no full `K x N` allocation in the recompute rung; three-axis correctness reports |
| 5 | Staged automatic selector | Extend the existing governor seam to select the fastest admitted full -> windowed -> tiled -> recompute combination from measured per-phase demand | Deterministic candidate report, hard-pin enforcement, safe-epoch replanning, actionable refusal, and selector replay tests |
| 6 | Long-prefix and larger-model stress | Acquire immutable nested 2K/5K/10K prefixes, run stop-on-failure 12B stress, then admit 27B only after asset/permanent-state inspection | Each failure identifies an irreducible or mechanism-owned resource; no rung passes through a one-off larger allocation |
| 7 | Opportunistic time work | Normalization launch structure, feature compute/replay, encoder materialization, staging/write overlap, cache qualification | One measured mechanism per arm after the owning survivability rung is stable |
| 8 | Consolidation and publication | Typed active-row policy, one backward resolver, prepared-launch interface cleanup, structured failures, and a measured envelope | One canonical runtime and selector; scoped survival/performance/fidelity publication |
| 9 | Deferred high-risk work | Incremental refresh state, exact cross-batch tape reuse, multi-GPU placement, and other distributed mechanisms | Reopen only when the irreducible single-device floor or measured walltime makes them necessary |

The rejected historical mapped-file-window and direct-`pread` experiments are
not the same mechanism as canonical column-tiled production or no-retention
replay. Exact-range LRU, pinned active CPU, generic larger Phase-4 batches,
unprofiled tile/source fusion, and shared-cache eviction remain rejected unless
genuinely new evidence changes their recorded mechanism or cost.

## 4. Immediate tranche: evidence hardening

- [x] Define one schema-owned validated compact-graph representation and strict
  loader covering array lengths, IDs/domains, finiteness, metadata, step/path
  identity, bucket schema, and future-position constraints.
- [x] Route post-run acceptance and analysis/audit entry points through it; a
  validation error must fail acceptance rather than becoming ignored metadata.
- [x] Persist CUDA device index, name, and stable UUID where the runtime exposes
  one; summarize samples per device rather than combining gifted visible GPUs.
- [x] Add a complete wall census plus deterministic stratified-jitter CUDA-event
  interval samples for the Phase-4 refresh and feature paths. Persist the
  sampling/estimator contract, an exact lifecycle event span, and
  reconciliation/residual fields at one lifecycle synchronization boundary.
- [x] Persist structured batched-VJP requested path, effective invocation,
  fallback-observation method, and observed/unknown fallback state.
- [x] Add focused CPU-safe tests for strict rejection, multi-device attribution,
  timing fallback/reconciliation, and VJP evidence serialization.
- [x] Create an immutable dual-repository snapshot and submit
  `EV-12B-1024-PROFILE-01` with the frozen v2 semantic pins. Submitted as
  Slurm job `1843641`; its completed trace was classified after the bounded
  UUID-validator repair described below.
- [x] Strictly accept the graph, compare it with the repeat-qualified reference,
  and re-extract phase/substage, CUDA-event, host-resource, and per-device GPU
  evidence before selecting priority 2.

## 5. Rerun acceptance gate

`EV-12B-1024-PROFILE-01` is accepted only when all of the following hold:

1. the fingerprinted trace specification preserves the frozen v2 semantic pins;
2. the schema-owned validator passes and reports the expected 8,192 selected
   features, the historical 20,000-edge legacy projection, and valid typed
   buckets;
3. the active CUDA device is identified unambiguously even when Slurm exposes
   or gifts more than one device;
4. the event stream and required phase/substage telemetry are complete;
5. CUDA-event and wall timings carry their synchronization scope and reconcile
   without double-counting asynchronous work;
6. the batched-VJP effective path and fallback state are explicit, including an
   honest `unknown` when the runtime cannot observe fallback;
7. the selected circuit and typed buckets are compared with jobs `1832086` and
   `1834396` under the existing claim boundaries; and
8. provenance records project and sibling revisions, dirty states, snapshot
   manifest, output root, Slurm job ID/node/allocation, cache/preheat state, and
   requested versus effective mechanisms.

### Result and priority-2 decision

The repaired acceptance replay passed all eight gates. The selected 8,192
features and historical 20,000-edge global projection were strictly exact
against both jobs `1832086` and `1834396`. All six typed buckets were present
with 1,323,157 typed
edges and no future-position violations. The only non-exact comparisons were
the already-bounded error buckets plus one equal-weight feature-to-feature
cutoff tie against job `1832086`; job `1834396` matched the complete
feature-to-feature bucket exactly.

Phase 4 took 405.18 seconds. The complete wall census and sampled current-stream
interval-latency estimate attributed it as follows; CUDA values include host
enqueue gaps and stream waits and are not GPU-active/kernel-only time.

These values came from the historical-v1 sampler. V1 could omit the final
partial stride block or, when it sampled that block, weight the observation as a
full stride. The 24,101-occurrence refresh population has a five-occurrence
tail, so the refresh estimates below carry a small unquantified tail error. Use
them to localize seams and rank independent experiments, not as exact substage
totals; the v2 sampler must close this evidence limitation on the next run.

| Substage | Wall census | Estimated current-stream interval | Interpretation |
|---|---:|---:|---|
| Feature compute batch | 121.83s | 157.36s | Largest measured stream-critical interval; retain as the leading post-residency seam. |
| Refresh row-store read | 45.86s | 116.02s | Window residency/transfer is far more critical than its legacy wall timer implied. |
| Encoder materialization | 44.26s | 44.26s | Real secondary device-path cost. |
| Row-store write | 37.03s | 37.04s | Interval includes the ordered host write gap; it is not 37s of GPU kernels. |
| Refresh normalization | 81.25s | 9.58s | Primarily host launch/synchronization overhead rather than an 81s GPU kernel. |
| CPU staging | 40.20s | 4.69s | Host/D2H wait-side cost. |
| Denominator | 4.54s | 4.54s | Small exact cost. |
| Other measured plus residual | 30.21s | 31.71s | Includes transfer/cast, direct accumulation, and the reconciled residual. |

The H200 averaged 34.4% SM utilization with p95 and maximum at 100%, while the
32-CPU allocation averaged 22.8% utilization. The workload is bursty rather
than continuously compute-saturated. Peak framebuffer use was 48,861 MiB of
143,771 MiB. This evidence selects `OPT-CUDA-FULL-01` next: it is the smallest
independent physical arm aimed directly at the 116.02-second row-read/window
interval and the measured HBM headroom. It remains the next tactical GPU arm,
but the 2026-08-23 revision makes survivability the strategic priority. Later
normalization work should target its host launch/synchronization structure rather
than treating 81.25 seconds as GPU kernel time, and all time work waits behind
the correctness contract plus tiled/recompute qualification.

## 6. Step 1: canonical artifacts and correctness evidence

### 6.1 Step 1a: typed-bucket-only canonical artifacts

The 12B/1,024 profile did not lose the full typed graph: its saved file contains
a legacy 20,000-edge global-top-K projection alongside six typed top-p/cap
buckets totaling 1,323,157 edges. The defect is architectural rather than a
missing-result incident: the obsolete projection is still written, summarized,
and consumed by active comparison paths, while the canonical multi-step writer
still emits only that legacy representation.

Replace this dual representation atomically. One deep `TypedCompactGraph`
module owns the serialization seam and exposes:

- selected feature identities and the token, error, and logit endpoint domains;
- signed COO edges in the canonical six `target <- source` buckets;
- a versioned `TypedEdgeRetentionPolicy` containing per-bucket top-p, cap, raw
  nonzero count/mass, retained count/mass, retained fraction, and whether a cap
  bound before the requested mass was reached; and
- graph, target, provider, trace, step, and policy fingerprints required for
  comparison and intervention evidence.

The initial policy preserves the already-established typed behavior:

| Bucket | Top-p | Cap |
|---|---:|---:|
| `feature<-feature` | 0.95 | 1,000,000 |
| `feature<-error` | 0.99 | 16,000 |
| `feature<-token` | 0.95 | 250,000 |
| `logit<-feature` | 1.00 | none |
| `logit<-error` | 1.00 | none |
| `logit<-token` | 1.00 | none |

Implementation and migration order:

1. define the typed-only schema and policy fingerprint, with the writer and
   strict loader as the shared test surface;
2. route both full-answer and canonical multi-step persistence through it;
3. remove generic `max_edges` from canonical requests, CLI/scenario generation,
   summaries, and new-run validation; expose only explicit versioned per-bucket
   policy selection;
4. migrate graph comparison, metric batteries, temporal analysis, audit,
   stability analysis, and promotion gates to per-bucket inputs and names;
5. remove `all_edge_*` metrics from current qualification. A cross-bucket view,
   if scientifically justified later, is a labeled derived analysis with its
   aggregation rule and source policies; it is never another persisted global
   top-K or a hidden promotion metric; and
6. retain a read-only historical adapter for old files. No canonical writer or
   loader fallback produces or silently accepts legacy-only output.

Step 1a exits only when a red-capable format test proves new files contain no
legacy `row_idx`/`col_idx`/`weights` edge projection, both writer paths produce
the same typed schema, all active consumers use typed buckets, historical files
remain explicitly readable, and the strict new-run gate refuses legacy-only or
mixed-schema drift. Existing 12B references remain usable because their typed
buckets are already present; no trace rerun is required merely to recover them.

### 6.2 Step 1b: three independent correctness verdicts

No mode receives one undifferentiated `correct` label. Every qualification
report carries these independent verdicts:

1. **Structural and algorithmic conformance.** Required automatically on every
   completed trace. Validate semantic/execution fingerprints, provider and target
   identity, finite tensors/weights, endpoint domains, no future positions,
   complete typed buckets, signed feature/token/error accounting, row
   denominators, refresh/frontier sequence integrity, mechanism requirements,
   and cleanup/terminal status. Failure invalidates the artifact.
2. **Numerical stability.** Compare candidate repeats to estimate within-mode
   noise, then compare the candidate with the canonical ordered-FP32 reference at
   intermediate checkpoints and the typed artifact. Report exact-stable,
   alias-stable, bounded, divergent, or unknown separately by scope; never infer
   correctness from repeat consistency or feature-ID overlap alone. One repeat
   may be deferred when the initial arm is only exploratory, but no mode is
   promoted without repeat evidence.
3. **Behavioral faithfulness.** Measure whether graph-predicted signed effects
   track actual model behavior under deterministic interventions. Report this
   independently from topology similarity and downstream CIE utility.

The canonical reference is a declared algorithmic contract, not a claim that one
hardware execution path is metaphysically correct. A numerically sensitive mode
can qualify only for a scope whose stability and behavioral floors it satisfies.

Reports describe evidence; a separate versioned promotion policy maps those
reports to an admission/default decision. No report contains an aggregate
`correct` flag, and an unknown axis cannot be hidden by success on another.

### 6.3 Equivalence-aware selection-frontier stability

The expected numerical failure mode is churn near the 8,192-feature selection
cutoff, where multiple active features may expose nearly interchangeable
downstream directions. Exact feature IDs remain the first reported comparison,
but they are not the complete stability verdict.

Every qualification run therefore persists a bounded frontier-evidence sidecar
covering selected and near-cutoff unselected candidates: feature identity,
activation, seed/selection influence and rank, selected status/rank, cutoff and
relative gap, tie/near-cutoff counts, and provider/decoder identity. Promote the
existing optional feature-semantic descriptor and cutoff telemetry into this
required bounded qualification evidence rather than retaining every active
feature descriptor.

The automatic comparison proceeds in three layers:

1. **Exact identity:** selected-feature and typed-bucket support/weight/sign
   comparisons remain visible and are never overwritten by soft matching.
2. **Decoder-soft frontier matching:** on entered/exited frontier features only,
   construct deterministic one-to-one same-layer matches from downstream
   decoder-signature cosine. Require compatible activation-scaled direction,
   signed target effect, and typed `feature<-feature`/`logit<-feature`
   neighborhoods. Report matched feature count and recovered selected/influence/
   edge mass at each frozen threshold.
3. **Sampled causal substitutability:** for at most one or two high-mass alias
   pairs, compare the signed endpoint and downstream response to intervening on
   feature A versus feature B; where admitted, ablate A and patch through B.
   Include a same-layer low-cosine control. This supports only prompt-and-target-
   scoped decoder-side substitutability, not universal semantic identity.

Decoder cosine is a candidate generator, not a forgiveness rule: similar write
directions can arise from features with different firing conditions. No alias
claim passes solely on cosine, many-to-one matching cannot double-count mass,
unmatched high-mass churn remains divergence, and exact plus alias-aware metrics
are both retained. Calibrate cosine, neighborhood, effect, and recovered-mass
floors from frozen within-mode repeats before using them for promotion; record
the calibration corpus and do not retrofit thresholds after seeing a candidate.

The offline exact/soft comparison belongs to numerical stability. The sampled
intervention response also contributes to behavioral faithfulness.

### 6.4 Automatic post-trace intervention-closure probe

The intended third-axis check is automatic and bounded, not a downstream CIE
run. Add a sibling-owned verification module invoked after graph construction
and before the loaded model/runtime is torn down. Its small interface consumes
the accepted graph, target state, canonical trace identity, a versioned probe
policy, and the still-live intervention-capable runtime; it returns a typed
`BehavioralFaithfulnessReport` and does not own graph acceptance or rendering.

Version-one probe contract:

- begin with an empty/no-op intervention control and one reusable baseline-state
  capture;
- deterministically choose two or three direct-effect closure samples across
  layer, position, sign, and influence rank, and compare their graph-column
  prediction with realized centered-target-logit/downstream activation deltas;
- run propagated necessity/ranking evidence for a high-influence cohort and a
  matched low-influence or unselected-active control;
- when the candidate/reference comparison exposes a qualified alias pair, use
  one or two of the remaining variants for the causal substitutability check in
  Section 6.3; otherwise record it as not applicable rather than inventing a
  pair;
- batch paired baseline/intervention forwards where the runtime permits;
- measure predicted versus realized target-objective change, sign agreement,
  direct-effect closure, necessity/ablation separation from control, and sampled
  alias substitutability;
- retain the exact intervention recipe, selected nodes/edges, baseline and
  intervened logits, raw effects, aggregate metrics, uncertainty/status, and all
  refusal/failure reasons;
- cap the first implementation at at most eight intervention variants and 120
  seconds after the trace. A later evidence-backed limit may change separately;
- expose `off | smoke | required`. Serious optimization/scaling campaigns use
  `required`; a timeout, unsupported provider, or failed intervention leaves the
  trace structurally valid but behavioral faithfulness `unknown` and blocks
  fidelity/default promotion. The time limit is hard: admit an adaptive variant
  count from measured forward/setup cost and return partial/inconclusive evidence
  rather than overrun it.

Version one intentionally does not claim full circuit sufficiency. Suppressing
everything outside a retained circuit requires settled semantics for unselected
features, reconstruction-error nodes, CLT cross-layer writes, attention, and
LayerNorm. Report `sufficiency=unknown` until a separately designed donor/
recovery or circuit-clamping mechanism earns that claim.

Before coding the GPU adapter, settle the exact CLT and PLT constrained/direct
versus propagated intervention semantics, the trace target functional, reusable
baseline capture and batching isolation, and the initial calibration protocol.
These are bounded Step-1b implementation decisions; they do not reopen the
correctness axes, artifact format, or execution order.

## 7. Immediate opportunistic arm: direct 12B/1,024 `cuda_full`

Queue latency dominates the difference between a roughly five-minute 4B/512
run and a sub-fifteen-minute 12B/1,024 run. The 4B/512 prerequisite is therefore
removed. Use 4B/512 only as an unscheduled/debugging fallback after a concrete
mechanical failure.

Prepare one immutable 12B/1,024 campaign that, when practical, runs the frozen
`cuda_windowed` control and `cuda_full` candidate in the same allocation after
one preheat. Preserve all semantic pins and vary only the candidate mechanism.
Require combined admission for permanent model/provider state, dynamic active
decoder rows, the complete GPU feature-row tier, transient phase working sets,
and the safety margin. `cuda_full` is provisionally numerically sensitive until
row placement/access is separated from reduction grouping and scoped evidence
earns a stronger classification.

This arm closes only an opportunistic fast rung. It does not delay or replace the
survivability work below, and success does not authorize `cuda_full` where it is
not admitted.

## 8. Survivability architecture and staged selection

The row subsystem must expose three independent policies rather than one mixed
mode:

1. **Row retention/ownership:** full file retention, canonical column-tiled full
   retention, or no-retention deterministic replay.
2. **Influence access/residency:** full CUDA residence, bounded CUDA windows, or
   CPU access, each with explicit byte budgets and transactional admission.
3. **Influence reduction execution:** canonical ordered reduction versus any
   separately qualified vectorized/batched implementation.

The first two are physical candidates only when their implementations preserve
the fixed reduction contract. The third is numerically sensitive whenever it
changes FP grouping. Each policy belongs to its sibling subsystem and declares
its fidelity class, freeze epoch, capability requirements, demand/lifetime
model, fallback rules, and telemetry. Project code serializes requests and
evidence but does not mirror the policy tables.

The automatic ladder is:

1. full retention plus full CUDA influence residency;
2. full retention plus bounded CUDA windows;
3. column-tiled production/full retention plus bounded streaming;
4. no-retention deterministic production/recompute with a bounded replay cache.

Encoder lazy/eager placement, decoder cache/streaming, active-decoder-row
residency, Phase-1 saved-tensor offload/recompute, microbatches, and transfer
overlap form independent subsystem ladders. The governor composes candidates
from provider capabilities, hard user pins, the immutable fidelity contract,
and per-phase concurrent demand. At each safe epoch it first rejects plans that
violate hard VRAM/host/disk/walltime constraints, then chooses the fastest
sufficiently supported survivor. It never treats minimum memory as the speed
objective and never weakens semantics to escape pressure.

`tiled` is qualified before `recompute` because deterministic no-retention replay
depends on canonical ordered tile production. The recompute gate must prove that
the runtime never creates or retains a full `K x N` tensor/file while preserving
the unavoidable `O(N)` state, compact output, structural conformance, stability,
and behavioral evidence.

## 9. Long-prefix and larger-model stress acquisition

Prompt acquisition is split by claim type:

1. **Source-agnostic shape stress.** A long natural-language trajectory need not
   have been generated by Gemma to exercise Gemma's tracing shapes. Provenance-
   permitted long Qwen/BonaFide outputs are eligible source text. Tokenize once
   with the frozen Gemma tokenizer, take nested immutable 2K/5K/10K prefixes,
   hash the exact token IDs, and trace Gemma's own next-token target at each
   endpoint. These runs support memory/shape/mechanism claims only, not native
   Gemma-trajectory or answer-quality claims.
2. **Native Gemma trajectory.** Separately attempt prompts/sampling that induce
   long Gemma continuations. Preserve generation provenance and exact token IDs.
   Failure to obtain 5K/10K native continuations does not block shape stress; it
   limits only native-trajectory claims.

Start with at least one nested natural trajectory and add a second contrasting
text family only when the first result shows prompt-dependent active-universe or
admission behavior. Do not use simple token repetition as the sole admission
case because it may underrepresent feature density; synthetic sequences remain
useful as explicit shape controls.

The execution ladder is 12B/2K -> 5K -> 10K with stop-on-first-architecture-
failure behavior. The stress point passes only with a complete artifact, all
required correctness reports, predicted/actual per-tier demand, the selected
survival rung, and no workload-specific capacity exception. Inspect 27B
model/provider assets and permanent-state demand separately; then begin 27B at
the strongest already-qualified prefix/rung that its irreducible floor admits.

## 10. Recording and stop rules

- Update `experiments/logs/2026-08.jsonl` after submission and again after result
  classification. Do not put a provisional run into `EXPERIMENTS.md`.
- Update `EXPERIMENTS.md` only if the accepted rerun changes the compact current
  baseline or interpretation.
- Update the optimization registry when a tranche is validated, not merely when
  code exists.
- Record the 2026-08-23 plan correction as a planning decision, not an experiment
  result or default promotion.
- Complete typed-bucket-only Step 1a before generating another qualification
  artifact or treating any edge comparison as a current promotion metric.
- Implement the three-axis report contract before relying on behavioral
  faithfulness for a numerical-mode decision.
- The next tactical GPU arm is direct paired 12B/1,024 `cuda_full`; it remains an
  opportunistic fast rung and may not displace tiled/recompute qualification.
- Do not combine `cuda_full`, normalization, or another feature-dataflow change
  in one candidate arm.
- Stop a 2K/5K/10K or 27B ladder on the first mechanism/resource failure and fix
  the general shape-dependent architecture before progression.

## 11. Plan completeness and remaining design gates

This document is complete at the execution-strategy level. It fixes the primary
objective, canonical artifact, correctness axes, immediate experiment, survival
mechanisms, selector behavior, scaling ladder, time-work ordering, cleanup, and
publication claim boundaries. Implementation can now proceed in order without a
new strategic choice.

The following values are intentionally derived during Steps 1a/1b rather than
guessed in advance:

- the concrete typed schema/policy version identifiers and full historical
  consumer migration inventory;
- decoder cosine, neighborhood, recovered-mass, and causal-effect floors fitted
  from frozen reference repeats;
- provider-specific CLT/PLT intervention adapter details and supported target
  functionals; and
- the admitted probe variant count under the hard 120-second ceiling after
  measuring reusable setup and forward cost.

Each is already assigned to an owner, evidence source, and exit gate above. None
may silently change the primary objective, canonical semantics, per-bucket
artifact policy, or promotion rules. Reopen the strategic plan only if new
evidence invalidates one of those premises.
