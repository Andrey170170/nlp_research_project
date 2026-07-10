# Memory Governor and Library Rearchitecture Spec

Status: Implementation-ready target design; A4 gate passed on Cardinal
Last updated: 2026-07-09

This is the "how it is supposed to be" document for the next major rework of
the sibling library `../circuit-tracer_chunked` and its project-side harness
surface. It folds together two previously separate threads:

1. the architecture restructure motivated by the scouting reports
   (`reports/circuit_tracer_architecture_scout.md`,
   `reports/harness_architecture_scout.md`), and
2. the reframing of the optimization-knob sprawl as a budget-driven,
   three-tier **memory governor**.

The execution plan lives in `plans/2026-07-03_governor_rearch.md`; section 10
below is the summary-level version. Step 1 (PLT merge) completed 2026-07-03:
project PR #3 (`e020a34`) and sibling PR #2 (`3abdc98`, incl. review fixes
such as `d77f5cc`), with `mlp.hook_in` now the default input hook for both
CLT and PLT presets on main.

## 1. Motivation and evidence

The current optimization surface grew iteratively and additively: each knob
was added to fix one break or speed up one phase. The result is a wide flat
bag of per-mechanism knobs (`decoder_chunk_size`,
`cross_batch_decoder_cache_bytes`, `chunked_feature_replay_window`,
`exact_encoder_residency`, `row_store_cache_control*`,
`error_vector_prefetch_lookahead`, batch sizes, offload/staging flags, read
caches, frontier buffers, ...) that cannot trade resources off against each
other. Nobody — including the library — owns the question "given this
hardware envelope, what is the fastest safe configuration?"

Concrete exhibit: the corrected-hook 12B PLT pilot (SLURM `12190937_0`,
2026-07-03, `plt_12b_small_initial/t00_828_det_g000`):

- timed out at the 2h wall limit while still inside Phase 3;
- CUDA peak reserved was 58.6 GiB — roughly 35 GiB of VRAM sat idle while
  conservative preset knobs (8 GiB decoder cache, replay window 4) throttled
  replay;
- MaxRSS was ~417 GiB of the 500G request, but process-anonymous memory was
  only ~20 GiB; cgroup **file** (page cache) memory was ~370 GiB, which is
  almost exactly the total transcoder checkpoint size for the 12B width_262k
  set (48 layers x (w_enc + w_dec) x 262144 x 3840 x fp32 ~ 386 GB). Grafana
  shows this cache component separately in the cgroup allocation, confirming
  the split.
- After the lazy readers touched every layer once, the working set plateaued
  (~400 GB) and stayed flat for the rest of the run. In the healthy regime,
  file-tier demand is bounded by total file bytes and saturates.

The pilot burned its budget in the file-cache tier while the GPU starved —
and no single existing knob owns that tradeoff. Only a budget-level planner
can make it.

Separately, the whole library is being restructured anyway (see the scout
reports: `attribute_nnsight.py` is a ~12.5k-line mega-module with low
locality and a knob-bag interface). The governor is not an add-on to that
restructure; it *is* the "exact-trace policy/config seam" the scout report
recommends, given a concrete job description.

## 2. Philosophy

> Teach the circuit-tracing library to be aware of its own weight and manage
> it inside a provided envelope — and give it granular instruments to do so.

Three principles:

1. **Mechanism/policy separation.** Each subsystem's job is to expose a
   ladder of *semantically identical* residency/execution modes (mechanism).
   The governor's job is to pick rungs given budgets (policy). Neither knows
   the other's internals.
2. **Degrade without changing semantics.** Every large structure should expose
   slower semantics-preserving rungs. “Never die; degrade” applies only while
   such a rung exists. If no semantics-preserving plan fits, `strict` admission
   refuses before model load with the binding resource and actionable ways to
   change the request or allocation. It must never survive by silently changing
   logical trace semantics. Walltime remains a first-class predicted constraint.
3. **Provider-agnostic coverage.** The governor is for every model/transcoder
   pair the library can currently run through the exact/chunked provider path,
   not just the Gemma/GemmaScope2 pairs that motivated the work. If Gemma 3
   12B + PLT, GPT-OSS 20B + PLT, or Llama 3.1 8B + a top-k transcoder is a
   supported provider pair, it should receive the same planning, ladder, and
   telemetry machinery. The governor may require a provider to declare
   capabilities/cost metadata; it must not require a governor-code special case
   for each model family or transcoder architecture.

### 2.1 Supported-provider coverage contract

The target boundary is **provider-semantics preserving**:

- For a fixed prompt, scenario, model, transcoder provider, checkpoint, hooks,
  dtype/caste settings, and explicit semantics knobs, governor decisions may
  only change residency/execution strategy, not the compact output. If a
  provider is approximate by construction (for example, a top-k transcoder), the
  governor preserves that provider's existing semantics; it does not decide the
  top-k threshold/cap from memory pressure.
- The governor core consumes a provider profile/capability object, not strings
  such as `gemma`, `GemmaScope2`, `clt`, or `plt` as policy branches. CLT vs PLT
  and future topologies are expressed as provider topology/capability fields
  (`decoder_output_topology`, lazy decoder/encoder support, row materialization
  support, checkpoint byte sizes, hook map, layer/feature dimensions, dtype, and
  measured/declared unit costs).
- A new model/transcoder pair should become governor-eligible by implementing or
  adapting to the provider contract. It should not require adding a new branch to
  the governor planner unless it introduces a genuinely new mechanism/ladder.
- Missing capabilities degrade cleanly. A provider that cannot support a fast
  rung (for example, no lazy encoder rows or no decoder cache) still gets a
  valid plan using the supported rungs or a clearly reported compatibility mode;
  unsupported capabilities are telemetry/admission facts, not hidden fallbacks.
- Cost formulas are parameterized by provider metadata and measurements. The
  Gemma/GemmaScope2 1B/4B/12B data calibrate constants and validate the first
  implementation, but they are not a lookup table defining the governor's scope.

## 3. Memory model

### 3.1 Three tiers

| Tier | Contents | Managed via |
|---|---|---|
| VRAM | model weights (permanent), phase working sets, decoder chunk cache, replay windows, resident encoder rows | fraction budget, headroom pool, ladders |
| Host RAM (anonymous) | staged CPU tensors, pinned buffers, read caches, process state | rigid budget from cgroup limit |
| File-backed | lazy checkpoint reads, row-store memmap, spill files; lives in page cache + NVMe/Lustre | eagerness policy, fadvise advisories, spill roots |

### 3.2 Demand classes: rigid vs elastic

Tier alone is not enough; the ledger must also track **demand class**,
because the failure modes are asymmetric:

- **Rigid** (VRAM allocations, host anonymous memory): exceeding the budget
  is fatal (CUDA OOM / cgroup OOM-kill). Rigid reservations are
  admission-blocking.
- **Elastic** (page cache from file-backed access): exceeding the allowance
  degrades — the kernel evicts and re-reads cost time, not the job. Elastic
  projections are throughput predictions, not admission blockers.

The 12B pilot was healthy by this metric: ~20 GiB rigid against a 500G
limit; the scary-looking 400 GB was almost entirely elastic. A per-run
governor report must present this split instead of leaving it to be
reverse-engineered from MaxRSS.

OSC nuance: cgroup v2 counts page cache toward the job's memory limit, so
tier-3 usage is **not free RAM relief**. The host budget must carry an
explicit file-cache allowance, and the existing fadvise machinery is
promoted from bolt-on to first-class tier-3 eviction policy.

### 3.3 The fourth budget: local disk capacity and spill targets

Node-local `/tmp` NVMe has its own capacity limit, distinct from RAM and
scratch. Multi-TB row stores may not fit on the node at all. Spill ladder:

```text
host RAM -> node-local NVMe (/tmp) -> project scratch (Lustre, ~100 TB/project quota)
```

Scratch is effectively unlimited for our purposes but shared and
lower-bandwidth; the governor treats the spill target as a derived choice
(capacity first, then bandwidth) built on the existing
`row_store_temp_root_policy` seam.

### 3.4 Current CHPC context and historical OSC evidence

Granite/H200 is the current execution and calibration environment. The concrete
Cardinal/Ascend measurements in this document are historical OSC evidence, not
current launch policy. The planner discovers the actual SLURM/cgroup, device,
local-disk, and scratch envelope rather than relying on a fixed 400--700 GB OSC
allocation heuristic. Small problems may run warm; large problems stream based
on projected file working set versus the discovered host allowance.

CHPC baselines calibrate cost and resource models only. They do not become
matched A4 execution-caste evidence unless a separately scoped validation
campaign promotes them as such.

### 3.5 Coupled NNSight trace-capacity constraint

The governor must not treat the current batch knobs as independent scalar
limits. In the current NNSight exact backend, the initial forward pass creates a
single trace capacity by expanding the prompt batch to:

```text
trace_capacity = max(source_batch_size, feature_batch_size, logit_batch_size)
```

Later Phase-3 and Phase-4 `compute_batch(...)` calls reuse the cached NNSight
activations from that forward pass, so their backward batch dimension cannot
exceed this trace capacity without rebuilding the trace/session. This means:

- lowering only `phase1_trace_batch_size_max` / source batch does **not** reduce
  Phase-1 forward VRAM if `feature_batch_size` or `logit_batch_size` remain
  larger;
- lowering Phase-3/Phase-4 microbatches below trace capacity can reduce later
  working sets and runtime shapes, but it does not recover the initial forward
  trace-capacity memory;
- any governor plan that intends to reduce trace-capacity memory must coordinate
  `attribution_batch_size` / source batch, `feature_batch_size`,
  `logit_batch_size`, and `phase1_trace_batch_size_max` as one coupled batch
  family;
- `feature_batch_size` additionally affects Phase-4 refresh/frontier cadence, so
  changing it is not merely a free memory dial unless validated for the scenario.

The plan output should therefore include an explicit `trace_capacity` line item
and a binding-reason report, e.g. `trace_capacity=max(...)=1024, binding=feature`
or `binding=logit`. A configuration that lowers one member of the family while
another still binds should be reported as such, not presented as a successful
memory reduction.

## 4. Correctness invariant and knob castes

**Logical trace semantics are fixed before admission and never selected from
runtime conditions such as free memory, measured frontier margins, or allocator
pressure. The governor controls physical execution only.**

### 4.1 Logical semantics versus physical execution

The current API conflates these concepts; the new types must separate them:

| Logical `TraceSemantics` (semantic fingerprint) | Physical `TracePlan` (execution fingerprint) |
|---|---|
| decoder reduction tile and reduction order | decoder fetch chunk and cache chunk/bytes |
| frontier refresh stride and explicit checkpoints | physical microbatch and prefetch depth |
| dtype, hooks, provider approximation, caps, row-store content | residency, spill target, cache policy, ladder rung |
| sequence/window state-reuse semantics | buffers, scheduling, transfer overlap |

The planner may change the right column under the selected fidelity contract.
It must not infer a right-column value by reinterpreting a legacy knob that also
changes the left column. Legacy arguments are translated deterministically into
both columns by the versioned compatibility layer (section 9.2).

Selection/frontier margins are validation diagnostics and runtime warnings.
They are never a semantic gate: a large measured margin does not authorize a
different reduction order or refresh checkpoint, and low free memory does not
authorize a semantics change.

### 4.2 Fidelity modes

- **`strict` (default):** only semantics-preserving physical plans and mechanism
  rungs are admissible. If none fits, admission refuses actionably before model
  load. “Never die; degrade” applies only within this semantics-preserving set.
- **`validated_relaxed`:** named semantic substitutions may be selected only
  from an explicit allowlist. Every entry names its semantic delta and carries a
  versioned evidence record with provider/checkpoint/hook, model, dtype,
  scenario/window, environment, compared configurations, metrics, and acceptance
  thresholds. The request must match that scope. Drift evidence is not a
  guarantee and cannot be generalized to an unlisted regime.
- **`research`:** the caller supplies explicit semantic overrides. The request,
  result, semantic fingerprint, and provenance label them as research; the
  runtime makes no equivalence claim.

Unclassified legacy knobs are semantics-touching and remain pinned in `strict`.
Performance mechanisms such as cache budgets, physical fetch chunks, prefetch,
residency, and spill may be governor-controlled only after tests establish that
their implementations preserve a fixed `TraceSemantics`.

### 4.3 A4 decision and exact validation scope

Cardinal A4 is sufficient to proceed with the sibling implementation. Ascend
job `6260319` is deferred, non-blocking, optional environment follow-up.

The metrics below are transcribed from the OSC-side Codex comparison summary
supplied by the user. Cardinal source artifacts and generated campaign files are
pending transfer to CHPC. This is sufficient to freeze the API and begin
mechanism work, but a validation profile cannot ship as a runtime default until
the transferred artifacts reproduce the derived report and provenance record.

Scope: corrected-hook Gemma-3-1B + GemmaScope2 PLT-small, Cardinal, fp32,
`t01_361_s1002_g300` and the explicit 298--300 window, using the indicated
configurations. Metrics are **feature / edge / weighted-edge similarity** and
**top64 / top256 / top1024 / top5000**:

| Comparison | Feature / edge / weighted | Top64 / 256 / 1024 / 5000 |
|---|---:|---:|
| chunk c4096/b256 vs A3 c8192/b256 | .994643 / .988269 / .987517 | 1 / .996 / .993 / .992 |
| batch c8192/b128 vs A3 c8192/b256 | .990040 / .987479 / .984281 | .969 / .984 / .989 / .991 |
| per-token g300 vs A3 | .998292 / .992627 / .992163 | 1 / 1 / .995 / .995 |
| window-reuse g300 vs A3 | 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| window-reuse vs per-token g298 | 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| window-reuse vs per-token g299 | .984737 / .973165 / .976543 | 1 / .992 / .992 / .991 |
| window-reuse vs per-token g300 | .998292 / .992627 / .992163 | 1 / 1 / .995 / .995 |

A4 has three tracks: **execution-caste** (the Cardinal comparisons above,
decision-bearing), **environment reproducibility** (Ascend deferred and
non-blocking), and **session regression** (the 298--300 comparisons). This
evidence does not clear 4B, 12B, CLT, another dtype, another cluster, or another
provider regime. Historical wrong-hook results remain provenance only.

## 5. Plan epochs

The governor is not a continuous memory manager. It acts at a small number
of discrete decision points, each with strictly more information:

1. **Admission (before load).** Inputs: model config, transcoder provider
   profile/capabilities (or a preset resolving to them), prompt/prefix length,
   hardware (VRAM queried; host budget auto-discovered from the SLURM/cgroup
   limit — not typed by the user). Only closed-form estimates are available.
   All *irrevocable* (semantics-caste) knobs are fixed here from the
   deterministic scenario/provider formula, conservatively.
   Output: a plan statement — predicted per-tier rigid/elastic demand,
   selected ladder rungs, estimated phase times vs walltime — or an early,
   explicit "will not finish / must shrink X" instead of a death 90 minutes
   into Phase 3.
2. **Post-load calibration.** The model is resident: measure it — that is
   the permanent VRAM line item. Load one decoder chunk and one encoder-row
   read: unit costs are now measured, not estimated. Everything remaining
   under `vram_fraction x total` after permanent + worst-case phase working
   set becomes the **headroom pool** (vLLM's profile-then-claim move).
3. **Post-Phase-0 re-plan.** Phase 0 measures the only real unknown: `nnz`.
   Row-store bytes, encoder residency, replay working sets all become
   arithmetic. Spend the headroom pool here: bigger decoder cache, wider
   replay window, deeper prefetch — performance-caste levers only.
4. **Phase transitions.** Each phase declares its working-set shape,
   receives a grant from the ledger, and returns it on exit. Ledger entries
   carry (tier, demand class, lifetime: permanent / phase / transient).

The hand-tuned size-aware presets for the 1B/4B/12B stress campaign (batch
1024/512/256, chunk 8192/4096/2048, cache 32/16/8 GiB) are a lookup-table
compilation of epochs 1–2 for one provider family; the governor is the
generalization of that table into provider-parameterized formulas plus
measurement, and reproducing those presets is one of its acceptance tests.

## 6. Degradation ladders (mechanism catalog)

Each large structure exposes ordered modes; **all rungs must produce
bitwise-identical outputs for the fixed provider semantics**. Rung selection is
policy (governor); rung implementation is mechanism (subsystem). Rung catalogs
are capability-filtered: if a provider lacks a fast mechanism, the governor
selects a supported slower rung or reports compatibility mode rather than
special-casing the provider name.

| Structure | Rungs (fast → survivable) |
|---|---|
| Transcoder weights | fully resident → lazy readers + warm page cache → streaming reads with aggressive advisories |
| Encoder rows | eager materialized → lazy per-request |
| Decoder chunks | large cross-batch cache → bounded cache → streamed, no cache |
| Row store (dense) | RAM-resident → file-backed full memmap (today) → **tiled/windowed** (bounded materialized slice, stream the rest) → recompute-on-demand (store nothing; re-derive rows when Phase 4 asks) |
| Spill target | host RAM → node NVMe `/tmp` → project scratch |
| Transfer overlap | double-buffered prefetch pipelines (one shared mechanism, per-phase configured), generalizing today's scattered `error_vector_prefetch_lookahead` / replay-window lookahead |

The table is a mechanism catalog, not a CLT/PLT matrix. Provider adapters map
their topology onto the catalog: cross-layer CLTs, same-layer PLTs, and future
top-k/other transcoders expose the chunks, rows, cacheability, and spillability
they actually support.

### 6.1 The tiled row store

Sizing: `(max_feature_nodes + 1) x nnz x 4 bytes`. Verified against the 12B
pilot: 8193 x 78,021 x 4 ≈ 2.56 GB (matches the logged
`preallocate_nbytes`). At corrected-regime L0, a ~13k-token CoT on a large
model legitimately reaches ~100M actives → **~3.3 TB**, which must never
require full materialization. The tiled rung keeps a bounded window of
columns materialized and streams the rest; the recompute rung trades
storage for compute entirely. Tiling and recompute are output-invariant
(performance caste). **Row truncation/compression at write time is not** —
it changes what Phase 4 sees and is excluded from the governor's authority
(scenario-level decision only).

Precedent that the slow rungs are workable: vLLM's swap-vs-recompute pair
for KV blocks under preemption; MegaTrain (arXiv 2604.05091) runs
permanently on the bottom rung (params live in host RAM, GPU as transient
compute engine, double-buffered streams hiding transfer latency).

## 7. User-facing surface

The sibling library owns five public value objects:

- `TraceRequest`: provider/model reference, input/targets, requested operation,
  `TraceSemantics`, fidelity mode, and optional named evidence/overrides.
- `TraceSemantics`: all logical choices that define the semantic fingerprint.
- `ResourceEnvelope`: device, VRAM fraction/bytes, host rigid/file allowances,
  local/scratch capacities, walltime, and explicit operator constraints.
- `TracePlan`: resolved physical execution, estimates, warnings/refusal reasons,
  profile/evidence versions, and semantic plus execution fingerprints.
- `TraceResult`: outputs/artifact references, fingerprints, provenance, terminal
  status, and telemetry stream summary.

First-class runtime entry points are:

```text
trace_one(request, resource_envelope) -> TraceResult
trace_batch(requests, resource_envelope) -> list[TraceResult]
open_session(request, resource_envelope) -> TraceSession
TraceSession.trace_sequence(...)
TraceSession.trace_window(..., reuse=explicit)
```

`trace_batch` plans shared loading/cache use without forcing requests to share
logical semantics. `open_session` makes lifecycle, sequence state, and window
reuse explicit; reuse is never inferred from available memory. `attribute(...)`
remains a compatibility facade over these APIs during migration.

The ordinary resource surface collapses to a handful of targets:

```text
vram_fraction:      0.90          # of detected device memory
host_budget:        auto          # from SLURM/cgroup limit; explicit override allowed
cache_policy:       auto          # warm | bounded | streaming | auto
planner:            v2            # planner/machinery version
spill_roots:        auto          # tmp -> scratch ladder; explicit override allowed
fidelity:           strict        # strict | validated_relaxed | research
```

- `cache_policy=auto` resolves from projected file working set vs host
  allowance: **warm** (small problem: let page cache grow, no advisories),
  **streaming** (big problem: aggressive DONTNEED after append/read, sized
  read caches as the only deliberate retention), **bounded** in between.
- All existing per-mechanism knobs survive as explicit *overrides* of the
  governor's derived values during the compatibility window. Logical overrides
  require `research` or a named `validated_relaxed` allowlist entry; physical
  overrides remain plan inputs.
- Existing explicit model/transcoder/provider selection remains outside the
  budget surface. The governor receives the resolved provider profile and plans
  from capabilities; it does not infer policy from repo names or model family
  strings.

## 8. Telemetry contract

Telemetry is a versioned, append-only stream emitted during planning, provider
loading, execution, cleanup, refusal, cancellation, and failure. It must not
wait for a successful `TraceResult` or retain the complete event history in
memory. Every event carries request/run ID, monotonic sequence number, timestamp,
event-schema version, semantic fingerprint, execution fingerprint, provider
profile version, and validation-evidence version (if any). Consumers tolerate
unknown additive events and can detect gaps or a truncated stream.

Every run reports, per phase and per tier:

1. predicted vs actual bytes, split rigid/elastic (the governor's own
   report card; feeds cost-model refinement);
2. ladder rungs selected and why (budget arithmetic shown);
3. cache hit/miss/eviction and load seconds (already exists; kept);
4. per-phase throughput, so the governor can conclude "Phase 3 got
   everything and is still slow — this is now a compute problem" (the
   signal that the next win is algorithmic, e.g. scheduler v2, not
   budgetary);
5. **selection-margin diagnostics** at Phase-4 selection: did the
   `max_feature_nodes` cap bind; relative score gap between the k-th and
   (k+1)-th feature; count of features within epsilon of the boundary.
   This is the per-run answer to "how brittle is this output to
   floating-point reordering" and the per-run form of the Track-A
   amplification question. Used as validation evidence and warning only, never
   as a semantic or free-memory-dependent runtime gate;
6. the Phase-0 sanity gate: `active_features ≈ tokens x layers x trained
   L0`, as a hard screaming check — the structural fix for the class of
   error that let the wrong-hook contamination shape conclusions for weeks.
7. provider identity and capabilities: architecture/topology, hook map,
   checkpoint/provider fingerprint, declared/measured dimensions and bytes,
   selected compatibility fallbacks, and any missing capability that forced a
   slower rung.
8. admission and lifecycle events: discovered `ResourceEnvelope`, predicted
   plan, binding constraints/actionable refusal, session open/reuse/reset/close,
   batch sharing decisions, cancellation, cleanup, terminal status, and dropped
   event count.

## 9. Relation to the module-split restructure

### 9.1 Ownership boundary

The sibling `../circuit-tracer_chunked` owns the complete tracing runtime:

- `TraceRequest`, `TraceSemantics`, `ResourceEnvelope`, `TracePlan`, and
  `TraceResult` schemas and fingerprint implementations;
- provider loading, capabilities, provider profiles, and profile-version
  validation;
- the pure resolver, governor/ledger/epochs, degradation mechanisms, row store,
  replay, caches, and streaming telemetry;
- first-class `trace_one`, `trace_batch`, and `open_session`, including sequence
  tracing and explicit window reuse;
- `attribute(...)` as a compatibility facade over the same runtime.

The project owns the experiment harness:

- scenarios, fixtures, campaign/wave definitions, and expected comparisons;
- SLURM submission and CHPC cluster/resource policy (Granite/H200 current; OSC
  profiles historical);
- immutable workspace snapshots and two-repo provenance;
- experiment layout, artifact promotion, extraction, comparison, and scientific
  interpretation;
- generation of calibration observations. The sibling consumes only promoted,
  versioned provider profiles, never project-internal campaign objects.

This follows the scout split: attribution phases become governor consumers,
policy resolution becomes a deep sibling module, and row-store/replay mechanisms
land beside the runtime that uses them. The governor consumes capabilities and
profiles, not model/provider name special cases.

Repository integration is an editable package during development and immutable
project+sibling snapshots for runs. A git submodule is explicitly rejected: it
does not create the API boundary and makes CHPC snapshot/provenance workflows
more brittle. A third package or plugin abstraction is deferred until the new
boundary reveals proven shared code.

### 9.2 Compatibility migration

1. Inventory each legacy argument and classify each effect as logical semantics,
   physical execution, provider selection, or harness policy.
2. Split conflated controls. At minimum, separate logical decoder reduction
   tile/order from physical decoder fetch/cache chunk, and logical frontier
   refresh stride/checkpoints from physical microbatch.
3. Add a versioned translator from legacy `attribute(...)` arguments into
   `TraceRequest` + `ResourceEnvelope`. Emit a structured deprecation event with
   the exact translation. Reject conflicting legacy and new fields before load.
4. Compute and persist semantic and execution fingerprints independently.
   Semantic fingerprints include logical fields, provider/checkpoint/hooks,
   dtype, fidelity mode, and evidence version. Execution fingerprints include
   physical plan, profile/planner version, environment, chunks, caches,
   microbatches, and ladder rungs.
5. Keep one documented compatibility window. New APIs and the facade must call
   the same resolver and mechanisms; no dual runtime implementation is allowed.

The governor package is unit-testable with synthetic cross-layer, same-layer,
and top-k/approximate providers. New provider eligibility comes from metadata or
a genuinely new mechanism rung, not a model-family branch.

## 10. Execution plan

Ordered; completed historical steps remain here for provenance.

### Step 1 — Merge PLT parity + hook fix (DONE 2026-07-03; history)

- Merge `feature/plt-optimization-parity-harness` (project) and
  `feature/plt-optimization-parity` (sibling) into their `main` branches,
  including the currently-dirty harness files in the PLT worktree.
- Apply the same `mlp.hook_in` input-hook fix to the CLT preset path —
  main's CLT path is still wrong-hook contaminated as of 2026-07-03.
- Re-establish corrected-hook CLT baselines; do not reuse any
  `hook_resid_mid`-era GemmaScope2 CLT artifacts as references.

### Step 2 — A3/A4 evidence (A4 GATE PASSED 2026-07-09; history)

A3 supplied survival/cost observations. A4 supplied the narrowly scoped
execution-caste and session evidence in section 4.3. Cardinal is sufficient to
proceed. Ascend `6260319` is optional environment reproducibility follow-up and
does not block implementation.

### Step 3 — Freeze semantics, evidence, and profile schemas (login-safe)

Extend `docs/knob_api_taxonomy.md`: for every knob — tier, bytes-cost
formula, caste, validated-under provenance, and whether it is provider-declared,
scenario-declared, or governor-derived. This is the requirements doc for the
governor. Define `TraceSemantics`, semantic/execution fingerprint schemas,
fidelity allowlist evidence schema, provider calibration profile schema, and the
legacy translation table before mechanisms move.

### Step 4 — Governor v0 as a pure resolver in the sibling

Implement the pure `TraceSemantics` + provider profile + `ResourceEnvelope` ->
`TracePlan` resolver directly in `../circuit-tracer_chunked`. Do not create a
disposable project-side resolver. The project generates calibration; promoted,
versioned profiles are sibling inputs. Recorded presets and synthetic providers
are arithmetic fixtures, not semantic-equivalence evidence.

### Step 5 — Runtime APIs, compatibility facade, and streaming telemetry

Land the five value objects and `trace_one`, then route `attribute(...)` through
the versioned translator. Add incremental success/failure/refusal telemetry and
separate fingerprints. Follow with `trace_batch` and `open_session`, including
sequence and explicit window-reuse tests, before removing old internal paths.

### Step 6 — Mechanism split and semantics-preserving ladders

Split the attribution mega-module, row store, replay, provider loaders, and
caches behind the new runtime. Implement bounded/tiled/recompute and admission
rungs. In `strict`, each rung must preserve the semantic fingerprint and pass
parity tests; otherwise it is unavailable, not an automatic degradation.

### Step 7 — Project harness migration and CHPC validation

Map scenarios and Granite allocations into sibling requests/envelopes; persist
streaming telemetry and both fingerprints. Keep scenarios, campaigns, SLURM
policy, snapshots, extraction, comparison, and interpretation project-side.
Validate on Granite/H200 and preserve Cardinal/Ascend records as historical.

### Step 8 — Dynamic post-Phase-0 spending

Enable epoch-3 re-planning only after the decoupled physical mechanisms pass
strict fixed-semantics parity tests. A4 motivates the split but its observed
drift is not such a proof. Dynamic decisions may change physical execution only;
frontier margins remain warnings.

## 11. Acceptance criteria

1. A run can be launched through `trace_one` with logical semantics and a
   resource envelope; all physical values are derived, logged, and separately
   overridable. `trace_batch` and `open_session` are first-class, not harness
   loops over private internals.
2. Semantic and execution fingerprints are independently stable and persisted.
   Changing physical fetch/cache chunk or microbatch in `strict` changes only
   the execution fingerprint; logical reduction or refresh changes the semantic
   fingerprint.
3. `strict` uses only semantics-preserving rungs and demonstrates actionable
   pre-load refusal when none fits. `validated_relaxed` requires a named,
   versioned, scope-matched allowlist record. `research` overrides are explicit
   in request, result, and provenance. Drift evidence is never represented as a
   guarantee.
4. The compatibility facade deterministically translates legacy knobs, warns
   with a schema version, rejects old/new conflicts, and matches new-API outputs
   and fingerprints during the compatibility window.
5. Streaming telemetry covers planning, success, failure, refusal, cancellation,
   session lifecycle, and cleanup; reports predicted/actual rigid and elastic
   demand, ladder decisions, margins as warnings, Phase-0 sanity, sequence gaps,
   and terminal status without buffering the full stream.
6. Mixed-shape batching tests verify result isolation, ordering, shared-resource
   accounting, partial failure, and cancellation. Session tests verify explicit
   reuse, independent sequence steps, the A4 298--300 window regression, reset,
   cleanup, and failure recovery.
7. The row store completes more slowly where full materialization exceeds local
   capacity via semantics-preserving tiled/recompute rungs, or strict admission
   refuses actionably if no validated rung fits.
8. The sibling resolver consumes promoted, versioned provider calibration
   profiles. The project retains calibration generation, campaigns, CHPC policy,
   snapshots, extraction, comparison, and interpretation.
9. Granite/H200 validation provenance records both repository states/snapshot
   IDs, SLURM job/allocation, output root, envelope, profile/evidence versions,
   and both fingerprints. CHPC calibration results are not mislabeled as A4
   semantic evidence.
10. Any supported provider obtains a plan from capabilities/profile metadata;
    missing capabilities are explicit. No git submodule, third package, plugin
    system, or model-family governor branch is introduced in this phase.

## 12. Open questions

1. Exact shape of the walltime estimator; Phase 3 on 12B may be compute-bound
   regardless of budgets.
2. Whether the recompute rung of the row-store ladder is ever cheaper than
   tiled streaming in practice on Lustre, or is kept only as the
   survivability floor.
3. Which additional regimes merit their own `validated_relaxed` evidence. A4
   itself remains limited to the section 4.3 scope.
4. Whether a third package/plugin boundary becomes justified after the sibling
   API and project adapter have stabilized; it is not part of this rework.
