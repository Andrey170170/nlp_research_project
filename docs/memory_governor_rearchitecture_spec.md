# Memory Governor and Library Rearchitecture Spec

Status: Target design, agreed 2026-07-03; not yet implemented
Last updated: 2026-07-08

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
2. **Never die; degrade.** Every large structure must have a
   degraded-but-correct fallback mode, so admission control never has to say
   "does not fit" — it says "fits at mode X with estimated slowdown Y".
   Memory stops being a cause of run death; **walltime becomes the honest
   binding constraint**, which the planner must therefore also estimate and
   warn about (the 12B pilot died of walltime, not memory).
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

### 3.4 Practical ceiling context (OSC)

Typical host allocations run 400–700 GB; ~1 TB is a practical upper bound.
Small problems should be allowed to run "warm" (let cache grow, everything
effectively RAM-speed); large problems must stream. That decision is
automatable: compare projected total file working set (checkpoint bytes
actually touched + row store + caches) against the host allowance.

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

**The governor may only dynamically control knobs that are proven
output-invariant under the currently validated provider regime. Anything
semantics-touching is derived deterministically from (model, transcoder
provider, scenario) — never from runtime conditions like free memory.**

Knobs therefore split into two castes:

- **Semantics-touching** (fixed per scenario, deterministic): batch sizes
  (backprop is pre-initialized on them and cannot change mid-run), internal
  dtype, row-store *content* policy, and — pending revalidation —
  `decoder_chunk_size`. Provider-defined approximation knobs, such as top-k
  feature caps in a top-k transcoder, are also semantics-touching unless
  separately proven invariant.
- **Performance-only** (governor-controllable, even mid-run): cache byte
  budgets, replay window, prefetch lookahead, read caches, fadvise
  eagerness, residency modes, ladder rung selection, spill targets.

### 4.1 Caste is empirical and provenance-tagged

The historical claim "`decoder_chunk_size` changes cause small
compact-output drift" was measured on the wrong-hook contaminated setup
(raw residual into JumpReLU → ~everything active → 1–2M actives competing
for an 8k extraction slot → a razor-thin, tie-dense selection boundary where
any float-summation reordering flips selections). That is a property of the
pathological regime, not necessarily of the knob.

Corrected-regime scale is structurally different:

- CLT short prompt: ~4k actives vs the 8k cap — selection is **non-binding**;
  there is no boundary to perturb. Chunk size is plausibly exactly invariant.
- PLT: 50k–78k actives vs 8k — selection binds (~10:1) but against a healthy
  score distribution. Sensitivity is an open empirical question.

Consequently the knob taxonomy (`docs/knob_api_taxonomy.md`, to be extended)
gets per-knob columns: **tier, bytes-cost formula, caste, validated-under**
(hook regime, provider family/topology, model family where relevant, scenario
family). Contaminated-era evidence does not count as validation. Unproven knobs
are treated as semantics-caste until cleared. If `decoder_chunk_size` is
cleared, the governor gains its single most powerful VRAM lever — this is the
biggest open fork in the design and is resolved by the probe campaign (section
10, step 2).

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

The knob set collapses to a handful of targets:

```text
vram_fraction:      0.90          # of detected device memory
host_budget:        auto          # from SLURM/cgroup limit; explicit override allowed
cache_policy:       auto          # warm | bounded | streaming | auto
planner:            v2            # planner/machinery version
spill_roots:        auto          # tmp -> scratch ladder; explicit override allowed
```

- `cache_policy=auto` resolves from projected file working set vs host
  allowance: **warm** (small problem: let page cache grow, no advisories),
  **streaming** (big problem: aggressive DONTNEED after append/read, sized
  read caches as the only deliberate retention), **bounded** in between.
- All existing per-mechanism knobs survive as explicit *overrides* of the
  governor's derived values, not as the primary interface.
- Existing explicit model/transcoder/provider selection remains outside the
  budget surface. The governor receives the resolved provider profile and plans
  from capabilities; it does not infer policy from repo names or model family
  strings.

## 8. Telemetry contract

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
   amplification question. Used as validation evidence, not as a per-run
   gate (knob choices must never depend on runtime measurements in ways
   that could differ across reruns);
6. the Phase-0 sanity gate: `active_features ≈ tokens x layers x trained
   L0`, as a hard screaming check — the structural fix for the class of
   error that let the wrong-hook contamination shape conclusions for weeks.
7. provider identity and capabilities: architecture/topology, hook map,
   checkpoint/provider fingerprint, declared/measured dimensions and bytes,
   selected compatibility fallbacks, and any missing capability that forced a
   slower rung.

## 9. Relation to the module-split restructure

From the scout report's deepening candidates:

- The governor **is** candidate #2 (exact-trace policy/config seam): policy
  resolution becomes one deep module, testable without running attribution.
- The attribution mega-module split (candidate #1) reshapes phases into
  governor *consumers*: each phase declares working sets and requests
  grants, instead of reading a flat knob bag.
- Row-store/replay locality (candidate #3) is where the ladder mechanism
  for the row store lands.
- The provider contract from the PLT parity work
  (`docs/plt_clt_optimization_parity_spec.md`) already made capabilities
  architecture-neutral; the governor consumes `TranscoderCapabilities`, a
  provider runtime profile, and per-provider cost formulas rather than CLT/PLT,
  Gemma/GPT/Llama, or checkpoint-name special cases.

Implementation rule: the governor package must be unit-testable with synthetic
providers that cover at least cross-layer, same-layer, and top-k/approximate
provider semantics. If a new supported model/transcoder pair requires editing
governor branching instead of supplying provider metadata or a new mechanism
rung, the abstraction has failed.

Public `attribute(...)` remains a compatibility facade during migration.

## 10. Execution plan

Ordered; each step is useful even if later steps slip.

### Step 1 — Merge PLT parity + hook fix into main (first, before anything)

- Merge `feature/plt-optimization-parity-harness` (project) and
  `feature/plt-optimization-parity` (sibling) into their `main` branches,
  including the currently-dirty harness files in the PLT worktree.
- Apply the same `mlp.hook_in` input-hook fix to the CLT preset path —
  main's CLT path is still wrong-hook contaminated as of 2026-07-03.
- Re-establish corrected-hook CLT baselines; do not reuse any
  `hook_resid_mid`-era GemmaScope2 CLT artifacts as references.

### Step 2 — Sensitivity + calibration probe campaign (1B/4B, before rewrite)

One campaign, two axes, on corrected hooks:

- **FP-sensitivity axis:** per (model, prefix size):
  `decoder_chunk_size` sweep {2048, 4096, 8192} with everything else
  pinned, plus a same-config repeat run as the determinism control; diff
  compact outputs; record selection-margin telemetry so results explain
  *why*, not just *whether*, outputs moved. This resolves the
  `decoder_chunk_size` caste — the biggest open design fork.
- **Cost-model axis:** the same runs across a spread of prefix lengths
  yield the nnz-vs-tokens curve, unit decoder-chunk/encoder-row timings,
  and rigid/elastic memory curves — the empirical constants for admission
  estimates, measured clean instead of inherited from contaminated
  telemetry. Store these as provider-profile calibration data, not as
  GemmaScope2-only constants.

### Step 3 — Taxonomy pass (login-safe)

Extend `docs/knob_api_taxonomy.md`: for every knob — tier, bytes-cost
formula, caste, validated-under provenance, and whether it is provider-declared,
scenario-declared, or governor-derived. This is the requirements doc for the
governor and the guardrail that keeps provider semantics knobs out of memory
policy.

### Step 4 — Governor v0 as a pure resolver, project-side

A pure function (model config, provider profile/capabilities, scenario,
hardware) → existing knob values, living next to `transcoder_config.py`. No
library changes. The hand-tuned 1B/4B/12B presets become test fixtures: the
resolver must reproduce them within tolerance, and must beat them where the
probe data says they were too conservative. Add synthetic provider fixtures so
the resolver is tested without Gemma/GemmaScope-specific branches.

### Step 5 — Library restructure with the governor as a deep module

The module split per the scout report, with the ledger, epoch model, provider
runtime profile, and ladder mechanisms (including the tiled row store) landing
behind the new seams. Existing login-safe tests listed in the scout report are
the safety rails; parity runs on canonical prompts guard exact outputs, and
provider-contract tests guard non-Gemma/top-k extensibility.

### Step 6 — Dynamic post-Phase-0 spending

Enable epoch-3 re-planning, gated per knob on parity proofs from step 2
(and equivalent proofs for any knob added later).

## 11. Acceptance criteria

1. A run can be launched with only budget-level inputs; all per-mechanism
   values are derived, logged, and overridable.
2. The governor reproduces (or justifiedly improves on) the hand-tuned
   stress presets for 1B/4B/12B.
3. No governor-controlled knob can change compact outputs — enforced by
   caste tests with validated-under provenance, not by convention.
4. Admission produces a plan or an actionable refusal; no memory-caused
   mid-run deaths in the validation matrix; walltime projections reported.
5. Telemetry reports predicted-vs-actual per tier with rigid/elastic split,
   ladder decisions, selection margins, and the Phase-0 L0 sanity gate.
6. The row store completes (slower) at problem sizes where full
   materialization would exceed node NVMe capacity, via tiled/recompute
   rungs spilling toward scratch.
7. Any model/transcoder pair supported by the exact/chunked provider contract
   can obtain a governor plan from provider metadata. Unsupported/missing
   provider capabilities are explicit in the plan and telemetry; they are not
   hidden Gemma/CLT/PLT special cases.

## 12. Open questions

1. `decoder_chunk_size` caste in the corrected regime (resolved by step 2).
2. Whether PLT selection at ~10:1 binding ratio shows meaningful
   FP-sensitivity, and whether tolerance-aware tie-breaking (Track-A
   follow-up) should land with the governor or stay separate.
3. Exact shape of the walltime estimator (per-phase throughput models need
   step-2 data; Phase 3 on 12B may be compute-bound regardless of budgets).
4. Whether the recompute rung of the row-store ladder is ever cheaper than
   tiled streaming in practice on Lustre, or is kept only as the
   survivability floor.
5. How much of the ledger/epoch machinery belongs library-side vs
   harness-side for non-harness library users.
