# Exact-trace large-model optimization plan

Status: proposed; no GPU work launched by this plan

Date: 2026-07-27

Branch: `perf/exact-trace-loop` in both project and sibling worktrees

Scope: Gemma 3 4B/12B PLT as the development and stress cases, with mechanisms
implemented against provider capabilities rather than model names

This succeeds `plans/2026-07-26_optimizations.md`. The 1B path is frozen as a
regression gate; it is no longer an optimization target. The aim is to use 4B
for relatively cheap iteration and 12B to expose storage, cache, and execution
regimes that lighter models hide, while landing mechanisms that can transfer to
other model sizes and exact-chunked providers.

This is an engineering plan. It does not promote governor observations,
response-model bundles, launch defaults, or the frozen baseline registry.

## Starting evidence

| Case | Current result | Interpretation |
|---|---:|---|
| 1B active rows, b256/c4096 | 84.82s completion; exact pass | frozen regression gate only |
| 4B active rows, b128/c4096 | 335.61s first transfer; 223-256s warm pairs; 27,685 MiB framebuffer | bounded engineering incumbent; not historical exact parity |
| 4B active rows, b128/c65536 | 228.43s paired mean vs 239.36s c4096 | dataset-frozen bounded regime; 4.6% gain is below promotion margin |
| 4B active rows, b512/c4096 | 327.32s vs paired b128 255.72s; 82,111 MiB framebuffer | rejected: 28% slower and about 3x framebuffer |
| 12B active rows, b64/c4096 | Phase 0 1,350.83s; first Phase-4 batches 187.85s/120.29s | capacity-viable at 250 GiB, throughput-inviable; no completed compact artifact |

The 12B run issued 3,030 decoder loads for 95.32 GB of logical decoder data.
Its attribution kernels took only 0.20-0.22s in the first measured Phase-4
batches, while the batch wall times were two to three orders of magnitude
larger. This makes storage and working-set lifecycle the first target, not more
GPU arithmetic tuning.

The previous contiguous safetensor range experiment must not be repeated
unchanged. The active feature rows are nearly random: narrow coalescing produces
almost one range per row, while wider coalescing approaches full-checkpoint
overfetch. A useful successor needs a different physical representation or a
block-indexed I/O owner, not another range threshold.

## Generality rules

1. Branch on `TranscoderCapabilities`, decoder topology, row shape, byte cost,
   and resource envelope. Do not branch on `4B`, `12B`, Gemma checkpoint names,
   or scenario IDs in the sibling runtime.
2. Keep model/provider, prompt, `max_feature_nodes=8192`, semantic batching,
   refresh checkpoints, retention, dtype, and reconstruction order frozen
   within a comparison.
3. Treat `decoder_chunk_size` as compatibility-mixed. A larger chunk is a
   separate bounded numerical regime, never a compact-strict physical
   optimization.
4. Preserve two comparison scopes:
   - same-regime compact-strict parity against the active-row c4096 control; and
   - bounded compact parity against the frozen historical artifact.
   Passing the first does not retroactively make active-row 4B historical-exact.
5. Preserve exact decoder values and occurrence order. Sorting and deduplicating
   physical reads is allowed only with an explicit inverse permutation before
   reconstruction and accumulation.
6. Report offline preparation, startup, Phase 0, Phase 3, Phase 4, completion,
   and subprocess time separately.
7. Treat a 250 GiB job as a deliberate cache-pressure engineering envelope.
   Candidate and control must use the same envelope. Final promotion evidence
   either repeats the canonical 4B/12B resource contract or deliberately freezes
   a new resource-normalized baseline.

### Parity and runtime gates

Both policies require exact target-token identity, aligned completions/steps,
and the worst aligned step to pass:

| Policy | Feature Jaccard | All-edge Jaccard | Top-256 | Weighted | Magnitude L1 |
|---|---:|---:|---:|---:|---:|
| compact-strict (`--fidelity exact`) | 1.0 | 1.0 | 1.0 | >=0.999999 | <=0.000001 |
| bounded | >=0.98 | >=0.98 | >=0.98 | >=0.98 | <=0.02 |

Compact artifacts store edge magnitudes, so neither gate establishes internal
edge-sign parity. The CLI retains the historical `exact` policy name, but this
plan calls that result `compact-strict`. A full exact-mechanism claim additionally
requires byte-exact provider values/order, identical semantic fingerprints, and
signed/raw intermediate evidence where the mechanism can affect them. For new
physical mechanisms, compact-strict is first evaluated against the same
active-row c4096 regime; bounded is also evaluated against the frozen historical
artifact.

Keep 600s as the final large-model working objective. Existing 4B results meet
it, but Stage A must add an explicit 4B gate rather than inherit the current 1B
gate. For 12B it remains a target, not an initial promotion gate: first obtain
a completed b64/c4096 artifact and fit a same-envelope control before setting a
hard runtime threshold that could otherwise be arbitrary.

## Stage A — capability-based profiles and evidence contracts

This is the remaining useful part of original work item I and is a prerequisite
for calling any mechanism reusable.

Replace model-string profile routing with a typed candidate contract:

```text
CandidateProfile
├── requires: provider capabilities and decoder topology
├── baseline_pins: semantic and compatibility-mixed values
├── physical: storage, placement, batching, and lifecycle controls
└── evidence_scope: model/provider/prompt/resource/parity requirements
```

Required enforcement:

- a profile applies when the loaded provider satisfies `requires`;
- exact mode refuses a baseline-pin mismatch, especially chunk size;
- topology-specific decoder controls cannot leak into unrelated providers;
- a mixed 4B/12B suite may use a profile only when both cases independently
  satisfy it;
- the run fails, rather than merely records metrics, when its declared
  per-model parity or runtime gate is missed;
- 4B/12B thresholds are explicit and are not inherited from the 1B 600s gate.
- same-regime active-row controls are registered separately as mechanism
  baselines and never overwrite the frozen scientific baseline.

The project owns profile/gate/provenance policy. The sibling owns executable
provider capabilities. Keep these contracts aligned without turning model names
into runtime capabilities.

**Gate A:** login-safe tests show capability selection, baseline-pin refusal,
mixed-suite behavior, and separate exact/bounded comparison scopes for synthetic
same-layer and cross-layer providers.

## Stage B — phase-scoped I/O and working-set attribution

Before changing cache policy, make the 12B stall directly attributable. Current
telemetry records decoder counters and elapsed time but cannot distinguish file
I/O, page-cache refault, model/offload reload, encoder loading, and GPU work.

Add diagnostic-only phase snapshots:

- process minor/major fault deltas;
- `/proc/self/smaps_rollup` anonymous/file-backed totals where available;
- versioned cgroup memory fields:
  - v1: `total_rss`, `total_cache`, `pgfault`, `pgmajfault`, `pgpgin`,
    `pgpgout`, and `memory.failcnt`;
  - v2: `anon`, `file`, `workingset_*`, and available page-scan/reclaim
    counters;
- process read-byte/syscall deltas;
- decoder and encoder load counts, logical bytes, physical/requested bytes, and
  load time;
- model offload/reload state and time;
- CUDA event time for the actual attribution kernels;
- cold/warm/cache-pressure provenance for every timing.

Add two diagnostic stop points:

- `phase0_probe`, immediately after active-row materialization and cleanup, for
  rejecting row-loading candidates without spending a full trace;
- `transition_probe`, after Phase 0 and a declared first-N Phase-4 batches, for
  observing the failing page-lifecycle transition.

Neither is a scientific trace result. Both terminate through a runner-owned
`probe_completed` outcome, finalize telemetry, and release the Slurm step
without calling the trace complete or failed. Add that outcome end to end:
runtime result/status, terminal observability event and sink close, runner/result
JSON, extraction/performance report, and explicit exclusion from compact
parity, baseline, and calibration ingestion. Tests must prove that a probe
cannot remain `running`, be mapped to unknown/infrastructure failure, or enter
ordinary success handling.

**Gate B:** one immutable 4B control and one 12B transition probe identify
asset-correlated file-backed working sets that are reclaimed and refaulted
across the Phase-0-to-Phase-4 transition. Record backend-owned requested and
materialized bytes separately from process/cgroup OS-read estimates; the latter
are attributable only inside asset-tagged intervals under the job-private cold
protocol. Do not implement a generic "model prefault" until this evidence shows
which owned mapping or offload handle is responsible. Instrumentation overhead
must remain below 2% on a paired 4B no-policy-change control. Record unavailable
cgroup fields explicitly; Gate B fails rather than silently attributing page
behavior when the required version-specific counters are absent.

## Stage C — checkpoint page and working-set lifecycle

The runtime has no typed owner for checkpoint-backed pages. Add a sibling-owned
`CheckpointPageLifecycle` beside provider loading, with asset roles such as
decoder, encoder, model-forward, and refresh. Providers expose immutable
file/range manifests; the runtime owns phase transitions.

```text
Phase 0 scan
  -> active decoder rows sealed and coverage-checked
  -> decoder handles/caches released
  -> owned decoder pages advised DONTNEED
  -> identified next-phase assets prefaulted if admitted
  -> Phase 4
  -> cleanup
```

Required behavior:

1. close/release sidecar and safetensor mappings after the active decoder rows
   pass admission;
2. clear owned decoder caches;
3. issue targeted `madvise`/`posix_fadvise(DONTNEED)` only for immutable,
   non-overlapping owned decoder byte ranges when the backend supports it;
   file-wide advice is refused for mixed-role files, and
   `posix_fadvise(DONTNEED)` is limited to job-private staged files or an
   explicitly exclusive node so it cannot evict a shared model cache used by
   other jobs;
4. never use global `drop_caches`;
5. record page-release support, requested bytes, observed cache delta, and
   refusal/failure without weakening correctness;
6. retain or prefault model/offload mappings only when Stage B identified them
   and memory admission says the working set fits;
7. make release/prefault synchronous, byte-budgeted, prioritized from cgroup
   headroom, idempotent, and cleanup-safe.

Record device, inode, byte range, asset role, shared/private scope, and effective
advice support. If exclusive range ownership cannot be proven, close the owned
mapping and use the no-advice fallback.

Represent the policy in bytes and capabilities:

```text
PhaseWorkingSetPlan
├── retain: active decoder rows, selected encoder placement, required model state
├── release: raw decoder files and obsolete caches
├── prefault: identified next-phase mappings, if admitted
└── fallback: no advisory/release support
```

Eviction and prefault are measured hints, not assumed effects. Never evict
model pages indiscriminately.

**Gate C:** cold/warm 4B b128/c4096 pairs remain within 3% of the incumbent.
For the declared first three 12B Phase-4 batches, define disparity as
`sum(batch_wall_seconds) / max(sum(cuda_kernel_seconds), epsilon)`. Continue
only if lifecycle policy reduces both this ratio by at least 50% and summed
batch wall time by at least 30%, with asset-tagged fault/refill telemetry moving
consistently, zero fail-count delta, and rigid RSS/HBM within the envelope. If
the identified refault source is not decoder or model/provider-owned storage,
stop and update the diagnosis rather than adding broader cache controls.

## Stage D — provider-owned selective decoder-row source

Phase 0 will remain a large target after lifecycle repair. Add a reusable
selective-row interface in the transcoder/provider layer:

```text
DecoderRowSource
├── estimate(keys) -> requested/physical bytes and request/block count
├── materialize(keys, destination, order) -> DecoderRowSeed
├── release(reason) -> lifecycle/resource report
└── fingerprint -> checkpoint/layout/value identity
```

Backends, in order:

1. current chunked safetensor scan, retained as the exact fallback;
2. one long-lived mapping/handle per tensor or layer during Phase 0, with
   sorted page-aware vectorized gathers, bounded CPU/pinned staging, and an
   inverse permutation to restore caller order;
3. an immutable block-indexed decoder sidecar only if telemetry shows that the
   existing row-major mapping still causes unacceptable syscall, page, or
   layout amplification.

Backend 2 must not create one mmap or explicit read per row. It should let the
OS fault selected pages through a stable mapping or issue coalesced/vectorized
block requests. It must refuse when its planned request shape degenerates to a
per-row operation.

If needed, the sidecar should:

- store original decoder values in their original dtype without quantization;
- use provider/topology metadata rather than PLT-specific tensor assumptions;
- group a small, fixed number of adjacent rows into aligned blocks;
- include checkpoint-content fingerprint, tensor metadata, dtype, shape, block
  geometry, and sidecar-content digest;
- live in external model/scratch cache, never in Git;
- be buildable in `setup_prefetch` independently from a trace;
- support batched/coalesced block reads;
- refuse on any fingerprint, dtype, topology, coverage, or byte-budget mismatch
  and fall back to the current path.

Block geometry is derived from row bytes, request count, measured
device/filesystem behavior, and overfetch. It must not be a hard-coded 4B/12B
constant. A login-safe planner can evaluate candidate geometries from the row
index without reading tensor data.

Required telemetry:

- unique and occurrence row counts;
- requested row bytes;
- backend-requested/materialized bytes and planned overfetch ratio;
- OS physical-read estimate only under the attributable cold local-file
  protocol;
- mapping/block/range/read counts;
- planning, fault/read, gather, reorder, H2D, and total Phase-0 time;
- sidecar build time and size, reported separately and amortized only when the
  campaign legitimately reuses it;
- fallback/refusal reason.

**Gate D1 — synthetic:** same-layer and cross-layer fixtures reproduce decoder
rows bit-for-bit and preserve reconstruction/accumulation order, including
duplicates and non-monotonic keys.

**Gate D2 — 4B:** paired immutable b128/c4096 controls show same-regime
compact-strict parity, historical bounded parity, byte-exact row values/order,
and no Phase-4 decoder loads. Repeat a winner twice.

**Gate D3 — 12B Phase-0 probe:** the minimum continuation gate is at least 50%
less backend-materialized decoder traffic and 30% less Phase-0 time under the
same cache-pressure envelope. When the cold local-file protocol makes OS reads
attributable, require the physical-read estimate to move consistently. The
promotion target is backend-materialized bytes at most 20% of the 95.32 GB
fallback traffic and at least a 2x Phase-0 improvement. Otherwise revise or
reject the backend before a full 12B run.

## Stage E — complete the 12B reference lane

Run b64/c4096 with the accepted byte-exact/compact-strict physical mechanisms and
unchanged semantic settings. Acceptance requires:

- a completed compact artifact and aligned completions/steps;
- exact target token;
- no fallback or post-Phase-0 decoder loads;
- the bounded compact gate at minimum;
- an explicit comparison against the frozen strict artifact before any exact
  claim;
- one completion before further optimization and a repeat before any
  profile/default promotion.

## Stage F — encoder placement

Once decoder and model-page churn is controlled, evaluate the existing
provider-generic encoder choices under one frozen execution plan:

- lazy per request;
- active CPU;
- active pinned CPU.

Use the existing exact row materialization API and byte admission. Add an
active-GPU mode only if profiling shows repeated encoder H2D/materialization is
at least 10% of Phase 4 and the full active encoder set fits with audited HBM
headroom. Do not add it speculatively.

**Gate F:** compact-strict same-regime parity, bounded historical parity, an
unchanged semantic fingerprint, a complete owned-byte account, and either at
least 10% Phase-4 improvement or a clear elimination of checkpoint loads
needed by the following 12B execution rung. Otherwise retain the simpler
placement.

## Stage G — physical Phase-4 execution envelopes

Only after the I/O/working-set gates pass:

1. use the completed Stage-E b64/c4096 result as the control;
2. test physical b128 and b256 at c4096 while keeping semantic batches,
   prepared frontier membership/order, and refresh checkpoints frozen;
3. test 4B b256/c4096 only if it helps fit the transferable cost curve;
4. do not repeat 4B b512;
5. do not combine a larger execution envelope with c65536 until each axis has
   independent evidence.

Split evaluation into two scopes:

1. an executor microgate using captured/prepared frontier inputs (or an
   equivalent replay fixture) so membership/order is frozen and the physical
   packing/commit path can be isolated;
2. an endogenous full run where numerical differences may change later
   frontier membership and that drift is measured explicitly.

Execution coalescing is physical only inside the microgate when it remains
within one prepared refresh frontier and commits rows in canonical order. The
endogenous run is a bounded transfer result if any later frontier differs; it
must not be called exact physical isolation. Historical Wave-B b256 evidence is
a prior, not proof under active-row residency.

The governor should ultimately select an envelope from measured row count,
tensor shape, HBM, and time models. Do not promote a universal `12B=b256`
constant.

**Gate G:** the executor microgate requires identical prepared-frontier
membership/order and compact-strict output. The endogenous run requires a
completed compact artifact, explicit frontier-drift report, per-model bounded
gate, two same-state timing repeats for a winner, and a Pareto improvement in
completion time/framebuffer. A 12B timing without a completed artifact is
operational evidence only.

## Stage H — compute optimizations only if compute becomes visible

Two ideas from the original plan remain conditional:

- source-layer contraction fusion; and
- fewer/larger fp32 contractions.

They change reduction grouping and are bounded-only by construction. Profile
them only if, after Stages C-G, contraction/kernel launch time is at least
10-15% of end-to-end or Phase-4 time on 4B/12B. At the current 12B observation,
0.20-0.22s kernels inside 120-188s batches do not meet this gate.

Do not revive decoder cache, VJP tape, gather-before-cast, frontier-512,
prepared-refresh cache, or decoder prefetch unless new telemetry shows that
their removed bottleneck has returned. Active-row residency already removed the
decoder reuse that justified them.

## Ownership and commit boundaries

| Stage | Primary ownership | Likely seams |
|---|---|---|
| A | project harness | `exact_trace_bench/perf_cli.py`, baseline gates, typed scenario/profile boundary |
| B | project reporting + sibling observability | `trace_runtime/support.py`, phase/resource telemetry, diagnostic runner outcomes |
| C | sibling provider/loading lifecycle | provider asset manifests, loader/file ownership, new checkpoint-page lifecycle module |
| D | sibling transcoder/provider | `transcoder/provider.py`, loaders, single/cross-layer providers, `DecoderRowSeed` handoff |
| E | project immutable campaign | 12B b64/c4096 suite, snapshot, comparison, and report |
| F/G | sibling physical execution + project profiles | encoder materialization and `phase4_batches.py` physical packing |
| H | sibling contraction path | active-row contraction only after the compute-share gate |

Commit in these boundaries rather than combining mechanism, telemetry, and
promotion decisions. After each accepted GPU gate, append the detailed run to
`experiments/logs/YYYY-MM.jsonl`; update `EXPERIMENTS.md` only for a
baseline-changing disposition or current interpretation.

## Experiment ladder

All GPU/model work is SLURM-only and uses one immutable read-only snapshot of
both repositories per source state.

1. login-safe synthetic/provider/profile tests;
2. 1B PLT and CLT smoke only as regression checks;
3. paired 4B b128/c4096 control/candidate, with cold/warm provenance;
4. 12B Phase-0 control/candidate probes under the same resource envelope;
5. completed 12B b64/c4096 candidate with compact parity;
6. isolated 12B physical execution-envelope ladder;
7. a 1k-prefix 4B/12B transfer probe only after the 124-token cases are stable
   and admitted. It is a new workload scope, not confirmation of the short
   prefix result.

For a controlled cold comparison, stage checkpoint assets under a job-private
cache identity or use targeted advice only on job-private files. Warm repeats
reuse the same paths. Never claim a cold result from an unverified cache state
and never use a node-global cache drop.

Before each resource-mutating launch, present the exact project/sibling commits,
snapshot, model/provider, prompt, semantic and physical knobs, host/HBM/time
envelope, cache-state protocol, candidate/control order, output root, and stop
conditions for review.

Every candidate report must include:

| Category | Required fields |
|---|---|
| Provenance | both commits/dirty states, snapshot manifest, checkpoint and sidecar fingerprints, Slurm/node/GPU |
| Semantics | model/provider/prompt, feature cap, semantic batches, refresh checkpoints, chunk regime, dtype |
| Correctness | completed steps, token identity, feature/all-edge/Top-256/weighted Jaccard, normalized L1, semantic fingerprints, prepared-frontier hashes/drift |
| Time | sidecar build, startup, completion, attribution, Phase 0/3/4, per-batch and CUDA-event time |
| I/O | logical/requested/materialized bytes, attributable OS-read estimate, blocks/ranges/reads, overfetch, decoder/encoder/model load time |
| Memory | framebuffer, CUDA alloc/reserved, active row bytes, RSS/anon/file cache, cgroup peak/failcnt/refault |

## Stop/go and promotion rules

- Stop a mechanism after its phase-scoped gate misses; do not spend a full 12B
  trace to confirm an I/O candidate that failed the Phase-0 probe.
- Require a completed compact artifact before any 12B parity statement.
- `compact-strict` means the historical CLI `exact` thresholds passed against
  the declared same-regime comparator. It does not establish signed internal
  edge parity or full exact semantics without the additional raw/value/order and
  semantic-fingerprint evidence above.
- Larger chunks always remain an independently named bounded regime.
- Promote a mechanism separately from a fitted plan, governor bundle, or launch
  default.
- Keep multi-GPU out of scope until Phase 4 sustains more than about 50% SM
  utilization or the admitted working set genuinely exceeds one H200.

## Disposition of the original plan

| Original item | Successor disposition |
|---|---|
| A/B active rows and plumbing | retained incumbent and prerequisite |
| B2 chunk-size confound | converted into baseline-pin enforcement |
| C source-layer fusion | Stage H, conditional bounded-only |
| D cost-model telemetry | complete; extended with Stage B page/I/O ownership |
| E CLT Phase-1 cap | closed; 1B regression only |
| F timing separation | retained as required reporting |
| G FP32/TF32 pin | retained unchanged |
| H feature cap | frozen at 8,192; not an optimization lever |
| I provider enforcement | Stage A prerequisite |
| J coalesced safetensor ranges | rejected unchanged; replaced by Stage D selective row source |
| Cache/tape/prefetch variants | do not revive without a measured bottleneck return |
| Multi-GPU | remains deferred |

The intended execution order is therefore:

```text
typed evidence contract
  -> page/I/O attribution
  -> checkpoint page lifecycle
  -> selective decoder-row source
  -> complete 12B b64/c4096
  -> encoder placement
  -> physical execution envelopes
  -> conditional compute fusion
```
