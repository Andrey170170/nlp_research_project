# Exact-trace performance review and optimization directions

Date: 2026-08-04

Updated: 2026-08-07

Status: completed LS4 4B/512 evaluation

Accepted comparison artifacts:

- job `1714134` — valid preheated `cpu_exact` control artifact, followed by a
  launch-wrapper validation failure before the candidate started;
- job `1721614` — completed preheated, fail-closed `cuda_windowed` candidate;
  and
- canonical comparison
  `runs/512/comparisons/cpu-exact-preheated-v1-vs-cuda-windowed-preheated-v2.json`.

This report separates measured facts, current inferences, and future
experiments. The result changes the measured 4B development-length
interpretation, but it does not change a global default or scientific baseline.

## Executive readout

The current optimization direction is technically coherent and is now
non-dominated on the frozen 4B/512 development workload. The fail-closed
candidate resolved to `cuda_windowed`, completed without fallback, and matched
the control compact graph exactly: all 8,192 features and all 20,000 edges,
including weights, signs, target token, and every Top-64 through Top-1,024
view. Its canonical fidelity classification is `strict_exact`.

Trace wall fell from 346.773 to 168.067 seconds: a 51.5% reduction and 2.06x
speedup. Phase 4 fell from 276.521 to 121.596 seconds, accounting for 86.7% of
the total saved trace time. Refresh fell from 201.158 to 53.521 seconds
(73.4%, 3.76x), while feature-batch work improved only 9.6%. This is strong
mechanism evidence for bounded CUDA-windowed refresh, not merely a warm-start
effect.

The observed cgroup totals are not a valid memory comparison. The control's
first preheat read 178.11 GiB at 599 MiB/s and charged about 191.5 GiB of file
cache to its cgroup. The candidate ran later on the same node, read the same
set at 6,366 MiB/s from an already warm shared cache, and was charged only
about 13.1 GiB of file pages. The candidate's anonymous maximum was higher
(21.1 versus 9.2 GiB), consistent with its 11.92 GiB host mirror, and both
runs had essentially the same CUDA peak. Therefore this result supports speed,
fidelity, and mechanism admission; it does not support a claim that
`cuda_windowed` reduced total host memory.

The recommended order is now:

1. retain `cuda_windowed` as a selectable non-dominated 4B mode without making
   it the arbitrary-prompt or global default;
2. improve resource/cache/fingerprint observability so later measurements are
   easier to interpret;
3. make benchmark mode requirements fail closed in the runtime itself, not
   only in the post-run wrapper;
4. profile and prototype an exact normalization-state optimization, because
   normalization is now 32.94 seconds or 61.6% of candidate refresh; and
5. take only a transition probe to the frozen 4B/1,024 development gate before
   deciding whether another complete pair is justified.

The strongest genuinely different storage direction is not another file-only
window. It is to make the pageable signed host tensor the canonical exact store
for an admitted run, use it for both CPU fallback and bounded CUDA windows, and
avoid writing the same complete row matrix into a second file-backed store.

## Review scope and provenance

The accepted control and candidate use separate read-only snapshots containing
the same runtime revisions:

- project commit `323554f8dd97c47569437648256197e3c4b97d98`;
- sibling commit `aec77af4660f5dae0cae134cd7fc177e8c181681`;
- frozen workload `ls4-4b-361-dev-512`;
- 512-token prefix hash
  `8a01ec6a0f7fe0f91a1ebf8c489f258af140086aac7db35eccb2e0016091b78b`;
- target token ID `33036` at absolute position 512; and
- one H200, 32 CPUs, 400 GiB host RAM, and two hours per trace.

The estimate-based prepared-workload stop guard is intentionally not active for
these optimization runs. The scheduler allocation limit remains active. This
is an execution-policy choice, not evidence that the frozen 200 GiB campaign
envelope has passed.

The code review centers on:

- project campaign preparation and full-answer request plumbing in
  `src/nlp_research_project/exact_trace_bench/`;
- sibling row ownership and CUDA-window execution in
  `circuit_tracer/attribution/nnsight/row_store.py`;
- Phase-4 refresh/read behavior in
  `circuit_tracer/attribution/nnsight/phases/phase4_influence.py`;
- selective provider materialization and checkpoint lifecycle; and
- the current telemetry, resource, and execution-fingerprint contracts.

## Evidence leading into the full result

### Accepted evidence

The 1B development curve measured the corrected CUDA-windowed implementation at
129, 256, 512, and 1,024 tokens. It improved runner wall at every rung, from
7.7% at 129 tokens to 48.0% at 1,024 tokens, while meeting `exact` or
`strict_exact` on each measured development point. This establishes that the
mechanism can scale usefully; it does not establish 4B transfer.

At 4B/129, the matched pair was `strict_exact`. Runner wall fell from 125.835
to 98.511 seconds, Phase 4 from 83.426 to 58.267 seconds, and refresh from
40.951 to 15.391 seconds.

At 4B/256, the matched pair met canonical `exact`. Runner wall fell from
204.179 to 129.403 seconds, Phase 4 from 162.783 to 84.898 seconds, and refresh
from 114.343 to 34.264 seconds. Feature-batch work was nearly flat, so the main
gain came from refresh row consumption rather than attribution compute.

The 4B/512 candidate transition probe completed successfully:

| Probe signal | Observed value |
|---|---:|
| Trace wall | 51.849 s |
| Phase 0 | 32.331 s |
| Four Phase-4 executions | 4.654 s |
| Active decoder rows | 390,468 |
| Active decoder bytes | 1,999,196,160 |
| Active decoder cap | 4,294,967,296 |
| Resolved row-influence mode | `cuda_windowed` |
| Logical host mirror | 12,796,417,296 bytes |
| CUDA window | 534,160,224 bytes |
| Window geometry | 171 rows x 2 buffers |
| Peak CUDA allocated | 61.21 GiB |
| Peak CUDA reserved | 63.08 GiB |
| Observed cgroup peak | 13.90 GiB |
| Telemetry events / drops / sink errors | 494 / 0 / 0 |

The probe used selective mapped Phase-0 decoder rows without fallback. It
materialized about 1.19 GB through the mapped-safetensors backend instead of
the reported 45.63 GB full-page baseline for that operation.

### Evidence that must not be overread

The 13.90 GiB probe cgroup peak is not a full-trace host peak. Only 512 of
8,192 rows were committed. The 11.92 GiB binary host-mirror size reported by
the provider is logical ownership; `torch.empty` does not prove that every
page has been physically charged before later appends.

Checkpoint cache state was recorded as unavailable. All 34 release requests,
covering 85 GiB of owned decoder ranges, were safely refused because the assets
were declared shared. Already-resident shared pages may remain charged to a
different cgroup, so the low observed charge is not a reproducible cold-cache
measurement.

The probe produced no graph artifact and did not establish fidelity or complete
runtime by itself. Those gaps are now closed by the full candidate. It issued
2,652 window reads covering 281.57 GB, completed with no fallback or copy
failure, and saved a compact graph with complete telemetry.

The accepted control and candidate are separate per-trace jobs. They are
matched in workload, code revisions, requested resources, GPU class, and exact
preheat file set, and both ran on `grn031`; they are not matched in cgroup page-
cache ownership. The 51.5% end-to-end margin and its Phase-4 concentration are
far larger than the remaining allocation-order uncertainty, but resource
charges must be reported by anonymous/file category rather than as a total-
memory winner.

## Current code-quality findings

### 1. The mechanism boundaries are mostly sound

The current profile composes independently meaningful mechanisms rather than a
single opaque fast path:

- selective decoder-row materialization owns Phase-0 checkpoint traffic;
- active decoder-row residency owns the post-Phase-0 decoder lifetime;
- physical batch splitting owns the HBM execution envelope;
- `cuda_windowed` owns Phase-4 feature-row consumption; and
- `cpu_exact` remains the stronger atomic row-influence fallback.

Requested/effective fields and detailed row-tier counters make silent provider
mis-resolution detectable. This already caught the earlier full-answer bridge
bug and is a strong part of the design.

### 2. Host and CUDA admission are asymmetric

`_GpuResidentFeatureRowStore` checks CUDA capacity, configured window bytes,
free HBM, and the HBM safety margin before admitting the CUDA window. It then
allocates the complete pageable host mirror without an equivalent host-headroom
decision. Because the allocation is lazy at the OS level, a short probe can
report logical ownership before the final anonymous footprint is committed.

This is acceptable for the current explicitly high-memory experiment, but the
provider's `admitted` status should not be interpreted as proof of complete
host admission. A future admission contract should distinguish:

- logical address-space allocation;
- configured host budget admission;
- committed or prefaulted host pages; and
- observed cgroup charge during execution.

Do not add a hard-coded model/length threshold. Host admission belongs in the
typed runtime/resource plan and must retain an explicit experimental override.

### 3. CUDA-windowed mode deliberately owns two canonical-sized stores

Every append first updates `_FileBackedFeatureRowStore`, then copies the same
signed rows into `_host_rows`. This gives a robust file-backed reference and
fallback, but it can charge both row-file cache and pageable anonymous memory.
At 4B/512 the two complete logical copies are each about 11.92 GiB before
temporary staging and metadata.

The rejected file-window experiments showed that deleting the host copy while
keeping file-backed refresh is the wrong direction: mapped reads were much
slower without a meaningful total-memory win, and direct reads became
storage-bound. That result does not prove that dual ownership is necessary; it
only rejects file-only consumption as implemented.

### 4. Resource policy is represented as one envelope but used in three ways

The frozen campaign contains a 200 GiB host stop, 64 GiB RSS stop, 90% HBM
fraction, and four-hour wall limit. The actual full jobs request 400 GiB and two
hours, and intentionally bypass the stop wrapper. These are legitimate
optimization-run choices, but the current schema makes a fitted/estimated
envelope, an allocation request, and a runtime abort policy look like one
contract.

Future preparation should record them separately:

- `planning_envelope`: prediction or previously demonstrated scope;
- `allocation_request`: actual Slurm GPU/CPU/RAM/wall request; and
- `runtime_resource_policy`: `off`, `measure_only`, or `enforce`, with any
  thresholds and override rationale.

`measure_only` is particularly important: optimization runs should be able to
disable estimate-based termination without losing periodic cgroup/RSS/HBM
telemetry and a final resource summary.

### 5. The prepared launch interface makes policy bypass implicit

`prepare-campaign-workload` records and prints the raw
`run-full-answer-shard` command. A separate command wraps that raw command with
resource monitoring. This makes it easy for two operationally different launch
paths to appear to execute the same prepared contract.

Preparation should emit one canonical launch specification containing the raw
runner plus an explicit resource-policy field. Rendering a local command or an
`sbatch` command should consume that specification. Bypass should be an
auditable choice, not a consequence of copying the command printed most
prominently.

### 6. Effective-plan changes need a delta, not only a new fingerprint

The 512 probe began with requested execution fingerprint `e41e...` and finished
with effective fingerprint `8388...`. Both values are persisted, but the
compact result does not provide a concise machine-readable list of changed
fields and reasons. Dynamic resolution is expected; unexplained resolution is
hard to audit.

Persist an `execution_resolution` record with, for each changed field:

- requested value;
- effective value;
- resolution stage;
- reason or observation ID; and
- whether the change affects semantics, fidelity scope, or only physical
  execution.

### 7. Large implementation modules are tolerable now but should not absorb the
next design

The current row-store class owns mode selection, admission, canonical backing,
host/GPU buffers, asynchronous transfer, fallback, lifecycle, and a large
string-keyed telemetry dictionary. These concerns are related enough that an
aesthetic refactor is not justified during the active campaign. Adding another
substantial storage mode directly to the same class would, however, make
invariants and fallback behavior harder to review.

If the single-owner direction is opened, introduce a typed canonical-row owner
and separate influence consumers instead of adding another branch to
`_GpuResidentFeatureRowStore`. Preserve one top-level Phase-2 storage decision
and avoid parallel legacy/new runtimes.

## Ranked optimization recommendations

### P0 — completed: retain the measured 4B/512 candidate

The full readout is recorded below. `cuda_windowed` is `strict_exact`, 2.06x
faster in trace wall, fail-closed, free of late fallback, and non-dominated on
this frozen development point. Retain the profile unchanged as a selectable 4B
mode and admit the next frozen development transition probe. Do not promote it
to the arbitrary-prompt or global default from one prompt family.

### P1 — make resource observation independent of guard enforcement

Implement `off | measure_only | enforce` as a typed launch/runtime policy.
Use `measure_only` for active optimization unless a known safety boundary
requires termination. Always persist effective Slurm resources and final
resource telemetry next to the trace.

This work is worthwhile regardless of the 512 outcome because it improves the
interpretability of every subsequent 4B/12B campaign without changing tracing
semantics.

### P2 — make mechanism-under-test overrides fail closed

Add a typed feature-row influence requirement alongside the requested mode,
with `preferred` preserving today's resilient fallback behavior and `required`
making the requested concrete mechanism a hard execution constraint. Expose a
user-facing convenience such as
`--require-feature-row-influence-mode cuda_windowed`, but compile it into the
canonical trace request rather than implementing it as an unrecorded
environment-only switch.

For `required`, fail with a typed, structured error if admission resolves to a
different mode, an append/copy failure would disable acceleration, or a Phase-4
read would fall back to the CPU-backed path. Record the requirement in the
trace specification, execution fingerprint, requested-to-effective telemetry,
and failure artifact. Reject `auto + required` as ambiguous. Capacity preflight
may provide an earlier refusal, but enforcement at the real row-store admission
and use sites remains authoritative.

Cover admission refusal, CUDA allocation failure, late append failure, and
fallback-read injection in the sibling-library tests, plus harness propagation
and provenance tests here. Keep post-run effective-mode assertions as
defense-in-depth. Use `required` for mechanism-isolation benchmarks and retain
`preferred` as the default for ordinary completion-oriented runs.

### P3 — add explicit execution-resolution provenance

Emit requested-to-effective plan deltas and resolution reasons. Treat missing
resolution provenance as an artifact-quality defect, not a trace failure. This
is low-risk observability work and should precede more complex adaptive
governor behavior.

### P4 — prototype a single-owner canonical host store only if triggered

Trigger this work if the full candidate is otherwise useful but host anonymous
plus row-file charge is the binding constraint, or if row append/writeback is a
material part of Phase 4.

The 512 result does not yet trigger this work. It confirms about 11.94 GiB of
additional anonymous host residency, but unequal shared page-cache ownership
prevents a reliable measurement of the duplicate row-file charge. Establish
that charge under job-private cache ownership or explicit file-residency
telemetry before accepting the complexity of a new canonical owner.

Target design:

1. estimate and admit the complete pageable signed host matrix before Phase 4;
2. if admitted, make that host matrix the canonical exact row owner;
3. let `cpu_exact` consume the same host rows if CUDA-window admission or a
   later CUDA copy fails;
4. layer the existing bounded pinned/CUDA double buffer over that owner;
5. do not create a complete row file in this admitted mode;
6. if host admission fails before row production, atomically select the current
   file-backed `cpu_exact` path; and
7. keep Phase-5 graph construction independent of the temporary row owner's
   physical representation.

This attacks duplicate ownership without reintroducing storage-bound refresh.
It must first pass focused signed-row, denominator, partial-commit, cleanup,
fallback, and failure-injection tests. Then run a 4B/256 matched proof against
the existing artifact before any 512 measurement.

Do not silently fall back from a partially populated host-only store to an
empty file. Either the selected canonical owner remains valid for the run or
the run fails explicitly with preserved diagnostics.

### P5 — attack normalization and repeated refresh work

The 4B/256 candidate streamed 147.7 GB through the bounded window from a 6.19
GB logical host mirror. The 1B/1,024 candidate streamed 362.2 GB from an 18.46
GB mirror. This amplification is expected from repeated refreshes and explains
why a fast host-memory path matters.

The 512 candidate streamed 281.57 GB through the bounded window from a 12.80 GB
logical row store. Refresh still consumed 53.52 seconds, but its internal
profile changed materially: row-store reads were 18.38 seconds, exact
normalization was 32.94 seconds, and direct accumulation was 0.54 seconds.
Normalization is therefore the largest measured candidate-refresh component,
whereas the control was dominated by transfer/cast/absolute-value work.

First test whether exact normalized row weights can be computed once per solver
iteration in contiguous form, or whether denominator application can be fused
with the ordered direct-accumulation path without changing reduction order.
The denominator pair is invariant within a refresh; the changing input is the
row-weight vector. Avoid recomputing normalization separately for every active
subrange when a single ordered vector operation can feed the same subranges.
Instrument kernel synchronization before attributing all 32.94 seconds to
arithmetic, because the current wall timers may also absorb queued CUDA work.

The higher-risk alternative is exact incremental refresh state: update only
the influence contribution affected by newly visited frontier rows instead of
re-solving from the logit seed. That attacks both the 281.57 GB repeated reads
and normalization. It requires explicit ordering, convergence, denominator,
and signed-FP32 parity proofs and should be developed as a separate semantic-
risk experiment.

The old exact-range prepared LRU remains rejected. Any new reuse design must be
chunk-aligned/window-aligned or maintain exact incremental state with explicit
ordering and reduction invariants. Treat an incremental refresh redesign as a
separate semantic-risk project, not an opportunistic extension of the row
store.

### P6 — control cache state when a decision depends on small timing deltas

Shared checkpoint lifecycle is correct to refuse destructive cache advice. If
the control/candidate difference is large and concentrated in refresh, the
current pair can still be informative. If the difference is small or Phase 0
dominates differently across nodes, run a narrower confirmation protocol.

Options, in increasing cost:

1. report phase-local timing and current cache-state uncertainty;
2. repeat the pair with reversed submission/order conditions;
3. precondition only the exact owned ranges identically when safe; or
4. use the already validated job-private checkpoint lifecycle protocol for a
   high-consequence cold comparison.

Do not copy the complete roughly 171 GiB provider repository per ordinary run
unless the decision actually requires controlled cold-cache ownership. Earlier
lifecycle work established safety but did not establish a robust speed win.

For the present result, a reversed timing replicate is not required to retain
the candidate: the margin is large and 86.7% of the savings localizes to Phase
4. A controlled ownership run is required before making any total-host-memory
claim or sizing an automatic host-admission threshold from these cgroup totals.

### P7 — defer model-general conclusions

A 4B/512 win now broadens the measured 4B development length curve only. It
does not establish the 828 holdout, 4B/1,024, 12B, arbitrary prompts, or a
global default. Continue probe-then-full admission and canonical comparison at
each new model/length boundary.

## Directions not to repeat without new evidence

The following have already been tested or deliberately deferred:

- mapped file-only CUDA windows: exact but substantially slower and no useful
  measured total-memory reduction at 4B/256;
- direct-pread CUDA windows: storage-bound and stopped before a complete
  artifact;
- exact-range prepared refresh cache: poor reuse and expensive misses across
  the prior 0/8/32/64 GiB matrix;
- active pinned-CPU encoder residency: slower than pageable active CPU in the
  measured 12B work;
- larger Phase-4 physical batches as a generic speed fix: higher HBM and no
  demonstrated win in the relevant prior ladder;
- larger contraction tiles/source-layer fusion without a renewed kernel-bound
  profile; and
- unsafe page-cache eviction for shared checkpoint assets.

Archived implementations remain useful provenance. Restore one only if new
profiling invalidates the reason it was rejected, and state that changed
premise before implementation.

## Post-run decision matrix

| Full-pair outcome | Action |
|---|---|
| Candidate is `strict_exact` or `exact`, materially faster, no late fallback, resources acceptable | Retain as non-dominated at 4B/512; consider scoped promotion review, then proceed to the next frozen LS4 gate |
| Candidate is `close` or `bounded` but materially faster | Consider scoped selectable retention only; no automatic exact promotion |
| Candidate meets fidelity but is flat or slower | Keep the earlier admitted 4B range; do not promote 512, and do not add storage complexity without a measured bottleneck |
| Candidate hits host duplication while HBM and compute remain healthy | Open the single-owner canonical host-store prototype at 4B/256 first |
| Candidate falls back from `cuda_windowed` | Treat as failed mechanism admission; diagnose the exact resolution reason before any new mode |
| Either run fails or emits incomplete telemetry/artifacts | Repair execution/artifact quality and rerun the same frozen pair; do not tune |
| Timing difference is small and cache histories differ materially | Run a controlled confirmation before a performance selection |

`exact` is permission to enter promotion review, not an automatic selection.
Retention still compares runtime, host/HBM, implementation complexity,
fallback strength, provider compatibility, and evidence breadth.

## Final 4B/512 result

Jobs `1712546_0` and `1712550_0` completed, but they are not a valid candidate
comparison. Both landed on the same 70 GiB H200 MIG slice; the requested
`cuda_windowed` candidate resolved to `cpu_exact`, and the apparent end-to-end
gain was dominated by a cold control Phase 0 followed by a warm candidate
Phase 0. The first pair therefore establishes only that the fallback path
completed and produced matching canonical graphs. It does not measure the
candidate mechanism.

Job `1714134` reached a full H200 on `grn031`, passed the 139.80 GiB CUDA-device
gate, preheated the complete 49-file 178.11 GiB model/provider set, and produced
a complete warm `cpu_exact` control. The launch wrapper then failed because its
validator incorrectly required accelerator-resolution telemetry from native
`cpu_exact`, which does not instantiate the accelerator wrapper. The candidate
never started; this was a wrapper failure after a valid control artifact, not a
trace or resource failure.

The validator now accepts native `cpu_exact` from its recorded requested mode
while still requiring explicit matching resolution telemetry for accelerated
modes. Focused tests pass, the real control passes, and the known MIG fallback
is rejected. Candidate-only job `1721614` repeated the same preheat on an
explicit full H200 and passed both the 130 GiB device and required
`cuda_windowed` gates.

| Field | Control | Candidate |
|---|---:|---:|
| Scheduler state / exit | trace complete; wrapper job `1714134` failed `1:0` afterward | job `1721614` `COMPLETED`, `0:0` |
| Node | `grn031` | `grn031` |
| Trace seconds | 346.773 | 168.067 |
| Attribute wall seconds | 335.566 | 159.635 |
| Phase 0 seconds | 35.891 | 19.247 |
| Phase 1 seconds | 2.591 | 2.361 |
| Phase 2 seconds | 1.688 | 1.463 |
| Phase 3 seconds | 0.877 | 0.759 |
| Phase 4 seconds | 276.521 | 121.596 |
| Refresh seconds | 201.158 | 53.521 |
| Feature-batch seconds | 71.521 | 64.658 |
| Phase 5 seconds | 2.990 | 2.220 |
| Peak CUDA allocated / reserved | 61.21 / 63.08 GiB | 61.21 / 63.08 GiB |
| Max sampled cgroup current / anon / file | 201.27 / 9.19 / 191.54 GiB | 34.37 / 21.13 / 13.12 GiB |
| Slurm batch MaxRSS | about 206.02 GiB | about 34.37 GiB |
| Row mode requested / resolved | `cpu_exact` / native `cpu_exact` | `cuda_windowed` / `cuda_windowed` |
| Logical row bytes | 12,796,417,296 | 12,796,417,296 |
| CUDA window | n/a | 534,160,224 bytes; 171 rows x 2 buffers |
| Window reads / bytes | n/a | 2,652 / 281,571,160,416 |
| Window host stage / transfer wall | n/a | 3.980 / 4.114 seconds |
| Fallback / copy failures | not applicable; native control | none / 0 |
| Telemetry drops / sink errors | 0 / 0 | 0 / 0 |
| Compact graph status | saved | saved |

Performance classification: trace wall improved 51.5% (2.06x), Phase 4
improved 56.0% (2.27x), and refresh improved 73.4% (3.76x). Phase 4 accounts
for 86.7% of total trace savings. Smaller differences outside Phase 4 retain
allocation/order uncertainty and should not be credited to the row-window
mechanism.

Canonical comparison: 8,192/8,192 features and 20,000/20,000 edges shared;
feature, edge, weighted-edge, and all-edge Jaccard are all 1.0; normalized and
signed normalized L1 deviations are 0.0; sign agreement and target-token match
are 1.0; and Top-64, 128, 256, 512, and 1,024 are identical.

Final fidelity classification: `strict_exact` under the canonical thresholds.

Resource interpretation: do not compare the cgroup-current or MaxRSS totals as
mode memory. The control cgroup owned the cold preheat's shared file-cache
pages; the candidate reused them after the pages were warm and largely charged
elsewhere. Anonymous memory increased by about 11.94 GiB in the candidate,
which is consistent with the admitted host mirror. HBM was effectively flat.

Operational fit conclusion: one full H200 with a 400 GiB host allocation is a
demonstrated safe contract for this frozen workload. Peak CUDA reservation was
63.08 GiB on a 139.80 GiB device, leaving substantial device headroom. The host
allocation was ample even when the control cgroup owned roughly 201.27 GiB in
total, but the unequal page-cache charge prevents a minimum-RAM estimate. Keep
400 GiB for the next formal 4B long-prefix gate; do not size from the
candidate's approximately 34.37 GiB MaxRSS.

Performance/retention decision: retain `cuda_windowed` as non-dominated and
selectable for the measured 4B development curve. Do not promote it to a
global or arbitrary-prompt default; the evidence remains one development
prompt family and separate allocations.

Next LS4 gate: run a fail-closed, preheated transition probe for frozen
`ls4-4b-361-dev-1024`. Use measured resource observation without the stale
200 GiB estimate stop, and admit a full pair only after checking host anonymous
commit, HBM, provider mode, late fallback counters, and projected runtime. The
frozen 828 holdout remains a later independent generalization gate.

## Evidence locations

Campaign root:

`/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/long_trace/performance_optimization/ls4-4b-transfer-v1`

Prepared full bundles:

- `prepared/512/control/full-job-v1`;
- `prepared/512/candidate/full-job-v1`.

Corrected run roots:

- `runs/512/control/ls4-4b-512-cpu-exact-full-preheated-v1`;
- `runs/512/candidate/ls4-4b-512-cuda-windowed-full-preheated-v2`;
- `runs/512/preheated-full-h200-pair-v1` for the control device, preheat, and
  GPU-monitor provenance;
- `runs/512/preheated-candidate-h200-v2` for the candidate-equivalent
  provenance; and
- `runs/512/comparisons/cpu-exact-preheated-v1-vs-cuda-windowed-preheated-v2.json`
  for the canonical graph comparison.

Launch provenance is recorded in `experiments/logs/2026-08.jsonl`.

This result updates the measured LS4 development-length interpretation but
does not change the global default, fidelity scope, governor bundle, or
scientific baseline.
