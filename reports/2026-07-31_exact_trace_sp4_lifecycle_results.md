# SP4.2 results: checkpoint lifecycle

Date: 2026-07-31

Status: gate passed; SP4 complete

## Scope and protocol

SP4.2 tests whether the provider-owned checkpoint lifecycle safely releases
owned decoder checkpoint pages after active decoder rows are sealed, without
causing Phase-4 reloads or harming steady-state execution.

The formal protocol ran in Slurm job 1685189 on Granite node `grn029` with one
H200. The 4B GemmaScope repository was copied from shared VAST into
`/scratch/local/u1653998/1685189/huggingface` with new device/inode identities.
The staged repository was 171 GiB; its blob inventory matched the source, and a
representative 5,370,816,904-byte blob had the same SHA-256 digest but a
different device/inode. The 8.1 GiB base-model repository remained a read-only
link to the shared cache because lifecycle advice targets only the transcoder
assets.

Before every cold arm, all 34 private transcoder blobs were individually
`fdatasync`ed and passed to `posix_fadvise(..., POSIX_FADV_DONTNEED)` with their
explicit file sizes. `fincore` verified zero resident bytes out of
182,607,774,736 bytes before each arm. No node-global cache operation was used.

Both policies used this same physical job-private cache identity:

- the reference profile declared `checkpoint_asset_scope=shared`, forcing the
  safe lifecycle-refusal path; and
- the private profile declared `checkpoint_asset_scope=job_private`, admitting
  exact-range release advice.

This intentionally conservative reference label isolates lifecycle policy from
filesystem identity and cold-state differences.

## Full exact runs

The full pair used immutable snapshot
`workspace_20260731_160432_sp4_lifecycle_private_v6`, project commit `4c3c232`,
and sibling commit `c78ab46`.

| Metric | Reference fallback | Private lifecycle | Change |
|---|---:|---:|---:|
| Completion | 193.663853 s | 189.475649 s | -2.163% |
| Attribution | 191.191502 s | 187.067745 s | -2.157% |
| Phase 0 | 22.347528 s | 18.235974 s | -18.398% |
| Phase 4 | 150.68 s | 144.33 s | -4.214% |
| Warm batches 4-65, mean | 811.285 ms | 791.889 ms | -2.391% |
| Warm batches 4-65, median | 722.108 ms | 726.406 ms | +0.595% |

Both runs passed the exact signed compact gate:

- feature, all-edge, top-256 edge, and weighted-edge Jaccard: 1.0;
- shared-edge sign agreement: 1.0;
- magnitude and signed normalized L1 deviation: 0.0; and
- target-token match: 1.

Both runs reported zero post-transition decoder page loads and zero decoder
load bytes. The private policy therefore did not refault decoder checkpoints
during Phase 4. The small timing differences are not promoted as a speedup;
they establish that accepted release does not harm warm steady-state timing.

## Lifecycle and cgroup evidence

The reference emitted 34 explicit refusals with reason
`scope_shared_is_not_advice_eligible`. The private policy issued all 34
requests without error. The exact owned ranges totaled 91,268,055,040 bytes
(85 GiB). Requested range bytes describe lifecycle ownership; they are not an
estimate of resident cache bytes.

The first cgroup-enabled probes exposed a missing cgroup-v1 adapter. Sibling
commit `b36359a` maps the legacy controller's current, inherited finite limit,
headroom, peak, RSS/cache, and active/inactive file counters into the generic
resource schema. Seventeen focused resource/telemetry tests and targeted lint
passed. Live validation resolved the allocation limit as 250 GiB. The reported
170.79 GiB cgroup peak includes the earlier 171 GiB staging copy and must not be
interpreted as trace peak memory.

Final three-batch probes used immutable snapshot
`workspace_20260731_162437_sp4_lifecycle_cgroup_v7`, project commit `4c3c232`,
and sibling commit `b36359a`:

| Transition evidence | Reference fallback | Private lifecycle |
|---|---:|---:|
| Release outcomes | 34 refused | 34 issued |
| cgroup current delta | +0.000031 GiB | -4.461796 GiB |
| cgroup file delta | 0 GiB | -4.425735 GiB |
| cgroup anonymous delta | +0.000008 GiB | -0.010109 GiB |
| process read-byte delta during advice | 0 | 0 |
| major-fault delta during advice | 0 | 0 |

The private transition therefore produced an observed, file-dominated cache
charge reduction rather than merely a successful syscall.

The synchronized first-three-batch CUDA-event evidence was:

| Batch | Reference wall / CUDA | Private wall / CUDA |
|---:|---:|---:|
| 1 | 1,446.357 / 376.646 ms | 1,436.176 / 421.932 ms |
| 2 | 1,029.786 / 404.958 ms | 1,187.931 / 468.307 ms |
| 3 | 1,078.497 / 442.259 ms | 1,188.272 / 454.645 ms |
| Sum | 3,554.640 / 1,223.863 ms | 3,812.379 / 1,344.884 ms |

The private synchronized probe was 7.251% slower in wall time and 9.888% slower
in CUDA time. This does not reproduce the historical transition speed win and
must not be presented as one. It is compatible with ordinary run variance and
the explicit synchronization probe; more importantly for this gate, all three
batches completed within the same resource envelope, showed no decoder reload,
and the full-run warm mean and Phase-4 total did not regress.

## Decision

SP4.2 passes:

- the cache identity was genuinely job-private and cold-state verified;
- shared-scope fallback refused every unsafe release;
- private scope issued every owned exact-range release;
- cgroup-v1 evidence observed a 4.426 GiB file-cache reduction;
- three matched transition batches completed with CUDA-event and OS/cgroup
  evidence;
- full signed compact output remained exact;
- no decoder pages reloaded after transition; and
- warm steady-state timing was not harmed.

This accepts the existing typed checkpoint lifecycle mechanism. It does not
change launch defaults and does not claim a repeatable lifecycle speedup.

## Provenance

- Full run IDs: `sp4-lifecycle-cold-reference-v6` and
  `sp4-lifecycle-cold-private-v6`.
- Preliminary probe IDs: `sp4-lifecycle-probe-reference-v6` and
  `sp4-lifecycle-probe-private-v6`.
- Final cgroup-enabled probe IDs: `sp4-lifecycle-probe-reference-v7` and
  `sp4-lifecycle-probe-private-v7`.
- Artifacts:
  `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/sweep/performance_optimization`.
- Plan: `plans/2026-07-29_exact_trace_performance_optimization_loop.md`.

SP4 is complete. Continue in the selected order with open-ended SP1, then
SP5/LS0 including deterministic trajectory generation.
