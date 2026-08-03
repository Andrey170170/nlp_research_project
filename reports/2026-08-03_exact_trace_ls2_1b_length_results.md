# Exact-trace LS2 1B length results

Date: 2026-08-03

Status: LS2 complete; development finalist frozen for LS3 holdouts

## Decision

Promote `ls2-1b-long-prefix-cuda-windowed-512mib-b128-v1` as the exact 1B
development finalist over the admitted 129--1024-token curve. Promotion is
scoped to this model/provider and measured length band. It does not make
`cuda_windowed` a global default or override the historically bounded evidence
for other model sizes.

The finalist combines:

- selective mapped-row Phase 0;
- final-token-only Phase-1 logit materialization on supported models;
- 1,536 MiB exact active-decoder-row admission;
- 256-row semantic and 128-row physical Phase-4 batches; and
- a double-buffered 512 MiB CUDA feature-row influence window with atomic
  fallback.

## Development curve

| Prefix tokens | Exact reference wall | Finalist wall | Wall improvement | Reference Phase 4 | Finalist Phase 4 | Fidelity |
|---:|---:|---:|---:|---:|---:|---|
| 129 | 61.185s | 56.485s | 7.7% | 39.785s | 33.600s | `strict_exact` |
| 256 | 108.443s | 73.584s | 32.1% | 85.864s | 47.957s | `exact` |
| 512 | 179.265s | 104.163s | 41.9% | 153.949s | 75.005s | `strict_exact` |
| 1024 | 316.037s | 164.218s | 48.0% | 282.750s | 129.723s | `strict_exact` |

The 256-token result differs only at floating-point scale: weighted Jaccard is
`0.9999999992505764`, normalized magnitude and signed L1 deviation are both
`7.494236341232103e-10`, and feature/edge support, signs, Top-64 through
Top-1024, and target token remain exact.

## Scaling and resources

| Prefix tokens | Logical row store / host mirror | Window bytes | Streamed window bytes | Max observed cgroup current | Max observed CUDA reservation |
|---:|---:|---:|---:|---:|---:|
| 129 | 2.30 GB | 536.56 MB | 40.82 GB | 71.53 GiB | 20.31 GiB |
| 256 | 4.49 GB | 535.93 MB | 85.54 GB | 75.88 GiB | 28.19 GiB |
| 512 | 9.07 GB | 535.63 MB | 168.31 GB | 85.19 GiB | 47.53 GiB |
| 1024 | 18.46 GB | 536.24 MB | 362.21 GB | 103.52 GiB | 111.49 GiB |

At 1024 tokens, refresh fell from 234.704 to 79.404 seconds while feature-batch
work remained essentially flat. The provider performed 1,643 window reads and
prefetches; host staging took 9.57 seconds and stream synchronization 0.008
seconds. The full run stayed inside the declared 90% H200 and 200 GiB host
envelopes.

## Implementation finding

The first full-answer diagnostic did not actually activate the requested
provider: the canonical full-answer `TraceRequest` bridge omitted the four
feature-row influence fields when constructing `RowStoragePlan`. Project commit
`dd8ac10` fixes the general bridge and adds focused tests. Telemetry now records
`feature_row_influence_mode_requested=cuda_windowed` and
`feature_row_influence_mode_resolved=cuda_windowed`.

Relevant commits:

- `4efd161` — registered LS2 windowed-CUDA profile;
- `dd8ac10` — full-answer row-influence bridge fix; and
- `abeff24` — corrected 1024-token result and retention decision.

The false pre-fix probe/full runs are integration diagnostics only and are not
scientific evidence.

## Evidence paths

Scratch campaign root:

`/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/long_trace/performance_optimization/ls1-1b-length-scaling-v1`

Canonical comparisons:

- `comparisons/129-cpu-exact-b256-vs-cuda-windowed-b128-v1.json`
- `comparisons/256-cpu-exact-b256-vs-cuda-windowed-b128-v1.json`
- `comparisons/512-cpu-exact-b256-vs-cuda-windowed-b128-v1.json`
- `comparisons/1024-cpu-exact-vs-cuda-windowed-wired-v2.json`

## Next gate

Freeze this profile and begin LS3. Run a contemporaneous development control
and at least two held-out prompts per selected length band. Do not tune after
observing holdout outcomes; accept, reject, or narrow the applicability scope.
