# 4B/12B exact-trace optimization results

Status: engineering execution complete; formal Gates B/C unresolved; no
launch-default or scientific-baseline promotion

Date: 2026-07-27

Plan: `plans/2026-07-27_large_model_optimizations.md`

Scope: Gemma 3 4B/12B PLT, `361_base`, 124-token prompt, one H200, 250 GiB
job allocation, `max_feature_nodes=8192`, `decoder_chunk_size=4096`, internal
FP32 attribution, and bounded or compact-strict gates as declared by the plan.

## Result

The provider-owned mapped selective-row source is accepted as an exact physical
mechanism. It preserves row values and caller order, removes Phase-4 decoder
loads, and reduces accounted 12B raw checkpoint bytes for decoder-row
materialization from 95.316 GB to 0.910 GB.

The reproducible 12B reference remains mapped b64/c4096 with lazy encoder
placement. It completed twice in 586.30s and 563.27s and produced identical
signed compact graphs under the permutation-insensitive comparison. The
serialized NPZ arrays are not byte-identical because two equal graph entries
exchange order. Active-CPU encoder residency is the fastest tested bounded
candidate at 509.80s and 508.67s, but its two graphs are only bounded-aligned,
not compact-strict. Pinned CPU, b128, b256, and larger FP32 contraction tiles
were rejected.

No governor bundle, fidelity scope, baseline registry, or default was changed.

## Provenance

- Slurm job: `1664400`
- node / GPU: `grn030` / NVIDIA H200
- allocation: 12 CPUs, 250 GiB, one H200
- project branch: `perf/exact-trace-loop`
- sibling branch: `perf/exact-trace-loop`
- main mechanism snapshots:
  - v4: project `7216e35`, sibling `65c94bb`
  - v5: project `7216e35`, sibling `a648e5f`
  - v6: project `0227724`, sibling `a648e5f`
  - v7: project `05e29a0`, sibling `a648e5f`
  - v9: project `75bee35`, sibling `17c8e41`
- every GPU run used a read-only project+sibling snapshot and the job-private
  offline Hugging Face cache
- cache state was warm/uncontrolled unless explicitly described; no global
  cache drop was used

The project manifest records the pre-existing untracked patch files and
`slurm-1664400.out`. They were not used as runtime source.

## Selective decoder-row source

### 4B compact-strict gate

Two immutable 4B runs passed feature, edge, Top-256, weighted, magnitude,
sign-agreement, and signed-L1 gates exactly:

| Run | Subprocess | Completion | Phase 0 | Phase 4 | Peak CUDA reserved |
|---|---:|---:|---:|---:|---:|
| mapped repeat 1 | 345.39s | 310.83s | 128.51s | 158.37s | 26.24 GiB |
| mapped repeat 2 | 234.46s | 206.61s | 39.89s | 149.12s | 26.24 GiB |

The timing spread is cache-state sensitivity; exact signed parity and logical
row-source accounting were stable. The external framebuffer peak was
27,685 MiB.

### 12B Phase 0

- unique decoder rows: 59,249
- decoder-row occurrences: 127,580
- retained output bytes: 455,032,320
- raw FP32 backend bytes: 910,064,640
- fallback raw checkpoint bytes: 95,315,558,400
- raw backend fraction: 0.955%
- stable mappings / backend requests: 48 / 48
- mapped blocks: 279,952
- explicit per-row reads: 0
- temporary staging high-water: 38,730,240 bytes
- Phase-4 decoder loads: 0

The 12B Phase-0 probe took 653.26s versus the prior 1,350.83s chunk-scan result,
a 51.6% reduction. Warm full runs reached 62.6-63.4s Phase 0. Backend byte
accounting is stable, but wall time is strongly affected by Linux page-cache
state; OS physical reads were not attributable under the shared-cache protocol.

## Completed 12B lane

All completed full runs produced token `Here`, passed the frozen historical
bounded gate, stayed below the 64 GiB rigid-RSS stop, and had zero post-Phase-0
decoder loads. The CLI reports them as measurement-only because a hard 12B
promotion gate was intentionally not installed.

| Candidate | Runs | Subprocess | Completion | Phase 0 | Phase 4 | Peak CUDA reserved | Disposition |
|---|---:|---:|---:|---:|---:|---:|---|
| lazy encoder, b64 | 2 | 586.30 / 563.27s | 555.62 / 539.42s | 63.43 / 62.59s | 458.42 / 443.74s | 38.82 GiB | reproducible reference |
| active CPU, b64 | 2 | 509.80 / 508.67s | 478.01 / 483.19s | 86.56 / 90.67s | 359.68 / 359.56s | 38.82 GiB | fastest bounded candidate |
| active pinned CPU, b64 | 1 | 520.60s | 490.59s | 91.21s | 367.76s | 38.82 GiB | reject |
| lazy encoder, b128 | 1 | 538.30s | 512.26s | 60.56s | 417.49s | 55.19 GiB | reject |
| lazy encoder, b256 | 1 | 533.37s | 505.96s | 58.57s | 408.86s | 86.97 GiB | reject |

The lazy-repeat permutation-insensitive signed compact-graph comparison is
exact. The active-CPU repeat comparison is bounded only:

- feature Jaccard: 0.999023914
- all-edge Jaccard: 0.988961265
- Top-256 Jaccard: 1.0
- weighted-edge Jaccard: 0.989621080

One active-CPU repeat is compact-graph-equal to the lazy reference and the
other has 130 rather than 129 Phase-4 execution batches. Therefore active CPU
is suitable only for a dataset-frozen bounded campaign whose placement policy
is fixed before tracing; it is not the compact-strict reference.

b128 passes the exact permutation-insensitive compact-graph comparison against
lazy b64 and b256 has weighted Jaccard 0.999999938, but both show endogenous
later-frontier identity drift. Neither is a Pareto winner: they are slower than
active CPU and reserve substantially more HBM.

The previous first-batch transition stall was removed. The mapped lazy
transition probe completed its first three Phase-4 batches in 5.86s; the prior
unmapped run required 308.14s for only its first two batches. This is strong
operational evidence, but not a formal matched three-batch cold/refault gate
because the old run did not retain the required third-batch/CUDA-event record.

## Compute gate

Diagnostic CUDA timing initially refused active-CPU residency because it
incorrectly keyed on the encoder-vector device. Sibling commit `17c8e41` fixes
that diagnostic-only guard.

The corrected active-CPU transition probe measured 2.043s of CUDA work inside
3.59s Phase-4 wall time (56.9%), admitting the lower-risk Stage-H contraction
tile ladder:

| FP32 contraction tile | CUDA time, three batches | Phase-4 wall | Result |
|---|---:|---:|---|
| incumbent 4096 | 2.043s | 3.59s | control |
| 16384 | 2.009s | 3.62s | reject: 1.7% CUDA gain, wall regression |
| 65536 | 2.082s | 3.67s | reject: CUDA and wall regression |

No tile earned a full run. Source-layer fusion was deferred rather than
experimentally rejected: the lower-risk tile proxy showed no material wall
benefit, while fusion would add concatenation/scatter work and change traversal
and accumulation grouping.

## Gate disposition

| Stage | Disposition |
|---|---|
| A typed evidence contracts | passed by focused capability/scope/pin tests |
| B phase-scoped diagnostics | implemented; probe lifecycle passed; formal <2% paired 4B overhead and attributable cold-refault subgate remain unavailable |
| C checkpoint lifecycle | implemented and transition stall removed; formal matched cold three-batch fault/disparity gate unavailable from the old two-batch artifact |
| D selective decoder rows | D1 synthetic passed; D2 4B compact-strict repeated; D3 traffic and Phase-0 continuation gates passed |
| E completed 12B reference | passed twice under bounded historical parity |
| F encoder placement | active CPU passes the speed/bounded gate but fails compact-strict repeatability; retain as bounded-only |
| G execution envelopes | b128/b256 completed and bounded-pass, but fail Pareto selection; no winner to repeat/combine |
| H compute | admitted, tile ladder executed, no winner; source fusion deferred after the proxy |

The 1k-prefix transfer rung was not launched because the prerequisite
compact-strict admission for the active placement/execution winner was not met.
It remains a separate workload scope rather than unfinished confirmation of the
124-token result.

## Current recommendation

1. Use mapped decoder rows universally where the provider capability and
   checkpoint identity gates pass.
2. Use lazy mapped b64/c4096 as the reproducible 12B reference.
3. Use active-CPU mapped b64/c4096 only for explicitly bounded,
   placement-frozen campaigns that value the approximately 11% mean total and
   20% Phase-4 improvement over strict repeatability.
4. Do not use pinned CPU, b128, b256, larger contraction tiles, or source-layer
   fusion as defaults; fusion remains unmeasured.
5. Keep c65536 as the separately named bounded chunk regime from the prior 4B
   study; this campaign did not combine it with other axes.
