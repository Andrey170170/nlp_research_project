# Exact-trace LS3 1B prompt generalization

Date: 2026-08-03

Status: LS3 complete; broad exact auto-promotion rejected, selectable mode
retained

## Decision

Retain `ls2-1b-long-prefix-cuda-windowed-512mib-b128-v1` as a non-dominated
selectable 1B row-influence profile, but do not auto-select it as exact for an
unseen prompt.

The frozen profile won every holdout by 36.7--47.8% runner wall and remained
inside the declared HBM and host envelopes. Its worst holdout was still
`close`, so it passes code-retention review and is useful for real traces when
that drift is acceptable. It does not pass automatic exact-promotion review
across arbitrary prompts: prompt 94 at 256 tokens was `close`, even though the
same prompt was `strict_exact` at 512 tokens. There is no safe prompt/length
selector for that distinction before running the trace.

Keep `cpu_exact` as the automatic exact route for unseen prompts. Preserve the
LS2 exact claim only for its measured development curve and the exact LS3
points reported below. No launch default or baseline registry changed.

## Frozen holdout matrix

The profile was frozen before holdout observation. Prompt 828 ran candidate
then control; prompt 94 reversed the order. No parameter was tuned after any
holdout result.

| Prompt | Prefix | CPU exact wall | CUDA windowed wall | Wall win | CPU Phase 4 | CUDA Phase 4 | Refresh reduction | Fidelity |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 828 | 256 | 119.094s | 75.390s | 36.7% | 95.677s | 49.135s | 69.7% | `exact` |
| 94 | 256 | 124.895s | 76.317s | 38.9% | 102.137s | 50.298s | 70.3% | `close` |
| 828 | 512 | 198.702s | 107.331s | 46.0% | 170.777s | 75.979s | 71.1% | `strict_exact` |
| 94 | 512 | 201.711s | 105.325s | 47.8% | 174.889s | 77.568s | 69.5% | `strict_exact` |

Feature-batch time remained stable within each pair while refresh produced the
gain. For prompt 94/512, for example, feature-batch work was 32.294 versus
32.266 seconds, while refresh fell from 139.968 to 42.746 seconds.

## Fidelity

| Prompt / prefix | Feature Jaccard | Edge Jaccard | Weighted Jaccard | Normalized / signed L1 | Top-K result | Class |
|---|---:|---:|---:|---:|---|---|
| 828 / 256 | 0.999512 | 0.999900 | 0.999946 | 0.0000544 | Top-64--1024 exact | `exact` |
| 94 / 256 | 0.993672 | 0.992528 | 0.991711 | 0.008323 | 63/64, 127/128, 255/256, 510/512, 1019/1024 | `close` |
| 828 / 512 | 1.0 | 1.0 | 1.0 | 0.0 | Top-64--1024 exact | `strict_exact` |
| 94 / 512 | 1.0 | 1.0 | 1.0 | 0.0 | Top-64--1024 exact | `strict_exact` |

All four pairs matched target tokens and shared edge signs. The 828/256 point
meets the canonical `exact` thresholds and exact Top-256 requirement. The
94/256 point meets `close`, not `exact`, because its worst Jaccard is 0.991711,
L1 deviation is 0.008323, and Top-256 support differs by one edge.

## Scaling and resources

| Prompt / prefix | Active decoder rows | Decoder-row bytes | Logical row store / host mirror | Window | Streamed bytes | Max cgroup current | Max CUDA reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| 828 / 256 | 149,001 | 343.30 MB | 4.88 GB | 536.40 MB | 92.17 GB | 76.75 GiB | 28.42 GiB |
| 94 / 256 | 154,266 | 355.43 MB | 5.06 GB | 536.85 MB | 96.60 GB | 77.11 GiB | 26.12 GiB |
| 828 / 512 | 300,170 | 691.59 MB | 9.84 GB | 535.50 MB | 182.78 GB | 86.63 GiB | 47.53 GiB |
| 94 / 512 | 305,262 | 703.32 MB | 10.00 GB | 534.82 MB | 194.77 GB | 87.36 GiB | 45.31 GiB |

Every candidate resolved `cuda_windowed` without fallback. Window staging took
2.48--5.24 seconds and synchronization stayed below 0.005 seconds. Every run
completed all 8,192 selected features, 33 semantic / 65 physical batches, nine
refreshes, and zero telemetry sink errors.

## Applicability boundary

- Automatic exact selection: retain `cpu_exact` for arbitrary/unseen prompts.
- Selectable fast mode: retain the frozen CUDA-windowed profile as
  close-or-better over the measured 1B 256--512 prompt holdouts, with atomic
  exact fallback on resource or compatibility refusal.
- Exact evidence: preserve the LS2 development curve and the LS3 828/256,
  828/512, and 94/512 points as measured exact scopes.
- Unsupported inference: do not infer that 512 tokens are universally exact,
  or introduce a prompt-identity-specific automatic selector, from two
  holdouts.
- Future tuning: none on these holdouts. Any new mechanism or budget is a new
  candidate and requires new development and holdout separation.

## Evidence paths

Scratch campaign root:

`/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/long_trace/performance_optimization/ls1-1b-length-scaling-v1`

Canonical comparisons:

- `comparisons/ls3-828-256-cpu-exact-vs-cuda-windowed-v1.json`
- `comparisons/ls3-94-256-cpu-exact-vs-cuda-windowed-v1.json`
- `comparisons/ls3-828-512-cpu-exact-vs-cuda-windowed-v1.json`
- `comparisons/ls3-94-512-cpu-exact-vs-cuda-windowed-v1.json`

## Next gate

Begin LS4 with the existing launch machinery and the frozen mechanism family.
Use a short 4B capacity probe at each new length, then complete admitted traces.
Do not carry the 1B exact label across model sizes; compare each 4B artifact
against its matched `cpu_exact` reference and reopen optimization only when a
new component consumes at least 10% of relevant wall time.
