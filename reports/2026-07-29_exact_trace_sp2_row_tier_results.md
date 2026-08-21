# Exact-trace SP2 feature-row tier results

Status: implementation complete; exact promotion rejected; bounded research
candidate retained default-off

Date: 2026-07-29

Plan: `plans/2026-07-29_exact_trace_performance_optimization_loop.md`

Scope: Gemma 3 12B PLT `361_base`, 124-token prompt, one H200,
`max_feature_nodes=8192`, mapped selective decoder rows, b64/c4096, lazy
encoder placement, and FP32 internal attribution.

## Outcome

SP2 produced a capability- and shape-based full GPU feature-row tier with
explicit HBM budget/safety admission, canonical file backing, atomic exact
fallback, cleanup, and cumulative traffic telemetry. The implementation is
disabled by default and is covered by focused CUDA tests.

No candidate simultaneously passed the exact signed-graph gate and the
predeclared 10% Phase-4 improvement gate:

| Variant | Phase 4 | Batches | Exact vs control | Disposition |
|---|---:|---:|---|---|
| mapped lazy file control | 421.81s | 130 | reference | exact incumbent |
| CUDA influence arithmetic | 189.09s | 129 | fail: feature Jaccard 0.999024, signed L1 0.010433 | bounded research only |
| exact signed host mirror | 427.84s | 130 | pass: all canonical metrics 1.0, signed L1 0 | reject: 1.4% slower |
| zero-copy prepared host mirror | 215.56s | 129 | same exact failure as CUDA path | rejected SP3 experiment |
| fresh-layout prepared host mirror | stopped at 65% | n/a | focused solver exact; no complete graph | reject: timing converged to control |

The fast CUDA candidate reduced Phase 4 by 55.2% and total harness time by
50.1%, but changed four of 8,192 selected features. Common-edge weights were
almost perfectly correlated and preserved sign; the divergence arose at the
CPU-versus-CUDA or zero-copy reduction/layout boundary, so it is not an exact
physical optimization.

The exact signed-host candidate restored all 8,192 features, all 20,000 edges,
all signed values, target token `Here`, and the 130 semantic batches. It also
raised process RSS from about 20 GiB during early Phase 4 to 24.04 GiB and CUDA
reserved from 38.82 to 42.71 GiB, while slightly regressing Phase 4.

## Traffic correction

The completed control's final cumulative row counter is 111,610,045,920 bytes
(103.97 GiB) across 218,706 row touches. The previously reported approximately
1.616 TB was obtained by summing cumulative snapshots from every refresh, so it
double-counted earlier traffic. SP2 eliminated the actual 111.61 GB of file
reads in admitted variants; it did not eliminate 1.616 TB of unique transfer.

This correction also explains why storage-only exact residency did not deliver
the expected payoff. In the control, aggregate refresh accounting was:

- partial influence: 244.02s;
- file-row read: 70.51s;
- transfer/cast/absolute value: 133.42s;
- normalization: 10.48s; and
- direct accumulation/matmul: 28.26s.

The exact signed-host variant shifted rather than removed the controlling CPU
memory pass: partial influence was 250.44s, row-reader time 66.61s,
transfer/absolute time 144.46s, normalization 9.72s, and matmul 28.37s.

The zero-copy prepared mirror reduced partial influence to 39.79s and removed
the repeated absolute-value pass, but changed the CPU tensor storage
offset/alignment contract and reproduced the same compact divergence as the
CUDA solver. Forcing a fresh contiguous allocation restored the reference
layout but also restored control-like wall time.

## Implementation and provenance

Project:

- branch: `perf/exact-trace-loop`
- profile/request commit: `7fcc5b5`
- profile: `plt-selective-mapped-rows-gpu-store-12b-v1`

Sibling:

- core GPU tier: `702ea2a`
- refresh telemetry: `ed90bc6`
- exact CPU arithmetic contract: `0e58b66`
- exact host mirror: `fff8650`
- rejected prepared experiment: `3fdc7cc`, `2fe273d`
- prepared experiment reverts / final source: `9b1eabb`, `e32c4aa`

Principal immutable snapshots:

- paired control / fast CUDA candidate:
  `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/workspace_snapshots/workspace_20260729_145326_sp2_gpu_row_tier_v2_telemetry`
- completed exact signed-host candidate:
  `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/workspace_snapshots/workspace_20260729_152902_sp2_gpu_row_tier_v4_exact_host_mirror`
- completed zero-copy prepared candidate:
  `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/workspace_snapshots/workspace_20260729_154904_sp2_sp3_exact_prepared_row_tier_v5`

All model runs used job `1671648` on `grn032`, one NVIDIA H200, the offline
shared Hugging Face cache, and immutable project+sibling snapshots. Cache state
was warm/uncontrolled; no global cache drop was performed. The harness ran with
the `bounded` comparison policy because no same-regime 12B baseline is
registered for the exact policy; exactness was evaluated directly with the
compact/signed artifact comparator.

Validation:

- focused SP2/telemetry tests: 43 passed on H200;
- broader affected Phase-4 suite: 81 passed;
- project performance-profile tests: 114 passed;
- affected-file Ruff checks: passed;
- one unrelated architecture guard still fails on the pre-existing 338-line
  `finish_phase4` function, which this work did not modify.

Pre-existing untracked patch files and Slurm logs in the project worktree were
recorded by snapshot provenance and were not used or committed.

## Decision

Keep the SP2 mechanism and 12B profile default-off for bounded research and
future hardware/long-prefix experiments. Do not promote it as the short-prefix
exact finalist and do not compose it with active-CPU encoder residency.

SP3's profile-driven prepared-row attempt is rejected and removed from the
final source state. No remaining non-batch mechanism from the current profile
has a demonstrated exact 10% opportunity, so the campaign should not advance
to formal SP4/SP5 promotion on this candidate. A future revisit needs a
fundamentally exact reduction implementation or a workload where storage
traffic, rather than the CPU reduction memory pass, is controlling.

## SP2.2 addendum

The exact-promotion decision above remains unchanged, but its final-source
disposition is superseded for the explicitly bounded lane. SP2.2 restores the
fast CUDA and prepared CPU consumers as named, selectable, default-off modes
and adds a pinned double-buffered bounded-HBM CUDA fallback. See
`reports/2026-07-29_exact_trace_sp2_2_selectable_modes_results.md`.
