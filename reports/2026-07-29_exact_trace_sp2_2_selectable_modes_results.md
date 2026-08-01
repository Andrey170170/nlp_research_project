# Exact-trace SP2.2 selectable row-execution results

Status: implementation complete; bounded CUDA speed and bounded-HBM modes
validated on 12B; exact default unchanged

Date: 2026-07-29

Plan: `plans/2026-07-29_exact_trace_performance_optimization_loop.md`

Scope: Gemma 3 12B PLT `361_base`, 124-token prompt, one H200,
`max_feature_nodes=8192`, mapped selective decoder rows, b64/c4096, lazy
encoder placement, and FP32 internal attribution.

## Outcome

SP2.2 turns the earlier SP2/SP3 experiments into explicit row-influence modes
over one canonical signed file-backed store:

| Mode | Storage/consumer | Fidelity disposition |
|---|---|---|
| `cpu_exact` | canonical file-backed signed rows and incumbent CPU preparation/reduction | exact incumbent and default |
| `cpu_prepared` | one 4.18 GB absolute-value host mirror | bounded opt-in |
| `cuda_full` | complete 4.18 GB signed matrix in HBM and CUDA influence consumer | bounded opt-in; fastest |
| `cuda_windowed` | 4.18 GB signed host mirror, two pinned staging buffers, and two bounded CUDA buffers | bounded opt-in; memory-pressure fallback |
| `auto` | try full CUDA, then windowed CUDA, then `cpu_exact` | capability/budget selector; resolved mode is reported |

Mode selection is separate from byte admission. The committed 12B profiles use
an 8 GiB full-residency budget, a 512 MiB total window budget, and a 16 GiB
free-HBM safety margin. Admission and allocation failures fall back atomically
to `cpu_exact`.

## 12B measurements

The selectable implementation reproduced the prior full-CUDA result and
measured both the initial synchronous window and the completed pinned,
double-buffered implementation:

| Variant | Harness | Completion | Phase 0 | Phase 4 | Batches | Peak CUDA reserved | Peak Phase-4 RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| prior `cpu_exact` control | n/a | 525.26s | 63.90s | 421.81s | 130 | 38.82 GiB | about 20 GiB |
| selectable `cuda_full` | 311.29s | 285.17s | 55.81s | 190.59s | 130 | 42.71 GiB | 20.13 GiB |
| synchronous `cuda_windowed` | 495.31s | 469.77s | 57.18s | 373.20s | 129 | 38.82 GiB | 24.02 GiB |
| pipelined `cuda_windowed` | 426.65s | 396.90s | 56.52s | 300.91s | 129 | 38.82 GiB | 24.77 GiB |

The selectable `cpu_prepared` transition probe completed three Phase-4 batches
in 3.66 seconds. Its earlier complete implementation measured 215.56 seconds
for Phase 4 and had the same bounded divergence as the CUDA consumer; SP2.2
retains it as an explicit bounded mode rather than an exact candidate.

At this workload:

- `cuda_full` cuts Phase 4 by 54.8% versus `cpu_exact`;
- pipelined `cuda_windowed` cuts Phase 4 by 28.7% versus `cpu_exact`;
- double buffering cuts windowed Phase 4 from 373.20 to 300.91 seconds
  (19.4%) and total harness time from 495.31 to 426.65 seconds (13.9%);
- `cuda_full` is still 110.32 seconds faster than pipelined windowing in
  Phase 4; and
- windowing saves 3.89 GiB of CUDA reservation versus full residency.

The first three batches do not expose this scaling cost: `cpu_prepared`,
`cuda_full`, synchronous windowing, and pipelined windowing measured 3.66,
3.89, 3.79, and 3.72 seconds respectively. Complete traces are required to
measure the increasing refresh prefixes.

## Window pipeline and traffic

The 512 MiB window budget is the total HBM budget for two buffers:

- 526 rows per buffer;
- 536,856,640 bytes (0.500 GiB) of total owned window HBM;
- 536,856,640 bytes of pinned host staging;
- 4,181,562,080 bytes (3.894 GiB) in the canonical signed host mirror;
- 2,616 ordered prefetches and matching CUDA-stream waits; and
- only 10.67 ms of host slot-reuse synchronization.

The complete pipelined run avoided 111,868,267,840 bytes (104.185 GiB) of file
reads, then streamed the same number of host/H2D bytes. Host staging accounted
for 3.59 seconds and transfer dispatch for 3.76 seconds. More importantly,
cumulative solver row-reader time fell from 170.41 seconds in the synchronous
implementation to 57.83 seconds, and partial-influence time fell from 196.80
to 123.28 seconds.

Windowing therefore bounds HBM but does not change the logical traffic
complexity. Each refresh consumes an increasing active-row prefix. At a fixed
8,192-node output cap, full-store and host-mirror bytes grow with active-feature
width; if both the selected-node cap and active-feature width grow, capacity
grows multiplicatively. The fixed window remains bounded by construction, but
its row capacity shrinks as each row becomes wider and its repeated H2D traffic
grows with the logical solver work.

For the committed 8 GiB `auto` full-residency budget, the current geometry
switches away from `cuda_full` at roughly 262,000 active feature columns,
about 2.05 times the present 127,580 columns. This is an experimental byte
threshold, not a model-name or governor policy; larger explicit budgets may be
appropriate on an H200 when the declared safety margin still passes.

## Fidelity

Both complete selectable CUDA modes pass the frozen bounded gate.

`cuda_full` versus the frozen historical anchor:

- feature Jaccard: 0.999023914;
- all-edge Jaccard: 0.999500125;
- weighted all-edge Jaccard: 0.999529079;
- signed normalized L1: 0.000471032;
- top-256 edge Jaccard, shared-edge sign agreement, and target token: 1.0.

Pipelined `cuda_windowed` versus the frozen anchor:

- feature Jaccard: 0.999755889;
- all-edge Jaccard: 0.989455884;
- weighted all-edge Jaccard: 0.990086381;
- signed normalized L1: 0.009963004;
- top-256 edge Jaccard, shared-edge sign agreement, and target token: 1.0.

The synchronous and pipelined window artifacts are canonically identical:
feature Jaccard, all-edge Jaccard, weighted Jaccard, top-256, sign agreement,
and target token are all 1.0, while signed normalized L1 is 0.0. Thus the
two-buffer overlap and change from 1,052-row to 526-row windows did not add
numerical drift.

`cuda_full` and windowed CUDA remain different bounded execution regimes. Their
direct comparison has feature Jaccard 0.999023914, all-edge Jaccard
0.988961265, weighted Jaccard 0.989621080, signed normalized L1 0.010433062,
and exact top-256/sign/token results.

## SP3 re-profile

After full residency removes row traffic, `cuda_full` Phase 4 contains:

- feature-batch execution: 145.28 seconds (76.2%);
- all refreshes: 26.16 seconds;
- partial influence: 19.96 seconds;
- normalization inside partial influence: 17.45 seconds (9.2%);
- direct accumulation: 0.98 seconds;
- row reads: 0.37 seconds;
- ranking: 0.25 seconds; and
- frontier planning: 0.40 seconds.

None of SP3's allowed non-batch targets clears its 10%-of-Phase-4 continuation
threshold. Normalization is close but below the threshold, while the dominant
remaining cost is feature-batch execution, which this plan intentionally keeps
out of the current non-batch ladder. SP3 therefore closes with no additional
implementation.

## Provenance and validation

Project:

- branch: `perf/exact-trace-loop`
- selectable profiles/knobs: `1eb82f7`
- SP2.2 planning commit: `772e015`

Sibling:

- selectable execution modes: `6324210`
- pinned double-buffer pipeline: `127a928`

Execution snapshots:

- selectable modes and synchronous/full comparison:
  `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/workspace_snapshots/workspace_20260729_170601_sp2_2_selectable_cuda_modes_v1`
- pinned double-buffer comparison:
  `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/workspace_snapshots/workspace_20260729_174716_sp2_2_pinned_double_buffer_v2`

All runs used Slurm allocation `1671648` on `grn032`, one NVIDIA H200, the
offline shared Hugging Face cache, and immutable project+sibling snapshots.
Cache state was warm/uncontrolled.

Validation:

- initial selectable-mode focused sibling suite: 60 passed;
- initial broader affected sibling suite: 88 passed;
- project profile/knob tests: 134 passed;
- double-buffer focused sibling suite: 70 passed;
- double-buffer broader affected sibling suite: 85 passed;
- path-sensitive Phase-4 resource architecture guard: passed from the sibling
  root; and
- affected-file Ruff checks: passed.

The apparent resource-architecture failure when run from the project root was
only the test's relative-path assumption; the same test passes from its
repository root. Pre-existing untracked patches and Slurm logs in the project
worktree were not modified or committed.

## Decision

This section records the original SP2.2 decision. The canonical policy was
subsequently clarified on 2026-07-31: a `bounded` pass permits code-retention
review but does not require retention. The modes below are therefore current
retention candidates pending comparative scaling/dominance review, not a
promise that every branch stays in runtime code.

Keep `cpu_exact` as the exact default. The measured roles for the other,
default-off bounded candidates are:

- use `cuda_full` for speed when the complete allocation and safety gate pass;
- use pipelined `cuda_windowed` when preserving about 3.9 GiB of HBM matters
  more than the remaining 110-second Phase-4 penalty;
- use `cpu_prepared` only when its bounded CPU arithmetic regime is acceptable;
  and
- use `auto` when capability-based full-to-window-to-exact fallback is desired
  and the resolved mode is recorded with the artifact.

No launch default or exact-promotion claim is made. The next informative
performance work is longer-prefix scaling with these modes, or SP4's formal
telemetry/lifecycle evidence; there is no justified SP3 non-batch
implementation at the current 12B short-prefix profile.
