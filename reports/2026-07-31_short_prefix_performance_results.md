# Exact-trace SP5 short-prefix results

Status: formal matrix executed; exact candidates admitted; LS0 complete

Date: 2026-07-31

## Outcome

The fast selective Phase-0 decoder-row path meets the canonical `exact` floor
over every measured short-prefix workload, including held-out 1B `94_base`,
and completes two 12B runs below ten minutes. It is currently retained and is
eligible for promotion consideration. Exactness is only the admission gate:
prompt-length scaling and comparative runtime/memory evidence still determine
whether it is selected for a default scope. The same selection rule applies to
code retention: `bounded` permits review but does not protect a dominated mode
from removal.

Accordingly:

- `sp5-canonical-phase0-reference-plt-v1` retains full-page Phase 0 as the
  `strict_exact` audit/reference path where its compact fields match;
- `sp5-selective-phase0-finalist-plt-v1` retains selective mapped rows as the
  faster `exact` candidate (historical profile aliases remain available);
- no launch default or scientific baseline changes;
- the approximately 470-second 12B result is exact for its measured workload
  scope; and
- LS1 is unblocked and must determine how far that scope extends with prefix
  length.

Fidelity names below follow
`plans/2026-07-31_exact_trace_canonical_fidelity_taxonomy.md`. They describe the
persisted compact-graph contract and do not prove equality of omitted raw
intermediates or the full mechanism.

## Immutable source

Final strict source snapshot:

`/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/workspace_snapshots/workspace_20260731_203817_sp5-strict-rebased-v5`

- project: `dfa9100` (profile split implementation at `f09d6c5`);
- sibling: `bfd804287d46a90cabd054d4da1732a6947703f8`;
- allocation: Slurm `1685189`, `grn029`, NVIDIA H200;
- cache: offline shared Hugging Face cache, warm/uncontrolled unless stated;
- host RSS stop: 64 GiB.

The earlier composed selective matrix used snapshot
`workspace_20260731_175923_sp5-exact-finalist-v1`. The cast-placement probe
used `workspace_20260731_193024_sp5-exact-cast-v2`.

## Fast selective Phase-0 matrix

All canonical `361_base` runs passed strict compact parity.

| Case | Harness seconds | Completion seconds | Phase 0 | Phase 4 | Result |
|---|---:|---:|---:|---:|---|
| 1B PLT r1/r2/r3 | 187.24 / 120.60 / 116.41 | 106.13 / 94.24 / 92.86 | 19.83 / 7.49 / 7.24 | 74.71 / 75.48 / 74.73 | exact on workload |
| 4B PLT r1/r2 | 216.76 / 201.30 | 192.85 / 178.16 | 34.01 / 18.51 | 139.79 / 140.91 | exact on workload |
| 12B PLT r1/r2 | 471.38 / 470.14 | 443.99 / 443.27 | 35.90 / 34.87 | 370.87 / 373.71 | exact on workload, below 10 min |
| 1B CLT | 95.36 | 72.06 | 7.15 | 36.62 | exact provider regression |

Prompt breadth distinguishes `exact` from `strict_exact`:

| 1B prompt | Harness | Completion | Phase 0 | Fidelity |
|---|---:|---:|---:|---|
| `828_base` | 101.60 | 76.43 | 7.35 | all strict metrics 1.0 |
| `94_base` | 104.13 | 79.82 | 7.49 | exact, not strict_exact: feature 0.999756, edge 0.997004, weighted 0.996279, signed L1 0.003728; Top-256/sign/token exact |

The initial `94_base` mismatch before the sibling cast adjustment was narrower
(feature Jaccard 0.999512 with edge values otherwise exact). Matching mapped
checkpoint casts to the provider device moved the cutoff and broadened the
retained-edge difference. The mechanism is therefore `exact`, not
`strict_exact`; no cast-fix claim is made.

On `94_base`, mapped selective telemetry requested 106.66 MB of raw checkpoint
rows and retained 53.33 MB of unique rows instead of traversing about 15.52 GB
of baseline decoder pages. That is why Phase 0 falls from roughly 122 seconds
cold to 7.49 seconds.

## Canonical full-page Phase-0 matrix

| Case | Harness | Completion | Phase 0 | Phase 4 | Fidelity |
|---|---:|---:|---:|---:|---|
| 1B `361_base` | 145.06 | 120.76 | 21.04 | 87.14 | exact |
| 1B `828_base` cold | 221.02 | 185.24 | 114.31 | 56.87 | exact |
| 1B `94_base` cold | 232.46 | 198.60 | 122.50 | 62.69 | exact after current-source control refresh |
| 4B `828_base` cold | 519.97 | 494.45 | 363.61 | 105.73 | exact |
| 4B `361_base` cold | 617.68 | 587.46 | 391.57 | 168.91 | exact, not strict_exact: feature 0.999024, edge 0.998401, weighted 0.998881, signed L1 0.001119 |
| 12B `361_base` cold | 2025.76 | 1905.57 | 1370.02 | 476.53 | exact graph; measurement-only gate status |

The 12B strict run loaded 95.32 GB of logical decoder pages in Phase 0. Direct
process I/O sampling observed roughly 400 GB of physical reads over the full
traversal. Peak CUDA reservation was 38.82 GiB, peak process RSS was 23.81 GiB,
and the cgroup reached its 250 GiB file-cache ceiling without tripping the
64-GiB process-RSS guard.

The repeated 1B `94_base` canonical artifacts from strict r1/r2 were
byte-identical in every NPZ field. The prior control differed by four selected
features total; all four were isolated with zero incident retained edges and
zero retained weight. The same-regime 94 control was refreshed from the
current-source artifact in commit `dfa9100`; the frozen scientific registry was
not changed. A subsequent cold r3 passed every strict metric.

## Interpretation

The experiments reject a simple global rule that “GPU/selective is bounded,
CPU/full-page is exact.” Phase-0 decoder reconstruction, seed capture, cutoff
selection, and the source revision together define the numerical regime:

- selective rows can be byte-equivalent on one prompt/model and drift on
  another;
- canonical current-source execution can disagree with an older same-regime
  artifact at 4B; and
- near-cutoff differences can be either isolated feature ties or materially
  retained edge changes.

The next task is LS1 prefix scaling. It should compare currently retained selective and
canonical/placement candidates at frozen 129/256/512/1,024-token workloads,
classify each result independently, and choose among exact candidates using
runtime, memory, provider admission, and prompt-length scope. It should also
decide whether each bounded execution mode owns a non-dominated scaling or
resource region; modes that do not should move to the indexed patch archive.
Phase-0
first-divergence localization remains useful for improving `strict_exact`, but
no longer blocks canonical `exact` scaling work.

## LS0

LS0 is complete, and the canonical taxonomy now unblocks LS1. The frozen
campaign manifest contains development prefixes at 129, 256, 512, and 1,024
tokens from one deterministic 1,329-token trajectory, plus two held-out
256-token workloads. Every workload records prompt, trajectory, prefix, and
target hashes. `list-campaign --require-frozen` passes without model loading.

## Validation and artifacts

- project profile tests: 7 focused passed during the split and control refresh;
- sibling mapped-row tests: 7 focused passed; Ruff passed;
- earlier full focused project suite: 133 passed; Ruff passed;
- no launch defaults or frozen scientific baselines changed.

Primary run roots are under
`/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/sweep/performance_optimization/`
with run IDs used in the tables, notably `sp5-exact-1b-361-strict-r1`,
`sp5-exact-1b-plt-breadth-strict-r3`, `sp5-exact-4b-plt-828-strict-r1`,
`sp5-exact-4b-361-strict-r1`, and `sp5-exact-12b-361-strict-r1`.
