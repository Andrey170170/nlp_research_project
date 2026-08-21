# SP1 active-CPU encoder exactness result

Date: 2026-07-31

Status: complete; retained exact candidate, promotion not yet selected

## Outcome

SP1 replaced the old active-CPU encoder path's GPU occurrence table with direct,
duplicate-aware host materialization and a bulk per-layer gather. The new path
removes the roughly 0.98 GiB temporary GPU occurrence table and reduces matched
12B Phase 4 from 389.50 seconds to 286.69 seconds. It does not pass
`strict_exact`: the matched full runs select two different features per arm and
execute 129 versus 130 semantic batches. It does pass the canonical `exact`
floor, including exact ranked support, signs, and target. Retain both placements
as selectable implementations. `active_cpu` is eligible for comparative
promotion, but the single 12B matched result is not by itself a reason to select
it over the broader-evidence mapped-lazy candidate.

## Implementation

The sibling runtime now:

- materializes only the duplicate-aware selected encoder rows directly on CPU;
- performs one bulk tensor fetch and advanced-index gather per layer rather
  than one host fetch per selected row;
- preserves requested Phase-0 and Phase-3 diagnostic captures when a
  transition probe intentionally returns no graph; and
- returns those captures as diagnostic artifacts for project-owned
  persistence.

The project harness now:

- defines active-CPU and lazy SP1 diagnostic profiles;
- persists transition-probe diagnostic sidecars and records their status in
  the step manifest; and
- permits the full SP1 profiles to compare against the scientific historical
  anchor while retaining their unchanged scenario knobs.

Implementation commits:

| Repository | Commit | Change |
|---|---|---|
| sibling | `5f543b2` | direct active encoder-row materialization on CPU |
| sibling | `d1fa822` | bulk host encoder-row gather |
| project | `6170509` | SP1 encoder diagnostic profiles |
| sibling | `2f63779` | retain diagnostic-probe captures |
| project | `0238415` | persist exact-trace probe sidecars |
| project | `eb93f1d` | enable full SP1 scientific comparisons |

## Diagnostic localization

The useful warm transition probes used the immutable v4 snapshot:

`/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/workspace_snapshots/workspace_20260731_171815_sp1-probe-artifacts-direct-host-v4`

| Placement | Run ID | Completion | Phase 0 | Phase 3 | Phase 4 | Probe batches |
|---|---|---:|---:|---:|---:|---:|
| direct active CPU | `sp1-active-cpu-direct-host-probe-v4` | 81.28s | 44.32s | 1.04s | 2.85s | 3 |
| mapped lazy | `sp1-lazy-reference-probe-v3` | 72.42s | 34.96s | 0.86s | 4.12s | 3 |

The active probe materialized 979,814,400 encoder bytes and completed the
three-batch Phase-4 probe with 189 rows. The persisted Phase-3 sidecars showed:

- byte-exact `phase3_feature_rows`;
- exact active features, activation values, target IDs/probabilities,
  duplicate-aware row hash, and pre/post frontier locality;
- a maximum absolute difference of 0.010731 in two row-absolute sums;
- 125,412 differing seed-feature influence entries, with maximum absolute
  difference `9.0245e-7`; and
- identical frontier membership and order through the first three Phase-4
  batches.

This localizes the first numerical difference after row selection, in the
floating reduction that creates seed-feature influences. Direct host
materialization repairs the earlier selected-frontier divergence seen with the
old GPU occurrence-table path, but it cannot make the downstream reduction
bit-identical without giving up the placement-specific computation.

Non-comparable setup attempts were retained only as engineering provenance:

- `sp1-active-cpu-direct-host-probe-v1` used an impractically small slice and
  was safely aborted;
- `sp1-active-cpu-direct-host-probe-v2` was a cold-cache observation
  (425.42s candidate, 395.11s completion, 359.26s Phase 0, approximately 96 GB
  of physical reads); and
- `sp1-active-cpu-direct-host-probe-v3` failed before model load because it
  inherited the wrong Hugging Face cache location and is excluded.

## Matched full 12B result

Both full runs used the immutable v5 snapshot:

`/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/workspace_snapshots/workspace_20260731_173012_sp1-full-direct-host-v5`

The snapshot contains project commit `eb93f1d` and sibling commit `2f63779`.
Both candidates completed successfully; their CLI exit status was nonzero only
because the separately declared historical-anchor comparison detected older
scientific-baseline drift.

| Placement | Run ID | Harness | Completion | Attribution | Phase 0 | Phase 3 | Phase 4 | Batches |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| direct active CPU | `sp1-active-cpu-direct-host-full-01` | 448.24s | 368.70s | 366.20s | 44.66s | 1.04s | 286.69s | 129 |
| mapped lazy | `sp1-lazy-current-full-01` | 487.75s | 457.30s | 454.81s | 34.43s | 1.04s | 389.50s | 130 |

Relative to the current matched lazy control, direct active CPU saved:

- 102.81 seconds, or 26.4%, in Phase 4;
- 88.60 seconds, or 19.4%, in completion time; and
- 39.51 seconds, or 8.1%, in outer harness time.

Completion and phase timing are the preferred comparison because setup and
post-run comparison work varied between arms. The direct active-CPU completion
is comfortably below ten minutes.

The current-control graph comparison is `exact`, but not `strict_exact`:

| Metric | Active CPU versus lazy |
|---|---:|
| Feature Jaccard | 0.9995118379301928 |
| All-edge Jaccard | 0.9995001249687578 |
| Weighted-edge Jaccard | 0.9995290793033402 |
| Top-64/128/256/512/1024 edge Jaccard | 1.0 |
| Signed normalized L1 | 0.0004710316058778251 |
| Magnitude normalized L1 | 0.0004710316058778251 |
| Sign agreement | 1.0 |
| Target token | exact (`Here`) |
| Semantic/execution batches | 129 versus 130 |

The feature sets share 8,190 of 8,192 features. Active CPU uniquely selects
features in layers 22 and 37; lazy uniquely selects features in layers 5 and
10. Five edge-support entries differ as a consequence of the differing
endpoints. Common-edge value deltas are tiny (maximum absolute delta about
`9.8e-8`) but nonzero.

## Decision

SP1 closes with a positive exact classification under the canonical taxonomy:

- `active_cpu`: retain as an exact, default-off promotion candidate with a
  substantial measured 12B speed and temporary-memory benefit;
- mapped lazy encoder materialization: retain as the broader-evidence exact
  SP5 input and atomic fallback;
- automatic promotion: permitted by exactness, but not yet selected; compare
  runtime, memory, prompt breadth, and LS scaling before changing defaults; and
- `strict_exact`: not claimed because selected support and semantic batch count
  differ.

No scientific baseline, fidelity authorization, governor bundle, or launch
default changed.

The 2026-07-31 code-retention clarification subsequently removed
`active_pinned_cpu` from the runtime and project profile surface. Its measured
520.60-second result was slower than `active_cpu`, used no less HBM, and offered
no stronger fidelity. Exact restoration patches are indexed under
`experiments/patches/exact_trace_performance_20260731/`.

## Validation

- sibling focused and affected suites: 160 passed; lint passed;
- diagnostic artifact changes: sibling 37 passed, project 16 passed; lint
  passed;
- full-profile scope change: project 2 passed; lint passed.
