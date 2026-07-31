# SP4 intermediate results: telemetry overhead

Date: 2026-07-29; updated 2026-07-31

Status: SP4.1 gate passed; SP4.2 is next

## Scope

SP4.1 compares the same mapped-lazy exact 4B PLT workload with telemetry
recording disabled and enabled. Both profiles retain attribution profiling so
the comparison changes the recorder/sampling mechanism, not the traced graph.
The preregistered gate is less than 2% completion-time overhead with exact
signed compact-graph parity.

All formal runs used Slurm job 1671648 on Granite node `grn032` with one H200,
the shared offline Hugging Face cache, and immutable project+sibling workspace
snapshots.

## Formal measurements

| Pair | Order | Telemetry off completion | Telemetry on completion | Overhead |
|---|---|---:|---:|---:|
| v3 | off, on | 163.614817 s | 179.769358 s | 9.873% |
| v4 | on, off | 157.858818 s | 178.782410 s | 13.254% |

Every run passed the exact signed compact gate:

- feature, all-edge, top-256 edge, and weighted-edge Jaccard: 1.0;
- shared-edge sign agreement: 1.0;
- magnitude and signed normalized L1 deviation: 0.0; and
- target-token match: 1.

The reversed v4 pair is the stronger order-control result. Its telemetry-on
profile recorded Phase 0 at 19.770 s, Phase 3 at 0.810 s, Phase 4 at 138.770 s,
and attribution at 176.545 s. The disabled observer intentionally omits those
internal phase fields; its completion time remains available and is the SP4.1
gate metric.

## Sparse resource sampling experiment

Sibling commit `9a6cd33` retains a timing event for every batch but takes the
heavy resource snapshot only for Phase-4 batches 1-3, every 32nd batch, and the
final batch. The 65-batch workload therefore takes six Phase-4 resource
samples instead of 65. Non-Phase-4 batches retain full sampling.

Focused validation passed:

- `21 passed` for the new sampling policy plus Phase-3/Phase-4 runtime tests;
- source diff check passed; and
- the v4 formal pair passed exact parity.

Those runs did not close the overhead gate. Telemetry-on moved from 179.769 s
in v3 to 178.782 s in v4, only 0.55%, while the v4 sidecar still contained
9,822 fine-grained events. This motivated a unified resource-sampling and
bounded-sink implementation, followed by a new reversed pair.

## Unified sampling and bounded-sink result

Sibling commit `c78ab46` applies the same first-three/every-32/final policy to
the inner compute batch, outer Phase-4 feature batch, and refresh resource
intervals. It also buffers the JSONL sink, flushes every 64 events and at
phase/run boundaries, and reports a maximum 63-event crash-loss window.

The recorder replay microbenchmark reduced 9,822-event sink flushes from 9,822
to 154. Median replay time changed from 0.148 s to 0.106 s on node-local
storage and from 0.292 s to 0.259 s on shared VAST. This established that sink
flushing was a secondary cost, but bounded buffering was retained for its
durability/performance contract.

The formal v5 reversed pair used Slurm job 1685189 on Granite node `grn029`,
one H200, and immutable snapshot
`workspace_20260731_153216_sp4_unified_telemetry_v5`:

| Pair | Order | Telemetry off completion | Telemetry on completion | Per-order delta |
|---|---|---:|---:|---:|
| v5 pair 3 | off, on | 204.085826 s | 176.572635 s | -13.481% |
| v5 pair 4 | on, off | 167.638877 s | 192.993689 s | +15.124% |
| paired mean | reversed | 185.862352 s | 184.783162 s | **-0.581%** |

The large, opposite single-order deltas show cold/runtime variance and must not
be interpreted individually. The preregistered reversed-pair mean is the gate
metric: measured telemetry overhead is -0.581%, below the 2% ceiling. All four
runs passed exact signed compact parity. Both telemetry-on sidecars recorded
9,835 events with 171 bounded flushes, a 64-event maximum pending window, zero
sink errors, and sampled Phase-4 feature batches 1-3, 32, 64, and 65.

SP4.1 therefore passes. The implementation does not claim telemetry makes the
workload faster; the small negative estimate is noise around zero overhead.

## Provenance

- v3 snapshot:
  `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/workspace_snapshots/workspace_20260729_205723_sp4_formal_evidence_v3`
- v4 snapshot:
  `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/workspace_snapshots/workspace_20260729_211457_sp4_sparse_telemetry_v4`
- v3 run IDs:
  `sp4-telemetry-pair1-off-v3`, `sp4-telemetry-pair1-on-v3`
- v4 run IDs:
  `sp4-telemetry-sparse-pair2-on-v4`,
  `sp4-telemetry-sparse-pair2-off-v4`
- v4 project commit: `4698c29`
- v4 sibling commit: `9a6cd33`
- v5 snapshot:
  `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/workspace_snapshots/workspace_20260731_153216_sp4_unified_telemetry_v5`
- v5 run IDs:
  `sp4-telemetry-unified-pair3-off-v5`,
  `sp4-telemetry-unified-pair3-on-v5`,
  `sp4-telemetry-unified-pair4-on-v5`, and
  `sp4-telemetry-unified-pair4-off-v5`
- v5 project commit: `5199fd5`
- v5 sibling commit: `c78ab46`

The run artifacts are under
`/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/sweep/performance_optimization`.

## Restart point

Continue with SP4.2 from a genuinely job-private staged 4B transcoder cache,
using matched cold first-three-batch and warm steady-state probes. Preserve the
accepted SP4.1 sampling and bounded-sink policy.

After SP4 closes, proceed in the user-selected order: open-ended SP1, then
SP5/LS0 including deterministic trajectory generation.
