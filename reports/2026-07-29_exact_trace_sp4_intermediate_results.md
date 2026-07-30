# SP4 intermediate results: telemetry overhead

Date: 2026-07-29

Status: SP4.1 gate failed; SP4.2 not yet executed

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

The change did not close the overhead gate. Telemetry-on moved from 179.769 s
in v3 to 178.782 s in v4, only 0.55%, while the v4 sidecar still contains 9,822
fine-grained events. Cross-run cache/order effects prevent treating that
0.55% as an isolated speedup, but the persistent 13.254% paired overhead
rejects the hypothesis that per-batch resource snapshots were the principal
cost. The next seam is event serialization and the line-flushed incremental
JSONL sink, while retaining crash-useful incremental durability.

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

The run artifacts are under
`/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/sweep/performance_optimization`.

## Restart point

Continue SP4.1 before SP4.2:

1. preserve the current per-batch event schema and sparse resource-sampling
   policy;
2. add bounded buffering to the incremental JSONL sink (flush periodically
   and at terminal/close events) and record the maximum crash-loss window;
3. run a focused recorder microbenchmark, then a fresh reversed 4B exact pair;
4. accept SP4.1 only if paired completion overhead is below 2%;
5. run SP4.2 from a genuinely job-private staged 4B transcoder cache with
   matched cold first-three-batch and warm steady-state probes.

After SP4 closes, proceed in the user-selected order: open-ended SP1, then
SP5/LS0 including deterministic trajectory generation.
