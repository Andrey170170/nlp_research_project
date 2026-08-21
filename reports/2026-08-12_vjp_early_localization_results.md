# VJP early-localization result and rerun design

Status: corrected Phase-3 factorial complete and classified; repair experiments
remain gated
Date: 2026-08-13

## Completed evidence

The corrected three-arm 4B/256 transition probes completed successfully:

| Arm | Job | Effective backward path | Trace seconds | Peak HBM MiB |
|---|---:|---|---:|---:|
| control | `1789268` | `logical_capacity+nnsight_injected`, 128 lanes | 50.51 | 34,443 |
| serial | `1789266` | `single_lane+autograd_serial`, 1 lane | 51.34 | 14,597 |
| batched | `1789265` | `single_lane+autograd_batched`, 1 lane | 63.91 | 14,597 |

All three jobs ended `COMPLETED 0:0`, resolved required `cuda_windowed`,
completed one Phase-4 execution batch, and recorded complete telemetry without
fallback or dropped events. The single-lane paths reduced measured peak HBM by
57.6 percent relative to the wide injected control.

The serial and batched paths selected the same Phase-3 frontier order and
membership and the same first Phase-4 scheduler batch. Their frontier cutoff
scores nevertheless differed by about 46.2 ppm, so this is not evidence of
numeric equality. The control differed from both single-lane paths before the
first Phase-4 execution batch. This localizes the observed discrete divergence
to Phase 3 or earlier and makes Phase-4 grouping an implausible first cause.

## Evidence loss and claim limit

The jobs requested Phase-0 donor and Phase-3 gradient, row, and seed captures,
but the full-answer probe branch persisted telemetry only and discarded the
diagnostic artifacts returned in `TraceResult.output`. No requested NPZ bundle
survived. Consequently these runs cannot distinguish Phase-0 state, target
state, Phase-3 gradient production, row contraction/normalization, or seed
ranking numerically. They are valid mechanism/resource probes and partial
localization evidence, not a completed fidelity diagnosis.

The persistence path is now repaired as a shared result contract. Full-answer
and canonical runtime completion use the same schema-specific, atomic capture
codecs and persist a structured requested/written/missing/failed report. A
successful probe is recorded only after every requested capture is losslessly
written and reopened successfully; missing, truncated, empty, or failed
requested captures fail closed. Phase-0 BF16 bytes and the Phase-3 gradient-v2
layout fields are preserved. The Slurm post-run validator independently checks
the report, request flags, paths, schemas, and nonempty files.

## Corrected diagnostic

The next experiment stops immediately after Phase 3 and uses four arms:

1. wide injected: logical-capacity graph and NNSight injection at capacity 128;
2. narrow injected: the same injected primitive at capacity and graph width 1;
3. narrow serial: single-lane direct serial autograd at capacity 1;
4. narrow batched: single-lane batched autograd at capacity 1.

This yields three scoped contrasts:

- wide injected versus narrow injected changes graph width/capacity while
  retaining the injected primitive;
- narrow injected versus narrow serial compares injected versus direct
  autograd at one physical lane, but still changes the graph-policy and
  execution stack and therefore remains a composite contrast;
- narrow serial versus narrow batched isolates `is_grads_batched` conditional
  on the shared single-lane direct-autograd implementation.

The comparison order is Phase-0 support and values, Phase-1 target state,
per-layer Phase-3 gradients, normalized Phase-3 row and denominator, then seed
influences and frontier. Stop at the first divergent boundary. No Phase-4,
performance, full-fidelity, default-promotion, or scaling-ladder claim follows
from this diagnostic alone.

Canonical inputs remain the frozen 4B/256 development point from
`361_length_v1`, with required `cuda_windowed`, one H200 per arm,
`measure_only`, deterministic file-cache preheat, and a one-hour scheduler
ceiling. Timing is operational telemetry only because cache placement and arm
order are not matched for performance inference.

## Corrected factorial result

The four Phase-3 probes were prepared from one read-only project+sibling
snapshot and passed prepared-workload validation before submission:

| Arm | Job | Frozen mechanism | Physical caps |
|---|---:|---|---:|
| wide injected | `1791882` | `logical_capacity+nnsight_injected` | 128 |
| narrow injected | `1791883` | `logical_capacity+nnsight_injected` | 1 |
| narrow serial | `1791884` | `single_lane+autograd_serial` | 1 |
| narrow batched | `1791885` | `single_lane+autograd_batched` | 1 |

All four jobs completed with exit `0:0` under `rai-gpu-grn`. Each resolved the
required `cuda_windowed` mode and the prepared engine, graph, kernel, and lane
count. Each wrote and reopened all four requested diagnostic bundles, reported
`probe_completed`, and executed zero Phase-4 batches. They used one H200, 32
CPUs, 400G, and a one-hour limit. The immutable snapshot is
`workspace_20260812_212814_ls4-4b-vjp-early-phase3-factorial-v2-20260812`
with snapshot-manifest SHA-256
`ed3c57bac54e0e07cbb23c0a1060e8c469ebdde1e1ee427449ae5380d547f11d`.

| Arm | Trace seconds | Peak sampled HBM MiB | Captures complete |
|---|---:|---:|---|
| wide injected | 70.44 | 34,443 | yes |
| narrow injected | 65.83 | 14,661 | yes |
| narrow serial | 58.47 | 14,597 | yes |
| narrow batched | 70.46 | 14,597 | yes |

The width-one arms reduce sampled HBM by 57.4--57.6 percent relative to wide
injected. This confirms the physical memory result at the Phase-3 boundary.
It is not a throughput result: the jobs ran on different nodes and cache
states, while diagnostic bundle construction and persistence occupied a large
fraction of the trace. Host-memory sampling was disabled, so this campaign
does not establish host-memory usage or a smaller RAM request.

### Wide injected versus narrow injected

Input tokens, target token, Phase-0 active support and BF16 activation values,
and Phase-1 target logits/probabilities are raw-exact. The first persisted
divergence is the Phase-3 gradient:

- all 34 layer-gradient payloads differ;
- gradient symmetric normalized L1 is `0.0262703`;
- feature-row symmetric normalized L1 is `0.0242801`;
- seed-influence symmetric normalized L1 is `0.0212439`;
- the pre-locality Top-64 set is exact, while Top-1,024 overlap is
  `1013/1024` with Jaccard `0.978744`;
- the post-locality Top-64 overlap is `63/64`, and Top-1,024 overlap remains
  `1013/1024`.

This strongly locates the large full-graph fidelity failure to width-sensitive
execution at or before Phase-3 gradient production. It does not yet name a
single operation: duplicated graph lanes, session/backward capacity, and the
actual Phase-1 trace width are structurally coupled in this path and all move
from 128 to 1, even though the NNSight-injected VJP primitive is unchanged. A
repeated matched pair is still required to
exclude run-to-run CUDA variability before treating the measured magnitude as
stable.

### Singleton contrasts

Narrow injected versus narrow serial, and narrow serial versus narrow batched,
have raw-exact Phase-3 gradients and active-feature rows. Their first persisted
difference is the nonfeature error-column L1 contribution, followed by the
total row denominator and seed influence:

| Contrast | Error-column symmetric L1 | Row-denominator symmetric L1 | Seed symmetric L1 | Frontier |
|---|---:|---:|---:|---|
| narrow injected vs narrow serial | `3.67346e-4` | `5.39847e-5` | `5.39952e-5` | exact order and membership |
| narrow serial vs narrow batched | `2.75880e-4` | `4.04665e-5` | `4.04986e-5` | exact order and membership |

Thus `is_grads_batched=True` is not the cause of the earlier large bounded
failure at singleton width. This does not validate multi-vector batched VJP:
the contrast contains only one physical lane. The residual 40--54 ppm
denominator/seed drift occurs after the captured active-feature rows and may
include ordinary run-to-run GPU reduction variability.

The initial comparator skipped `error_abs_sums` and `row_abs_sums` in its
declared checkpoint order and therefore mislabeled this residual as first
appearing in seed influence. The implementation now requires and checks those
arrays before the seed checkpoint; focused validation passes. Full integration
validation is part of the next campaign-preparation gate.

## Decision and next gates

The one-lane mechanism remains blocked for 4B/512 and the scaling ladder. The
result supports keeping batched autograd as a viable memory mechanism, but the
current narrow graph is not a bounded-fidelity replacement for the canonical
wide path.

Proceed in this order:

1. run two new repeats of each injected endpoint in one preheated allocation,
   counterbalanced as `128, 1, 1, 128`;
2. advance only if within-width repeat spread is much smaller than the endpoint
   effect and the effect repeats across all 34 gradient layers;
3. then run middle widths in cache-counterbalanced order
   `2, 64, 4, 32, 8, 16` on the same frozen 4B/256 Phase-3 probe;
4. use the new signed per-layer error/token captures to localize the small
   singleton denominator discrepancy;
5. if width itself is causal, prototype a general memory mechanism that keeps
   the canonical wide numerical path, prioritizing checkpointed recomputation
   and selective saved-tensor offload over workload-specific caps;
6. require repaired full 4B/256 fidelity, then 4B/512 fidelity and resource
   admission, before retrying 4B/1,024 or entering the 12B ladder.

No default, baseline registry, fidelity scope, or scaling-rung status changes
from this diagnostic.

## Endpoint-repeat launch

Job `1794733` was submitted pending on Granite for the first gate only. It uses
one H200, 32 CPUs, 400G, and one hour, with counterbalanced order
`128, 1, 1, 128` after one deterministic preheat. Every entry is a separate
immutable prepared workload with required `cuda_windowed`,
`logical_capacity+nnsight_injected`, Phase-3 stop, all four diagnostic bundles,
and `measure_only` resource policy. A content-addressed sequence runner
preflights all entries before creating outputs and applies the mechanism and
capture validator after every trace. The six middle widths remain unlaunched
until the endpoint repeatability gate passes.
