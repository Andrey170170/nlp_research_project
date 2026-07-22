# Governor Calibration Run Plan

Status: Wave A, static Phase-4 correction, and Wave B closed; observation ingestion active
Last updated: 2026-07-21

This campaign supplies training and validation observations for a staged
constrained optimizer. It measures the joint resource, runtime, and fidelity
response above and below historical settings. It does not divide rows into
globally accepted and rejected governor configurations.

Each row is useful in one or more response families. Exact repeats estimate
noise and certify narrow invariant scopes. Numerically sensitive and semantic
contrasts estimate fidelity loss. OOM/refusal constrain feasibility, timeout
constrains runtime from below, and infrastructure failures carry provenance but
no scientific response value.

The request chooses the policy: `exact` requires exact or scope-certified
candidates; `bounded` enforces conservative metric floors; `best_effort`
treats fidelity loss as a soft penalty; and `research` permits safe
extrapolation while labeling unsupported predictions unknown.

All runs use one Granite H200, fp32 exact tracing, `361_base`, one trace,
incremental telemetry, verbose profiling, compact graph comparison, and one
immutable project+sibling snapshot per shared code state. Array concurrency is
one so cache order and resource use remain inspectable. Every row, including
reference rows, compares independently against the pinned corrected-hook
registry in `experiments/baselines/governor_calibration_granite_20260719.json`;
there is no Slurm array-order dependency. `mode=metrics` keeps graph drift
advisory, while a missing baseline or structurally incomplete comparison is a
job failure.

The executable source is
`src/nlp_research_project/exact_trace_bench/scenarios/governor_calibration.py`.
Generate Wave A with:

```bash
UV_CACHE_DIR=.uv-cache uv run python experiments/build_governor_calibration_configs.py
```

Wave A deliberately sets `governor_admission_mode=advisory`. This is a
calibration-only force override: every admission checkpoint and resource-budget
violation is still computed and recorded, but a governor refusal does not stop
the trace. Ordinary launch policy remains `enforce`; CUDA OOMs, OS/cgroup kills,
provider or structural validation failures, and arbitrary runtime errors remain
terminal in both modes.

## Wave A - 1B Upward Search

Wave A contains 36 runs: 17 CLT and 19 PLT. Historical references are controls,
not ceilings.

### 1B CLT, 17 rows

1. Coupled logical batches: `1000`, `1500`, `2000`, `3000`, `4096`.
2. Decoder-fetch chunks: `8192`, `10080` against reference `4096`.
3. Strict decoder caches: `0`, `16`, `32 GiB` against reference `8 GiB`.
4. Coupled corners: `(b2000,c8192,cache16)`,
   `(b3000,c8192,cache32)`, `(b4096,c10080,cache32)`.
5. At batch `1500`, isolated source, feature, and logit semantic rows.
6. One exact reference repeat.

### 1B PLT, 19 rows

1. Coupled logical batches: `128`, `256`, `512`, `768`, `1024`, `1536`.
2. Decoder-fetch chunks: `8192`, `16384`, `32768` against reference
   `4096`.
3. Strict decoder-cache proof: `4 GiB` against reference zero.
4. Coupled corners: `(b512,c8192)`, `(b1024,c16384)`,
   `(b1536,c32768)`.
5. At batch `256`, isolated source, feature, and logit semantic rows.
6. Research refresh cadences `2` and `8` against reference `4`.
7. One exact reference repeat.

Logical-batch, isolated-axis, and refresh rows declare
`governor_fidelity_mode=research` plus the exact `PlanningWorkload` fields they
override. The current governor contract still represents fetch chunk as a
physical requirement, but Wave A shows that non-reference PLT chunks introduce
small numerical graph drift. Classify fetch and physical execution grouping as
numerically sensitive until the coupling is removed. Decoder cache remains an
exact axis. All observations remain available to the response models; fidelity
policy controls runtime eligibility.

### Wave A outcome

| Provider/case | Speedup | Peak reserved | Feature J | Edge J | Weighted J | Top-256 J | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| CLT `c10080` | 1.43x | 89.34 GiB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | useful fetch candidate |
| CLT `b1500` | 1.05x | 132.98 GiB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | memory knee; do not transfer |
| PLT `b256` | 1.68x | 24.67 GiB | 0.9932 | 0.9946 | 0.9948 | 0.9845 | primary batch candidate |
| PLT `b512` | 2.96x | 46.57 GiB | 0.9697 | 0.9784 | 0.9784 | 0.9542 | upper drift probe only |
| PLT `c16384` | 2.64x | 13.30 GiB | 0.9978 | 0.9996 | 0.9997 | 1.0000 | opt-in fetch candidate |
| PLT `c32768` | 4.01x | 13.30 GiB | 0.9973 | 0.9972 | 0.9975 | 1.0000 | fastest row above all floors |
| PLT `b512/c8192` | 5.08x | 46.57 GiB | 0.9766 | 0.9833 | 0.9829 | 0.9617 | misses Top-256 floor |

CLT completed 11 of 17 rows; all `b2000+` logical and coupled rows OOMed. PLT
completed all 19 rows. The provider-local reference rows match the original
corrected-hook Granite artifacts. Wave A bounds the next search but does not fit
or promote governor coefficients: loaded-state walltime estimates remain several
times too conservative, and peak-VRAM estimates are about 9% high for CLT and
31% high at PLT `b1536`.

Do not rerun the complete Wave A matrix after the static Phase-4 coalescing
correction. Preserve its coupled PLT batch rows as measurements of logical
batch semantics and stress behavior, but do not confuse them with isolated
physical Phase-4 execution cost. The immutable `128/256/512` static gate
is the corrected physical supplement: it held the canonical `128x4` semantic
schedule fixed, preserved prepared frontiers and compact topology, and selected
256 execution rows as the current 1B PLT efficiency knee. Decoder-fetch rows,
cache rows, CLT bounds, and explicit semantic rows remain valid for their stated
factors.

### Phase-4 feedback diagnostic

Before Wave B, run a five-row 1B PLT causal diagnostic to separate frontier
feedback cadence from physical batch numerics. Keep source/logit semantics at
`128`, decoder chunk at `4096`, decoder cache at zero, update policy standard,
and all other tracing semantics fixed.

| Row | Logical feature batch | Refresh interval | Frontier window | Session | Phase-4 microbatch | Contrast |
|---|---:|---:|---:|---:|---:|---|
| canonical reference | 128 | 4 | 512 | 128 | 128 | corrected-hook control |
| constant-window logical | 256 | 2 | 512 | 128 | 128 | logical grouping at fixed feedback, session, and execution |
| session control | 256 | 2 | 512 | 256 | 128 | session capacity only |
| constant-window physical | 256 | 2 | 512 | 256 | 256 | physical matmul/backtrace shape only |
| stale-window | 256 | 4 | 1024 | 256 | 128 | feedback staleness only |

Jobs `1639225` and replacement `1639431` closed the diagnostic. Evaluate the
paired contrasts rather than treating the rows as independent scores:

| Contrast | Speedup | Feature J | Edge J | Weighted J | Top-64 J | Top-256 J | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| logical grouping `128x4 -> 256x2` | 1.18x | 0.99367 | 0.99521 | 0.99387 | 0.93939 | 0.98450 | semantic control |
| session capacity `128 -> 256` | 0.95x observed | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | physical control |
| Phase-4 microbatch `128 -> 256` | 1.39x | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | scalable physical control |
| frontier window `512 -> 1024` | 1.21x | 0.99464 | 0.99203 | 0.99243 | 1.00000 | 1.00000 | semantic control |

The physical-microbatch comparison differs only below compact graph precision
(`weighted J=0.9999999979`); the session comparison is exactly equal. Logical
grouping explains the canonical-to-`b256` graph change even when the aggregate
frontier window remains 512, while a larger frontier window introduces separate
tail-graph drift. The fastest tested constant-window configuration is 1.55x the
canonical runtime, with all of its measured drift attributable to logical
grouping rather than session or physical microbatch size.

Governor candidate generation must therefore classify logical feature grouping
and frontier feedback window as fidelity-controlled semantic choices. Session
capacity and Phase-4 compute microbatch remain resource-optimized physical
choices. The comparison bundle is under
`granite/analysis/governor_calibration/phase4_feedback_diagnostic_20260719`
inside the exact-trace VAST root.

### Stop Rules

Stop increasing a dimension after any hard condition:

- governor admission refusal or CUDA OOM;
- peak allocated/reserved VRAM reaches 90% of the usable envelope;
- the research graph falls below `0.97` feature, edge, weighted-edge, or Top-256
  Jaccard against the corrected-hook Granite reference.

Treat these as soft knees and inspect before continuing:

- peak VRAM reaches 85%;
- two consecutive increases each improve trace runtime by less than 3%;
- GPU utilization plateaus while memory/cache traffic grows;
- host RSS, local spill, or decoder-cache pressure becomes the active bound.

Admission refusal is a valid calibration observation. In Wave A it is recorded
without terminating the trace so the deliberately forced configuration can
produce measured evidence. It identifies a safety
boundary; it must remain distinguishable from infrastructure failure in the
run records.

## Wave B - Larger-Model Transfer

Wave B transfers the corrected independent-control design rather than the old
coupled logical-batch ladder. Canonical source, feature, and logit batches stay
fixed at 128 for 4B and 64 for 12B. The canonical refresh stride remains four;
Phase-1 and Phase-3 widths remain fixed. Physical-envelope rows raise
session capacity and `phase4_execution_batch_max_rows` together without changing
the prepared semantic batches or crossing a prepared refresh frontier. Wave B
tests transferable fitting configurations; Wave C later isolates the session
and execution cost terms.

| Provider | Canonical semantic batch | Session/execution envelope | Fetch chunks | Host RAM | Walltime |
|---|---:|---|---|---:|---:|
| Gemma 3 4B PLT | `128` | `128, 256, 512` | `4096, 16384, 32768` | 400 GiB | 2h |
| Gemma 3 12B PLT | `64` | `64, 128, 256` | `4096, 16384, 32768` | 600 GiB | 8h |

Each model has seven causal rows:

1. exact canonical reference;
2. middle session/Phase-4 execution envelope;
3. upper session/Phase-4 execution envelope;
4. isolated opt-in decoder fetch chunk 16384 at reference execution capacity;
5. isolated opt-in decoder fetch chunk 32768 at reference execution capacity;
6. coupled middle execution capacity plus decoder fetch chunk 32768;
7. one semantic-transfer row that doubles logical feature grouping, halves the
   refresh stride to preserve the aggregate frontier window, and uses the
   middle physical execution capacity.

The two non-reference fetch chunks and the semantic-transfer row are explicit
research probes. Do not run a Cartesian product. The bounded matrix runs all
seven independent rows; VRAM, graph-floor, and diminishing-return conditions
are campaign continuation checks rather than global solver eligibility rules.
An upper execution row is headroom and response-surface evidence, not a
prospective default by itself.

Launch Wave B only through `exact-trace-bench launch-plan` with immutable
workspace mode. The shared Slurm template defaults are not the campaign
resources; the rendered command must carry the scenario metadata's regular RAI
QOS, `400G/2h` or `600G/8h`, 12 CPUs, H200 GRES, and serialized array range.
Run `sbatch --test-only` on that exact rendered command before submission.

The first corrected Wave B launch uses snapshot
`workspace_20260720_050003_governor-wave-b-20260720`. Granite arrays
`1642077` (4B) and `1642078` (12B) each contain seven rows serialized with
`%1`; both were pending for ordinary priority at launch rather than a policy
or resource-limit reason.

### Wave B outcome

All 14 tasks completed successfully with complete compact artifacts,
incremental telemetry, and GPU-monitor logs. Performance transfer is strong and
the rows reveal scale-dependent numerical sensitivity:

| Model/case | Speedup | Peak reserved | Feature J | Edge J | Weighted J | Prepared-frontier contract |
|---|---:|---:|---:|---:|---:|---|
| 4B execution `128 -> 256` | 1.52x | 42.95 GiB | 1.00000 | 1.00000 | 0.99999999 | failed at 2/17 refreshes |
| 4B execution `128 -> 512` | 1.60x | 77.64 GiB | 0.99902 | 0.99840 | 0.99888 | failed at 6/17 refreshes |
| 12B execution `64 -> 128` | 1.70x | 54.41 GiB | 0.99976 | 0.98946 | 0.99009 | failed at 17/33 refreshes |
| 12B execution `64 -> 256` | 2.14x | 83.22 GiB | 0.99902 | 0.99950 | 0.99953 | failed at 13/33 refreshes |

The semantic batch sizes and refresh count remained canonical, but numerical
differences from larger physical execution groups changed later frontier
membership. The 12B `256` arm also produced 130 semantic batches versus 129 in
the reference after the frontier diverged. Therefore none of the larger-model
execution envelopes is certified for exact mode by the existing
prepared-frontier contract, even though 4B `256` has effectively exact final
compact output. The 1B `256` exact certification remains limited to its recorded
1B PLT scope. All larger-model rows remain valid runtime, resource, and fidelity
observations for bounded/best-effort/research planning.

The opt-in chunk rows remain useful research evidence. At 4B, `c16384` and
`c32768` reached 2.49x and 3.76x with weighted-edge Jaccard 0.99667 and 0.98508.
At 12B they reached 2.94x and 4.11x with weighted-edge Jaccard 0.98977 and
0.98957. Coupling the middle execution envelope with `c32768` reached 4.59x at
4B and 4.23x at 12B. Do not infer a cross-model exact rung. Fit these as
scope-labeled observations with conservative uncertainty; bounded requests may
use them only when predicted metric floors pass, while best-effort/research can
surface the speed/fidelity tradeoff explicitly.

## Wave C - Local Mechanism Fit

Wave C fits the remaining physical cost terms around measured model-local knees.
It samples both sides of each knee and reserves held-out rows so the response
models learn local shape and uncertainty instead of one promoted setting:

- session, Phase-1, Phase-3, and Phase-4 widths at lower/selected/higher rungs;
- cache neighbors around the selected size;
- replay windows `2/4/8` and prefetch depths `0/2/4`;
- full-file, tiled, and recompute row policies;
- lazy/eager encoder residency and local/scratch spill where admissible;
- reference repeats for run-noise estimation.

These are local causal contrasts, not another broad grid. Hold all unrelated
controls at the selected configuration and reserve held-out rows for checking
the fitted model.

## Required Measurements

Every row must record:

- all planning epochs, candidate scores, hard constraints, selected values,
  support classification, and refusal reasons;
- phase elapsed time and operation/batch counts;
- phase-local CUDA peak allocated/reserved bytes and end allocation;
- GPU utilization and memory telemetry over time;
- host RSS plus detected cgroup/allocation budget;
- decoder/replay-cache hits, misses, evictions, and bytes;
- row-store bytes read/written and spill placement;
- semantic and execution fingerprints;
- compact graph parity against the original corrected-hook Granite artifacts.

Every row also emits one normalized calibration observation containing:

- outcome class (`completed`, `refused`, `oom`, `timeout`, or
  `infrastructure_failure`) and censoring semantics;
- hardware, provider/checkpoint/hooks/dtype, both code snapshots, workload, and
  reference artifact identity;
- complete requested/selected decision vectors and changed-axis classes;
- phase/runtime, CUDA/host/disk/cache observations and uncertainty;
- graph metrics, frontier diagnostics, confidence/support classification, and
  pointers to source telemetry/comparison artifacts.

Campaign-reference comparison is automatic when a reference is configured.
The observation must join graph metrics, lifecycle telemetry, Slurm accounting,
and GPU-monitor summaries without depending on array order or parsing human
logs.

The resource envelope walltime must be capped to the live Slurm allocation's
remaining time. Scenario walltime and `timeout_minutes` must match the submitted
allocation before launch; neither may promise time the job does not own.

## Ingestion, authorization, and defaults

These are separate operations:

1. **Ingest observations.** Preserve every scientifically interpretable row as
   resource/runtime/fidelity evidence, including censored outcomes.
2. **Publish a response bundle.** Fit conservative models, record support and
   held-out diagnostics, and content-address the exact observation set.
3. **Authorize fidelity scope.** Exact certification and bounded metric budgets
   are reviewed claims tied to provider/model/hardware/code scope. Observations
   alone do not broaden them.
4. **Change defaults.** A launch default changes only after held-out validation
   and an explicit project decision. Solver evidence does not silently mutate
   scenarios.

Values inside implementation safety limits may be explored outside calibrated
support only in research mode and must be labeled extrapolated/unknown. The MVP
predictor uses deterministic nearest-supported evidence and conservative bounds;
hierarchical transfer and learned interaction models are deferred until Wave C
contains enough held-out data.
