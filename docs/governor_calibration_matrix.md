# Governor Calibration Run Plan

Status: Wave A analyzed; Phase-4 feedback diagnostic next; Wave B/C pending
Last updated: 2026-07-19

This campaign finds the fastest fitting governor configuration instead of only
measuring settings below the historical reference. It deliberately separates:

- strict physical controls, which must preserve the canonical trace;
- opt-in research semantics, which may improve throughput with measured graph
  drift;
- later mechanism fitting, which begins only after the upward resource knee is
  known.

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
small numerical graph drift. Treat those rows as opt-in research evidence until
that physical/reduction coupling is removed or explicitly covered by a
versioned validated-relaxed package. Decoder-cache rows remain strict.

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

Evaluate the paired contrasts rather than treating these as five independent
scores. The primary hypothesis is that the stale-window contrast explains most
graph drift, while the physical-microbatch contrast retains most of the speed
benefit with much smaller drift. Every row self-scores against the pinned
corrected-hook artifact; also compare each paired row directly after completion.
Do not redesign the governor or launch Wave B until this diagnostic is
adjudicated.

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

Wave B starts only after choosing the top two useful 1B PLT fetch chunks and a
bounded batch range from Wave A.

| Provider | Logical batch candidates | Fetch chunks | Host RAM | Walltime |
|---|---|---|---:|---:|
| Gemma 3 4B PLT | `128, 256, 512` | `16384, 32768` | 300-400 GiB | up to 2h |
| Gemma 3 12B PLT | `64, 128, 256` | `16384, 32768` | 400-600 GiB | 4-5h |

Each model gets an exact reference, upward batch rows, the selected opt-in
fetch-chunk rows, a bounded `b256` coupled candidate, and only the semantic
research rows needed to test whether the 1B relation transfers. `b512` is an
upper drift probe, not a prospective default. Do not blindly run the full 1B
Cartesian space.

## Wave C - Local Mechanism Fit

Wave C fits the remaining physical cost terms around the Wave A/B knee:

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

The resource envelope walltime must be capped to the live Slurm allocation's
remaining time. Scenario walltime and `timeout_minutes` must match the submitted
allocation before launch; neither may promise time the job does not own.

## Promotion

Wave A discovers the feasible upward range and semantic tradeoff. It does not
by itself promote coefficients. Fit only after Waves B/C provide transfer and
local-mechanism evidence. Keep explicit support ranges and evidence IDs;
extrapolated values remain legal only inside implementation safety limits and
must be labeled as extrapolation.

Research rows remain opt-in. A range may move to `validated_relaxed` only with a
named, versioned evidence package that records provider/model scope, parity
metrics, runtime benefit, and the accepted drift threshold. Strict mode never
consumes that evidence implicitly.
