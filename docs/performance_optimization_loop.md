# Exact-trace performance optimization loop

Status: isolated engineering workflow  
Branch: `perf/exact-trace-loop`

This loop measures candidate runtime changes without disturbing governor
calibration work. It runs the canonical exact-trace pipeline, compares the
resulting compact graph with frozen Granite H200 references, and fails when the
selected fidelity policy is exceeded. It does not fit, promote, or modify
governor profiles or launch defaults.

Keep changes and reports on the dedicated project and sibling worktrees:

```text
projects/worktrees/exact-trace-perf/
├── nlp_research_project/
└── circuit-tracer_chunked/
```

Both worktrees should be on `perf/exact-trace-loop`. The CLI records the branch,
commit, dirty-file list, and content hash for both repositories before execution,
then compares that recorded state before and after every case. A detected
source-state change aborts the suite. This is after-the-fact change detection,
not a filesystem lock: the operator remains responsible for making no edits
while a case runs.

This is an explicit live-workspace exception to the ordinary immutable
experiment policy: `run_manifest.json` records `workspace_mode=live`, the
non-empty rationale that the isolated worktree itself is the candidate under
test, and `no_edits_during_run_enforced=true`. Do not edit either worktree while
a case is running.

## Suites

| Suite | Provider | Fixtures | Intended use |
|---|---|---|---|
| `clt-smoke` | Gemma 3 1B GemmaScope-2 CLT | `361_base` | shortest inner loop |
| `clt-pair` | Gemma 3 1B GemmaScope-2 CLT | `828_base`, `361_base` | broader CLT gate |
| `plt-hard` | Gemma 3 1B GemmaScope-2 PLT | `361_base` | longer, harder gate |
| `all` | both above | CLT pair plus PLT hard | 1B optimization-candidate gate |

List the fixed cases without loading a model:

```bash
uv run exact-trace-perf list
```

## Running inside an H200 job

The command is intentionally allocation-local: it does not submit or wrap a
Slurm job. Start or enter a Granite H200 allocation using the normal project
workflow, then run:

```bash
uv run exact-trace-perf run clt-smoke
```

The default fidelity policy is `bounded`. For timing evidence, the command
requires `SLURM_JOB_ID`, cluster `granite`, partition `rai-gpu-grn`, and exactly
one visible full H200 with at least 130,000 MiB reported by `nvidia-smi`.
`--dry-run` is login-safe and prints the underlying commands without creating
output:

```bash
uv run exact-trace-perf run all --dry-run
```

Use a stable, unique ID when comparing iterations:

```bash
uv run exact-trace-perf run clt-pair \
  --run-id perf-column-kernel-v2 \
  --fidelity bounded
```

The longer PLT checkpoint is:

```bash
uv run exact-trace-perf run plt-hard --run-id perf-column-kernel-v2-plt
```

Named profiles make the tested physical controls auditable:

| Profile | PLT physical controls | Fidelity scope |
|---|---|---|
| `canonical` | frozen scenario defaults | bounded or exact |
| `plt-bounded-fast-v1` | decoder chunk 32,768; Phase 1/3 cap 128; Phase 4 execution cap 256; session 256 | bounded only |
| `plt-bounded-fast-v2` | same, with decoder chunk 65,536 | bounded only |
| `plt-bounded-fast-v3` | v2 plus a 16 GiB cross-batch decoder cache | bounded only |

The 16 GiB cache in v3 is sized to retain the reusable 1B PLT decoder, which is
about 14.6 GiB in bf16. It is not an instruction to fill HBM. The cache remains
a physical candidate, not a promoted default:

```bash
uv run exact-trace-perf run plt-hard \
  --candidate-profile plt-bounded-fast-v3 \
  --fidelity bounded \
  --run-id perf-plt-cache16g-c65536-01
```

The default run goal can be replaced with iteration-specific wording, and is
persisted in both the performance manifest/report and the existing runner's
metadata:

```bash
uv run exact-trace-perf run plt-hard \
  --run-id perf-column-kernel-v2-plt \
  --run-goal "Evaluate the column kernel candidate under bounded parity."
```

`--allow-non-h200-test-only` exists only for focused automated tests. Results
produced with that override are not H200 performance evidence.

## Acceptance gates

The harness reuses `graph_compare.compare_artifact_dirs` through the existing
sparsification runner's baseline gate.

| Policy | Feature Jaccard | All-edge Jaccard | All-edge Top-256 | All-edge weighted | All-edge magnitude L1 | Token identity |
|---|---:|---:|---:|---:|---:|---:|
| `bounded` | >=0.98 | >=0.98 | >=0.98 | >=0.98 | <=0.02 | exact |
| `exact` | 1.0 | 1.0 | 1.0 | >=0.999999 | <=0.000001 | exact |

The policy applies to the worst aligned generation step, not only the overall
means. All-edge metrics include feature-to-feature and feature-to-logit compact
edges. All-edge magnitude normalized L1 deviation is
`sum(|candidate_magnitude-reference_magnitude|) / max(sum(candidate_magnitude), sum(reference_magnitude), 1e-12)`
over the union of labeled endpoints. Compact artifacts store absolute weights,
so this gate cannot establish internal sign parity.
The comparison must also align every completion and step; missing or
structurally incomplete comparisons fail independently of numeric thresholds.
Overall means remain in the artifacts for compatibility, while
`worst_step_evidence` identifies the completion and step responsible for each
gate value.

`bounded` is an engineering characterization budget for finding promising
performance changes. Passing it is not exact-semantics evidence and must not
promote a runtime mechanism, governor profile, calibration observation, or
launch default. Use `exact`, followed by the project’s separately reviewed
immutable gates, for any exact-semantics claim.

Parity and runtime are independent gates. Every case report records
`parity_passed`, `performance_target_seconds`, `performance_passed`, and the
overall `passed` result. PLT `361_base` has a hard
`performance_target_seconds=600.0`; its runtime gate passes only when the
candidate duration is at most 600 seconds. A missing duration or a duration
above 600 seconds fails the run and makes the CLI exit nonzero even if parity
passes. CLT cases currently have no hard runtime budget, so they record
`performance_target_seconds=null` and `performance_passed=null`; their overall
result still requires parity and successful runner execution.

PLT `361_base` also records a 300-second stretch target. Stretch attainment is
reported independently and does not weaken or replace the hard gate. The
working objective is reproducible sub-five-minute 1B PLT execution; a candidate
should be repeated on the same allocation and source state before that timing is
treated as stable.

The suite report aggregates the same three outcomes. `parity_passed` covers
only graph comparison, `performance_passed` covers all applicable hard runtime
targets (and is null when the suite contains none), and overall `passed`
requires parity, every applicable runtime target, and zero runner exit codes.
Timing is the end-to-end candidate duration reported by the existing
sparsification runner, not the sum of selected profiling phases.

The frozen registry is
`experiments/baselines/exact_trace_performance_granite_20260709.json`. It points
to the corrected-hook `chpc-baseline-gemma-stack-20260709-03` H200 artifacts:

| Case | Baseline duration |
|---|---:|
| CLT `828_base` | 156.50s |
| CLT `361_base` | 105.07s |
| PLT `361_base` | 2805.70s |

These are engineering comparison anchors, not governor calibration observations.
Changing the registry is a reviewed baseline change and should not be folded
into an optimization patch.

The source run recorded a live project workspace at base commit `6a91f1b`
with only its Slurm launcher dirty, plus a clean sibling at `690fc04`; it did
not record an immutable snapshot. The registry pins SHA-256 digests for each
referenced `step_000.npz`, `completion.json`, and `result.json` and verifies
them before tracing, but those artifact digests do not make the historical
live workspace immutable. Treat this as a provenance limitation.

## Outputs

By default, reports are written outside the worktree under:

```text
/scratch/general/vast/$USER/nlp_research_project/exact_trace_bench/
  granite/sweep/performance_optimization/<run-id>/
```

Each run contains:

- `run_manifest.json`: suite, thresholds, baseline reference, Slurm/GPU
  provenance, content hashes for tracked and untracked source, both repository
  states, and timing-evidence eligibility;
- `configs/`: generated single-case scenario files;
- `candidates/`: ordinary sparsification runner logs, compact artifacts,
  results, scenario metrics, and baseline comparisons;
- `candidates/<case>/gpu_samples.csv` and `resource_summary.json`: one-second
  GPU samples plus aggregate SM, memory-controller, power, framebuffer, and
  process-tree CPU utilization; sampler status, exit code, usable sample count,
  and an explicit resource-validation result prevent an empty sampler from
  looking like valid utilization evidence;
- `performance_report.json`: candidate and baseline duration, speedup, all
  worst-step parity metrics and evidence, the runner's `profiling_summary`,
  resource summary, independent parity/performance/resource/overall results,
  and per-case/overall pass status. Resource validation remains diagnostic
  rather than changing the fidelity/runtime gate because utilization is a
  secondary optimization objective.

`--allow-non-h200-test-only` is recorded as
`test_environment_override_used=true` and
`timing_evidence_eligible=false`; any timing from such a run is explicitly
non-evidence.

The CLI invokes `experiments/run_sparsification_experiment.py` rather than
adding a tracing path. It tails each `run.log` while the candidate is running
and prints one compact report row after comparison, including the available
Phase 3, Phase 4, and Phase 4 batch timings.

## Resource efficiency

Runtime is the primary objective and parity is the hard scientific constraint.
Resource utilization is diagnostic: high allocation alone is not success.
Inspect useful work, cache reuse, and throughput together:

- mean, p95, and maximum GPU SM and memory-controller utilization;
- mean/maximum power and peak framebuffer residency;
- process-tree CPU time relative to wall time and allocated CPUs;
- runner peak RSS and CUDA allocated/reserved memory;
- Phase 3/4 durations and Phase 4 execution-batch distribution.

A memory-for-speed candidate is justified when retained data is repeatedly
consumed and its residency removes measured I/O or recomputation. The v3 cache
is an example: it retains the 1B PLT decoder across Phase 4 batches. Do not raise
cache budgets merely to increase reported memory use.

### Long-prefix direction

The fixed gate currently uses a 124-token prefix. Passing it does not establish
runtime or memory safety for 1k-token or longer prefixes. Prefix length increases
activation, gradient, and attention-state pressure while the decoder size stays
fixed, so a full-decoder cache becomes a lower-priority HBM consumer as context
grows.

Keep the runtime capable of opportunistic, reclaimable decoder caching. A
long-prefix plan must derive the cache allowance from residual HBM after model,
forward, retained-gradient, and row-buffer preflight; it must be able to reduce
or disable the cache rather than OOM. Do not hard-code the 16 GiB short-prefix
candidate as a universal launch default. Until a tractable long-prefix gate is
frozen, report long-context performance as unverified and retain prompt length
in every timing claim.

## Optimization cadence

1. Make one cohesive runtime change on the isolated branch.
2. Run `clt-smoke`.
3. If it is faster and passes, run `clt-pair`.
4. Run `plt-hard` for changes that may affect provider topology, batching,
   reduction order, decoder access, or memory behavior.
5. Use `all` before treating a change as a viable Gemma 3 1B optimization
   candidate.
6. Keep only changes with a reproducible speedup and a passing parity report.

## 2026-07-24 H200 characterization

On Granite job `1654070`, the v3 profile completed the 1B PLT `361_base`
one-token trace twice from the same project and sibling source state:

| Run | Decoder cache | End-to-end | Phase 3 | Phase 4 | Speedup | Bounded gate | 300s stretch |
|---|---:|---:|---:|---:|---:|---|---|
| `perf-plt-cache16g-c65536-20260724-01` | 16 GiB | 279.89s | 15.80s | 224.40s | 10.024x | pass | pass |
| `perf-plt-cache16g-c65536-20260724-02` | 16 GiB | 280.58s | 15.93s | 225.34s | 10.000x | pass | pass |
| `perf-plt-streaming-baseline-c65536-20260724-03` | disabled | 391.32s | 15.52s | 338.54s | 7.170x | pass | miss |

All three runs produced the same compact comparison metrics: feature Jaccard
0.994643, all-edge Jaccard 0.996008, Top-256 Jaccard 0.992218, weighted-edge
Jaccard 0.995133, normalized magnitude L1 deviation 0.004879, and exact target
token identity. This is reproducible 1B short-prefix bounded engineering
evidence, not exact semantics, a long-prefix result, or a 4B/12B result.

The clean cache-disabled run establishes the bounded-memory architectural
baseline. It used only 26,113 MiB peak framebuffer memory, but whole-run GPU SM
utilization averaged 10.86% (p95 16%, maximum 21%), GPU power averaged 118.63 W,
and the twelve allocated CPUs averaged 16.79% utilization. Phase 4 issued 34
execution batches and averaged 8.68s per batch. The same graph comparison
metrics as the cache-resident runs show that the 113.20s Phase-4 difference is
decoder reuse cost, while the remaining 225s cache-resident Phase 4 is a
separate repeated-contraction and launch-utilization floor.

Passing this live-workspace loop supports only a post-run engineering claim
that the tested Gemma 3 1B source state is an optimization candidate under the
selected parity policy and the applicable timing target. It is not a release
gate, does not establish an exact-semantics claim under `bounded`, and does not
promote a runtime mechanism, governor profile, calibration observation, or
launch default. A broader or confirmatory claim requires the separately
reviewed immutable project gates. This loop does not yet define or authorize
4B or 12B performance targets.

Custom Triton or CUDA operations belong in the sibling runtime, but only after
the project-side loop is green. Such changes still use this same CLI and frozen
registry; do not create a separate comparison implementation.

## Bounded-throughput architecture

Survival is a hard constraint; throughput is optimized inside the declared
resource envelope. Decoder cache capacity is one elastic reuse decision, not
the architecture. Every optimized rung must retain a bounded fallback that
makes monotonic, checkpointable progress with one execution batch, one decoder
page, and one row tile. If the irreducible unit cannot fit, admission must refuse
before the phase starts rather than discovering infeasibility through a late
memory spike.

The next Phase-3/4 mechanism is a byte-bounded cross-execution-batch VJP tape
within one already-fixed semantic refresh frontier:

```text
fixed semantic frontier
  -> capture each execution batch VJP once
  -> bounded FeatureVjpTape (GPU, pinned host, host, or file-backed)
  -> source-layer / decoder-page-major contraction
  -> bounded row tiles and durability fence
  -> existing ordered influence refresh
  -> frontier checkpoint
```

The current `chunked_feature_replay_window` is not this mechanism. It retains a
small number of output-layer gradients inside one backward pass and then scans
the decoder for that execution batch. It does not reuse a decoder page across
Phase-4 execution batches. The cross-batch tape changes physical loop order so
one decoder page serves several captured batches before eviction, without
increasing `cross_batch_decoder_cache_bytes`.

The canonical runtime should own explicit domain objects rather than another
unstructured knob group:

- `FrontierSlice` owns contiguous execution batches up to, but never across, a
  semantic refresh boundary.
- `FeatureVjpTape` owns immutable per-batch/per-layer gradients, storage
  placement, byte accounting, checksums, and cleanup.
- `DecoderPagePipeline` owns a bounded decoder page source, at most one
  lookahead page in the first implementation, CUDA events, and deterministic
  slot reuse.
- `RowTileSink` accepts disjoint completed tiles and exposes the durability
  fence required before influence refresh.
- `PhaseSliceGrant` is the governor decision containing independent GPU, pinned
  host, ordinary host, and spill limits plus the permitted execution rung.

The exact rung invokes the existing per-batch contraction with the same shapes,
dtype, row mapping, and reduction order after loop interchange. Its main gain
is decoder-load amortization. A separately identified `coalesced_bounded` rung
may concatenate compatible tape entries into larger contractions to reduce the
thousands of roughly 0.14s per-layer calls observed on H200. That rung is
numerically sensitive and must pass the bounded gate; it must never be implied
by the exact rung.

The memory contract for one frontier slice is:

```text
permanent model and session state
+ one VJP execution lane
+ bounded tape staging
+ decoder page and optional prefetch slot
+ bounded row-tile slots
+ influence working tile
+ allocator safety reserve
<= ResourceEnvelope
```

Host and spill tiers have independent bounds. The governor shrinks only
physical axes at safe boundaries: contraction coalescing, tape batch count,
prefetch depth, optional hot residency, row/decoder tile width, VJP microbatch,
then tape placement. All queues and transient tensors require leases. Runtime
pressure produces a recorded plan revision or checkpointed restart with a
smaller grant, never an unrecorded fallback.

Initial acceptance under an intentionally constrained decoder-cache envelope:

1. window one reproduces the current streaming result and remains the survival
   path;
2. exact tape windowing preserves strict compact parity and reduces decoder
   loads by at least 35% and Phase 4 by at least 20% against the same-budget
   cache-disabled baseline;
3. bounded coalescing passes the existing bounded graph floors, beats
   prefetch-only and exact windowing, and reduces Phase 4 by at least 25%;
4. telemetry reports tape bytes and tier high-watermarks, decoder pages and
   bytes loaded, contraction calls, transfer/compute overlap, durable tile
   progress, and plan revisions;
5. no result may be described as a 1k-prefix, 4B/12B, or extreme-regime result
   until a corresponding gate exists.

The extreme path also needs bounded forward activation replay/offload,
file-backed active-feature ranking, model placement or parallelism when
permanent weights do not fit, and checkpoints that can continue across Slurm
jobs. Until those mechanisms are certified, the guarantee is bounded progress
or early actionable refusal—not that every finite request fits one process.
