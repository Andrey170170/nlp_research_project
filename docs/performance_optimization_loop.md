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
| `plt-4b` | Gemma 3 4B GemmaScope-2 PLT | `361_base` | model-scoped transfer gate |
| `plt-12b` | Gemma 3 12B GemmaScope-2 PLT | `361_base` | model-scoped transfer gate |
| `plt-large` | 4B and 12B PLT | both `361_base` cases | canonical-only mixed-model inspection; model-specific profiles are rejected |
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

Larger models must use their model-specific suites and profiles. Do not pair a
4B or 12B profile with `plt-large`; the CLI rejects this to prevent the other
model from silently running canonical:

```bash
uv run exact-trace-perf run plt-4b \
  --candidate-profile plt-active-rows-4b-c4096-v1 \
  --fidelity bounded \
  --run-id perf-plt-4b-active-rows-c4096-01
```

`--host-memory-stop-gib` is a total cgroup-charge cutoff. Decoder checkpoints
are clean mmap-backed file cache, so a low total-charge threshold can terminate
a healthy run immediately before kernel reclaim. For accepted-risk 250 GiB
engineering runs, use the cache-aware RSS/anonymous guard and keep the cgroup
hard limit as the cache boundary:

```bash
--host-rss-stop-gib 64
```

A 250 GiB result is an engineering envelope observation, not a replacement for
the documented 400G/600G baseline resource contracts.

Named profiles make the tested physical controls auditable:

| Profile | PLT physical controls | Fidelity scope |
|---|---|---|
| `canonical` | frozen scenario defaults | bounded or exact |
| `plt-bounded-fast-v1` | decoder chunk 32,768; Phase 1/3 cap 128; Phase 4 execution cap 256; session 256 | bounded only |
| `plt-bounded-fast-v2` | same, with decoder chunk 65,536 | bounded only |
| `plt-bounded-fast-v3` | v2 plus a 16 GiB cross-batch decoder cache | bounded only |
| `plt-bounded-tape-v1` | v2 plus a two-execution-batch VJP tape with a 12 GiB simultaneous-owned-byte cap; decoder cache disabled | bounded only |
| `plt-bounded-tape-prefetch-v1` | tape-v1 plus one physically fenced decoder-page lookahead slot; decoder cache disabled | bounded only |
| `plt-bounded-frontier-v1` | v2 with one bounded 512-row physical execution batch per four-semantic-batch refresh frontier; decoder cache disabled | bounded only |
| `plt-active-rows-4b-c4096-v1` | 4B b128/c4096 with a 4 GiB active-row cap | bounded only; measured exact fail against the frozen non-residency graph |
| `plt-active-rows-4b-c65536-v1` | 4B b128/c65536 with a 4 GiB active-row cap | bounded only |
| `plt-active-rows-4b-b512-c4096-v1` | 4B execution cap 512 at c4096 | bounded only |
| `plt-active-rows-4b-b512-c65536-v1` | combined 4B execution cap 512 and c65536 | bounded only; measured slower than b128/c4096 |
| `plt-active-rows-12b-c4096-v1` | 12B b64/c4096 with an 8 GiB active-row cap | exact gate allowed; no completed parity result yet |
| `plt-active-rows-12b-c32768-v1` / `c65536-v1` | 12B b64 with a larger decoder chunk | bounded only |
| `plt-active-rows-12b-b256-c4096-v1` / `c65536-v1` | 12B execution cap 256, optionally with c65536 | bounded only |

The 16 GiB cache in v3 is sized to retain the reusable 1B PLT decoder, which is
about 14.6 GiB in bf16. It is not an instruction to fill HBM. The cache remains
a physical candidate, not a promoted default:

```bash
uv run exact-trace-perf run plt-hard \
  --candidate-profile plt-bounded-fast-v3 \
  --fidelity bounded \
  --run-id perf-plt-cache16g-c65536-01
```

The tape profile is the first cache-independent architectural candidate. Its
12 GiB value is a hard ceiling across simultaneous host gradients, replay-device
materializations, and row buffers, not a reservation or utilization target. It
falls back to a one-batch flush whenever the next captured record would exceed
the byte limit:

```bash
uv run exact-trace-perf run plt-hard \
  --candidate-profile plt-bounded-tape-v1 \
  --fidelity bounded \
  --run-id perf-plt-tape-w2-c65536-01
```

The prefetch profile is an opt-in diagnostic candidate built on the same tape
and cache-disabled fallback. It owns one persistent worker and CUDA stream for
Phase 4 and uses a consumer-completion fence so that final decoder-page
residency is physically bounded to the current and next pages. It is not a
promoted speed profile; the H200 characterization below rejected direct-GPU
prefetch for timing.

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
6. Promote performance profiles only with a reproducible speedup and passing
   parity. A default-off bounded-safety primitive may remain when its memory
   contract is verified and its negative timing result is recorded explicitly.

## Job 1654070 H200 characterization

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

### Cache-disabled architectural candidates

The two-execution-batch VJP tape was then tested under the same 65,536-row
decoder-page and zero-cache envelope. Source state changed only at the recorded
candidate checkpoints; each row was protected by the loop's before/after
content hashes.

| Run | Mechanism | Project / sibling | End-to-end | Phase 4 | Peak framebuffer | Decision |
|---|---|---|---:|---:|---:|---|
| `perf-plt-streaming-baseline-c65536-20260724-03` | one-batch streaming | `49da3d7` / `f324d70` | 391.32s | 338.54s | 26,113 MiB | bounded survival reference |
| `perf-plt-tape-w2-c65536-20260725-01` | tape window 2, 12 GiB cap | `9b26258` / `1ec99f7` | 307.34s | 250.81s | 26,265 MiB | historical pre-residency winner; superseded by active rows |
| `perf-plt-tape-w2-c65536-20260725-02` | repeat of tape window 2 | `9b26258` / `1ec99f7` | 312.65s | 255.31s | 26,125 MiB | repeat supports approximately 5.1–5.2 min |
| `perf-plt-tape-gather-c65536-20260725-01` | tape plus gather-before-cast | `9b26258` / `793e2ef` | 318.07s | 261.58s | 26,127 MiB | retain workspace bound; reject for speed |
| `perf-plt-frontier512-c65536-20260725-01` | one 512-row execution group | `c55a222` / `793e2ef` | 341.44s | 285.60s | 49,669 MiB | reject for speed and memory |
| `perf-plt-tape-prefetch-c65536-20260725-01` | unfenced direct-GPU lookahead | `e6a6daa` / `e1b8ad6` | 465.72s | 409.74s | 38,609 MiB | invalid physical bound; superseded |
| `perf-plt-tape-prefetch-fenced-c65536-20260725-01` | persistent fenced direct-GPU lookahead | `38825a8` / `c8c578c` | 571.31s | 495.74s | 26,917 MiB | bound verified; reject for speed |

Both tape repeats reduced Phase 4 by 24.6–25.9% relative to cache-disabled
streaming and completed in 307.34s and 312.65s. The 12 GiB tape cap was not
filled for appearance: its observed simultaneous high-water was 8,022,227,568
bytes, comprising 2,631,260,160 pinned-host bytes, 5,262,520,320 replay-device
bytes, and 141,792,240 row bytes. It executed 17 decoder traversals for 34
execution batches. Phase 4 loaded 1,500 decoder pages totaling 226,492,416,000
bytes.

The tape, gather-before-cast, Frontier-512, and fenced-prefetch rows passed the
bounded compact gate with the same reported metrics. Because every row also
uses c65536, this establishes bounded engineering behavior but cannot attribute
numerical drift to an individual mechanism. Active-row residency later removed
the decoder replay traffic that made tape/cache useful; a direct c4096
post-residency tape run preserved exact parity but was slower. None of this
establishes internal edge-sign parity. The gather-before-cast change remains
valuable because its FP32 replay workspace is bounded by selected rows rather
than a whole decoder page, but it is not a speed promotion. The 512-row
frontier candidate was slower and raised peak framebuffer residency by roughly
23 GiB.

The first direct-GPU lookahead implementation was only logically depth one:
Python could enqueue additional page uses faster than the consumer completed,
and recreating CUDA streams across tape windows fragmented allocator pools.
The corrected implementation uses one Phase-4 owner, one worker, one producer
stream, and a consumer-completion retirement fence. Its live evidence matched
the intended contract:

- final-page pipeline high-water: 2 pages / 301,989,888 bytes;
- owner high-water: 1, with one open and one close;
- all active, retained, in-flight, and owner gauges: zero at Phase-4 exit;
- Phase-4 loads unchanged at 1,500 pages / 226,492,416,000 bytes;
- peak framebuffer: 26,917 MiB, stable rather than climbing to 38,609 MiB.

That candidate nevertheless spent 426.11s in decoder replay and 495.74s in
Phase 4, versus 189.83–205.56s replay and 250.81–255.31s Phase 4 for tape-only.
Whole-run GPU SM utilization was 9.58% and allocated-CPU utilization was 15.12%.
It passed the 600s hard gate at 571.31s but missed the 300s stretch target and
the candidate-specific 313s no-go threshold, so it was not repeated or
promoted. The result isolates the next architectural seam: if page overlap is
revisited, use fixed GPU slots with bounded pinned-CPU staging and CPU-side
checkpoint reads rather than direct safetensors CUDA allocations. Separately,
the dominant remaining tape-only floor is many small decoder contractions; a
bounded fused/coalesced contraction kernel is the more promising sub-five-minute
direction.

CLT regression controls completed in 89.58s before gather-before-cast, 89.43s
after it, and 85.84s at the committed prefetch-profile source state. Their
compact artifacts matched exactly. Because CLT does not advertise the PLT page
prefetch capability, the final control exercised the depth-zero fallback, not
the lookahead mechanism. These are compatibility controls, not a new general
CLT timing claim.

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

The implemented Phase-3/4 mechanism is a byte-bounded cross-execution-batch VJP
tape within one already-fixed semantic refresh frontier:

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

Acceptance status under the intentionally constrained decoder-cache envelope:

1. window one reproduces the current streaming result and remains the survival
   path;
2. tape window two passes the bounded compact gate and reduces Phase 4 by
   24.6–25.9% against the same-budget cache-disabled baseline; strict/exact
   compact parity remains unestablished and must not be inferred;
3. the tested 512-row bounded coalescing and direct-GPU prefetch candidates are
   rejected for speed; neither is a launch-default candidate;
4. telemetry reports tape bytes and tier high-watermarks, decoder pages and
   bytes loaded, page-pipeline ownership, contraction/replay time, durable tile
   progress, and plan revisions; transfer/compute overlap is not established by
   the current evidence;
5. no result may be described as a 1k-prefix, 4B/12B, or extreme-regime result
   until a corresponding gate exists.

The extreme path also needs bounded forward activation replay/offload,
file-backed active-feature ranking, model placement or parallelism when
permanent weights do not fit, and checkpoints that can continue across Slurm
jobs. Until those mechanisms are certified, the guarantee is bounded progress
or early actionable refusal—not that every finite request fits one process.
