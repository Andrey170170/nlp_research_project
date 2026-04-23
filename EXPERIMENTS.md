# Experiments inventory

This file is a living note for experiment artifacts currently stored on scratch.

For now it documents:

- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench`

The goal is to make it easy to answer:

- what experiment families exist,
- where they live,
- what each family was trying to test,
- and which runs should be treated as baseline / debug / anomaly / exploratory.

## Scratch layout

Current top-level layout:

```text
/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/
  ascend/
    fast/
    anomaly/
    long_eval/
    matched_debug/
  cardinal/
    fast/
    anomaly/
    long_eval/
    matched_debug/
  manual_scenarios/
  manual_worktrees/
  workspace_snapshots/
```

### Meaning of the subtrees

- `ascend/fast`: main fast-loop benchmark runs on Ascend, usually `828_base` and
  `361_base`
- `ascend/anomaly`: prompt-94 watch / anomaly runs on Ascend
- `ascend/long_eval`: late-prefix stress runs on Ascend
- `ascend/matched_debug`: matched debug or manual comparison runs on Ascend
- `cardinal/*`: Cardinal equivalents where available
- `manual_scenarios/`: custom JSON scenario files for one-off comparison
  matrices
- `manual_worktrees/`: manually created worktrees for branch / commit isolation
- `workspace_snapshots/`: immutable project + library snapshots used for actual
  launches

## Current inventory summary

From the current extractor pass over `exact_trace_bench/`:

- total extracted scenario rows: `66`

By cluster / group:

- `ascend/fast`: `17`
- `ascend/anomaly`: `14`
- `ascend/long_eval`: `4`
- `ascend/matched_debug`: `13`
- `cardinal/fast`: `3`
- `cardinal/anomaly`: `6`
- `cardinal/long_eval`: `4`
- `cardinal/matched_debug`: `5`

## Major experiment families

### 1. Historical unnamed baselines

These are older direct benchmark outputs that do not carry a `run_name` in the
current extraction tables.

They cover a mix of:

- `361_base`
- `828_base`
- `94_base`
- late-prefix fixtures (`361_late`, `828_late`, `94_late`)

Use these as historical references only; prefer named runs when possible.

### 2. Memory-refactor smoke runs

Purpose: validate early exact-mode memory/system improvements before Phase 4
work.

Current named families:

- `lazy encoder smoke ascend`
- `memmap row-store fast ascend`

These are primarily Ascend runs on:

- `361_base`
- `828_base`
- `94_base`

### 3. Phase 4 locality / autoscale / planner runs

Purpose: test the first Phase 4 scheduling optimization and then batch-size
control strategies.

Current named families:

- `phase4 locality reorder fast ascend`
- `phase4 autoscale fast ascend cuda fix`
- `phase4 autoscale anomaly ascend`
- `phase4 autoscale anomaly ascend cuda fix`
- `phase4 planner fast ascend`
- `phase4 planner anomaly ascend`

Interpretation notes:

- locality reorder is the first real Phase 4 optimization family
- old autoscale runs are historical only; planner/preflight is the current path
- planner runs are the important batch-size comparison set

### 4. Telemetry validation runs

Purpose: validate structured telemetry artifacts and extraction.

Current named family:

- `telemetry validation fast ascend`

This run family is important because it established:

- `telemetry.jsonl`
- completion-level timing summaries
- downstream extraction via `experiments/telemetry_gathering.py`

### 5. Prompt-94 anomaly debug campaigns

Purpose: investigate Phase 4 anomaly behavior on prompt 94.

Ascend families:

- `prompt94 anomaly debug ascend`
- `prompt94 anomaly debug ascend v2`
- `prompt94 anomaly debug ascend v3`
- `prompt94 standard ascend float64 norm`
- `prompt94 anomaly debug ascend float64 norm general`

Cardinal families:

- `prompt94 anomaly debug cardinal`
- `prompt94 anomaly debug cardinal v2`
- `prompt94 anomaly debug cardinal v3`
- `prompt94 standard cardinal float64 norm`
- `prompt94 anomaly debug cardinal float64 norm general`

Interpretation notes:

- these are watch / diagnosis runs, not normal optimization baselines
- `*standard* float64 norm` runs are especially important as post-float64
  non-debug references

### 6. Prompt-828 float64 debug runs

Purpose: check whether the prompt-94 anomaly story generalizes to a healthy base
prompt.

Current named families:

- `prompt828 debug ascend float64 norm general`
- `prompt828 debug cardinal float64 norm general`

Interpretation notes:

- these are diagnostic runs, not clean performance baselines
- use them for matched graph/debug comparisons, not direct speed claims

### 7. Matched-debug fixture campaigns

Purpose: compare a larger base-fixture set under the matched-debug protocol.

Current named families:

- `matched debug ascend b256 c4096`
- `matched debug cardinal b256 c4096`

Current fixtures in this family:

- `828_base`
- `613_base`
- `999_base`
- `1046_base`
- `1075_base`

These runs live under `*/matched_debug/` and use the matched-debug fixture
catalog produced from:

- `experiments/generated/weekend_exact_chunked_fixtures_matched_debug/`

### 8. Phase 4 refresh cache run

Purpose: test the first Direction-A refresh-cache implementation.

Current named family:

- `phase4 refresh cache fast ascend`

This family includes:

- `361_base`
- `828_base`
- `94_base`

Interpretation note:

- later analysis showed the measured speedups were not well explained by cache
  hits, so this family should be treated as an exploratory checkpoint rather
  than a settled optimization win

### 9. fp32 / float64 normalization comparison matrix

Purpose: isolate whether the Phase 4 normalization precision change explains the
large runtime shifts.

Current named families:

- `fp32 norm debug ascend`
- `float64 norm debug ascend`
- `fp32 norm 361 baseline ascend`
- `float64 norm 361 baseline ascend`

These were launched from custom scenario files under:

- `manual_scenarios/ascend_phase4_norm_compare_debug_matrix.json`
- `manual_scenarios/ascend_phase4_norm_compare_361_baseline.json`

and custom library worktrees under:

- `manual_worktrees/circuit-tracer_chunked_fp32_norm_9afff02`
- `manual_worktrees/circuit-tracer_chunked_float64_norm_62f5271`

Important interpretation note:

- commit `9afff02` already uses float64 normalization sums in anomaly-debug mode
- therefore `fp32 norm debug ascend` is **not** a true fp32-normalization debug
  run; it is only fp32-branch + debug-mode float64 normalization
- the clean normalization A/B currently available is the non-debug
  `361_base` baseline pair

Status:

- completed / analyzed

### 10. True fp32 / fp64 internal-dtype comparison matrix

Purpose: rerun the normalization comparison using the new external runtime knob
so the comparison is a **true** fp32-vs-fp64 test on the current codebase.

Current named families:

- `true fp32 norm debug ascend`
- `true fp64 norm debug ascend`
- `true fp32 norm 361 baseline ascend`
- `true fp64 norm 361 baseline ascend`

These runs use the new runtime control knob:

- `exact_trace_internal_dtype=fp32|fp64`

and were launched from custom scenario files under:

- `manual_scenarios/ascend_true_norm_fp32_debug_fast.json`
- `manual_scenarios/ascend_true_norm_fp32_debug_anomaly.json`
- `manual_scenarios/ascend_true_norm_fp64_debug_fast.json`
- `manual_scenarios/ascend_true_norm_fp64_debug_anomaly.json`
- `manual_scenarios/ascend_true_norm_fp32_361_baseline.json`
- `manual_scenarios/ascend_true_norm_fp64_361_baseline.json`

Current output roots:

- fast / fp32 debug:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260421_124700_true-fp32-norm-debug-ascend`
- anomaly / fp32 debug:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260421_124700_true-fp32-norm-debug-ascend`
- fast / fp64 debug:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260421_124700_true-fp64-norm-debug-ascend`
- anomaly / fp64 debug:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260421_124700_true-fp64-norm-debug-ascend`
- fast / fp32 361 baseline:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260421_124700_true-fp32-norm-361-baseline-ascend`
- fast / fp64 361 baseline:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260421_124700_true-fp64-norm-361-baseline-ascend`

Interpretation note:

- unlike the older `fp32 norm debug ascend` family, these runs are intended to be
  the real fp32/fp64 comparison on the current branch because the runtime dtype
  is controlled externally rather than inferred from debug mode

Status:

- **in progress**

Update:

- now completed / analyzed

Main findings from the true-dtype comparison (pre-fix code state):

- `828_base` and `361_base` genuinely collapse under true fp32 internal dtype
- `94_base` does not collapse via the same mechanism
- for that pre-fix state, fp64 was the safer immediate runtime default
- the healthy-prompt speedups previously observed there were best explained by
  the float64 normalization path, not by the refresh-cache attempt

Observed healthy-prompt fp32 collapse signature:

- Phase-3 logit-row normalization sums overflow to `inf`
- refresh ranking becomes effectively all-zero across all refreshes
- runtime shifts into a degenerate regime with:
  - very small `phase4.refresh`
  - huge `phase4.feature_batch` / `context.compute_batch`
  - very large decoder-load counts
- compact outputs keep the same feature set, but fp32 healthy-prompt runs retain
  only `8192` edges instead of the full `20000`

Observed clean `361_base` baseline comparison:

- `fp32` baseline:
  - completion `5167.95s`
  - Phase 4 `4829.21s`
  - RSS `236.12 GiB`
- `fp64` baseline:
  - completion `4303.92s`
  - Phase 4 `3943.07s`
  - RSS `333.90 GiB`

Interpretation:

- fp64 improves `361_base` by roughly `16.7%` end-to-end and `18.3%` in Phase 4
- the improvement comes with much higher host RAM usage
- this pre-fix behavior is superseded by the validated permanent overflow fix in
  §14, where fp32/fp64 match exactly on the validation prompts

### 11. Long-eval runs

Purpose: late-prefix stress validation rather than fast-loop iteration.

These live under:

- `ascend/long_eval`
- `cardinal/long_eval`

Typical fixtures:

- `361_late`
- `828_late`
- `94_late`

Treat these as stress tests, not as default optimization benchmarks.

## Recommended baseline interpretation by topic

### General fast exact baselines

Prefer:

- named fast runs on `828_base` and `361_base`
- especially the most recent non-debug runs that match the code state you care
  about

### Prompt-94 anomaly reference

Prefer:

- `prompt94 standard ascend float64 norm`
- `prompt94 standard cardinal float64 norm`

### Planner reference

Prefer:

- `phase4 planner fast ascend`
- `phase4 planner anomaly ascend`

### Normalization precision reference

Prefer:

- `fp32 norm 361 baseline ascend`
- `float64 norm 361 baseline ascend`

These are currently the cleanest direct precision comparison pair.

## Notes on debug vs non-debug runs

Runs with `phase4_anomaly_debug=true` should be treated as:

- diagnosis / collapse / stability runs
- not clean performance baselines

Matched analysis so far suggests:

- debug mode can add very large runtime overhead
- but often leaves the final compact graph unchanged

So performance claims should come from matched non-debug runs whenever possible.

## Manual assets currently on scratch

### `manual_scenarios/`

Currently includes custom scenario JSON for:

- normalization precision comparison matrix
- matched `361_base` baseline comparison

### `manual_worktrees/`

Currently includes detached library worktrees for:

- commit `9afff02` (`fp32_norm` label, but debug still uses float64 sums)
- commit `62f5271` (`float64_norm` default)

### `workspace_snapshots/`

Contains immutable project + library snapshots created by:

- `uv run python -m experiments.exact_trace_bench snapshot-workspace ...`

These are the workspaces actually used by benchmark jobs, so long-running runs
do not observe live source edits.

## How to refresh this inventory

Re-extract from scratch:

```bash
uv run python -m experiments.exact_trace_bench extract \
  --input-root /fs/scratch/PAS3272/kopanev.1/exact_trace_bench \
  --output-dir /tmp/exact_trace_extract_all_scratch \
  --skip-slurm
```

Then summarize `benchmark_enriched.csv` by:

- cluster / group
- run name
- fixture set

## Recent launch update — overflow-fix validation

Purpose: validate the permanent overflow-fix implementation on the optimization
project+library pair after landing the two library commits:

- `3d9d01a` — `fix exact compact normalization overflow with stable row denominators`
- `ac368bb` — `harden shared normalization overflow paths and retained-mass stats`

Provenance:

- launch time: `2026-04-21 23:57:57`
- project workspace: `exact-trace-bench-opt@94e4283`
- library workspace: `exact-trace-hidden-knobs-opt@ac368bb`

Launches submitted on Ascend:

- fast validation array job: `5024044`
- anomaly/gate validation array job: `5024045`

Scenario files used:

- `experiments/generated/exact_trace_overflow_fix_fast_ascend_scenarios.json`
- `experiments/generated/exact_trace_overflow_fix_anomaly_ascend_scenarios.json`

Run metadata:

- fast run id: `20260421_235757_overflow-fix-fast-ascend`
- anomaly run id: `20260421_235757_overflow-fix-anomaly-ascend`

Submitted scenario matrix:

- `828_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32`
- `828_base` on `ascend/fast` with `exact_trace_internal_dtype=fp64`
- `361_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32`
- `361_base` on `ascend/fast` with `exact_trace_internal_dtype=fp64`
- `94_base` on `ascend/anomaly` with `exact_trace_internal_dtype=fp32`
- `94_base` on `ascend/anomaly` with `exact_trace_internal_dtype=fp64`

Shared runtime config:

- `attribution_batch_size=256`
- `feature_batch_size=256`
- `logit_batch_size=256`
- `decoder_chunk_size=4096`
- `cross_batch_decoder_cache_bytes=0`
- `completions=1`
- `temperature=0.0`
- `max_steps=1`
- `cross_cluster_debug=false`
- `phase4_anomaly_debug=false`
- `save_raw=false`

Expected output roots:

- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_fast_828_base_overflow_fix_fp32_b256_c4096_cache0`
- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_fast_828_base_overflow_fix_fp64_b256_c4096_cache0`
- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_fast_361_base_overflow_fix_fp32_b256_c4096_cache0`
- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_fast_361_base_overflow_fix_fp64_b256_c4096_cache0`
- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/ascend_anomaly_94_base_overflow_fix_fp32_b256_c4096_cache0`
- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/ascend_anomaly_94_base_overflow_fix_fp64_b256_c4096_cache0`

Interpretation goal:

- confirm healthy prompts (`828_base`, `361_base`) no longer show the old fp32
  collapse signature,
- confirm `fp64` remains sane after the permanent fix,
- keep `94_base` as the gate/anomaly prompt for non-collapse regressions.

Status:

- completed and analyzed.

Analysis provenance:

- analysis time: `2026-04-22 14:25:58`
- project workspace: `exact-trace-bench-opt@94e4283`
- library workspace: `exact-trace-hidden-knobs-opt@ac368bb`

Observed result summary:

- all 6 scenarios finished successfully.
- all 6 runs retained the full configured `20000` edges.
- fp32 and fp64 compact artifacts matched exactly for all three prompts under
  `compare-compact`:
  - `mean_feature_jaccard = 1.0`
  - `mean_edge_jaccard = 1.0`
  - `mean_weighted_edge_jaccard = 1.0`

Current fp32/fp64 paired results:

- `828_base`:
  - fp32: `808.97s` completion, `542.90s` Phase 4 wall, `1107` decoder loads,
    `20000` edges retained
  - fp64: `927.49s` completion, `659.29s` Phase 4 wall, `1107` decoder loads,
    `20000` edges retained
- `361_base`:
  - fp32: `2130.47s` completion, `1722.67s` Phase 4 wall, `1650` decoder loads,
    `20000` edges retained
  - fp64: `2335.41s` completion, `1948.25s` Phase 4 wall, `1650` decoder loads,
    `20000` edges retained
- `94_base`:
  - fp32: `920.13s` completion, `650.00s` Phase 4 wall, `1200` decoder loads,
    `20000` edges retained
  - fp64: `984.86s` completion, `696.68s` Phase 4 wall, `1200` decoder loads,
    `20000` edges retained

Primary interpretation:

- the old healthy-prompt fp32 collapse signature is absent after the permanent
  overflow fix.
- on `828_base` and `361_base`, fp32 no longer falls into the previous degenerate
  regime with truncated edge retention and inflated decoder-load counts.
- current fp32 and fp64 outputs are not merely similar; for these three probes
  they are artifact-identical at the compact saved-output level.
- runtime policy update from this validated post-fix matrix: fp32 is now the
  default exact compact tracing dtype; fp64 remains an optional parity mode.

Important contrast vs the earlier true-dtype comparison runs:

- earlier true-fp32 runs showed the collapse signature on healthy prompts,
  including:
  - only `8192` retained edges on `828_base` / `361_base`
  - much larger decoder-load counts (`21164` on `828_base`, `22254` on
    `361_base`)
- the new overflow-fix fp32 runs instead show:
  - full `20000` retained edges
  - decoder-load counts aligned with fp64 at the same runtime config
  - non-degenerate Phase 4 refresh behavior

Performance interpretation for the new code state:

- with the overflow fix in place, fp32 is now modestly faster than fp64 on this
  Ascend validation matrix while preserving identical compact outputs:
  - `828_base`: fp32 about `12.8%` faster end-to-end, `17.7%` faster in Phase 4
  - `361_base`: fp32 about `8.8%` faster end-to-end, `11.6%` faster in Phase 4
  - `94_base`: fp32 about `6.6%` faster end-to-end, `6.7%` faster in Phase 4

Caveat:

- do not compare absolute wall times directly against the older true-dtype runs
  without accounting for config differences (`b256/c4096` here vs older
  `b128/c2048`, and older debug-mode runs for some 828 comparisons).
- the important conclusion from this validation pass is the disappearance of the
  fp32 collapse signature and the exact fp32/fp64 artifact match on the tested
  prompts.

## Recent launch update — post-RSS-workstream-A validation

Purpose: validate whether the recent RSS/upcast cleanup phases materially reduce
Grafana-visible RSS jumps while preserving the already validated post-fix exact
compact outputs.

Library commits included in this validation point:

- `d0a567c` — `gate expensive refresh summaries outside debug modes`
- `5ca5a66` — `reduce row append copies in compact attribution`
- `d67ff6e` — `reduce packaging copies in phase5 materialization`
- `2ea911b` — `avoid whole-buffer pinning during cpu staging`
- `bcee92d` — `reuse cpu row staging buffers across batches`

Provenance:

- launch time: `2026-04-22 16:38:22`
- project workspace: `exact-trace-bench-opt@88321f8`
- library workspace: `exact-trace-hidden-knobs-opt@bcee92d`

Launches submitted on Ascend:

- fast validation array job: `5041841`
- anomaly/gate validation array job: `5041842`

Scenario files used:

- `experiments/generated/exact_trace_rss_validation_fast_ascend_scenarios.json`
- `experiments/generated/exact_trace_rss_validation_anomaly_ascend_scenarios.json`

Run metadata:

- fast run id: `20260422_163822_rss-validation-fast-ascend`
- anomaly run id: `20260422_163822_rss-validation-anomaly-ascend`

Submitted scenario matrix:

- `828_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32`
- `828_base` on `ascend/fast` with `exact_trace_internal_dtype=fp64`
- `361_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32`
- `361_base` on `ascend/fast` with `exact_trace_internal_dtype=fp64`
- `94_base` on `ascend/anomaly` with `exact_trace_internal_dtype=fp32`
- `94_base` on `ascend/anomaly` with `exact_trace_internal_dtype=fp64`

Shared runtime config:

- `attribution_batch_size=256`
- `feature_batch_size=256`
- `logit_batch_size=256`
- `decoder_chunk_size=4096`
- `cross_batch_decoder_cache_bytes=0`
- `completions=1`
- `temperature=0.0`
- `max_steps=1`
- `cross_cluster_debug=false`
- `phase4_anomaly_debug=false`
- `save_raw=false`

Expected analysis targets:

- compare Grafana/cgroup RSS jump shape against the earlier post-fix baseline
  validation,
- confirm retained edges remain `20000`,
- confirm fp32/fp64 compact artifacts still match exactly,
- check that runtime and decoder-load counts remain sane while RSS spikes are
  reduced.

Status:

- completed and analyzed.

Analysis provenance:

- analysis time: `2026-04-22 17:05:00`
- project workspace: `exact-trace-bench-opt@88321f8`
- library workspace: `exact-trace-hidden-knobs-opt@bcee92d`

Observed result summary:

- all 6 scenarios finished successfully.
- fp32/fp64 compact artifacts still matched exactly on all three prompts:
  - `mean_feature_jaccard = 1.0`
  - `mean_edge_jaccard = 1.0`
  - `mean_weighted_edge_jaccard = 1.0`
- retained edges remained `20000` in all 6 runs.
- decoder-load counts were unchanged relative to the earlier post-fix baseline:
  - `828_base`: `1107`
  - `361_base`: `1650`
  - `94_base`: `1200`

Runtime comparison vs the earlier post-fix baseline:

- `828_base`:
  - fp32: completion `808.97s -> 684.25s`, Phase 4 wall `542.90s -> 432.78s`
  - fp64: completion `927.49s -> 788.93s`, Phase 4 wall `659.29s -> 532.19s`
- `361_base`:
  - fp32: completion `2130.47s -> 972.95s`, Phase 4 wall `1722.67s -> 609.18s`
  - fp64: completion `2335.41s -> 2456.98s`, Phase 4 wall `1948.25s -> 2088.23s`
- `94_base`:
  - fp32: completion `920.13s -> 721.37s`, Phase 4 wall `650.00s -> 454.93s`
  - fp64: completion `984.86s -> 1062.06s`, Phase 4 wall `696.68s -> 836.20s`

Process RSS snapshot comparison vs the earlier post-fix baseline:

- `828_base`:
  - fp32: `259.01 GiB -> 133.30 GiB`
  - fp64: `258.98 GiB -> 270.45 GiB`
- `361_base`:
  - fp32: `333.97 GiB -> 229.07 GiB`
  - fp64: `333.80 GiB -> 334.25 GiB`
- `94_base`:
  - fp32: `291.01 GiB -> 149.52 GiB`
  - fp64: `291.07 GiB -> 303.80 GiB`

Grafana/cgroup observation from the queue batch (important):

- the major RSS spikes are **still present** at the cgroup/Grafana level.
- job `5041841` was the notable worst case.
- reported pattern:
  - early in the run, `rss` and `cache` rise together steadily,
  - after about 8 minutes, the first large spike appears (`rss ~215 GiB`,
    `cache ~86 GiB`),
  - roughly 4 minutes later, `rss` spikes hard enough that cache begins to clear
    (`rss ~335 GiB`, `cache ~30 GiB`),
  - similar rise/drop cycles then continue roughly every 2 minutes until the end.

Interpretation:

- the recent RSS workstream-A patches clearly helped some **process-level** memory
  behavior, especially on the fp32 path, where completion time and process RSS
  snapshots improved substantially on all three validation prompts.
- however, the Grafana/cgroup evidence says the larger scheduling-relevant RSS
  spike pattern is **not solved yet**.
- this strongly suggests that at least one important source of memory pressure is
  still happening at the cgroup/job level and is not fully captured by the
  current in-process summary metrics.
- fp32/fp64 semantic parity remains excellent, so the remaining problem is now a
  systems/memory-behavior issue, not a correctness issue.

Important instrumentation note:

- the memory-log parser was updated after this run family to handle the new
  `rss_current_gib` field and the affected `result.json` files were reparsed.
- `profiling_summary.peak_rss_gib` / CUDA peak fields are no longer `null` for
  these runs, but they remain **in-process** metrics and should not be confused
  with the larger cgroup/Grafana RSS spike pattern.

Current conclusion:

- Workstream A produced useful improvements, but **not enough** to declare the
  RSS redesign finished.
- the next step should stay inside Workstream A and target the remaining cyclic
  cgroup RSS spikes before moving on to refresh-path speed work.

## Recent launch update — targeted Phase 4 memory telemetry probe

Purpose: use the new lightweight Phase 4 telemetry to localize whether the
remaining scheduler-relevant RSS spikes are dominated by refresh
rescans/materialization or by `compute_batch(...)` / replay transient
allocations.

Library commit included in this probe point:

- `b79706e` — `add targeted phase4 memory attribution telemetry`

Provenance:

- launch time: `2026-04-22 20:04:03 -0400`
- project workspace: `exact-trace-bench-opt@88321f8` (**dirty**)
- library workspace: `exact-trace-hidden-knobs-opt@b79706e` (clean, ahead of
  remote by the telemetry commit)

Project dirty state at launch:

- modified: `EXPERIMENTS.md`
- modified: `PLAN.md`
- modified: `docs/phase4_refresh_optimization_spec.md`
- modified: `experiments/exact_trace_bench/extract.py`
- modified: `experiments/extract_runlog_metrics.py`
- modified: `experiments/run_sparsification_experiment.py`
- untracked: `TODO.md`
- untracked: `experiments/generated/exact_trace_bench/`
- untracked: `experiments/generated/weekend_exact_chunked_fixtures_matched_debug/`

Launch submitted on Ascend:

- fast validation array job: `5044503`

Scenario file used:

- `experiments/generated/exact_trace_phase4_memory_probe_fast_ascend_scenarios.json`

Run metadata:

- fast run id: `20260422_200403_phase4-memory-probe-fast-ascend`
- run name: `phase4_memory_probe_fast_ascend`

Submitted scenario matrix:

- `828_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32`
- `361_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32`

Shared runtime config:

- `attribution_batch_size=256`
- `feature_batch_size=256`
- `logit_batch_size=256`
- `decoder_chunk_size=4096`
- `cross_batch_decoder_cache_bytes=0`
- `completions=1`
- `temperature=0.0`
- `max_steps=1`
- `cross_cluster_debug=false`
- `phase4_anomaly_debug=false`
- `save_raw=false`

Expected analysis targets:

- determine whether the repeated Phase 4 RSS cycles line up more strongly with
  `phase4.refresh` or `context.compute_batch`,
- compare process-vs-cgroup memory deltas from the new telemetry fields,
- inspect row-store read/materialization counters during refresh,
- inspect `compute_batch` transient-allocation indicators (`inject_values_*`,
  `_batch_buffer`, replay-window size/peak),
- choose the next Phase 4 memory fix from evidence instead of guesswork.

Status:

- completed and analyzed.

Analysis provenance:

- analysis time: `2026-04-22 20:33:15 -0400`
- project workspace: `exact-trace-bench-opt@88321f8` (**dirty**)
- library workspace: `exact-trace-hidden-knobs-opt@b79706e`

Observed result summary:

- both scenarios finished successfully.
- job accounting:
  - `5044503_0` (`828_base`) completed in `00:12:11`
  - `5044503_1` (`361_base`) completed in `00:16:50`
- output roots:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_fast_828_base_phase4_memory_probe_fp32_b256_c4096_cache0/`
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_fast_361_base_phase4_memory_probe_fp32_b256_c4096_cache0/`
- runtime snapshots:
  - `828_base`: completion `727.51s`, Phase 4 wall `423.24s`, final resource
    snapshot `rss_gib=133.32`
  - `361_base`: completion `1006.56s`, Phase 4 wall `600.88s`, final resource
    snapshot `rss_gib=229.10`

Telemetry findings (main result):

- each run showed:
  - `8` Phase 4 refreshes,
  - `32` Phase 4 feature batches,
  - `32` Phase 4 `feature_row_store.append_rows` events.
- refresh itself does **not** look like the dominant source of the large cyclic
  RSS growth:
  - `828_base` refresh median cgroup-current delta was only about `+0.011 GiB`
    (max about `+0.090 GiB`),
  - `361_base` refresh median cgroup-current delta was only about `+0.020 GiB`
    (max about `+0.156 GiB`),
  - refresh file-delta was effectively zero in both runs,
  - Phase 4 refresh reported growing row-store reads as expected
    (`1, 4100, 10245, ..., 32772` rows),
  - but refresh reported no Phase 4 dense materialization activity.
- `context.compute_batch(...)` also does **not** look like the main direct source
  of the large net growth:
  - Phase 4 feature-batch median cgroup-current delta per `context.compute_batch`
    event was approximately zero on both prompts,
  - median file delta per `context.compute_batch` event was `0.0 GiB`,
  - median anon delta per `context.compute_batch` event was approximately zero,
  - so the large persistent growth is not happening *inside* the start→end window
    of `compute_batch(...)`.

Stronger localization from event ordering:

- the large stepwise memory growth happens **between** one `context.compute_batch`
  finishing and the next one starting,
- that interval consistently contains `feature_row_store.append_rows`,
- the cgroup/process **file-backed** growth per batch closely matches the Phase 4
  batch buffer size.

Measured pattern:

- `828_base`:
  - `batch_buffer_nbytes ~= 2.857 GiB`
  - median jump between successive Phase 4 `compute_batch.start` cgroup-current
    snapshots: `2.867 GiB`
  - median jump in process file RSS over the same boundary: `2.853 GiB`
- `361_base`:
  - `batch_buffer_nbytes ~= 4.984 GiB`
  - median jump between successive Phase 4 `compute_batch.start` cgroup-current
    snapshots: `5.003 GiB`
  - median jump in process file RSS over the same boundary: `4.981 GiB`

Interpretation:

- the remaining major Phase 4 growth is now much more consistent with
  **file-backed row append / staging behavior** than with refresh rescans or with
  `compute_batch(...)` itself,
- the memory signature is dominated by **file-backed RSS/cache accumulation**,
  not sustained anonymous growth inside the compute step,
- the first Phase 4 batch on each prompt shows an extra one-buffer-scale jump in
  anonymous memory, but the recurring steady growth after that is primarily the
  file-backed row-store path.

New conclusion / next-step recommendation:

- the next implementation pass should target the **Phase 4 row append / CPU
  staging path** in and around:
  - `_copy_rows_to_cpu_staging(...)`
  - `_FileBackedFeatureRowStore.append_rows(...)`
  - surrounding Phase 4 feature-batch post-processing in
    `attribute_nnsight.py`
- refresh rescans are still expensive in time, but they are **not** the best next
  target for the scheduler-relevant RSS spikes,
- `compute_batch(...)` transient allocation cleanup is now secondary to the
  row-store append/writeback/cache-growth path.

Instrumentation note:

- `result.json` `profiling_summary.peak_rss_gib` remained `null` for this probe
  family, so the key conclusions above were taken from `telemetry.jsonl`, job
  accounting, and completion resource snapshots rather than the run-log peak-RSS
  parser.

## Recent launch update — Phase 4 row-store file-IO rerun

Purpose: rerun the same `828_base` / `361_base` Phase 4 memory probe after
replacing the persistent writable row-store memmap with offset-based file I/O,
to test whether the batch-aligned file-backed RSS growth materially improves.

Library commits included in this rerun point:

- `b79706e` — `add targeted phase4 memory attribution telemetry`
- `a370f0d` — `replace exact row store memmap with file io`

Provenance:

- launch time: `2026-04-22 21:25:22 -0400`
- project workspace: `exact-trace-bench-opt@88321f8` (**dirty**)
- library workspace: `exact-trace-hidden-knobs-opt@a370f0d` (clean, ahead of
  remote by two local commits)

Launch submitted on Ascend:

- fast validation array job: `5044673`

Scenario file reused:

- `experiments/generated/exact_trace_phase4_memory_probe_fast_ascend_scenarios.json`

Run metadata:

- fast run id: `20260422_212522_phase4-rowstore-file-io-rerun-fast-ascend`
- run name: `phase4_rowstore_file_io_rerun_fast_ascend`

Submitted scenario matrix:

- `828_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32`
- `361_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32`

Expected analysis targets:

- compare batch-to-batch file RSS growth against the earlier Phase 4 memory probe,
- check whether cgroup/Grafana Phase 4 spike shape improves materially,
- verify that process file RSS no longer tracks batch-buffer size nearly 1:1,
- confirm exact compact outputs remain stable enough for follow-up validation.

Status:

- first submission failed immediately because the scenario names reused existing
  output directories from the earlier Phase 4 memory probe.
- failure cause was operational, not algorithmic:
  `run_sparsification_experiment.py::_assert_fresh_scenario_root(...)` rejected
  reuse of the existing scenario roots for both prompts.
- failed array job: `5044673`
- corrected by creating a new scenario file with unique scenario names and
  resubmitting.

Resubmission:

- resubmission time: `2026-04-22 21:28:21 -0400`
- replacement scenario file:
  `experiments/generated/exact_trace_phase4_rowstore_file_io_rerun_fast_ascend_scenarios.json`
- replacement array job: `5044682`
- replacement run id: `20260422_212821_phase4-rowstore-file-io-rerun-fast-ascend`

- resubmitted; results pending.

## Status of this note

This file is descriptive, not normative.

It is meant to answer “what do we currently have on scratch?” rather than “what
should we run next?”.
