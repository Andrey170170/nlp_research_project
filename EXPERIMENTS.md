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

- completed and analyzed.

Analysis provenance:

- analysis time: `2026-04-22 22:10:00 -0400`
- project workspace: `exact-trace-bench-opt@e73486a` (**dirty**)
- library workspace: `exact-trace-hidden-knobs-opt@a370f0d`

Observed result summary:

- `5044682_0` (`828_base`) completed in `00:30:15`
- `5044682_1` (`361_base`) hit the `01:00:00` walltime and timed out
- the broad read/write file-I/O rewrite was a bad direction:
  - `828_base` completion regressed from about `678.3s` to `1771.1s`
  - `828_base` Phase 4 wall regressed from about `423.2s` to `1499.7s`
  - `828_base` Phase 5 wall regressed from about `0.73s` to `24.51s`
  - `828_base` refresh total regressed from about `160.3s` to `1277.6s`
- process file RSS behavior improved, but the overall tradeoff was unacceptable.

Interpretation:

- replacing the **read path** with explicit file-I/O copies destroyed the cheap
  demand-paged access pattern used by refresh and Phase 5,
- the broad rewrite was therefore rejected,
- the correct lesson was: fix **append/writeback residency** without replacing
  the cheap read/materialization path.

## Recent launch update — Phase 4 hybrid row-store rerun

Purpose: rerun the canonical `828_base` / `361_base` Phase 4 memory probe after
the **hybrid** row-store fix, which preserves the cheap read path and changes
only append/writeback behavior.

Library commits included in this rerun point:

- `b79706e` — `add targeted phase4 memory attribution telemetry`
- `e485fd3` — `Revert "replace exact row store memmap with file io"`
- `db4705a` — `use direct writes for exact row-store appends`

Provenance:

- launch time: `2026-04-22 22:51:15 -0400`
- project workspace: `exact-trace-bench-opt@e73486a` (**dirty**)
- library workspace: `exact-trace-hidden-knobs-opt@db4705a`

Launch submitted on Ascend:

- fast validation array job: `5045541`

Scenario file used:

- `experiments/generated/exact_trace_phase4_hybrid_rowstore_rerun_fast_ascend_scenarios.json`

Run metadata:

- fast run id: `20260422_225115_phase4-hybrid-rowstore-rerun-fast-ascend`
- run name: `phase4_hybrid_rowstore_rerun_fast_ascend`

Submitted scenario matrix:

- `828_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32`
- `361_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32`

Expected analysis targets:

- compare batch-to-batch file RSS growth against the original Phase 4 probe,
- confirm refresh wall time stays near the original probe baseline,
- confirm Phase 5 wall time stays sane,
- check whether the hybrid cut improves cgroup/Grafana behavior without the broad
  file-I/O regression.

Status:

- completed and analyzed.

Analysis provenance:

- analysis time: `2026-04-22 23:20:00 -0400`
- project workspace: `exact-trace-bench-opt@e73486a` (**dirty**)
- library workspace: `exact-trace-hidden-knobs-opt@db4705a`

Observed result summary:

- both scenarios finished successfully.
- job accounting:
  - `5045541_0` (`828_base`) completed in `00:12:21`
  - `5045541_1` (`361_base`) completed in `00:23:36`

Exactness / artifact comparison:

- compared against the original Phase 4 memory probe, the hybrid rerun preserved
  exact compact outputs on both prompts:
  - `mean_feature_jaccard = 1.0`
  - `mean_edge_jaccard = 1.0`
  - `mean_weighted_edge_jaccard = 1.0`

Runtime comparison vs the original Phase 4 memory probe:

- `828_base`:
  - completion `678.30s -> 686.40s`
  - Phase 4 wall `423.24s -> 434.24s`
  - Phase 5 wall `0.73s -> 0.83s`
  - refresh total `160.27s -> 141.14s`
  - final RSS snapshot `133.32 GiB -> 133.29 GiB`
- `361_base`:
  - completion end-to-end `958.62s -> 1365.45s`
  - attribution-only time `956.88s -> 1049.37s`
  - Phase 4 wall `600.89s -> 674.39s`
  - Phase 5 wall `1.08s -> 1.04s`
  - refresh total `291.27s -> 247.71s`
  - final RSS snapshot `229.10 GiB -> 229.08 GiB`

Memory-behavior comparison vs the original Phase 4 memory probe:

- the hybrid fix removed the strong **process file RSS** batch-to-batch growth
  signature:
  - `828_base`: median process-file jump between successive Phase 4 batch starts
    fell from about `2.853 GiB` to about `0.0 GiB`
  - `361_base`: median process-file jump fell from about `4.981 GiB` to about
    `0.001 GiB`
- however, the broader cgroup-visible growth still remained:
  - `828_base`: median cgroup-current jump stayed about `2.867 GiB -> 2.861 GiB`
  - `361_base`: median cgroup-current jump stayed about `5.003 GiB -> 4.993 GiB`

Interpretation:

- the hybrid fix is much better than the rejected broad file-I/O rewrite,
- `828_base` is effectively back near the original probe baseline,
- `361_base` still has a meaningful end-to-end slowdown, but the core
  attribution-path timings remain much closer to baseline than the rejected
  broad rewrite,
- the remaining large cgroup growth looks more like **file-cache / kernel-level
  accounting pressure** than a process-mapped file-RSS problem.

About the weird `361_base` low-memory / low-utilization tail:

- the log shows the main attribution work finished normally:
  - `Feature attributions completed in 674.39s`
  - `Phase 5` remained only about `1.04s`
  - `teardown.cleanup` telemetry was only about `12.91s`
- by teardown start, current memory had already collapsed sharply:
  - `rss_current ≈ 10.25 GiB`
  - `proc_file ≈ 0.45 GiB`
  - `cg_current ≈ 42.95 GiB`
- `completion_end_to_end_seconds` exceeded the attribution-only time by about
  `316s`, so the long tail appears to live **outside** the instrumented
  attribution phases.

Current interpretation of the tail:

- most likely outer pipeline / process-finalization / filesystem cleanup time,
- not ongoing Phase 4 graph exploration,
- not a Phase 5 packaging blow-up,
- and not active GPU attribution work.

Current conclusion:

- the hybrid append/writeback fix is **good enough for now**.
- RSS still grows, but not in a way that currently looks dramatic enough to make
  the jobs unusable.
- the remaining growth appears to be dominated more by **file-cache / broader
  system accounting behavior** than by the original process file-RSS signature.
- the only likely way to remove the problem much more aggressively would be a
  broader redesign of Phase 4 (and possibly even Phase 3), which is out of scope
  for the current safe fix.

## Recent launch update — prompt-diversity validation after the hybrid fix

Purpose: check how the current hybrid fix behaves on other healthy prompts beyond
the canonical one-token base probes.

Library commit included in this run point:

- `db4705a` — `use direct writes for exact row-store appends`

Provenance:

- launch time: `2026-04-23 00:43:14 -0400`
- project workspace: `exact-trace-bench-opt@e73486a` (**dirty**)
- library workspace: `exact-trace-hidden-knobs-opt@db4705a`

Launch submitted on Ascend:

- fast validation array job: `5046318`

Scenario file used:

- `experiments/generated/exact_trace_prompt_diversity_fast_ascend_scenarios.json`

Run metadata:

- fast run id: `20260423_004314_prompt-diversity-fast-ascend`
- run name: `prompt_diversity_fast_ascend`

Submitted scenario matrix:

- `828_late` on `ascend/fast` with `exact_trace_internal_dtype=fp32`
- `361_late` on `ascend/fast` with `exact_trace_internal_dtype=fp32`

Expected analysis targets:

- see whether the hybrid fix generalizes beyond the canonical base one-token
  probes,
- inspect whether later-prefix healthy prompts produce qualitatively different
  RSS or tail behavior,
- compare runtime scaling against the known `828` / `361` base results.

Status:

- completed / analyzed.

Observed result summary:

- both scenarios finished successfully.
- job accounting:
  - `5063198_0` / child job `5065500` (`828_base`) completed in about
    `00:11:32`; harness duration `690.58s`.
  - `5063198_1` (`361_base`) completed in about `00:16:34`; harness duration
    `992.11s`.
- output roots:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_fast_828_base_phase4_locality_validation_v2_fp32_b256_c4096_cache0/`
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_fast_361_base_phase4_locality_validation_v2_fp32_b256_c4096_cache0/`

Exactness / artifact comparison vs the hybrid baseline:

- `828_base`:
  - completion text unchanged: `Here`
  - active features unchanged: `2993540`
  - compact feature Jaccard: `1.0`
  - compact edge Jaccard: `0.995943`
  - weighted edge Jaccard: `0.995224`
- `361_base`:
  - completion text unchanged: `Let`
  - active features unchanged: `5223267`
  - compact feature Jaccard: `1.0`
  - compact edge Jaccard: `0.999492`
  - weighted edge Jaccard: `0.999863`

Runtime comparison vs hybrid baseline and first locality pass:

### `828_base`

- Phase 4 wall-clock:
  - hybrid baseline: `434.24s`
  - locality v1: `449.27s`
  - locality v2: `392.82s`
- attribution time:
  - hybrid baseline: `685.01s`
  - locality v1: `705.71s`
  - locality v2: `645.08s`
- completion end-to-end:
  - hybrid baseline: `686.40s`
  - locality v1: `707.12s`
  - locality v2: `646.57s`
- refreshes / feature batches:
  - hybrid baseline: `8 / 32`
  - locality v1: `10 / 40`
  - locality v2: `9 / 33`
- decoder loads:
  - hybrid baseline: `1107`
  - locality v1: `1077`
  - locality v2: `1095`
- final RSS snapshot:
  - hybrid baseline: `133.29 GiB`
  - locality v1: `136.05 GiB`
  - locality v2: `141.99 GiB`
- cgroup-current max at Phase 4 compute-batch starts remained essentially flat:
  - hybrid baseline: `129.39 GiB`
  - locality v1: `130.19 GiB`
  - locality v2: `129.53 GiB`

Interpretation for `828_base`:

- the refined heuristic fixed the v1 over-splitting regression and turned the
  prompt into a real win:
  - Phase 4 improved about `9.5%` vs hybrid,
  - attribution improved about `5.8%` vs hybrid,
  - v2 reduced v1 batch count from `40` to `33`.
- the final process RSS snapshot was higher, but cgroup-current during Phase 4 was
  not materially worse; treat this as a watch item rather than a blocker.

### `361_base`

- Phase 4 wall-clock:
  - hybrid baseline: `674.39s`
  - locality v1: `504.12s`
  - locality v2: `587.04s`
- attribution time:
  - hybrid baseline: `1049.37s`
  - locality v1: `867.39s`
  - locality v2: `947.01s`
- completion end-to-end:
  - hybrid baseline: `1365.45s`
  - locality v1: `869.12s`
  - locality v2: `948.87s`
- refreshes / feature batches:
  - hybrid baseline: `8 / 32`
  - locality v1: `10 / 37`
  - locality v2: `9 / 34`
- decoder loads:
  - hybrid baseline: `1650`
  - locality v1: `1623`
  - locality v2: `1635`
- final RSS snapshot:
  - hybrid baseline: `229.08 GiB`
  - locality v1: `244.25 GiB`
  - locality v2: `241.62 GiB`
- cgroup-current max at Phase 4 compute-batch starts:
  - hybrid baseline: `197.65 GiB`
  - locality v1: `190.38 GiB`
  - locality v2: `199.70 GiB`

Interpretation for `361_base`:

- locality v2 preserved a useful win vs hybrid, but gave back a large part of v1's
  best-case speedup:
  - Phase 4 remained about `13.0%` faster than hybrid,
  - attribution remained about `9.8%` faster than hybrid,
  - but Phase 4 was about `16.4%` slower than locality v1.
- v2 did reduce v1's batch count from `37` to `34`, and final RSS improved
  slightly vs v1, but cgroup-current max was not better than the hybrid baseline.

Overall conclusion from locality v2:

- the refined conservative split is a cleaner default candidate than locality v1:
  - it removes the `828_base` regression,
  - keeps a meaningful `361_base` speedup,
  - and keeps exact compact artifacts stable.
- it is not the final speed answer:
  - refresh total grew on both prompts vs the hybrid baseline,
  - RSS/cgroup behavior did not materially improve,
  - and the harder prompt still leaves substantial Phase 4 time on the table.
- this supports the current roadmap: keep locality v2 as the safer fallback and
  move the main optimization effort to the explicit Phase 4 frontier planner /
  scheduler path rather than another blind threshold tweak.

## Recent launch update — small multistep validation after the hybrid fix

Purpose: inspect per-step cleanup/tail behavior and confirm that modest multistep
generation remains operationally sane under the current hybrid fix.

Library commit included in this run point:

- `db4705a` — `use direct writes for exact row-store appends`

Provenance:

- launch time: `2026-04-23 00:43:14 -0400`
- project workspace: `exact-trace-bench-opt@e73486a` (**dirty**)
- library workspace: `exact-trace-hidden-knobs-opt@db4705a`

Launch submitted on Ascend:

- fast validation array job: `5046319`

Scenario file used:

- `experiments/generated/exact_trace_multistep_fast_ascend_scenarios.json`

Run metadata:

- fast run id: `20260423_004314_multistep-fast-ascend`
- run name: `multistep_fast_ascend`

Submitted scenario matrix:

- `828_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32` and
  `max_steps=3`
- `361_base` on `ascend/fast` with `exact_trace_internal_dtype=fp32` and
  `max_steps=2`

Expected analysis targets:

- measure whether the weird `361` tail reproduces in multistep generation,
- inspect whether cleanup/finalization cost appears per step or mostly at the end
  of the completion,
- understand whether step-to-step timing stays operationally reasonable.

Status:

- completed / analyzed

Observed result summary:

- `828_base` multistep succeeded for `max_steps=3`
- `361_base` multistep succeeded for `max_steps=2`
- the earlier weird one-step `361` completion tail did **not** reproduce as a
  strong multistep pathology,
- both prompts remained operationally sane under the current hybrid fix,
- Phase 4 still remained the dominant wall-clock cost.

Interpretation:

- the hybrid append/writeback fix is operationally acceptable for modest
  multistep base runs,
- the remaining main within-trace speed target is still Phase 4,
- the old weird `361` tail now looks more like a run-specific outer-tail effect
  than a stable per-step tracing failure.

## Recent launch update — Phase 4 locality / batch-shaping validation

Purpose: validate the first fixed-frontier Phase 4 locality / batch-shaping pass
against the current hybrid baseline on canonical fast prompts.

Code state used:

- library repo commit: `77393db` — `improve phase4 locality-shaped batching`

Scenario file:

- `experiments/generated/exact_trace_phase4_locality_validation_fast_ascend_scenarios.json`

Launch metadata:

- validation array job: `5058313`
- run id: `20260423_phase4-locality-validation-fast-ascend`
- run name: `phase4_locality_validation_fast_ascend`

Scenarios:

- `ascend_fast_828_base_phase4_locality_validation_fp32_b256_c4096_cache0`
- `ascend_fast_361_base_phase4_locality_validation_fp32_b256_c4096_cache0`

Status:

- completed / analyzed

Result summary vs the hybrid fp32 baseline:

### `828_base`

- completion text unchanged: `Here`
- active features unchanged: `2993540`
- retained edges unchanged: `20000`
- Phase 4 wall-clock **regressed slightly**:
  - baseline: `434.24 s`
  - locality pass: `449.27 s`
- total attribution also regressed slightly:
  - baseline: `685.01 s`
  - locality pass: `705.71 s`
- RSS increased modestly:
  - baseline: `133.29 GiB`
  - locality pass: `136.05 GiB`
- decoder loads improved slightly:
  - baseline: `1107`
  - locality pass: `1077`
- Phase 4 refreshes / batches increased:
  - baseline: `8 / 32`
  - locality pass: `10 / 40`

Interpretation for `828_base`:

- the first shaping heuristic appears to have **over-split** the easier prompt,
- small decoder-load / refresh improvements were not enough to offset the added
  batch churn.

### `361_base`

- completion text unchanged: `Let`
- active features unchanged: `5223267`
- retained edges unchanged: `20000`
- Phase 4 wall-clock improved materially:
  - baseline: `674.39 s`
  - locality pass: `504.12 s`
- total attribution improved materially:
  - baseline: `1049.37 s`
  - locality pass: `867.39 s`
- RSS increased:
  - baseline: `229.08 GiB`
  - locality pass: `244.25 GiB`
- decoder loads improved slightly:
  - baseline: `1650`
  - locality pass: `1623`
- Phase 4 refreshes / batches increased:
  - baseline: `8 / 32`
  - locality pass: `10 / 37`

Interpretation for `361_base`:

- the locality direction is likely **real** on the harder base prompt,
- but the first heuristic was too aggressive overall.

Overall conclusion from the first locality pass:

- not ready to call finished,
- keep the locality direction,
- refine the split heuristic to reduce gratuitous over-splitting before deciding
  whether the approach is a clean win.

## Recent launch update — Phase 4 locality / batch-shaping validation v2

Purpose: rerun the canonical fast validation after making the locality split
heuristic more conservative.

Code state used:

- library repo commit: `3592685` — `tune phase4 locality split thresholds`

What changed relative to the first locality pass:

- avoid too-small leading split batches,
- only preserve relatively short trailing layer/chunk runs,
- keep the locality direction while reducing over-splitting pressure.

Scenario file:

- `experiments/generated/exact_trace_phase4_locality_validation_v2_fast_ascend_scenarios.json`

Launch metadata:

- validation array job: `5063198`
- run id: `20260423_phase4-locality-validation-v2-fast-ascend`
- run name: `phase4_locality_validation_v2_fast_ascend`

Scenarios:

- `ascend_fast_828_base_phase4_locality_validation_v2_fp32_b256_c4096_cache0`
- `ascend_fast_361_base_phase4_locality_validation_v2_fp32_b256_c4096_cache0`

Expected analysis targets:

- see whether the `828_base` regression disappears or shrinks,
- see whether a useful share of the `361_base` improvement remains,
- compare refresh counts / batch counts against both the hybrid baseline and the
  first locality run,
- decide whether locality is ready to keep or still needs another refinement.

Status:

- completed / analyzed.

Result summary:

- `828_base`: v2 fixed the first-pass regression and improved over the hybrid
  baseline.
  - hybrid Phase 4 / attribution / scenario duration: `434.24 s` / `674.01 s` /
    `735.71 s`
  - locality v1: `449.27 s` / `705.71 s` / `778.97 s`
  - locality v2: `392.82 s` / `636.25 s` / `690.58 s`
  - refreshes / batches: `8 / 32` hybrid, `10 / 40` v1, `9 / 33` v2
- `361_base`: v2 remained clearly better than hybrid, but gave back some of the
  unusually strong first-pass locality win.
  - hybrid Phase 4 / attribution / scenario duration: `674.39 s` / `1036.10 s` /
    `1412.69 s`
  - locality v1: `504.12 s` / `867.39 s` / `914.68 s`
  - locality v2: `587.04 s` / `932.08 s` / `992.11 s`
  - refreshes / batches: `8 / 32` hybrid, `10 / 37` v1, `9 / 34` v2

Interpretation:

- locality v2 is the safer current fallback/baseline because it removes the
  `828_base` regression while preserving a useful `361_base` improvement over
  hybrid,
- locality v1 remains an interesting hint that harder prompts can benefit more
  from aggressive locality, but it is not the default because it regressed the
  easier prompt.

## Recent launch update — Phase 4 Planner V1 validation

Purpose: validate the first membership-preserving Phase 4 Frontier Planner V1
execution path on canonical fast prompts, using the locality v2 behavior as the
current fallback/baseline candidate.

Code state used:

- project repo commit: `04f1eea` — `prefer effective phase4 scheduler metadata`
- library repo commit: `99b8873` — `tighten phase4 planner plumbing`

Scenario file:

- `experiments/generated/exact_trace_phase4_planner_v1_fast_ascend_scenarios.json`

Launch metadata:

- validation array job: `5066332`
- run id: `20260423_phase4-planner-v1-fast-ascend`
- run name: `phase4_planner_v1_fast_ascend`
- output root: `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast`

Scenarios:

- `ascend_fast_828_base_phase4_planner_v1_fp32_b256_c4096_cache0`
- `ascend_fast_361_base_phase4_planner_v1_fp32_b256_c4096_cache0`

Key settings:

- `phase4_scheduler_mode=planner_v1`
- `phase4_scheduler_debug=true`
- `phase4_scheduler_telemetry_detail=debug`
- `exact_trace_internal_dtype=fp32`
- `feature_batch_size=256`
- `decoder_chunk_size=4096`
- `cross_batch_decoder_cache_bytes=0`

Expected analysis targets:

- confirm Planner V1 selected-frontier membership remains stable relative to the
  current locality path,
- confirm compact outputs / retained edges remain acceptable,
- compare Phase 4 wall-clock, refresh count, and batch count against locality v2,
- inspect new planner telemetry:
  - plan membership/order hashes,
  - boundary reason counts,
  - locality fragmentation summaries,
  - per-batch layer/chunk locality stats.

Status:

- completed / analyzed.

Validation result:

- both array tasks completed successfully (`5066332_0`, `5066332_1`),
- completion text unchanged relative to the comparison runs:
  - `828_base`: `Here`
  - `361_base`: `Let`
- active feature counts unchanged:
  - `828_base`: `2993540`
  - `361_base`: `5223267`
- retained edges unchanged: `20000` in both runs,
- Planner V1 compact outputs matched locality v2 exactly:
  - feature Jaccard: `1.0` for both prompts,
  - edge Jaccard: `1.0` for both prompts,
  - weighted edge Jaccard: `1.0` for both prompts.

Performance summary vs locality v2:

- `828_base`:
  - locality v2 Phase 4 / attribution / scenario duration: `392.82 s` /
    `636.25 s` / `690.58 s`
  - Planner V1 Phase 4 / attribution / scenario duration: `314.78 s` /
    `553.91 s` / `603.81 s`
  - change vs locality v2: Phase 4 `-78.04 s` (`-19.9%`), attribution
    `-82.34 s` (`-12.9%`), scenario duration `-86.77 s` (`-12.6%`)
  - refreshes / batches: unchanged at `9 / 33`
  - refresh elapsed: `180.19 s -> 173.89 s`
  - Phase 4 batch elapsed sum: `212.43 s -> 140.67 s`
- `361_base`:
  - locality v2 Phase 4 / attribution / scenario duration: `587.04 s` /
    `932.08 s` / `992.11 s`
  - Planner V1 Phase 4 / attribution / scenario duration: `580.01 s` /
    `941.48 s` / `998.25 s`
  - change vs locality v2: Phase 4 `-7.03 s` (`-1.2%`), attribution `+9.40 s`
    (`+1.0%`), scenario duration `+6.14 s` (`+0.6%`)
  - refreshes / batches: unchanged at `9 / 34`
  - refresh elapsed: `261.03 s -> 266.20 s`
  - Phase 4 batch elapsed sum: `325.81 s -> 313.59 s`

Telemetry notes:

- Planner V1 emitted expected structured metadata in completion manifests and
  telemetry events:
  - `phase4_scheduler_mode=planner_v1`,
  - `phase4_scheduler_version=planner_v1`,
  - `phase4_scheduler_policy=membership_preserving_locality`,
  - `phase4_scheduler_telemetry_detail=debug`.
- Event counts were consistent with locality v2:
  - `828_base`: `1499` telemetry events,
  - `361_base`: `1968` telemetry events.
- Planner refresh events included membership/order hashes, boundary reason
  counts, invariant summaries, and fragmentation stats.
- Planner batch events showed monotonic chunk order for all Phase 4 batches.

Interpretation:

- Planner V1 is correct enough to keep: it preserved locality-v2 compact outputs
  exactly while enabling richer scheduler telemetry,
- it produced a meaningful `828_base` speedup at the same refresh/batch counts,
  mostly through lower feature-batch elapsed time rather than fewer refreshes,
- `361_base` is effectively neutral against locality v2 and still clearly better
  than hybrid, but worse than the aggressive locality v1 outlier,
- next optimization should use Planner V1 as the instrumentation/control surface
  for a policy experiment rather than returning to hidden cache sweeps.

## Recent launch update — Phase 4 Planner V2 validation

Purpose: validate the first bounded membership-aware Planner V2 policy on
canonical fast prompts, using Planner V1 and locality v2 as the direct comparison
baselines.

Code state used:

- project repo commit: `73ec4b0` — `align planner v2 metadata roundtrip test`
- library repo commit: `aba297b` — `fix planner_v2 scheduler semantics and fallback telemetry`

Scenario file:

- `experiments/generated/exact_trace_phase4_planner_v2_fast_ascend_scenarios.json`

Launch metadata:

- validation array job: `5066777`
- run id: `20260423_phase4-planner-v2-fast-ascend`
- run name: `phase4_planner_v2_fast_ascend`
- output root: `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast`

Scenarios:

- `ascend_fast_828_base_phase4_planner_v2_fp32_b256_c4096_cache0`
- `ascend_fast_361_base_phase4_planner_v2_fp32_b256_c4096_cache0`

Key settings:

- `phase4_scheduler_mode=planner_v2`
- `phase4_scheduler_debug=true`
- `phase4_scheduler_telemetry_detail=debug`
- `exact_trace_internal_dtype=fp32`
- `feature_batch_size=256`
- `decoder_chunk_size=4096`
- `cross_batch_decoder_cache_bytes=0`

Expected analysis targets:

- confirm V2 executes or records explicit Planner V1 fallback per refresh,
- measure membership drift vs Planner V1 reference:
  - selected-node Jaccard,
  - replacement count/fraction,
  - score-sum ratio,
  - rank displacement summaries,
- compare compact artifacts against Planner V1 and locality v2:
  - feature Jaccard,
  - edge Jaccard,
  - weighted edge Jaccard,
  - retained-edge count,
- compare runtime against Planner V1:
  - Phase 4 wall-clock,
  - refresh elapsed,
  - feature-batch elapsed,
  - refresh/batch counts,
- check whether `828_base` keeps the Planner V1 win and whether `361_base` moves
  toward the aggressive locality v1 result without an `828_base` regression.

Status:

- completed / analyzed.

Job status:

- `5066777_0`: completed successfully in `00:10:21`
- `5066777_1`: completed successfully in `00:18:31`

Correctness / artifact summary:

- both scenarios completed successfully,
- completion text unchanged relative to Planner V1 comparison runs:
  - `828_base`: `Here`
  - `361_base`: `Let`
- active feature counts unchanged:
  - `828_base`: `2993540`
  - `361_base`: `5223267`
- retained edges unchanged: `20000` in both runs.

Compact comparison vs Planner V1:

- `828_base`:
  - feature Jaccard: `1.0`
  - edge Jaccard: `0.948515`
  - weighted edge Jaccard: `0.955868`
- `361_base`:
  - feature Jaccard: `1.0`
  - edge Jaccard: `0.922501`
  - weighted edge Jaccard: `0.907005`

Planner V2 telemetry summary:

- V2 executed without fallback on all refreshes:
  - fallback count: `0 / 9` refreshes for both prompts,
  - `scheduler_planner_v2_fallback_reason=None` throughout.
- V2 selected membership changed only on a minority of refreshes:
  - `828_base`: `3 / 9` refreshes changed membership,
  - `361_base`: `2 / 9` refreshes changed membership.
- Average replacement count was small:
  - `828_base`: mean `1.67` replaced nodes / refresh, max `8`,
  - `361_base`: mean `0.89` replaced nodes / refresh, max `5`.
- Score loss stayed within the intended conservative bound:
  - `828_base`: min score-sum ratio `0.999496`, mean `0.999920`,
  - `361_base`: min score-sum ratio `0.999773`, mean `0.999964`.
- Candidate window/reference checks behaved as expected:
  - candidate windows included the reference frontier on all refreshes,
  - candidate window multiplier was `2.0`,
  - no fallback was needed for candidate-window construction.

Performance summary vs Planner V1:

- `828_base`:
  - Planner V1 Phase 4 / attribution / scenario duration: `314.78 s` /
    `562.51 s` / `603.81 s`
  - Planner V2 Phase 4 / attribution / scenario duration: `372.25 s` /
    `586.92 s` / `620.26 s`
  - change vs Planner V1: Phase 4 `+57.47 s` (`+18.3%`), attribution
    `+24.41 s` (`+4.3%`), scenario duration `+16.45 s` (`+2.7%`)
  - refreshes / batches: unchanged at `9 / 33`
  - refresh elapsed: `173.89 s -> 247.55 s`
  - feature-batch elapsed sum improved: `140.67 s -> 124.49 s`
- `361_base`:
  - Planner V1 Phase 4 / attribution / scenario duration: `580.01 s` /
    `956.02 s` / `998.25 s`
  - Planner V2 Phase 4 / attribution / scenario duration: `709.79 s` /
    `1072.91 s` / `1109.54 s`
  - change vs Planner V1: Phase 4 `+129.78 s` (`+22.4%`), attribution
    `+116.90 s` (`+12.2%`), scenario duration `+111.29 s` (`+11.1%`)
  - refreshes / batches: unchanged at `9 / 34`
  - refresh elapsed: `266.20 s -> 380.78 s`
  - feature-batch elapsed sum also worsened: `313.59 s -> 328.80 s`

Performance summary vs other baselines:

- `828_base`: Planner V2 is still faster than locality v2 and hybrid:
  - Phase 4 vs locality v2: `-20.57 s` (`-5.2%`),
  - Phase 4 vs hybrid: `-61.99 s` (`-14.3%`).
- `361_base`: Planner V2 regressed relative to Planner V1, locality v2, and even
  the hybrid Phase 4 baseline:
  - Phase 4 vs locality v2: `+122.75 s` (`+20.9%`),
  - Phase 4 vs hybrid: `+35.40 s` (`+5.2%`),
  - Phase 4 vs aggressive locality v1: `+205.67 s` (`+40.8%`).

Interpretation:

- Planner V2 is **not** worth promoting over Planner V1 in its current form.
- The policy stayed conservative and did not fall back, but even tiny membership
  changes caused noticeable compact-edge drift:
  - about `4.4%` weighted edge drift on `828_base`,
  - about `9.3%` weighted edge drift on `361_base`.
- The main performance problem is refresh cost: V2 increased refresh elapsed by
  about `42–43%` on both prompts. On `828_base`, feature-batch execution improved,
  but the refresh overhead more than erased that gain. On `361_base`, both refresh
  and feature-batch costs worsened.
- Keep Planner V1 as the current best scheduler baseline.
- Keep Planner V2 code/telemetry as an experimental surface, but the next policy
  should either:
  - make V2 selection much cheaper and only apply it when expected locality gain is
    large, or
  - abandon membership-changing selection for now and use Planner V1 plus a
    lower-risk optimization such as adaptive refresh or guided decoder-cache tuning.

## Recent launch update — Phase 4 Execution V1 validation

Purpose: validate the first execution-side optimization tracks under the Planner
V1 scheduler baseline:

1. refresh optimization V1 only,
2. streaming row executor V1 only,
3. combined refresh optimization V1 + streaming row executor V1.

Code state used:

- project repo commit: `d2c9514` — `add phase4 execution validation scenarios`
- library repo commit: `7554b35` — `add phase4 streaming row executor v1`

Scenario files:

- `experiments/generated/exact_trace_phase4_refresh_opt_v1_fast_ascend_scenarios.json`
- `experiments/generated/exact_trace_phase4_streaming_executor_v1_fast_ascend_scenarios.json`
- `experiments/generated/exact_trace_phase4_refresh_streaming_v1_fast_ascend_scenarios.json`

Launch metadata:

- refresh-only array job: `5068122`
  - run id: `20260424_phase4-refresh-opt-v1-fast-ascend`
  - run name: `phase4_refresh_opt_v1`
- streaming-only array job: `5068134`
  - run id: `20260424_phase4-streaming-executor-v1-fast-ascend`
  - run name: `phase4_streaming_executor_v1`
- combined array job: `5068136`
  - run id: `20260424_phase4-refresh-streaming-v1-fast-ascend`
  - run name: `phase4_refresh_streaming_v1`
- output root: `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast`

Scenarios:

- refresh-only:
  - `ascend_fast_828_base_phase4_refresh_opt_v1_fp32_b256_c4096_cache0`
  - `ascend_fast_361_base_phase4_refresh_opt_v1_fp32_b256_c4096_cache0`
- streaming-only:
  - `ascend_fast_828_base_phase4_streaming_executor_v1_fp32_b256_c4096_cache0`
  - `ascend_fast_361_base_phase4_streaming_executor_v1_fp32_b256_c4096_cache0`
- combined:
  - `ascend_fast_828_base_phase4_refresh_streaming_v1_fp32_b256_c4096_cache0`
  - `ascend_fast_361_base_phase4_refresh_streaming_v1_fp32_b256_c4096_cache0`

Key settings:

- `phase4_scheduler_mode=planner_v1`
- `phase4_scheduler_debug=true`
- `phase4_scheduler_telemetry_detail=debug`
- `exact_trace_internal_dtype=fp32`
- `feature_batch_size=256`
- `decoder_chunk_size=4096`
- `cross_batch_decoder_cache_bytes=0`

Variant settings:

| Variant | `phase4_refresh_optimization` | `phase4_row_executor` |
|---|---|---|
| Planner V1 baseline | `off` | `batched` |
| refresh-only | `v1` | `batched` |
| streaming-only | `off` | `streaming_v1` |
| combined | `v1` | `streaming_v1` |

Status:

- all six array tasks completed successfully,
- analyzed against the Planner V1 baseline runs:
  - `ascend_fast_828_base_phase4_planner_v1_fp32_b256_c4096_cache0`,
  - `ascend_fast_361_base_phase4_planner_v1_fp32_b256_c4096_cache0`.

Correctness / artifact summary:

- completion text unchanged relative to Planner V1:
  - `828_base`: `Here`,
  - `361_base`: `Let`,
- active feature counts unchanged:
  - `828_base`: `2993540`,
  - `361_base`: `5223267`,
- retained edges unchanged: `20000` in all runs,
- compact outputs matched Planner V1 exactly for all six variant/prompt runs:
  - feature arrays equal,
  - edge arrays exact,
  - feature Jaccard: `1.0`,
  - edge Jaccard: `1.0`,
  - weighted edge Jaccard: `1.0`,
  - max common retained-edge weight difference: `0.0`.

Performance summary vs Planner V1:

### `828_base`

| Variant | Phase 4 wall-clock | Refresh elapsed | Feature-batch / executor elapsed | Attribution | Completion / end-to-end | RSS snapshot |
|---|---:|---:|---:|---:|---:|---:|
| Planner V1 | `314.78 s` | `173.89 s` | inferred `140.90 s` | `562.51 s` | `563.91 s` | `142.05 GiB` |
| refresh-only | `420.51 s` | `233.23 s` | `187.07 s` | `675.23 s` | `676.73 s` | `227.94 GiB` |
| streaming-only | `532.00 s` | `176.87 s` | `354.40 s` | `770.50 s` | `771.92 s` | `139.91 GiB` |
| combined | `587.88 s` | `237.26 s` | `349.87 s` | `785.34 s` | `786.78 s` | `225.84 GiB` |

Changes vs Planner V1:

- refresh-only:
  - Phase 4 `+105.73 s` (`+33.6%`),
  - refresh `+59.34 s` (`+34.1%`),
  - completion/end-to-end `+112.82 s` (`+20.0%`),
  - RSS snapshot `+85.90 GiB` (`+60.5%`).
- streaming-only:
  - Phase 4 `+217.22 s` (`+69.0%`),
  - refresh `+2.99 s` (`+1.7%`),
  - feature-batch/executor `140.90 s -> 354.40 s`,
  - completion/end-to-end `+208.01 s` (`+36.9%`),
  - RSS snapshot `-2.14 GiB` (`-1.5%`).
- combined:
  - Phase 4 `+273.10 s` (`+86.8%`),
  - refresh `+63.38 s` (`+36.4%`),
  - feature-batch/executor `140.90 s -> 349.87 s`,
  - completion/end-to-end `+222.87 s` (`+39.5%`),
  - RSS snapshot `+83.79 GiB` (`+59.0%`).

### `361_base`

| Variant | Phase 4 wall-clock | Refresh elapsed | Feature-batch / executor elapsed | Attribution | Completion / end-to-end | RSS snapshot |
|---|---:|---:|---:|---:|---:|---:|
| Planner V1 | `580.01 s` | `266.20 s` | inferred `313.81 s` | `956.02 s` | `957.78 s` | `241.59 GiB` |
| refresh-only | `766.20 s` | `424.62 s` | `341.33 s` | `1126.38 s` | `1128.52 s` | `333.47 GiB` |
| streaming-only | `551.50 s` | `272.66 s` | `278.11 s` | `912.48 s` | `914.42 s` | `237.88 GiB` |
| combined | `777.41 s` | `399.43 s` | `377.26 s` | `1093.60 s` | `1095.36 s` | `334.12 GiB` |

Changes vs Planner V1:

- refresh-only:
  - Phase 4 `+186.19 s` (`+32.1%`),
  - refresh `+158.41 s` (`+59.5%`),
  - completion/end-to-end `+170.74 s` (`+17.8%`),
  - RSS snapshot `+91.89 GiB` (`+38.0%`).
- streaming-only:
  - Phase 4 `-28.51 s` (`-4.9%`),
  - refresh `+6.45 s` (`+2.4%`),
  - feature-batch/executor `313.81 s -> 278.11 s`,
  - completion/end-to-end `-43.36 s` (`-4.5%`),
  - RSS snapshot `-3.70 GiB` (`-1.5%`).
- combined:
  - Phase 4 `+197.40 s` (`+34.0%`),
  - refresh `+133.23 s` (`+50.0%`),
  - feature-batch/executor `313.81 s -> 377.26 s`,
  - completion/end-to-end `+137.58 s` (`+14.4%`),
  - RSS snapshot `+92.53 GiB` (`+38.3%`).

Telemetry / substage notes:

- Refresh-only V1 did reduce row-reader over-read to zero, but row-store read time
  was tiny relative to the full refresh cost:
  - `828_base`: row-store read `3.62 s` inside `233.23 s` refresh total,
  - `361_base`: row-store read `7.09 s` inside `424.62 s` refresh total.
- Refresh-only V1 made the partial-influence loop much slower:
  - `828_base`: partial influence `229.26 s`; matmul `113.50 s`; remaining
    loop/zero/scatter overhead about `110.53 s`,
  - `361_base`: partial influence `418.44 s`; matmul `160.71 s`; remaining
    loop/zero/scatter overhead about `249.10 s`.
- The active-row refresh strategy therefore saved little I/O while adding large
  zero-fill/scatter and RSS cost.
- Streaming-only changed executor granularity from scheduler/reference batches to
  microbatches:
  - `828_base`: `33` scheduler batches -> `132` executor microbatches,
  - `361_base`: `34` scheduler batches -> `131` executor microbatches.
- On `828_base`, streaming-only regressed primarily through denominator work:
  - executor denominator elapsed `266.50 s`,
  - total executor elapsed `354.40 s`.
- On `361_base`, streaming-only improved feature-batch/executor elapsed:
  - inferred Planner V1 executor `313.81 s`,
  - streaming-only executor `278.11 s`,
  - but refresh was still slightly slower (`+6.45 s`).

Interpretation:

- **Do not promote refresh optimization V1.** It is exact, but it is a clear
  performance and RSS regression on both canonical fast prompts.
- The active-row refresh approach targeted the wrong bottleneck: row-store read
  time is only a few seconds, while fixed-shape zero-fill/scatter and matmul work
  dominate the refresh loop.
- **Do not run the combined variant further** until refresh V1 is replaced or
  disabled; the refresh penalty dominates the combined behavior.
- Streaming row executor V1 is **experimental only**, not a default:
  - it is a severe regression on `828_base`,
  - it is a modest win on `361_base`,
  - the hard-coded `64` microbatch cap is likely too small for easier/smaller
    prompts and should be made adaptive before further promotion.
- Candidate next actions:
  - leave refresh V1 feature-gated and mark it rejected/not promoted,
  - keep streaming V1 feature-gated,
  - test a larger/adaptive streaming microbatch cap, e.g. `128` or “use 64 only
    for very large active-feature cases,” before considering another validation
    matrix.

## Recent launch update — Planner V1 hidden optimization knobs matrix

Purpose: compare the current Planner V1 baseline against a set of newly added,
explicitly opt-in exact/chunked hidden optimization knobs on Ascend/A100. This
matrix is intended to isolate each new knob before any combination runs.

Code state used at launch:

- project repo commit: `d2c9514` — `add phase4 execution validation scenarios`
  - project worktree had uncommitted plumbing/scenario/doc changes that were
    included in the immutable launch snapshot,
- library repo commit: `f7f7318` — `stage active encoders on cpu`
  - library worktree was clean and ahead of origin by six local commits.

Scenario file:

- `experiments/generated/exact_trace_bench/exact_trace_hidden_knobs_planner_v1_ascend_scenarios.json`

Launch metadata:

- SLURM array job: `5101442` on cluster `ascend`, array indices `0-13`
- run id / launch id: `20260427_165337_015086_planner-v1-hidden-knobs-matrix`
- run name: `planner-v1 hidden knobs matrix`
- output root:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260427_165337_015086_planner-v1-hidden-knobs-matrix`
- immutable project snapshot:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260427_165337/nlp_research_project`
- immutable library snapshot:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260427_165337/circuit-tracer_chunked`

Scenarios:

- Planner V1 baseline for `828_base` and `361_base`
- `phase1_cap64` for `828_base` and `361_base`
- `deferred_refresh_x2` for `828_base` and `361_base`
- `topk_ranker` for `828_base` and `361_base`
- `rowstore_fadvise` for `828_base` and `361_base`
- `active_encoder_cpu` for `828_base` and `361_base`
- `active_encoder_pinned_cpu` for `828_base` and `361_base`

Key settings:

- `phase4_scheduler_mode=planner_v1`
- `attribution_batch_size=128`
- `feature_batch_size=128`
- `logit_batch_size=128`
- `decoder_chunk_size=2048`
- `cross_batch_decoder_cache_bytes=0`
- `exact_trace_internal_dtype=fp32`
- `phase4_refresh_optimization=off`
- `phase4_row_executor=batched`
- `phase4_scheduler_telemetry_detail=normal`

Variant settings under test:

| Variant | Isolated setting |
|---|---|
| baseline | hidden knobs left at default / legacy behavior |
| `phase1_cap64` | `phase1_trace_batch_policy=cap_effective_batches`, `phase1_trace_batch_size_max=64` |
| `deferred_refresh_x2` | `phase4_refresh_policy=deferred_v1`, `phase4_refresh_interval_multiplier=2` |
| `topk_ranker` | `phase4_ranker=topk_v1` |
| `rowstore_fadvise` | `row_store_cache_control=fadvise_dontneed_after_append_v1` |
| `active_encoder_cpu` | `exact_encoder_residency=active_cpu` |
| `active_encoder_pinned_cpu` | `exact_encoder_residency=active_pinned_cpu` |

Pre-submit validation:

- project lint passed for the touched launcher/scenario plumbing,
- scenario list loaded all 14 scenarios,
- dry-run for one deferred-refresh scenario confirmed the expected Planner V1 and
  hidden-knob CLI flags were emitted,
- launch-plan created immutable project + sibling library snapshots before
  submission.

Status / interpretation notes:

- all 14 array tasks completed successfully,
- extraction / comparison outputs:
  - `experiments/extracted/exact_trace_bench/planner_v1_hidden_knobs_matrix/benchmark_enriched.csv`,
  - `experiments/extracted/exact_trace_bench/planner_v1_hidden_knobs_matrix/analysis_summary.csv`,
  - `experiments/extracted/exact_trace_bench/planner_v1_hidden_knobs_matrix/compact_comparisons.csv`,
  - per-variant `compare_<prompt>_<variant>_vs_baseline.json` files,
- use this run, not the earlier locality/default sanity job, as the main hidden
  knobs comparison matrix,
- treat `deferred_refresh_x2` as deliberately non-exact-equivalent: record drift
  rather than requiring strict compact equality.

Performance summary vs in-matrix Planner V1 baseline:

### `828_base`

| Variant | Duration | Δ duration | Phase 4 | Δ Phase 4 | Refresh count | Refresh elapsed | Feature-batch elapsed | sacct MaxRSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `960.77 s` | — | `574.39 s` | — | `18` | `333.94 s` | `240.03 s` | `173.64 GiB` |
| `phase1_cap64` | `1236.84 s` | `+28.7%` | `974.43 s` | `+69.7%` | `35` | `724.63 s` | `248.96 s` | `174.22 GiB` |
| `deferred_refresh_x2` | `1310.68 s` | `+36.4%` | `1026.60 s` | `+78.7%` | `9` | `171.17 s` | `855.02 s` | `172.53 GiB` |
| `topk_ranker` | `889.19 s` | `-7.5%` | `622.92 s` | `+8.5%` | `18` | `311.41 s` | `311.08 s` | `173.08 GiB` |
| `rowstore_fadvise` | `871.86 s` | `-9.2%` | `607.12 s` | `+5.7%` | `18` | `323.48 s` | `283.22 s` | `173.13 GiB` |
| `active_encoder_cpu` | `694.82 s` | `-27.7%` | `498.51 s` | `-13.2%` | `18` | `319.55 s` | `178.55 s` | `148.01 GiB` |
| `active_encoder_pinned_cpu` | `790.34 s` | `-17.7%` | `574.92 s` | `+0.1%` | `18` | `327.25 s` | `247.24 s` | `150.58 GiB` |

### `361_base`

| Variant | Duration | Δ duration | Phase 4 | Δ Phase 4 | Refresh count | Refresh elapsed | Feature-batch elapsed | sacct MaxRSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `1404.36 s` | — | `1004.43 s` | — | `18` | `614.22 s` | `389.80 s` | `265.74 GiB` |
| `phase1_cap64` | `1723.88 s` | `+22.8%` | `1384.11 s` | `+37.8%` | `36` | `1015.65 s` | `367.65 s` | `276.89 GiB` |
| `deferred_refresh_x2` | `987.72 s` | `-29.7%` | `609.79 s` | `-39.3%` | `9` | `280.82 s` | `328.57 s` | `273.72 GiB` |
| `topk_ranker` | `1392.67 s` | `-0.8%` | `1101.06 s` | `+9.6%` | `18` | `554.98 s` | `545.62 s` | `240.40 GiB` |
| `rowstore_fadvise` | `1223.19 s` | `-12.9%` | `848.12 s` | `-15.6%` | `18` | `512.64 s` | `335.07 s` | `273.43 GiB` |
| `active_encoder_cpu` | `1329.42 s` | `-5.3%` | `975.60 s` | `-2.9%` | `18` | `610.63 s` | `364.54 s` | `263.76 GiB` |
| `active_encoder_pinned_cpu` | `1281.57 s` | `-8.7%` | `950.49 s` | `-5.4%` | `18` | `585.31 s` | `364.75 s` | `256.87 GiB` |

Compact comparison vs in-matrix baseline:

- active feature sets matched exactly for every variant/prompt pair
  (`feature_jaccard=1.0`, active features `2993540` for `828_base` and
  `5223267` for `361_base`),
- exact compact match by edge set / weighted edge Jaccard for:
  - `topk_ranker`,
  - `rowstore_fadvise`,
  - `active_encoder_cpu`,
  - `active_encoder_pinned_cpu`,
- `deferred_refresh_x2` drifted slightly, as expected:
  - `828_base`: edge Jaccard `0.998815`, weighted edge Jaccard `0.998569`,
  - `361_base`: edge Jaccard `0.997632`, weighted edge Jaccard `0.997697`,
- `phase1_cap64` unexpectedly drifted in retained edge weights/membership:
  - `828_base`: edge Jaccard `1.0`, weighted edge Jaccard `0.999991`,
  - `361_base`: edge Jaccard `0.994426`, weighted edge Jaccard `0.994663`.

Interpretation:

- **Do not promote `phase1_cap64`.** It regressed runtime and also introduced
  compact-output drift, especially on `361_base`.
- **Do not combine or promote `deferred_refresh_x2` yet.** It is promising for
  `361_base` but a severe `828_base` regression and deliberately changes retained
  edge outputs.
- **`topk_ranker` is exact in this matrix but not a clear speed win.** Keep it
  feature-gated; useful mostly as a low-risk implementation cleanup / tie-behavior
  experiment.
- **`rowstore_fadvise` is exact and improved wall time in both prompts**
  (`-9.2%` / `-12.9%`), but did not reduce sacct MaxRSS here; keep as a good
  candidate for combination because it is simple and output-preserving.
- **`active_encoder_cpu` is the strongest isolated result for `828_base`**:
  `-27.7%` duration and `173.64 -> 148.01 GiB` sacct MaxRSS, with exact compact
  output.
- **`active_encoder_pinned_cpu` is also exact and reduces memory**, but plain CPU
  was better on `828_base`; pinned CPU was modestly better than plain CPU on
  `361_base` duration / MaxRSS.
- Best next combination candidates are `rowstore_fadvise` plus one active encoder
  residency mode. Prefer testing both `active_encoder_cpu` and
  `active_encoder_pinned_cpu` in combination because their prompt-wise behavior
  differs.

Preceding sanity run:

- job `5100027` completed successfully on Ascend and verified the new hidden knobs
  defaulted to legacy/off behavior, but it accidentally used
  `phase4_scheduler_mode=locality` rather than Planner V1, so it is only a
  default-safety/completion check.

## Status of this note

This file is descriptive, not normative.

It is meant to answer “what do we currently have on scratch?” rather than “what
should we run next?”.
