# Experiments inventory

This file is a living note for experiment artifacts currently stored on scratch
and a dated investigation log for important decisions, launches, findings, and
result reinterpretations.

For now it documents:

- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench`

The goal is to make it easy to answer:

- what experiment families exist,
- where they live,
- what each family was trying to test,
- and which runs should be treated as baseline / debug / anomaly / exploratory.

## Investigation log conventions

For important updates, include:

- date,
- project repo provenance,
- sibling library provenance,
- whether the run came from a live workspace or immutable snapshot,
- enough context to reconstruct later why we trusted or reinterpreted a
  baseline.

## Recent investigation updates

### 2026-04-22 — intended next diagnostic batch: 6 matched `94_base` jobs

Purpose:

- use one queued batch to both **diagnose** the Phase-0 divergence and **test** a
  narrow mitigation without needing a second edit-submit cycle,
- keep the next batch focused on `94_base`, since that is the prompt of most
  immediate interest for the cross-cluster anomaly story.

Planning provenance:

- project repo: `nlp_research_project`
  - workspace: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - commit baseline for planning: `94e4283a9e2b0c04d63409890b5936c07c17108c`
  - current workspace state: live workspace with pending local edits to docs,
    scripts, and generated scenario files before the next implementation snapshot
- sibling library repo: `circuit-tracer_chunked`
  - workspace: main sibling workspace `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit baseline for planning: `d1f3df3e456fdb5430b5462a0a6834bdaf7fa716`

Queueing / launch policy for the next batch:

- use immutable workspace snapshots only,
- keep default sbatch walltime at `01:00:00` on both clusters,
- keep the batch fully matched across clusters except for the explicit Phase-0
  compare-mode variant under test.

Shared intended config across all six jobs:

- fixture: `94_base`
- tier: `anomaly`
- `max_steps=1`
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
- `decoder_chunk_size=2048`
- `cross_batch_decoder_cache_bytes=0`
- `temperature=0.0`
- `completions=1`
- `max_feature_nodes=8192`
- `max_edges=20000`
- `attribution_batch_size=128`
- `feature_batch_size=128`
- `logit_batch_size=128`
- `max_n_logits=3`
- `desired_logit_prob=0.8`
- `attribution_update_interval=4`
- `feature_batch_target_reserved_fraction=0.9`
- `feature_batch_min_free_fraction=0.05`
- `feature_batch_probe_batches=1`
- `verbose_attribution=true`
- `profile_attribution=true`

Planned six-job matrix:

1. Ascend baseline Phase-0 compare path
2. Cardinal baseline Phase-0 compare path
3. Ascend Phase-0 compare upcast = `fp32`
4. Cardinal Phase-0 compare upcast = `fp32`
5. Ascend Phase-0 compare upcast = `fp64`
6. Cardinal Phase-0 compare upcast = `fp64`

Additional diagnostics intended in this batch:

- Phase-0:
  - canonical sorted active-feature hash,
  - near-threshold counts,
  - sample borderline features near the JumpReLU threshold,
  - clearer separation of membership drift vs ordering-only drift.
- Phase-3:
  - top-K seed influences,
  - cutoff margin / near-tie counts,
  - richer shadow-frontier overlap diagnostics,
  - more explicit row-input / row-abs-sum comparison fields.

Decision rationale:

- a 4-job batch (baseline + one mitigation) would tell us whether *some*
  upcast helps,
- the chosen 6-job batch should tell us whether:
  - no targeted upcast helps,
  - fp32 is already sufficient,
  - or fp64 gives a materially better reduction in Phase-0 and Phase-3 drift.

Current expectation before launch:

- if the Phase-0 divergence is primarily caused by bf16 threshold flips in the
  transcoder sparse encode path, then the fp32/fp64 compare variants should
  reduce active-feature membership drift relative to the baseline pair,
- if drift remains essentially unchanged across all three modes, then the root
  cause is likely earlier model/transcoder activation differences rather than the
  final threshold compare precision alone.

### 2026-04-22 — first analysis of matched cross-cluster debug pairs (`828_base` / `94_base`)

Runs analyzed:

- `828_base` fast:
  - Ascend output: `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260422_001758_409577_matched-cross-cluster-828-fast-ascend/ascend_matched_cross_cluster_828_base_fast_b128_c2048_cache0g`
  - Cardinal output: `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/20260422_001758_483504_matched-cross-cluster-828-fast-cardinal/cardinal_matched_cross_cluster_828_base_fast_b128_c2048_cache0g`
- `94_base` anomaly:
  - Ascend output: `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_001758_549185_matched-cross-cluster-94-anomaly-ascend/ascend_matched_cross_cluster_94_base_anomaly_b128_c2048_cache0g`
  - Cardinal output: `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260422_001758_619013_matched-cross-cluster-94-anomaly-cardinal/cardinal_matched_cross_cluster_94_base_anomaly_b128_c2048_cache0g`

Important scope note:

- these are still **single-step** (`max_steps=1`) debug runs,
- so they are good for locating the first divergence checkpoint, but they do not
  directly test later-step "refresh every step" behavior.

Observed `828_base` results:

- both clusters generated the same first token: `Here` (`token_id=8291`),
- next-token logprob remained very close (`Δ ≈ 1.45e-05`),
- Phase-1 target-logit hash matched exactly,
- first hash divergence appeared already at `phase0_sparse_setup`,
- compact artifact comparison:
  - feature Jaccard `0.99343`
  - edge Jaccard `0.45940`
  - weighted edge Jaccard `0.49887`
- Phase-4 wall-clock still differed substantially:
  - Ascend `1051.30s`
  - Cardinal `564.60s`

Observed `94_base` results:

- both clusters generated the same first token: `Let` (`token_id=6481`),
- next-token logprob stayed close but less tightly than `828_base`
  (`Δ ≈ 7.65e-04`),
- Phase-1 target-logit hash matched exactly,
- first hash divergence again appeared at `phase0_sparse_setup`,
- compact artifact comparison:
  - feature Jaccard `0.99362`
  - edge Jaccard `0.57094`
  - weighted edge Jaccard `0.61281`
- Phase-4 wall-clock still differed substantially:
  - Ascend `1201.76s`
  - Cardinal `818.57s`

Checkpoint interpretation:

- for **both prompts**, the first structural divergence appears in
  `phase0_sparse_setup` via differing retained feature sets / active-feature
  hashes,
- despite that early structural difference, `phase1_target_logits` still matched
  exactly on the chosen first token for both prompts,
- by `phase3_seed_ranking_pre_phase4`, the ranking/frontier hashes diverged on
  both prompts, so Phase 3 is the first checkpoint where the difference is more
  clearly expressed in the actual frontier/ranking state,
- `phase4_entry` itself had no dedicated hash difference record, but Phase-4
  runtime remained materially different across clusters.

Current interpretation:

- under matched `fp64` + matched `decoder_chunk_size=2048`, the first-step
  cross-cluster story is **not** "Cardinal immediately collapses while Ascend
  stays healthy" for `94_base`,
- instead, both prompts show the same pattern:
  - early feature-set divergence at Phase 0,
  - matching target-logit state at Phase 1,
  - frontier/ranking divergence visible by Phase 3,
  - substantial Phase-4 runtime differences despite the matched tracing
    contract,
- this supports treating Phase 0 as the first structural divergence point and
  Phase 3 as the first obviously downstream-meaningful divergence point,
- to test the earlier overflow / repeated-refresh hypothesis directly, we likely
  need a later-step or multi-step matched run on `94_base`, not just the current
  single-step check.

### 2026-04-22 — launched matched cross-cluster debug pairs for `828_base` and `94_base`

Purpose:

- start the main-branch cross-cluster investigation with **matched configs** so
  cluster is the intended primary source of variance,
- pair one healthy control prompt (`828_base` in `fast`) with one anomalous
  prompt (`94_base` in `anomaly`).

Launch provenance:

- source project workspace before snapshot: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - commit: `94e4283a9e2b0c04d63409890b5936c07c17108c`
- source sibling library before snapshot: `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `d1f3df3e456fdb5430b5462a0a6834bdaf7fa716`
- immutable snapshot container:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260422_001720_matched_cross_cluster_828_94_fp64`
- snapshot project root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260422_001720_matched_cross_cluster_828_94_fp64/nlp_research_project`
- snapshot library root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260422_001720_matched_cross_cluster_828_94_fp64/circuit-tracer_chunked`

Matched launch config used on **all four** runs:

- exact tracing, single-step (`max_steps=1`)
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
- `decoder_chunk_size=2048`
- `cross_batch_decoder_cache_bytes=0`
- `temperature=0.0`
- `completions=1`
- `max_feature_nodes=8192`
- `max_edges=20000`
- `attribution_batch_size=128`
- `feature_batch_size=128`
- `logit_batch_size=128`
- `max_n_logits=3`
- `desired_logit_prob=0.8`
- `attribution_update_interval=4`
- `feature_batch_target_reserved_fraction=0.9`
- `feature_batch_min_free_fraction=0.05`
- `feature_batch_probe_batches=1`
- `verbose_attribution=true`
- `profile_attribution=true`

Scenario files used:

- `experiments/generated/exact_trace_bench/matched_cross_cluster_828_fast_ascend.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_828_fast_cardinal.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_ascend.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_cardinal.json`

Submitted jobs:

- Ascend `828_base` fast:
  - SLURM job `5024202`
  - run id `20260422_001758_409577_matched-cross-cluster-828-fast-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260422_001758_409577_matched-cross-cluster-828-fast-ascend`
- Cardinal `828_base` fast:
  - SLURM job `8694599`
  - run id `20260422_001758_483504_matched-cross-cluster-828-fast-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/20260422_001758_483504_matched-cross-cluster-828-fast-cardinal`
- Ascend `94_base` anomaly:
  - SLURM job `5024203`
  - run id `20260422_001758_549185_matched-cross-cluster-94-anomaly-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_001758_549185_matched-cross-cluster-94-anomaly-ascend`
- Cardinal `94_base` anomaly:
  - SLURM job `8694600`
  - run id `20260422_001758_619013_matched-cross-cluster-94-anomaly-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260422_001758_619013_matched-cross-cluster-94-anomaly-cardinal`

Interpretation / intent:

- these launches intentionally remove the earlier `decoder_chunk_size`
  asymmetry (`2048` on Ascend vs `4096` on Cardinal) from the initial
  cross-cluster comparison,
- the point of this run set is not to preserve each cluster's historical native
  benchmark config, but to isolate where the two clusters first diverge under a
  **matched tracing contract**,
- analysis is still pending; this entry records the exact launch decision and
  provenance so later interpretation can reconstruct what was submitted.

### 2026-04-22 — post-library-merge within-cluster baseline consistency check

Purpose:

- verify that the merged main baseline did not silently change `828_base`
  outputs **within each cluster** before starting cross-cluster drift analysis.

Project + library provenance for the checked baseline pair:

- project repo: `nlp_research_project`
  - workspace: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - commit: `94e4283a9e2b0c04d63409890b5936c07c17108c`
- sibling library repo: `circuit-tracer_chunked`
  - workspace: main sibling workspace `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `d1f3df3e456fdb5430b5462a0a6834bdaf7fa716`

Runs checked:

- Ascend current baseline-check run:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_quick_fp64_cross_cluster_828_base_postlibmerge_b128_c2048_cache0g`
  - launched from the live main workspace
- Ascend same-cluster reference run:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_quick_fp64_cross_cluster_828_base_b128_c2048_cache0g`
- Cardinal current baseline-check run:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/20260421_214702_004280_quick-cross-cluster-fp64-828-postlibmerge-snapshot/cardinal_quick_fp64_cross_cluster_828_base_postlibmerge_b128_c4096_cache0g`
  - launched from immutable snapshot workspace `workspace_20260421_214702_cardinal_828_postlibmerge`
- Cardinal same-cluster reference run:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/20260420_131422_618249_prompt828-debug-cardinal-float64-norm-general/cardinal_fast_828_base_b128_c4096_cache0g`

Validation notes:

- live main `trace_pipeline_chunked.py` matched the Cardinal snapshot copy by
  file hash,
- live main `circuit_tracer/attribution/attribute_nnsight.py` matched the
  Cardinal snapshot copy by file hash.

Observed within-cluster comparisons:

- Ascend matched exactly against the earlier same-cluster fp64 quick run for:
  - generated token (`Here`),
  - next-token id/logprob,
  - active feature count (`2993540`),
  - telemetry event count (`2760`),
  - saved `step_000.npz` file hash,
  - saved `feature_ids`, `row_idx`, `col_idx`, and `weights` hashes.
- Cardinal matched exactly against the earlier same-cluster prompt-828 debug
  run for:
  - generated token (`Here`),
  - next-token id/logprob,
  - active feature count (`2993606`),
  - telemetry event count (`1952`),
  - saved `step_000.npz` file hash,
  - saved `feature_ids`, `row_idx`, `col_idx`, and `weights` hashes.
- Cardinal run-metadata/debug-mode fields differed (`cross_cluster_debug` in
  the new run versus `phase4_anomaly_debug` in the older debug campaign), but
  the saved compact step artifact for `828_base` was unchanged within Cardinal.

Interpretation:

- this check supports that we did **not** accidentally move onto a bad merged
  baseline for either cluster,
- treat the project/library pair `94e4283` + `d1f3df3` as a validated baseline
  pair for continuing the **cross-cluster** investigation,
- remaining Ascend/Cardinal differences are expected and should be investigated
  as actual cross-cluster drift rather than as evidence of baseline corruption.

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

Main findings from the true-dtype comparison:

- `828_base` and `361_base` genuinely collapse under true fp32 internal dtype
- `94_base` does not collapse via the same mechanism
- fp64 should remain the default runtime dtype for exact compact tracing
- the healthy-prompt speedups previously observed are best explained by the
  float64 normalization path, not by the refresh-cache attempt

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
- for long-token / long-trace scaling, fp64 is likely the right immediate
  default but not the final permanent solution if raw normalization sums can
  keep growing toward fp64 overflow

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

## Status of this note

This file is descriptive, not normative.

It is meant to answer “what do we currently have on scratch?” rather than “what
should we run next?”.
