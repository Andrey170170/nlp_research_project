# Generated experiment configs

Status: Scenario provenance index
Last updated: 2026-05-18

This directory mixes current exact-bench templates with historical one-off launch
configs. Treat this README as the first routing check before copying a JSON file
for a new run.

## Current canonical exact-bench templates

Use these for ordinary `experiments.exact_trace_bench` launches:

- `exact_trace_bench/exact_trace_bench_fast_ascend_scenarios.json`
- `exact_trace_bench/exact_trace_bench_fast_cardinal_scenarios.json`
- `exact_trace_bench/exact_trace_bench_anomaly_ascend_scenarios.json`
- `exact_trace_bench/exact_trace_bench_anomaly_cardinal_scenarios.json`
- `exact_trace_bench/exact_trace_bench_long_eval_ascend_scenarios.json`
- `exact_trace_bench/exact_trace_bench_long_eval_cardinal_scenarios.json`

Regenerate them with:

```bash
uv run python -m experiments.exact_trace_bench build-scenarios --all-tiers --all-clusters
```

Canonical rows should keep only ordinary fixture/source metadata, batch sizes,
`decoder_chunk_size`, and `cross_batch_decoder_cache_bytes`. Default-valued
advanced/debug settings live in the config defaults and are guarded by tests.

## Explicit advanced sweep configs

These are not ordinary templates, but remain useful as prior sweep/probe inputs:

- `exact_trace_bench/exact_trace_cache8g_interaction_ascend_scenarios.json`
- `exact_trace_bench/exact_trace_hidden_knobs_planner_v1_ascend_scenarios.json`
- `exact_trace_bench/exact_trace_wave1_cache_chunk_resweep_ascend_scenarios.json`
- root-level `exact_trace_phase4_*_scenarios.json`
- `exact_batch_tuning_scenarios.json`
- `exact_cache_sweep_scenarios.json`
- `exact_trace_multistep_fast_ascend_scenarios.json`
- `exact_trace_prompt_diversity_fast_ascend_scenarios.json`

Before reusing one, confirm the sibling library commit and current knob taxonomy;
these files may encode pre-cleanup defaults or old run-family assumptions.

## Wave 0 sweep-baseline configs

The next exact-trace sweep campaign uses an expanded Wave 0 prompt baseline before
larger knob sweeps. Fixture targets are declared in:

- `../exact_trace_wave0_fixture_targets.json`

Prepare the fixture catalog inside a SLURM job, not on a login node, because this
loads Gemma/GemmaScope machinery. The exact-bench helper renders/submits the
appropriate cluster script:

```bash
uv run python -m experiments.exact_trace_bench submit-fixture-prep \
  --cluster cardinal \
  --no-immutable-workspace \
  --print-only
```

Remove `--print-only` to submit. The underlying fixture-prep command remains
`experiments/prepare_weekend_prefix_fixtures.py` for direct SLURM script use.

After the catalog exists, generate Wave 0 scenario files on a login node:

```bash
uv run python -m experiments.exact_trace_bench build-wave0-scenarios \
  --all-tiers \
  --all-clusters \
  --fixture-catalog experiments/generated/exact_trace_wave0_fixtures/fixture_catalog.json
```

Expected generated files:

- `exact_trace_bench/exact_trace_wave0_baseline_fast_ascend_scenarios.json`
- `exact_trace_bench/exact_trace_wave0_baseline_anomaly_ascend_scenarios.json`
- `exact_trace_bench/exact_trace_wave0_baseline_long_eval_ascend_scenarios.json`
- `exact_trace_bench/exact_trace_wave0_baseline_fast_cardinal_scenarios.json`
- `exact_trace_bench/exact_trace_wave0_baseline_anomaly_cardinal_scenarios.json`
- `exact_trace_bench/exact_trace_wave0_baseline_long_eval_cardinal_scenarios.json`

These configs still write under the normal `{cluster}/{tier}` scratch roots; use
`run_id`, run metadata, and scenario names to identify the Wave 0 campaign.

After Wave 0 completes, build the pinned baseline registry from the scratch roots:

```bash
uv run python -m experiments.exact_trace_bench build-baseline-registry \
  --run-id wave0-baseline-20260520-01 \
  --all-clusters \
  --all-tiers \
  --output /fs/scratch/PAS3272/kopanev.1/exact_trace_bench/baselines/wave0-baseline-20260520-01.json
```

## Debug/replay and validation configs

These are Track-A or validation fixtures, not normal benchmark templates:

- `exact_trace_bench/matched_cross_cluster_*`
- `exact_trace_bench/phase0_donor_capture_94_base_*`
- `exact_trace_bench/phase0_replay_matrix_94_base_*`
- `exact_trace_bench/phase3_gradient_donor_capture_94_base_*`
- `exact_trace_bench/phase3_replay_matrix_94_base_*`
- `exact_trace_bench/phase3_row_replay_smoke_94_base_ascend.json`
- `exact_trace_bench/quick_cross_cluster_fp64_828_*`
- `cross_cluster_phase1_*_scenarios.json`
- `prompt94_compare_*_scenarios.json`

Use these only when intentionally capturing/replaying donor state or reproducing a
specific debug campaign. Record donor bundle paths, source commits, and SLURM job
IDs in run metadata/provenance.

## Historical/pre-consolidation configs

These are provenance artifacts and should not be used as new-run templates without
manual review:

- `weekend_exact_chunked_*`
- `weekend_exact_chunked_fixtures_matched_debug/`
- `exact_reference_overnight_scenarios.json`
- older RSS/overflow/locality validation configs outside the exact-bench package

Physical archiving can happen in a later repo-cleanup pass. For now, this index
marks the intended ownership without breaking historical paths.
