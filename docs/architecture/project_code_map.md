# Project code map

Status: Current-state map
Last updated: 2026-07-01

## Package and CLI

| Surface | Location |
|---|---|
| Console script `exact-trace-bench` | `pyproject.toml` → `nlp_research_project.exact_trace_bench.cli:main` |
| Main package | `src/nlp_research_project/exact_trace_bench/` |

Unless noted, paths in this page are relative to the project repo root.

## Project-root runtime modules

These modules are not part of the packaged harness, but current full-answer and
trace execution paths still use them.

| Module | Role |
|---|---|
| `trace_pipeline.py` | Older multi-prompt tracing/model-loading pipeline |
| `trace_pipeline_chunked.py` | Project-local exact/chunked tracing entry point |
| `circuit_utils.py` | Compact graph metrics and `.npz` helpers |
| `prefix_caching/trace_pipeline_cached.py` | Prefix-cached tracing variant |

## Key modules

| Module | Role |
|---|---|
| `config.py` | Roots, defaults, canonical `fp32` exact-trace dtype |
| `workspace.py` | Workspace snapshots, import verification, sibling detection |
| `jobs.py` | SLURM launch plans, templates, environment assembly |
| `presets.py` | Preset fanout |
| `fixtures.py` | Fixture catalog |
| `io_utils.py` | File and artifact helpers |
| `scenarios/{base,canonical,wave0,waves,tier_api}` | Scenario builders |
| `extract.py` | Data extraction workflows |
| `graph_compare.py` | Graph comparison utilities |
| `phase0_replay_matrix_compare.py` | Phase-0 replay comparison |
| `phase3_seed_bundle_compare.py` | Phase-3 seed bundle comparison |
| `semantic_feature_compare.py` | Semantic feature comparison |
| `baselines.py` | Baseline definitions |
| `full_answer/*` | Full-answer trace planning, execution, aggregation, audit, temporal/stability, role/calibration, decoder signature cache |

## CLI command groups

| Group | Typical modules |
|---|---|
| Scenario / baseline builders | `scenarios/*`, `baselines.py`, `fixtures.py` |
| Launch / safety / preset | `jobs.py`, `workspace.py`, `presets.py` |
| Extraction / comparison | `extract.py`, `graph_compare.py`, comparison helpers |
| Full-answer planning / execution | `full_answer/*`, `jobs.py` |
| Analysis / calibration | `semantic_feature_compare.py`, `phase0_replay_matrix_compare.py`, `phase3_seed_bundle_compare.py` |

All current commands are routed through `exact_trace_bench/cli.py`. The table
below is a map for finding the implementation owner, not a complete CLI
reference.

| Command area | Commands | Main implementation |
|---|---|---|
| Scenario generation | `build-scenarios`, `build-wave0-scenarios`, `build-wave2a-phase1-scenarios`, `build-wave2b-phase4-scenarios`, `build-wave2c-row-encoder-scenarios`, `build-wave3-interaction-confirmation-scenarios`, `build-wave4-generalization-scenarios` | `scenarios/*`, `fixtures.py`, `config.py` |
| Baselines / fixtures | `build-baseline-registry`, `describe-fixtures` | `baselines.py`, `fixtures.py` |
| Launch and workspace safety | `launch-plan`, `submit-fixture-prep`, `submit-preset`, `snapshot-workspace`, `verify-imports` | `jobs.py`, `presets.py`, `workspace.py` |
| Extraction and compact comparison | `extract`, `compare-compact`, `compare-phase0-replay-matrix`, `compare-phase3-seed-bundles`, `compare-semantic-features` | `extract.py`, `graph_compare.py`, comparison helpers |
| Full-answer planning/execution | `build-full-answer-trace-specs`, `build-full-answer-shards`, `run-full-answer-shard`, `aggregate-full-answer-shards`, `audit-full-answer-prefix-views`, `launch-full-answer-shards` | `full_answer/schemas.py`, `full_answer/selection.py`, `full_answer/sharding.py`, `full_answer/runner.py`, `full_answer/aggregate.py`, `full_answer/audit.py`, `jobs.py` |
| Full-answer trajectory | `run-full-answer-trajectory`, `sample-full-answer-trajectories`, `submit-full-answer-trajectory` | `full_answer/trajectory.py`, `jobs.py` |
| Full-answer analysis/calibration | `analyze-full-answer-temporal`, `plot-full-answer-temporal`, `compare-full-answer-stability`, `diagnose-full-answer-stability`, `classify-full-answer-roles`, `build-role-matched-calibration-manifest`, `apply-role-classification-reviews`, `build-decoder-signature-cache`, `run-metric-calibration`, `rebucket-metric-calibration`, `plot-metric-calibration` | `full_answer/temporal.py`, `full_answer/temporal_plots.py`, `full_answer/stability.py`, `full_answer/diagnostics.py`, `full_answer/role_classification.py`, `full_answer/decoder_signature_cache.py`, `full_answer/calibration.py`, `full_answer/metric_battery.py`, `full_answer/calibration_plots.py` |

## Artifact flow

`scenario JSON` → `launch plans / SLURM env` → `fixture prep` → `full-answer shard outputs`
(`shard.json`, `trace_results.jsonl`, token dirs, `trace.json`, `graph.npz`/`.pt`,
optional phase bundles/debug sidecars) → `aggregate / audit` → `extraction CSV/JSONL`
→ `calibration role artifacts`

## Current safe commands

- `uv run exact-trace-bench --help`
- `uv run exact-trace-bench build-full-answer-trace-specs --help`
- `uv run exact-trace-bench describe-fixtures`
- `uv run exact-trace-bench build-scenarios --all-tiers --all-clusters`
- `uv run exact-trace-bench snapshot-workspace --print-path-only`
- `uv run exact-trace-bench verify-imports --workspace-root <repo-root>`
- data-only extraction/comparison/aggregation/audit commands on known artifacts

## Tests commonly used for this area

- `tests/test_exact_trace_bench_package.py`
- `tests/test_exact_trace_bench_path_safety.py`
- monkeypatched full-answer CLI/sharding/trajectory/aggregate tests

## Current compatibility surfaces to remember

- `experiments/exact_trace_bench/__init__.py` and `__main__.py` are compatibility
  shims for older import paths.
- `scripts/archive/` is historical; current launch templates live under
  `slurm/exact_trace_bench/` and are normally reached through the CLI.
- Scenario taxonomy still has compatibility aliases such as
  `auto_scale_feature_batch_size` and legacy/compact/full parser names.
- Wave-specific scenario builders are campaign-specific and should be checked
  against current docs before being copied into new campaigns.
