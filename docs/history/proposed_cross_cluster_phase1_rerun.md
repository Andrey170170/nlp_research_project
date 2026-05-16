# Proposed cross-cluster Phase 1 rerun

This note describes the **prepared but not launched** paired rerun config for
the next cross-cluster investigation pass.

## Goal

Use the new Phase-1 instrumentation to collect one matched Ascend/Cardinal rerun
pair that gives:

- explicit internal precision contract,
- broad early-phase cross-cluster checkpoints,
- per-batch Phase 3 / Phase 4 debug streams,
- and prompt-94-specific Phase-4 anomaly diagnostics without requiring a second
  dedicated rerun.

## Prepared scenario files

- `experiments/generated/cross_cluster_phase1_ascend_scenarios.json`
- `experiments/generated/cross_cluster_phase1_cardinal_scenarios.json`

## Prompt selection

### 1. `94_base`

Purpose:

- primary anomaly/watch prompt,
- runs both:
  - `cross_cluster_debug=true`
  - `phase4_anomaly_debug=true`

### 2. `828_base`

Purpose:

- fast control prompt,
- runs:
  - `cross_cluster_debug=true`
  - `phase4_anomaly_debug=false`

Rationale:

- this keeps broad early-phase coverage on both prompts,
- but avoids paying the heavier Phase-4 shadow-debug cost on the control prompt.

## Shared runtime settings

- `attribution_batch_size=256`
- `feature_batch_size=256`
- `logit_batch_size=256`
- `decoder_chunk_size=4096`
- `cross_batch_decoder_cache_bytes=0`
- `internal_precision=float64`
- `cross_cluster_debug=true`
- `max_feature_nodes=8192`
- `max_edges=20000`
- `max_steps=1`
- `temperature=0.0`

## Proposed launch commands

Do **not** launch automatically. These are only the proposed commands.

### Ascend

```bash
SCENARIOS_FILE=experiments/generated/cross_cluster_phase1_ascend_scenarios.json \
sbatch scripts/trace_weekend_exact_chunked.ascend.sbatch \
  --run-id 2026xxxx_xxxxxx_cross-cluster-phase1-ascend \
  --run-name "cross cluster phase1 ascend" \
  --run-description "Phase 1 paired rerun on Ascend with explicit precision contract and broad cross-cluster debug coverage for prompts 94 and 828" \
  --run-goal "localize earliest Ascend/Cardinal divergence with one paired rerun while keeping prompt 94 under anomaly-watch instrumentation"
```

### Cardinal

```bash
SCENARIOS_FILE=experiments/generated/cross_cluster_phase1_cardinal_scenarios.json \
sbatch scripts/trace_weekend_exact_chunked.cardinal.sbatch \
  --run-id 2026xxxx_xxxxxx_cross-cluster-phase1-cardinal \
  --run-name "cross cluster phase1 cardinal" \
  --run-description "Phase 1 paired rerun on Cardinal with explicit precision contract and broad cross-cluster debug coverage for prompts 94 and 828" \
  --run-goal "localize earliest Ascend/Cardinal divergence with one paired rerun while keeping prompt 94 under anomaly-watch instrumentation"
```

## Expected artifacts of interest

Per completion:

- `cross_cluster_debug_summary.json`
- `cross_cluster_debug_checkpoints.jsonl`
- `cross_cluster_debug_batches.jsonl`
- `phase4_anomaly_debug.json` (prompt 94 only in this proposal)
- `telemetry.jsonl`
- `completion.json`

## Notes

- Ascend target here is the normal nextgen path used by the existing fast Ascend
  batch script, not the quad high-memory path.
- If this proposed rerun is still too expensive, the first simplification should
  be dropping `phase4_anomaly_debug` from `94_base`, not dropping
  `cross_cluster_debug`.
