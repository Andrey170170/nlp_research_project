# Metric calibration toolset

Status: Current workflow reference
Last updated: 2026-06-12

This page describes the Phase-0/Phase-1 analysis tooling for calibrated temporal
graph metrics. It is login-node safe except for the one-time decoder-signature
cache build, which should run as a SLURM CPU/high-memory job when the cache is
not already present.

## Phase 0: decoder-signature cache

Soft feature matching uses normalized GemmaScope-2 CLT downstream decoder
signatures. The canonical cache built for current work is:

```text
/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/decoder_signature_cache/
  gemma-scope-2-1b-it__clt_width_262k_l0_medium_affine__float32_chunk512/
```

It stores one normalized flattened row for each `(source_layer, feature_id)`:

```text
w_dec[feature_id, source_layer:, :].reshape(-1) / ||...||_2
```

The cache is chunked by source layer and feature row. Analysis code reads it via
`DecoderSignatureStore`; metric runs do not load the model or raw transcoders.

Rebuild command template:

```bash
uv run exact-trace-bench build-decoder-signature-cache \
  --source-dir /path/to/gemma-scope-2-1b-it/clt/width_262k_l0_medium_affine \
  --output-dir /fs/scratch/PAS2836/kopanev.1/exact_trace_bench/decoder_signature_cache/gemma-scope-2-1b-it__clt_width_262k_l0_medium_affine__float32_chunk512 \
  --chunk-size 512 \
  --storage-dtype float32
```

## Phase 1: metric calibration

Run the calibration harness from a JSON pair manifest:

```bash
uv run exact-trace-bench run-metric-calibration \
  --pair-manifest experiments/generated/exact_trace_bench/metric_calibration_manifest.json \
  --output-dir /fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/fast/metric-calibration-v1 \
  --decoder-cache-dir /fs/scratch/PAS2836/kopanev.1/exact_trace_bench/decoder_signature_cache/gemma-scope-2-1b-it__clt_width_262k_l0_medium_affine__float32_chunk512
```

Outputs:

- `metric_rows.jsonl` — long-form metric rows with structured `params`;
- `metric_rows.csv` — tabular copy with `params_json`;
- `metric_rows.parquet` — tabular parquet when pandas/pyarrow are available;
- `scorecard.{csv,json}` — null/temporal/noise separation summary;
- `pair_manifest_resolved.json` — exact graph paths and pair metadata;
- `calibration_summary.json` — run summary.

Plot and report template command:

```bash
uv run exact-trace-bench plot-metric-calibration \
  --analysis-dir /fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/fast/metric-calibration-v1 \
  --output-dir reports/metric-calibration-v1
```

This writes calibration distribution plots, derived-K curves, lag-decay curves,
scorecard pass-count plots, and `metric_calibration_findings.md`.

## Manifest schema

The manifest can mix three pair sources.

### Explicit pairs

Use for Stage A/B/C/D noise pairs or hand-picked null pairs:

```json
{
  "pairs": [
    {
      "pair_id": "stage_a_828_token0_prefix_vs_fullseq",
      "left_graph": "/scratch/.../left/token_000000/graph.npz",
      "right_graph": "/scratch/.../right/token_000000/graph.npz",
      "pair_category": "noise",
      "sub_category": "prefix_vs_fullseq",
      "fixture": "828_base",
      "position_band": "early",
      "generated_index": 0
    }
  ]
}
```

### Temporal trajectory pairs

Use for adjacent and lagged pairs within a full-answer trajectory:

```json
{
  "trajectories": [
    {
      "name": "828_base_temp0_correct",
      "run_root": "/scratch/.../typed-bucketed-full-reruns-20260527/...",
      "pair_category": "temporal",
      "lags": [1, 2, 5, 10, 20],
      "stride": 1,
      "max_pairs_per_lag": 2000
    }
  ]
}
```

### Matched-relative null pairs

Use for cross-prompt or same-prompt/different-sample nulls:

```json
{
  "matched_relative_pairs": [
    {
      "name": "828_vs_361_quantile_null",
      "left_run_root": "/scratch/.../828/...",
      "right_run_root": "/scratch/.../361/...",
      "pair_category": "null",
      "sub_category": "cross_prompt",
      "sample_count": 20
    }
  ]
}
```

## Metric families

The battery emits per-bucket and feature-node rows for:

- derived-K plain/weighted Jaccard curves;
- top-p mass-core overlap;
- extrapolated RBO;
- raw union-support cosine and intersection Spearman;
- decoder-cosine soft feature matching when `--decoder-cache-dir` is supplied;
- collapsed layer/positionless flow weighted Jaccard, normalized L1, and total
  variation distance.

The scorecard treats most metrics as higher-is-more-similar and treats `distance`
metrics as lower-is-more-similar.
