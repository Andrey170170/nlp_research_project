# Current Implementation Plan — Phase-0 / Phase-3 Cross-Cluster Diagnostics

## Objective

Use the main cross-cluster investigation workspace to explain why the matched
single-step fp64 runs diverge first at `phase0_sparse_setup`, and to prepare one
high-information queued batch that can both diagnose and test a narrow
mitigation.

This plan is for the main workspace only.

## Current state

- Within-cluster baseline consistency has already been validated.
- Matched single-step cross-cluster runs for `828_base` and `94_base` completed.
- Current interpretation:
  - first structural divergence appears in `phase0_sparse_setup`,
  - Phase-1 target logit state still matches,
  - Phase-3 seed ranking/frontier diverges downstream.
- The likely root cause is that the transcoder Phase-0 sparse encode path still
  runs in bf16 and applies a hard JumpReLU threshold before the exact-trace
  internal fp64 path begins.

## Immediate goals

1. Reduce queue friction for future launches.
2. Add Phase-0 instrumentation to distinguish:
   - true active-feature membership drift,
   - near-threshold bf16 flips,
   - sparse ordering-only differences.
3. Add richer Phase-3 instrumentation to explain when early drift becomes a
   ranking/frontier divergence.
4. Add a narrow Phase-0 mitigation toggle that upcasts only the encoder
   activation + threshold-compare path.
5. Launch one high-value **6-job** matched batch on `94_base`.

## Required implementation changes

### A. Queue-friendly defaults

- Keep benchmark sbatch walltime at `01:00:00` on both clusters.
- Files:
  - `scripts/trace_weekend_exact_chunked.ascend.sbatch`
  - `scripts/trace_weekend_exact_chunked.cardinal.sbatch`

### B. Phase-0 diagnostics

Target file:

- `../circuit-tracer_chunked/circuit_tracer/transcoder/cross_layer_transcoder.py`
- `../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`

Add diagnostics for the first-step cross-cluster capture path:

- canonical / sorted active-feature hash,
- per-layer near-threshold counts,
- small sample of borderline features near the JumpReLU threshold,
- explicit separation of:
  - raw sparse index hash,
  - sorted membership hash,
- optional shadow summary for alternative compare dtypes.

Goal:

- confirm whether Phase-0 divergence is caused by bf16 threshold flips on
  marginal features.

### C. Phase-3 diagnostics

Target file:

- `../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
- `../circuit-tracer_chunked/circuit_tracer/graph.py`

Add logging around `phase3_seed_ranking_pre_phase4`:

- top-K seed influences with `(layer, pos, feat_id)`,
- cutoff score and next-below-cutoff score,
- cutoff margin / near-tie count,
- more explicit shadow-frontier overlap diagnostics,
- hashes or compact stats for row inputs / row abs sums used for ranking.

Goal:

- distinguish "Phase-3 only amplifies earlier Phase-0 drift" from "Phase-3 adds
  extra ranking instability of its own".

### D. Narrow mitigation toggle

Target file:

- `../circuit-tracer_chunked/circuit_tracer/transcoder/cross_layer_transcoder.py`
- project-side launch plumbing where needed

Add a toggle for Phase-0 encoder activation / threshold compare mode:

- baseline: current bf16 behavior,
- mitigation candidate 1: Phase-0 compare in fp32,
- mitigation candidate 2: Phase-0 compare in fp64.

Constraint:

- do **not** change the broader exact-trace internal contract,
- do **not** convert the whole model/transcoder pipeline to fp64,
- keep this as a narrow targeted mitigation for the Phase-0 sparse-encode path.

## Intended next launch batch

Run a **6-job** matched batch, all on `94_base`, all via immutable snapshot
launches, all with the same tracing contract except for the Phase-0 compare mode.

Shared config across all six jobs:

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

Six planned jobs:

1. Ascend baseline
2. Cardinal baseline
3. Ascend Phase-0 compare upcast = fp32
4. Cardinal Phase-0 compare upcast = fp32
5. Ascend Phase-0 compare upcast = fp64
6. Cardinal Phase-0 compare upcast = fp64

## Success criteria

Implementation success:

- new diagnostics appear in cross-cluster debug artifacts,
- launch plumbing can select the Phase-0 compare mode explicitly,
- no GPU work is run locally outside SLURM.

Experimental success:

- we can tell whether Phase-0 divergence is mainly caused by threshold flips near
  the JumpReLU boundary,
- we can tell whether fp32 or fp64 Phase-0 upcast reduces:
  - active-feature count deltas,
  - Phase-0 membership-hash drift,
  - downstream Phase-3 frontier divergence.

## Non-goals

- do not move optimization work into this workspace,
- do not treat this as the permanent overflow fix,
- do not widen scope into multi-step replay yet unless the single-step evidence
  remains ambiguous,
- do not use live-workspace launches for the next diagnostic batch.
