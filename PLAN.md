# Current Implementation Plan — Upstream Phase-0 Boundary Fingerprinting

## Objective

Use the main cross-cluster investigation workspace to localize **where Phase-0
first diverges** for `94_base`:

1. before CLT encode (`mlp_in_cache` input into CLT),
2. inside CLT encode preactivation / margin computation,
3. only at JumpReLU mask/post-mask.

The narrow Phase-0 compare-upcast hypothesis (baseline vs `fp32` vs `fp64`
compare mode) is now treated as negative for `94_base`.

This plan is for the main workspace only.

## Current state

- Within-cluster baseline consistency has already been validated.
- Matched single-step cross-cluster runs for `828_base` and `94_base` completed.
- Current interpretation:
  - first structural divergence appears in `phase0_sparse_setup`,
  - Phase-1 target logit state still matches,
  - Phase-3 seed ranking/frontier diverges downstream.
- New result from the 6-job compare matrix: Phase-0 compare-mode upcasts
  (`fp32`, `fp64`) did **not** reduce cross-cluster divergence for `94_base`.

## Immediate goals

1. Add upstream Phase-0 boundary fingerprints in library + checkpoint summaries.
2. Keep artifacts compact and JSON-friendly (hashes + compact stats only).
3. Run one matched `94_base` baseline pair with the new fingerprints.
4. Use first-divergence boundary evidence to choose the next investigation batch.

## Required implementation changes

### A. Phase-0 boundary fingerprint instrumentation

Target files:

- `../circuit-tracer_chunked/circuit_tracer/replacement_model/replacement_model_nnsight.py`
- `../circuit-tracer_chunked/circuit_tracer/transcoder/cross_layer_transcoder.py`
- `../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`

Add per-layer fingerprints for:

- pre-CLT input (`mlp_in_cache`) hash + compact stats,
- transcoder encode constants (W_enc/b_enc/threshold fingerprints),
- preactivation hash + compact stats,
- compare-margin (`preactivation - threshold`) hash + compact stats,
- JumpReLU mask hash / membership hash,
- post-mask activation hash + compact stats.

Also expand near-threshold epsilon counts for better boundary-mass coverage.

Constraints:

- preserve existing cross-cluster artifact compatibility (add fields, do not
  remove existing ones),
- keep payload scalar/compact (no large tensor dumps).

### B. Project docs + durable strategy note

Update:

- `PLAN.md` (this file),
- `EXPERIMENTS.md` with a dated note that compare-upcast was negative and
  boundary fingerprinting is now the chosen direction,
- `TODO.md` immediate tasks,
- `docs/phase0_boundary_fingerprinting_spec.md` as durable strategy/tradeoff
  reference.

## Intended next launch batch

Run a **2-job matched baseline pair** for `94_base` via immutable snapshot.

Shared config:

- fixture: `94_base`
- tier: `anomaly`
- `max_steps=1`
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
- `phase0_activation_threshold_compare_mode=baseline`
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

Jobs:

1. Ascend baseline
2. Cardinal baseline

## Success criteria

Implementation success:

- boundary fingerprint fields appear in `phase0_sparse_setup`,
- per-layer hashes are available for pre-CLT input, preactivation, margin,
  mask membership, and post-mask activation,
- no GPU work is run locally outside SLURM.

Experimental success:

- we can identify the earliest divergent boundary among:
  - pre-CLT input,
  - preactivation/margin,
  - mask/post-mask.

## Non-goals

- do not move optimization work into this workspace,
- do not treat this as the permanent overflow fix,
- do not widen scope into multi-step replay yet unless boundary-localization
  evidence remains ambiguous,
- do not use live-workspace launches for the next diagnostic batch.
