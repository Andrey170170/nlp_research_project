# Current Implementation Plan — Phase-0 → Phase-3 Causality Experiment

## Objective

Implement the next stronger cross-cluster experiment for `94_base` so we can
move from:

- "Phase-0 diverges first, and Phase-3 diverges later"

to a stronger test of:

- whether the Phase-3 divergence is mostly downstream amplification of earlier
  Phase-0 drift,
- or whether Phase-3 has an additional cluster-sensitive instability of its own.

The full experiment has two layers:

1. **Core layer (required)** — Phase-0 boundary fingerprints + saved Phase-3
   seed bundle for offline shared-vs-unique decomposition.
2. **Extra layer (defer unless still needed)** — stronger donor/replay-style
   intervention that reruns Phase-3/4 from saved early-state artifacts.

The current queued boundary-fingerprint pair is still useful, but by itself it
is **not** enough to prove a causal Phase-0 → Phase-3 link.

## Current state

- Within-cluster baseline consistency has already been validated.
- Matched single-step cross-cluster runs for `828_base` and `94_base` completed.
- Current interpretation:
  - first structural divergence appears in `phase0_sparse_setup`,
  - Phase-1 target logit state still matches,
  - Phase-3 seed ranking/frontier diverges downstream.
- The 6-job compare matrix showed that Phase-0 compare-mode upcasts (`fp32`,
  `fp64`) do **not** reduce the `94_base` cross-cluster split.
- Phase-0 boundary-fingerprint logging has already been implemented and a new
  matched baseline pair has been launched.

## Why the current queued pair is not enough

Boundary fingerprints can tell us **where the first mismatch starts**:

1. before CLT encode (`mlp_in_cache` input into CLT),
2. inside CLT encode preactivation / margin computation,
3. only at JumpReLU mask/post-mask.

But even a clean early mismatch does not, by itself, prove that the later
Phase-3 split is entirely downstream of Phase-0. We need an additional test that
asks whether Phase-3 disagreement is mostly concentrated in Phase-0-unique
features or persists even after restricting to shared early-state support.

## Immediate goals

1. Keep the current Phase-0 boundary-fingerprint work as the earliest-divergence
   localization layer.
2. Add a compact **Phase-3 seed bundle artifact** that can be saved per traced
   step without changing run behavior.
3. Add offline comparison support for shared-vs-unique decomposition of Phase-3
   influence/frontier behavior.
4. Relaunch a new matched `94_base` pair from immutable snapshot using the
   stronger artifact set.
5. Only if ambiguity remains, plan an explicit donor/replay intervention mode as
   an extra follow-up.

## Required implementation changes

### A. Preserve / validate Phase-0 boundary localization layer

Keep and validate the already chosen Phase-0 boundary instrumentation:

- `../circuit-tracer_chunked/circuit_tracer/replacement_model/replacement_model_nnsight.py`
- `../circuit-tracer_chunked/circuit_tracer/transcoder/cross_layer_transcoder.py`
- `../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`

Required fields remain:

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

### B. Add Phase-3 seed bundle capture (**core stronger-evidence layer**)

Goal:

- capture enough Phase-3 state to test, offline, whether the later divergence is
  mostly explained by the earlier Phase-0 split.

Target files:

- `../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
- `trace_pipeline_chunked.py`
- `experiments/run_sparsification_experiment.py`
- `experiments/exact_trace_bench/scenarios.py` (only if a scenario knob is used)
- extractors / manifests if needed for indexing

Preferred artifact:

- `step_000_phase3_seed_bundle.npz`

Minimum bundle contents:

- `active_features`
- `activation_values`
- `seed_feature_influences`
- `frontier_pre_locality`
- `frontier_post_locality`
- `queue_size`
- `actual_max_feature_nodes`
- any small metadata needed to align bundle rows with existing Phase-0 support

Constraints:

- do not change planner behavior,
- do not change seed ranking/frontier selection semantics,
- keep this as a passive saved artifact for offline comparison.

### C. Add offline comparison / interpretation support (**core stronger-evidence layer**)

Goal:

- compare Ascend/Cardinal saved Phase-3 bundles and quantify whether the Phase-3
  mismatch is concentrated in Phase-0-unique features.

Required outputs:

- shared vs unique Phase-0 feature counts,
- influence mass on shared vs unique feature sets,
- frontier overlap before/after restricting to shared features,
- summary verdict on whether Phase-3 mismatch largely disappears once the
  earlier support mismatch is controlled.

This can live in a small CPU-only compare helper or notebook-friendly script.

### D. Docs + durable strategy note

Update:

- `PLAN.md` (this file),
- `EXPERIMENTS.md` with a dated note that compare-upcast was negative and
  the next stronger causality experiment now adds Phase-3 bundle capture,
- `TODO.md` immediate tasks,
- `docs/phase0_boundary_fingerprinting_spec.md` as durable strategy/tradeoff
  reference.

### E. Extra follow-up — donor/replay intervention (**extra; do not block core experiment**) 

Only do this if the core bundle-capture experiment still leaves ambiguity.

Possible direction:

- load a saved donor Phase-0 or Phase-3 bundle,
- rerun downstream ranking/frontier logic on the other cluster,
- test whether Phase-3 follows the donor early-state support.

This is stronger causal evidence, but it is also much more invasive and should
not be the first implementation step.

## Intended next launch batch

Run a **new 2-job matched baseline pair** for `94_base` via immutable snapshot
after the core stronger-evidence artifacts land.

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
- Phase-0 boundary fingerprints enabled
- Phase-3 seed bundle capture enabled

Jobs:

1. Ascend baseline
2. Cardinal baseline

## Success criteria

Implementation success:

- boundary fingerprint fields appear in `phase0_sparse_setup`,
- per-layer hashes are available for pre-CLT input, preactivation, margin,
  mask membership, and post-mask activation,
- Phase-3 seed bundle artifacts are emitted and indexed cleanly,
- no GPU work is run locally outside SLURM.

Experimental success:

- we can identify the earliest divergent boundary among:
  - pre-CLT input,
  - preactivation/margin,
  - mask/post-mask.
- we can quantify whether Phase-3 divergence is mostly carried by Phase-0-unique
  features,
- we can say whether the evidence supports "Phase-3 is downstream amplification"
  versus "Phase-3 likely adds its own extra instability".

## Non-goals

- do not move optimization work into this workspace,
- do not treat this as the permanent overflow fix,
- do not implement donor/replay intervention until the passive bundle-capture
  experiment has been analyzed,
- do not widen scope into multi-step replay unless the single-step causality
  evidence remains ambiguous,
- do not use live-workspace launches for the next diagnostic batch.
