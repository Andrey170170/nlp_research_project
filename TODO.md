# TODO — main repo (cross-cluster investigation)

This workspace is for **cross-cluster investigation**, not optimization.

## Immediate tasks

1. Close the compare-upcast branch as negative for `94_base` in active planning
   and experiments notes.

2. Land upstream Phase-0 boundary-fingerprinting logs in the paired repos:
   - project: `./`
   - sibling library: `../circuit-tracer_chunked`

3. Verify cross-cluster debug artifacts now include compact boundary fields for:
   - pre-CLT input fingerprints,
   - transcoder constant fingerprints,
   - preactivation/margin/mask/post-mask fingerprints,
   - expanded near-threshold counts.

4. Launch one matched `94_base` baseline pair (Ascend + Cardinal) from an
   immutable snapshot with:
   - `cross_cluster_debug=true`
   - `exact_trace_internal_dtype=fp64`
   - `phase0_activation_threshold_compare_mode=baseline`
   - single-step (`max_steps=1`) configuration.

5. From `phase0_sparse_setup`, identify the first divergence boundary:
   - pre-CLT input vs
   - preactivation/margin vs
   - mask/post-mask.

## Questions to answer next

- Does divergence already exist in `mlp_in_cache` before CLT encode?
- If not, does it first appear in preactivation/margin fingerprints?
- If not, is it introduced only at mask/post-mask boundary logic?
- Does the earliest boundary explain downstream Phase-3 frontier divergence?

## Rules for this workspace

- Do not use this workspace for optimization experiments.
- Keep canonical debug schema and interpretation here.
- Use normal run folders only:
  - `ascend/fast`
  - `ascend/anomaly`
  - `ascend/long_eval`
  - `cardinal/fast`
  - `cardinal/anomaly`
  - `cardinal/long_eval`
