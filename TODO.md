# TODO — main repo (cross-cluster investigation)

This workspace is for **cross-cluster investigation**, not optimization.

## Immediate tasks

1. Treat the current queued boundary-fingerprint pair as a localization run,
   not the final causality test.

2. Keep the upstream Phase-0 boundary-fingerprinting logs as the earliest-boundary
   layer in the paired repos:
   - project: `./`
   - sibling library: `../circuit-tracer_chunked`

3. Implement passive Phase-3 seed bundle capture for the next stronger rerun.

4. Verify cross-cluster debug artifacts now include compact boundary fields for:
   - pre-CLT input fingerprints,
   - transcoder constant fingerprints,
   - preactivation/margin/mask/post-mask fingerprints,
   - expanded near-threshold counts,
   - plus saved Phase-3 seed bundle artifacts.

5. Add offline Ascend/Cardinal bundle comparison for:
   - shared vs unique Phase-0 support,
   - Phase-3 influence mass split,
   - frontier overlap after restricting to shared support.

6. Relaunch one matched `94_base` baseline pair (Ascend + Cardinal) from an
   immutable snapshot with the stronger artifact set:
   - `cross_cluster_debug=true`
   - `exact_trace_internal_dtype=fp64`
   - `phase0_activation_threshold_compare_mode=baseline`
   - single-step (`max_steps=1`) configuration.

7. From `phase0_sparse_setup`, identify the first divergence boundary:
   - pre-CLT input vs
   - preactivation/margin vs
   - mask/post-mask.

## Questions to answer next

- Does divergence already exist in `mlp_in_cache` before CLT encode?
- If not, does it first appear in preactivation/margin fingerprints?
- If not, is it introduced only at mask/post-mask boundary logic?
- Does Phase-3 disagreement mostly disappear once Phase-0-unique support is
  removed?
- If not, does Phase-3 appear to add its own extra instability and therefore
  justify a later replay/intervention step?

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
