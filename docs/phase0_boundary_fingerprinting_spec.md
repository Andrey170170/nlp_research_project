# Phase-0 Boundary + Phase-3 Causality Probe Spec

## Purpose

Localize the first cross-cluster divergence for matched single-step `94_base`
runs and add enough passive state capture to test whether later Phase-3
divergence is mostly downstream of the earlier Phase-0 split.

The earlier compare-mode experiment (baseline vs `fp32` vs `fp64` threshold
compare) was negative, so this spec shifts the investigation upstream.

## Decision summary

Instrument the Phase-0 boundary in three segments:

1. **Pre-CLT input** — the tensor entering CLT (`mlp_in_cache`)
2. **CLT pre-mask path** — transcoder constants, preactivation, and margin
3. **Post-mask path** — JumpReLU membership and post-mask activation

Use compact fingerprints only:

- hashes,
- compact sampled stats,
- per-layer counts,
- global aggregate hashes.

Do not emit dense tensors into JSON artifacts.

For stronger evidence, add a **passive Phase-3 seed bundle artifact** that can
be compared offline across Ascend/Cardinal without changing runtime behavior.

The stronger donor/replay-style intervention is explicitly deferred and treated
as an **extra** follow-up, not the first implementation step.

## Required fields

### Pre-CLT input

For each layer:

- shape
- element count
- sampled hash
- sampled stats

Also emit a global hash across per-layer hashes.

### CLT encode constants

For each layer:

- sampled encoder-weight fingerprint
- encoder-bias hash/stats
- JumpReLU threshold hash/stats
- per-layer constant fingerprint hash

Also emit a global hash across layer constants.

### Preactivation / margin / mask / post-mask

For each layer:

- preactivation sampled hash/stats
- compare-margin sampled hash/stats
- mask membership hash
- post-mask activation sampled hash/stats
- near-threshold counts for expanded epsilons

Also emit global hashes across per-layer:

- preactivation
- margin
- mask membership
- post-mask activation

## Phase-3 causality probe (core stronger-evidence layer)

### Goal

Test whether the observed Phase-3 divergence is largely explained by the earlier
Phase-0 support split.

### Chosen mechanism

Save a compact per-step Phase-3 seed bundle, then compare Ascend/Cardinal
offline.

Preferred artifact:

- `step_000_phase3_seed_bundle.npz`
- `step_000_feature_semantic_descriptors.npz` when semantic descriptor capture is
  enabled

Minimum contents:

- `active_features`
- `activation_values`
- `seed_feature_influences`
- `frontier_pre_locality`
- `frontier_post_locality`
- `queue_size`
- `actual_max_feature_nodes`

The offline comparator should report:

- shared vs unique Phase-0 feature counts,
- influence mass on shared vs unique features,
- frontier overlap before/after restricting to shared support,
- whether the Phase-3 mismatch largely disappears once Phase-0-unique features
  are removed.

The semantic descriptor artifact should remain bounded and passive. It captures
top seed/frontier candidate labels, ranks, activations, influences, Phase-4
selection membership when available, and a compact descriptor sketch. The current
first implementation uses `fallback_identity_metadata_v1`; future SLURM-only
descriptor jobs may replace or augment this with decoder-vector sketches if exact
ID mismatch remains semantically ambiguous.

Semantic comparison should report:

- exact candidate support overlap,
- shared-candidate activation/influence stability,
- high-mass unmatched candidates,
- same-layer/same-position descriptor-nearest substitutes,
- mass-weighted coverage of unmatched features by high-confidence substitutes.

## Chosen tradeoffs

### Chosen

- **Sampled dense fingerprints** instead of full dense dumps or full dense hashes
  for every tensor.
- Keep full near-threshold counts / membership summaries where those are already
  part of the diagnostic decision path.
- Preserve existing checkpoint structure and add fields instead of replacing the
  current schema.
- Use **passive bundle capture + offline decomposition** as the first stronger
  Phase-0 → Phase-3 test instead of immediately building a replay mode.
- Add bounded semantic descriptor capture before the next expensive run so exact
  ID graph churn can be separated from possible semantic stability.

### Rejected for now

- Full dense tensor serialization: too large and unnecessary.
- Immediate deterministic shadow replay: higher cost before we know whether the
  mismatch starts before or after CLT encode.
- More compare-mode variants: low information after the negative 6-job matrix.
- Donor/swap/replay intervention as the first step: stronger evidence, but too
  invasive before the passive bundle-capture experiment is tried.

## Expected interpretation logic

- If `mlp_in_cache` already differs, the root cause is upstream of CLT encode.
- If pre-CLT input matches but preactivation / margin differs, focus on CLT
  encode precision / determinism.
- If preactivation matches but mask/post-mask differs, focus on boundary logic.
- If the offline Phase-3 bundle comparison shows that most Phase-3 disagreement
  is carried by Phase-0-unique features and shrinks sharply after restricting to
  shared support, treat that as strong evidence for downstream amplification.
- If substantial Phase-3 disagreement remains even after controlling for shared
  support, treat that as evidence that Phase-3 likely contributes additional
  instability and consider the deferred replay/intervention step.
- If exact-ID feature/edge overlap is weak but high-mass unmatched features have
  high-confidence semantic substitutes, treat the graph as semantically more
  stable than exact-ID metrics alone suggest.
- If high-mass unmatched features lack semantic substitutes, treat the drift as a
  candidate genuine graph instability requiring mitigation or replay analysis.

## Extra follow-up (deferred)

If the passive bundle-capture experiment remains ambiguous, implement a stronger
intervention mode that consumes a saved donor early-state bundle and reruns
downstream ranking/frontier logic. This is the closest thing to a direct causal
counterfactual, but it should be a separate follow-up task.

## Validation

- Extend existing cross-cluster debug artifact tests to check new fields are
  preserved through summary / stream emission.
- Extend CLT diagnostic tests to assert new Phase-0 fingerprint fields exist and
  reflect post-zero-position masking.
- Add bundle-capture, descriptor-capture, graph-decomposition, and semantic
  comparison CPU-only checks.
- Only run safe local validation (`uv run ...`), no GPU workloads outside SLURM.
