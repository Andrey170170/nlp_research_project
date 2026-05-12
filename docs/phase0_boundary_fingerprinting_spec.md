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

## Phase-0 donor swap/replay design (deferred stronger causal check)

### Problem statement

The passive `94_base` rerun with fixed Phase-3 seed bundles supports the current
interpretation that Ascend/Cardinal drift starts upstream of CLT encode, while
the important downstream circuit remains mostly stable. It does not by itself
prove a direct causal link from Phase-0 state to the observed Phase-3 frontier
and compact-graph exact-ID churn.

The next stronger test is a counterfactual replay:

> Keep the host cluster's prompt/model/logit context, replace only the Phase-0
> active feature state with a saved donor cluster Phase-0 bundle, then rerun
> Phase-3/frontier/graph extraction. If the downstream artifacts move toward the
> donor cluster, Phase-0 state is causally carrying the later split.

### Scope / non-goals

Scope:

- Start with matched single-step `94_base` only.
- Run a four-condition matrix:
  - Ascend normal baseline,
  - Cardinal normal baseline,
  - Ascend host with Cardinal Phase-0 donor bundle,
  - Cardinal host with Ascend Phase-0 donor bundle.
- Include self-replay sanity checks before trusting cross-swaps:
  - Ascend host with Ascend Phase-0 bundle,
  - Cardinal host with Cardinal Phase-0 bundle.
- Compare Phase-3 seed bundles, Phase-4 frontier/compact graph artifacts, and
  semantic descriptor artifacts using the existing offline comparators.

Non-goals:

- Do not replay or swap full dense model activations.
- Do not try to make the whole model forward deterministic across hardware.
- Do not change generation policy or sample additional tokens in the first
  implementation.
- Do not treat fallback semantic descriptors as strong semantic proof.
- Do not run replay/model code outside SLURM.

### Required Phase-0 donor bundle

Add an opt-in artifact, for example:

- `step_000_phase0_donor_bundle.npz`

The bundle must be **order preserving** because downstream row indices depend on
Phase-0 feature order.

Minimum fields:

- `schema_version`
- `active_features`: `(N, 3)` int64 `[layer, position, feature_idx]`
- `activation_values`: replay-ready activation values aligned to
  `active_features`
- `activation_values_float32`: analysis copy when replay dtype is not NumPy
  native
- `activation_values_original_dtype`
- optional exact dtype sidecar for unsupported NumPy dtypes, e.g.
  `activation_values_bf16_bits` when source activations are `bfloat16`
- `active_feature_count`
- `active_feature_indices_hash`
- `active_feature_membership_hash_canonical`
- `activation_value_hash`
- compact activation stats
- `phase0_activation_threshold_compare_mode`
- `exact_trace_internal_dtype_requested`
- `resolved_dtype_map`
- `phase0_pre_clt_input_global_hash`
- `transcoder_constants_global_hash`
- prompt/context identity:
  - `fixture_name`, `gsm8k_index`, `prompt_token_count`, prefix/token hash
  - target-token ids hash / Phase-1 target-logit state hash when available
- provenance:
  - cluster, run id, scenario name,
  - project commit, library commit,
  - snapshot roots or live workspace paths.

The existing `step_000_phase3_seed_bundle.npz` already contains
`active_features` and `activation_values`, but the donor-swap mode should use a
dedicated Phase-0 bundle so replay prerequisites and hashes are explicit and not
coupled to successful Phase-3 completion.

### Proposed replay approach

Add a controlled exact-trace mode with flags similar to:

- `--capture-phase0-donor-bundle`
- `--phase0-donor-bundle PATH`
- `--phase0-replay-mode donor-active-features`
- `--phase0-donor-context-policy strict|warn`

Primary replay behavior:

1. Run the host model/prompt normally far enough to construct the attribution
   context and Phase-1 target logits/logit probabilities.
2. Load the donor Phase-0 bundle.
3. Validate strict context compatibility:
   - same fixture/prompt hash,
   - same token prefix length,
   - same CLT constants hash,
   - same target token ids hash,
   - target-logit state hash either matches or the run explicitly records a
     `warn` policy deviation.
4. Replace the runtime `activation_matrix` and all derived Phase-0 row metadata
   with the donor bundle:
   - `total_active_feats`,
   - `feat_layers`, `feat_pos`, `feat_ids`,
   - `row_to_node_index` / row ordering,
   - chunked decoder state source layers / positions / feature ids,
   - activation values in the intended replay dtype.
5. Rerun Phase-3 seed ranking, Phase-4 frontier expansion, and compact graph
   packaging exactly as in a normal run.
6. Emit normal artifacts plus replay metadata:
   - `phase0_replay_enabled`,
   - donor bundle path/hash,
   - donor cluster/run id,
   - host cluster/run id,
   - context validation status,
   - any dtype reconstruction status.

Keep host target logits/logit probabilities for the primary test. This isolates
the intervention to Phase-0 feature support/activations. If a later prompt shows
target-logit mismatch, add a separate donor-logit replay variant rather than
mixing both interventions in the first causal test.

### Comparison plan

For each swapped run, compute similarity to both the host baseline and donor
baseline:

- Phase-3 seed bundle:
  - support Jaccard,
  - shared influence Pearson/Spearman,
  - top-k seed overlap,
  - frontier pre/post Jaccard,
  - frontier rank drift.
- Compact graph:
  - feature Jaccard,
  - edge Jaccard and weighted edge Jaccard,
  - common-edge weight Pearson,
  - top-k edge overlap,
  - shared/unique endpoint mass decomposition.
- Semantic descriptor artifact:
  - top candidate exact-ID overlap,
  - shared candidate influence/activation stability,
  - unmatched high-mass candidate summary.

Report a simple donor-movement score for each metric where larger means more
similar:

```text
movement_to_donor = sim(swapped, donor_baseline) - sim(swapped, host_baseline)
```

Positive movement on Phase-3 frontier and compact graph metrics indicates that
the donor Phase-0 state causally pulls downstream artifacts toward the donor
cluster.

### Expected interpretation logic

- **Self-replay sanity passes and cross-swap becomes donor-like:** strong causal
  evidence that Phase-0 support/activation state carries the downstream Phase-3
  and graph exact-ID split.
- **Self-replay sanity passes and cross-swap remains host-like:** Phase-3/Phase-4
  computations or host target-logit context contribute additional independent
  cluster drift beyond Phase-0 state.
- **Self-replay sanity passes and cross-swap is intermediate:** both Phase-0
  state and later host computations matter.
- **Self-replay sanity fails:** do not interpret cross-swap results; first fix
  replay serialization, row ordering, dtype reconstruction, or context rebuild.

### Acceptance criteria

- Capture mode emits a Phase-0 donor bundle without changing normal attribution
  outputs when replay is disabled.
- Bundle load validates shape, dtype metadata, row ordering, prompt identity, CLT
  constants hash, and target token ids.
- Self-replay on each cluster reproduces its normal baseline within tight
  tolerances:
  - feature support Jaccard `1.0`,
  - Phase-3 seed influence Pearson at least `0.9999`,
  - Phase-3 top-1024 overlap at least `0.999`,
  - compact graph weighted edge Jaccard at least `0.999` unless explicitly
    explained by dtype round-trip loss.
- Cross-swap outputs include enough metadata to identify host vs donor provenance
  in `EXPERIMENTS.md` without reading SLURM logs.
- Offline comparison script produces host-vs-donor movement summaries for the
  four-condition matrix.
- CPU-only tests cover:
  - donor bundle save/load round trip, including `bfloat16` reconstruction,
  - strict context validation failure modes,
  - movement-score calculation on synthetic summaries.

### Risks and open questions

- `bfloat16` exact replay may require storing raw bf16 bits, not just float32
  analysis values.
- Rebuilding all derived row metadata from donor state is fragile; row order must
  remain identical to the original Phase-0 order.
- Host target logits are currently matched for `94_base`, but other prompts may
  have target-logit or probability differences; those need separate treatment.
- A donor Phase-0 activation matrix may be numerically inconsistent with the
  host model's hidden states. This is intentional for the counterfactual, but the
  run metadata must make the intervention clear.
- Replay may require changes in the sibling `circuit-tracer_chunked` library, not
  just project-side artifact plumbing.

## Validation

- Extend existing cross-cluster debug artifact tests to check new fields are
  preserved through summary / stream emission.
- Extend CLT diagnostic tests to assert new Phase-0 fingerprint fields exist and
  reflect post-zero-position masking.
- Add bundle-capture, descriptor-capture, graph-decomposition, and semantic
  comparison CPU-only checks.
- Only run safe local validation (`uv run ...`), no GPU workloads outside SLURM.

## Phase-3 gradient / row boundary follow-up

### Problem statement

The completed `94_base` Phase-0 replay matrix changed the interpretation of the
causal boundary. Same-cluster self-replay controls passed, and both cross-swaps
copied donor Phase-0 support exactly, but Phase-3 influence scores, frontiers,
and compact edge weights remained host-like.

Therefore Phase-0 active feature support/activation replacement is insufficient
to transfer the downstream graph. The next suspected boundary is Phase-3
direct-effect scoring: host backward gradients, error/token vectors, row
construction, row normalization, and influence ranking.

### Scope / non-goals

Scope:

- add passive capture for Phase-3 gradient/direct-row state on `94_base`,
- compare baseline/self-replay/cross-swap gradients and rows offline,
- decide whether to implement Phase-3 row replay or gradient replay next.

Non-goals:

- do not rename this as a richer Phase-0 bundle; gradients are Phase-3 state,
- do not make generation or forward pass deterministic across clusters in this
  step,
- do not emit dense tensors into JSON artifacts,
- do not replay gradients before the passive capture tells us whether row replay
  is the cleaner causal boundary.

### Proposed capture artifacts

Add opt-in artifacts, for example:

- `step_000_phase3_gradient_bundle.npz`
- `step_000_phase3_row_bundle.npz`

Gradient bundle minimum fields:

- `schema_version`
- target token ids/probabilities and hashes,
- prompt/context hashes,
- active feature support/value hashes for the runtime state,
- per-layer gradient hashes and summary stats,
- compact gradient tensor when feasible, shaped like
  `[n_layers, n_targets, n_pos, d_model]`,
- missing-layer mask / layer ids,
- provenance: host cluster/run/scenario, project/library commit, snapshot roots.

Row bundle minimum fields:

- Phase-3 logit feature rows or bounded row slices if full rows are too large,
- row-L1 totals used by the influence solver,
- row-L1 splits by column family when available: feature/error/token/total,
- row hashes/stats per target,
- seed influence hash and frontier hashes derived from these rows.

Both artifacts must be passive: enabling capture must not alter Phase-3 ranking,
Phase-4 frontier expansion, or compact graph packaging.

### Comparison plan

Run the same matrix shape as the Phase-0 replay test, with richer capture enabled:

1. Ascend self-replay with Ascend Phase-0 donor.
2. Ascend host with Cardinal Phase-0 donor.
3. Cardinal self-replay with Cardinal Phase-0 donor.
4. Cardinal host with Ascend Phase-0 donor.

Compare each cross-swap to both host and donor baselines for:

- gradient hash/stats similarity,
- raw Phase-3 row similarity,
- row-L1 split similarity,
- seed influence vector similarity,
- frontier pre/post locality overlap,
- compact graph similarity.

Expected if the current hypothesis is right:

- donor Phase-0 support/activation hashes remain donor-like,
- gradients and Phase-3 rows remain host-like,
- influence/frontier/edge metrics remain host-like.

### Causal replay preference after capture

Prefer Phase-3 row replay as the next causal intervention:

- row replay directly tests whether fixed Phase-3 rows make the influence solver
  and frontier construction donor-like,
- gradient replay is a second-order follow-up if row replay localizes the drift
  to row construction but we still need to distinguish gradients from
  decoder/activation contraction.

Interpretation:

- **Donor row replay moves frontier/graph donor-like:** drift is before or inside
  Phase-3 row construction, likely host gradients/direct effects.
- **Donor row replay remains host-like:** investigate row mapping,
  normalization, influence solver, or ranking/frontier logic.
- **Gradient replay moves rows donor-like but row replay was already sufficient:**
  gradients are a plausible root cause, but row replay remains the cleaner
  operational mitigation/checkpoint.
