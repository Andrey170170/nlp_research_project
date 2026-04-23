# Current Implementation Plan — Full Downstream/Semantic Stability Capture

## Problem statement

The finished `94_base` boundary-localization pair showed that Ascend/Cardinal
diverge already at the **pre-CLT input** boundary while CLT constants match. That
is plausible hardware/runtime numerical drift. The next expensive run should
therefore be loaded with enough artifacts to answer the more important question:

> Does upstream numerical drift materially change the extracted circuit, or is
> the graph semantically stable despite exact-ID / score perturbations?

We should implement the full passive capture and offline analysis stack **before**
launching the next matched Ascend/Cardinal pair.

## Current state

- Within-cluster baseline consistency has already been validated.
- The 6-job Phase-0 compare-mode matrix was negative for `94_base`:
  `baseline`, `fp32`, and `fp64` all produced the same cross-cluster split.
- Boundary localization pair completed successfully:
  - first token still matched: `Let`,
  - Phase-1 target-logit hash still matched: `437d56a13df41ec1`,
  - transcoder constants matched globally: `31c83df182f3f365`,
  - pre-CLT input fingerprints differed in all 26 layers,
  - downstream preactivation/margin/mask/post-activation also differed in all 26
    layers.
- Phase-3 seed bundle capture is implemented and committed.
- Enriched offline Phase-3 seed-bundle, compact-graph, and semantic-descriptor
  comparison helpers are implemented and committed.
- Passive feature semantic descriptor capture is implemented and committed in the
  project and sibling library.
- The queued boundary-localization pair did **not** include Phase-3 seed bundles
  or semantic descriptors, so it cannot answer the full downstream-stability
  question.

## Scope

Implement additional **passive** capture and CPU-offline comparison so the next
run can quantify four stability levels:

1. **Support stability** — how much Phase-0/Phase-3 mass is on shared vs unique
   features.
2. **Score/rank stability** — whether shared features keep similar activations,
   influences, ranks, and frontier status.
3. **Graph stability** — whether edge structure and weights are stable after
   decomposing shared/unique endpoints.
4. **Semantic stability** — whether exact-ID-unmatched features are still close
   semantic substitutes by descriptor / decoder-neighborhood evidence.

## Non-goals

- Do not attempt to eliminate all raw floating-point drift before this diagnostic
  run.
- Do not implement donor/swap/replay intervention yet.
- Do not save full dense tensors or full decoder matrices into artifacts.
- Do not run GPU/model-loading code outside SLURM.
- Do not launch the next matched pair until the run-readiness checklist below
  passes.

## Proposed implementation phases

### Phase 1 — enrich Phase-3 seed-bundle offline comparison

Extend `experiments/exact_trace_bench/phase3_seed_bundle_compare.py` and CLI
output to report shared-support score/rank stability.

Inputs already available in `step_000_phase3_seed_bundle.npz`:

- `active_features`
- `activation_values`
- `seed_feature_influences`
- `frontier_pre_locality`
- `frontier_post_locality`
- `queue_size`
- `actual_max_feature_nodes`

Add metrics:

- shared-support activation statistics:
  - Pearson correlation,
  - Spearman/rank correlation or deterministic rank-delta summary,
  - absolute/relative delta quantiles,
  - top-k activation overlap for configurable `k` values.
- shared-support seed-influence statistics:
  - Pearson correlation,
  - rank correlation / rank-delta quantiles,
  - sign agreement if signed influences are retained,
  - top-k overlap for `k in {64, 128, 256, 512, 1024}` clipped to available rows.
- unique-support contribution:
  - shared/left-only/right-only absolute influence mass,
  - top unique features by influence,
  - whether top unique features are near the seed cutoff / tie boundary.
- frontier decomposition:
  - pre/post frontier overlap overall,
  - pre/post frontier overlap restricted to shared support,
  - rank drift for shared frontier features,
  - improvement ratio from shared-support restriction.

Acceptance criteria:

- Existing basic comparator output remains backwards-compatible.
- Synthetic tests cover shared-only, unique-heavy, and rank-reordered cases.
- Local validation: `uv run ruff check ...` and `uv run pytest
  tests/test_phase3_seed_bundle_compare.py -q`.

### Phase 2 — add shared/unique compact graph decomposition

Extend `experiments/exact_trace_bench/graph_compare.py` or add a sibling helper
for richer compact graph analysis.

Current compact graph comparison reports global feature/edge Jaccard. Add:

- feature support decomposition:
  - shared, left-only, right-only counts by layer and position bucket.
- edge endpoint-class decomposition:
  - shared→shared,
  - shared→unique,
  - unique→shared,
  - unique→unique,
  - feature→logit / feature→error-like rows separately when identifiable.
- edge stability on shared endpoints:
  - unweighted Jaccard,
  - weighted Jaccard,
  - common-edge weight Pearson correlation,
  - common-edge absolute/relative delta quantiles,
  - top-k edge overlap by absolute weight.
- graph-mass accounting:
  - fraction of retained edge mass involving only shared features,
  - fraction involving any unique feature,
  - fraction going directly to logit rows.

Acceptance criteria:

- Existing `compare-compact` command still works.
- Enhanced output makes it clear whether low global edge Jaccard is due to
  unique endpoint churn or changed edges among shared features.
- Add small CPU-only tests with hand-built `StepData` fixtures.

### Phase 3 — implement passive semantic descriptor capture for candidate features

Add an opt-in runtime artifact, for example:

- `step_000_feature_semantic_descriptors.npz`

The goal is not to dump full decoder matrices. The goal is to save compact
descriptors for a bounded candidate set so unmatched features can be compared
semantically after the run.

Candidate feature sources:

- top seed-influence features,
- Phase-3 pre-locality frontier,
- Phase-3 post-locality frontier,
- final selected/retained feature set if accessible after Phase-4,
- compact-graph edge endpoints if practical without a second GPU pass.

Descriptor fields:

- `candidate_features`: `(M, 3)` `[layer, position, feature_idx]`,
- source masks / ranks:
  - `is_top_seed`, `seed_rank`, `seed_influence`,
  - `is_frontier_pre`, `frontier_pre_rank`,
  - `is_frontier_post`, `frontier_post_rank`,
  - `is_selected_phase4` if available,
- `activation_value`,
- compact decoder/semantic sketch:
  - fixed-size deterministic projection, e.g. 32–128 floats,
  - sampled coordinate sketch and/or top-absolute-coordinate sketch,
  - descriptor metadata including projection seed/version and source tensor kind.

Implementation notes:

- Add flags such as:
  - `--capture-feature-semantic-descriptors`,
  - `--semantic-descriptor-top-k`, default bounded, e.g. `2048`,
  - `--semantic-descriptor-dim`, default bounded, e.g. `64`.
- Prefer capturing from already-loaded decoder/provider state to avoid a large
  second pass.
- If full decoder access is too invasive in Phase 3, implement a fallback
  descriptor that still captures candidate labels, ranks, activations, influence,
  and local graph-neighborhood signatures; then plan a separate SLURM semantic
  descriptor job only if needed.
- Keep artifact size bounded and record truncation/candidate-count metadata.

Acceptance criteria:

- Artifact is emitted only when the flag is enabled.
- Descriptor capture does not change attribution ranking/frontier behavior.
- Manifest/extractor/indexing reports descriptor artifact path/status.
- Unit tests validate descriptor artifact schema on small fake tensors/helpers.

### Phase 4 — implement semantic comparison and remapped graph analysis

Add an offline comparator, for example:

- `experiments/exact_trace_bench/semantic_feature_compare.py`
- CLI command: `compare-semantic-features`

Inputs:

- left/right Phase-3 seed bundles,
- left/right compact graph `.npz`,
- left/right semantic descriptor `.npz` if present.

Outputs:

- exact-ID unmatched candidate summary:
  - top left-only and right-only features by seed influence / edge mass,
  - layer/position distribution of unmatched high-mass features.
- candidate semantic matching:
  - nearest-neighbor matches constrained by same layer and same/near position,
  - descriptor cosine similarity,
  - seed-influence / activation similarity,
  - local edge-neighborhood Jaccard when compact graph endpoints are available.
- semantic-substitute report:
  - fraction of left-only high-mass features with a strong right-side semantic
    match,
  - fraction of right-only high-mass features with a strong left-side match,
  - mass-weighted semantic-match coverage.
- optional remapped graph metrics:
  - exact graph metrics before remapping,
  - graph metrics after mapping high-confidence semantic substitutes,
  - list of high-confidence and ambiguous matches.

Interpretation labels:

- `exact_id_stable`
- `semantic_substitutes_explain_mismatch`
- `shared_support_scores_unstable`
- `unique_features_semantically_unmatched`
- `insufficient_descriptor_coverage`

Acceptance criteria:

- Works entirely CPU-offline on saved artifacts.
- Has synthetic tests where exact IDs differ but descriptors intentionally match.
- Fails clearly when descriptor artifacts are absent, or degrades to
  neighborhood-only comparison with an explicit warning.

### Phase 5 — scenario plumbing, extraction, docs, and launch readiness

Add scenario/config passthrough for all new flags:

- `capture_phase3_seed_bundle=true`,
- `capture_feature_semantic_descriptors=true`,
- descriptor top-k / dimension knobs.

Update:

- `trace_pipeline_chunked.py`,
- `experiments/run_sparsification_experiment.py`,
- `experiments/exact_trace_bench/scenarios.py`,
- extractors/indexers,
- `EXPERIMENTS.md`,
- `TODO.md`,
- `docs/phase0_boundary_fingerprinting_spec.md` or a new dedicated semantic
  stability spec if this becomes large.

Acceptance criteria:

- Scenario JSON can request the richer capture set.
- Completion manifest records artifact statuses and paths.
- Extracted benchmark index includes both seed-bundle and semantic-descriptor
  artifacts.
- Docs state that this is passive capture, not an intervention.

## Intended next launch batch after implementation

Run a new **2-job matched baseline pair** for `94_base` via immutable snapshot.

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
- feature semantic descriptor capture enabled

Jobs:

1. Ascend baseline
2. Cardinal baseline

## Post-run analysis order

1. Confirm provenance and run success for both project and sibling library
   snapshots.
2. Confirm first-token and Phase-1 target-logit invariants.
3. Re-run Phase-0 boundary comparison and record earliest divergent boundary.
4. Run enriched Phase-3 seed-bundle comparison.
5. Run enhanced compact graph shared/unique decomposition.
6. Run semantic descriptor matching for unmatched high-mass features.
7. Decide whether the graph is:
   - exact-ID stable,
   - semantically stable despite ID churn,
   - or genuinely unstable and in need of mitigation/replay work.

## Run-readiness checklist

Before launching the next matched pair:

- [x] Phase-3 bundle comparator reports rank/score/top-k stability.
- [x] Compact graph comparator reports shared/unique edge decomposition.
- [x] Semantic descriptor capture emits bounded artifact under an opt-in flag.
- [x] Semantic descriptor comparator can match intentionally substituted features
      in a synthetic test.
- [x] Scenario plumbing exposes all capture flags.
- [x] Completion manifests and extractors report all new artifacts.
- [ ] Safe local validation passes with `uv run` only.
- [ ] Project and sibling library commits are clean and recorded.
- [ ] Immutable paired workspace snapshot is created from the clean commits.

## Risks and open questions

- Decoder-vector access may be expensive or awkward in the exact chunked path. If
  so, implement candidate/rank/neighborhood descriptors first and schedule a
  separate SLURM-only descriptor job for decoder sketches.
- Semantic matching can create false comfort if descriptors are too weak. Use
  conservative thresholds and report ambiguous matches separately.
- Candidate caps may miss low-rank but high-edge-mass features. Include graph
  endpoint candidates if practical, and record descriptor coverage.
- Hash/fingerprint differences are sampled diagnostics. They are strong
  localization evidence but not full dense equality proofs.

## Local validation commands

Use only safe login-node checks:

```bash
uv run ruff check .
uv run pytest tests/test_phase3_seed_bundle_compare.py -q
uv run pytest tests/test_cross_cluster_debug_artifacts.py -q
```

Add targeted tests for any new graph/semantic comparator modules as they land.
