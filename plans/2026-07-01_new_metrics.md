# ADAG-lite temporal stability metrics plan

Status: implementation-ready design spec for the next analysis pass  
Date: 2026-07-01  
Scope: project repo `nlp_research_project`; no new model tracing required

## Problem statement

The 19 real-prompt correct/wrong pairs have already been fully traced at every
generated token. The previous downstream analysis mixed several confounds:

- target-token role churn, e.g. comparing punctuation graphs to math-number
  graphs;
- exact feature-ID churn, where semantically similar transcoder features may
  substitute for each other across adjacent tokens;
- metric fishing over a large metric battery whose individual meanings were not
  always clear;
- role-matched pair construction that depended on weak role labels.

The new analysis should measure temporal graph stability while separating these
confounds. The core scientific question is whether reasoning/math/final-answer
regions contain a stable circuit core, whether stability increases toward the
final answer, and whether correct trajectories are more stable than wrong ones.

This plan defines a reusable analysis substrate and a small ADAG-inspired metric
stack. Role definitions are intentionally left as a separate next design step;
this plan only defines the interfaces that role tables must satisfy.

## Inputs and existing artifacts

Canonical inventory already prepared:

- `reports/real_prompt_19_inventory_20260701/README.md`
- `reports/real_prompt_19_inventory_20260701/prompt_pairs.csv`
- `reports/real_prompt_19_inventory_20260701/trajectories.csv`
- `reports/real_prompt_19_inventory_20260701/tokens.csv`
- `reports/real_prompt_19_inventory_20260701/trajectory_texts.md`

Validated inventory counts:

- prompt pairs: `19`,
- trajectories: `38`,
- token graph rows: `7714`,
- all traced trajectory aggregates are OK.

Trace roots:

- v1 prior 9-pair wave:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/real-prompt-screen-v1/trace_launch_prep_fullseq_wreuse_w1600`
- v2 combined 10-pair wave:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/real-prompt-screen-v2/trace_launch_prep_fullseq_wreuse_w1600_combined10`

Each token graph is `graph.npz` and contains typed-bucketed compact graph data
with buckets including:

- `feature<-token`,
- `feature<-feature`,
- `feature<-error`,
- `logit<-feature`,
- `logit<-error`,
- `logit<-token`.

Important implementation note: existing `GraphSnapshot` helpers mostly expose
absolute-weight comparison maps. ADAG-lite profile extraction must preserve signed
bucket weights where available, especially for output/contribution profiles and
opposing-effect cluster diagnostics.

## Scope

In scope:

1. Build signed feature attribution profiles from existing `graph.npz` files.
2. Include `feature<-feature` structure as a first-class profile view, not only
   as a post-hoc metric.
3. Cluster feature identities into ADAG-lite supernodes using multi-view
   functional similarity.
4. Compute a small, interpretable set of exact, decoder-soft, and
   cluster-collapsed graph stability metrics.
5. Make role labels versioned and replaceable without re-extracting graph
   profiles or retracing.
6. Aggregate scientific comparisons by prompt pair, not raw token count.

Non-goals for this pass:

- No new GPU/model tracing.
- No LLM-generated natural-language explanations of clusters in v1.
- No causal steering/intervention validation in v1.
- No final role schema in this document; roles will be designed next.
- No reuse of the prior role-matched pct20 scorecard as a scientific result.
- No promotion of a large metric battery as the main result.

## Proposed analysis root layout

Large derived artifacts should live on scratch, not in git history. Use a root
like:

```text
/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/
  real-prompt-adag-lite-19prompts-20260701/
    source_manifest.json
    inventory_snapshot/
    graph_profiles/
    roles/
    pair_manifests/
    feature_clusters/
    metrics/
    summaries/
```

Small human-readable summaries may be copied into a repo-local report directory:

```text
reports/real_prompt_adag_lite_19prompts_20260701/
```

The scratch root must record:

- project repo branch/commit/dirty files,
- sibling `../circuit-tracer_chunked` branch/commit/dirty files,
- source inventory manifest path,
- source trace roots,
- profile/role/cluster/metric version IDs.

## Primary analysis objective and normalization

The headline analysis is temporal stability evolution over the generated chain of
thought. For each prompt/trajectory and each approved role grouping, the analysis
should answer:

- does graph stability stay flat, increase, or fall over generation time?
- does the trend differ between `problem_setup`, `work_step`,
  `final_calculation`, and `answer_statement` regions?
- does the trend differ between broad groups such as `reasoning_prose`,
  `math_core`, and `answer_core`?
- do correct and wrong trajectories differ at the prompt-pair level?

Stable-core and substructure discovery are secondary analyses. They should help
explain the primary stability signal, e.g. by showing whether a stability
increase comes from persistent cluster nodes/flows, but should not replace the
temporal-stability trend as the headline result.

Use two position normalizations everywhere temporal trends are reported:

1. **Global generated-token fraction**: position over the full generated
   trajectory, so different-length completions can be compared.
2. **Region-relative fraction**: position within a role-assigned region, so
   different-length `work_step` or `final_calculation` spans contribute
   comparably.

Use prompt-level aggregation for headline claims:

- compute token-pair metrics,
- aggregate to trajectory/region/group summaries,
- aggregate to prompt-pair correct-vs-wrong deltas where relevant,
- then average over prompts with equal prompt weight.

Do not let longer completions dominate merely because they contain more token
pairs. Always report raw support counts beside normalized summaries.

## Implementation order decision

Implement the full pipeline, but stage it in this order:

1. **Roles first**
   - generate deterministic `roles_regex.csv`,
   - review selected traces,
   - freeze `roles_v1.csv`.
2. **ADAG-lite clustering**
   - extract signed graph profiles,
   - build within-layer feature clusters for v1.
3. **Metrics computation**
   - compute exact, decoder-soft, and cluster-collapsed core metrics.
4. **E2E prompt-pair test run**
   - run the complete pipeline on 2--4 prompt pairs before the full run,
   - each phase still needs its own local/probe tests before this E2E run.
5. **Full 19-prompt run**
   - launch only after the E2E run produces coherent role joins, clusters,
     metric rows, and prompt-level summaries.

Suggested E2E prompt-pair set:

- `486` — short, near-equal v1 pair,
- `709` — longer v1 pair with LaTeX/fractions/rates and larger length
  difference,
- `401` — v2 equal-length pair with clean structure,
- `877` — v2 long pair with weird/caveat-style wrong trajectory.

This order is operational, not a narrowing of scope: the implementation plan is
for the complete roles → clustering → metrics → summaries pipeline.

## Implementation placement decision

Keep this analysis code distinct from the normal tracing/run harness so future
readers do not confuse post-hoc analysis with SLURM trace production.

Default package location:

```text
src/nlp_research_project/circuit_stability_analysis/
```

Suggested submodules:

```text
circuit_stability_analysis/
  __init__.py
  cli.py                     # separate analysis CLI, not trace-launch CLI
  inventory.py               # read neutral 19-prompt inventory tables
  roles.py                   # span detection, regex roles, review export/import
  signed_graph.py            # signed graph.npz bucket loader/endpoint decoder
  profiles.py                # ADAG-lite profile extraction
  clustering.py              # within-layer multi-view feature clustering
  pairs.py                   # role-aware and role-independent pair manifests
  metrics.py                 # exact / decoder-soft / cluster-collapsed metrics
  summaries.py               # prompt-normalized summaries and plot inputs
  schemas.py                 # dataclasses/schema validation helpers
```

The package may import reusable read-only helpers from
`nlp_research_project.exact_trace_bench` where appropriate, for example JSON/CSV
I/O utilities, decoder signature cache helpers, or already-tested graph endpoint
decoders. However, new analysis-specific logic should live in the separate
package above, not in the trace-planning/runner modules.

Expose commands through a separate CLI entrypoint, e.g.

```text
circuit-stability-analysis =
  nlp_research_project.circuit_stability_analysis.cli:main
```

or equivalently via `uv run python -m nlp_research_project.circuit_stability_analysis.cli`
during early development. Do not overload ordinary `exact-trace-bench` trace-run
commands with these analysis workflows unless a small shared utility command is
clearly justified.

Tests should also be separate, e.g. `tests/test_circuit_stability_*.py`, with
small synthetic graph fixtures and optional smoke tests against one explicit
scratch `graph.npz` path.

## Data model

### Stable token key

Use this key for role tables, pair manifests, and metric joins:

```text
(wave, prompt_id, label, generated_index)
```

where `label in {correct, wrong}`. Also carry `trajectory_name` and
`graph_path` as redundant path-oriented keys for debugging.

### Feature identity key

Primary cluster unit:

```text
feature_key = (layer, feature_id)
```

Feature observations retain position:

```text
feature_observation_key = (
  wave, prompt_id, label, generated_index,
  layer, feature_position, feature_id
)
```

Default clustering should be **within layer** unless a later validation pass shows
cross-layer decoder/profile clustering is reliable. Cross-layer clustering can be
implemented as an explicit optional mode.

### Role table interface

Role definitions are external and versioned. A role table must contain at least:

```text
wave,prompt_id,label,trajectory_name,generated_index,
region,coarse_role,fine_role,role_confidence,role_version
```

The analysis code must not hardcode particular role names beyond reading these
columns. Pair-manifest builders can then filter by whichever role tiers we decide
next.

## ADAG-lite feature profiles

The paper's useful idea is to cluster features by **functional profile**, not by
feature ID or decoder similarity alone. For our existing artifacts, build four
profile views per `feature_key`.

Implementation note 2026-07-02: profile extraction now records observed bucket
weight sign statistics in `graph_profile_audit.json`. A one-graph local smoke and
a 32-graph CPU-SLURM smoke both observed
`bucket_sign_mode=nonnegative_observed`, which means the current historical
`graph.npz` buckets appear to contain retained absolute magnitudes for the
sampled artifacts. ADAG-lite code preserves signs when present and tests
synthetic signed buckets, but scientific claims that depend on
support-vs-suppression must be gated on this audit. If the full run also reports
nonnegative-only buckets, treat v1 clustering as an absolute-mass
structural/profile clustering, not a proper signed-output-effect clustering,
unless signed bucket artifacts are regenerated later.

32-graph smoke:

```text
/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/real-prompt-adag-lite-smoke32-cap5k-20260702
SLURM job: 12164946
graphs_seen: 32
max_edges_per_bucket: 5000
decode_errors: 0
cluster_count: 12
assigned_feature_count: 256
```

### View A — token input attribution profile

Source bucket: `feature<-token`.

Purpose: capture which prior/context tokens feed a feature.

Per feature observation, extract signed and absolute mass from token-source edges
into a sparse vector over stable coordinates such as:

- source token absolute position,
- relative position to target token,
- relative position to feature position when available,
- source token text / token id,
- later: source token region/coarse/fine role when a role table is supplied.

Default cluster profile should avoid overfitting exact token text. Keep both:

1. raw token-coordinate profiles for diagnostics, and
2. binned profiles for clustering, e.g. relative-distance bins and source-role
   bins once roles exist.

### View B — output/logit contribution profile

Source bucket: `logit<-feature`.

Purpose: capture what next-token logits a feature supports or suppresses.

Use signed bucket weights. Coordinates should include:

- logit token id,
- decoded logit token text when available,
- rank/order among retained logits,
- sign of contribution,
- optional grouping by emitted target token role once role tables exist.

This view is the main guard against clustering features that have similar inputs
but opposite output effects.

### View C — feature-to-feature structural profile

Source bucket: `feature<-feature`.

Purpose: capture circuit neighborhood and multi-step computation structure.

This view is required for the first proper ADAG-lite implementation. Build three
subprofiles:

1. **Layer-flow F2F profile**: incoming/outgoing mass by
   `(source_layer, target_layer, direction)`.
2. **Position-flow F2F profile**: incoming/outgoing mass by relative source/target
   position buckets, e.g. same position, previous 1--2, previous 3--8,
   previous 9+, future/leakage bucket for audit only.
3. **Neighbor-identity F2F profile**: sparse bag of neighboring
   `(layer, feature_id)` endpoints, used as an ablation/diagnostic and later for
   iterative refinement. Do not let this exact-neighbor view dominate v1
   clustering, or it may reintroduce exact-ID churn.

For `feature<-feature`, existing code shows endpoint IDs can be decoded with the
`layer * (n_pos * 1_000_000) + position * 1_000_000 + feature_id` scheme. The new
extractor should reuse or harden this decoder and add tests.

### View D — decoder-vector profile

Source: GemmaScope transcoder decoder vectors via the existing decoder signature
cache machinery.

Purpose: identify alias features whose learned decoder vectors point in similar
directions.

Use decoder cosine as a prior, not as the only cluster criterion. A high decoder
cosine should not override opposite output contribution or incompatible F2F
structure.

## Multi-view similarity and clustering

### Similarity rule

For two feature keys `i,j`, compute per-view cosine similarities:

```text
s_input(i,j)
s_output(i,j)
s_f2f_layerflow(i,j)
s_f2f_positionflow(i,j)
s_decoder(i,j)
```

Use clamped non-negative similarities for affinity construction:

```text
p_view = max(0, s_view)
```

Default ADAG-lite affinity should be a weighted harmonic mean over required
functional views, with decoder as a bounded prior:

```text
functional = harmonic_mean(
  p_input,
  p_output,
  p_f2f_layerflow,
  p_f2f_positionflow
)

affinity = functional * (1 + decoder_prior_weight * p_decoder)
```

Default `decoder_prior_weight` should be small, e.g. `0.10` to `0.25`, and must be
recorded in the cluster manifest.

Rationale:

- harmonic mean penalizes disagreement in any functional view;
- negative/opposing output effects become zero affinity;
- decoder similarity can help merge aliases but cannot force a bad merge.

### Feature universe and scaling

The 19-prompt set contains 7714 token graphs and potentially many unique feature
IDs. Dense all-pairs clustering may be too expensive. Implement scalable defaults:

1. Extract profiles for all observed feature keys that pass minimal sanity checks.
2. Define a clusterable universe by configurable support thresholds, e.g.:
   - `min_observation_count`,
   - `min_total_abs_mass`,
   - optional top-N per layer by total contribution mass.
3. Assign low-support excluded features to explicit singleton/unclustered buckets
   during collapsed metric computation.
4. Build sparse kNN affinity graphs per layer using profile-vector nearest
   neighbors rather than dense all-pairs matrices.
5. Use scipy sparse linear algebra for spectral embeddings; avoid adding heavy new
   dependencies unless explicitly justified.

### Clustering algorithm

Default v1:

1. Cluster within layer.
2. Build sparse affinity graph from multi-view kNN candidates.
3. Construct normalized graph Laplacian.
4. Compute spectral embedding for a configurable `k` or a small `k` sweep.
5. Run k-means on the embedding.
6. Emit cluster assignment and QC summaries.

Decision: v1 clustering is within-layer only. Cross-layer clustering is a later
exploratory mode after within-layer ADAG-lite metrics are validated.

Implementation can use `scipy.sparse.csgraph`, `scipy.sparse.linalg.eigsh`, and
`scipy.cluster.vq.kmeans2` to avoid introducing scikit-learn as a hard dependency.

### Cluster quality diagnostics

For every clustering run, write:

- cluster size distribution and coefficient of variation,
- sampled silhouette score by final affinity distance,
- fraction of intra-cluster pairs with opposing signed output profiles,
- mean/quantiles of input/output/F2F/decoder similarities within clusters,
- ablation comparison for clusterings with/without decoder prior,
- ablation comparison for clusterings with/without F2F structural views,
- top exemplar tokens/contexts per cluster by contribution mass.

These diagnostics are descriptive. They are not final scientific conclusions.

## Collapsed graph metrics

After clusters exist, compute graph stability at three levels.

### Level 1 — exact graph metrics

Keep exact-ID metrics as a lower-bound diagnostic:

- exact feature-node weighted Jaccard,
- exact `feature<-feature` weighted Jaccard,
- exact `feature<-token` and `logit<-feature` weighted Jaccard,
- exact layer-flow sanity metrics.

### Level 2 — decoder-soft metrics

For feature-node comparisons, compute decoder-soft matched mass:

- greedy or optimal matching within layer above cosine thresholds,
- thresholds such as `tau in {0.70, 0.80, 0.90}`,
- report matched mass fraction for both graphs and soft weighted Jaccard.

This separates exact-ID churn from likely semantic alias churn.

### Level 3 — cluster-collapsed metrics

Collapse feature endpoints using ADAG-lite cluster IDs. Compute at multiple
granularities:

1. **Positionless cluster nodes**:
   - node key: `(cluster_id)`.
   - best for semantic-core stability.
2. **Layer/cluster nodes**:
   - node key: `(layer, cluster_id)`.
   - default if cluster IDs are only layer-local.
3. **Position-binned cluster nodes**:
   - node key: `(cluster_id, relative_position_bucket)`.
   - useful to test whether stability is semantic or merely positional.

Collapsed edge metrics:

- `feature<-feature`: `(target_cluster, source_cluster)` weighted maps,
- `feature<-token`: `(target_cluster, source_token_role_or_distance_bin)` maps,
- `logit<-feature`: `(logit_token_id_or_role, source_cluster)` maps,
- coarse layer/cluster flow maps.

Primary collapsed stability metrics should be simple:

- weighted Jaccard,
- cosine over aligned collapsed edge/node mass vectors,
- top-K overlap for a small K set only if K is interpretable,
- stable-core persistence over adjacent windows.

Avoid making derived-K mass fractions primary outcomes; those are mostly
single-graph concentration diagnostics, not direct graph similarity measures.

## Pair-manifest strategy

Pair manifests must be regenerated whenever roles change, but graph profiles and
cluster assignments should remain reusable.

### Role-independent base pairs

Always generate these from `tokens.csv`:

1. Adjacent temporal pairs within each trajectory: `t -> t+1`.
2. Lagged temporal pairs: e.g. `1, 2, 5, 10, 20` generated-token lags.
3. Correct-vs-wrong relative-position pairs within the same prompt.
4. Same-prompt final-window correct-vs-wrong pairs.
5. Cross-prompt null controls matched by wave, label, token fraction, and later
   role tier.

### Role-aware derived pairs

After roles are decided, derive:

1. Same-region adjacent temporal pairs.
2. Same-coarse-role adjacent and lagged pairs.
3. Same-fine-role pairs where enough support exists.
4. Reasoning/math/final-answer window pairs.
5. Punctuation/formatting controls.

All pair manifests must include:

- role version,
- cluster version,
- pair-category and sub-category,
- prompt id and label,
- generated indices for both sides,
- token texts and role fields for both sides,
- graph paths for both sides.

## Scientific summaries to produce

Once roles and metrics are available, produce prompt-level summaries:

1. **Temporal stability curves**
   - x-axis: generated-token fraction or region-relative progress,
   - y-axis: exact / decoder-soft / cluster-collapsed stability,
   - separate correct and wrong trajectories,
   - aggregate by prompt pair with confidence intervals over prompts.
   - headline curves: `problem_setup`, `work_step`, `final_calculation`,
     `answer_statement`; `reasoning_prose`, `math_core`, `answer_core`.

2. **Final-answer stability**
   - compare last N reasoning/math/final-answer tokens,
   - test whether stability increases near final output,
   - report prompt-level correct-minus-wrong deltas.

3. **Stable core persistence**
   - for each trajectory, identify cluster nodes/edges persistent across a rolling
     window,
   - measure persistent mass fraction and core overlap over time.

4. **Churn decomposition**
   - exact-ID churn,
   - alias-resolved decoder/cluster churn,
   - residual churn after collapsing,
   - by region/coarse/fine role.

5. **Substructure discovery**
   - top stable clusters and cluster-to-cluster flows in reasoning/math/final
     regions,
   - compare cluster involvement in correct vs wrong final regions,
   - list exemplar tokens/contexts for each high-mass stable cluster.

6. **Control-group behavior**
   - formatting and punctuation controls should be plotted separately from
     headline reasoning/math plots,
   - use the same y-axis metric units and the same global/region-relative
     normalization,
   - purpose: show whether headline stability behavior is specific to reasoning
     and math regions or merely a formatting/tokenization artifact.

## Implementation stages

### Stage 0 — Preserve neutral inventory baseline

Use `reports/real_prompt_19_inventory_20260701/` as the canonical input. Do not
modify it during metric development.

Acceptance criteria:

- Analysis root stores a copy or checksum of `source_manifest.json`.
- Full extraction checks still report 19 pairs / 38 trajectories / 7714 token
  graph rows.

### Stage 1 — Add signed bucket endpoint decoder

Implement a small loader separate from existing absolute-weight comparison code,
for example:

- `src/nlp_research_project/exact_trace_bench/full_answer/signed_graph.py`

Responsibilities:

1. Load `graph.npz` without model/GPU dependencies.
2. Preserve signed `bucket_weights`.
3. Decode endpoints for all required buckets:
   - `feature<-feature`,
   - `feature<-token`,
   - `logit<-feature`.
4. Expose both signed and absolute masses.
5. Record unknown/unsupported endpoint patterns as audit rows, not silent drops.

Acceptance criteria:

- Unit tests cover endpoint decoding on synthetic bucket arrays.
- Smoke test on one real `graph.npz` reconciles:
  - bucket names,
  - retained nnz counts,
  - retained absolute mass versus metadata within tolerance.
- Existing `temporal.py` comparisons are unchanged.

### Stage 2 — Extract feature observations and profiles

Add a CLI such as:

```bash
uv run exact-trace-bench extract-adag-lite-profiles \
  --inventory reports/real_prompt_19_inventory_20260701/source_manifest.json \
  --output-dir <scratch-analysis-root>/graph_profiles \
  --workers 8
```

Outputs:

```text
graph_profiles/
  feature_observations.jsonl[.gz]
  feature_profile_vectors.npz
  feature_profile_index.csv
  graph_profile_audit.json
```

`feature_observations` should include:

- stable token key,
- feature observation key,
- feature key `(layer, feature_id)`,
- feature position,
- per-view mass totals,
- top input-token contributors,
- top output-logit contributors,
- top F2F incoming/outgoing contributors,
- graph path.

`feature_profile_vectors.npz` should store sparse matrices for the four views:

- input/token,
- output/logit,
- F2F layer/position flow,
- F2F neighbor diagnostic,
- decoder vectors or decoder-cache references.

Acceptance criteria:

- Full run processes all 7714 graph paths or reports explicit skipped paths.
- Profile audit has zero missing graphs and zero decode failures for required
  buckets.
- Extraction can be resumed without duplicating rows.
- A small two-trajectory probe can run on a login node; full extraction should be
  launched as a CPU SLURM job if runtime or scratch I/O is heavy.

### Stage 3 — Build ADAG-lite feature clusters

Add a CLI such as:

```bash
uv run exact-trace-bench cluster-adag-lite-features \
  --profiles <scratch-analysis-root>/graph_profiles \
  --output-dir <scratch-analysis-root>/feature_clusters/clusters_v1 \
  --cluster-mode within_layer \
  --min-observation-count 5 \
  --decoder-prior-weight 0.15 \
  --include-f2f true
```

Outputs:

```text
feature_clusters/clusters_v1/
  cluster_manifest.json
  feature_cluster_assignments.csv
  cluster_summary.csv
  cluster_quality.json
  cluster_exemplars.jsonl
  view_ablation_quality.csv
```

Acceptance criteria:

- Every cluster assignment records profile version and cluster version.
- Low-support features are explicitly marked `unclustered` or singleton, not
  silently omitted.
- Cluster QC includes F2F ablation and decoder-prior ablation.
- Opposing-output-effect rate is reported and is acceptably low for selected
  clustering settings, or the run is marked exploratory.

### Stage 4 — Add role-table and pair-manifest builders

Roles are designed next, but implement the interface now.

Add CLIs such as:

```bash
uv run exact-trace-bench build-temporal-pair-manifest \
  --tokens reports/real_prompt_19_inventory_20260701/tokens.csv \
  --roles <roles.csv> \
  --role-version <role-version> \
  --output <scratch-analysis-root>/pair_manifests/<name>.json
```

Acceptance criteria:

- Pair manifests can be rebuilt after editing only `roles.csv`.
- Role-independent pair manifests work with no role table.
- Role-aware manifests include support counts by region/coarse/fine role.
- Pair manifests are deterministic and diffable.

### Stage 5 — Compute exact, decoder-soft, and cluster-collapsed metrics

Add a CLI such as:

```bash
uv run exact-trace-bench run-adag-lite-metrics \
  --pair-manifest <pair-manifest.json> \
  --clusters <cluster_manifest.json> \
  --output-dir <scratch-analysis-root>/metrics/<metric-version> \
  --metric-set core_v1 \
  --resume
```

Core v1 metric set:

1. exact feature-node weighted overlap,
2. exact bucket-level weighted overlap for `feature<-feature`, `feature<-token`,
   and `logit<-feature`,
3. proper decoder-soft feature-node matched mass / soft weighted Jaccard,
4. cluster-collapsed feature-node weighted overlap,
5. cluster-collapsed `feature<-feature` flow weighted overlap/cosine,
6. cluster-collapsed `logit<-feature` weighted overlap/cosine,
7. stable-core persistence over rolling windows.

Acceptance criteria:

- Metric rows are checkpointed per pair and resumable.
- Metric output records exact role version, cluster version, and metric version.
- Re-running with a new role table only recomputes missing/new pairs; graph
  profiles and cluster assignments are reused.
- Prompt-level aggregation is available directly from output summaries.

### Stage 6 — Summaries and plots

Only after roles and metric choices are approved, generate plots from metric rows.

Required summaries:

- prompt-level correct vs wrong deltas,
- region/coarse/fine role summaries,
- final-window stability summaries,
- churn decomposition exact vs decoder-soft vs cluster-collapsed,
- cluster exemplars for high-stability cores.

Acceptance criteria:

- Every plot has a backing CSV/JSON table.
- Plots aggregate by prompt pair as the statistical unit.
- No plot uses an implicit role or metric version.

## Validation plan

1. **Loader validation**
   - synthetic endpoint decoding tests,
   - one real graph smoke audit,
   - compare absolute bucket totals to metadata.

2. **Profile validation**
   - two trajectories end-to-end profile extraction,
   - inspect top input/output/F2F contributors for a few features,
   - verify signed output profiles preserve positive/negative effects.

3. **Clustering validation**
   - run small layer subset first,
   - inspect cluster exemplars,
   - compare no-F2F vs F2F clustering QC,
   - compare decoder-only nearest neighbors vs ADAG-lite clusters.

4. **Metric validation**
   - identity/self pairs should score at the noise ceiling,
   - adjacent exact-ID stability should be lower than decoder/cluster-collapsed
     stability when alias churn is present,
   - null controls should remain lower than role-matched temporal pairs if the
     metric is meaningful.

5. **Scientific sanity checks**
   - punctuation/formatting controls should show different stability behavior
     than math/reasoning/final-answer regions,
   - correct-vs-wrong claims must be prompt-level, not token-count-level.

## Operational notes

- GPU/model-loading work is not required for this plan.
- Full extraction/metric passes over scratch graphs may still be heavy filesystem
  and CPU work; use CPU SLURM jobs for full runs if probes are nontrivial.
- Do not run broad filesystem searches; all source paths are explicit in the
  inventory.
- Do not commit large metric rows, graph profile matrices, or cluster artifacts
  unless intentionally promoted as small fixtures. Commit only small specs,
  code, tests, and compact summaries.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Bucket endpoint encoding is partially misunderstood. | Build signed endpoint decoder with audits and real-graph mass reconciliation before any metrics. |
| Exact-neighbor F2F profile reintroduces exact-ID churn. | Use F2F layer/position flow as default; keep exact-neighbor F2F as ablation/diagnostic. |
| Decoder similarity merges functionally opposite features. | Use signed output profile and harmonic-mean affinity; report opposing-effect rate. |
| Too many unique features for dense clustering. | Use support thresholds, per-layer sparse kNN graphs, and explicit unclustered/singleton handling. |
| Role changes invalidate prior pair manifests. | Treat roles as versioned input; regenerate pair manifests while reusing graph profiles/clusters. |
| Metrics become too numerous again. | Freeze `core_v1` small metric set; keep exploratory metrics separate and hidden from primary summaries. |
| Token-level aggregation exaggerates significance. | Aggregate by prompt pair for primary correct-vs-wrong claims. |

## Remaining open questions

1. Exact implementation rules for the reviewed `roles_v1` table are tracked in
   `plans/2026-07-01_roles.md`.
2. How much cluster interpretability output is needed before plotting: exemplars
   only, or LLM descriptions later.
3. Whether to run a conservative and expanded role variant in parallel after the
   first reviewed role table exists.
