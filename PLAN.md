# Current Execution Plan — Phase-3 Gradient Boundary Probe

## Problem statement

The completed `94_base` Phase-0 replay matrix showed that replacing only the
Phase-0 active feature support/activation values copies donor support exactly,
but Phase-3 scores/frontiers and compact edge weights remain host-like.

That means the next causal boundary is not richer Phase-0 support alone. We need
to capture the Phase-3 gradient/direct-effect inputs that score those features,
then test whether gradient or row replay moves the graph toward the donor.

## Working hypothesis

Cross-cluster drift is dominated by host-side Phase-3 scoring state:

- backward gradients through the host cached forward graph,
- error/token vectors and direct-effect row construction,
- row-L1 normalization and influence ranking from those rows.

Phase-0 donor replay currently creates a mixed counterfactual:

```text
donor feature support/activation values × host Phase-3 gradient field
```

The observed host-like downstream result is expected if the gradient/direct-row
field dominates the ranking.

## Status checklist

- [x] Phase-0 donor capture/replay implemented and validated.
- [x] `94_base` Phase-0 replay matrix completed on Ascend/Cardinal.
- [x] Self-replay controls passed on both clusters.
- [x] Cross-swaps stayed host-like in Phase-3/edge metrics despite donor support.
- [ ] Implement passive Phase-3 gradient/direct-row capture.
- [ ] Launch matched gradient donor-capture pair.
- [ ] Launch Phase-0 + Phase-3-gradient capture matrix.
- [ ] Compare gradients/rows host-vs-donor and decide replay boundary.
- [ ] Implement Phase-3 row replay or gradient replay if capture confirms the
      drift is in row construction.

## Implementation phase

### 1. Add passive Phase-3 gradient bundle capture

Suggested flag:

- `--capture-phase3-gradient-bundle`

Suggested artifact:

- `step_000_phase3_gradient_bundle.npz`

Capture gradients from `phase3_logits` attribution only. Minimum contents:

- `schema_version`
- `target_token_ids`, `target_probabilities`, target hashes
- prompt/context hashes already used by Phase-0 validation
- active-feature support/value hashes for the runtime state
- per-layer gradient hashes and stats
- compact full gradient tensor if feasible:
  - shape `[n_layers, n_targets, n_pos, d_model]`
  - dtype `float32` initially, optionally `float16/bfloat16` only for analysis
    copies if storage pressure requires it
- missing-layer mask / layer ids
- provenance: host cluster/run/scenario, project/library commit, snapshot roots

Keep this capture passive: it must not alter Phase-3 rows or frontier selection.

### 2. Add optional Phase-3 row forensic bundle capture

Suggested flag:

- `--capture-phase3-row-bundle`

Suggested artifact:

- `step_000_phase3_row_bundle.npz`

Minimum contents:

- raw Phase-3 logit feature rows or bounded row slices if full rows are too large
- row-L1 total used by the influence solver
- row-L1 split by column family when available:
  - feature columns,
  - error columns,
  - token columns,
  - total row abs sum
- row hashes/stats per target
- seed influence hash/frontier hashes derived from these rows

This is the cleanest causal boundary for later replay: if donor rows replayed
into the host produce donor-like frontiers, the drift is before/in row
construction.

### 3. Project-side plumbing

Wire new capture flags through:

- `trace_pipeline_chunked.py`
- `experiments/run_sparsification_experiment.py`
- scenario config schema/generation helpers
- manifest/result metadata
- extraction/index summaries

Add CPU-only tests for:

- gradient bundle payload shape/schema,
- metadata propagation through manifests,
- row-bundle stats/hash extraction if implemented,
- comparator behavior on synthetic gradient/row bundles.

## Run phase

### A. Matched donor-capture rerun

Run matched `94_base` baseline captures on Ascend and Cardinal with:

- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
- `capture_phase0_donor_bundle=true`
- `capture_phase3_seed_bundle=true`
- `capture_phase3_gradient_bundle=true`
- `capture_phase3_row_bundle=true` if implemented in the first pass
- `max_steps=1`, `temperature=0.0`, `completions=1`

Purpose:

- create baseline gradient/row donors for each cluster,
- verify artifact sizes and schema before launching the full matrix.

### B. Repeat replay matrix with richer capture

Use the same four replay conditions as the completed Phase-0 matrix:

1. Ascend self-replay with Ascend Phase-0 donor.
2. Ascend host with Cardinal Phase-0 donor.
3. Cardinal self-replay with Cardinal Phase-0 donor.
4. Cardinal host with Ascend Phase-0 donor.

Keep `phase0_replay_mode=donor_phase0` and strict donor validation. Add
Phase-3 gradient/row capture to every task.

Primary readout:

- Do cross-swap Phase-3 gradients match host or donor?
- Do raw Phase-3 rows and row-L1 splits match host or donor?
- Do self-replay gradient/row bundles reproduce baseline exactly enough?

Expected result if current hypothesis is right:

- cross-swap support remains donor-like,
- Phase-3 gradients and rows remain host-like,
- influence/frontier/edge metrics remain host-like.

## Replay phase after capture

Prefer **Phase-3 row replay** as the first causal intervention after capture:

- `--phase3-row-bundle PATH`
- `--phase3-row-replay-mode donor_phase3_rows`

Reason: row replay bypasses gradient contraction and directly tests whether the
influence solver/frontier logic is deterministic once Phase-3 rows are fixed.

If row replay makes the frontier donor-like:

- drift is before or inside Phase-3 row construction, likely gradients/direct
  effects.

If row replay still stays host-like:

- investigate row mapping, normalization, influence solver, or ranking/frontier
  logic.

Only implement gradient replay after row replay if we need to distinguish
gradient tensors from decoder/activation contraction.

## Acceptance criteria

- Capture flags are opt-in and do not affect normal outputs when disabled.
- Completed capture runs produce declared gradient/row artifacts and manifest
  paths/statuses.
- Self-replay controls pass both existing graph gates and new gradient/row gates.
- Cross-swap analysis can state whether gradients/rows are host-like or
  donor-like using explicit similarity metrics.
- `EXPERIMENTS.md` records run provenance and interpretation.

## Guardrails

- Do not run GPU/model-loading jobs outside SLURM allocations.
- Keep run placement under `{cluster}/{fast|anomaly|long_eval}` only.
- Keep gradient/row artifacts bounded; avoid dense JSON dumps.
- Treat gradients as Phase-3 state, not Phase-0 donor state, in naming/docs.
- Record durable design changes in `docs/phase0_boundary_fingerprinting_spec.md`
  and run decisions/results in `EXPERIMENTS.md`.
