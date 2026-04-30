# Current Execution Plan — Phase-3 Enhanced Donor Replay Matrix

## Problem statement

The completed `94_base` Phase-0 replay matrix showed that replacing only the
Phase-0 active feature support/activation values copies donor support exactly,
but Phase-3 scores/frontiers and compact edge weights remain host-like.

The matched Phase-3 donor-capture pair has now completed successfully on both
Ascend and Cardinal. It produced Phase-0 donor bundles, Phase-3 seed bundles,
Phase-3 gradient bundles, and Phase-3 row bundles for `94_base`.

Next goal: implement **Phase-3 enhanced donor replay** and run an enlarged matrix
that separately tests donor gradients, donor rows, and both together.

## Working hypothesis

Cross-cluster drift is dominated by host-side Phase-3 scoring state:

- backward gradients through the host cached forward graph,
- direct-effect row construction from gradients and decoder/error/token vectors,
- row-L1 normalization and influence ranking from those rows.

Phase-0 donor replay currently creates a mixed counterfactual:

```text
donor feature support/activation values × host Phase-3 gradient/row field
```

Enhanced replay should let us test the causal boundary:

```text
donor Phase-0 state × host/donor gradient field × host/donor row field
```

## Status checklist

- [x] Phase-0 donor capture/replay implemented and validated.
- [x] `94_base` Phase-0 replay matrix completed on Ascend/Cardinal.
- [x] Self-replay controls passed on both clusters.
- [x] Cross-swaps stayed host-like in Phase-3/edge metrics despite donor support.
- [x] Passive Phase-3 gradient/direct-row capture implemented.
- [x] Matched `94_base` Phase-3 donor-capture pair completed.
- [x] Donor captures verified: gradient/row/seed/Phase-0 artifacts all present.
- [x] Implement Phase-3 row donor replay.
- [x] Implement Phase-3 gradient donor replay.
- [x] Add project plumbing and lightweight tests.
- [x] Reviewer follow-up complete: strict donor provenance validation, batched
  gradient slicing, row split consistency checks, and extractor provenance fixes.
- [ ] Generate and dry-run enlarged matrix scenarios.
- [ ] Launch enlarged matrix from a fresh immutable snapshot.
- [ ] Compare self controls first, then interpret cross-swaps.

## Implementation plan

### 1. Add explicit Phase-3 replay controls

Add independent controls for gradient and row replay:

```text
--phase3-gradient-replay-mode {disabled,donor}
--phase3-gradient-donor-bundle PATH

--phase3-row-replay-mode {disabled,donor}
--phase3-row-donor-bundle PATH

--phase3-replay-validation-policy strict
```

All enhanced-replay matrix scenarios should also keep:

```text
phase0_replay_mode=donor_phase0
phase0_donor_context_policy=strict
```

Reason: row replay is indexed by the active feature list, so it is only
meaningful after Phase-0 donor replay has imposed donor feature support/order.
Gradient replay is indexed by layer/target/position/d_model, but it is most
interpretable when paired with the donor Phase-0 active-feature state.

### 2. Define replay modes

For each host/donor pair, run four modes:

| Mode | Phase-0 state | Gradient field | Direct rows |
|---|---|---|---|
| `baseline` | donor | host-computed | host-computed |
| `row_donor` | donor | host-computed | donor row bundle |
| `gradient_donor` | donor | donor gradient bundle | recomputed from donor gradient |
| `gradient_row_donor` | donor | donor gradient bundle | donor row bundle override |

Apply order:

1. Phase-0 donor replay.
2. Optional Phase-3 gradient replay.
3. Host recomputes rows from the active state and current gradient field.
4. Optional Phase-3 row replay overrides recomputed rows.

If both gradient and row donor bundles are supplied, the donor row bundle is the
final row source. Manifest metadata must make that override explicit.

### 3. Library implementation touchpoints

In `../circuit-tracer_chunked`:

- Load and validate `step_000_phase3_gradient_bundle.npz`.
- Load and validate `step_000_phase3_row_bundle.npz`.
- Add replay state/status fields to attribution context/output payloads.
- Replace Phase-3 gradients before row construction when gradient replay is
  enabled.
- Replace final Phase-3 feature rows before influence/frontier ranking when row
  replay is enabled.
- Current gradient replay uses the captured per-layer feature/error gradients;
  token-gradient columns remain host-computed because the current gradient bundle
  schema does not store a separate token-embedding gradient. The row-donor modes
  remain the direct test for fully fixed Phase-3 row normalization.

Strict validation for gradient replay:

- target token ids/hash match runtime targets,
- target count exactly matches gradient columns,
- active feature count/hash and activation-value hash match the current
  post-Phase-0-replay runtime state,
- layer count compatible,
- prefix/position count compatible,
- `d_model` compatible,
- finite values only,
- expected dtype conversion is explicit and recorded.

Strict validation for row replay:

- target token ids/hash match runtime targets,
- active feature count matches current post-Phase-0-replay active features,
- active feature hash/order matches current active features,
- activation-value hash matches current active-feature values,
- row shape is `[target_count, active_feature_count]`,
- row abs sums are finite in fp64 representation,
- feature split sums match the stored feature rows,
- row source/override status is recorded.

### 4. Project implementation touchpoints

In this repo:

- `trace_pipeline_chunked.py`
  - add CLI args for Phase-3 gradient/row replay modes and donor paths,
  - pass settings into the circuit-tracer attribution call,
  - write manifest statuses/errors and per-step replay metadata.
- `experiments/run_sparsification_experiment.py`
  - pass new scenario JSON knobs through to the CLI.
- `experiments/exact_trace_bench/scenarios.py`
  - allow the new exact-mode knobs.
- `experiments/exact_trace_bench/extract.py`
  - index replay mode/status/path fields.
- `experiments/extract_benchmark_index.py`
  - mirror extractor fields for standalone indexing.
- Tests
  - add CPU-only synthetic tests for CLI plumbing, validation failures, and
    manifest status behavior.

## Test plan

### CPU-only tests before any SLURM launch

Do not load models or run GPU code on login nodes.

Add or update lightweight tests for:

1. Scenario/CLI plumbing:
   - dry-run includes `--phase3-gradient-replay-mode donor`,
   - dry-run includes `--phase3-gradient-donor-bundle ...`,
   - dry-run includes `--phase3-row-replay-mode donor`,
   - dry-run includes `--phase3-row-donor-bundle ...`.
2. Gradient validation:
   - valid synthetic bundle passes,
   - target token mismatch fails,
   - target count mismatch fails,
   - gradient shape mismatch fails,
   - nonfinite gradients fail strict validation.
3. Row validation:
   - valid synthetic bundle passes,
   - active feature count/hash mismatch fails,
   - target token mismatch fails,
   - row shape mismatch fails,
   - nonfinite rows or row sums fail strict validation.
4. Manifest/status behavior:
   - baseline: gradient and row replay disabled,
   - row donor: row replayed, gradient disabled,
   - gradient donor: gradient replayed, row host/recomputed,
   - both: gradient replayed, row replayed with donor-row override.

Safe validation commands should use `uv run`, for example:

```bash
uv run ruff check <changed project files>
uv run python tests/test_cross_cluster_debug_artifacts.py
```

For the sibling library, use its own uv environment and only lightweight tests.

## Enlarged matrix run configuration

### Donor artifact roots

Ascend donor capture root:

```text
/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260429_184917_527785_phase3-gradient-donor-capture-94-base-ascend/ascend_phase3_gradient_donor_capture_94_base_anomaly_b128_c2048_cache0g/artifacts/prompt_000/completion_000
```

Cardinal donor capture root:

```text
/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260429_184917_613970_phase3-gradient-donor-capture-94-base-cardinal/cardinal_phase3_gradient_donor_capture_94_base_anomaly_b128_c2048_cache0g/artifacts/prompt_000/completion_000
```

Each root provides:

```text
step_000_phase0_donor_bundle.npz
step_000_phase3_gradient_bundle.npz
step_000_phase3_row_bundle.npz
```

### Host/donor pairs

Run four host/donor pairs:

| Host | Donor | Purpose |
|---|---|---|
| Ascend | Ascend | Ascend self-control |
| Ascend | Cardinal | Cardinal donor on Ascend host |
| Cardinal | Cardinal | Cardinal self-control |
| Cardinal | Ascend | Ascend donor on Cardinal host |

For each pair, run the four replay modes listed above. Total: **16 scenarios**.

### Shared scenario defaults

Use:

```text
fixture_name=94_base
tier=anomaly
max_steps=1
completions=1
cross_cluster_debug=true
exact_trace_internal_dtype=fp64
phase0_activation_threshold_compare_mode=baseline
phase0_replay_mode=donor_phase0
phase0_donor_context_policy=strict
capture_phase0_donor_bundle=true
capture_phase3_seed_bundle=true
capture_phase3_gradient_bundle=true
capture_phase3_row_bundle=true
capture_feature_semantic_descriptors=true
semantic_descriptor_top_k=2048
semantic_descriptor_dim=64
attribution_batch_size=128
feature_batch_size=128
logit_batch_size=128
decoder_chunk_size=2048
cross_batch_decoder_cache_bytes=0
```

### Scenario naming convention

Use explicit names that include host, donor, and Phase-3 replay mode, for example:

```text
ascend_phase3_baseline_with_ascend_donor_94_base_anomaly_b128_c2048_cache0g
ascend_phase3_row_donor_with_cardinal_donor_94_base_anomaly_b128_c2048_cache0g
ascend_phase3_gradient_donor_with_cardinal_donor_94_base_anomaly_b128_c2048_cache0g
ascend_phase3_gradient_row_donor_with_cardinal_donor_94_base_anomaly_b128_c2048_cache0g
```

Mirror names for Cardinal host runs.

## Success gates before interpretation

Self controls must pass before interpreting cross-swaps:

- Ascend host + Ascend donor:
  - `baseline`, `row_donor`, `gradient_donor`, `gradient_row_donor`
- Cardinal host + Cardinal donor:
  - `baseline`, `row_donor`, `gradient_donor`, `gradient_row_donor`

Required gates:

- jobs complete with SLURM exit `0:0`,
- no strict validation warnings/errors,
- replay statuses match scenario mode,
- row donor self-run row hash matches donor row hash,
- gradient donor self-run gradient hash matches donor gradient hash,
- generated token remains `Let`,
- compact graph/frontier metrics are exact or any drift is explicitly explained.

Only after self gates pass should cross-swap movement be interpreted.

## Primary readouts

For each cross-swap, compare against both host baseline and donor baseline:

- compact feature Jaccard,
- weighted edge Jaccard,
- Phase-3 seed influence Pearson/Spearman,
- frontier pre/post Jaccard,
- Phase-3 gradient hash/stats similarity,
- Phase-3 row hash/stats similarity,
- row abs sums and feature/error/token splits.

Interpretation guide:

- row donor moves graph donor-like: drift is at/before row construction and the
  influence solver is mostly deterministic once rows are fixed.
- gradient donor moves graph donor-like but row donor also works: gradients are a
  sufficient upstream carrier of the row drift.
- gradient donor does not move graph but row donor does: row construction has
  additional host-side dependence beyond captured gradients.
- neither row nor gradient donor moves graph: investigate row mapping,
  normalization, influence ranking/frontier logic, or incomplete replay wiring.

## Guardrails

- Do not run GPU/model-loading code outside SLURM allocations.
- Keep run placement under `{cluster}/{fast|anomaly|long_eval}` only.
- Keep gradient/row artifacts in `.npz`; avoid dense JSON dumps.
- Keep Phase-3 replay explicitly named as Phase-3 state, not Phase-0 donor state.
- Treat the sibling `circuit-tracer_chunked` checkout as part of experiment
  provenance.
- Before serious launches, snapshot both project and sibling library together.
- Record durable design changes in `docs/phase0_boundary_fingerprinting_spec.md`
  and run decisions/results in `EXPERIMENTS.md`.
