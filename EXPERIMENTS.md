# Experiments inventory

This file is a living note for experiment artifacts currently stored on scratch
and a dated investigation log for important decisions, launches, findings, and
result reinterpretations.

For now it documents:

- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench`

The goal is to make it easy to answer:

- what experiment families exist,
- where they live,
- what each family was trying to test,
- and which runs should be treated as baseline / debug / anomaly / exploratory.

## Investigation log conventions

For important updates, include:

- date,
- project repo provenance,
- sibling library provenance,
- whether the run came from a live workspace or immutable snapshot,
- enough context to reconstruct later why we trusted or reinterpreted a
  baseline.

## Recent investigation updates

### 2026-04-29 — Launched Phase-3 enhanced replay matrix for `94_base`

Purpose:

- test whether donor Phase-3 gradients, donor Phase-3 row normalizers/feature
  rows, or both move the downstream compact graph/frontier toward the donor
  cluster after Phase-0 donor replay,
- run same-cluster self controls before interpreting cross-cluster swaps.

Implementation/launch provenance:

- project repo:
  - live workspace: `/users/PAS2119/andreykopanev/nlp_research_project`
  - branch: `exact-trace-bench-harness`
  - commit: `bea89a1d37549e414beb2febd661c8483c4cb854`
    (`Plumb Phase-3 donor replay matrix`)
  - generated matrix scenario JSONs were untracked at snapshot time.
- sibling library repo:
  - live workspace: `/users/PAS2119/andreykopanev/circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `cee7e4c52140994ebda9d970963a421c37506491`
    (`Replay Phase-3 donor gradients and rows`)
- immutable snapshot container:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260429_230804_phase3_replay_matrix_94_base`
- snapshot project root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260429_230804_phase3_replay_matrix_94_base/nlp_research_project`
- snapshot library root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260429_230804_phase3_replay_matrix_94_base/circuit-tracer_chunked`

Scenario files:

- Ascend:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260429_230804_phase3_replay_matrix_94_base/nlp_research_project/experiments/generated/exact_trace_bench/phase3_replay_matrix_94_base_ascend.json`
- Cardinal:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260429_230804_phase3_replay_matrix_94_base/nlp_research_project/experiments/generated/exact_trace_bench/phase3_replay_matrix_94_base_cardinal.json`
- each contains eight tasks:
  - tasks `0-3`: same-cluster donor self controls,
  - tasks `4-7`: opposite-cluster donor cross-swaps,
  - mode order per donor: `baseline`, `row_donor`, `gradient_donor`,
    `gradient_row_donor`.

Donor roots:

- Ascend donor:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260429_184917_527785_phase3-gradient-donor-capture-94-base-ascend/ascend_phase3_gradient_donor_capture_94_base_anomaly_b128_c2048_cache0g/artifacts/prompt_000/completion_000`
- Cardinal donor:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260429_184917_613970_phase3-gradient-donor-capture-94-base-cardinal/cardinal_phase3_gradient_donor_capture_94_base_anomaly_b128_c2048_cache0g/artifacts/prompt_000/completion_000`
- each mode uses the donor's `step_000_phase0_donor_bundle.npz`; donor-gradient
  modes also use `step_000_phase3_gradient_bundle.npz`; donor-row modes also use
  `step_000_phase3_row_bundle.npz`.

Submitted jobs and intended outputs:

- Ascend:
  - SLURM job `5147972` (`5147972_[0-7]` array)
  - run id `phase3_replay_matrix_94_base_ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/phase3_replay_matrix_94_base_ascend`
- Cardinal:
  - SLURM job `8978498` (`8978498_[0-7]` array)
  - run id `phase3_replay_matrix_94_base_cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/phase3_replay_matrix_94_base_cardinal`

Shared replay config:

- fixture: `94_base`
- tier: `anomaly`
- `max_steps=1`, `temperature=0.0`, `completions=1`
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
- `phase0_activation_threshold_compare_mode=baseline`
- `phase0_replay_mode=donor_phase0`
- `phase0_donor_context_policy=strict`
- `phase3_replay_validation_policy=strict`
- `capture_phase0_donor_bundle=true`
- `capture_phase3_seed_bundle=true`
- `capture_phase3_gradient_bundle=true`
- `capture_phase3_row_bundle=true`
- `capture_feature_semantic_descriptors=true`
- `semantic_descriptor_top_k=2048`, `semantic_descriptor_dim=64`
- `decoder_chunk_size=2048`, `cross_batch_decoder_cache_bytes=0`
- batch sizes `128`
- requested walltime: `01:00:00`

Analysis gates after completion:

- first check all same-cluster self controls complete with strict replay status
  and no validation failures,
- then compare cross-swaps against both host and donor baselines,
- specifically interpret `row_donor`, `gradient_donor`, and
  `gradient_row_donor` movement separately.

### 2026-04-27 — Launched Phase-0 donor-bundle capture pair for replay matrix

Purpose:

- capture matched `94_base` Phase-0 donor bundles on Ascend and Cardinal as the
  baseline inputs for the self-replay and cross-swap causality checks,
- keep the same single-step anomaly/debug configuration while adding
  `step_000_phase0_donor_bundle.npz` capture.

Implementation/launch provenance:

- project repo:
  - live workspace: `/users/PAS2119/andreykopanev/nlp_research_project`
  - branch: `exact-trace-bench-harness`
  - commit: `16ca5a2f6bce7218ac9ee0724c212c7d845b89c9`
    (`Plumb Phase-0 replay run metadata`)
  - untracked launch/local files were present in the live workspace, including
    presentation/report drafts and the generated donor-capture scenario JSONs.
- sibling library repo:
  - live workspace: `/users/PAS2119/andreykopanev/circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `a6946fa7875d6e510845b38e44d66db88098a8f1`
    (`Add Phase-0 donor replay plumbing`)
- immutable snapshot container:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260427_143848_phase0_donor_capture_94_base`
- snapshot project root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260427_143848_phase0_donor_capture_94_base/nlp_research_project`
- snapshot library root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260427_143848_phase0_donor_capture_94_base/circuit-tracer_chunked`

Submitted jobs and intended outputs:

- Ascend:
  - SLURM job `5098340` (`5098340_0` array task)
  - run id `20260427_143849_028151_phase0-donor-capture-94-base-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260427_143849_028151_phase0-donor-capture-94-base-ascend`
  - completed `2026-04-27T15:19:17`, state `COMPLETED`, exit `0:0`,
    elapsed `00:29:30`
- Cardinal:
  - SLURM job `8887574` (`8887574_0` array task)
  - run id `20260427_143849_120767_phase0-donor-capture-94-base-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260427_143849_120767_phase0-donor-capture-94-base-cardinal`
  - completed `2026-04-27T15:14:07`, state `COMPLETED`, exit `0:0`,
    elapsed `00:16:38`

Shared config:

- fixture: `94_base`
- tier: `anomaly`
- `max_steps=1`, `temperature=0.0`, `completions=1`
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
- `phase0_activation_threshold_compare_mode=baseline`
- `capture_phase0_donor_bundle=true`
- `capture_phase3_seed_bundle=true`
- `capture_feature_semantic_descriptors=true`
- `semantic_descriptor_top_k=2048`, `semantic_descriptor_dim=64`
- `decoder_chunk_size=2048`, `cross_batch_decoder_cache_bytes=0`
- batch sizes `128`

Artifact status:

- both jobs succeeded (`status=success`, `returncode=0`),
- both generated token id `6481`, text `"Let"`,
- both captured:
  - `cross_cluster_debug_summary.json`,
  - `step_000.npz`,
  - `step_000_phase0_donor_bundle.npz`,
  - `step_000_phase3_seed_bundle.npz`,
  - `step_000_feature_semantic_descriptors.npz`.
- donor bundle status is `captured` on both sides.
- active feature counts:
  - Ascend: `3371343`,
  - Cardinal: `3370036`.
- donor bundle artifact sizes:
  - Ascend: `18407266` bytes,
  - Cardinal: `18399886` bytes.
- Phase-3 seed bundle and semantic descriptor artifacts also saved successfully
  on both sides.

Follow-up:

- superseded by the replay-matrix launch below: same-cluster self-replay and
  cross-cluster donor-swap tasks were launched together to reduce queue wait,
  with cross-swap interpretation gated on self-replay passing.

### 2026-04-27 — Launched Phase-0 self/cross replay matrix for `94_base`

Purpose:

- test whether replacing the host Phase-0 activation state with a captured donor
  bundle is sufficient to move the downstream Phase-3 seed set and compact graph
  toward the donor cluster,
- run same-cluster self-replay controls in parallel with cross-cluster donor
  swaps,
- treat cross-swap interpretation as provisional until self-replay controls pass.

Implementation/launch provenance:

- project repo:
  - live workspace: `/users/PAS2119/andreykopanev/nlp_research_project`
  - branch: `exact-trace-bench-harness`
  - commit: `16ca5a2f6bce7218ac9ee0724c212c7d845b89c9`
    (`Plumb Phase-0 replay run metadata`)
- sibling library repo:
  - live workspace: `/users/PAS2119/andreykopanev/circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `a6946fa7875d6e510845b38e44d66db88098a8f1`
    (`Add Phase-0 donor replay plumbing`)
- immutable snapshot container:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260427_164501_phase0_replay_matrix_94_base`
- snapshot project root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260427_164501_phase0_replay_matrix_94_base/nlp_research_project`
- snapshot library root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260427_164501_phase0_replay_matrix_94_base/circuit-tracer_chunked`

Scenario files:

- Ascend:
  `/users/PAS2119/andreykopanev/nlp_research_project/experiments/generated/exact_trace_bench/phase0_replay_matrix_94_base_ascend.json`
- Cardinal:
  `/users/PAS2119/andreykopanev/nlp_research_project/experiments/generated/exact_trace_bench/phase0_replay_matrix_94_base_cardinal.json`
- each contains two tasks:
  - task `0`: same-cluster self-replay using that cluster's donor bundle,
  - task `1`: host cluster with the other cluster's donor bundle.

Donor bundles used:

- Ascend donor:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260427_143849_028151_phase0-donor-capture-94-base-ascend/ascend_phase0_donor_capture_94_base_anomaly_b128_c2048_cache0g/artifacts/prompt_000/completion_000/step_000_phase0_donor_bundle.npz`
- Cardinal donor:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260427_143849_120767_phase0-donor-capture-94-base-cardinal/cardinal_phase0_donor_capture_94_base_anomaly_b128_c2048_cache0g/artifacts/prompt_000/completion_000/step_000_phase0_donor_bundle.npz`

Submitted jobs and intended outputs:

- Ascend:
  - SLURM job `5101354` (`5101354_[0-1]` array)
  - run id `20260427_164504_282386_phase0-replay-matrix-94-base-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260427_164504_282386_phase0-replay-matrix-94-base-ascend`
  - queue check after submission: `PENDING` on `nextgen` with reason
    `Priority`
- Cardinal:
  - SLURM job `8888658` (`8888658_[0-1]` array)
  - run id `20260427_164504_455131_phase0-replay-matrix-94-base-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260427_164504_455131_phase0-replay-matrix-94-base-cardinal`
  - queue check after submission: `PENDING` on `gpu` with reason `Priority`

Shared replay config:

- fixture: `94_base`
- tier: `anomaly`
- `max_steps=1`, `temperature=0.0`, `completions=1`
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
- `phase0_activation_threshold_compare_mode=baseline`
- `phase0_replay_mode=donor_phase0`
- `phase0_donor_context_policy=strict`
- `capture_phase0_donor_bundle=true`
- `capture_phase3_seed_bundle=true`
- `capture_feature_semantic_descriptors=true`
- `semantic_descriptor_top_k=2048`, `semantic_descriptor_dim=64`
- `decoder_chunk_size=2048`, `cross_batch_decoder_cache_bytes=0`
- batch sizes `128`
- requested walltime: `01:00:00`

Analysis gates after completion:

- first verify self-replay controls before interpreting cross-swaps,
- expected self-replay pass thresholds:
  - feature support Jaccard `1.0`,
  - Phase-3 seed influence Pearson `>= 0.9999`,
  - Phase-3 top-1024 overlap `>= 0.999`,
  - compact weighted edge Jaccard `>= 0.999`,
- then run `compare-phase0-replay-matrix` over the donor-capture baselines,
  self-replay outputs, and cross-swap outputs.

Completion / analysis update:

- job states checked after launch:
  - Ascend task `5101354_0` completed, exit `0:0`, elapsed `00:28:57`;
    this is `ascend_phase0_self_replay_94_base_anomaly_b128_c2048_cache0g`.
  - Ascend task `5101354_1` completed, exit `0:0`, elapsed `00:31:15`;
    this is `ascend_phase0_with_cardinal_donor_94_base...`. It had previously
    remained `PENDING` with reason `Priority` and no dependency.
  - Cardinal tasks `8888658_0` and `8888658_1` both completed, exit `0:0`,
    elapsed `00:21:00` and `00:20:31` respectively.
- completed replay artifacts all captured `step_000.npz`,
  `step_000_phase0_donor_bundle.npz`, `step_000_phase3_seed_bundle.npz`, and
  `step_000_feature_semantic_descriptors.npz`.
- replay metadata for all completed replay runs reported
  `phase0_replay_mode=donor_phase0`, `phase0_replay_status=applied`, strict
  donor context, and no dtype roundtrip loss.
- self-replay gates passed exactly enough to trust cross-swap diagnostics:
  - Ascend self-replay vs Ascend baseline:
    - compact feature Jaccard `1.0`, weighted edge Jaccard `1.0`,
    - Phase-3 support Jaccard `1.0`, seed influence Pearson
      `0.999999999999988`, top-1024 overlap `1.0`, frontier post Jaccard `1.0`.
  - Cardinal self-replay vs Cardinal baseline:
    - compact feature Jaccard `1.0`, weighted edge Jaccard `1.0`,
    - Phase-3 support Jaccard `1.0`, seed influence Pearson
      `0.9999999999999999`, top-1024 overlap `1.0`, frontier post Jaccard `1.0`.
- full comparison written to:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/analysis/phase0_replay_matrix_94_base_compare_no_semantic.json`
  (`include_semantic=false`; semantic descriptors in this run are still the
  fallback identity-metadata descriptors, so the main interpretation uses compact
  graph and Phase-3 seed-bundle metrics).
- baseline and cross-swap similarities:
  - baseline Ascend-vs-Cardinal similarity remains the earlier reference:
    compact feature Jaccard / Phase-3 support Jaccard `0.9936243685806376`,
    weighted edge Jaccard `0.6128084788664958`, Phase-3 seed influence Pearson
    `0.9843850642912175`, top-1024 overlap `0.828125`, frontier post Jaccard
    `0.7009966777408638`.
  - `Ascend host + Cardinal donor` vs Ascend host baseline:
    compact feature Jaccard / Phase-3 support Jaccard `0.9936243685806376`,
    weighted edge Jaccard `0.9995229242169169`, Phase-3 seed influence Pearson
    `0.9999974420907568`, top-1024 overlap `1.0`, frontier post Jaccard `1.0`.
  - `Ascend host + Cardinal donor` vs Cardinal donor baseline:
    compact feature Jaccard / Phase-3 support Jaccard `1.0`, weighted edge
    Jaccard `0.6127990686429539`, Phase-3 seed influence Pearson
    `0.984378196700129`, top-1024 overlap `0.828125`, frontier post Jaccard
    `0.7009966777408638`.
  - `Cardinal host + Ascend donor` vs Cardinal host baseline:
    compact feature Jaccard / Phase-3 support Jaccard `0.9936243685806376`,
    weighted edge Jaccard `0.9991256837503769`, Phase-3 seed influence Pearson
    `0.9999971844045109`, top-1024 overlap `0.9990234375`, frontier post
    Jaccard `1.0`.
  - `Cardinal host + Ascend donor` vs Ascend donor baseline:
    compact feature Jaccard / Phase-3 support Jaccard `1.0`, weighted edge
    Jaccard `0.612793730082947`, Phase-3 seed influence Pearson
    `0.9843768836438769`, top-1024 overlap `0.8271484375`, frontier post
    Jaccard `0.7009966777408638`.
- final replay-matrix interpretation:
  - both cross-swaps copy the donor Phase-0 support set exactly (`donor`
    feature/support Jaccard `1.0`),
  - but downstream edge weights, Phase-3 influence scores, top-k ranking, and
    frontier locality remain overwhelmingly host-like,
  - therefore the dominant cross-cluster drift is not explained by Phase-0 active
    feature support identity. The current suspect is host-cluster-dependent
    processing after Phase-0, especially Phase-3 scoring/ranking/frontier
    construction and/or later edge-weight construction from those host-side
    quantities.

### 2026-04-29 — Chosen next diagnostic: Phase-3 gradient / row capture

Decision:

- Do not expand the Phase-0 donor bundle to include gradients. The Phase-0 replay
  bundle is intentionally limited to active feature support/activation state.
- Treat gradients and direct-effect rows as Phase-3 state, because they are
  produced during `phase3_logits` backward attribution from the host forward
  graph.
- Add a new passive capture layer for Phase-3 gradient and direct-row bundles,
  then rerun the `94_base` self/cross matrix with richer capture.

Rationale:

- The completed Phase-0 replay matrix created a mixed counterfactual:
  `donor feature support/activation values × host Phase-3 gradient field`.
- Both directions stayed host-like in Phase-3 influence/frontier/edge metrics, so
  the next hypothesis is that host-side gradients/direct-effect row construction
  dominate downstream drift.
- Existing debug summaries already show host-like Phase-3 row-L1 scale under
  cross-swap, reinforcing the need to capture the gradient/row boundary directly.

Planned implementation/run sequence:

1. Implement opt-in `step_000_phase3_gradient_bundle.npz` capture.
2. Implement opt-in `step_000_phase3_row_bundle.npz` capture if feasible in the
   same pass; otherwise start with gradient bundle plus row-family scalar stats.
3. Wire flags through exact tracing, scenario configs, manifests, extraction, and
   CPU-only tests.
4. Launch matched Ascend/Cardinal `94_base` donor-capture baselines with Phase-3
   gradient/row capture enabled.
5. Launch the four-condition replay matrix again with strict Phase-0 donor replay
   plus Phase-3 gradient/row capture.
6. Compare whether cross-swap gradients/rows are host-like or donor-like.
7. Prefer Phase-3 row replay as the next causal intervention if passive capture
   confirms the drift is already present in direct-effect rows.

Current working plan:

- root `PLAN.md` now tracks the Phase-3 gradient boundary probe.
- durable design update is in
  `docs/phase0_boundary_fingerprinting_spec.md` under
  “Phase-3 gradient / row boundary follow-up”.

### 2026-04-29 — Launched Phase-3 gradient/row donor-capture pair for `94_base`

Purpose:

- produce matched Ascend/Cardinal baseline donor artifacts with the new Phase-3
  gradient and direct-row bundle capture enabled,
- verify artifact schema/size/runtime before launching the richer self/cross
  replay matrix.

Implementation/launch provenance:

- project repo:
  - live workspace: `/users/PAS2119/andreykopanev/nlp_research_project`
  - branch: `exact-trace-bench-harness`
  - commit: `3eb5103` (`Wire Phase-3 gradient row capture`)
- sibling library repo:
  - live workspace: `/users/PAS2119/andreykopanev/circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `050c877` (`Capture Phase-3 gradients and rows`)
- immutable snapshot container:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260429_184917_phase3_gradient_capture_94_base`
- snapshot project root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260429_184917_phase3_gradient_capture_94_base/nlp_research_project`
- snapshot library root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260429_184917_phase3_gradient_capture_94_base/circuit-tracer_chunked`

Scenario files:

- Ascend:
  `/users/PAS2119/andreykopanev/nlp_research_project/experiments/generated/exact_trace_bench/phase3_gradient_donor_capture_94_base_ascend.json`
- Cardinal:
  `/users/PAS2119/andreykopanev/nlp_research_project/experiments/generated/exact_trace_bench/phase3_gradient_donor_capture_94_base_cardinal.json`

Submitted jobs and intended outputs:

- Ascend:
  - SLURM job `5146702` (`5146702_[0]` array)
  - run id `20260429_184917_527785_phase3-gradient-donor-capture-94-base-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260429_184917_527785_phase3-gradient-donor-capture-94-base-ascend`
  - SLURM state: `COMPLETED`, exit code `0:0`, elapsed `00:32:34`
    (`2026-04-29T20:44:30`–`2026-04-29T21:17:04`)
- Cardinal:
  - SLURM job `8975505` (`8975505_[0]` array)
  - run id `20260429_184917_613970_phase3-gradient-donor-capture-94-base-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260429_184917_613970_phase3-gradient-donor-capture-94-base-cardinal`
  - SLURM state: `COMPLETED`, exit code `0:0`, elapsed `00:18:40`
    (`2026-04-29T20:16:37`–`2026-04-29T20:35:17`)

Shared config:

- fixture: `94_base`
- tier: `anomaly`
- `max_steps=1`, `temperature=0.0`, `completions=1`
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
- `phase0_activation_threshold_compare_mode=baseline`
- `capture_phase0_donor_bundle=true`
- `capture_phase3_seed_bundle=true`
- `capture_phase3_gradient_bundle=true`
- `capture_phase3_row_bundle=true`
- `capture_feature_semantic_descriptors=true`
- `semantic_descriptor_top_k=2048`, `semantic_descriptor_dim=64`
- `decoder_chunk_size=2048`, `cross_batch_decoder_cache_bytes=0`
- batch sizes `128`
- requested walltime: `01:00:00`

Completion/artifact check:

- both `result.json` files report `status=success`, `returncode=0`;
  `completion.json` reports first generated token `Let` and `n_steps_traced=1`.
- both artifact directories contain:
  - `completion.json`
  - `cross_cluster_debug_summary.json`
  - `cross_cluster_debug_batches.jsonl`
  - `cross_cluster_debug_checkpoints.jsonl`
  - `telemetry.jsonl`
  - `step_000.npz`
  - `step_000_phase0_donor_bundle.npz`
  - `step_000_phase3_seed_bundle.npz`
  - `step_000_phase3_gradient_bundle.npz`
  - `step_000_phase3_row_bundle.npz`
  - `step_000_feature_semantic_descriptors.npz`
- manifest statuses are `captured` for Phase-0 donor, Phase-3 seed,
  Phase-3 gradient, Phase-3 row, and feature semantic descriptor bundles.
- active feature counts:
  - Ascend: `3,371,343`
  - Cardinal: `3,370,036`
- common fixed-input/target checks:
  - input token hash `d081924d7fcce7ec`
  - target token id `[6481]` (`Let`)
  - target-token hash `1dea56c43da95d32`
  - target-probability hash `437d56a13df41ec1`
  - CLT constants hash `31c83df182f3f365`
- divergent captured state hashes:
  - Phase-0 canonical membership: Ascend `2042e3cf71dc07b7`, Cardinal
    `ba1ab906c0c08e5e`
  - Phase-0 values: Ascend `456ae070fe38a548`, Cardinal
    `13a754ff3967d1b4`
  - Phase-3 gradient hash: Ascend `9439db929b7bd065`, Cardinal
    `201828f60d3cf292`
  - Phase-3 row hash: Ascend `fed34f11c8491988`, Cardinal
    `74cebf03c3fa4ed5`
- notable row-abs sums:
  - Ascend row abs sum `3.306366250278821e38` with feature split
    `2.9732265315158493e38`
  - Cardinal row abs sum `3.707134498858746e38` with feature split
    `3.332438773222376e38`
  - Cardinal exceeds float32 max (`~3.4028235e38`), reinforcing that row-sum
    overflow remains a plausible boundary for permanent normalization work even
    though this diagnostic capture itself completed.
- artifact sizes were bounded:
  - Phase-3 gradient bundle: about `5.44 MiB` each, with stored dense gradient
    member shape `(26, 128, 80, 1152)` compressed from about `1170 MiB`.
  - Phase-3 row bundle: about `11.36 MiB` each, with row shape
    `(1, active_feature_count)`.
  - Phase-0 donor bundle: about `17.55 MiB` each.
  - Phase-3 seed bundle: about `31.16 MiB` each.
- log scan found only expected/benign matches such as `error` in phase labels and
  the cuBLAS first-context `UserWarning`; no failed manifest status, traceback,
  OOM, or nonzero SLURM exit was observed.

Next action:

- build and dry-run the richer self/cross replay matrix using these new donor
  captures, then launch from a fresh immutable snapshot if dry-run checks pass.

### 2026-04-24 — Phase-3 seed-bundle rerun supports normal upstream numerical drift interpretation

Purpose:

- close the missing Phase-3 evidence gap from the first semantic-capture rerun,
  where both jobs completed but `step_000_phase3_seed_bundle.npz` saving failed,
- determine whether the `94_base` cross-cluster graph differences are carried by
  cluster-unique Phase-0 features or mostly by ranking/frontier churn within a
  shared high-mass feature universe.

Implementation/launch provenance:

- project repo:
  - workspace used for launch prep:
    `/users/PAS2119/andreykopanev/worktrees_phase3_seed_fix/nlp_research_project`
  - branch/source before clean worktree: `exact-trace-bench-harness`
  - commit: `e6636edafbfc62848981c5b77b25646319e92e06`
    (`Fix diagnostic artifact serialization`)
  - relevant fix: convert NumPy-unsupported torch float dtypes such as
    `bfloat16` activation values to `float32` before saving Phase-3 seed-bundle
    `.npz` artifacts; record `phase3_seed_bundle_error` on future save failures.
- sibling library repo:
  - workspace used for launch prep:
    `/users/PAS2119/andreykopanev/worktrees_phase3_seed_fix/circuit-tracer_chunked`
  - branch/source before clean worktree: `exact-trace-hidden-knobs`
  - commit: `fc22162ba75aca736ae1a088a83e64d0fac20f93`
    (`Capture semantic feature descriptors`)
- immutable snapshot container:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260423_200451_matched_cross_cluster_94_phase3_seed_fix`
- snapshot project root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260423_200451_matched_cross_cluster_94_phase3_seed_fix/nlp_research_project`
- snapshot library root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260423_200451_matched_cross_cluster_94_phase3_seed_fix/circuit-tracer_chunked`

Submitted jobs and outputs:

- Ascend:
  - SLURM job `5066410`
  - run id `20260423_200451_matched-cross-cluster-94-phase3-seed-fix-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260423_200451_matched-cross-cluster-94-phase3-seed-fix-ascend`
  - completed `2026-04-23T20:32:26`, state `COMPLETED`, exit `0:0`
- Cardinal:
  - SLURM job `8757353`
  - run id `20260423_200451_matched-cross-cluster-94-phase3-seed-fix-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260423_200451_matched-cross-cluster-94-phase3-seed-fix-cardinal`
  - completed `2026-04-24T05:28:24`, state `COMPLETED`, exit `0:0`

Shared config:

- fixture: `94_base`
- tier: `anomaly`
- `max_steps=1`, `temperature=0.0`, `completions=1`
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
- `phase0_activation_threshold_compare_mode=baseline`
- `capture_phase3_seed_bundle=true`
- `capture_feature_semantic_descriptors=true`
- `semantic_descriptor_top_k=2048`, `semantic_descriptor_dim=64`
- `decoder_chunk_size=2048`, `cross_batch_decoder_cache_bytes=0`
- batch sizes `128`

Artifact status:

- both jobs succeeded (`status=success`, `returncode=0`),
- both generated token id `6481`, text `"Let"`,
- both captured:
  - `cross_cluster_debug_summary.json`,
  - `cross_cluster_debug_checkpoints.jsonl`,
  - `cross_cluster_debug_batches.jsonl`,
  - `step_000.npz`,
  - `step_000_phase3_seed_bundle.npz`,
  - `step_000_feature_semantic_descriptors.npz`.
- Phase-3 seed bundle status is now `captured` on both sides.

Analysis outputs written:

- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260423_200451_matched-cross-cluster-94-phase3-seed-fix-ascend/compact_compare_vs_cardinal.json`
- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260423_200451_matched-cross-cluster-94-phase3-seed-fix-ascend/phase3_seed_compare_vs_cardinal.json`
- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260423_200451_matched-cross-cluster-94-phase3-seed-fix-ascend/semantic_compare_vs_cardinal.json`

Key debug hashes:

- CLT/transcoder constants match:
  - `transcoder_constants_global_hash = 31c83df182f3f365`
- pre-CLT input hashes differ:
  - Ascend `b2e3040ca6df7a43`
  - Cardinal `3da7761e872c6f45`
- Phase-0 active membership hashes differ:
  - Ascend `2042e3cf71dc07b7`
  - Cardinal `ba1ab906c0c08e5e`
- Phase-1 target logit state matches:
  - `437d56a13df41ec1`
- Phase-3 seed ranking hashes differ, as expected after the upstream activation
  split:
  - Ascend feature influence hash `c39f2e379488507f`
  - Cardinal feature influence hash `de8bf95e9ef1701c`

Compact graph comparison:

- active feature support:
  - Ascend active features: `3,371,343`
  - Cardinal active features: `3,370,036`
  - shared features: `3,359,910`
  - feature Jaccard: `0.9936243685806376`
  - Ascend-only: `11,433`
  - Cardinal-only: `10,126`
- retained feature-edge comparison:
  - edge Jaccard: `0.5709439233685891`
  - weighted edge Jaccard: `0.6128084788664958`
  - common edges: `8,583 / 11,808`
  - common-edge weight Pearson: `0.9834548009895984`
  - top-edge overlap:
    - top-64: `0.8125`
    - top-128: `0.84375`
    - top-1024: `0.818359375`
- edge-class decomposition:
  - retained feature-to-feature mass is effectively all `shared_to_shared`,
  - no retained edge mass is carried by unique-feature endpoints.

Phase-3 seed-bundle comparison:

- support is the same as compact active-feature support:
  - feature Jaccard `0.9936243685806376`,
  - shared features `3,359,910`.
- Phase-3 absolute influence mass is overwhelmingly on shared support:
  - Ascend total abs influence mass: `0.8957302851643294`
  - Ascend shared mass fraction: `0.9999483349442382`
  - Ascend unique mass fraction: `0.000051665055761584634`
  - Cardinal total abs influence mass: `0.895414351781944`
  - Cardinal shared mass fraction: `0.9999551620772623`
  - Cardinal unique mass fraction: `0.00004483792273787384`
- shared-support score stability:
  - seed influence Pearson: `0.9843850642912244`
  - seed influence Spearman: `0.9986320355329672`
  - seed influence sign agreement: `1.0`
  - activation Pearson: `0.9998822818760333`
  - activation Spearman: `0.9993906288285449`
  - activation sign agreement: `1.0`
- Phase-3 seed top-k overlap:
  - top-64: `0.734375`
  - top-128: `0.765625`
  - top-256: `0.7890625`
  - top-512: `0.82421875`
  - top-1024: `0.828125`
- Phase-3 frontier overlap:
  - pre-locality Jaccard: `0.7009966777408638`
  - post-locality Jaccard: `0.7009966777408638`
  - post-locality shared frontier rank drift:
    - median abs rank delta: `8`
    - q90 abs rank delta: `17`
    - max abs rank delta: `22`
- comparator interpretation:
  - `phase3_mismatch_persists_on_shared_support`
  - important nuance: this means exact frontier membership/ranking still churns
    on shared support; it does **not** mean unique Phase-0 support carries much
    influence mass. Unique-feature influence mass is only about `0.005%` per side.

Semantic descriptor comparison:

- descriptor kind is still the bounded fallback descriptor:
  - `fallback_identity_metadata_v1`
  - not a true decoder-vector semantic descriptor.
- top-2048 candidate support:
  - shared candidates: `1748 / 2048`
  - feature Jaccard: `0.7444633730834753`
  - left-only: `300`
  - right-only: `300`
- shared candidate scores:
  - seed influence Pearson: `0.9295630779229779`
  - activation Pearson: `0.9999988975796839`
- unmatched top-2048 candidate mass:
  - Ascend-only: `0.012183350530644847`
  - Cardinal-only: `0.011181063223906407`
- comparator interpretation remains:
  - `unique_features_semantically_unmatched`
  - caveat: because descriptors are fallback identity metadata, do not overread
    this as semantic evidence.

Current interpretation:

- This rerun substantially strengthens the conclusion that the `94_base`
  Ascend/Cardinal difference is ordinary upstream numerical/model-forward drift,
  not mismatched model/CLT assets.
- The drift appears before CLT encode, then perturbs Phase-0 active membership
  and downstream Phase-3/frontier/edge exact IDs.
- The important circuit structure is nevertheless largely stable for this prompt:
  - same generated token,
  - matched target-logit state,
  - nearly identical active feature universe,
  - >`99.994%` of Phase-3 influence mass on shared features,
  - highly correlated shared Phase-3 influence scores,
  - stable common-edge weights,
  - strong top-edge overlap.
- Treat exact top-k/frontier/edge membership near thresholds as hardware-sensitive,
  but the extracted graph remains credible at the high-importance circuit level
  for this single diagnostic prompt.

Remaining evidence gap:

- This is still one deep diagnostic prompt (`94_base`). To claim this generally
  across the project, rerun a larger matched sample with Phase-3 seed bundles and
  compare distributions of same-token rate, feature Jaccard, shared influence
  mass fraction, seed top-k overlap, compact graph weighted Jaccard, top-edge
  overlap, and common-edge weight Pearson.

### 2026-04-23 — implemented full passive downstream/semantic stability capture before rerun

Purpose:

- prepare the next expensive matched `94_base` rerun to answer whether upstream
  numerical drift materially changes the extracted circuit or mostly causes
  exact-ID/score churn around a semantically similar graph.

Implementation summary:

- Added enriched Phase-3 seed-bundle comparison for shared-support score/rank
  stability, top-k overlap, unique-support mass, and frontier rank drift.
- Added compact graph decomposition by shared/unique feature endpoints and
  shared-endpoint edge weight stability.
- Added opt-in feature semantic descriptor artifact capture:
  - `step_<idx>_feature_semantic_descriptors.npz`,
  - bounded top seed/frontier candidate set,
  - Phase-4 selection membership when available,
  - current descriptor kind: `fallback_identity_metadata_v1`.
- Added semantic descriptor comparison for high-mass unmatched features and
  mass-weighted semantic substitute coverage.
- Plumbed scenario/extractor support so future run indexes report descriptor
  artifact presence, status, and paths.

Provenance after implementation:

- project repo:
  - workspace: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - relevant commits through the scenario/extractor/docs plumbing commit that
    introduced this entry; see git log after implementation
- sibling library repo:
  - workspace: `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `fc22162` (`Capture semantic feature descriptors`)

Next launch requirement:

- new matched Ascend/Cardinal `94_base` anomaly baseline pair should enable both:
  - `capture_phase3_seed_bundle=true`,
  - `capture_feature_semantic_descriptors=true`.
- This is still passive capture, not a donor/replay intervention.

### 2026-04-23 — chose stronger Phase-0 → Phase-3 causality experiment design

Purpose:

- record the follow-up design decision after discussing how to link an early
  Phase-0 mismatch to the later Phase-3 divergence more confidently,
- clarify that the currently queued boundary-fingerprint pair is useful but not
  sufficient for a strong causal claim by itself.

Decision summary:

- Treat the current queued boundary-fingerprint pair as an **earliest-divergence
  localization** run only.
- The next stronger matched rerun should add a passive **Phase-3 seed bundle**
  artifact so we can test offline whether Phase-3 disagreement is mostly carried
  by Phase-0-unique features.
- Defer donor/swap/replay intervention as an **extra** follow-up rather than the
  first stronger implementation step.

Chosen stronger-evidence direction:

- keep the Phase-0 boundary fingerprints,
- add per-step saved Phase-3 seed bundle state,
- add offline comparison for:
  - shared vs unique Phase-0 feature counts,
  - Phase-3 influence mass on shared vs unique support,
  - frontier overlap before/after restricting to shared support.

Interpretation target:

- if Phase-3 disagreement mostly disappears once Phase-0-unique support is
  removed, treat that as strong evidence that the later split is downstream
  amplification of earlier drift,
- if substantial Phase-3 disagreement remains even on shared support, treat that
  as evidence that Phase-3 likely contributes additional instability and reserve
  replay/intervention for the next escalation step.

### 2026-04-23 — launched matched `94_base` boundary-fingerprinting baseline pair

Purpose:

- run the next matched single-step `94_base` pair after the compare-upcast
  hypothesis closed negative,
- use the new upstream Phase-0 boundary fingerprints to localize whether the
  first cross-cluster divergence appears:
  - before CLT encode (`mlp_in_cache`),
  - in preactivation / margin,
  - or only at mask / post-mask.

Launch provenance:

- project repo source commit:
  - repo: `nlp_research_project`
  - workspace used for launch prep: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - commit: `f86c7e1c7f24c03c822a2682202cd9b5c44db4ad`
- sibling library source commit:
  - repo: `circuit-tracer_chunked`
  - workspace used for launch prep: main sibling workspace `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `b06d35e73d8ec782acd6f6911789ee424b44eb6a`
- clean paired worktrees used before snapshot:
  - `/users/PAS2119/andreykopanev/worktrees_probe_94_boundary/nlp_research_project`
  - `/users/PAS2119/andreykopanev/worktrees_probe_94_boundary/circuit-tracer_chunked`
- immutable snapshot container:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260423_124304_matched_cross_cluster_94_boundary_probe`
- snapshot project root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260423_124304_matched_cross_cluster_94_boundary_probe/nlp_research_project`
- snapshot library root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260423_124304_matched_cross_cluster_94_boundary_probe/circuit-tracer_chunked`

Shared config across both jobs:

- fixture: `94_base`
- tier: `anomaly`
- `max_steps=1`
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
- `phase0_activation_threshold_compare_mode=baseline`
- `decoder_chunk_size=2048`
- `cross_batch_decoder_cache_bytes=0`
- batch sizes `128`
- `temperature=0.0`
- `completions=1`
- walltime override: `01:00:00`

Scenario files used:

- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_baseline_ascend.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_baseline_cardinal.json`

Submitted jobs:

- Ascend boundary probe:
  - SLURM job `5059551`
  - run id `20260423_124423_578647_matched-cross-cluster-94-anomaly-boundary-probe-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260423_124423_578647_matched-cross-cluster-94-anomaly-boundary-probe-ascend`
- Cardinal boundary probe:
  - SLURM job `8726006`
  - run id `20260423_124423_723348_matched-cross-cluster-94-anomaly-boundary-probe-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260423_124423_723348_matched-cross-cluster-94-anomaly-boundary-probe-cardinal`

Intended interpretation:

- if divergence already appears in pre-CLT input fingerprints, move upstream of
  CLT encode in the next debug pass,
- if pre-CLT input matches but preactivation / margin diverges, focus next on
  CLT encode precision / determinism,
- if preactivation matches but mask / post-mask diverges, focus next on the
  boundary logic itself,
- only after localizing the earliest differing boundary should we decide whether
  the Phase-3 divergence is downstream amplification or needs its own separate
  root-cause investigation.

### 2026-04-23 — compare-upcast hypothesis closed negative; move to upstream boundary fingerprinting

Purpose:

- record the decision boundary after full baseline/`fp32`/`fp64` compare-matrix
  readout for `94_base`,
- document the new immediate investigation direction.

Decision summary:

- The narrow Phase-0 compare-mode hypothesis is now treated as **negative** for
  `94_base` in the matched single-step setup.
- Baseline, `fp32`, and `fp64` compare modes preserved the same cross-cluster
  divergence pattern (Phase-0 structural split, Phase-1 match, downstream
  Phase-3 frontier differences).
- The next step is to localize divergence **upstream** with boundary
  fingerprints rather than iterate more compare-mode variants.

New implementation direction:

- Add compact per-layer fingerprints for:
  - pre-CLT input (`mlp_in_cache`),
  - CLT encode constants (W_enc/b_enc/threshold),
  - preactivation,
  - margin (`preactivation - threshold`),
  - mask membership,
  - post-mask activation.
- Expand near-threshold epsilon counts to better characterize borderline mass.
- Keep artifact payloads compact (hashes + compact stats only).

Planning provenance at decision time:

- project repo:
  - workspace: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - commit baseline: `bad09e38da850ece4cae085eefeabbe7e63ca056`
- sibling library repo:
  - workspace: `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit baseline: `bd9f3c16bbfddfe499706eb357863c5d9ac0d1b1`

Immediate batch direction:

- Run one new matched baseline pair (`94_base`, single-step,
  `cross_cluster_debug=true`) with upstream boundary fingerprints enabled and
  compare mode left at baseline.

### 2026-04-22 — launched 6-job matched `94_base` Phase-0 compare matrix

Purpose:

- execute the planned single-step `94_base` diagnostic matrix in one queue wave,
- compare the baseline Phase-0 threshold path against narrow Phase-0 compare
  upcasts (`fp32`, `fp64`) while keeping the rest of the tracing contract
  matched across clusters.

Launch provenance:

- source project workspace before snapshot:
  - repo: `nlp_research_project`
  - workspace: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - commit: `bad09e38da850ece4cae085eefeabbe7e63ca056`
- source sibling library before snapshot:
  - repo: `circuit-tracer_chunked`
  - workspace: main sibling workspace `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `bd9f3c16bbfddfe499706eb357863c5d9ac0d1b1`
- immutable snapshot container:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260422_154606_matched_cross_cluster_94_phase0_compare_matrix`
- snapshot project root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260422_154606_matched_cross_cluster_94_phase0_compare_matrix/nlp_research_project`
- snapshot library root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260422_154606_matched_cross_cluster_94_phase0_compare_matrix/circuit-tracer_chunked`

Shared config across all six jobs:

- fixture: `94_base`
- tier: `anomaly`
- `max_steps=1`
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
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
- walltime override: `01:00:00`

Scenario files used:

- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_baseline_ascend.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_baseline_cardinal.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_fp32_ascend.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_fp32_cardinal.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_fp64_ascend.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_fp64_cardinal.json`

Submitted jobs:

- Ascend baseline:
  - SLURM job `5040750`
  - run id `20260422_154658_571947_matched-cross-cluster-94-anomaly-baseline-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_154658_571947_matched-cross-cluster-94-anomaly-baseline-ascend`
- Cardinal baseline:
  - SLURM job `8706829`
  - run id `20260422_154658_754153_matched-cross-cluster-94-anomaly-baseline-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260422_154658_754153_matched-cross-cluster-94-anomaly-baseline-cardinal`
- Ascend Phase-0 compare `fp32`:
  - SLURM job `5040751`
  - run id `20260422_154658_838439_matched-cross-cluster-94-anomaly-p0-fp32-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_154658_838439_matched-cross-cluster-94-anomaly-p0-fp32-ascend`
- Cardinal Phase-0 compare `fp32`:
  - SLURM job `8706832`
  - run id `20260422_154658_929983_matched-cross-cluster-94-anomaly-p0-fp32-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260422_154658_929983_matched-cross-cluster-94-anomaly-p0-fp32-cardinal`
- Ascend Phase-0 compare `fp64`:
  - SLURM job `5040752`
  - run id `20260422_154659_015092_matched-cross-cluster-94-anomaly-p0-fp64-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_154659_015092_matched-cross-cluster-94-anomaly-p0-fp64-ascend`
- Cardinal Phase-0 compare `fp64`:
  - SLURM job `8706834`
  - run id `20260422_154659_103228_matched-cross-cluster-94-anomaly-p0-fp64-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260422_154659_103228_matched-cross-cluster-94-anomaly-p0-fp64-cardinal`

Intended interpretation:

- baseline pair measures current Phase-0 divergence with richer diagnostics,
- `fp32` pair tests whether a moderate upcast at the Phase-0 threshold compare
  boundary is enough to reduce membership drift,
- `fp64` pair tests whether a stronger compare upcast further reduces
  Phase-0/Phase-3 divergence,
- all six runs remain single-step, so they still target the first-next-logit
  divergence question rather than later-step replay behavior.

### 2026-04-22 — preliminary Ascend-only readout for the 6-job `94_base` Phase-0 compare matrix

Purpose:

- record the early signal available while Cardinal remained queued / incomplete,
- check whether the Ascend baseline, `fp32`, and `fp64` variants already show
  any within-cluster behavioral change from the narrow Phase-0 compare-mode
  edit.

Analysis provenance:

- project repo provenance from the immutable launch snapshot:
  - repo: `nlp_research_project`
  - source workspace before snapshot: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - commit: `bad09e38da850ece4cae085eefeabbe7e63ca056`
- sibling library provenance from the immutable launch snapshot:
  - repo: `circuit-tracer_chunked`
  - source workspace before snapshot: main sibling workspace `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `bd9f3c16bbfddfe499706eb357863c5d9ac0d1b1`
- snapshot container:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260422_154606_matched_cross_cluster_94_phase0_compare_matrix`

Runs analyzed:

- Ascend baseline:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_154658_571947_matched-cross-cluster-94-anomaly-baseline-ascend/ascend_matched_cross_cluster_94_base_anomaly_p0baseline_b128_c2048_cache0g`
- Ascend Phase-0 compare `fp32`:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_154658_838439_matched-cross-cluster-94-anomaly-p0-fp32-ascend/ascend_matched_cross_cluster_94_base_anomaly_p0fp32_b128_c2048_cache0g`
- Ascend Phase-0 compare `fp64`:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_154659_015092_matched-cross-cluster-94-anomaly-p0-fp64-ascend/ascend_matched_cross_cluster_94_base_anomaly_p0fp64_b128_c2048_cache0g`

Observed Ascend-only results:

- all three Ascend runs completed successfully,
- the generated first token matched exactly across modes: `Let`,
- the compact traced artifact `step_000.npz` matched exactly across modes,
  including:
  - `row_idx`, `col_idx`, and `weights`,
  - `feature_ids`,
  - token text and logprob,
- Phase-0 active-feature membership was unchanged across baseline / `fp32` /
  `fp64`:
  - `active_feature_indices_hash = 7c2fdf6b069843d1`
  - `active_feature_membership_hash_canonical = 2042e3cf71dc07b7`
  - total near-threshold count (`abs_lte_1e-05`) remained `209` in all three
    runs,
- Phase-1 target-logit state also matched exactly:
  - `target_logit_state_hash = 437d56a13df41ec1`,
- Phase-2 feature ordering matched exactly,
- Phase-3 frontier selection remained effectively unchanged:
  - `frontier_pre_locality_hash = 060b1515c58914bf`
  - `frontier_post_locality_hash = ab0d84d2e04d36d7`
  - top seed identities and ranks matched across all three Ascend runs,
- the only differences observed were tiny floating-point changes in some
  Phase-3 scalar summaries / hashes (for example influence summaries and
  normalization summary values), with top-seed influence deltas only around
  `1e-12` and no seed-rank or frontier change.

Interim interpretation:

- for `94_base` on Ascend, the narrow Phase-0 compare-mode knob did **not**
  produce a meaningful within-cluster behavioral change,
- this does **not** yet rule out a cross-cluster benefit, because the key
  remaining question is whether Cardinal moves toward the Ascend baseline once
  the matched Cardinal `fp32` / `fp64` runs finish,
- but the Ascend-only readout already says the new compare modes are not by
  themselves perturbing the Ascend result for this prompt.

Current status after this note:

- treat this as a preliminary within-Ascend sanity check only,
- do not reinterpret the Phase-0 hypothesis yet until the Cardinal half of the
  matrix completes and we can compare the matched cross-cluster pairs directly.

### 2026-04-23 — full readout of the 6-job matched `94_base` Phase-0 compare matrix

Purpose:

- finish the cross-cluster analysis once the delayed Cardinal jobs completed,
- determine whether the narrow Phase-0 threshold-compare upcasts (`fp32`,
  `fp64`) reduced the previously observed Ascend/Cardinal divergence for
  `94_base`.

Analysis provenance:

- project repo provenance from the immutable launch snapshot:
  - repo: `nlp_research_project`
  - source workspace before snapshot: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - commit: `bad09e38da850ece4cae085eefeabbe7e63ca056`
- sibling library provenance from the immutable launch snapshot:
  - repo: `circuit-tracer_chunked`
  - source workspace before snapshot: main sibling workspace `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `bd9f3c16bbfddfe499706eb357863c5d9ac0d1b1`
- snapshot container:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260422_154606_matched_cross_cluster_94_phase0_compare_matrix`

Runs analyzed:

- Ascend baseline / `fp32` / `fp64`:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_154658_571947_matched-cross-cluster-94-anomaly-baseline-ascend/ascend_matched_cross_cluster_94_base_anomaly_p0baseline_b128_c2048_cache0g`
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_154658_838439_matched-cross-cluster-94-anomaly-p0-fp32-ascend/ascend_matched_cross_cluster_94_base_anomaly_p0fp32_b128_c2048_cache0g`
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_154659_015092_matched-cross-cluster-94-anomaly-p0-fp64-ascend/ascend_matched_cross_cluster_94_base_anomaly_p0fp64_b128_c2048_cache0g`
- Cardinal baseline / `fp32` / `fp64`:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260422_154658_754153_matched-cross-cluster-94-anomaly-baseline-cardinal/cardinal_matched_cross_cluster_94_base_anomaly_p0baseline_b128_c2048_cache0g`
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260422_154658_929983_matched-cross-cluster-94-anomaly-p0-fp32-cardinal/cardinal_matched_cross_cluster_94_base_anomaly_p0fp32_b128_c2048_cache0g`
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260422_154659_103228_matched-cross-cluster-94-anomaly-p0-fp64-cardinal/cardinal_matched_cross_cluster_94_base_anomaly_p0fp64_b128_c2048_cache0g`

Observed within-cluster behavior:

- Ascend remained invariant across baseline / `fp32` / `fp64`:
  - identical `step_000.npz` compact artifacts,
  - identical generated token / logprob,
  - identical Phase-0 active-feature membership hashes,
  - only tiny Phase-3 floating-point summary drift.
- Cardinal showed the same pattern:
  - identical `step_000.npz` compact artifacts across baseline / `fp32` /
    `fp64`,
  - identical generated token / logprob,
  - identical Phase-0 active-feature membership hashes,
  - only tiny Phase-3 floating-point summary drift.

Observed cross-cluster behavior (same for baseline, `fp32`, and `fp64`):

- both clusters still generated the same first token: `Let`,
- next-token logprob delta stayed unchanged at approximately `7.65e-04`,
- Phase-1 target-logit state still matched exactly in all three compare modes:
  - `target_logit_state_hash = 437d56a13df41ec1`,
- the **first structural divergence still appeared at Phase-0** in all three
  compare modes:
  - Ascend Phase-0 hashes stayed
    - `active_feature_indices_hash = 7c2fdf6b069843d1`
    - `active_feature_membership_hash_canonical = 2042e3cf71dc07b7`
  - Cardinal Phase-0 hashes stayed
    - `active_feature_indices_hash = 62b94f74a7172516`
    - `active_feature_membership_hash_canonical = ba1ab906c0c08e5e`
- the resulting compact graph similarity also stayed unchanged across all three
  compare modes:
  - feature Jaccard `0.9936243685806376`
  - edge Jaccard `0.5709439233685891`
  - weighted edge Jaccard `0.6128084788664958`
- shared / differing feature counts also stayed unchanged across all three
  compare modes:
  - shared features: `3,359,910`
  - Ascend-only features: `11,433`
  - Cardinal-only features: `10,126`
- Phase-3 divergence likewise remained unchanged at the structural level:
  - frontier hashes still differed between clusters,
  - the top-seed lists still agreed on the first several strongest seeds but not
    the entire top-8 set,
  - compare-mode upcasts did not improve that overlap.

Wall-clock notes from this batch:

- Ascend durations stayed around `1719.95s` to `1815.66s`,
- Cardinal durations stayed around `1057.32s` to `1157.31s`,
- the compare-mode choice did not reveal a meaningful runtime tradeoff at this
  batch scale.

Interpretation:

- this 6-job matrix does **not** support the hypothesis that the current
  Ascend/Cardinal `94_base` divergence is primarily caused by the final
  bf16-vs-fp32/fp64 threshold comparison at the narrow Phase-0 JumpReLU
  membership boundary,
- more strongly: the tested compare-mode upcasts produced **no measurable
  reduction** in cross-cluster divergence for this prompt under the matched
  single-step configuration,
- the most likely remaining explanation is that the important drift already
  exists in earlier model / transcoder activations entering Phase-0 sparse setup,
  rather than in the final threshold-compare precision alone.

Updated takeaway:

- treat the narrow Phase-0 compare-upcast idea as **negative for `94_base`** in
  this matched single-step test,
- future cross-cluster investigation should move upstream of the final threshold
  compare boundary rather than expecting `fp32` / `fp64` compare-mode alone to
  resolve the anomaly.

### 2026-04-22 — intended next diagnostic batch: 6 matched `94_base` jobs

Purpose:

- use one queued batch to both **diagnose** the Phase-0 divergence and **test** a
  narrow mitigation without needing a second edit-submit cycle,
- keep the next batch focused on `94_base`, since that is the prompt of most
  immediate interest for the cross-cluster anomaly story.

Planning provenance:

- project repo: `nlp_research_project`
  - workspace: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - commit baseline for planning: `94e4283a9e2b0c04d63409890b5936c07c17108c`
  - current workspace state: live workspace with pending local edits to docs,
    scripts, and generated scenario files before the next implementation snapshot
- sibling library repo: `circuit-tracer_chunked`
  - workspace: main sibling workspace `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit baseline for planning: `d1f3df3e456fdb5430b5462a0a6834bdaf7fa716`

Queueing / launch policy for the next batch:

- use immutable workspace snapshots only,
- keep default sbatch walltime at `01:00:00` on both clusters,
- keep the batch fully matched across clusters except for the explicit Phase-0
  compare-mode variant under test.

Shared intended config across all six jobs:

- fixture: `94_base`
- tier: `anomaly`
- `max_steps=1`
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
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

Planned six-job matrix:

1. Ascend baseline Phase-0 compare path
2. Cardinal baseline Phase-0 compare path
3. Ascend Phase-0 compare upcast = `fp32`
4. Cardinal Phase-0 compare upcast = `fp32`
5. Ascend Phase-0 compare upcast = `fp64`
6. Cardinal Phase-0 compare upcast = `fp64`

Additional diagnostics intended in this batch:

- Phase-0:
  - canonical sorted active-feature hash,
  - near-threshold counts,
  - sample borderline features near the JumpReLU threshold,
  - clearer separation of membership drift vs ordering-only drift.
- Phase-3:
  - top-K seed influences,
  - cutoff margin / near-tie counts,
  - richer shadow-frontier overlap diagnostics,
  - more explicit row-input / row-abs-sum comparison fields.

Decision rationale:

- a 4-job batch (baseline + one mitigation) would tell us whether *some*
  upcast helps,
- the chosen 6-job batch should tell us whether:
  - no targeted upcast helps,
  - fp32 is already sufficient,
  - or fp64 gives a materially better reduction in Phase-0 and Phase-3 drift.

Current expectation before launch:

- if the Phase-0 divergence is primarily caused by bf16 threshold flips in the
  transcoder sparse encode path, then the fp32/fp64 compare variants should
  reduce active-feature membership drift relative to the baseline pair,
- if drift remains essentially unchanged across all three modes, then the root
  cause is likely earlier model/transcoder activation differences rather than the
  final threshold compare precision alone.

### 2026-04-22 — first analysis of matched cross-cluster debug pairs (`828_base` / `94_base`)

Runs analyzed:

- `828_base` fast:
  - Ascend output: `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260422_001758_409577_matched-cross-cluster-828-fast-ascend/ascend_matched_cross_cluster_828_base_fast_b128_c2048_cache0g`
  - Cardinal output: `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/20260422_001758_483504_matched-cross-cluster-828-fast-cardinal/cardinal_matched_cross_cluster_828_base_fast_b128_c2048_cache0g`
- `94_base` anomaly:
  - Ascend output: `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_001758_549185_matched-cross-cluster-94-anomaly-ascend/ascend_matched_cross_cluster_94_base_anomaly_b128_c2048_cache0g`
  - Cardinal output: `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260422_001758_619013_matched-cross-cluster-94-anomaly-cardinal/cardinal_matched_cross_cluster_94_base_anomaly_b128_c2048_cache0g`

Important scope note:

- these are still **single-step** (`max_steps=1`) debug runs,
- so they are good for locating the first divergence checkpoint, but they do not
  directly test later-step "refresh every step" behavior.

Observed `828_base` results:

- both clusters generated the same first token: `Here` (`token_id=8291`),
- next-token logprob remained very close (`Δ ≈ 1.45e-05`),
- Phase-1 target-logit hash matched exactly,
- first hash divergence appeared already at `phase0_sparse_setup`,
- compact artifact comparison:
  - feature Jaccard `0.99343`
  - edge Jaccard `0.45940`
  - weighted edge Jaccard `0.49887`
- Phase-4 wall-clock still differed substantially:
  - Ascend `1051.30s`
  - Cardinal `564.60s`

Observed `94_base` results:

- both clusters generated the same first token: `Let` (`token_id=6481`),
- next-token logprob stayed close but less tightly than `828_base`
  (`Δ ≈ 7.65e-04`),
- Phase-1 target-logit hash matched exactly,
- first hash divergence again appeared at `phase0_sparse_setup`,
- compact artifact comparison:
  - feature Jaccard `0.99362`
  - edge Jaccard `0.57094`
  - weighted edge Jaccard `0.61281`
- Phase-4 wall-clock still differed substantially:
  - Ascend `1201.76s`
  - Cardinal `818.57s`

Checkpoint interpretation:

- for **both prompts**, the first structural divergence appears in
  `phase0_sparse_setup` via differing retained feature sets / active-feature
  hashes,
- despite that early structural difference, `phase1_target_logits` still matched
  exactly on the chosen first token for both prompts,
- by `phase3_seed_ranking_pre_phase4`, the ranking/frontier hashes diverged on
  both prompts, so Phase 3 is the first checkpoint where the difference is more
  clearly expressed in the actual frontier/ranking state,
- `phase4_entry` itself had no dedicated hash difference record, but Phase-4
  runtime remained materially different across clusters.

Current interpretation:

- under matched `fp64` + matched `decoder_chunk_size=2048`, the first-step
  cross-cluster story is **not** "Cardinal immediately collapses while Ascend
  stays healthy" for `94_base`,
- instead, both prompts show the same pattern:
  - early feature-set divergence at Phase 0,
  - matching target-logit state at Phase 1,
  - frontier/ranking divergence visible by Phase 3,
  - substantial Phase-4 runtime differences despite the matched tracing
    contract,
- this supports treating Phase 0 as the first structural divergence point and
  Phase 3 as the first obviously downstream-meaningful divergence point,
- to test the earlier overflow / repeated-refresh hypothesis directly, we likely
  need a later-step or multi-step matched run on `94_base`, not just the current
  single-step check.

### 2026-04-22 — launched matched cross-cluster debug pairs for `828_base` and `94_base`

Purpose:

- start the main-branch cross-cluster investigation with **matched configs** so
  cluster is the intended primary source of variance,
- pair one healthy control prompt (`828_base` in `fast`) with one anomalous
  prompt (`94_base` in `anomaly`).

Launch provenance:

- source project workspace before snapshot: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - commit: `94e4283a9e2b0c04d63409890b5936c07c17108c`
- source sibling library before snapshot: `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `d1f3df3e456fdb5430b5462a0a6834bdaf7fa716`
- immutable snapshot container:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260422_001720_matched_cross_cluster_828_94_fp64`
- snapshot project root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260422_001720_matched_cross_cluster_828_94_fp64/nlp_research_project`
- snapshot library root:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260422_001720_matched_cross_cluster_828_94_fp64/circuit-tracer_chunked`

Matched launch config used on **all four** runs:

- exact tracing, single-step (`max_steps=1`)
- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`
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

Scenario files used:

- `experiments/generated/exact_trace_bench/matched_cross_cluster_828_fast_ascend.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_828_fast_cardinal.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_ascend.json`
- `experiments/generated/exact_trace_bench/matched_cross_cluster_94_anomaly_cardinal.json`

Submitted jobs:

- Ascend `828_base` fast:
  - SLURM job `5024202`
  - run id `20260422_001758_409577_matched-cross-cluster-828-fast-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260422_001758_409577_matched-cross-cluster-828-fast-ascend`
- Cardinal `828_base` fast:
  - SLURM job `8694599`
  - run id `20260422_001758_483504_matched-cross-cluster-828-fast-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/20260422_001758_483504_matched-cross-cluster-828-fast-cardinal`
- Ascend `94_base` anomaly:
  - SLURM job `5024203`
  - run id `20260422_001758_549185_matched-cross-cluster-94-anomaly-ascend`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260422_001758_549185_matched-cross-cluster-94-anomaly-ascend`
- Cardinal `94_base` anomaly:
  - SLURM job `8694600`
  - run id `20260422_001758_619013_matched-cross-cluster-94-anomaly-cardinal`
  - output root `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/anomaly/20260422_001758_619013_matched-cross-cluster-94-anomaly-cardinal`

Interpretation / intent:

- these launches intentionally remove the earlier `decoder_chunk_size`
  asymmetry (`2048` on Ascend vs `4096` on Cardinal) from the initial
  cross-cluster comparison,
- the point of this run set is not to preserve each cluster's historical native
  benchmark config, but to isolate where the two clusters first diverge under a
  **matched tracing contract**,
- analysis is still pending; this entry records the exact launch decision and
  provenance so later interpretation can reconstruct what was submitted.

### 2026-04-22 — post-library-merge within-cluster baseline consistency check

Purpose:

- verify that the merged main baseline did not silently change `828_base`
  outputs **within each cluster** before starting cross-cluster drift analysis.

Project + library provenance for the checked baseline pair:

- project repo: `nlp_research_project`
  - workspace: main workspace `./`
  - branch: `exact-trace-bench-harness`
  - commit: `94e4283a9e2b0c04d63409890b5936c07c17108c`
- sibling library repo: `circuit-tracer_chunked`
  - workspace: main sibling workspace `../circuit-tracer_chunked`
  - branch: `exact-trace-hidden-knobs`
  - commit: `d1f3df3e456fdb5430b5462a0a6834bdaf7fa716`

Runs checked:

- Ascend current baseline-check run:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_quick_fp64_cross_cluster_828_base_postlibmerge_b128_c2048_cache0g`
  - launched from the live main workspace
- Ascend same-cluster reference run:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/ascend_quick_fp64_cross_cluster_828_base_b128_c2048_cache0g`
- Cardinal current baseline-check run:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/20260421_214702_004280_quick-cross-cluster-fp64-828-postlibmerge-snapshot/cardinal_quick_fp64_cross_cluster_828_base_postlibmerge_b128_c4096_cache0g`
  - launched from immutable snapshot workspace `workspace_20260421_214702_cardinal_828_postlibmerge`
- Cardinal same-cluster reference run:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/20260420_131422_618249_prompt828-debug-cardinal-float64-norm-general/cardinal_fast_828_base_b128_c4096_cache0g`

Validation notes:

- live main `trace_pipeline_chunked.py` matched the Cardinal snapshot copy by
  file hash,
- live main `circuit_tracer/attribution/attribute_nnsight.py` matched the
  Cardinal snapshot copy by file hash.

Observed within-cluster comparisons:

- Ascend matched exactly against the earlier same-cluster fp64 quick run for:
  - generated token (`Here`),
  - next-token id/logprob,
  - active feature count (`2993540`),
  - telemetry event count (`2760`),
  - saved `step_000.npz` file hash,
  - saved `feature_ids`, `row_idx`, `col_idx`, and `weights` hashes.
- Cardinal matched exactly against the earlier same-cluster prompt-828 debug
  run for:
  - generated token (`Here`),
  - next-token id/logprob,
  - active feature count (`2993606`),
  - telemetry event count (`1952`),
  - saved `step_000.npz` file hash,
  - saved `feature_ids`, `row_idx`, `col_idx`, and `weights` hashes.
- Cardinal run-metadata/debug-mode fields differed (`cross_cluster_debug` in
  the new run versus `phase4_anomaly_debug` in the older debug campaign), but
  the saved compact step artifact for `828_base` was unchanged within Cardinal.

Interpretation:

- this check supports that we did **not** accidentally move onto a bad merged
  baseline for either cluster,
- treat the project/library pair `94e4283` + `d1f3df3` as a validated baseline
  pair for continuing the **cross-cluster** investigation,
- remaining Ascend/Cardinal differences are expected and should be investigated
  as actual cross-cluster drift rather than as evidence of baseline corruption.

## Scratch layout

Current top-level layout:

```text
/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/
  ascend/
    fast/
    anomaly/
    long_eval/
    matched_debug/
  cardinal/
    fast/
    anomaly/
    long_eval/
    matched_debug/
  manual_scenarios/
  manual_worktrees/
  workspace_snapshots/
```

### Meaning of the subtrees

- `ascend/fast`: main fast-loop benchmark runs on Ascend, usually `828_base` and
  `361_base`
- `ascend/anomaly`: prompt-94 watch / anomaly runs on Ascend
- `ascend/long_eval`: late-prefix stress runs on Ascend
- `ascend/matched_debug`: matched debug or manual comparison runs on Ascend
- `cardinal/*`: Cardinal equivalents where available
- `manual_scenarios/`: custom JSON scenario files for one-off comparison
  matrices
- `manual_worktrees/`: manually created worktrees for branch / commit isolation
- `workspace_snapshots/`: immutable project + library snapshots used for actual
  launches

## Current inventory summary

From the current extractor pass over `exact_trace_bench/`:

- total extracted scenario rows: `66`

By cluster / group:

- `ascend/fast`: `17`
- `ascend/anomaly`: `14`
- `ascend/long_eval`: `4`
- `ascend/matched_debug`: `13`
- `cardinal/fast`: `3`
- `cardinal/anomaly`: `6`
- `cardinal/long_eval`: `4`
- `cardinal/matched_debug`: `5`

## Major experiment families

### 1. Historical unnamed baselines

These are older direct benchmark outputs that do not carry a `run_name` in the
current extraction tables.

They cover a mix of:

- `361_base`
- `828_base`
- `94_base`
- late-prefix fixtures (`361_late`, `828_late`, `94_late`)

Use these as historical references only; prefer named runs when possible.

### 2. Memory-refactor smoke runs

Purpose: validate early exact-mode memory/system improvements before Phase 4
work.

Current named families:

- `lazy encoder smoke ascend`
- `memmap row-store fast ascend`

These are primarily Ascend runs on:

- `361_base`
- `828_base`
- `94_base`

### 3. Phase 4 locality / autoscale / planner runs

Purpose: test the first Phase 4 scheduling optimization and then batch-size
control strategies.

Current named families:

- `phase4 locality reorder fast ascend`
- `phase4 autoscale fast ascend cuda fix`
- `phase4 autoscale anomaly ascend`
- `phase4 autoscale anomaly ascend cuda fix`
- `phase4 planner fast ascend`
- `phase4 planner anomaly ascend`

Interpretation notes:

- locality reorder is the first real Phase 4 optimization family
- old autoscale runs are historical only; planner/preflight is the current path
- planner runs are the important batch-size comparison set

### 4. Telemetry validation runs

Purpose: validate structured telemetry artifacts and extraction.

Current named family:

- `telemetry validation fast ascend`

This run family is important because it established:

- `telemetry.jsonl`
- completion-level timing summaries
- downstream extraction via `experiments/telemetry_gathering.py`

### 5. Prompt-94 anomaly debug campaigns

Purpose: investigate Phase 4 anomaly behavior on prompt 94.

Ascend families:

- `prompt94 anomaly debug ascend`
- `prompt94 anomaly debug ascend v2`
- `prompt94 anomaly debug ascend v3`
- `prompt94 standard ascend float64 norm`
- `prompt94 anomaly debug ascend float64 norm general`

Cardinal families:

- `prompt94 anomaly debug cardinal`
- `prompt94 anomaly debug cardinal v2`
- `prompt94 anomaly debug cardinal v3`
- `prompt94 standard cardinal float64 norm`
- `prompt94 anomaly debug cardinal float64 norm general`

Interpretation notes:

- these are watch / diagnosis runs, not normal optimization baselines
- `*standard* float64 norm` runs are especially important as post-float64
  non-debug references

### 6. Prompt-828 float64 debug runs

Purpose: check whether the prompt-94 anomaly story generalizes to a healthy base
prompt.

Current named families:

- `prompt828 debug ascend float64 norm general`
- `prompt828 debug cardinal float64 norm general`

Interpretation notes:

- these are diagnostic runs, not clean performance baselines
- use them for matched graph/debug comparisons, not direct speed claims

### 7. Matched-debug fixture campaigns

Purpose: compare a larger base-fixture set under the matched-debug protocol.

Current named families:

- `matched debug ascend b256 c4096`
- `matched debug cardinal b256 c4096`

Current fixtures in this family:

- `828_base`
- `613_base`
- `999_base`
- `1046_base`
- `1075_base`

These runs live under `*/matched_debug/` and use the matched-debug fixture
catalog produced from:

- `experiments/generated/weekend_exact_chunked_fixtures_matched_debug/`

### 8. Phase 4 refresh cache run

Purpose: test the first Direction-A refresh-cache implementation.

Current named family:

- `phase4 refresh cache fast ascend`

This family includes:

- `361_base`
- `828_base`
- `94_base`

Interpretation note:

- later analysis showed the measured speedups were not well explained by cache
  hits, so this family should be treated as an exploratory checkpoint rather
  than a settled optimization win

### 9. fp32 / float64 normalization comparison matrix

Purpose: isolate whether the Phase 4 normalization precision change explains the
large runtime shifts.

Current named families:

- `fp32 norm debug ascend`
- `float64 norm debug ascend`
- `fp32 norm 361 baseline ascend`
- `float64 norm 361 baseline ascend`

These were launched from custom scenario files under:

- `manual_scenarios/ascend_phase4_norm_compare_debug_matrix.json`
- `manual_scenarios/ascend_phase4_norm_compare_361_baseline.json`

and custom library worktrees under:

- `manual_worktrees/circuit-tracer_chunked_fp32_norm_9afff02`
- `manual_worktrees/circuit-tracer_chunked_float64_norm_62f5271`

Important interpretation note:

- commit `9afff02` already uses float64 normalization sums in anomaly-debug mode
- therefore `fp32 norm debug ascend` is **not** a true fp32-normalization debug
  run; it is only fp32-branch + debug-mode float64 normalization
- the clean normalization A/B currently available is the non-debug
  `361_base` baseline pair

Status:

- completed / analyzed

### 10. True fp32 / fp64 internal-dtype comparison matrix

Purpose: rerun the normalization comparison using the new external runtime knob
so the comparison is a **true** fp32-vs-fp64 test on the current codebase.

Current named families:

- `true fp32 norm debug ascend`
- `true fp64 norm debug ascend`
- `true fp32 norm 361 baseline ascend`
- `true fp64 norm 361 baseline ascend`

These runs use the new runtime control knob:

- `exact_trace_internal_dtype=fp32|fp64`

and were launched from custom scenario files under:

- `manual_scenarios/ascend_true_norm_fp32_debug_fast.json`
- `manual_scenarios/ascend_true_norm_fp32_debug_anomaly.json`
- `manual_scenarios/ascend_true_norm_fp64_debug_fast.json`
- `manual_scenarios/ascend_true_norm_fp64_debug_anomaly.json`
- `manual_scenarios/ascend_true_norm_fp32_361_baseline.json`
- `manual_scenarios/ascend_true_norm_fp64_361_baseline.json`

Current output roots:

- fast / fp32 debug:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260421_124700_true-fp32-norm-debug-ascend`
- anomaly / fp32 debug:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260421_124700_true-fp32-norm-debug-ascend`
- fast / fp64 debug:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260421_124700_true-fp64-norm-debug-ascend`
- anomaly / fp64 debug:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260421_124700_true-fp64-norm-debug-ascend`
- fast / fp32 361 baseline:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260421_124700_true-fp32-norm-361-baseline-ascend`
- fast / fp64 361 baseline:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260421_124700_true-fp64-norm-361-baseline-ascend`

Interpretation note:

- unlike the older `fp32 norm debug ascend` family, these runs are intended to be
  the real fp32/fp64 comparison on the current branch because the runtime dtype
  is controlled externally rather than inferred from debug mode

Status:

- **in progress**

Update:

- now completed / analyzed

Main findings from the true-dtype comparison:

- `828_base` and `361_base` genuinely collapse under true fp32 internal dtype
- `94_base` does not collapse via the same mechanism
- fp64 should remain the default runtime dtype for exact compact tracing
- the healthy-prompt speedups previously observed are best explained by the
  float64 normalization path, not by the refresh-cache attempt

Observed healthy-prompt fp32 collapse signature:

- Phase-3 logit-row normalization sums overflow to `inf`
- refresh ranking becomes effectively all-zero across all refreshes
- runtime shifts into a degenerate regime with:
  - very small `phase4.refresh`
  - huge `phase4.feature_batch` / `context.compute_batch`
  - very large decoder-load counts
- compact outputs keep the same feature set, but fp32 healthy-prompt runs retain
  only `8192` edges instead of the full `20000`

Observed clean `361_base` baseline comparison:

- `fp32` baseline:
  - completion `5167.95s`
  - Phase 4 `4829.21s`
  - RSS `236.12 GiB`
- `fp64` baseline:
  - completion `4303.92s`
  - Phase 4 `3943.07s`
  - RSS `333.90 GiB`

Interpretation:

- fp64 improves `361_base` by roughly `16.7%` end-to-end and `18.3%` in Phase 4
- the improvement comes with much higher host RAM usage
- for long-token / long-trace scaling, fp64 is likely the right immediate
  default but not the final permanent solution if raw normalization sums can
  keep growing toward fp64 overflow

### 11. Long-eval runs

Purpose: late-prefix stress validation rather than fast-loop iteration.

These live under:

- `ascend/long_eval`
- `cardinal/long_eval`

Typical fixtures:

- `361_late`
- `828_late`
- `94_late`

Treat these as stress tests, not as default optimization benchmarks.

## Recommended baseline interpretation by topic

### General fast exact baselines

Prefer:

- named fast runs on `828_base` and `361_base`
- especially the most recent non-debug runs that match the code state you care
  about

### Prompt-94 anomaly reference

Prefer:

- `prompt94 standard ascend float64 norm`
- `prompt94 standard cardinal float64 norm`

### Planner reference

Prefer:

- `phase4 planner fast ascend`
- `phase4 planner anomaly ascend`

### Normalization precision reference

Prefer:

- `fp32 norm 361 baseline ascend`
- `float64 norm 361 baseline ascend`

These are currently the cleanest direct precision comparison pair.

## Notes on debug vs non-debug runs

Runs with `phase4_anomaly_debug=true` should be treated as:

- diagnosis / collapse / stability runs
- not clean performance baselines

Matched analysis so far suggests:

- debug mode can add very large runtime overhead
- but often leaves the final compact graph unchanged

So performance claims should come from matched non-debug runs whenever possible.

## Manual assets currently on scratch

### `manual_scenarios/`

Currently includes custom scenario JSON for:

- normalization precision comparison matrix
- matched `361_base` baseline comparison

### `manual_worktrees/`

Currently includes detached library worktrees for:

- commit `9afff02` (`fp32_norm` label, but debug still uses float64 sums)
- commit `62f5271` (`float64_norm` default)

### `workspace_snapshots/`

Contains immutable project + library snapshots created by:

- `uv run python -m experiments.exact_trace_bench snapshot-workspace ...`

These are the workspaces actually used by benchmark jobs, so long-running runs
do not observe live source edits.

## How to refresh this inventory

Re-extract from scratch:

```bash
uv run python -m experiments.exact_trace_bench extract \
  --input-root /fs/scratch/PAS3272/kopanev.1/exact_trace_bench \
  --output-dir /tmp/exact_trace_extract_all_scratch \
  --skip-slurm
```

Then summarize `benchmark_enriched.csv` by:

- cluster / group
- run name
- fixture set

## Status of this note

This file is descriptive, not normative.

It is meant to answer “what do we currently have on scratch?” rather than “what
should we run next?”.
