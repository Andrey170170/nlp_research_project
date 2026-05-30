# Experiments inventory

Status: Current compact index and interpretation summary
Last updated: 2026-05-30

This file is the readable front page for experiment provenance. It should stay
small enough to edit by hand.

Detailed historical narrative was archived to:

- `docs/history/experiment_log_legacy_2026-05-16.md`

Structured append-only event records now live under:

- `experiments/logs/`

For important future launches, baseline decisions, and reinterpretations:

1. update the compact index here when the result changes current interpretation,
2. append a structured record to `experiments/logs/YYYY-MM.jsonl`, and
3. preserve enough project + sibling-library provenance to reconstruct the run.

## Current baseline

| Item | Current value |
|---|---|
| Project workspace | `/users/PAS2119/andreykopanev/nlp_research_project` |
| Project branch / commit | `main` / `9314f30` (`Record post-consolidation cleanup plan`) |
| Sibling library workspace | `/users/PAS2119/andreykopanev/circuit-tracer_chunked` |
| Sibling branch / commit | `main` / `e91370a` (`Fix Phase-3 row replay effective state`) |
| Editable dependency path | `../circuit-tracer_chunked` |
| Canonical exact-trace dtype | `exact_trace_internal_dtype=fp32` |
| Canonical prompt tiers | `828_base`, `361_base` in `fast`; `94_base` in `anomaly` |
| Scratch root | `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench` |
| Run placement | cluster (`ascend`/`cardinal`) × tier (`fast`/`anomaly`/`long_eval`) |

Baseline preservation notes:

- `828_base` and `361_base` matched same-generation Ascend baselines exactly
  after consolidation.
- `94_base` matched the pre-consolidation optimization-control output exactly;
  mismatch against older Apr-20/21 references predates consolidation.
- The permanent row-L1 overflow fix is part of the validated baseline.
- Within a fixed `decoder_chunk_size`, cache-size changes were exact; changing
  `decoder_chunk_size` can cause small compact-output drift relative to the
  current `c2048` reference.

## Current interpretation

Track-A cross-cluster localization is mostly complete:

- the observed Ascend/Cardinal divergence is primarily driven by Phase-3 gradient
  differences,
- later exact-trace stages amplify those differences enough to affect compact
  circuit outputs,
- debug/replay machinery should remain available as internal validation tooling,
  but it should not dominate the ordinary user-facing workflow.

Clean/current toy parity follow-up (May 2026):

- Current exact-chunked outputs are semantically useful for continued research,
  but strict clean parity depends on NNSight trace/source batch semantics.
- Phase 3 is trace-batch sensitive in the underlying NNSight/replacement path;
  dense/non-chunk and exact-chunked runs agree at the same trace batch, while
  both can differ from singleton. Use singleton trace/batch settings for strict
  clean-like Phase-3 validation.
- Phase 4 is much more stable in same-feature micro-tests, but trace batch >1 can
  produce sparse row-specific differences. Treat batched Phase-4 rows as
  meaningful but not bit-exact clean parity evidence.
- The observed batch sensitivity is not attributed to the exact-chunked decoder
  implementation itself; it appears to be general NNSight/replacement behavior.

Near-term cleanup focus:

1. keep the current exact-trace baseline readable and reproducible,
2. keep docs and scenario defaults from drifting,
3. separate normal benchmark workflow from Track-A replay/debug tooling,
4. add lightweight tests before deeper harness or library refactors.

Track-2 full-answer typed-bucketed interpretation (May 2026):

- Typed-bucketed compact saves replace the old global top-K full-answer save path
  for current temporal comparisons. The old global top-K view is still useful for
  compatibility, but not as the scientific readout for feature topology.
- Across the four completed full-answer traces, the three `828_base` trajectories
  are close by adjacent, lag, and phase metrics; the sampled temp0.8 correct/wrong
  pair does not show a clean correct-vs-wrong separation.
- The wrong `361_base` temp0.8 trajectory is substantially more locally stable,
  especially in middle/late phases and positionless/layer-flow summaries.
- Exact `feature<-feature` edge identity remains highly churny even with about
  95% retained feature-feature mass, so the next useful topology analysis should
  collapse by layer/positionless feature classes rather than comparing exact edge
  IDs alone.

## Current run families

| Family | Meaning | Current status |
|---|---|---|
| `exact_trace_bench/ascend/fast` | Ascend quick validation/debug for normal prompts | Current |
| `exact_trace_bench/cardinal/fast` | Cardinal quick validation/debug for normal prompts | Current |
| `exact_trace_bench/{ascend,cardinal}/anomaly` | `94_base` anomaly/debug/parity work | Current |
| `exact_trace_bench/{ascend,cardinal}/long_eval` | Longer exact-bench evaluation tier | Current but SLURM-only |
| `workspace_snapshots/` | Immutable project + sibling-library launch snapshots | Current provenance mechanism |
| historical `matched_debug` artifacts | Old matched-debug campaign outputs/configs | Historical only; do not use as an ordinary bucket |

## Recent durable decisions

### 2026-05-30 — Typed-bucketed full-answer temporal comparison

The typed-bucketed all-token reruns and temporal analyses completed for the four
current comparison traces:

1. deterministic temp0 correct `828_base`,
2. sampled temp0.8 correct `828_base` seed2003,
3. sampled temp0.8 wrong `828_base` seed1002,
4. sampled temp0.8 wrong `361_base` seed1002.

Key output root:

- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/typed-bucketed-full-reruns-20260527`

Cross-run deep comparison package:

- `deep_temporal_comparison_v1/deep_temporal_comparison_summary.json`
- `deep_temporal_comparison_v1/metric_summary.csv`
- `deep_temporal_comparison_v1/phase_summary.csv`
- `deep_temporal_comparison_v1/bucket_summary.csv`
- combined plots for adjacent metrics, lag decay, phase means, and bucket means.

Mean adjacent metrics from the deep comparison:

| Run | Feature J | All-edge weighted J | Positionless feature J | Layer-flow weighted J |
|---|---:|---:|---:|---:|
| `828` temp0 correct | 0.4148 | 0.4038 | 0.6751 | 0.9477 |
| `828` temp0.8 correct | 0.4209 | 0.4116 | 0.6751 | 0.9408 |
| `828` temp0.8 wrong | 0.4129 | 0.4018 | 0.6782 | 0.9489 |
| `361` temp0.8 wrong | 0.5130 | 0.5104 | 0.7391 | 0.9668 |

Bucket-level interpretation:

- `feature<-feature` exact weighted Jaccard remains near zero across runs
  (`0.0054`--`0.0084`) despite retained fractions near `0.95`.
- `feature<-token` is much more stable (`0.1988`--`0.2671`) and is highest for
  the `361` wrong trajectory.
- logit-facing buckets are stable enough for local output-readout comparisons,
  but they should not be mixed with feature topology in a single global edge cap.

Decision: retain typed-bucketed saves as the current full-answer comparison format
and treat exact feature-feature edge identity as a low-level diagnostic. For
scientific topology comparisons, add collapsed layer/positionless feature-flow
metrics before drawing stronger correct-vs-wrong conclusions.

### 2026-05-27 — Prefix-view validation starting point

The trajectory-batching Phase-1 seam is now committed and SLURM-validated:
full-answer traces pass explicit `prefix_view_metadata` into the sibling
attribution path, where it is validated before attribution without changing graph
math.

Commits:

- project: `20e9625` (`Forward full-answer prefix views`),
- sibling library: `469687f` (`Validate full-answer prefix views`).

Validation run:

- job: Cardinal `10690579_[0]`, completed in `00:06:07`,
- output root:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/full-answer-828-prefix-view-validation-20260527`,
- target: `828_base_full_answer_20260523_02`, generated index `0`, token
  `Here`, target position / prefix length `73`,
- trace time: `304.89s`, status `ok`,
- `trace.json` records `prefix_view_metadata` with mode `independent_prefix`,
  target token ID `8291`, and prefix-token SHA256
  `29b69e3c31366bb2c455f4b719a453c8499b6eafd76df2ae0662cfb143f7a805`.

Comparison result:

- exact compact match against Cardinal Wave-1 cache8g, Cardinal Wave-0 cache0,
  and the prior baseline-parameter full-answer run,
- feature Jaccard `1.0`, edge Jaccard `1.0`, weighted-edge Jaccard `1.0`,
  all-edge-weighted Jaccard `1.0`.

Decision: use `20e9625` + sibling `469687f` as the starting point for
trajectory-level optimization. Future shared-Phase-0 or row-reuse prototypes must
preserve this prefix-view contract and reject target/prefix/hash mismatches
before attribution.

### 2026-05-26 — Full-answer optimized-default Cardinal smoke

After merging the full-answer harness into `opt-phase34-gpu-residency`, a
one-token `828_base` full-answer smoke validated that the optimized trace knobs
are emitted in specs, forwarded by the runner, and executable on Cardinal.

Run:

- project worktree commit: `0c39a7b`, sibling library commit: `6445795`,
- job: Cardinal `10438978_[0]`, completed in `00:06:30`,
- output root:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/full-answer-828-smoke-perf-default-20260526`,
- target: `828_base_full_answer_20260523_02`, generated index `0`, token
  `Here`, prefix length `73`,
- trace time: `330.82s`, status `ok`, graph saved as `graph.npz`.

Validated knobs:

- `exact_trace_internal_dtype=fp32`,
- `cross_batch_decoder_cache_bytes=8589934592`,
- `phase4_row_reduction=gpu_v1`,
- `phase4_refresh_optimization=v1`,
- `phase4_refresh_active_row_accumulation=direct_v1`,
- `row_store_preallocate=true`,
- `verbose_attribution=false`, `profile_attribution=false`.

Interpretation:

- Treat this as a successful plumbing/performance smoke, not a new bitwise
  baseline.
- The best same-cluster historical comparison found so far is an older Cardinal
  Wave-1 cache8g single-token compact artifact: feature Jaccard `0.998536`,
  edge Jaccard `0.693050`, weighted-edge Jaccard `0.816245`, and
  all-edge-weighted Jaccard `0.999173`.
- Newer Cardinal performance-validation artifacts with millions of stored
  features are not comparable compact baselines for this full-answer smoke.
- Next 5x-class work remains trajectory-level reuse/batching, measured against
  this merged full-answer performance path rather than the older pre-merge path.

Follow-up baseline-parameter run:

- job: Cardinal `10497418_[0]`, completed in `00:06:18`,
- output root:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/full-answer-828-baseline-params-20260526`,
- matching compact/reference knobs: `max_edges=20000`,
  `decoder_chunk_size=4096`, `cross_batch_decoder_cache_bytes=8589934592`,
  `phase4_refresh_optimization=off`, `phase4_row_reduction=off`,
  `row_store_preallocate=false`, `verbose_attribution=true`,
  `profile_attribution=true`,
- trace time: `320.61s`, status `ok`,
- comparison against both Cardinal Wave-1 cache8g and Wave-0 cache0 `828_base`
  compact baselines: feature Jaccard `1.0`, edge Jaccard `1.0`,
  weighted-edge Jaccard `1.0`, all-edge-weighted Jaccard `1.0`.

Decision: the full-answer harness path can reproduce the canonical Cardinal
one-token compact baseline exactly when run with matching compact/resource knobs;
the earlier optimized-default smoke's lower edge Jaccard was due to the smoke's
smaller `max_edges=16384` cap and non-baseline performance knobs, not graph drift.

### 2026-05-25 — Optimization worktree performance default candidate

The optimization worktree validated a faster exact-trace performance candidate on
the post-fix compact path before merging the full-answer harness.

Current candidate:

- `exact_trace_internal_dtype=fp32`,
- `phase4_row_reduction=gpu_v1`,
- `phase4_refresh_optimization=v1`,
- `phase4_refresh_active_row_accumulation=direct_v1`,
- `row_store_preallocate=true`,
- performance mode: `cross_batch_decoder_cache_bytes=8GiB`.

Interpretation:

- The strict cache0 variant with `direct_v1 + row_store_preallocate=true` was
  bitwise exact against the cache0 baseline on the broader five-fixture check and
  reduced Phase 4 for every fixture.
- The cache8g performance mode improved end-to-end runtime on all five broader
  fixtures (`-12.8%` to `-36.1%`, mean `-23.1%`) and reduced artifact RSS, but
  retained-edge arrays are cutoff-sensitive rather than bitwise exact against
  cache0.
- Keep cache0 as the bitwise-retained-edge fallback. Use cache8g only when the
  near-exact/cutoff-sensitive performance tradeoff is acceptable.
- The next 5x-class optimization path is trajectory-level reuse/batching in the
  full-answer harness, not another isolated Phase 4 micro-optimization.

Key provenance:

- broader default validation output root:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/20260525_broader-default-validation-fast`,
- snapshot root:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260525_190714_broader_default_validation`,
- optimization project commit before merge: `a01ab07`,
- optimization library commit before merge: `6445795`.

### 2026-05-22 — Sweep Wave 4 prompt generalization

Wave 4 broadened the finalist validation from sentinel prompts to the Wave 0
fast/anomaly/long-eval prompt coverage.

Run:

- `wave4-generalization-20260521-01`
- jobs: Ascend `5381290`, `5381291`, `5381292`; Cardinal `10306425`,
  `10306426`, `10306427`
- Cardinal submissions excluded `c0811`.

Effective result:

- 144/144 successful scenarios, all compared against Wave 0 fp32 baselines.
- Minimum feature/edge Jaccard: `1.0`.
- Minimum weighted-edge Jaccard: `0.9999999998808901`; the only sub-1.0 rows
  were three Ascend `row_subchunk_512` cases with feature/edge Jaccard still
  exactly `1.0`, so this is treated as tiny numerical weighted-edge noise rather
  than structural compact-graph drift.

Runtime/resource interpretation:

- `plan_feature_batch_size=true` was the best broad finalist overall:
  mean runtime ratio `0.981`, median `0.975`, geometric mean `0.971`, with
  26/48 prompt-cluster-tier cases faster than baseline by >2%.
- `row_subchunk_size=512` was mixed: mean ratio `1.015`, median `1.002`, with
  22/48 faster and 22/48 slower by >2%.
- Long-eval favored `plan_feature_batch_size=true` (`0.955` mean ratio) more than
  `row_subchunk_size=512` (`0.993` mean ratio).
- Slurm MaxRSS differences were small overall; neither finalist showed a broad
  memory reduction relative to baseline.

Decision:

- Promote `plan_feature_batch_size=true` as the safest broad default/finalist
  candidate from Wave 4.
- Keep `row_subchunk_size=512` as an opt-in prompt/resource tuning knob rather
  than a global default.
- Continue not promoting combined Wave 3 interactions or streaming as ordinary
  defaults.

Structured record:

- `experiments/logs/2026-05.jsonl`

### 2026-05-21 — Sweep Wave 3 interaction confirmation

Wave 3 combined the selected Wave 2 candidates, including the optional speed
interaction, across Ascend/Cardinal fast/anomaly sentinel prompts.

Run:

- `wave3-interaction-20260521-01`
- jobs: Ascend `5379152`, `5379153`; Cardinal `10303141`, `10303142`
- Cardinal submissions excluded `c0811`.

Effective result:

- 42/42 successful scenarios, all compared against Wave 0 fp32 baselines.
- Minimum feature/edge/weighted-edge Jaccard across all Wave 3 scenarios: `1.0`.
- No promoted interaction caused compact graph drift.

Runtime/resource interpretation:

- Best aggregate mean runtime ratio: `row_subchunk_size=512` (`0.969`).
- `phase4_refresh_policy=deferred_v1` was close (`0.978`) and helped Cardinal,
  but was mixed on Ascend.
- `plan_feature_batch_size=true` was near-neutral (`0.986`) and remains a
  conservative memory/planning candidate rather than a speed win.
- Combined candidates did not beat the best singleton:
  `deferred_v1 + row_subchunk_size=512` was neutral (`0.997`),
  `deferred_v1 + plan_feature_batch_size=true` was slower (`1.029`), and
  `deferred_v1 + streaming_v1 + row_subchunk_size=512` was exact but slower
  overall (`1.042`).

Decision:

- Promote `row_subchunk_size=512` as the primary next default candidate to test
  more broadly.
- Keep `deferred_v1` as a Cardinal/throughput candidate, not a universal default
  yet.
- Keep `plan_feature_batch_size=true` for conservative memory-sensitive launches.
- Do not promote the combined Wave 3 interactions or the streaming interaction as
  global defaults.

Structured record:

- `experiments/logs/2026-05.jsonl`

### 2026-05-21 — Sweep Wave 2 complete; Wave 3 candidate set

Wave 2 completed the independent advanced-family screens on sentinel prompts.

Results and decisions:

- Wave 2A Phase-1 trace batch: all effective results were exact; keep ordinary
  `phase1_trace_batch_policy=legacy`. Retain `cap16` only as an optional later
  resource candidate, not a Wave 3 default.
- Wave 2B Phase-4 family: promote `phase4_refresh_policy=deferred_v1` as the
  primary candidate; keep `phase4_row_executor=streaming_v1` as a secondary speed
  interaction candidate; reject `planner_v2` and `refresh_opt_v1`.
- Wave 2C row/encoder/staging/planner: all variants were exact. Promote
  `row_subchunk_size=512` as the primary row/memory candidate and
  `plan_feature_batch_size=true` as the conservative memory candidate.

Wave 3 should combine only a small candidate set:

1. locked stable-resource baseline from Wave 1,
2. `deferred_v1` alone,
3. `row_subchunk_size=512` alone,
4. `plan_feature_batch_size=true` alone,
5. `deferred_v1 + row_subchunk_size=512`,
6. `deferred_v1 + plan_feature_batch_size=true`,
7. optional speed interaction: `deferred_v1 + streaming_v1 + row_subchunk_size=512`.

Keep Cardinal node `c0811` excluded for exact-trace launches unless it is
explicitly being diagnosed.

Structured record:

- `experiments/logs/2026-05.jsonl`

### 2026-05-21 — Sweep Wave 2C row/encoder/staging decision

Wave 2C screened row-store, encoder residency, CPU staging, row-subchunk, and
feature-batch planner variants independently, without combining Wave 2B winners.

Effective result:

- 42/42 successful scenarios, all compared against Wave 0 baselines.
- Every Wave 2C variant preserved exact compact graph agreement against Wave 0
  (`min_weighted_edge_jaccard=1.0`).
- `row_subchunk_size=512` had the best mean runtime ratio (`0.941`) and lower mean
  RSS than legacy (`192.8 GiB` vs `202.5 GiB`), but was mixed by prompt.
- `plan_feature_batch_size=true` had the most conservative memory profile
  (`191.5 GiB` mean RSS) with near-neutral runtime (`0.992` mean ratio).
- Do not promote `row_fadvise`, active CPU encoder residency, active pinned CPU
  encoder residency, or no-CPU-staging as ordinary candidates.

Structured record:

- `experiments/logs/2026-05.jsonl`

### 2026-05-20 — Sweep Wave 0/1/2A baseline and Phase-1 decision

Sweep campaign status:

- Wave 0 established the pinned cross-cluster baseline registry:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/baselines/wave0-baseline-20260520-01.json`.
- Wave 1 locked stable resource settings for later waves:
  - Ascend fast: `batch=128`, `decoder_chunk_size=2048`, cache `0`.
  - Ascend anomaly: `batch=256`, `decoder_chunk_size=4096`, cache `0`.
  - Cardinal fast/anomaly: `decoder_chunk_size=4096`, cache `0`.
- Wave 2A Phase-1 trace-batch screening completed successfully after replacing
  Cardinal `c0811` node failures with reruns excluding that node.

Wave 2A effective result:

- 24/24 successful scenarios, all compared against Wave 0 baselines.
- Minimum feature/edge/weighted-edge Jaccard across effective results: `1.0`.
- `cap16` is the safer optional Phase-1 resource candidate, but the ordinary
  Wave 2B path keeps `phase1_trace_batch_policy=legacy` because Phase-1 caps did
  not produce a consistent cross-prompt speed win.
- Cardinal node `c0811` produced CUDA misaligned/illegal-access/CUBLAS failures
  and should be avoided or reported for exact-trace runs.

Structured record:

- `experiments/logs/2026-05.jsonl`

### 2026-05-20 — Sweep Wave 2B Phase-4 family decision

Wave 2B screened curated Phase-4 scheduler/refresh/ranker/executor variants on
the sentinel prompts while keeping Wave 2A's ordinary `legacy` Phase-1 setting.

Effective result:

- 42/42 successful scenarios, all compared against Wave 0 baselines.
- `planner_v2` caused compact graph drift and is not promoted
  (`min_weighted_edge_jaccard=0.866600`).
- All other Phase-4 variants preserved exact compact graph agreement against Wave
  0 (`min_weighted_edge_jaccard=1.0`).
- Primary Wave 2B promotion candidate: `phase4_refresh_policy=deferred_v1`.
- Secondary candidate for later interaction testing: `phase4_row_executor=streaming_v1`.
- Wave 2C should still screen the row/encoder/staging family independently rather
  than combining Wave 2B winners before Wave 3.

Structured record:

- `experiments/logs/2026-05.jsonl`

### 2026-05-15/16 — Phase-3 row capture/replay fix validated and committed

Project commit:

- `9314f30` (`Record post-consolidation cleanup plan`)

Sibling library commit:

- `e91370a` (`Fix Phase-3 row replay effective state`)

Validation summary:

- local CPU-only library tests passed:
  - `uv run pytest tests/test_attribute_nnsight_telemetry.py -q`
  - `uv run pytest tests/test_phase3_replay_validation.py -q`
  - `uv run pytest tests/test_partial_influences.py -q`
  - targeted `ruff check`
- Ascend SLURM validation completed for `828_base`, `361_base`, `94_base`, and
  the Phase-3 row donor replay smoke.
- Compact comparisons remained exact for canonical scenarios:
  `feature_jaccard=1.0`, `edge_jaccard=1.0`, `weighted_edge_jaccard=1.0`.
- Row donor replay smoke confirmed finite `float64` row denominators around the
  previous overflow boundary.

Relevant scratch roots:

- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/phase1-row-replay-fix-20260515-01`
- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/phase1-row-replay-fix-20260515-01`
- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/phase1-row-replay-smoke-20260515-01`

Structured record:

- `experiments/logs/2026-05.jsonl`

### 2026-05-12 — Track-0B consolidation baseline validated

Project and sibling `main` were consolidated and validated as the working
baseline before post-consolidation cleanup.

Key references:

- same-generation post-consolidation references:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260512_193309_605201_post-consolidation-ascend-validation`
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260512_193309_737038_post-consolidation-ascend-validation`
- pre-consolidation optimization `94_base` control:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260512_201118_810555_pre-consolidation-optimization-94-control`

Structured record:

- `experiments/logs/2026-05.jsonl`

## Where to add future information

| Information type | Destination |
|---|---|
| Current baseline, interpretation, run-family meaning | This file |
| Append-only event/run records | `experiments/logs/YYYY-MM.jsonl` |
| Old long-form narratives | `docs/history/` |
| Current workflow policy | `AGENTS.md` |
| Current exact-bench harness usage | `docs/harness.md` and `src/nlp_research_project/exact_trace_bench/README.md` |
| Current cleanup plan | `docs/current_project_roadmap.md` |
| Durable design/spec tradeoffs | Current specs linked from `docs/README.md` |
