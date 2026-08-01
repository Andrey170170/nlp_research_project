# Experiments inventory

Status: Current compact index and interpretation summary
Last updated: 2026-07-31

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
| Project workspace | `/uufs/chpc.utah.edu/common/home/u1653998/projects/nlp_research_project` |
| Project branch / run commit | `chpc-granite-baseline-setup` / `5cb554a` |
| Sibling library workspace | `/uufs/chpc.utah.edu/common/home/u1653998/projects/circuit-tracer_chunked` |
| Sibling branch / run commit | `chpc-incremental-telemetry` / `3ad7fc9` |
| Editable dependency path | `../circuit-tracer_chunked` |
| Canonical exact-trace dtype | `exact_trace_internal_dtype=fp32` |
| Canonical prompt tiers | `828_base`, `361_base`, and `94_base` in `fast` for new work |
| GemmaScope-2 feature input hook | `mlp.hook_in` (`pre_feedforward_layernorm.output`) for CLT and PLT |
| Scratch root | `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench` |
| New-run placement | Granite plus operational class (`smoke`, `baseline`, `sweep`, `long_trace`, `full_answer`, or `analysis`) |

Baseline preservation notes:

- `828_base` and `361_base` matched same-generation Ascend baselines exactly
  after consolidation.
- `94_base` matched the pre-consolidation optimization-control output exactly;
  mismatch against older Apr-20/21 references predates consolidation.
- The permanent row-L1 overflow fix is part of the validated baseline.
- Within a fixed `decoder_chunk_size`, cache-size changes were exact; changing
  `decoder_chunk_size` can cause small compact-output drift relative to the
  current `c2048` reference.
- Corrected-hook CLT fast baselines were regenerated on 2026-07-06/07 for
  `828_base`, `361_base`, and `94_base` on both Cardinal and Ascend. Treat older
  GemmaScope-2 `hook_resid_mid` CLT/PLT artifacts as contaminated provenance only.
- The first CHPC Gemma-3-12B PLT strict baseline completed successfully on
  Granite H200 from one immutable snapshot:

  | Fixture | Trace duration | Phase 4 | Slurm MaxRSS | Compact graph |
  |---|---:|---:|---:|---:|
  | `828_base` | 20,878.41s | 20,417.06s | 38,719,956K | 8,192 features / 20,000 edges |
  | `361_base` | 23,051.72s | 20,787.43s | 244,429,132K | 8,192 features / 20,000 edges |
  | `94_base` | 22,588.42s | 20,324.72s | 208,527,604K | 8,192 features / 20,000 edges |

  Run root:
  `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/baseline/chpc-baseline-gemma3-12b-20260710-04`.
  Keep the 600G request until the large fixture-dependent RSS spread is
  understood and repeated.

## Current interpretation

The 4B/12B mapped-row campaign ran the remaining optimization lanes in
`reports/2026-07-27_large_model_optimization_results.md`; formal diagnostic
Gates B/C remain unresolved. Provider-owned mapped decoder rows are the
accepted transferable physical mechanism: the 4B compact-strict gate passed
twice with exact signed evidence, while 12B raw decoder-row materialization
accounting fell from 95.316 GB to 0.910 GB with zero post-Phase-0 decoder
loads. The 12B lazy b64/c4096 reference repeated in 586.30s and 563.27s and
passed an exact permutation-insensitive signed compact-graph comparison.

Active-CPU encoder residency repeated at 509.80s and 508.67s with Phase 4 near
359.6s, but its two compact artifacts were only bounded-aligned
(`0.999024/0.988961/1.0/0.989621` feature/all-edge/Top-256/weighted Jaccard).
Keep lazy mapped b64 as the reproducible reference and active CPU only as an
explicitly bounded, placement-frozen candidate. Pinned CPU, b128, b256, and
16,384/65,536 FP32 contraction tiles were slower or failed the Pareto margin.
No default, governor bundle, fidelity scope, or baseline registry changed.

SP4.1 phase-scoped telemetry now passes its formal overhead gate. A reversed
4B exact pair after unified sparse resource sampling and bounded JSONL flushing
measured telemetry-on at 184.783s versus telemetry-off at 185.862s in the
paired mean (-0.581%, interpreted as noise around zero). All four artifacts
were exact; both on runs recorded 9,835 events with zero sink errors and a
maximum 63-event crash-loss window.

SP4.2 checkpoint lifecycle also passes. A verified 171 GiB job-private 4B
transcoder cache produced 34/34 safe release refusals under shared scope and
34/34 issued releases under job-private scope. The private transition reduced
cgroup file charge by 4.426 GiB, caused zero Phase-4 decoder reloads, preserved
the exact signed compact graph, and did not harm full-run warm steady state.
The synchronized three-batch private probe was 7.25% slower, so the lifecycle
is accepted for ownership/reclamation correctness without claiming a repeatable
speedup. See `reports/2026-07-31_exact_trace_sp4_lifecycle_results.md`.

SP1 closes active-CPU encoder placement as a high-value bounded/research path,
not an exact finalist. Direct duplicate-aware host materialization removed the
old approximately 0.98 GiB GPU occurrence table and reduced a matched 12B
Phase 4 from 389.50s to 286.69s (26.4%) and completion from 457.30s to 368.70s
(19.4%). The full graphs nevertheless selected two different features per arm
and executed 129 versus 130 semantic batches: feature/all-edge/weighted-edge
Jaccard was `0.999512/0.999500/0.999529`, signed normalized L1 was `0.000471`,
and the target token and Top-64 through Top-1024 support remained exact.
Persisted probes localized the first numerical difference after byte-exact
Phase-3 feature rows to seed-influence reduction. Keep mapped lazy in the SP5
exact finalist and retain direct active CPU as a default-off bounded option.
See `reports/2026-07-31_exact_trace_sp1_results.md`.

SP5 is fully characterized but not promoted. Selective mapped Phase-0 rows
produced two exact-on-workload 12B `361_base` runs at 471.38s and 470.14s, but
failed held-out 1B `94_base` (feature/edge/weighted Jaccard
`0.999756/0.997004/0.996279`, signed L1 `0.003728`). Canonical full-page Phase
0 restored held-out exactness but took 1,905.57s completion at 12B, including
1,370.02s in Phase 0, and current canonical 4B `361_base` disagreed with the
older mechanism control. Keep selective rows as an explicit bounded opt-in,
make no default or scientific-baseline change, and resolve the source-invariant
Phase-0 reduction boundary before LS1. LS0 itself is complete with frozen
129/256/512/1,024-token development workloads and two 256-token holdouts. See
`reports/2026-07-31_short_prefix_performance_results.md`.

Governor Wave A completed on Granite H200 and was analyzed on 2026-07-19. The
provider-local CLT/PLT references match the original corrected-hook Granite
`361_base` artifacts. CLT has a useful `c10080` fetch candidate at 1.43x warm
speed with effectively exact output; `b1500` consumes 132.98 GiB reserved VRAM
for only 1.05x and `b2000+` OOMs. For PLT, `b256` gives 1.68x at minimum
Jaccard `0.9845`; `c32768` is the fastest measured row above all `0.97`
feature/edge/weighted-edge/Top-256 floors at 4.01x. The `b512/c8192` corner
reaches 5.08x but misses the Top-256 floor at `0.9617`; the 12-16x `b1024+`
corners are not acceptable defaults because broad graph overlap falls as low as
`0.51-0.87`. The full report is under
`exact_trace_bench/granite/analysis/governor_calibration/wave_a_20260716/`.

Wave A/B rows are calibration observations, not a global pass/fail list. They
fit separate compact statistical feasibility/resource, phase-runtime, and
fidelity response models. This does not imply neural training. Exact mode
remains scope-certified; bounded mode enforces explicit
metric floors; best-effort mode assigns fidelity loss a soft objective penalty;
research mode permits labeled extrapolation. Campaign ingestion, fidelity-scope
authorization, and launch-default changes are separate decisions.

Wave C uses a ten-row local causal matrix: nine serialized 4B PLT rows at
`400G/2h` and one 12B PLT row at `600G/8h`. The response-model implementation
is now split at the intended ownership boundary: project artifacts become typed
fit/held-out observations, while the sibling owns model families, uncertainty,
immutable bundles, validation, and staged runtime evaluation. Bundle activation
is explicit and remains off for calibration launches.

Decoder-fetch chunk size is numerically sensitive for PLT: `c32768`
gives 4.01x speed with about 0.25-0.28% broad graph drift. Keep non-reference
chunks out of exact mode unless scope-certified; retain them as bounded,
best-effort, and research calibration evidence.

The isolated 1B PLT performance loop now has a reproducible short-prefix
bounded candidate on Granite H200. `c65536` plus a 16 GiB cross-batch decoder
cache, Phase-1/3 cap 128, Phase-4 execution cap 256, and session 256 completed
`361_base` in 279.89s and 280.58s versus the 2805.70s frozen anchor. Both runs
passed the bounded compact gate with identical comparison metrics (minimum
feature/all-edge/Top-256/weighted Jaccard
`0.994643/0.996008/0.992218/0.995133`, magnitude L1 `0.004879`, exact target
token). This is a 124-token, 1B engineering result, not an exact-mode default,
long-prefix validation, or evidence that the same cache envelope is safe for
4B/12B.

Fused active-row residency supersedes that pre-residency cache/tape family.
Under the current 1B active-row path, c65536 completes in `79.37s` versus
`84.82s` for c4096 and passes bounded parity, but its graph drift is the chunk
regime and it is not exact-equivalent. Cache, tape, gather, frontier, and
prefetch variants did not show additional detectable compact drift beyond their
shared chunk regime; they remain rejected because residency removes their reuse
benefit or because they are slower/more memory-heavy. See
`reports/2026-07-26_bounded_candidate_reassessment.md`.

The first active-row larger-model transfer on Granite H200 established:

- 4B b128/c4096 completed in `335.61s` (`16.30x` versus the frozen baseline),
  with Phase 0 `191.09s`, Phase 4 `113.21s`, and 27,685 MiB framebuffer. It
  failed exact but passed bounded at feature/all-edge/Top-256/weighted Jaccard
  `0.999024/0.998401/1.000000/0.998881`, L1 `0.001119`, token exact.
- 4B b512/c65536 passed bounded but regressed to `418.58s`: Phase 0 improved to
  `147.32s`, Phase 4 worsened to `220.12s`, framebuffer rose to 82,111 MiB, and
  bounded metrics moved near the floor (`0.992460/0.983143/0.992218/0.985026`,
  L1 `0.015087`). Do not select the combined profile.
- 12B b64/c4096 proved 250 GiB capacity viability but not throughput viability.
  Phase 0 completed in `1350.83s` with 3,030 decoder loads / 95.32 GB logical
  bytes; rigid anonymous memory reached 19.46 GiB and HBM stayed around 40.6
  GiB. Phase 3 took `0.39s`, but the first two Phase-4 batches took `187.85s`
  and `120.29s` because model/refresh pages repeatedly faulted under cgroup
  reclaim. The step was stopped before completion, so there is no 12B compact
  parity result.

Paired warm 4B follow-ups separated chunk size from execution batching:

- At c4096, b512 and b128 had identical compact metrics, but the matched b512
  run took `327.32s` / Phase 4 `192.72s` / 82,111 MiB versus b128 `255.72s` /
  `139.58s` / 27,685 MiB. Larger execution batches add no detectable compact
  drift here, but are slower and use roughly 3x framebuffer.
- At b128, c65536 and c4096 were run in both orders. c65536 won both pairs
  (`234.99s` versus `255.72s`, then `221.87s` versus `223.00s`), averaging
  `228.43s` versus `239.36s`, a `4.6%` gain. Its compact metrics were identical
  with and without b512, attributing the measured drift to chunk partitioning.
  Retain c65536 as an explicit dataset-frozen bounded regime, but do not promote
  it: the average gain misses 5% and all-edge Jaccard `0.983143` / L1 `0.015087`
  sit close to the preregistered floors.

In both model sizes the 250 GiB cgroup reached its hard limit almost entirely
through clean file cache while rigid RSS remained small and `memory.failcnt`
stayed zero. This validates the capacity hypothesis, but the 12B result shows
why nominal cache is operationally useful: insufficient cache headroom can turn
refresh into the bottleneck even when no OOM occurs. Guard future accepted-risk
runs by anonymous/RSS growth rather than low total cgroup charge.

Wave B completed all 14 Gemma 3 4B/12B PLT rows on Granite H200. Larger
Phase-4 execution envelopes improved total runtime by `1.52-1.60x` at 4B and
`1.70-2.14x` at 12B, but did not receive cross-model exact certification under
the prepared-frontier contract.
The 4B 256 arm is compact-graph exact to numerical tolerance despite two later
frontier-membership changes; the 12B 128 arm drifts by about 1% on edge metrics,
and the 12B 256 arm is closer but still changes later frontiers. Treat the
validated 1B execution knee as model-scoped, not as a universal physical rung.

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

Near-term implementation focus:

1. backfill Wave A/B into the v2 observation schema and publish the preliminary
   response bundle;
2. run the separate Wave C 4B/12B arrays and dependent finalizer;
3. inspect held-out coverage, prediction error, resource use, and graph parity
   before authorizing any new fidelity scope;
4. consolidate governed harness workflows in Phase F after the calibrated
   solver passes its final Granite gates.

Static Phase-4 coalescing gate (Granite H200, 1B PLT, `361_base`, 2026-07-19):

- execution caps `128/256/512` preserved the same semantic fingerprint, 65
  semantic batches, 17 prepared frontiers, frontier membership and execution
  order, and exact compact feature/edge/top-256 topology;
- weighted-edge Jaccard versus 128 was `0.9999999979` at 256 and
  `0.9999999964` at 512;
- execution calls fell `65 -> 33 -> 17`; total runtime was
  `3125.24s -> 2178.85s -> 2147.83s`, while reserved CUDA memory rose
  `13.30 -> 24.67 -> 46.57 GiB`;
- use 256 as the current 1B PLT efficiency knee. The 512 cap is validated
  headroom, but its additional `1.4%` total speedup does not justify nearly
  doubling reserved CUDA memory for this workload.

Phase C1 closed on July 11 with immutable jobs `1613108`/`1613109`: compact NPZ
artifacts were byte-identical to baseline, live/final telemetry counts matched,
and both sinks closed without errors. The CLT job reached its `32G` host-memory
request and slowed materially; this was accepted as a memory-headroom outlier,
not a clean timing measurement. Request at least `64G` for future 1B CLT gates.

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
- Collapsed `feature<-feature` topology confirms that the retained feature graph is
  much more stable at coarse levels than exact edge identity suggests: adjacent
  layer-flow weighted Jaccard is about `0.57`--`0.65`, and positionless feature-flow
  weighted Jaccard is about `0.41`--`0.49`, versus exact feature-feature edge
  weighted Jaccard near `0.005`--`0.008`.

Full-answer stability follow-up (May 2026):

- Same-cluster full-sequence versus independent-prefix runs should be treated as a
  stability/amplification lab, not a bitwise parity target.
- For `828_base` token 0, tiny Phase-0 drift under full sequence is amplified by
  Phase-3/Phase-4 hard cutoffs into large compact-graph support differences.
- Phase-3-only frontier buffering helps modestly; dynamic Phase-4 frontier
  buffering has a much larger stability effect and should remain an explicit
  robust-mode experiment knob while broader support definitions are evaluated.

Metric-calibrated temporal stability decision (June 2026):

- Phase-1 metric calibration over Stage A--C noise pairs, four existing
  trajectories, and matched null pairs selected a simple primary ruler:
  `logit<-error / union_raw_weight_cosine / {}` for mid/late answer positions.
- The selected metric separates controls in the desired order. Calibration means:
  late null `0.194`, temporal-adjacent `0.602`, noise `0.998`; mid null `0.244`,
  temporal-adjacent `0.588`, noise `0.991`.
- Secondary interpretable readouts are `logit<-error /
  top_p_core_shared_mass_fraction_{a,b}` at `p=0.8`; rank/core checks include
  `logit<-error / top_p_core_jaccard / {"p":0.95}` and derived-K weighted
  Jaccard.
- Near-case feature identity is not the main problem: feature-node raw-weight
  cosine is high at mid/late (`~0.980`/`~0.991`), and decoder-soft matched
  feature mass is near complete (`~0.998`). The brittle object is exact compact
  edge support, especially exact `feature<-feature` edges and early token-0
  support boundaries.
- Default full-answer tracing for the next confirmation/scaling phase should be
  `full_sequence`, unbuffered, typed-bucketed, fp32, with the caveat that the
  first-pass full-sequence implementation needs an efficiency/hardening pass
  before large all-token runs. Do not mix `full_sequence` and
  `independent_prefix` inside one comparison set.
- Keep Phase-3/Phase-4 frontier buffering as an explicit robustness/debug knob,
  not the ordinary default, unless a unified Stage-D rerun shows it materially
  narrows the calibrated mid/late noise band on the frozen primary metric.

Stage-D full-matrix calibration follow-up (June 2026):

- The unified Cardinal Stage-D-style matrix with `window_reuse_v1` and read-side
  row-store fadvise completed cleanly and re-calibrated metrics on 39 sparse
  selected-token pairs.
- Same-token noise anchors are now effectively exact under robust metrics:
  independent-prefix versus full-sequence and late rep-a/rep-b repeats are at
  `1.0` or numerical-equivalent `1.0` for feature-node/readout/core metrics.
- This confirms the operational default: `full_sequence`, `window_reuse_v1`,
  unbuffered, typed-bucketed, fp32, and
  `row_store_cache_control=fadvise_dontneed_after_append_and_read_v1`.
- The modest Phase-4 buffer did not narrow the Stage-D noise band because the
  noise ceiling was already exact; keep it as an explicit robustness/debug arm,
  not the ordinary default.
- The Phase-1 primary `logit<-error / union_raw_weight_cosine / {}` does **not**
  separate this sparse selected-neighbor Stage-D calibration regime: selected
  large-hop temporal pairs fall below the matched cross-fixture null. Do not use
  that metric as the primary for sparse selected-neighbor Stage-D scoring; retest
  it on dense adjacent all-token trajectories before making Phase-3 claims.

Full-sequence telemetry pilot (June 2026):

- Full-sequence prefix views now execute NNSight traces only over the
  prefix-effective token span while retaining full-sequence metadata for audit.
- On Cardinal, large full-sequence selected-token traces completed under the QOS
  cap with `--mem=460G` plus `row_store_cache_control=fadvise_dontneed_after_append_v1`;
  lower memory requests failed or were canceled during hardening.
- For `94_base` and `828_base` selected early/final tokens, final full-sequence
  per-token, full-sequence experimental-reuse, and independent-prefix compact
  graphs matched exactly in the structural comparator (`feature_jaccard=1.0`,
  `edge_jaccard=1.0`, `weighted_edge_jaccard=1.0`).

## Current run families

| Family | Meaning | Current status |
|---|---|---|
| `exact_trace_bench/granite/setup_prefetch` | Model/transcoder download and cache preparation | Current CHPC operational class |
| `exact_trace_bench/granite/smoke` | Minimal load/trace validation | Current CHPC operational class |
| `exact_trace_bench/granite/baseline` | Canonical strict baseline rebuilds, including the 12B `20260710-04` run | Current CHPC operational class |
| `exact_trace_bench/granite/sweep` | Parameter/resource campaigns | Current CHPC operational class |
| `exact_trace_bench/granite/long_trace` | Large-context/high-memory traces | Current CHPC operational class |
| `exact_trace_bench/granite/full_answer` | Multi-token/full-answer campaigns | Current CHPC operational class |
| `exact_trace_bench/granite/analysis` | Extraction, comparison, and plotting | Current CHPC operational class |
| `exact_trace_bench/{ascend,cardinal}/{fast,anomaly,long_eval}` | OSC baselines, parity, and historical evaluation tiers | Historical provenance; do not use for new placement |
| `workspace_snapshots/` | Immutable project + sibling-library launch snapshots | Current provenance mechanism |
| historical `matched_debug` artifacts | Old matched-debug campaign outputs/configs | Historical only; do not use as an ordinary bucket |

## Recent durable decisions

### 2026-07-07 — Phase A2/A3 corrected-regime interim readout

Phase A2 completed corrected-hook CLT fast re-baselines for `828_base`,
`361_base`, and `94_base` on Cardinal (`12242778_[0-2]`) and Ascend
(`6242253_[0-2]`); all six runs succeeded with `feature_input_hook=mlp.hook_in`
and `exact_trace_internal_dtype=fp32`.

Phase A3 was partial, not a final caste decision. The PLT small-affine chunk/cost
matrix succeeded only for the shortest-prefix `t00_828_det_g000` target
(`prefix_token_count=73`) and failed with CUDA OOM for every longer-prefix target
(`prefix_token_count>=424`) across both 1B/4B providers and all requested
chunks/repeats. The failures happened under aggressive memory-for-speed knobs
(large decoder caches, prepared refresh cache, high replay/prefetch, active pinned
encoder residency), so rerun the failed targets with a survival profile before
turning this into a durable governor conclusion.

Instrumentation caveat: the A1 ranker-frontier selection-margin telemetry is
present in ordinary `trace_pipeline_chunked.py` telemetry artifacts, but the A3
full-answer runner originally did not persist generic `telemetry_events` into
`trace_results.jsonl`/`trace.json`. Project commit `620e781` fixed the success
path by writing per-token `telemetry.jsonl` sidecars plus
`telemetry_event_count` / `telemetry_events_path` fields. The subsequent failed
pilot still exited before a compact result was available, so error-path telemetry
persistence remains a follow-up.

Survival pilot result: Cardinal jobs `12255846` (`plt_1b_small`, chunk `8192`)
and `12255847` (`plt_4b_small`, chunk `4096`) reran `t01_361_s1002_g300`
(`prefix_token_count=424`) with lower memory-for-speed knobs
(`cross_batch_decoder_cache_bytes=0`, prepared refresh cache `0`, replay window
`4`, prefetch `2`, lazy encoder residency, no row-store preallocation). Both
failed with CUDA OOM during Phase 1 forward, not host OOM or timeout:

- `12255846`: `FAILED|1:0`, elapsed `00:02:01`, MaxRSS `66124448K`, trace row
  `OutOfMemoryError` allocating `5.59 GiB`; Phase-0 active features `227051`,
  precompute `60.21s`, CUDA peak allocated `91.14 GiB`.
- `12255847`: `FAILED|1:0`, elapsed `00:04:14`, MaxRSS `189595184K`, trace row
  `OutOfMemoryError` allocating `4.14 GiB`; Phase-0 active features `313100`,
  precompute `172.59s`, CUDA peak allocated `89.65 GiB`.

The lower memory-for-speed knobs did reduce noncritical residency/caches but did
not change the Phase-1 source trace batch (`legacy`, effective `1024` for 1B and
`512` for 4B). The survival-v2 pilot then capped Phase-1 source batches on the
same target with `phase1_trace_batch_policy=cap_effective_batches`, using
`phase1_trace_batch_size_max=128` for 1B and `64` for 4B. Cardinal jobs
`12260794` (1B c8192 cap128) and `12260795` (4B c4096 cap64) still failed with
CUDA OOM during Phase 1 forward, but the cap was applied and the new error-path
telemetry persistence was proven on real failed GPU artifacts:

- `12260794`: effective Phase-1 source batch `1024 -> 128`; `FAILED|1:0`,
  elapsed `00:01:55`, MaxRSS `66121944K`, CUDA OOM allocating `5.59 GiB`,
  Phase-0 active features `227051`, precompute `62.45s`, CUDA peak allocated
  `91.14 GiB`; token `telemetry.jsonl` contains 4 events.
- `12260795`: effective Phase-1 source batch `512 -> 64`; `FAILED|1:0`,
  elapsed `00:04:11`, MaxRSS `189633676K`, CUDA OOM allocating `4.14 GiB`,
  Phase-0 active features `313100`, precompute `175.01s`, CUDA peak allocated
  `89.65 GiB`; token `telemetry.jsonl` contains 4 events.

Conclusion: Phase-1 source-batch capping alone is insufficient for this
long-prefix PLT target. Any next survival pilot should reduce the broader
Phase-1 pressure surface, not just source batch size: start with smaller source
caps plus lower feature/logit/attribution batch sizes, or wait for governor
admission estimates before relaunching a broad A3 matrix.

Follow-up launch: after committing the error-path telemetry and Phase-1 cap
plumbing fixes, submitted a less-drastic all-main-batch pilot on Cardinal:
`12263597` (1B c8192, source/feature/logit/attribution batches all `256`) and
`12263598` (4B c4096, all main batches `128`). These use immutable snapshots of
project commit `5d855de` and sibling commit `690fc04`.

Outcome (2026-07-08): both survival-v3 all-main-batch jobs completed and saved
compact graphs for the failed long-prefix `t01_361_s1002_g300` target:

- `12263597`: `COMPLETED|0:0`, elapsed `00:15:05`, MaxRSS `79193196K`,
  trace `ok`, trace time `857.36s`, telemetry events `4623`, graph artifact
  `9.80 MiB`. Effective trace capacity was `256`, bound equally by source,
  feature, and logit batches. Phase-0 active features `227051`, precompute
  `57.63s`, Phase-1 forward `1.23s`, forward CUDA peak allocated/reserved
  `79.69/85.90 GiB`.
- `12263598`: `COMPLETED|0:0`, elapsed `01:34:05`, MaxRSS `211938236K`,
  trace `ok`, trace time `5557.93s`, telemetry events `16551`, graph artifact
  `9.92 MiB`. Effective trace capacity was `128`, bound equally by source,
  feature, and logit batches. Phase-0 active features `313100`, precompute
  `170.30s`, Phase-1 forward `1.44s`, forward CUDA peak allocated/reserved
  `65.30/75.35 GiB`.

Runtime emphasis: Phase 1 is an admission/survival gate, not the performance
bottleneck in these completed runs. Phase 3 + Phase 4 dominate wall time:

- 1B b256: Phase-3 logit attribution `44.63s` and Phase-4 feature attribution
  `744.66s`, about `92%` of trace time together; Phase 1 was only `1.23s`.
- 4B b128: Phase-3 logit attribution `147.25s` and Phase-4 feature attribution
  `5220.87s`, about `97%` of trace time together; Phase 1 was only `1.44s`.

During Phase 3/4, sampled CUDA allocated memory stayed near `23.0 GiB` for 1B
and `33.2 GiB` for 4B while PyTorch retained the earlier forward-pass reserve.
For governor/performance work, optimize Phase-3/4 throughput and refresh cadence;
use Phase-1 peaks mainly as an admission constraint.

Conclusion update: coherent all-main-batch reduction can make this long-prefix
PLT target survive where legacy and source-only caps OOMed, but the actionable
performance target is Phase 3/4, especially Phase 4. The batch-coupling
interpretation is still supported: `phase1_trace_batch_size_max` only reduced the
NNSight trace-capacity reserve once feature/logit/attribution batches were
lowered together. This is a survival and throughput-tuning result, not broad PLT
chunk-sensitivity evidence; use Phase-3/4 timing, refresh count, and memory
samples to choose an upward bracket or bounded relaunch before any broad A3
matrix rerun.

Interim decision:

- Keep `decoder_chunk_size` scenario-pinned for the next rerun wave. Do not let
  the governor auto-change it as a purely performance/VRAM lever until a broader
  low-memory corrected-regime rerun proves tolerance-stable behavior.
- A3 is insufficient to answer broad PLT FP sensitivity. The completed
  shortest-prefix chunk sweep and the single survival-v3 long-prefix success are
  useful as warning/survival signals, not a final verdict: larger chunks
  substantially reduce wall time and compact artifacts differ across chunk sizes,
  but broad long-prefix tolerance evidence is still missing.

A4 launch (2026-07-08): after committing the A3 outcome/A4 plan at project
`700b408` with sibling `690fc04` clean, submitted a bounded first-pass
frontier/caste sweep from immutable snapshots. This is not a memory-fitting
sweep; it tests whether candidate governor knobs move compact graph/frontier
membership for the long-prefix 1B PLT target.

- `12275056` Cardinal: decoder-chunk axis, c4096 vs existing c8192/b256 anchor.
- `12275057` Cardinal: coupled batch/refresh axis, c8192 all main batches `128`
  vs existing c8192/b256 anchor.
- `6260319` Ascend: cluster axis, c8192/b256 vs existing Cardinal anchor.
- `12275058` Cardinal: tiny full-sequence multi-token `per_token` control for
  generated indices `298,299,300`.
- `12275061` Cardinal: matching tiny-window `window_reuse_v1` run with
  Phase-0 window-state reuse and target-logit reuse enabled.

A4 Cardinal readout (reported 2026-07-09; OSC artifacts pending transfer to
CHPC): the completed Cardinal comparisons are sufficient to unblock the scoped
knob-caste design. The delayed Ascend job `6260319` is now optional
cross-environment reproducibility evidence rather than a Phase-B gate.

| Pair | Feature J | Edge J | Weighted Edge J | Top64 | Top256 | Top1024 | Top5000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| chunk c4096/b256 vs A3 c8192/b256 | 0.994643 | 0.988269 | 0.987517 | 1.000 | 0.996 | 0.993 | 0.992 |
| batch c8192/b128 vs A3 c8192/b256 | 0.990040 | 0.987479 | 0.984281 | 0.969 | 0.984 | 0.989 | 0.991 |
| per-token g300 vs A3 | 0.998292 | 0.992627 | 0.992163 | 1.000 | 1.000 | 0.995 | 0.995 |
| window-reuse g300 vs A3 | 1.000000 | 1.000000 | 1.000000 | 1.000 | 1.000 | 1.000 | 1.000 |
| window-reuse vs per-token g298 | 1.000000 | 1.000000 | 1.000000 | 1.000 | 1.000 | 1.000 | 1.000 |
| window-reuse vs per-token g299 | 0.984737 | 0.973165 | 0.976543 | 1.000 | 0.992 | 0.992 | 0.991 |
| window-reuse vs per-token g300 | 0.998292 | 0.992627 | 0.992163 | 1.000 | 1.000 | 0.995 | 0.995 |

Validated-under scope: corrected-hook Gemma-3-1B with GemmaScope2 PLT-small,
Cardinal, fp32, long-prefix `t01_361_s1002_g300` and full-sequence window indices
`298--300`, using the listed chunk/batch settings. This does not validate 4B,
12B, CLT, other prompts, or Granite/H200.

Decision:

- strict mode keeps decoder reduction order, coupled batch/refresh semantics,
  and session mode scenario-pinned;
- decoder chunk variation is eligible for an explicit validated-relaxed policy
  in this narrow regime because the Top64 core was exact and weighted drift was
  about 1.25%;
- coupled batch/refresh variation needs a separate, stronger permission because
  it changed the Top64 set (`0.969`) and had about 1.57% weighted drift;
- `window_reuse_v1` remains the canonical full-answer session mode, not a memory
  governor choice; its g300 output exactly matched the A3 anchor;
- future implementation should decouple logical reduction/refresh checkpoints
  from physical fetch chunks and compute microbatches so strict mode can recover
  more performance without accepting semantic drift.

Key roots:

- A2 Cardinal:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/phase-a2-corrected-clt-fast-cardinal-20260706`
- A2 Ascend:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/fast/phase-a2-corrected-clt-fast-ascend-20260706`
- A3 manifest:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/manual_scenarios/phase-a3-fp-cost-plt-small-affine-20260706/manifest.json`
- A3 output base:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/long_eval/phase-a3-fp-cost-plt-small-affine-20260706`
- A3 survival pilot manifest:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/manual_scenarios/phase-a3-survival-telemetry-lowmem-mem250-20260707/manifest.json`
- A3 survival pilot output base:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/long_eval/phase-a3-survival-telemetry-lowmem-mem250-20260707`
- A3 survival-v2 phase1cap manifest:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/manual_scenarios/phase-a3-survival-v2-phase1cap-20260707/manifest.json`
- A3 survival-v2 phase1cap output base:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/long_eval/phase-a3-survival-v2-phase1cap-20260707`
- A3 survival-v3 allbatch manifest:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/manual_scenarios/phase-a3-survival-v3-allbatch-lessdrastic-20260707/manifest.json`
- A4 frontier/caste manifest:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/manual_scenarios/phase-a4-frontier-caste-v1-20260708/manifest.json`

### 2026-07-02 — GemmaScope-2 CLT/PLT hook correction

GemmaScope-2 CLT and PLT checkpoint configs train on
`pre_feedforward_layernorm.output`, not the pre-layernorm residual input. The
harness defaults now use `feature_input_hook=mlp.hook_in` for GemmaScope-2 CLT
and PLT providers while keeping `feature_output_hook=hook_mlp_out`.

Decision:

- Treat prior `hook_resid_mid` GemmaScope-2 CLT/PLT exact-trace runs as a legacy
  baseline using the wrong input site.
- Use corrected-hook CLT/PLT active-count probes launched on 2026-07-02 as the
  baseline migration evidence before larger PLT feasibility conclusions.

### 2026-06-15 — Stage-D full matrix metric calibration

Metric calibration completed over the Stage-D-style selected-token full matrix.

Key outputs:

- matrix root:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/native-instability-stage-d-full-matrix-read-fadvise-v1`,
- analysis root:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/native-instability-stage-d-full-matrix-read-fadvise-v1/analysis_full_matrix/metric_calibration_v1`,
- pair count: `39` (`24` noise, `6` sparse selected-neighbor temporal, `9`
  null), metric rows: `25545`, scorecard rows: `1965`.

Decision:

- Adopt `full_sequence` + `trajectory_session_mode=window_reuse_v1` with
  `reuse_phase0_window_state=true`, `reuse_target_logits=true`, typed-bucketed
  saves, fp32 internal dtype, and read-side row-store fadvise as the next
  operational tracing default.
- Keep ordinary runs unbuffered. The `phase4buf01_128_2048` arm showed no
  measurable noise-band improvement in this matrix because same-token noise was
  already exact on robust metrics.
- For sparse selected-neighbor Stage-D scoring, prefer feature/readout/core
  metrics that still separate null < selected-neighbor < noise. Do not promote
  the older `logit<-error / union_raw_weight_cosine / {}` primary for this sparse
  regime.

Selected Stage-D scorecard means:

| Bucket/metric | Band | Null | Selected-neighbor temporal | Noise |
|---|---:|---:|---:|---:|
| `feature_nodes / union_raw_weight_cosine` | mid | 0.547 | 0.764 | 1.000 |
| `feature_nodes / union_raw_weight_cosine` | late | 0.619 | 0.900 | 1.000 |
| `logit<-feature / union_raw_weight_cosine` | mid | 0.532 | 0.717 | 1.000 |
| `logit<-feature / union_raw_weight_cosine` | late | 0.604 | 0.846 | 1.000 |
| `all_edges / derived_k_weighted_jaccard / {"K":2048}` | mid | 0.411 | 0.636 | 1.000 |
| `all_edges / derived_k_weighted_jaccard / {"K":2048}` | late | 0.495 | 0.830 | 1.000 |

Caveat:

- This manifest's temporal anchors are sparse selected-neighbor hops, not dense
  adjacent all-token trajectories. The older Phase-1 primary should be retested
  in the dense adjacent setting before being retired for full trajectory work.

### 2026-06-14 — Full-sequence window/session reuse validation

The `window_reuse_v1` validation matrix completed on Cardinal for selected
`94_base` and `828_base` early/late targets.

Key output root:

- `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/fullseq-window-reuse-v1`

Decision:

- Use `trajectory_session_mode=window_reuse_v1` with
  `reuse_phase0_window_state=true` and `reuse_target_logits=true` as the next
  production full-sequence mode for Stage-D-style confirmation runs.
- Keep `phase0_window_scope=shard_window` and
  `phase0_window_reference_checks=off` until trajectory-wide sessions and live
  sampled/all reference checks are implemented.
- Validation showed exact agreement for resource-session-only and logit-only
  controls. Combined Phase-0/logit reuse introduced only tiny fixed-configuration
  early-token drift under calibrated metrics (worst checked
  `logit<-error / union_raw_weight_cosine / {}` was `~0.9999966`) with no
  prefix-audit failures.
- Runtime improvement was modest on the selected two-token matrix (about
  `~6%` for combined mode), but this mode is the right operational default for
  dense full-sequence windows where setup amortization can compound.

### 2026-06-14 — Full-sequence telemetry/resource pilot

The first selected-token full-sequence telemetry matrix completed on Cardinal for
`94_base` and `828_base` after prefix-effective trace hardening and row-store
cache-control pass-through.

Key output root:

- `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/fullseq-telemetry-v1`

Final successful fadvise rerun roots use run id:

- `rerun_prefix-trace-fix2_mem460_fadvise_20260614`

Successful final jobs:

| Fixture/mode | Job | Elapsed | ReqMem | MaxRSS | Shard seconds | Max token seconds |
|---|---:|---:|---:|---:|---:|---:|
| `94_base` `full_sequence_per_token` | `11569735` | `00:28:40` | `460G` | `482391168K` | `1706.2` | `1310.3` |
| `94_base` `full_sequence_experimental_reuse` | `11569736` | `00:25:07` | `460G` | `482379748K` | `1491.5` | `1184.3` |
| `828_base` `full_sequence_per_token` | `11569737` | `00:29:49` | `460G` | `482352784K` | `1776.7` | `1448.0` |
| `828_base` `full_sequence_experimental_reuse` | `11569738` | `00:30:43` | `460G` | `482353536K` | `1802.5` | `1480.9` |

Control independent-prefix jobs also completed:

| Fixture/mode | Job | Elapsed | ReqMem | MaxRSS | Shard seconds | Max token seconds |
|---|---:|---:|---:|---:|---:|---:|
| `94_base` `independent_prefix_per_token` | `11561805` | `01:42:04` | `360G` | `377482464K` | `6096.1` | `4844.0` |
| `828_base` `independent_prefix_per_token` | `11561802` | `01:48:05` | `360G` | `377482708K` | `6434.1` | `5407.8` |

Validation summary:

- All final full-sequence shards completed with `failed_token_count=0` and
  `retry_recommended=false`.
- Sibling-library prefix-effective trace hardening was committed as `2bf8c28`
  (`Trace full-sequence prefix views over prefixes`).
- `row_store_cache_control` was recorded as
  `fadvise_dontneed_after_append_v1` on final full-sequence traces.
- Experimental reuse was effective for reuse-mode traces (`session_reuse_effective`
  and `decoder_cache_reuse_effective` true) and produced exact compact-graph
  agreement with per-token full-sequence traces for all four checked tokens.
- Independent-prefix versus full-sequence structural comparisons also matched
  exactly for the checked `94_base` token indices `0`/`220` and `828_base` token
  indices `0`/`227`.
- Cardinal resource constraint found during hardening: `460G` was accepted and
  completed; `470G+` and exclusive-node memory requests were rejected by QOS, and
  `180G`/`360G` attempts were insufficient for this matrix.

Operational decision:

- Keep `full_sequence`, unbuffered, typed-bucketed, fp32 as the next default, and
  use prefix-effective tracing plus row-store fadvise cache control for large
  Cardinal selected-token/fullseq runs under the current QOS cap.

### 2026-06-13 — Phase-1 calibrated temporal graph metrics

Phase-1 calibration completed on existing Stage A--C selected-token artifacts and
the four typed-bucketed full-answer trajectories. The calibration compared three
control families: same-token perturbed/noise pairs, within-trajectory temporal
pairs, and matched null pairs.

Key outputs:

- analysis root:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/fast/metric-calibration-v1`,
- report root: `reports/metric-calibration-v1/`,
- pair count: `502`, metric rows: `328810`, scorecard rows: `1965`,
- decoder signature cache:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/decoder_signature_cache/gemma-scope-2-1b-it__clt_width_262k_l0_medium_affine__float32_chunk512`.

Decision:

- Freeze `logit<-error / union_raw_weight_cosine / {}` as the primary temporal
  stability ruler for mid/late answer positions.
- Interpret scores against the calibrated anchors rather than as absolute truth:
  near noise means reproducible/same-core, near temporal-adjacent means normal
  within-answer evolution, and near null means weak temporal relation or major
  divergence.
- Treat this as metric/instrument calibration only. General temporal-relation
  claims require applying the frozen metric to held-out prompts/trajectories.

Selected calibration means:

| Bucket/metric | Band | Null | Temporal adjacent | Noise |
|---|---:|---:|---:|---:|
| `logit<-error / union_raw_weight_cosine` | mid | 0.244 | 0.588 | 0.991 |
| `logit<-error / union_raw_weight_cosine` | late | 0.194 | 0.602 | 0.998 |

Near-case interpretation:

- Full-sequence versus independent-prefix and cross-cluster near cases preserve a
  strong feature/readout core. Mid/late feature-node raw-weight cosine is about
  `0.980`/`0.991`, and decoder-soft matched feature mass is about `0.998`.
- Exact compact support differences are therefore mostly boundary/selection
  amplification rather than wholesale semantic-feature replacement.

Operational default for the next confirmation/scaling step:

- Use `full_sequence`, unbuffered, typed-bucketed saves, fp32 internal dtype.
- Keep early positions reliability-caveated.
- Keep Phase-3/Phase-4 buffering as a robustness/debug knob, not the default,
  unless a unified Stage-D rerun proves a meaningful reduction in the frozen
  primary metric's mid/late noise band.
- Before large all-token full-answer runs, harden/optimize the first-pass
  full-sequence implementation; the instability campaign validated it enough for
  selected-token comparisons but did not finish throughput engineering.

### 2026-05-30 — Typed-bucketed full-answer temporal comparison

The typed-bucketed all-token reruns and temporal analyses completed for the four
current comparison traces:

1. deterministic temp0 correct `828_base`,
2. sampled temp0.8 correct `828_base` seed2003,
3. sampled temp0.8 wrong `828_base` seed1002,
4. sampled temp0.8 wrong `361_base` seed1002.

Key output root:

- `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/fast/typed-bucketed-full-reruns-20260527`

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

Collapsed topology follow-up:

- implementation commit: `b2bc97c` (`Add collapsed feature topology metrics`),
  with `5ed2d82` caching collapsed flows for full-run analysis,
- result summary files:
  `deep_temporal_comparison_v1/collapsed_topology_summary.{json,csv}`.

Mean adjacent collapsed `feature<-feature` topology metrics:

| Run | Exact F<-F edge J | Collapsed layer-flow J | Collapsed positionless-flow J |
|---|---:|---:|---:|
| `828` temp0 correct | 0.0054 | 0.5817 | 0.4142 |
| `828` temp0.8 correct | 0.0063 | 0.5685 | 0.4175 |
| `828` temp0.8 wrong | 0.0074 | 0.6418 | 0.4556 |
| `361` temp0.8 wrong | 0.0084 | 0.6537 | 0.4909 |

Interpretation: exact feature-to-feature edge IDs churn almost completely, but the
mass distribution over source/target layers and position-collapsed feature pairs is
moderately stable. The wrong `361` trajectory remains the most stable. The sampled
wrong `828` is somewhat more stable than the sampled correct `828` under these
collapsed feature-feature metrics, but this still does not establish a general
correct-vs-wrong split from the current four traces.

### 2026-05-29 — Full-answer dynamic Phase-4 buffering result

A2 dynamic Phase-4 frontier-buffer runs completed on Cardinal for `828_base`
token 0, comparing independent-prefix and full-sequence traces while excluding
Cardinal node `c0811`.

Commits:

- project: `241b102` (`Plumb Phase 4 frontier buffer knobs`),
- sibling library: `3bc2056` (`Add dynamic Phase 4 frontier buffer`).

Validation:

- six one-token jobs completed successfully (`11100861`--`11100866`),
- all aggregates reported `status_counts={"ok": 1}`,
- all prefix/future-position audits passed with `future_position_violations=0`.

Stability trend versus unbuffered and Phase-3-only buffering:

| Run | Features (ind/fullseq) | Feature Jaccard | Edge Jaccard | Weighted edge Jaccard | All-edge weighted |
|---|---:|---:|---:|---:|---:|
| Unbuffered | 8192 / 8192 | 0.6069 | 0.2252 | 0.1922 | 0.5723 |
| Phase3 buffer 5% | 8908 / 8999 | 0.6147 | 0.2541 | 0.2140 | 0.5785 |
| Phase4 1%, 64/1024 cap | 9932 / 10023 | 0.6277 | 0.2880 | 0.2410 | 0.5880 |
| Phase4 1%, 128/2048 cap | 10956 / 11047 | 0.6394 | 0.3582 | 0.2835 | 0.5965 |
| Phase4 5%, 700/8192 cap | 17100 / 17191 | 0.6656 | 0.5240 | 0.3900 | 0.6207 |

Interpretation: dynamic Phase-4 boundary buffering has a substantially larger
effect than Phase-3-only buffering. The moderate `1%, 128/2048` setting gives a
useful cost/stability tradeoff; the `5%, 700/8192` setting gives the best
stability but approaches one-hour runtime for this single-token pair. Keep these
as robust-mode experiment knobs rather than ordinary exact-baseline defaults.

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
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/full-answer-828-prefix-view-validation-20260527`,
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
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/full-answer-828-smoke-perf-default-20260526`,
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
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/full-answer-828-baseline-params-20260526`,
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
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/20260525_broader-default-validation-fast`,
- snapshot root:
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260525_190714_broader_default_validation`,
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
  `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/baselines/wave0-baseline-20260520-01.json`.
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

- `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/fast/phase1-row-replay-fix-20260515-01`
- `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/anomaly/phase1-row-replay-fix-20260515-01`
- `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/anomaly/phase1-row-replay-smoke-20260515-01`

Structured record:

- `experiments/logs/2026-05.jsonl`

### 2026-05-12 — Track-0B consolidation baseline validated

Project and sibling `main` were consolidated and validated as the working
baseline before post-consolidation cleanup.

Key references:

- same-generation post-consolidation references:
  - `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/fast/20260512_193309_605201_post-consolidation-ascend-validation`
  - `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/anomaly/20260512_193309_737038_post-consolidation-ascend-validation`
- pre-consolidation optimization `94_base` control:
  - `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/anomaly/20260512_201118_810555_pre-consolidation-optimization-94-control`

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
