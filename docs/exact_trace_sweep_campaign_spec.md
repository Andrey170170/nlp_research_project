# Exact-Trace Sweep Campaign Spec

Status: Current sweep campaign spec
Last updated: 2026-05-20

## 1. Problem statement

Knob taxonomy, light cleanup, and sanity tests are done. The clean-upstream
comparison showed the remaining divergence is rooted in NNSight/replacement
machinery, so strict exact clean/hardware parity is not the target anymore.

Next objective: characterize stability and resource scaling of exact-trace
library internals across prompts, knobs, and hardware. Use Cardinal fully,
including larger settings that Ascend may not fit.

Assumptions:

- Wave 0 creates the pinned baseline used by later waves.
- Sweep runs stay inside the existing scratch layout and provenance scheme.

## 2. Scope / non-goals

### In scope

1. Expand prompt coverage and establish per-hardware Wave 0 baselines/timing
   references.
2. Sweep stable resource knobs and then advanced knob families.
3. Add baseline-registry and self-scoring support to the harness.
4. Rank scenarios by concatenating per-scenario CSV outputs.
5. Record wave provenance and promotion decisions in docs/logs.

### Non-goals

1. Do not chase strict Ascend/Cardinal exact parity.
2. Do not run a full factorial across all knobs.
3. Do not introduce new scratch buckets beyond the current layout.

## 3. Proposed approach

### 3.1 Harness changes before large sweeps

Implement these first:

1. Prompt expansion/scenario builder.
2. Baseline registry backed by pinned artifacts only; no latest lookup.
3. Self-scoring jobs.
4. Scenario-level `baseline_check` opt-in with mode metrics/gate referencing the
   registry.
5. Post-trace comparison against the Wave 0 baseline using existing
   `compare_artifact_dirs`, writing:
   - `baseline_compare.json`
   - `scenario_metrics.json`
   - `scenario_metrics.csv`
6. Per-scenario CSV emission so later ranking jobs can concatenate them.

#### Baseline registry contract

Use a pinned JSON registry, not a floating "latest" lookup. Each entry should
record at least:

- registry key, cluster, tier, fixture name/kind, prompt identity,
- `scenario_root`, `artifacts_dir`, and `result_json`,
- expected successful status,
- project repo commit/dirty state,
- sibling `../circuit-tracer_chunked` commit/dirty state,
- comparison contract fields such as dtype, method, completions, temperature,
  `max_steps`, `max_feature_nodes`, `max_edges`, `max_n_logits`, and
  `desired_logit_prob`.

Suggested key pattern:

```text
wave0/<fixture_name>/<cluster>/<tier>/<profile>
```

Example profile names: `fp32_default`, `fp32_matched_hw_c4096_cache0g`,
`fp32_cardinal_large_cache`.

#### Scenario `baseline_check` contract

Sweep scenarios may opt into self-scoring with:

```json
"baseline_check": {
  "enabled": true,
  "mode": "metrics",
  "registry_key": "wave0/828_base/ascend/fast/fp32_default",
  "baseline_required": true,
  "thresholds": null
}
```

Modes:

- `metrics`: compute comparison and write metrics, but do not fail the scenario.
- `gate`: compute comparison, apply thresholds, and record pass/fail.
- absent/disabled: preserve current runner behavior.

Each self-scored scenario should emit:

```text
scenario.json
run.log
result.json
baseline_compare.json
scenario_metrics.json
scenario_metrics.csv
artifacts/
```

The CSV should include scenario identity, knob values, status, duration,
Phase-3/Phase-4 timings, memory, cache diagnostics, baseline key/status, and
feature/edge/weighted-edge Jaccard against the pinned baseline.

### 3.2 Wave 0 — baseline creation

Build the baseline set from the start with:

- the current canonical 3 base prompts,
- the current 3 late prompts,
- ~10-12 new GSM base prompts,
- ~4-6 late-prefix variants.

For each hardware cluster, record baseline artifacts and timing references. For
noise-floor measurement, repeat only the canonical 3 prompts per hardware.

Approximate first-pass size:

| Component | Per cluster | Clusters | Scenario-runs |
|---|---:|---:|---:|
| Canonical base repeat set: `828_base`, `361_base`, `94_base` × 2 | 6 | 2 | 12 |
| Current late fixtures: `828_late`, `361_late`, `94_late` | 3 | 2 | 6 |
| New GSM base prompts | 10-12 | 2 | 20-24 |
| New late-prefix subset | 4-6 | 2 | 8-12 |

Expected total: about 46-54 scenario-runs. This is acceptable because it is the
baseline corpus reused by later waves.

Prompt selection should cover:

- prompt lengths near and above the current fixtures,
- shorter and longer reasoning chains,
- arithmetic/numeric edge cases,
- examples whose greedy completions have enough length to support late-prefix
  fixtures.

When comparing hardware, use matched settings. Outside hardware comparisons,
allow hardware-specific optimal settings.

Scratch layout remains:

`/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/{ascend,cardinal}/{fast,anomaly,long_eval}/<run_id>`

Use `run_id`, `run_name`, `run_description`, `run_goal`, and scenario names only;
do not add new buckets.

### 3.3 Wave 1 — stable resource knobs

Fix `exact_trace_internal_dtype=fp32`.

Sweep `decoder_chunk_size` and `cross_batch_decoder_cache_bytes` on sentinel
prompts on both Ascend and Cardinal. Cardinal should also test larger
scales/caches/batches that Ascend may not fit.

Sentinel prompts:

- `828_base`,
- `361_base`,
- `94_base`.

Initial matched grid:

- `decoder_chunk_size`: `1024`, `2048`, `4096`,
- `cross_batch_decoder_cache_bytes`: `0`, `8 GiB`.

Approximate matched size: `3 prompts × 3 chunks × 2 cache settings × 2 clusters`
= 36 scenario-runs.

Cardinal-only extensions may include larger cache, larger batch, or larger chunk
settings when they are useful for scaling characterization but not feasible on
Ascend. Mark these profiles as Cardinal-only; do not treat them as hardware
parity checks.

Measure:

- success / OOM / NaN,
- walltime,
- Phase-3 and Phase-4 timings,
- peak memory / RSS,
- cache diagnostics,
- feature / edge / weighted-edge Jaccard vs baseline,
- prompt-specific regressions.

### 3.4 Wave 2 — advanced knob-family screening

Screen families with curated bundles, not a full factorial:

- Phase-1 trace batch,
- Phase-4 scheduler / refresh / ranker / executor,
- row / encoder / staging / planner family.

Use sentinel prompts only. Promote only winners to anomaly and broader prompt
coverage.

Candidate bundles:

| Family | Example variants |
|---|---|
| Phase-1 trace batch | current/legacy, capped effective batches, small cap/singleton-like diagnostic |
| Phase-4 scheduler/refresh/ranker/executor | current, `planner_v1`, optional `planner_v2`, `deferred_v1`, `topk_v1`, `refresh_optimization=v1`, `streaming_v1` |
| Row/encoder/staging/planner | current, row cache-control, active CPU encoder residency, pinned CPU residency, row-subchunk/staging/planner variants |

Run each family separately. Do not cross Phase-1 × Phase-4 × row/planner until
Wave 3.

### 3.5 Wave 3 — interaction confirmation

Combine only winners:

- baseline,
- best stable,
- best Phase-1,
- best Phase-4,
- best row / memory,
- partial combos,
- full finalist if safe.

Test canonical prompts first, then a late/broader subset.

Do not keep adding compensating knobs if a combined config regresses. Prefer the
smallest partial combination that preserves stability and delivers a measurable
resource win.

### 3.6 Wave 4 — prompt generalization / finalist validation

Because Wave 0 created the prompt-diverse baseline, validate only:

- baseline,
- conservative finalist,
- speed finalist.

Use the broad prompt set.

Because broad baselines already exist from Wave 0, this wave should stay small:
only baseline, one conservative finalist, and one speed/resource finalist cross
the broad prompt set.

### 3.7 Wave 5 — hardware/resource confirmation

Use Ascend and Cardinal to understand scaling and fragility, not to enforce
strict hardware parity. Compare hardware effects only with matched scenarios;
otherwise use Cardinal for larger feasible settings.

For matched hardware effects, keep batch, chunk, cache, dtype, and scenario
fixtures fixed. For resource scaling, explicitly label Cardinal-only profiles and
compare them against Cardinal Wave 0 baselines rather than Ascend references.

### 3.8 Metrics and promotion gates

Wave 0 defines the noise floor for thresholds.

Promotion requires:

- no failures,
- no OOM / NaN,
- no unexplained prompt-specific regressions beyond Wave 0 noise,
- a meaningful stability or resource gain on the target metric.

Recommended promotion shape:

- Wave 1 promotes at most two stable-resource configs.
- Each Wave 2 family promotes at most one or two configs.
- Wave 3 promotes at most one conservative finalist and one speed/resource
  finalist.
- Wave 4/5 validate finalists; they do not expand the candidate set unless all
  finalists fail.

## 4. Acceptance criteria

1. Expanded baseline artifacts and pinned registry exist for both clusters.
2. Self-scoring outputs are emitted for each sweep scenario.
3. Per-scenario CSVs can be concatenated into ranking tables.
4. At most 1 conservative finalist and 1 speed finalist are promoted.
5. Docs and experiment provenance record the wave decisions and thresholds.

## 5. Risks and open questions

1. Prompt selection criteria for the GSM base and late-prefix sets.
2. Baseline storage longevity and pinning policy.
3. Threshold selection after Wave 0 noise-floor measurement.
4. Self-scoring overhead.
5. Cardinal-only settings that are not meaningfully comparable to Ascend.
