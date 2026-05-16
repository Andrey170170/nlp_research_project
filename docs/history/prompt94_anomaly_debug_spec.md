# Prompt 94 Anomaly Debug Spec

## 1. Problem statement

`94_base` is a correctness and stability watchpoint.

Current evidence shows:

- nearly identical Phase-0 active feature counts across clusters,
- but drastically different Phase-4 runtime,
- and strong evidence that Phase-4 selected node sets diverge despite similar
  starting sparse state.

This suggests that small early differences in Phase-4 ranking may be amplified
 by repeated frontier refreshes.

We want a diagnostic mode that maximizes information from a single run while
preserving baseline execution behavior.

## 2. Scope / non-goals

### Scope

- exact chunked Phase-4 anomaly diagnostics for `94_base`
- shadow/debug-only analysis of frontier selection behavior
- per-refresh artifacts that explain where divergence may appear

### Non-goals

- no behavior change to the actual traced result in debug mode
- no queue-heuristic optimization in this phase
- no attempt to solve the anomaly before first collecting stronger evidence

## 3. Proposed approach

Add a dedicated Phase-4 anomaly debug mode that records shadow diagnostics at
every frontier refresh.

Suggested flag:

- `--phase4-anomaly-debug`

Suggested artifacts per completion:

- `phase4_anomaly_debug.json`
- optionally `phase4_refreshes.jsonl`

## 4. Diagnostic package

### 4.1 Refresh snapshots

At every Phase-4 queue refresh, record:

- `refresh_index`
- `n_visited`
- `queue_size`
- selected frontier size
- compact signature / hash of selected `pending`
- overlap with previous refresh
- overlap with first refresh

Purpose:

- test how path-dependent the frontier evolution is

### 4.2 Cutoff / tie diagnostics

At every refresh, around the frontier cutoff `K`, record:

- influence at rank `K`
- influence at rank `K+1`
- cutoff margin
- local summary around the cutoff window (e.g. `K-8..K+8`)
- count of exact equal values near the cutoff
- count of values within epsilon of the cutoff

Purpose:

- test whether small numeric perturbations could flip membership at the cutoff

### 4.3 Shadow deterministic ranking

Without changing execution behavior, compute a shadow deterministic ranking
using a stable tie-break, such as:

- `(-influence, feature_index)`

or, if useful:

- `(-influence, layer, chunk_id, position, feature_index)`

Record:

- top-K overlap with actual selection
- first differing rank
- count of changed selected nodes

Purpose:

- test whether nondeterministic/implicit tie ordering is a likely cause

### 4.4 Shadow float64 ranking

For early refreshes only by default, recompute ranking input in float64 on CPU
and compare against the normal float32 ranking.

Record:

- top-K overlap
- first differing rank
- cutoff margin differences

Default recommendation:

- first refresh only, or first 1-2 refreshes

Purpose:

- test whether ranking is highly precision-sensitive

### 4.5 Frozen-frontier comparison

Capture the first Phase-4 frontier as a frozen reference set.

At later refreshes, record:

- overlap between the current frontier and the frozen first frontier
- aggregate drift across refreshes

Purpose:

- separate initial ranking instability from later refresh amplification

### 4.6 Environment fingerprint

Persist:

- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- torch version
- CUDA version
- relevant module `__file__` paths
- workspace snapshot / commit identifiers

Purpose:

- rule out environment ambiguity during anomaly investigation

## 5. Storage format

### `phase4_anomaly_debug.json`

Top-level summary should include:

- environment fingerprint
- run config / prompt id
- aggregate refresh counts
- aggregate cutoff/tie stats
- aggregate deterministic-shadow overlap stats
- aggregate float64-shadow overlap stats
- frozen-frontier drift summary

### `phase4_refreshes.jsonl`

One record per refresh, with:

- refresh metadata
- cutoff/tie diagnostics
- actual frontier signature
- overlap stats
- shadow deterministic comparison
- optional float64 comparison

## 6. Acceptance criteria

- one prompt-94 debug run produces enough information to answer:
  - whether frontier membership drifts strongly over refreshes
  - whether cutoff ties / near-ties are common
  - whether a deterministic tie-break materially changes top-K membership
  - whether float32 vs float64 ranking materially changes top-K membership
  - whether later divergence mainly comes from refresh amplification

## 7. Risks and open questions

### Debug overhead

Risk:

- collecting too much shadow data could slow prompt-94 runs further

Mitigation:

- keep payloads scalar-only
- limit float64 shadow ranking to early refreshes by default
- keep actual traced behavior unchanged

### Interpretation risk

Risk:

- one cluster run alone cannot prove cross-cluster causality

Mitigation:

- use the same debug artifact on both clusters so runs are directly comparable
