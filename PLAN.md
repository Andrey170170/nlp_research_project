# Current Plan — Phase 3/4 Design + Phase 1 Decoupling

## Current status

The last Ascend matrix (`5101442`) gives the current decision basis:

- `active_encoder_cpu` is the strongest exact win.
- `rowstore_fadvise` is an exact, modest win.
- `phase1_cap64` is not a candidate: coupling/regression/drift make it invalid.
- `deferred_refresh_x2` is mixed and non-exact, so keep it out of the exact path.
- `topk` is exact, but the win is not yet clear enough to promote blindly.

Working rule: no global defaults until a variant is validated. All runs must keep flags explicit.

## Work order

1. Design Phase 3/4 first.
2. Pick the best exact-preserving candidate.
3. Update this plan for implementation details.
4. Implement that candidate together with Phase 1 decoupling.
5. Run the full combined Ascend array once the implementation is ready, because cluster pressure is high.

## Implementation needed

Status: Phase 1 decoupling and next-array scenario generation are implemented
locally but not yet committed or launched.

### Phase 1 decoupling

- Keep current behavior as the default.
- Add only an optional Phase 1 cap path.
- The cap must affect Phase 1 only; it must not change Phase 3/4 behavior.
- Use the combo scenario as the scenario default only; do not turn it into a library default.
- Preserve explicit requested/effective metadata.
- Implemented behavior: `phase1_trace_batch_policy=cap_effective_batches` now
  caps only the Phase-1 source/invoke trace batch size. Feature, logit, and
  Phase-4 max feature batch sizes remain at their requested/default values.

### Phase 3/4 design target

- Chosen next-array candidate: **budgeted Phase 4 decoder-chunk cache probe**.
- This is exact-preserving if it only changes decoder chunk residency / reuse:
  same Planner V1 frontier, same row order, same refresh cadence, same row-store
  semantics.
- Use the existing `cross_batch_decoder_cache_bytes` machinery rather than a new
  global default. Keep the budget explicit in scenario JSON.
- Context: earlier cross-batch decoder-cache work suggested `361_base` may need a
  large budget (`~8-12 GiB`) and did not obviously help enough at the time. That
  was before the fp32 overflow/stability fixes and before the current
  `active_encoder_cpu + rowstore_fadvise` baseline, so it is worth one bounded
  re-test, but still treat this as a probe rather than a likely default.
- Run it on top of the combo scenario (`active_encoder_cpu + rowstore_fadvise`),
  because `active_encoder_cpu` frees enough memory to make a bounded GPU cache a
  more realistic A100 probe.
- Start conservative: one explicit budget only unless implementation/telemetry
  shows the need for a second budget. Prefer `4 GiB` or `8 GiB`; do **not** add an
  auto-budgeting policy before the first combined array.
- Required telemetry / analysis fields:
  - requested/effective `cross_batch_decoder_cache_bytes`,
  - decoder cache hits, misses, evictions, skips, resident bytes,
  - decoder load count/time,
  - CUDA peak reserved and sacct MaxRSS,
  - compact comparison against the combo baseline.

Design alternatives considered but deferred:

- **Cache-aware scheduler / stronger locality reorder:** plausible, but changes
  execution order and can alter floating-point accumulation/frontier behavior.
- **GPU-side denominator:** potentially useful, but GPU reduction order may perturb
  refresh scores; needs a separate parity-focused design. This remains a desired
  follow-up path after the cache/Phase-1 combined array.
- **Incremental influence propagation:** highest algorithmic upside, but too risky
  for the next shared array. Start later in shadow mode, not as the next live path.
- **Row-store read/cache refactor:** previous broad read-path work regressed; keep
  `rowstore_fadvise` as the narrow exact piece for this run.

## Planned run matrix

Scenario file:

- `experiments/generated/exact_trace_bench/exact_trace_next_combo_phase1_cache_ascend_scenarios.json`

### Group A — combo matrix

Run baseline / `active_encoder_cpu` / `rowstore_fadvise` / combo across:

- `828_base`
- `361_base`

Total: 8 jobs.

### Group B — Phase 1 decoupling

Use combo as the scenario default, not the library default.

Minimum matrix:

- `828_base`: Phase 1-only cap vs combo baseline
- `361_base`: Phase 1-only cap vs combo baseline

Total: 2 jobs.

Optional if fixture readiness is confirmed:

- `361_late` combo
- `361_late` combo + Phase 1-only cap

### Group C — selected Phase 3/4 optimization

Run the selected budgeted decoder-cache probe on top of combo for:

- `828_base`
- `361_base`

Total: 2+ jobs.

Suggested scenario shape:

- `exact_encoder_residency=active_cpu`
- `row_store_cache_control=fadvise_dontneed_after_append_v1`
- `phase4_scheduler_mode=planner_v1`
- `cross_batch_decoder_cache_bytes=<explicit 4GiB or 8GiB budget>`

If the cache budget causes OOM or churn-heavy telemetry, treat the candidate as
falsified for the next default-combo path rather than immediately adding more
budget variants.

## Acceptance criteria

- Phase 1 default behavior remains unchanged.
- Phase 1 cap is opt-in and isolated from Phase 3/4.
- No global defaults are introduced before validation.
- The chosen Phase 3/4 candidate is exact-preserving and justified by design.
- Decoder-cache probe success requires:
  - exact compact output vs combo baseline,
  - substantial cache hits without eviction-dominated churn,
  - material decoder load count/time reduction,
  - at least a modest wall-clock improvement on both base prompts,
  - no unacceptable CUDA peak / sacct MaxRSS increase.
- The combined Ascend array covers all three groups with explicit flags only.

## Local validation

- Keep validation lightweight.
- Use only doc / plumbing checks and focused unit tests.
- Do not run GPU/model work outside SLURM.

Suggested checks:

- scenario/config round-trip for explicit flags and requested/effective metadata,
- Phase 1 cap only changes Phase 1 fields,
- exact-preserving Phase 3/4 candidate matches baseline on a small fixture,
- `uv run ruff check ...` for touched Python/docs-adjacent plumbing where applicable.
