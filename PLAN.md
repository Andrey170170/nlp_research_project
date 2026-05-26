# Phase 4 Next Optimization Plan

Scratch plan for the optimization worktree. Durable conclusions belong in
`EXPERIMENTS.md` and design tradeoffs in the relevant docs/specs.

## Current baseline

- Keep staged `phase4_row_reduction=gpu_v1` as the default exact compact path.
- `phase4_row_reduction=off` remains the CPU-reference fallback.
- Promoted performance default is `phase4_row_reduction=gpu_v1` +
  `phase4_refresh_optimization=v1` +
  `phase4_refresh_active_row_accumulation=direct_v1` +
  `row_store_preallocate=true` + `cross_batch_decoder_cache_bytes=8GiB`.
- Cache8g is acceptable for this performance default as a near-exact,
  cutoff-sensitive tradeoff: decoder chunk fingerprints are bitwise stable, but
  retained edge sets can shift at the few-e-6 weight level near cutoff. Keep a
  strict cache0 path documented for bitwise-retained-edge checks.
- Negative results already recorded:
  - writable memmap assignment was slower than `os.pwrite`,
  - forced-copy fused refresh transfer/cast/abs was slower.
- Current exact-range prepared refresh cache is exact but not promotion-ready:
  telemetry shows poor hit rates, large too-large skips, and high prepared-read
  miss cost at 8GiB. Larger 32/64GiB budget tuning did not make the behavior
  robust enough to promote. It is retired for this optimization track; do not
  include `phase4_refresh_prepared_chunk_cache_bytes>0` in promotion or
  interaction matrices.

## Candidate speedups

These are not mutually exclusive, but test them independently before combining.

### 1. Row-store placement / preallocation

Goal: reduce row-store write cost without changing row math or refresh order.

Implementation shape:

- Add an explicit row-store temp-root policy for the file-backed feature row
  store, preferring node-local scratch when configured/available.
- Surface telemetry for chosen row-store path/root and whether preallocation ran.
- Add optional file preallocation (`posix_fallocate` if available, otherwise keep
  current `truncate` behavior).
- Preserve existing `os.pwrite` write path; do not reintroduce memmap assignment.

Validation:

- Local tests for path selection, fallback, and preallocation telemetry.
- Cardinal fast `828_base` / `361_base`, default `gpu_v1` plus `off` reference if
  needed.

### 2. Prepared refresh chunk cache

Goal: avoid repeated transfer/cast/abs work during refresh.

Implementation shape:

- Cache post-transfer/post-abs chunks or active subranges by row range, dtype, and
  device.
- Keep the same accumulation order and normalization math.
- Add a memory budget and hit/miss/eviction telemetry.

Validation:

- Exact output parity vs baseline default `gpu_v1`.
- Watch CUDA/RSS memory and refresh partial-influence timing.
- Final narrow tuning pass with `direct_v1` only and prepared budgets `0`,
  `8GiB`, `32GiB`, `64GiB` is complete. Larger budgets did not materially improve
  hit rate and were not robust across prompts; stop pursuing exact-range prepared
  caching and only revisit a chunk-aligned/windowed redesign later.

### 3. Active-row direct accumulation

Goal: avoid zero-filled full chunks in active-row mode.

Implementation shape:

- Process each active subrange directly into the accumulator.
- Keep as a separate flag because accumulation order may shift floating-point
  behavior.

Validation:

- Strict compact parity first on `828_base` and `361_base`.
- Then anomaly `94_base` if parity and speed are promising.

## Test matrix

Completed interaction matrix:

- exact cache0 candidate (`direct_v1 + row_store_preallocate=true`):
  - `828_base`: total `457.10 s -> 424.32 s` (`-7.2%`), Phase 4
    `117.12 s -> 89.26 s`,
  - `361_base`: total `729.83 s -> 557.24 s` (`-23.6%`), Phase 4
    `250.23 s -> 132.42 s`,
  - saved compact `.npz` outputs were bitwise-identical to baseline.
- proposed performance default including cache8g:
  - best `828_base`: `direct_v1 + cache8g` (`370.31 s`, `-19.0%`),
  - best `361_base`: `direct_v1 + row_store_preallocate + cache8g`
    (`538.35 s`, `-26.2%`),
  - retained-edge arrays remain cutoff-sensitive, not bitwise exact.

## Immediate next step

Completed broader immutable generalization matrix on the matched base fixtures
(`828_base`, `613_base`, `999_base`, `1046_base`, `1075_base`) comparing:

1. current baseline,
2. strict-exact candidate (`direct_v1 + row_store_preallocate`, cache0),
3. proposed performance default (`direct_v1 + row_store_preallocate + cache8g`).

If the speedup translates, promote the performance default while preserving clear
documentation that cache8g is near-exact/cutoff-sensitive and cache0 is the
bitwise-retained-edge fallback.

Status: completed on Cardinal as SLURM array job `10434364` (`0-14`, all
`ExitCode 0:0`).

- run id / output root:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/20260525_broader-default-validation-fast`
- immutable snapshot:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/workspace_snapshots/workspace_20260525_190714_broader_default_validation`
- scenario file:
  `experiments/generated/exact_trace_bench/exact_trace_broader_default_validation_fast_cardinal_scenarios.json`

Result summary:

- strict cache0 candidate was bitwise exact for all five fixtures and improved
  Phase 4 on all five, though `613_base` regressed end-to-end (`+6.5%`).
- performance cache8g candidate improved end-to-end time for all five fixtures:
  `-12.8%` to `-36.1%` (mean `-23.1%`).
- cache8g preserved token/logprob/active features but changed retained edges as
  expected; positional max retained-weight drift was up to `7.20e-6`, while
  edge Jaccard ranged from `0.955799` to `0.998801`.

Implementation status: default-promotion patch is applied in the optimization
worktree. It keeps strict-exact fallbacks (`--cross-batch-decoder-cache-bytes 0`,
`--phase4-refresh-optimization off`, `--phase4-refresh-active-row-accumulation
zero_fill`, `--no-row-store-preallocate`, and `--phase4-row-reduction off`) while
making the validated performance path the default.
