# Phase 4 Next Optimization Plan

Scratch plan for the optimization worktree. Durable conclusions belong in
`EXPERIMENTS.md` and design tradeoffs in the relevant docs/specs.

## Current baseline

- Keep staged `phase4_row_reduction=gpu_v1` as the default exact compact path.
- `phase4_row_reduction=off` remains the CPU-reference fallback.
- Negative results already recorded:
  - writable memmap assignment was slower than `os.pwrite`,
  - forced-copy fused refresh transfer/cast/abs was slower.

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

1. Baseline default `gpu_v1`.
2. Row-store placement / preallocation only.
3. Prepared refresh chunk cache only.
4. Active-row direct accumulation only.
5. Best combined variant only after individual wins are known.

## Immediate next step

Implement candidate 1: row-store placement / preallocation.
