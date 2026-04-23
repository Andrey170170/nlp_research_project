# TODO — optimization repo

This workspace is for **optimization work**, not cross-cluster interpretation.

## Immediate tasks

1. Confirm the refreshed optimization project+library pair is operational.

2. Keep the permanent overflow fix semantics-preserving:
   - no reliance on fp32 collapse
   - no frontier-selection policy changes
   - no silent approximation

3. Fix the current Phase 4 memory problem with the **safe hybrid row-store cut**:
   - preserve cheap mmap-style reads/materialization
   - change only append/writeback behavior
   - target batch-aligned file-backed RSS growth directly

4. Validate the hybrid fix on the canonical fast prompts:
   - `828_base`
   - `361_base`
   - compare file RSS growth, refresh wall time, Phase 5 wall time, and exact
     output invariants

5. Only after the main hybrid fix is validated, consider smaller cleanup:
   - staging buffer shrink/reset
   - read-cache tuning
   - other narrow hygiene passes

## Later scheduler-redesign considerations

These are **not** standalone optimization tasks for now.

Only revisit them if they become useful during a later scheduler/frontier
redesign, where we can evaluate them together with graph exploration behavior
and the memory operations implied by that redesign.

- two-tier hot/cold row storage
- cheaper scheduler indexes / metadata summaries that reduce rereads
- incremental refresh state
- approximate ranking / prefilter ideas
- larger storage-layout redesigns coordinated with scheduler work

## Optimization order

1. Permanent overflow fix
2. Safe hybrid row-store memory fix
3. Smaller staging/cache cleanup if still needed
4. Later scheduler redesign, with memory implications considered in that context

## Rules for this workspace

- Do not redefine the canonical debug schema here without explicit review.
- Keep runs in the standard folders only:
  - `ascend/fast`
  - `ascend/anomaly`
  - `ascend/long_eval`
  - `cardinal/fast`
  - `cardinal/anomaly`
  - `cardinal/long_eval`
- Distinguish experiments using run metadata, not new top-level folders.
