# Exact-trace performance patch archive

Date indexed: 2026-07-31

These patches preserve rejected, superseded, or not-yet-selected implementation
hypotheses from the exact-trace performance campaign. They are evidence and
recovery material, not current source and not a sequential patch series. Their
target paths belong to the sibling `circuit-tracer_chunked` repository.

Before applying one, identify the source revision it was produced against,
inspect it with `git apply --stat` and `git apply --check`, and re-run its
focused tests. A fidelity pass alone does not justify restoring a branch to
runtime code; restoration requires a non-dominated performance/memory role or
another explicit operational advantage.

## Index

| Patch | SHA-256 | Purpose / status |
|---|---|---|
| `active-pinned-cpu-project-restore.patch` | `7864323d22149690797cf8f170ea2eb4ca14b3ec6e08d6b7a060b995d50fb57c` | Restores the project CLI, candidate profile, and experiment-envelope registration for the rejected active-pinned encoder mode. Apply from the project root only if new evidence justifies revival. |
| `active-pinned-cpu-sibling-restore.patch` | `aa6b862382f8e6735eae60b0c1b21630fa56674169af3821a5b0a350c29d98c8` | Restores the sibling runtime implementation and public type surface for active-pinned encoder residency. Apply from the sibling root before the project-side patch. Focused tests must be restored or rewritten separately. |
| `phase0_budget.patch` | `7e02ef61203977825900afa3d0d6691ebcd422d746c7b7d360533075116e355e` | Alternative Phase-0 range planner that enforces the total overfetch-row budget while groups are built. Preserved hypothesis; not selected into current source. |
| `seed-binding-source.patch` | `c0c9aaef79c3506b84d87a4d2e88c63eda483eb51a84b5dd44dd605f5dfbdb54` | First source-context version of caller-provider fingerprint binding for `DecoderRowSeed`. Superseded by later rebases below. |
| `seed-binding-source-v2.patch` | `aba22eebd5a4ec6b388a7e656905e76aba47f99c851f3941e42d40322cf2362a` | Context-rebased form of the same seed-binding hypothesis. Alternative, not cumulative. |
| `seed-binding-source-v3.patch` | `64af0855d7de540f10156dfb0da37fe603104de30aad20c7b272a7dbdccc011c` | Corrected hunk context for the same seed-binding hypothesis. Alternative, not cumulative. |
| `seed-binding-source-v4.patch` | `6d6806520c35c77970f6317c724dcc37b7dcdfd3a726f24743336bd890fe150e` | Later source-context rebase of the same seed-binding hypothesis. Alternative, not cumulative. |
| `seed-binding-source-v5.patch` | `a7e937a97fded73488902608c49e90762af7f3c252fbdcfd7acf19b1270ef2ff` | Last preserved source-context version of the seed-binding hypothesis. Pair with the test patch if revisited. |
| `seed-binding-test.patch` | `4e1eb48cf6240c4eee355be8ab8c88215ea324b1b9002a744efbfa33fdfb1de8` | Focused tests for rebinding a decoder-row seed to the caller provider and rejecting a cross-provider seed. |

The five `seed-binding-source*` files encode revisions of one idea. Apply at
most one source patch, normally the newest revision that cleanly rebases, plus
`seed-binding-test.patch`; do not apply the source revisions in sequence.

The active-pinned restoration is split by repository. Both patches pass
`git apply --check` against the post-removal worktrees as archived. They recover
the implementation, but intentionally do not restore deleted tests: revival
requires tests that express the new evidence and current contracts rather than
blindly reinstating historical assertions.

## Retention convention

Future removed variants should be stored in a dated subdirectory with:

1. the patch against a named commit;
2. a short result/evidence link;
3. the reason it lost comparative selection;
4. the dimensions on which its neighbor dominated it; and
5. a checksum and minimal restoration instructions.
