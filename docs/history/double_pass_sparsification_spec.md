# Double-pass sparsification spec

## Problem statement
Final-graph sparsification is too late. If we prune only after the full graph is reconstructed, we still pay the dominant cost of enumerating and combining the full active set in phase 0 and phase 3/4.

The fork needs a sparsification path that reduces work before expensive reconstruction/attribution while preserving `attribute(...)` compatibility and the current `Graph` schema/node ordering.

## Scope
- Single-GPU tracing only.
- No new output graph schema.
- No multi-GPU tracing.
- Keep existing `attribute(...)` calls working when sparsification is disabled.

## Non-goals
- Rewriting the tracer backend.
- Changing graph node ordering.
- Making the exact path slower for non-sparsified runs.

## Proposed approach

### Recommended design
Use a two-pass flow:
1. **Pass 1: candidate screening** computes a cheap retained-feature mask.
2. **Pass 2: exact-on-candidates tracing** runs reconstruction and chunked attribution only on retained candidates.

This is the right place to cut cost because the expensive phases should never see the full active set.

### API surface options
1. **Optional flags on `attribute(...)`** with a small sparsification config object. **Recommended.**
   - Best fit for compatibility.
   - Keeps the feature opt-in.
   - Avoids GemmaScope-2-only branching in call sites.
2. **GemmaScope-2-specific transcoder mode/config.**
   - Faster to prototype, but leaks library details into the API.
3. **Separate backend/helper object.**
   - Cleaner separation, but more invasive for the current fork and harder to adopt incrementally.

Recommendation: add optional sparsification flags/config on `attribute(...)` (or the replacement-model config it already consumes), default off.

### Candidate screening strategies
Initial options:
- **Global top-K by activation**: existing monkeypatch behavior; fastest but too coarse.
- **Per-layer top-K by activation**: better layer coverage, still simple.
- **Per-layer-per-position top-K by activation**: best initial balance of locality and fidelity.
- **Future proxy:** activation × gradient or similar salience score.

Recommended initial implementation:
- start with **per-layer-per-position top-K**,
- allow a global cap as a safety valve,
- derive budgets from calibration rather than hardcoding one number.

Rationale: global top-K over-favors a few dominant layers; per-layer-per-position keeps candidate coverage closer to the exact path and reduces the risk of deleting useful local features early.

### Where to hook in
Hook candidate selection in the fork **before reconstruction** and **before chunked attribution state is finalized**.
- Phase 0 should build or consult the candidate mask before expensive component reconstruction.
- Phase 3/4 should reuse the same mask so attribution only iterates over retained candidates.
- Do not rely on final graph pruning to recover cost.

### Data contracts that must stay stable
- `attribute(...)` remains callable with sparsification disabled.
- `Graph` output format stays unchanged.
- Node ordering stays `[features, errors, tokens, logits]`.
- Any candidate mask is internal state, not serialized as part of the public graph schema.

### Cost savings target
The same candidate set should reduce both:
- phase 0 reconstruction cost, and
- phase 3/4 chunked attribution cost.

If a candidate set only helps the final graph size, it is not enough.

### Observability
Add profile-mode logging for:
- total candidate count,
- per-layer retained counts,
- retained attribution mass proxy, if available,
- phase timings for encode/screen/reconstruct/attribution,
- peak memory by phase.

## Validation plan
- Compare sparsified runs against exact on tiny cases first (2-3 steps, 1 deterministic completion).
- Check overlap, rank correlation, and retained attribution mass.
- Verify the new path matches the exact path when budgets are effectively unconstrained.
- Confirm that disabling sparsification produces unchanged outputs and timings close to the current fork baseline.

## Acceptance criteria
The implementation is acceptable if:
- exact tiny-scope comparisons stay close on overlap and ranking,
- phase 0 and phase 3/4 become materially cheaper,
- the API remains backward compatible,
- and the output graph schema/node ordering do not change.

## Risks / open questions
- The best budget may vary by prompt length and layer depth.
- Per-layer-per-position screening may still be too expensive if the candidate enumeration itself is naive.
- We may need a second-stage proxy later if activation-only screening misses important low-activation features.
- Need to confirm how much logging is safe before it becomes noisy on large arrays.

## Assumptions
- Exact tracing remains available as a calibration/reference mode.
- The current fork can expose candidate masks internally without changing the public graph format.
