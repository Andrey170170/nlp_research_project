# Direction 2 Prefix Caching — Handoff

**Purpose:** self-contained snapshot so a fresh chat can pick up the work
without scrolling history. Covers what's done, what's not, and the plan
to actually demonstrate a speedup via KV-cache integration.

---

## TL;DR

- **Fidelity proven.** Forward prefix cache + Strategy D decoder-chunk
  session both produce byte-identical traces vs uncached baseline
  (Feature Jaccard = 1.000, Edge Jaccard = 1.000, max |Δ| = 0.0 over 5
  steps on prompt 94).
- **No speedup proven.** The forward cache is record-only — it never
  skips Phase 0 compute. Strategy D persists decoder chunks across
  steps but the savings (~16 s on a 2400 s step) are within
  inter-run GPU variance.
- **Next step (this is what we want to build):** wire a real Phase 0
  skip via nnsight KV-cache integration. Expected ~10 % total
  speedup. The activations are already cached; the consumer is
  missing.

---

## Repos and branches

- **Library** (mostly Andrei's; Jay has a fork):
  `/fs/ess/PAS2136/jjivandas/projects/circuit-tracer_chunked` →
  branch `jay/prefix-caching`, pushed to fork
  `https://github.com/jjivandas/circuit-tracer_chunked`. Latest commit
  `38197b7` adds Strategy D plumbing.
- **Harness** (Andrei's, Jay has direct write access):
  `/fs/ess/PAS2136/jjivandas/projects/nlp_research_project` →
  branch `jay/prefix-caching` on `Andrey170170/nlp_research_project`.
  Latest commit `e3ebe1f` wires the harness side.
- No PRs opened yet; user wants to PR after the speedup story is in.

---

## Brief deliverable status (from `prefix_caching/BRIEF.md`)

| # | Ask | Status |
|---|---|---|
| D1 | Forward prefix cache that **reuses** activations | ⚠ infra ✓, reuse ✗ (record-only) |
| D2 | Diagnostics (cached / reused / recomputed / why) | ✅ |
| D3 | Backward reuse prototype **or** study | ✅ both — study + Strategy D |
| D4 | 3–5 scenarios | ❌ only prompt 94 |
| D5 | Recommendation note | ✅ `RECOMMENDATION.md` |
| Fidelity axis | "cached matches uncached closely" | ✅ byte-identical |
| Systems axis | "runtime meaningfully lower" | ❌ +0.9 % vs baseline |

---

## What's already implemented

### In the library (`circuit-tracer_chunked`)

1. `circuit_tracer/attribution/prefix_cache.py` — `PrefixActivationCache`
   class storing pre-sparsification `mlp_in` / `mlp_out` keyed by
   `(prefix_token_ids, model_fingerprint)`. Strict prefix-equality
   invalidation. CPU storage by default.
2. `circuit_tracer/replacement_model/replacement_model_nnsight.py`
   `setup_attribution()` — accepts `prefix_cache=None` and
   `decoder_chunk_cache=None` kwargs. Currently just **calls
   `lookup()` and `store()` on the prefix cache** but always runs the
   full forward pass via `with self.trace(tokens):` (line 549). The
   forward cache is never consumed.
3. `circuit_tracer/attribution/{attribute_nnsight.py, attribute.py,
   context_nnsight.py}` — `prefix_cache` and `decoder_chunk_cache`
   threaded through.
4. `AttributionContext` accepts `external_decoder_chunk_cache` kwarg;
   `_owns_decoder_chunk_cache` flag prevents `cleanup()` from
   destroying caller-owned caches.
5. `tests/test_prefix_cache.py` — 15 CPU unit tests, all green.

### In the harness (`nlp_research_project`)

1. `prefix_caching/trace_pipeline_cached.py` — driver with
   `--use-library-prefix-cache` and `--use-decoder-chunk-session`
   flags. Instantiates one cache per completion; passes into every
   `attribute()`.
2. `prefix_caching/compare_cached_vs_uncached.py` — fidelity comparator
   (Jaccard, max |Δ|) over `step_NNN.npz`.
3. `prefix_caching/reconstruct_diagnostics.py` — rebuilds
   `cache_validation.json` from surviving artifacts when SLURM kills a
   run mid-flush.
4. `prefix_caching/docs/BACKWARD_REUSE_STUDY.md` (D3 study).
5. `prefix_caching/RECOMMENDATION.md` (D5).
6. `prefix_caching/docs/SLIDE.md` (1-slide contribution summary).
7. SLURM scripts:
   - `scripts/trace_prefix_cache_bench.jay.ascend.sbatch` — uncached baseline
   - `scripts/trace_prefix_cache_run.jay.ascend.sbatch` — cached (forward only)
   - `scripts/trace_prefix_cache_strategyD.jay.ascend.sbatch` — cached + Strategy D

---

## What's NOT implemented (the speedup gap)

The forward prefix cache stores activations but **no code path consumes
them to skip work**. Specifically, in `setup_attribution()`:

```python
with self.trace(tokens):                     # line 549
    mlp_in_cache, mlp_out_cache = [], []
    for feature_input_loc, feature_output_loc in zip(
        self.feature_input_locs, self.feature_output_locs
    ):
        mlp_in_cache.append(feature_input_loc.output)
        ...
        mlp_out_cache.append(y)
    mlp_in_cache = save(torch.cat(mlp_in_cache, dim=0))
    mlp_out_cache = save(torch.cat(mlp_out_cache, dim=0))
    logits = save(self.output.logits)
```

This trace runs the **full transformer forward pass over every prefix
token, every step**, even though every prefix position's activations
are already in `prefix_cache`. That's the redundant work the brief is
asking us to skip.

---

## Numbers we have so far

All on prompt 94 (3 completed steps unless noted), single A100-80GB,
quad partition.

| Run | Step 0 | Step 1 | Step 2 | Total |
|---|---|---|---|---|
| Uncached baseline (5011889, 5 steps) | ~1730 s | ~1730 s | ~1730 s | mean |
| Forward cache only (5066786, 3 steps) | 651.1 | 2435.0 | 590.3 | 3676.4 s |
| Forward cache + Strategy D (5073456, 3 steps) | 652.7 | 2418.9 | 636.8 | 3708.4 s |

Per-step times vary wildly on the shared quad node (590 s vs 2435 s on
the *same* step number across runs). Inter-run variance >> any savings
we'd plausibly extract. Strategy D demonstrably worked mechanically
(`entries=130, bytes_resident=8.15 GiB` stayed flat across all 3 steps,
proving step 1 and 2 hit cached chunks instead of rebuilding) but the
savings are noise-bound.

---

## Plan: wire the actual Phase-0 forward skip via KV cache

**Target:** when `prefix_cache` has a hit covering `[0..P-1]` of the
prompt, run the transformer forward **only on tokens `[P..N-1]`** with
`past_key_values` from the cached prefix; splice cached `mlp_in`/
`mlp_out` for prefix positions; concatenate with new positions for
sparsification + downstream attribution.

### Why this should yield speedup

- Phase 0 is ~10 % of per-step time (~200–250 s of a 2400 s step,
  per the study).
- Skipping the forward pass on prefix positions saves ~95 % of Phase 0.
- Realistic ceiling: **~10 % total wall-clock speedup** on long-prefix
  consecutive steps (more if Phase 0 is a bigger chunk than estimated;
  this should be confirmed via `--profile-attribution` first — see
  step 1 below).
- Fidelity-neutral by construction *if* the splice is exact (KV math is
  deterministic on frozen weights with greedy decoding).

### Concrete implementation steps

1. **Capture per-phase timings first.** Re-run the existing cached
   sbatch with `--profile-attribution` flag (already supported in
   `trace_pipeline_cached.py`). Look for stdout lines
   `phase0.setup.trace_done elapsed_s=...` and the
   `setup_diagnostic_stats` keys (`trace_seconds`,
   `component_seconds`, `error_seconds`). This tells us exactly how
   much of per-step time Phase 0 actually is, so the Phase 0 skip
   budget is grounded in measurement, not estimate. **5-min code
   change, one GPU run.**

2. **Extend `PrefixActivationCache` to also store KV state.** Add
   `past_key_values: tuple[tuple[torch.Tensor, ...], ...]` to the
   cache entry (per-layer, per-head K and V tensors over prefix
   positions). Add a `mlp_logits_last_token` slot for the final-token
   logits we still need from the trace.

3. **Modify `setup_attribution()` to branch on cache state.** When
   `prefix_cache.lookup()` returns a hit covering `[0..P-1]`:
   - Run `with self.trace(new_tokens, past_key_values=cached_kv):`
     instead of `self.trace(tokens)`.
   - Capture `mlp_in_new` / `mlp_out_new` only for new positions.
   - Concatenate cached `mlp_in[:, :P]` with `mlp_in_new` along the
     position axis; same for `mlp_out`.
   - Capture the new `past_key_values` from the trace and store in
     `prefix_cache` for the next step.
   - Logits: only the last token's logits are typically needed, and
     those come from the new-tokens trace.
   - Continue with `compute_attribution_components(mlp_in_full, ...)`
     unchanged.

4. **Verify fidelity.** Re-use `compare_cached_vs_uncached.py` against
   the existing `baseline_exact/` artifacts. Required: same
   Feature/Edge Jaccard = 1.000 we hit before. If even 1 edge weight
   drifts, the splice is wrong; debug before merging.

5. **Measure speedup.** Compare new cached run vs uncached baseline.
   Phase 0 portion should drop on every step after step 0.

### Risks specific to KV-cache work

- **nnsight `past_key_values` support.** Need to verify the nnsight
  `trace()` API accepts `past_key_values` and exposes them as save-able
  proxies. If nnsight's tracing layer doesn't surface them, fallback is
  to use HuggingFace's `model.generate(use_cache=True)` paths and
  bypass the nnsight intervention machinery for Phase 0 only — uglier
  but viable.
- **Sparsification global top-K.** Sparsification picks top-K across
  all positions every step. If we splice cached `mlp_in[:, :P]` with
  fresh `mlp_in[:, P:]`, the sparsification still runs over all
  positions — we're not changing its semantics, just the source of
  the activation tensors. So fidelity should hold.
- **bfloat16 nondeterminism.** Even with frozen weights and same
  inputs, bfloat16 reductions can drift in the last bit. We've already
  seen byte-identical results in the record-only path (because we
  re-ran the full forward pass), but a KV-cache splice swaps the
  arithmetic order. Acceptable threshold: max edge-weight |Δ| should
  remain at 0.0 in practice; if it goes nonzero, decide whether
  trace-level Jaccard = 1.000 is enough or if we need exact
  reconstruction.

---

## Where to look first when continuing

1. `circuit_tracer/replacement_model/replacement_model_nnsight.py`
   line 549 — the `with self.trace(tokens):` block. This is the surgery
   site.
2. `circuit_tracer/attribution/prefix_cache.py` — already-built cache,
   add KV state to it.
3. `prefix_caching/trace_pipeline_cached.py` — driver, no changes
   needed unless we add diagnostics for "skipped forward pass on N
   positions."
4. nnsight docs on `trace()` kwargs — need to confirm
   `past_key_values` plumbing works.

---

## Useful commands

```bash
# Re-run the cached sbatch with profiling
# Edit scripts/trace_prefix_cache_strategyD.jay.ascend.sbatch and add:
#     --profile-attribution \
sbatch scripts/trace_prefix_cache_strategyD.jay.ascend.sbatch

# Library tests (CPU, fast)
cd ../circuit-tracer_chunked
uv run --active pytest tests/test_prefix_cache.py -q

# Compare fidelity (post-run)
uv run python -m prefix_caching.compare_cached_vs_uncached \
    --baseline /fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/baseline_exact \
    --cached   /fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/strategy_d \
    --output   /fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/analysis/fidelity_report_kv.json
```

---

## Constraints to keep in mind

- OSC: GPU work only on SLURM, never login nodes. Linting / unit
  tests are fine on login.
- Environment manager is `uv`. Always `uv run` or `uv run --active`.
- Output to scratch: `/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/`.
- Don't run more than one cached sbatch in parallel against the same
  output directory; results will collide.

---

## Open questions for the next chat

1. Does nnsight's `trace()` accept `past_key_values=...` as a kwarg?
   This is the single biggest unknown — the Phase 0 skip plan
   depends on it. First action: read nnsight docs / source for
   the underlying HF model invocation.
2. Should KV state live inside `PrefixActivationCache` or in a
   separate `KVCacheSession` object analogous to the decoder-chunk
   session? The library already has the plumbing pattern for
   session objects; reusing it keeps the API consistent.
3. If KV splice introduces sub-1-bit drift in bfloat16, is that
   acceptable? The brief says "preserving exact traced results" —
   need to clarify how strict "exact" is.
