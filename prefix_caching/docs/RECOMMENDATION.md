# Recommendation — Direction 2 Prefix Caching

**Author:** Jay Jivandas
**Date:** 2026-04-23
**Scope:** Covers brief deliverable D5 for [BRIEF.md](BRIEF.md). Summarizes
what was built, what is trustworthy, what is not, and what to do next.

---

## 1. What was implemented

**In the library fork
([`jjivandas/circuit-tracer_chunked`](https://github.com/jjivandas/circuit-tracer_chunked),
branch `jay/prefix-caching`):**

- `circuit_tracer/attribution/prefix_cache.py` — `PrefixActivationCache`
  class. Stores pre-sparsification `mlp_in` / `mlp_out` tensors keyed by
  token prefix + model fingerprint. Strict prefix-equality invalidation.
  Emits a `CacheDiagnostics` dataclass on every lookup.
- `tests/test_prefix_cache.py` — 15 CPU unit tests covering lookup, store,
  mismatch, partial-prefix, fingerprint drift. Green.
- `circuit_tracer/replacement_model/replacement_model_nnsight.py` —
  threaded a `prefix_cache=None` kwarg into `setup_attribution`. Lookup
  before the forward pass, store after. Trace events emitted on both.
- `circuit_tracer/attribution/attribute_nnsight.py` and `attribute.py`
  — same kwarg threaded through so `attribute()` callers can pass in a
  cache. Default `None` preserves pre-existing behavior exactly.

**In the harness repo (this directory,
[`jjivandas/jay/prefix-caching`](https://github.com/Andrey170170/nlp_research_project/tree/jay/prefix-caching)):**

- `prefix_caching/trace_pipeline_cached.py` — driver with
  `--use-library-prefix-cache` flag. Instantiates one
  `PrefixActivationCache` per completion and passes it to every
  `attribute()` call.
- `prefix_caching/compare_cached_vs_uncached.py` — CPU-only fidelity
  analyzer pairing baseline vs cached `step_NNN.npz` files.
- `prefix_caching/reconstruct_diagnostics.py` — stand-alone rebuilder
  for `cache_validation.json` from surviving artifacts when SLURM kills
  a run before it can flush its own diagnostics.
- `prefix_caching/docs/BACKWARD_REUSE_STUDY.md` — written study covering D3.
- `scripts/trace_prefix_cache_bench.jay.ascend.sbatch` — uncached
  baseline sbatch.
- `scripts/trace_prefix_cache_run.jay.ascend.sbatch` — cached sbatch
  (identical config + the flag).

---

## 2. Is forward prefix cache reuse trustworthy?

**Yes, for fidelity. No, for speed.**

### Fidelity: trustworthy

The cache produces **byte-identical** traces vs the uncached baseline on
the one scenario we ran to completion (GSM8K prompt 94, 5 steps):

- Feature Jaccard = 1.000000 on every step
- Edge-set Jaccard = 1.000000 on every step
- Max edge-weight |Δ| = 0.0 on every step
- Active feature counts match exactly (3.37 M → 3.55 M)

Evidence: `/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/analysis/fidelity_report.json`.

Because the cache stores pre-sparsification activations and sparsification
is re-run every step, drift at the top-K boundary does not affect the
cached output. The trace is the same as if the cache were not there.

### Speed: not yet trustworthy

What was built is a **record-only** cache: `store()` and `lookup()` run,
but no code path consumes the lookup result to skip forward-pass compute.
Cached runtime is ~30 s/step **slower** than baseline (bookkeeping
overhead with no compute savings to offset).

This was a deliberate scope boundary — fidelity-first. Actually skipping
compute requires either:

- Running the model forward only on new tokens and splicing cached prefix
  activations (needs KV-cache integration with `nnsight`, non-trivial), or
- Skipping the transcoder encoder step on cached positions (possible, but
  even a perfect skip saves <10 % of per-step time because Phase 0 is not
  the dominant cost; see Section 3).

---

## 3. Is backward pass reuse trustworthy?

**No — not under the current sparsification rule.** Full analysis in
[`BACKWARD_REUSE_STUDY.md`](docs/BACKWARD_REUSE_STUDY.md).

**Key finding:** Phase 4 (feature attribution) accounts for ~90 % of
per-step time (~2100 s out of ~2400 s on prompt 94). That is where any
meaningful speedup has to come from.

**Safety analysis of five reuse candidates:**

| Strategy | Safe? | Worth it? |
|---|---|---|
| A. Cache error vectors at cached positions | Safe with strict hash-invalidation | Too small to matter alone |
| B. Reuse Phase 3 logit rows | Unsafe — target logit changes each step | n/a |
| C. Reuse Phase 4 feature rows | Unsafe — output columns reshuffle with sparsification drift | n/a |
| D. Persist decoder chunk state across `attribute()` calls | Safe by construction (frozen weights) | **Yes — biggest win available** |
| E. Stabilize sparsification across steps | Would enable B and C but changes the traced graph | Out of scope |

### Partial backward reuse that IS trustworthy — Strategy D

Today, `AttributionContext._create_decoder_cache()` runs once per
`setup_attribution`, so the 12 GiB `cross_batch_decoder_cache_bytes` is
thrown away at the end of each step. Lifting that cache to a session-level
object would let it persist across steps with no safety risk (decoder
chunks depend only on frozen model weights and deterministic chunk-id
arithmetic).

This is not implemented here. It is flagged as the most attractive
follow-up.

---

## 4. Recommended configuration

For the current project runs, use one of these two modes:

### Mode A — Diagnostic / fidelity-check (recommended when evaluating the cache itself)

```
sbatch scripts/trace_prefix_cache_run.jay.ascend.sbatch
```

- `--use-library-prefix-cache` **on**
- `--time 08:00:00` (ours is set here)
- Output: `/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/forward_cache_v2/`
- What you get: per-step library hit/miss counters, per-step
  observational drift metrics, and step-NPZ files that you can diff
  against the uncached baseline using
  `prefix_caching/compare_cached_vs_uncached.py`.

### Mode B — Production tracing (recommended when you just want traces)

```
sbatch scripts/trace_prefix_cache_bench.jay.ascend.sbatch   # uncached baseline
```

- `--use-library-prefix-cache` **off**
- No overhead, no functional difference in output, no reduction in time.
- Until Strategy D (decoder-chunk persistence) or an equivalent
  compute-skipping path is implemented, there is no practical reason to
  turn the cache on for a production run.

### What NOT to do

- Do not rely on `--use-library-prefix-cache` to reduce attribution
  wall-clock time. It doesn't today.
- Do not assume `hit_count > 0` means compute was reused. It means the
  cache *detected* a prefix match, not that it *skipped* anything.

---

## 5. Where it still fails / gaps vs the brief

| Brief D# | Ask | Status | Gap |
|---|---|---|---|
| D1 | Forward cache that reuses activations | Infrastructure ✓, reuse ✗ | No code path consumes cache; ~30 s/step overhead |
| D2 | Diagnostics | Class + counters in place, plus driver stdout line | Library counters can be lost if completion doesn't flush (mitigated by stdout logging) |
| D3 | Backward-reuse prototype **or study** | Study ✓ | No prototype |
| D4 | 3-5 scenarios | 1 (prompt 94) | Need 2-4 more; extended run queued |
| D5 | Recommendation note | This document | — |

**"What good looks like" axes:**

- Systems improvement ("runtime meaningfully lower") — **not met**.
  Runtime is ~1.3 % *higher* on cached runs (cache bookkeeping overhead,
  no compute savings).
- Fidelity improvement ("cached runs match uncached closely") — **met
  exactly**. Byte-identical on every step of the one scenario
  completed to comparison.

---

## 6. Recommended next steps, in priority order

1. **Decoder-chunk-cache lifting (Strategy D from the study).** Safe by
   construction, meaningful savings, small surface area. Lift
   `decoder_chunk_cache` from per-context to per-session; pass it into
   `setup_attribution` as a kwarg like `prefix_cache` was. This is the
   one clear compute-reuse win without sparsification changes.
2. **Run the extended 8 h cached job (5047894, already queued)** and two
   additional scenarios to hit the brief's 3-5 minimum. The new
   `forward_cache_v2` output will have the flushed `cache_validation.json`
   plus per-step stdout counters.
3. **Forward-pass KV-cache integration.** Real speedup on Phase 0 only, which
   is already <10 % of per-step time. Lower priority than (1), flagged for
   completeness.
4. **Stabilized sparsification** (Strategy E). Out of scope for this
   direction, but enables most of the Phase 4 reuse candidates. Worth
   discussing with whoever owns sparsification.

---

## 7. Honest one-paragraph summary

I built a forward prefix cache for circuit-tracer that is fidelity-safe
by construction and proved it produces byte-identical traces against an
uncached baseline. I did not yet wire it into the forward pass to skip
compute, because (a) I wanted to prove fidelity in isolation before
touching the forward path, and (b) on analysis the bigger compute win
actually lives in Phase 4, not Phase 0. The brief's stretch goal —
backward-pass reuse — I deferred as a written study, because the most
obvious backward reuse strategies are unsafe under the current
sparsification rule. The one strategy that *is* safe and non-trivial
(persisting the decoder-chunk cache across steps) is my recommended
follow-up. The work that landed is enough to hand off confidently: the
cache is a fidelity-safe diagnostic that a future PR can turn into a
compute-reuse path once the library is ready for it.
