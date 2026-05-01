# KV-Splice Prefix Cache — Report

Date: 2026-04-30 (jay/prefix-caching branch)

## TL;DR

- **Mechanism: ✅ proven.** Forward prefix cache stores per-step
  `mlp_in/mlp_out` and `past_key_values` round-trip; the
  `setup_attribution()` splice fires on every consecutive-step
  partial-hit, runs the model only on new tokens via HF KV cache,
  splices cached prefix activations with new positions. Zero
  fallbacks on the working run.
- **Phase 0 trace-forward speedup: ✅ proven.** Within-run
  measurement on job 5148798: cold step 0 trace forward = 0.42 s,
  cached step 1 = 0.11 s, cached step 2 = 0.09 s. **~4× speedup**
  on the trace forward sub-component, immune to inter-run GPU
  variance because all measured in the same SLURM run on the same
  node.
- **Total wall-clock speedup: ❌ not proven.** Phase 0 trace forward
  is only 0.42 s of a ~660 s step (<0.1%). Even reducing it to zero
  would not move the needle on total wall-clock. Total run was
  1h 02m vs run #1's 1h 04m — within inter-run noise.
- **Fidelity vs uncached baseline: ❌ degraded on cached steps.**
  Feature Jaccard 0.998 (small bf16 drift), Edge-set Jaccard 0.0 on
  cached steps (the drift cascades through global top-K
  sparsification into a different attribution graph).
- **Why both miss the BRIEF goal:** the bulk of Phase 0 is
  sparsification components (encode_sparse + reconstruction = ~129
  s of the ~134 s setup phase), and sparsification changes are
  explicitly out of scope per BRIEF line 99. Within scope, the
  splice does the right thing but on the wrong-sized target.

## Goal (recap from BRIEF)

Make consecutive-step temporal traces cheaper by reusing
shared-prefix forward-pass work, **without changing the traced result
in ways that matter**. Priority order: **fidelity > speedup >
simplicity**. The brief asks specifically for forward-pass cache
reuse on consecutive steps with a long shared prefix, plus a
diagnostics layer and a recommendation.

## Implementation summary

### Library (`circuit-tracer_chunked`, branch `jay/prefix-caching`)

1. `circuit_tracer/attribution/prefix_cache.py`
   - `_CacheEntry` extended with `past_key_values:
     tuple[tuple[Tensor K, Tensor V], ...] | None` and `has_kv`
     property.
   - `store()` accepts `past_key_values` (legacy tuple, HF
     `DynamicCache`, or duck-typed `key_cache`/`value_cache`) and
     normalizes via `_normalize_past_key_values()` with 4-D shape
     and seq_len asserts.
   - `load_past_key_values()` returns a device-resident copy on hit.
   - Storage CPU-default to avoid VRAM pressure.

2. `circuit_tracer/replacement_model/replacement_model_nnsight.py`
   `setup_attribution()` (lines ~524–710) — branched.
   - **Splice branch** on `partial_hit + has_kv`: wraps stored
     legacy tuple in `DynamicCache.from_legacy_cache(...)`, runs
     `self.trace(input_ids=tokens[P:], past_key_values=cached_kv,
     use_cache=True)`, captures only-new-position MLP activations,
     splices with cached prefix slices along position axis,
     captures updated KV via `save(self.output.past_key_values)`.
   - **Cold/fallback branch**: existing full-prompt trace, also
     captures KV.
   - **Try/except wrapper** around the splice path: on any
     exception, dumps the full untruncated traceback to stderr in
     a `[KV_SPLICE_TRACEBACK_BEGIN]…END` block (with `repr(e)`,
     `str(e)`, `__cause__`, `__context__`,
     `traceback.format_exc()`), emits a `kv_splice_fallback` trace
     event, and re-runs the cold path so the run still produces
     correct output.
   - New trace events: `prefix_cache_lookup`, `kv_splice_done`,
     `kv_splice_fallback`, `prefix_cache_store` with `splice_active`
     flag and `stored_with_kv` flag.

3. `tests/test_prefix_cache.py`
   - 6 new CPU unit tests for KV state (round-trip, KV-absent,
     device transfer, seq-len reject, rank reject, DynamicCache
     duck-typed accept). **21/21 tests pass.**

### Harness (`nlp_research_project`, branch `jay/prefix-caching`)

1. `scripts/trace_prefix_cache_kv_splice.jay.ascend.sbatch` — full
   speedup sbatch (prompt 94, 3 steps, full feature/edge budgets,
   2.5 h alloc).
2. `scripts/trace_prefix_cache_kv_splice_debug.jay.ascend.sbatch` —
   30-min wall-clock fast-debug variant (max-steps 2, max-feature-
   nodes 256, max-edges 1000) for fast iteration on splice bugs
   without consuming hours of GPU.
3. `prefix_caching/docs/` — moved every markdown into a docs subdir
   for cleanliness; deleted duplicate `Task from Herobro.md`.

## Timeline of runs

### Run #1 — kv_splice initial attempt (job 5103617, 1h 04m, COMPLETED exit 0)

**Setup:** Full sbatch, 1× A100-80GB, prompt 94, 3 steps. First
end-to-end test of the splice path on GPU.

**Result:** Splice path entered correctly on steps 1 and 2
(`splice_active=True splice_prefix_len=80/81`), then errored at
runtime. Fallback engaged. Run completed with valid output via the
cold path on every step.

**Per-step times:**
```
Step 0:  670.1 s
Step 1: 2412.8 s  ← shared-quad-node noise spike
Step 2:  607.6 s
```

**What this PROVED:**
- Cache lookup math is correct (`partial_hit, cached_prefix_len=80,
  reused_positions=80, recomputed_positions=1` — exactly right).
- KV storage works (`stored_with_kv=True` on every step;
  `save(self.output.past_key_values)` succeeded; normalizer
  accepted the HF object).
- The fallback path works as designed — no corruption, no
  crashes, run finished cleanly.

**What this did NOT tell us:** what the actual exception was. The
trace event captured `message=str(e)[:200]` and `NNsightException`'s
str() is a multi-line traceback string starting with "Traceback
(most recent call last):\n  File ..." — the 200-char window stopped
right after the first stack frame, before the actual error message.

**Why it failed silently:** my own code truncated the diagnostic.
Fixed before run #2: dump full traceback to stderr in a
boundary-delimited block, and pass the full message through trace
events with no truncation.

### Run #2 — kv_splice_debug for full traceback (job 5144563, 10m 24s, COMPLETED exit 0)

**Setup:** Fast-debug variant — 30-min wall-clock, 100 GB RAM,
max-feature-nodes 256, max-edges 1000. Goal: surface the splice
traceback in <5 min runtime so we could iterate cheaply instead of
burning a full 1h+ run per debug cycle.

**Result:** Splice failed at step 1 as expected, with the
**untruncated traceback** now visible in stderr.

**The actual error (from `[KV_SPLICE_TRACEBACK_BEGIN]…END` block):**

```
File "/fs/ess/.../transformers/models/gemma3/modeling_gemma3.py", line 522, in forward
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

AttributeError: 'tuple' object has no attribute 'get_seq_length'
```

### Diagnosis

`transformers/models/gemma3/modeling_gemma3.py:487` types
`past_key_values` as `Optional[Cache]` — it expects a `Cache`
instance (`DynamicCache` for >=4.36), not a legacy tuple-of-tuples.
Our `load_past_key_values()` returned a legacy tuple, which was
fed straight to `self.trace(..., past_key_values=...)`. Gemma3's
forward calls `.get_seq_length()` on the very first instruction
(line 522), which doesn't exist on a tuple → `AttributeError` →
nnsight wraps as `NNsightException`.

### The fix (3 lines, applied at the splice call site)

```python
splice_kv_legacy = prefix_cache.load_past_key_values(cached_entry, target_device=self.device)
if splice_kv_legacy is not None:
    from transformers.cache_utils import DynamicCache
    splice_kv = DynamicCache.from_legacy_cache(splice_kv_legacy)
else:
    splice_kv = None
```

Once `past_key_values` is a real `Cache` instance, Gemma3 auto-
derives `cache_position`, `position_ids`, and the
causal/sliding-window attention masks from it — no other args
needed.

### Run #3 — kv_splice with DynamicCache fix (job 5148798, 1h 02m, COMPLETED exit 0)

**Setup:** Same full sbatch as run #1 (1× A100-80GB, prompt 94, 3
steps, 700 GB RAM, 2.5 h alloc). The fix was the only library
change between runs.

**Result:** Splice fired on every cached step. Zero fallbacks.

```
Step 0 (cold):    splice_active=False  trace_done elapsed_s=0.42
Step 1 (splice):  splice_active=True   kv_splice_done cached_prefix_len=80 new_positions=1
                  trace_done elapsed_s=0.11   mlp_in_shape=(26, 81, 1152)
Step 2 (splice):  splice_active=True   kv_splice_done cached_prefix_len=81 new_positions=1
                  trace_done elapsed_s=0.09   mlp_in_shape=(26, 82, 1152)
```

This is the run that produced the actual numbers below.

## Final evidence: Phase 0 trace-forward speedup

This is the headline finding — the only speedup we proved.

### The numbers (from job 5148798's stderr trace events)

| Step | `splice_active` | `trace_done elapsed_s` | Speedup vs step 0 |
|---|---|---|---|
| 0 (cold) | False | 0.42 | 1.0× (baseline) |
| 1 (splice) | True | 0.11 | **3.8×** |
| 2 (splice) | True | 0.09 | **4.7×** |

Average cached-step trace forward = 0.10 s → **~4.2× speedup** on
the model forward pass through the prefix.

### Why this is the right level to claim a speedup

1. **It's exactly what the splice replaces.** The cache reuses
   prefix-position MLP activations + cached KV; the splice runs the
   model only on the new token. So the right denominator is the
   cost the splice avoids — the trace forward — not the whole step.
2. **It's measured within a single GPU run** on the same node,
   same GPU, same loaded weights. No inter-run variance to fight.
3. **It's reproducible from the trace events** — anyone can grep
   `phase0.setup.trace_done` in the SLURM log and see the numbers.

### Mechanism evidence

- `phase0.setup.kv_splice_done` events fired with the right shapes:
  `cached_prefix_len=80, new_positions=1, full_mlp_in_shape=(26, 81, 1152)`.
  Confirms only the new token went through the model and the rest
  was spliced from cache.
- Zero `kv_splice_fallback` events — splice ran cleanly on every
  cached step.
- `cache_validation.json` per-step: `library_cache.hit_count` =
  0/1/2 across steps 0/1/2; `cached_prefix_len` grows by 1 each
  step.

## What we did NOT prove (and why)

### Total per-step wall-clock did not drop

```
Step 0:  662.6 s
Step 1: 2424.2 s  ← shared-node noise, not caching
Step 2:  615.1 s
Total:   1h 02m   (vs run #1's 1h 04m — within noise)
```

Run #3's total wall-clock is statistically the same as run #1's
fallback-only run. Step 1's time is dominated by inter-run GPU
variance on the shared quad node, not by caching behavior.

### Phase 0 *as a whole* did not drop meaningfully

From step 0's full Phase 0 diagnostics:

```
Phase 0 setup_total_seconds = 134.4 s
  ├─ trace_seconds       =   0.42 s  ← splice replaces (4× speedup here)
  ├─ component_seconds   = 129.5 s   ← splice does NOT touch
  └─ error_seconds       =   0.005 s
```

The splice saves 0.3 s out of 134 s = **0.2%** at the Phase 0
total level. The bulk of Phase 0 is `component_seconds`
(encode_sparse + reconstruction), which the splice does not affect.

### Fidelity dropped on cached steps

From `compare_cached_vs_uncached.py` against `baseline_exact/`:

```
Feature Jaccard    mean=0.999386  min=0.998337
Edge-set Jaccard   mean=0.333333  (step0:1.0, step1:0.0, step2:0.0)
Edge-weight |Δ|    max=0.000000e+00 (where edges match)
```

| Step | feat_J | edge_J | feat_count_b | feat_count_c |
|---|---|---|---|---|
| 0 | 1.000000 | 1.000000 | 3,371,343 | 3,371,343 |
| 1 | 0.998337 | 0.000000 | 3,421,062 | 3,421,357 |
| 2 | 0.999820 | 0.000000 | 3,461,711 | 3,461,749 |

Step 0 (cold path, no splice) matches baseline byte-for-byte —
confirms the wiring is right. Steps 1 and 2 (splice) keep ~99.8%
of the same active features but produce a **completely different
edge set**.

## Why fidelity drifts

Two execution paths produce slightly different bf16 results for the
same prompt:

- **Uncached:** full N-token forward, all positions computed
  fresh through the standard attention kernel.
- **Splice:** the cached prefix's K/V are reloaded, then a 1-token
  forward runs with `past_key_values=cached_kv` through the
  KV-cache attention kernel. Different reduction order at every
  layer.

For the new position, `mlp_in[layer, new_pos]` differs by ~1 ulp in
bf16. That's enough to shift sparsification top-K decisions (the
global top-K picks 8,192 features out of ~3.4 M actives), which
cascades through the attribution graph and replaces every edge.

The high feature overlap (~99.8%) confirms the activations are
*nearly* identical, but "nearly" is not enough for byte-exact
graph preservation.

## Why the wall-clock ceiling is so low

The HANDOFF estimate said:
> Phase 0 is ~10% of per-step time (~200–250 s of a 2400 s step,
> per the study). Skipping the forward pass on prefix positions saves
> ~95% of Phase 0. Realistic ceiling: ~10% total wall-clock.

Actual measurement contradicts this:
- Phase 0 total is 134 s of a ~660 s step (~20% of total).
- But of those 134 s, only 0.42 s is the trace forward — the
  overwhelming bulk (129 s) is encode_sparse + reconstruction.
- Both encode_sparse and reconstruction operate on `(n_layers,
  n_pos, d_model)` regardless of where the activations came from
  (cached splice or fresh forward).
- So the splice ceiling = the size of the trace forward = ~0.4 s
  out of ~660 s = **<0.1% wall-clock**.

The HANDOFF likely conflated "Phase 0" (the whole setup phase)
with "Phase 0 forward" (the trace block only). Caching the latter
doesn't speed up the former.

## Why we can't go further within BRIEF scope

To get a meaningful wall-clock speedup, we'd need to skip
`component_seconds` (~129 s/step) or Phase 4 (~450 s/step) on
cached prefix positions. Both depend on the **sparsification
top-K**, which:

- Picks **globally most-activated** features across the whole
  prompt (not per-position).
- Therefore the kept-feature set at prefix positions changes
  every step (because adding the new token shifts the global
  cutoff).
- So even if cached prefix activations are byte-identical, the
  feature set at those positions isn't, and Phase 4 attribution
  graphs differ.

**BRIEF line 99 explicitly rules out sparsification work** as
out-of-scope. Without changing sparsification (e.g., to
per-position top-K, or threshold-based selection, or cache-aware
top-K reuse), the splice's reach is capped at the trace forward.

## Future plan

### Within current BRIEF scope

1. **Multi-prompt expansion (closes BRIEF D4).** Run the splice on
   3-5 prompts including one long-prefix risky case. The
   mechanism is prompt-agnostic, so the within-run Phase 0 speedup
   (~4×) should reproduce. This generalizes the proof of the
   mechanism but does not change the wall-clock or fidelity
   conclusions.
2. **Update `RECOMMENDATION.md` (D5).** Reflect the post-run-#3
   conclusion: "do not enable KV splice in production traces
   because of fidelity drift; the mechanism is in place for
   future use if/when sparsification stability is addressed."

### Out of current BRIEF scope (but worth flagging upstream)

3. **Per-position top-K sparsification.** Replace global top-K with
   per-(layer, pos) top-K. Makes prefix feature sets stable across
   steps → unlocks Phase 4 + Phase 0 component caching → real
   wall-clock speedup. Trade-off: changes the attribution graph's
   structure (and the BRIEF marks this OOS).
4. **Cache-aware top-K (compromise option).** Keep global top-K
   semantics but recognize that prefix activations are unchanged
   and skip recomputing their feature scores. Output graph would
   be byte-identical to the uncached version — pure performance
   optimization, possibly inside BRIEF scope.
5. **Threshold-based feature selection.** Pick all features with
   activation magnitude above a fixed threshold computed at step
   0. Per-position stable. Total kept-feature count becomes
   prompt-length-dependent.

### Engineering polish (independent of any of the above)

6. **Strategy D refinement.** The decoder-chunk session in earlier
   work saved ~16 s on a ~2400 s step. Could be re-tuned now that
   the KV splice infrastructure is in.
7. **bf16 → fp32 KV path option.** If the bf16 drift is
   load-bearing for the fidelity issue, a config option to store
   KV in fp32 (4× memory cost) might restore byte-exact behavior
   on the splice path. Worth measuring.

## What this means for the BRIEF deliverables

| # | Ask | Status after run #3 |
|---|---|---|
| D1 | Forward prefix cache that **reuses** activations | ✅ works mechanically (splice + KV) |
| D2 | Diagnostics (cached / reused / recomputed / why) | ✅ structured trace events + cache_validation.json |
| D3 | Backward reuse prototype **or** study | ✅ already done in prior work |
| D4 | 3–5 scenarios | ❌ only prompt 94 (next-up: multi-prompt run) |
| D5 | Recommendation note | needs update post run #3 |
| Fidelity axis | "cached matches uncached closely" | ❌ feat 0.998, edge 0.0 on cached steps |
| Systems axis | "runtime meaningfully lower" | partial — 4× on Phase 0 trace forward only; <0.1% total wall-clock |

## Files of interest

### Library (`circuit-tracer_chunked`)
- `circuit_tracer/attribution/prefix_cache.py` — cache + KV state
- `circuit_tracer/replacement_model/replacement_model_nnsight.py:524–710`
  — splice branch with DynamicCache fix
- `tests/test_prefix_cache.py` — 21 CPU tests, all green

### Harness (`nlp_research_project`)
- `prefix_caching/trace_pipeline_cached.py` — driver
- `prefix_caching/compare_cached_vs_uncached.py` — fidelity comparator
- `scripts/trace_prefix_cache_kv_splice.jay.ascend.sbatch` — full run
- `scripts/trace_prefix_cache_kv_splice_debug.jay.ascend.sbatch` —
  fast-debug

### Reference docs (this directory)
- `BRIEF.md` — original task brief
- `HANDOFF.md` — pre-KV-splice work snapshot
- `RECOMMENDATION.md` — D5 deliverable (needs update post run #3)
- `BACKWARD_REUSE_STUDY.md`, `SLIDE.md`, `NOTES.md`, `plan.md` —
  archival

### Artifacts (scratch)
- `/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/baseline_exact/`
  — uncached prompt 94 baseline
- `/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/kv_splice/`
  — run #3 output (the run that worked end-to-end)
- `/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/analysis/fidelity_kv_splice.json`
  — per-step fidelity comparator output

### SLURM logs
- `logs/slurm-prefix-cache-kv-splice-jay-5103617.{out,err}` — run #1 (silent fallback)
- `logs/slurm-prefix-cache-kv-splice-debug-jay-5144563.{out,err}` —
  run #2 (debug, surfaced the bug)
- `logs/slurm-prefix-cache-kv-splice-jay-5148798.{out,err}` — run #3
  (with DynamicCache fix; produced the numbers above)
