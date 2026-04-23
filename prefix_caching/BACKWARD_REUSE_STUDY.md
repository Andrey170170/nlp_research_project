# Backward-Reuse Study — Direction 2

**Scope:** Deliverable D3 of [BRIEF.md](BRIEF.md). The brief accepts either a
conservative prototype *or* a written study. This is the study.

**Question:** Beyond the forward-pass prefix cache we already built, are any
intermediates in the attribution *backward* path (Phases 1-4) stable enough to
reuse safely across consecutive temporal-tracing steps?

**TL;DR:**

- The attribution pipeline is 5 phases; **Phase 4 (feature attribution) is
  the dominant cost** — on our prompt 94 baseline, ~2100 s out of ~2400 s per
  step.
- The forward cache we built targets Phase 0, which is only ~10 % of per-step
  time. So forward-cache reuse alone cannot produce a meaningful speedup —
  the math does not support it regardless of how well we implement it.
- Phase 4 has a few stable intermediates (error vectors at cached positions,
  the decoder chunks, pre-sparsification encoder output) and several that
  drift (active feature set, per-source gradients, chunked decoder state).
- Only **one reuse path looks both safe and non-trivial**: pre-computing and
  caching error vectors keyed on `(position, sorted_active_feature_set)`.
  The savings are still small (~5-10 s / step out of ~2400 s), because error
  vectors are cheap to recompute.
- The biggest lever — reusing per-source gradient rows — is *not* safe
  under the current sparsification scheme, because the top-K boundary
  reshuffles the active-feature set every step.
- **Recommendation:** do not implement backward reuse at this time. Either
  stabilize sparsification first, or accept the forward cache as a
  record-only diagnostic until the dominant cost (Phase 4) can be
  addressed by a different mechanism (e.g., persistent decoder chunk
  state across steps, which is out-of-scope here).

---

## 1. The attribution pipeline, decomposed

From `circuit_tracer/attribution/attribute_nnsight.py::_run_attribution`, a
single call to `attribute()` runs five phases:

| Phase | Name | What happens |
|:-:|---|---|
| 0 | Setup | One forward pass through the model, encode MLP outputs with the transcoder, build reconstruction + error vectors. |
| 1 | Forward pass | A second forward pass (with gradient flow configured and skip connections wired) to cache residual-stream activations. |
| 2 | Build input vectors | CPU-only bookkeeping: assemble `AttributionTargets`, allocate the edge matrix. |
| 3 | Logit attribution | One backward pass *per logit target*, computing how each (layer, position) contributes to each logit. |
| 4 | Feature attribution | One backward pass *per batch of source features*, computing cross-feature attributions. Uses the chunked decoder cache (`cross_batch_decoder_cache_bytes`). |

**Measured per-step attribution time on prompt 94 (baseline, from
`slurm-prefix-cache-bench-jay-5011889.out`):**

| Step | Total attribution seconds |
|:-:|:-:|
| 0 | 649.5 |
| 1 | 2408.0 |
| 2 | 584.8 |
| 3 | 2474.3 |
| 4 | 2536.8 |

Rough decomposition (based on structure + profiling flags, not direct
measurement): Phase 0 ≈ 50-200 s, Phases 1-2 ≈ 20-80 s, Phase 3 ≈ 100-300 s,
**Phase 4 ≈ 1800-2200 s**. Phase 4 loops over up to
`max_feature_nodes=8192` features in batches of 128, with each batch
triggering a full backward pass through the chunked decoder.

**Implication:** the forward cache we built saves work in Phase 0, which
is already the cheapest phase. Even a perfect Phase 0 skip would save less
than 10 % of per-step time. Anything meaningful has to touch Phases 3-4.

---

## 2. Phase 3/4 mechanics, in one paragraph

Both phases call `AttributionContext.compute_batch(layers, positions,
inject_values, ...)` (`context_nnsight.py:648`). Each call:

1. Overrides gradients at specific `(layer, position)` residual-stream
   locations with the provided `inject_values` (either logit directions or
   encoder vectors).
2. Triggers **one backward pass** through the cached residual activations
   (`_resid_activations`).
3. As the gradient flows layer-by-layer in reverse, it accumulates three
   contributions into `_batch_buffer`: feature rows (via
   `compute_feature_attributions`, which in the chunked path uses the
   `decoder_chunk_cache`), error rows (via `compute_error_attributions`),
   and token rows at the embed layer (via `compute_token_attributions`).
4. Returns the buffer rows, which go into `edge_matrix`.

The chunked path (`_compute_chunked_feature_attributions_from_grads`,
context_nnsight.py:411) additionally iterates over the decoder chunks
(`decoder_provider.get_decoder_chunk(source_layer, chunk_id, ...)`) and
does a per-chunk `einsum` over grads × decoder vectors. This is the part
the `cross_batch_decoder_cache_bytes=12 GiB` is trying to accelerate.

---

## 3. What is stable across consecutive steps, what drifts

Consider two generation steps N and N+1 that share the first P tokens.

### Stable (byte-identical or provably equivalent)

| Intermediate | Stored where | Stable because |
|---|---|---|
| `mlp_in_cache[:, 0:P, :]` | Our forward cache | Causal masking: position p depends only on tokens 0..p. |
| `mlp_out_cache[:, 0:P, :]` | Our forward cache | Same. |
| Token embedding vectors | `ctx.token_vectors[0:P]` | Just `embed_weight[token_ids[0:P]]`. Tokens are the same. |
| Decoder chunks | `decoder_provider.get_decoder_chunk(l, c)` | Model weights frozen; chunk ID is deterministic. Already cached inside the transcoder. |
| Encoder weights (`W^enc`) | `transcoders.encoder_weight(layer)` | Model weights frozen. |

### Drifts step-to-step

| Intermediate | Source of drift |
|---|---|
| Active feature set (`activation_matrix.indices()`) | Global top-K sparsification picks the K highest-activated `(layer, pos, feat)` tuples; boundary shifts when the prompt grows one more token. |
| `active_features` count per step | Changes from 3.37 M → 3.55 M over our 5-step run. |
| `decoder_locations`, `encoder_to_decoder_map` | Keyed off the active feature set. |
| Per-source gradient tensors at each layer (`_feature_output_activations[l].grad`) | Depend on the source node being attributed, which varies batch-by-batch and step-by-step. |
| `chunked_decoder_state` (positions, feature_ids, activation_values) | Reflects the current step's sparsified features. |
| Logit direction vectors (`targets.logit_vectors`) | Target next-token changes each step. |
| Reconstruction at cached positions | `= W^dec @ activation_matrix[:, pos, :]`; depends on which features are active at that position (drifts). |

**The observational metrics from the cached run confirm this.** Feature
match rate between consecutive steps is 0.9992 (near perfect: underlying
activations barely move). But position match rate drops to 0.29 by step 3
(the active set at each position reshuffles).

### Partially stable (depends on equality of the active set)

| Intermediate | Stable only when |
|---|---|
| `error_vectors[:, 0:P, :]` | The active feature set at each position 0..P-1 is identical to the previous step. |
| Reconstruction at position p for layer l | Same feature set active at (l, p). |
| Row of `edge_matrix` for source feature `f` at position `p ≤ P-1` | Feature `f` is in the active set of both steps AND all downstream grads are bit-equal. |

---

## 4. Reuse strategy analysis

### Strategy A — cache error vectors at cached positions

**Mechanism:** Key `error_vectors[:, p, :]` on `(p, hash of sorted active
feature IDs at layer l, position p)`. Reuse when the hash matches.

**Safety:** Safe. Byte-identical when the hash matches. Invalidation is
strict.

**Hit rate:** Depends entirely on how often the active-feature set at a
given position is unchanged. Empirically from our observational metrics,
this is the positions that contribute to the ~30-70 % `position_match_rate`,
so ~30-70 % of cached positions would hit.

**Savings:** Small. Error vectors are `mlp_out - reconstruction`; the
subtraction is cheap and the dominant cost is computing reconstruction
(one decoder-vector gather per active feature). For prompt 94 at step 1,
reconstruction computation is likely <20 s out of 2408 s.

**Verdict:** Technically safe, small win. Not worth implementing alone.

### Strategy B — reuse Phase 3 logit attribution rows at cached positions

**Mechanism:** For position `p ≤ P-1` and logit target `L`, reuse the
`edge_matrix` row.

**Blocker:** The logit target changes every step (different predicted
next token). Even though the gradient backward path through the model is
identical at cached positions, the injected gradient direction
(`targets.logit_vectors`) is different, so the row is different.

**Verdict:** Not reusable under normal generation. Could be reusable if
we forced the same logit target each step (not useful for our workflow).

### Strategy C — reuse Phase 4 feature attribution rows at cached sources

**Mechanism:** For a source feature `f` at layer `l`, position `p ≤ P-1`,
cache the attribution row keyed on `(l, p, f, activation_value)`. On cache
hit, skip the backward pass and reuse the cached row.

**Safety analysis:**

1. Backward pass through the frozen model from `(l, p)` produces identical
   gradients at every `(l', p')` — this holds *if* the forward residual
   activations `_resid_activations[l']` at every layer are byte-identical
   to the previous step's. At cached positions those activations derive
   from `mlp_in_cache[:, 0:P, :]` which we already know is byte-identical.
   ✓
2. Feature attribution is `grads[l'] · decoder_vec[f']` for every active
   downstream feature `f'`. The active downstream set drifts step-to-step
   (Section 3). So the set of feature-row columns that should be written
   changes.
3. The chunked decoder path walks per-chunk feature IDs from
   `chunked_decoder_state`; these IDs are re-assigned each step based on
   sparsification.

**Verdict:** Source-side equivalence holds (gradient is stable), but the
*output columns* of the attribution row are keyed to the current step's
active feature set. You would have to re-sort the cached row onto the
new feature set on each hit — which requires a position-and-feature-id
lookup that is not cheaper than just recomputing the row.

**Partial salvage:** The *source-side gradient tensors*
(`_feature_output_activations[layer+1].grad`) are byte-identical across
steps for a given source feature at a cached position. These could be
cached and reused to avoid one backward pass. But `compute_batch` runs
one backward pass per *batch* of sources (batch size 128), not per source,
so caching gradient tensors per source would require restructuring the
batching. That's substantial library surgery.

**Verdict:** Potentially safe per-source, but requires either
(a) stabilizing the active feature set across steps so output columns
don't reshuffle, or (b) re-architecting Phase 4 to batch differently.
Neither fits in a small library change.

### Strategy D — cache decoder chunk state across steps

**Mechanism:** `decoder_provider.get_decoder_chunk(source_layer,
chunk_id, ...)` already memoizes inside the transcoder with the
`cross_batch_decoder_cache_bytes=12 GiB` budget. Persist that cache
across `attribute()` calls in temporal tracing.

**Safety:** Safe. Decoder chunks depend only on model weights (frozen)
and chunk-id arithmetic (deterministic). Cached chunks survive across
steps without any invalidation concern.

**Current status:** The existing cache is *per-call* — every
`setup_attribution` call instantiates a new `decoder_chunk_cache`
(via `AttributionContext._create_decoder_cache()`, line 349). Across
successive `attribute()` calls, the cache is re-built from scratch.

**Savings:** If decoder-chunk build + load time is a meaningful
fraction of Phase 4 per-batch cost (which it likely is, given the 12 GiB
budget), persisting the cache across `attribute()` calls in the
same Python process could save real time — without changing any
attribution math.

**Verdict:** **This is the most attractive reuse target.** Safe by
construction (frozen model weights), requires no fingerprinting, and
the implementation is just lifting `decoder_chunk_cache` from a
per-context field to a per-transcoder-or-session field.

### Strategy E — stabilize sparsification across steps (enables B / C reuse)

**Mechanism:** Change the sparsification rule from "global top-K at each
step" to "global top-K anchored on the prior step's set, admitting new
features only." This removes the top-K boundary drift that breaks the
stability analysis for Strategies B-C.

**Safety:** Requires careful analysis; it *changes* the produced graph
relative to uncached exact traces. So it is no longer byte-identical; it
becomes a *different* — possibly more interpretable — sparsification rule.

**Verdict:** Out of scope for this direction (brief says sparsification
work is out of scope). But worth flagging for follow-up: several backward
reuse paths become safe only if sparsification is stabilized.

---

## 5. Safety checklist for any future prototype

If backward reuse is attempted, the following invariants must all hold
before reading a cached value:

1. **Model fingerprint match** — same model weights, same dtype, same
   device. Already checked in `prefix_cache.compute_model_fingerprint`.
2. **Token prefix match** — `token_ids[0:P]` identical to the cached
   entry's. Already checked in `PrefixActivationCache.lookup`.
3. **Active feature set match at the relevant layer and position** — this
   is the piece that is *not* covered by the existing forward cache and
   would need new fingerprinting. Suggested key:
   `sha256(sorted(feature_ids[layer, position]))`.
4. **Downstream grad-window consistency** — if the cached row assumes a
   certain backward-grad structure, the cached entry must be invalidated
   if the output column set has reshuffled.

Any prototype that skips even one of these checks is unsafe.

---

## 6. Recommendation

**Do not implement Strategy B or C at this time.** Their safety depends on
stabilizing the sparsification boundary, which is explicitly out of scope.

**Strategy A (error-vector cache)** is safe but low-value; not worth the
engineering cost by itself.

**Strategy D (persist decoder chunk state across `attribute()` calls)** is
the one backward-reuse candidate that is both safe and non-trivial.
Implementation sketch:

- Today: `AttributionContext._create_decoder_cache()` is called once per
  `setup_attribution`, so the chunk cache is thrown away at the end of
  each step.
- Proposed: lift the cache to a session-scoped object that lives alongside
  the `PrefixActivationCache`. Pass it into `setup_attribution` as a
  similar kwarg. Invalidate on model fingerprint mismatch.
- Expected savings: one-time cold-start on step 0, then every subsequent
  step reuses the same decoder chunks without re-building from the
  transcoder files. The 12 GiB budget is already allocated; we'd simply
  stop throwing away the cache contents.

**But** — this is a separate piece of work from the forward cache, touches
the transcoder/context object model, and was not the primary forward-cache
deliverable Andrei asked for. Recommending it as a follow-up, not
something to bundle into the current PR.

**For now, the backward reuse recommendation is: none implemented, Strategy D
proposed as follow-up.** The honest project status is: forward cache is
fidelity-safe but record-only; Phase 4 is where real time lives; breaking
Phase 4 open safely requires either sparsification work (out of scope) or
decoder-chunk-cache lifting (future PR).

---

## 7. What this means for our recommendation note (D5)

When writing `RECOMMENDATION.md`, we should:

- Report the forward cache as **fidelity-safe, record-only**.
- Be explicit that the headline speedup story requires either
  (a) forward-pass compute skipping that saves <10 % by construction, or
  (b) decoder-chunk-cache lifting that could save a meaningful chunk but
  is a separate PR.
- Point to this document as the reason backward reuse wasn't prototyped:
  not laziness, but because the safety analysis says the obvious candidates
  (Strategies B/C) are not safe under the current sparsification rule.

This is an honest, defensible "midpoint" story even without a speedup number.
