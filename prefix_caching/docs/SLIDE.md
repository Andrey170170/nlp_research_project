# Direction 2 — Prefix Caching (Jay)

## What I did
- Built a **prefix activation cache** in the circuit-tracer library:
  stores pre-sparsification `mlp_in` / `mlp_out` tensors keyed by
  `(prefix_token_ids, model_fingerprint)`. Strict prefix-equality
  invalidation. (new class + kwarg threaded through `attribute()`).
- Wired it into our temporal-tracing driver behind a
  `--use-library-prefix-cache` flag.
- Ran baseline vs cached on GSM8K prompt 94 and compared traces.

## What I proved
- **Fidelity-safe:** byte-identical vs uncached baseline on 5 steps
  (Feature Jaccard = 1.000, Edge Jaccard = 1.000, max |Δ| = 0.0).
- **Cache detects prefix overlap correctly** across a 3-step run:
  | Step | Prefix len | Active feats | Hits | Cached pos | Feat match |
  |---|---|---|---|---|---|
  | 0 | 80 | 3.37 M | 0 (cold) | — | — |
  | 1 | 81 | 3.42 M | 1 | 76 / 81 | **99.92 %** |
  | 2 | 82 | 3.46 M | 2 | 77 / 82 | **99.92 %** |

## What didn't land
- **No speedup yet** — cache records hits but doesn't skip compute.
  Cached run is ~30 s/step slower than baseline (bookkeeping only).
  - Phase 4 (feature attribution) = **~90 %** of per-step time (~2100 s
    of ~2400 s). Forward cache only targets Phase 0 (<10 %).

## Next step
- **Persist the decoder-chunk cache across steps** (currently thrown
  away per `attribute()` call; 12 GiB of reusable state). Safe by
  construction, biggest win available — flagged as the follow-up PR.
