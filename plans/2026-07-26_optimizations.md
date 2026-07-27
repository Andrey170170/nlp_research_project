# Exact-trace optimization plan: active-row residency and Phase-0 selective I/O

Status: active-row residency and its fused Phase-0 seed are implemented and
exact-H200 verified; work item J is the next optimization path
Date: 2026-07-26
Branch: `perf/exact-trace-loop` (both project and sibling worktrees)
Scope: sibling `../circuit-tracer_chunked` chunked attribution path + project
knob plumbing, candidate profiles, and numeric-precision provenance

Work items, in priority order:

| Item | Subject | Status |
|---|---|---|
| A | Active-row decoder residency — removes the Phase-4 decoder scan | **implemented; exact H200 pass** |
| B | Project knob plumbing and `plt-active-rows-v1` profile | **implemented** |
| B2 | **Parity confound: every candidate changed `decoder_chunk_size`** | fixed for active-row exact; general guard proposed |
| C | Source-layer fusion (`coalesced_bounded` rung) | proposed, gated on A |
| D | Confirm or refute the per-visit cost model | proposed, parallel to A |
| E | Phase-1 batch cap is never applied to CLT — the long-prefix blocker | diagnosed, fix proposed |
| F | Separate startup cost from attribution cost in reporting | proposed |
| G | Pin float32 matmul precision (no silent TF32) | **implemented** |
| H | `max_feature_nodes` as the real scaling knob | decision, not a task |
| I | Provider agnosticism: enforce the caste boundary in the harness | proposed, after A |
| J | Phase-0 coalesced safetensor row-range loading | **proposed; next bottleneck** |

A and B are now the verified incumbent. **B2 must still be read before any
profile value is set** because exact comparisons must keep the baseline chunk
size pinned. J is now the primary performance path. E remains independent and
decides whether long-prefix CLT is possible. I generalises E and is the
prerequisite for a second model family.

Companion documents:

- `docs/performance_optimization_loop.md` — the measurement loop, gates, frozen
  baseline registry, and the results of the five candidates already tested.
- `AGENTS.md` — durable operating policy. Login-node safety and SLURM-only rules
  in that file apply to every step below.

This plan does not change governor profiles, calibration observations, launch
defaults, or the frozen baseline registry. Those are separately reviewed actions.

## Problem statement

Five performance candidates have been tested on the branch (16 GiB decoder
cache, cross-batch VJP tape, gather-before-cast, 512-row frontier coalescing,
fenced decoder page prefetch). Four of the five optimize **how decoder pages
move**. The tape won (391s -> 307s on 1B PLT `361_base`); the other three were
rejected for speed.

The measurements below indicate that decoder page movement is the wrong target,
because the pages should not be read at all.

### Measured evidence

From `perf-plt-tape-w2-c65536-20260725-01` (1B PLT `361_base`, Granite job
`1654070`), scratch path
`granite/sweep/performance_optimization/.../perf_gemma3_1b_plt_361_base/run.log`:

- `n_layers=26, d_model=1152, d_transcoder=262144`
- one full decoder traversal = 104 chunks = `15,703,474,176` bytes
- Phase 4 loaded `226,492,416,000` bytes (~14.4 traversal-equivalents)
- every source layer logs `total_chunks=4` at `decoder_chunk_size=65536`,
  i.e. all 262,144 decoder rows of each layer are read on every traversal
- Phase 3 performs one further complete traversal (15.47s for one batch)

The original plan incorrectly treated `max_feature_nodes=8192` as the complete
active decoder universe. It is the selected-frontier budget. On the exact 1B
PLT `361_base` gate, Phase 0 contains 66,158 active occurrences and 29,973
unique `(source_layer, feature_id)` rows. The final occurrence-aligned resident
table is `66,158 x 1152 x 2 B = 152,428,032` bytes; the keyed unique Phase-0
seed is `29,973 x 1152 x 2 B = 69,057,792` bytes.

The waste diagnosis still holds, but the denominator is 152.4 MB of resident
occurrences (69.1 MB unique), not 18.9 MB. Before residency, Phase 4 read
226 GB to consume those rows. At `c4096`, the observed Phase-0 reconstruction
still touches 1,660 decoder pages and logically materializes 15,665,725,440
bytes because sparse feature IDs cover nearly every page.

### Derived cost model

Define chunk visits as `n_layers x ceil(d_transcoder / decoder_chunk_size) x
n_execution_batches`. Against the three strict Granite baselines at
`c4096` (`chpc-baseline-gemma-stack-20260709-03`, and the 12B run
`chpc-baseline-gemma3-12b-20260710-04`):

| variant | visits | measured | ms/visit |
|---|---:|---:|---:|
| 1B PLT b128 | 106,496 | 2,805.7s | 25.4 |
| 4B PLT b128 | 139,264 | 5,471.7s | 38.5 |
| 12B PLT b64 | 393,216 | 23,051.7s (Phase 4 20,787s) | 52.9 |

Fitting `per_visit = fixed + chunk_bytes / BW` to the 1B and 12B rows gives
`fixed ~= 13.6 ms`, `BW ~= 0.8 GB/s`, which predicts the 4B row to within 3%
(39.8 vs 38.5 ms). Residual non-scan time is ~105s (1B), ~110s (4B), ~2,250s
(12B).

**This model is a fit over three points and must be confirmed, not assumed.**
Work item D below exists to confirm or refute it cheaply. If it is right, the
PLT decoder scan is essentially the entire PLT cost and is ~99% waste.

Note that the scan term is prefix-length independent. Long-prefix cost is
currently hidden behind it; after this change, prefix length becomes the
dominant term.

### 2026-07-26 execution update

All runs below used Granite job `1657613` on one H200, exact fidelity,
`decoder_chunk_size=4096`, and live isolated project/runtime worktrees whose
source hashes were checked before and after each run. They are engineering
gates, not frozen baseline promotions.

The first exact active-row run,
`perf-plt-active-rows-exact-dev-20260726-01`, verified the contraction but
falsified the original 65-90 second model:

- 287.26s end to end;
- exact compact parity: feature/all-edge/top-256 Jaccard 1.0, weighted Jaccard
  0.999999995889084, normalized edge-magnitude L1 deviation
  4.110916e-09, and target-token match 1.0;
- Phase 0: 137.41s, 1,660 decoder-page loads, 15,665,725,440 logical bytes;
- standalone active-row build: 57.54s and the same 1,660 pages/15.67 GB again;
- resident table: 66,158 rows / 152,428,032 bytes;
- Phase 4: 62.97s with zero decoder-page loads/bytes;
- peak framebuffer: 26,111 MiB.

The duplicate build traversal was then fused into Phase 0. Corrected exact
runs `perf-plt-active-rows-fused-exact-dev-20260726-02` and `-03` both passed:

| run | end to end | Phase 0 | Phase 3 | Phase 4 | framebuffer |
|---|---:|---:|---:|---:|---:|
| `...-02` | 94.47s | 15.67s | 0.25s | 60.38s | 26,111 MiB |
| `...-03` | 98.36s | 15.32s | 0.26s | 64.16s | 26,111 MiB |

Both fused runs report `build.source=phase0_fused_seed`, zero incremental
build traversal/page loads/bytes, 1,660 shared Phase-0 page loads and
15,665,725,440 shared logical bytes, 29,973 unique seed rows / 69,057,792
bytes, 152,428,032 H2D materialization bytes, zero missing keys, and zero
Phase-4 decoder loads. Exact parity matches the first run.

The 15-second Phase-0 values are warm-page-cache observations and must not be
substituted for the earlier 137.41-second cold-ish observation. The verified
claim is that fusion removes the second traversal exactly. J targets the sole
remaining Phase-0 traversal and requires explicit cache-state provenance.

## Work item A — active-row decoder residency (sibling)

**Goal.** Materialize the decoder rows for the fixed active feature set once,
before Phase 3, and remove the page loop from the chunked attribution path.

**Where.** `circuit_tracer/attribution/context_nnsight.py`, method
`_compute_chunked_feature_attributions_from_grad_batches_impl`
(currently lines ~960-1180). The page loop is lines ~1056-1155.

**Why it is safe to precompute.** The active set is fixed at Phase 0.
`self.chunked_decoder_state` holds `source_layers`, `feature_ids`, `positions`,
and `activation_values`; `_build_chunked_layer_spans`
(`context_nnsight.py:541`) already derives contiguous per-layer row spans from
it, and rows are required to be sorted by source layer.
`self._produced_feature_range` only windows that ordering. Phase 4 varies which
gradients are contracted, never which decoder rows exist.

### A1. New owner: `ActiveDecoderRows`

Add a small module (suggested:
`circuit_tracer/attribution/nnsight/active_decoder_rows.py`) owning:

- `rows`: dense `[n_active_rows, n_slots, d_model]` in the provider dtype,
  laid out in the existing global row ordering so `layer_rows` indexing is a
  contiguous slice, not a gather;
- per-layer span reuse from `_build_chunked_layer_spans`;
- byte accounting, device placement, and deterministic release;
- a `get_diagnostic_snapshot()` reporting `active_row_count`,
  `active_row_bytes`, `build_seconds`, `build_traversal_bytes`, and
  `residency_device`.

For PLT `n_slots == 1` (`decoder_output_topology="same_layer"`,
`single_layer_transcoder.py:474`). For CLT `n_slots` is `n_layers - source_layer`,
which makes the compact object larger — see Risks.

### A2. Build path

Two candidate sources for the rows. **Use the first.**

1. **One page traversal, then gather** (primary). Iterate the same chunks the
   current code would visit, gather the active rows out of each page, and drop
   the page. Cost is exactly one traversal (~15s at 1B, ~45s at 4B, ~90s at 12B)
   against 17-128 traversals saved. It reuses the existing, already-validated
   page reader, so the resulting rows are bit-identical to today's by
   construction.
2. **Row-selective safetensors reads** (do not use initially).
   `SingleLayerTranscoder._get_decoder_vectors`
   (`single_layer_transcoder.py:172`) and
   `CrossLayerTranscoder._get_decoder_vectors`
   (`cross_layer_transcoder.py:755`) already support this, but the PLT helper
   `_slice_rows` (`single_layer_transcoder.py:36`) issues one mmap slice per row
   in a Python loop. That is a correctness-preserving but slow path and it
   changes the read pattern. Leave it as a later optimization of A2, not part of
   the first landing.

Build must happen after Phase 0 completes and before Phase 3 runs, since Phase 3
uses the same contraction implementation and should benefit too.

### A3. Contraction rewrite

Replace the `for chunk_position, chunk_id_tensor in enumerate(ordered_chunk_ids)`
loop with a single per-source-layer contraction over the layer's contiguous row
span. Keep, unchanged:

- the `for source_layer in range(...)` ordering,
- the `for output_layer in relevant_output_layers` ordering,
- the `for output_layer_grads, batch_buffer, grad_batch_index in grad_batches`
  ordering,
- the `einsum("batch position d_model, position d_model -> position batch")`
  contraction and its `d_model` reduction,
- the `batch_buffer[write_rows] +=` accumulation semantics.

Delete: the per-chunk `.item()` synchronisations
(`int(chunk_id_tensor.item())`, `int(ordered_chunk_counts[...].item())`), the
`decoder_vectors[row_chunk_local_feat_ids]` gather, the
`chunk_local_feat_ids` arithmetic, and the `DecoderPagePrefetch`
`get`/`schedule`/`finish` calls on this path.

**Exactness argument.** Today each row is written by exactly one chunk
(`chunk_rows` partitions `layer_rows`), so removing the chunk split reorders no
accumulation. The einsum reduces only over `d_model`, which is unchanged per
row. This candidate therefore has a real chance of passing `exact`, unlike
every candidate landed so far. Verify it; do not assume it — torch may select a
different reduction kernel at larger tensor sizes. If `exact` fails but
`bounded` passes, that is still a valid result: record it and say so plainly.

Keep `row_subchunk_size` honoured as a bound on the contraction width so the
bounded fallback rung still exists.

### A4. Bounded fallback and admission

Per `AGENTS.md`, an optimized rung must retain a bounded fallback and refuse
early rather than OOM. Add:

- `decoder_active_row_max_bytes`: hard ceiling on the compact object. If the
  estimated size exceeds it, do not build, log an explicit refusal reason, and
  fall through to the existing page-scan path unchanged.
- The estimate is `n_active_rows x n_slots x d_model x itemsize` and is exactly
  computable before any allocation.
- Never silently degrade: emit a recorded plan revision, as the bounded
  architecture section of `docs/performance_optimization_loop.md` requires.

### A5. Telemetry

Existing decoder counters (`decoder_load_count`, `decoder_load_bytes`,
`decoder_chunk_request_count`, `decoder_cache_hit_count`) must stay wired and
should go to near-zero for Phase 4 when residency is active. That is the primary
falsification signal for this whole plan: if Phase 4 `decoder_load_bytes` drops
to ~0 and the run does not get much faster, the cost model in this document is
wrong and work item D's profile is the next step, not more page work.

Add the `ActiveDecoderRows` snapshot to the existing Phase 4 diagnostics
aggregation in `phases/phase4_diagnostics.py` alongside the tape and prefetch
counters.

## Work item B — project plumbing and candidate profile

Follow the exact chain used by `9b26258` (VJP tape) and `e6a6daa` (prefetch):

1. `src/nlp_research_project/exact_trace_bench/config.py` — add
   `decoder_active_row_residency` (default `False`) and
   `decoder_active_row_max_bytes` (default `0` = disabled) to the knob defaults
   near line 74.
2. `scenarios/base.py` — add both names to the knob passthrough tuple at
   line ~228.
3. `trace_runtime/request.py` — read both from the scenario near line 260.
4. `full_answer/runner.py` — forward both near line 675 and validate them
   alongside the `decoder_chunk_size` checks at line ~493.
5. `perf_cli.py` — add candidate profile `plt-active-rows-v1`:
   **`decoder_chunk_size=4096`** (match the frozen baseline — see work item B2;
   do not inherit 65536 from `plt-bounded-fast-v2`),
   `cross_batch_decoder_cache_bytes=0`, `feature_vjp_tape_batch_window=1`,
   `decoder_page_prefetch_depth=0`, `decoder_active_row_residency=True`,
   `decoder_active_row_max_bytes=1073741824`, plus the Phase 1/3/4 caps from
   `plt-bounded-fast-v2`.

Note the tape window is deliberately **1** in this profile. If the decoder scan
is gone, the tape's reason to exist is gone; keeping it on would confound the
measurement. Add `plt-active-rows-tape-v1` (window 2) as a separate profile so
the interaction can be measured independently.

Do **not** add this profile to `BOUNDED_ONLY_CANDIDATE_PROFILES` — the point is
to run it under `--fidelity exact`.

## Work item B2 — the parity confound: every candidate changed `decoder_chunk_size`

**Read this before setting any profile value. It invalidates the "bounded-only"
label on all five landed candidates.**

Parity metrics across every recorded PLT candidate:

| Run | mechanism | chunk | worst-step feature Jaccard |
|---|---|---:|---:|
| `perf-plt-cache16g-c65536-...-01` | 16 GiB decoder cache | 65536 | 0.9946432919405892 |
| `perf-plt-streaming-baseline-c65536-...-03` | one-batch streaming | 65536 | 0.9946432919405892 |
| `perf-plt-tape-w2-c65536-...-01` | VJP tape window 2 | 65536 | 0.9946432919405892 |
| `perf-plt-frontier512-c65536-...-01` | 512-row frontier | 65536 | 0.9946432919405892 |
| `perf-plt-tape-prefetch-fenced-c65536-...-01` | fenced page prefetch | 65536 | 0.9946432919405892 |
| `perf-plt-bounded-fast-v1-...-01` | same as v2 | **32768** | **0.9973180543703523** |

Five architecturally unrelated mechanisms produce graphs that agree to sixteen
significant digits. Changing only `decoder_chunk_size` produces a different
value, and moves monotonically toward 1.0 as the chunk shrinks toward the
baseline's setting.

The frozen registry `exact_trace_performance_granite_20260709.json` points at
`chpc-baseline-gemma-stack-20260709-03`, which ran at **`decoder_chunk_size=4096`**.
Every candidate profile in `perf_cli.py` sets 32768 or 65536.

`decoder_chunk_size` is classified `compatibility-mixed` in
`docs/knob_api_taxonomy.md` — "mixed: reduction order plus decoder work" — and
EXPERIMENTS.md already records that "changing `decoder_chunk_size` can cause
small compact-output drift" while "within a fixed `decoder_chunk_size`,
cache-size changes were exact."

Conclusion: **the deviation tracks the chunk size and nothing else. None of the
five candidates has been shown to introduce any numerical deviation at all.**
They were labelled bounded-only because of a confound that sat in every profile,
not because of the mechanism under test. The loop has so far been structurally
incapable of detecting whether a candidate is exact.

Actions:

1. Set `decoder_chunk_size=4096` in `plt-active-rows-v1` to match the frozen
   baseline. With active-row residency the chunk size only affects the one-time
   build traversal, so this costs nothing — Track A converts a semantically
   live, drift-causing knob into a build-time detail.
2. Re-run one existing candidate (the tape) at `c4096` under `--fidelity exact`
   as a control. If it passes, the bounded-only labels in
   `docs/performance_optimization_loop.md` need correcting, and the tape may be
   promotable on stronger evidence than currently recorded.
3. Treat any future profile that changes a `compatibility-mixed` knob relative
   to the baseline as untestable for exactness until that knob is matched.

## Work item C — source-layer fusion (follow-up rung, not first landing)

Once the pages are gone, the remaining structure is
`n_layers x n_execution_batches` contraction calls (884 at 1B b256/tape2, vs
3,536 today). Fusing the source-layer axis into one contraction per grad batch
is the `coalesced_bounded` rung already described in
`docs/performance_optimization_loop.md`. It changes the `d_model` reduction
grouping and is therefore **bounded-only by construction**.

Land and measure A first. C is only worth doing if A lands and the residual is
still dominated by contraction call count.

## Work item D — confirm or refute the cost model

Cheap, login-safe where possible, and worth doing **in parallel** with A:

1. Add a CUDA-event or `torch.profiler` breakdown of one Phase-4 frontier
   behind a diagnostic flag, reporting separately: page fetch, row gather,
   dtype cast, grad gather, einsum, buffer accumulate.
2. Run it once on `plt-hard` with `plt-bounded-fast-v3` (cache resident, so
   page I/O is ~0). That run measured Phase 4 = 224.40s with the decoder fully
   in HBM. The cost model says that time is per-visit fixed overhead. The
   profile will say what it actually is.

This matters because the 13.6 ms fixed term is the part of the model with the
weakest evidence, and it is the part A removes.

## Work item E — Phase-1 batch cap is never applied to CLT (landed diagnosis)

**This is a wiring gap, not a missing mechanism. The fix already exists.**

`perf-prefetch-clt-control-20260725-01` records `cuda_peak_alloc=83.34 GiB`,
`cuda_reserved=91.62 GiB`, and 94,643 MiB peak framebuffer — on a 1B model with
a 124-token prompt. The comparable PLT run peaks at 22.73 GiB allocated /
24.84 GiB reserved / 26,265 MiB framebuffer. Same model, same prompt, 3.7x the
memory, with the peak landing at Phase 1 forward completion.

Root cause, confirmed from the run log:

```text
phase1_trace_batch_policy=legacy (effective=legacy, size_max=None, size_max_effective=None)
```

The spike control is implemented and working —
`circuit_tracer/attribution/nnsight/phase1_policy.py` provides
`cap_effective_batches` with `phase1_trace_batch_size_max`, and
`_resolve_phase1_trace_batch_config` correctly falls back to `legacy` when the
cap is requested without a size. CLT simply never requests it:

1. `_PHASE1_TRACE_BATCH_POLICY_DEFAULT` in the sibling is `legacy`.
2. The project's `config.py` knob defaults do not set
   `phase1_trace_batch_policy` at all.
3. The CLT baseline scenario (`scenarios/chpc_baseline.py`) does not set it, and
   runs at `batch=1000` (`logit_batch_size=1000`).
4. Every PLT candidate profile in `perf_cli.py` sets
   `phase1_trace_batch_policy="cap_effective_batches"` and
   `phase1_trace_batch_size_max=128` — but
   `_candidate_overrides` (`perf_cli.py:310`) returns `{}` for any case whose
   variant is not `gemma3_1b_plt`, so **a CLT case cannot receive the cap
   regardless of which `--candidate-profile` is selected.**

So the cap has only ever been exercised on PLT. That also explains the
EXPERIMENTS.md note that 1B CLT consumed essentially all of a `32G` request and
degraded to 240.72s, prompting the "request at least 64G" rule.

Why this matters beyond memory accounting: **this is the long-prefix blocker.**
If 124 tokens costs 83 GiB on a 1B CLT run, 1k tokens will not fit, and no
amount of decoder work changes that. Track A makes PLT fast; this decides
whether CLT at realistic prompt lengths is possible at all.

Actions:

1. Make `_candidate_overrides` variant-aware rather than PLT-only: split the
   profile table into PLT physical controls and provider-agnostic controls, and
   let CLT cases receive the latter. Do not silently apply PLT decoder knobs
   (`decoder_chunk_size=65536`) to CLT.
2. Add a CLT candidate profile that sets
   `phase1_trace_batch_policy="cap_effective_batches"` with an explicit
   `phase1_trace_batch_size_max`, and measure peak allocation and runtime
   against the current `legacy` CLT control.
3. Treat the resulting number as the input to the long-prefix question. If a
   capped 1B CLT still peaks near 80 GiB, the Phase-1 materialization itself
   needs work, not just the batch policy.
4. Record whether Phase 1 memory scales with prompt length, batch size, or both.
   That is the coefficient the 1k-prefix rows in the target table currently lack.

This is independent of Track A and can be measured on the same allocation.

## Work item F — separate startup cost from attribution cost

The GPU sample timeline shows that the first ~20 seconds of every run never
touch the GPU at all:

| Phase | 1B CLT | 1B PLT |
|---|---|---|
| GPU idle, 0% SM, **0 MiB framebuffer** | t=0-20s | t=0-21s |
| model onto GPU | t=20-24s | t=21-35s |
| attribution | t=24-87s | t=35-307s |

That segment is Python startup, torch/transformers/nnsight import, and weight
reading. It is essentially identical in both variants, so it is a constant, not
a CLT property. It is ~24% of the CLT benchmark and ~7% of PLT, and it is paid
once per process — in a real full-answer campaign over many completions it
amortizes to nothing.

`duration_seconds` in `result.json` is subprocess wall time and therefore
includes it. Action: report an attribution-only duration alongside
`duration_seconds` in `performance_report.json`, so optimization work is not
graded against a fixed ~20s constant. This is a reporting change, not an
optimization; do not spend effort shrinking import time.

Consequence for targets: a "sub-30s CLT" benchmark figure is mostly a fight
against `import torch`. State CLT goals in attribution-only terms.

## Work item G — float32 matmul precision (IMPLEMENTED 2026-07-26)

Landed ahead of the rest of this plan because it is small, independent, and
protects the exactness-critical path.

Neither repository set `allow_tf32` or `float32_matmul_precision` anywhere, so
the exact path inherited the PyTorch default. TF32 silently truncates float32
matmul inputs to a 10-bit mantissa. The pinned torch is 2.10.0, whose default
happens to be true IEEE float32 for matmul, but that default has changed across
releases and a version bump would alter exactness-critical arithmetic with
nothing catching it — a bounded gate at 0.98 Jaccard would pass straight
through a TF32 downgrade.

Implemented:

- `src/nlp_research_project/exact_trace_bench/trace_runtime/numeric_precision.py`
  — `pin_float32_matmul_precision()` forces `float32_matmul_precision="highest"`
  and disables `cudnn.allow_tf32`, then returns the resolved backend state.
  Version-tolerant reads (`allow_tf32`, `fp32_precision`) so it survives the
  ongoing PyTorch API migration.
- `trace_runtime/campaign.py` calls it at `run_campaign` entry, prints the
  resolved state, and records it in `run_config.json` under `numeric_precision`.
- `tests/test_trace_runtime_numeric_precision.py` — five tests covering the
  forced value, every exposed TF32 switch being off, idempotence, JSON-safety
  with torch version recorded, and the `run_config.json` wiring.

Rationale worth keeping durable: reconstruction error is the scientific object
here — error nodes are already in the graph. If attribution accumulation adds
its own numerical error, "the transcoder does not reconstruct this" becomes
indistinguishable from "the arithmetic drifted." EXPERIMENTS.md already records
the 12B 128 arm drifting about 1% on edge metrics from batching changes alone;
stacking dtype noise on that makes it unattributable. The cost of fp32
accumulation is near zero at 1B, where the contraction is microseconds of real
math.

Remaining question for after Track A: the contraction runs fp32 because bf16
gradients are cast up at `typed_grads = grads.to(dtype=batch_buffer.dtype)`, so
it does not use bf16 tensor cores. At 1B this is irrelevant. At 12B with a 1k
prefix it may become visible once the decoder scan is gone. If so, the answer is
fewer and larger fp32 contractions (work item C), **not** bf16 accumulation.

## Work item H — `max_feature_nodes` is the real scaling knob

`max_feature_nodes=8192` (`config.py:54`) is currently identical across 1B, 4B,
and 12B and across all prefix lengths. Total attribution cost is linear in it.
A 1k-token prompt arguably deserves more than 8192 features, and raising it
carries a directly proportional runtime price.

This is a scientific decision with a mechanical cost, not a performance knob to
tune. It should be made deliberately rather than inherited. Every target in this
document assumes 8192; state that assumption whenever a target is quoted.

## Work item I — provider agnosticism: the boundary exists, the harness ignores it

The intended split is **already defined and is good**. This is not a design gap;
it is an enforcement gap concentrated in the project-side perf loop.

### What is already correct

`docs/knob_api_taxonomy.md` defines a caste system that answers "what is
agnostic vs what needs its own knob" precisely:

| Caste | Meaning |
|---|---|
| `scenario-declared semantic` | fixed before admission; exact and bounded never change it |
| `provider-declared semantic/capability` | provider identity; the resolver consumes it and never invents it |
| `governor-derived physical` | selectable from resource conditions after a fixed-semantics parity gate |
| `compatibility-mixed` | legacy field controlling both logical and physical behaviour; must be split |
| `telemetry/artifact/debug` | never a degradation lever |

`TranscoderCapabilities` (`circuit_tracer/transcoder/provider.py`) is the runtime
half of that contract: `architecture`, `decoder_output_topology`
(`cross_layer` / `same_layer`), eleven `supports_*` flags,
`default_decoder_chunk_size`, `default_cross_batch_decoder_cache_bytes`.

The sibling hot loop honours it. `_compute_chunked_feature_attributions_from_grad_batches_impl`
branches on `decoder_output_layers_for_source`, `decoder_output_slot`, and
`supports_exact_chunked_provider` — capability queries, never model names.
`SingleLayerTranscoderSet` declares `architecture="plt"` /
`decoder_output_topology="same_layer"`; `CrossLayerTranscoder` declares
`architecture="clt"` / `cross_layer`. That is exactly the right shape, and it is
why Track A can be written once and work for both.

### Where it breaks

All in `src/nlp_research_project/exact_trace_bench/`:

1. `perf_cli.py:310` — `_candidate_overrides` branches on the literal string
   `"gemma3_1b_plt"`. AGENTS.md forbids this directly: "may branch on provider
   capabilities and topology; it must not branch on model-family or
   checkpoint-name special cases." This single line is the cause of work item E.
2. `CANDIDATE_PROFILES` is a flat `dict[str, Any]` mixing castes that the
   taxonomy separates: `decoder_chunk_size` (compatibility-mixed, semantically
   live — see B2), `cross_batch_decoder_cache_bytes` (physical override),
   `phase1_trace_batch_size_max` (compatibility/resource intent),
   `phase4_execution_batch_max_rows` (physical),
   `feature_vjp_tape_batch_window` (physical candidate). One blob keyed to one
   model string, with no record of which entries are safe to vary against a
   frozen baseline.
3. Suite definitions hardcode `Case("gemma3_1b_clt", ...)` /
   `Case("gemma3_1b_plt", ...)`, and registry keys are
   `performance/gemma3_1b_plt/361_base`.
4. `transcoder_config.py::PROVIDER_PRESETS` contains only Gemma families.

### Target shape

Replace the flat profile dict with a typed profile carrying a capability
predicate and caste-separated knob groups:

```python
CandidateProfile(
    name="active-rows-v1",
    requires=Capability(supports_exact_chunked_provider=True),
    baseline_pins={"decoder_chunk_size": 4096},   # must match the frozen baseline
    physical={"decoder_active_row_residency": True, ...},
)
```

Rules the harness should enforce mechanically:

- a profile applies to a case when the case's provider satisfies `requires`,
  never when a variant string matches;
- `baseline_pins` entries are compared against the frozen baseline's recorded
  values, and a mismatch makes the case ineligible for `--fidelity exact` with
  an explicit reason (this is B2 turned into a guardrail);
- `physical` entries are the only ones free to vary between candidate and
  baseline;
- topology-specific values (decoder paging knobs) may only be set for providers
  declaring the matching `decoder_output_topology`, so a CLT case can receive
  provider-agnostic controls such as `phase1_trace_batch_policy` without
  inheriting PLT decoder knobs. That is work item E's fix, generalised.

### What Llama 3.1 8B would actually need

Mostly harness work, not runtime work:

- **Sibling:** a transcoder set that declares `TranscoderCapabilities` honestly.
  The activation machinery already covers `jump_relu`, `relu`, and `topk`
  (`transcoder/activation_functions.py`), and the loaders already support both
  per-layer and cross-layer checkpoint layouts. A top-k PLT for Llama should
  need a loader and a capability declaration, not new attribution code.
- **Project:** a `PROVIDER_PRESETS` entry, fixtures, and a baseline registry
  keyed by something other than `gemma3_*`. Registry keys should be
  `<provider_family>/<fixture>` rather than embedding a model nickname.
- **Blocking issue:** the perf loop cannot express a non-Gemma case at all
  today, because suites, targets, and overrides are all keyed to the four
  hardcoded variant strings.

Do this restructuring **after** Track A lands. Track A is written against the
capability contract and does not depend on it; doing the restructuring first
would delay the change that matters most. But do it before adding a second model
family, because every hardcoded variant string added between now and then is
another one to remove.

## Work item J — Phase-0 coalesced safetensor row-range loading

**Goal.** Reduce the sole remaining PLT decoder traversal in Phase 0. The
cold-ish exact run measured 137.41s, 1,660 logical decoder-page loads, and
15,665,725,440 materialized bytes. Once the fused active-row path is enabled,
this is expected to be the main bottleneck.

### J1. Primary I/O design

The existing safetensor path already supports contiguous range reads:
`safe_open(...).get_slice(key)[start:stop]`. Do not reuse `_slice_rows`; it
issues one safetensor slice per row in a Python loop and is exactly the mmap
access pattern to avoid.

For each source layer after sparsification:

1. derive sorted unique active feature IDs;
2. coalesce adjacent IDs into half-open row ranges `[start, stop)`;
3. optionally merge small gaps under an explicit maximum-overfetch policy;
4. issue one safetensor slice per coalesced range and gather requested rows into
   a compact keyed table;
5. replay the existing sorted decoder-chunk groups from that table.

Only the I/O source changes. Preserve, byte for byte:

- the existing sorted chunk traversal;
- each chunk mask and active-occurrence order;
- activation scaling;
- the number, order, and row membership of reconstruction `index_add_` calls;
- bias/skip addition order;
- the keyed Phase-0 seed and its canonical provider fingerprint.

This is intentionally not "one mmap read per row." Range fragmentation is a
measured planning input. If coalescing would produce too many requests or
exceed the overfetch bound, refuse explicitly and use the current full-page
path unchanged.

### J2. Ownership and failure safety

Stage at most one bounded range (or one admitted range window) on the GPU at a
time. Release range tensors and gather intermediates on normal and exceptional
paths. The compact unique-row table may be reused as the fused Phase-0 seed;
do not build a second copy. Admission must account separately for:

- the current range staging buffer;
- the unique CPU seed;
- the final occurrence-aligned GPU resident table.

Provider/checkpoint identity remains caller-bound and is checked before any
seed-to-resident transfer.

### J3. Telemetry and claim boundary

Report Phase 0 separately from the full run:

- total Phase-0 wall time;
- range planning, safetensor read, row gather, reconstruction, and seed-capture
  seconds;
- unique row count/bytes;
- coalesced range request count and rows per range;
- merged-gap rows and overfetch bytes;
- logical requested/materialized bytes;
- fallback/refusal reason and baseline full-page count/bytes;
- Phase-0 CUDA allocated/reserved peaks and sampled framebuffer;
- end-to-end duration and whole-run framebuffer as separate fields.

Current `decoder_load_bytes` measures logical materialized tensor volume, not
storage-device physical I/O. If physical read bytes are collected, publish
them under a separate process/cgroup counter with filesystem page-cache state
and its limitations. Never relabel logical safetensor bytes as physical disk
traffic.

### J4. Independent larger-chunk candidate

Keep larger decoder chunks as a separate candidate/profile. Changing
`decoder_chunk_size` changes the partition and boundaries of Phase-0
`index_add_` calls, so it can change floating-point accumulation order even
when decoder values are identical. Do not combine it with coalesced row-range
loading in the first experiment. It needs its own exact-parity result against
the `c4096` frozen baseline; bounded parity is not exact semantics.

### J5. Validation and H200 gates

Synthetic gates:

- Phase-0 reconstruction is `torch.equal` with range loading on/off;
- selected rows equal full-page gathers for contiguous, fragmented, duplicate,
  empty, and final-short-range cases;
- the sequence and row membership of reconstruction `index_add_` operations is
  unchanged;
- fallback/refusal preserves output and releases all range owners;
- provider mismatch and injected second-range failure are cleanup-safe.

H200 gates, on fixed source and workload:

1. run the current fused profile as the control;
2. run `plt-phase0-coalesced-rows-v1` under exact fidelity;
3. repeat the winner twice on the same allocation;
4. record cache state explicitly and retain at least one cold-ish comparison;
5. require exact worst-step parity, zero post-Phase-0 decoder loads, no
   framebuffer regression above 26,111 MiB (26,113 MiB audited bound), and
   separately report Phase-0 time, logical bytes/pages/ranges, end-to-end time,
   and framebuffer.

## Explicitly deferred: multi-GPU

Multi-GPU is **out of scope for this plan and should not be started until A is
landed and measured.**

Rationale, recorded so it is not re-litigated: the cache-disabled architectural
baseline measured whole-run GPU SM utilization of 10.86% (p95 16%, max 21%),
118.63 W average power, 26,113 MiB peak framebuffer of ~140 GiB available, and
16.79% average utilization across twelve allocated CPUs. Adding a second H200 to
a workload that uses ~10% of the first one buys nothing; it multiplies idle
hardware and adds partitioning, collective, and provenance complexity on top of
a single-device inefficiency that is still unfixed.

The per-second GPU samples sharpen this further. The run is not uniformly idle;
it is idle punctuated by short bursts. On `perf-plt-tape-w2-c65536-20260725-01`
the GPU sits at 13-15% SM for most of Phase 4, with brief spikes to 43%, 70%,
and 50% at frontier boundaries. Adding a second device multiplies the idle
fraction, not the useful work.

Revisit multi-GPU only when single-H200 utilization is high enough that the
device is genuinely the constraint. Concretely: reconsider when a PLT run holds
SM utilization above ~50% sustained through Phase 4, or when the compact
decoder plus a real prefix (1k+) at 12B genuinely exceeds one device's HBM.
Until then the answer to "should we get a second H200" is no.

Re-measuring the SM profile after Track A lands is the specific evidence that
settles this. Add it to the acceptance reporting for the Track A runs.

## Validation protocol

Order matters. Do not skip ahead.

### Login-safe (no GPU, no model load)

```bash
uv run ruff check .
uv run ty check .
uv run pytest tests/test_exact_trace_perf_cli.py \
  tests/test_exact_trace_bench_knob_taxonomy.py \
  tests/test_c2_project_trace_runtime.py \
  tests/test_full_answer_runner_aggregate.py \
  tests/test_trace_runtime_numeric_precision.py
uv run exact-trace-perf list
uv run exact-trace-perf run all --dry-run
```

In the sibling worktree:

```bash
uv run ruff check .
uv run pytest tests/test_chunked_decoder_optimizations.py \
  tests/test_nnsight_phase4.py tests/test_feature_vjp_tape.py \
  tests/test_decoder_page_prefetch.py tests/test_nnsight_phase3.py
```

New sibling tests required (synthetic fixtures, no model load), following the
pattern in `tests/test_feature_vjp_tape.py`:

- `ActiveDecoderRows` builds rows that are element-wise identical to the rows
  the page path would have gathered, for both a same-layer (PLT) and a
  cross-layer (CLT) synthetic provider;
- the contraction produces bit-identical `batch_buffer` output with residency on
  vs off, on a synthetic fixture, for at least one multi-chunk layer;
- `decoder_active_row_max_bytes` below the required size refuses to build,
  records the refusal, and falls back to the page path with unchanged output;
- byte accounting and cleanup gauges return to zero after release;
- `_produced_feature_range` windowing still selects the correct rows.

New project test required: `plt-active-rows-v1` appears in the profile table,
is **not** bounded-only, and forwards both new knobs end to end.

### H200 (SLURM only)

Inside a Granite `rai-gpu-grn` H200 allocation, on the isolated worktrees, with
no edits during a run:

```bash
# 1. CLT regression control first — cheapest signal that nothing broke.
uv run exact-trace-perf run clt-smoke \
  --run-id perf-active-rows-clt-control-01 --fidelity exact

# 2. The headline run, under exact.
uv run exact-trace-perf run plt-hard \
  --candidate-profile plt-active-rows-v1 \
  --fidelity exact \
  --run-id perf-plt-active-rows-exact-01 \
  --run-goal "Active-row decoder residency: remove Phase-4 decoder scan."

# 3. Only if exact fails, repeat under bounded and record the failure explicitly.
uv run exact-trace-perf run plt-hard \
  --candidate-profile plt-active-rows-v1 \
  --fidelity bounded --run-id perf-plt-active-rows-bounded-01

# 4. Repeat the winning configuration on the same allocation and source state.
uv run exact-trace-perf run plt-hard \
  --candidate-profile plt-active-rows-v1 \
  --fidelity exact --run-id perf-plt-active-rows-exact-02

# 5. Tape interaction, measured separately.
uv run exact-trace-perf run plt-hard \
  --candidate-profile plt-active-rows-tape-v1 \
  --fidelity exact --run-id perf-plt-active-rows-tape-01

# 6. Full gate before treating it as a viable 1B candidate.
uv run exact-trace-perf run all --run-id perf-active-rows-all-01
```

## Acceptance criteria

A run is a success only if all of the following hold:

1. `parity_passed` is true at the declared fidelity on the worst aligned step,
   not only the means;
2. fused residency reports `build.source=phase0_fused_seed`, zero incremental
   build traversal/page loads/bytes, and zero Phase-4 decoder loads/bytes;
3. the 1B gate reports 29,973 unique seed rows / 69,057,792 bytes and 66,158
   occurrence-aligned resident rows / 152,428,032 bytes, rather than treating
   the 8,192 frontier budget as the active universe;
4. end-to-end duration is reproducible across two runs on the same allocation
   and source state;
5. peak framebuffer does not increase relative to
   `perf-plt-streaming-baseline-c65536-20260724-03` (26,113 MiB);
6. the CLT control still matches its compact artifacts;
7. Phase-0 selective-I/O candidates separately report Phase-0 time,
   page/range counts, logical bytes, end-to-end time, and framebuffer.

The original **65-90s** prediction was falsified by the non-fused 287.26s run,
then reconciled by identifying and removing its duplicated decoder traversal.
The corrected warm-cache repeats are 94.47s and 98.36s. These do not establish
a cold-cache target: Phase 0 alone ranged from 137.41s cold-ish to 15.32-15.67s
warm. Work item J must report both cache state and Phase-0 time rather than
promoting the warm number as a general baseline.

## Targets and floors

Compute floor per cell is `8192 rows x one backward (~2x forward) x 2NS FLOP`,
at ~150 TFLOPS effective for 1B and ~450 TFLOPS for 12B. It assumes perfect
utilization on one H200 and the current 8192-feature cap.

| Case | Today | Floor | Target after this plan |
|---|---:|---:|---|
| 1B CLT, 124 tok | 85.8s | ~15s | sub-60s expected; sub-30s needs the 22.9s non-attribution segment |
| 1B PLT, 124 tok | 94-98s warm-cache verified | ~27s | Phase-0 range-loading target pending cold/warm measurement |
| 4B PLT, 124 tok | 5,472s | ~90s | 200-400s |
| 12B PLT, 124 tok | 23,052s | ~110s | ~35 min, then 18-25 min with larger execution envelopes |
| 1B PLT, 1k prefix | unmeasured | ~225s | 5-8 min; sub-60s is not physical |
| 12B PLT, 1k prefix | unmeasured | ~15 min | 45-60 min single-GPU |

**Dtype (resolved 2026-07-26).** The floors above assume a bf16 backward, and
that is what runs. Evidence: the trace log records `Device: cuda, Dtype:
torch.bfloat16`; there is no `autocast` anywhere in the sibling library, so
autograd produces gradients in the forward dtype; and
`exact_trace_internal_dtype` accepts only `fp32`/`fp64`
(`circuit_tracer/attribution/nnsight/numerics.py:16`) and governs the
attribution accumulators, not the model. The boundary is visible in the hot
loop as `typed_grads = grads.to(dtype=batch_buffer.dtype)`, which casts bf16
gradients up to fp32 where accumulation begins. Model forward/backward is bf16;
attribution accumulation is fp32. No floor correction is needed.

One caveat remains: prefix length barely affects cost today because the decoder
scan is prefix-independent, so the 1k-prefix rows are the least certain in the
table and remain unverified until a long-prefix gate exists.

`docs/performance_optimization_loop.md` already states that no result may be
described as a 1k-prefix, 4B/12B, or extreme-regime result until a
corresponding gate exists. That still holds. The 4B/12B rows above are
projections from the fitted model, not authorized targets.

## Risks

- **CLT compact object is larger.** For CLT, `n_slots = n_layers - source_layer`,
  so the compact object is roughly `L/2` times the PLT size — order 250 MB at 1B,
  still trivially resident, but the estimate must be computed per topology and
  not assumed from the PLT case. The `decoder_active_row_max_bytes` admission
  check is what makes this safe.
- **`exact` may fail on kernel selection.** Larger contractions may pick a
  different reduction kernel. Falling back to `bounded` is an acceptable outcome;
  silently claiming exact is not.
- **The 13.6 ms fixed per-visit term may be misattributed.** If it is something
  that survives the page removal, A will underdeliver. Work item D is the hedge
  and should run in parallel.
- **Build-time traversal on 12B is ~90s.** Acceptable against ~20,000s saved,
  but it must be measured and reported separately, not folded into Phase 4.
- **Provider-agnosticism.** Per `AGENTS.md`, branch on provider capabilities and
  topology (`decoder_output_topology`, `supports_exact_chunked_provider`), never
  on model family or checkpoint name.
