# PLT/CLT Optimization Parity Spec

Status: Current implementation spec
Last updated: 2026-06-30

## 1. Problem statement

The sibling library `../circuit-tracer_chunked` currently behaves like a
**CLT-shaped** exact-tracing stack: the exact/chunked optimizations are wired
through CLT-specific load, cache, and runtime gates, while PLT support remains a
separate path with different defaults and less direct validation.

We need one parity plan that:

- keeps the current **Gemma 3 1B CLT** baseline/defaults intact,
- adds proper support for both **PLTs** and **CLTs** under the same exact/
  chunked optimization model,
- focuses initial PLT work on **GemmaScope2 width_262k PLTs** for Gemma 3 1B/4B,
- preserves exact semantics and existing chunked optimizations,
- keeps GPU/model-loading work off login nodes.

## 2. Current findings: why the library is CLT-shaped

### 2.1 Code-path index

| Path | Current shape | Why it matters |
|---|---|---|
| `circuit_tracer/utils/hf_utils.py:121-211` | Loader dispatch infers CLT exact behavior from `repo_id`/`scan` strings and routes GemmaScope2 CLTs through `load_gemma_scope_2_clt(...)`; `exact_chunked_decoder` is effectively a CLT capability flag. | PLT/CLT capability is conflated with repository naming, so optimization availability is not modeled explicitly. |
| `circuit_tracer/utils/caching.py:308-380` | Cache save/load has a dedicated GemmaScope2 CLT branch and serializes CLTs with exact-chunked assumptions. | Cache artifacts can encode CLT-only expectations, making PLT parity and cache reuse harder. |
| `circuit_tracer/transcoder/cross_layer_transcoder.py:72-126, 1874-1974` | `CrossLayerTranscoder` is documented and instantiated around CLT semantics; `load_gemma_scope_2_clt(...)` hard-sets `exact_chunked_decoder=True`. | The main optimized exact path is CLT-first by construction. |
| `circuit_tracer/attribution/attribute.py:251-275` | The public `attribute(...)` entrypoint rejects planner/override combinations unless the caller uses the NNSight compact exact path. | This keeps the public API aligned to the CLT compact path rather than a shared PLT/CLT capability model. |
| `circuit_tracer/attribution/attribute_nnsight.py:1622-1788, 1930-2808, 3084-3887, 6964-8056` | Phase-4 locality shaping, refresh policy, row executor, row reduction, row-store cache control, and exact encoder residency are all gated on `compact_output and exact_chunked_decoder`. | These optimizations are not expressed as backend-neutral capabilities, so PLT support is incomplete by default. |
| `circuit_tracer/attribution/attribute_transformerlens.py:214-269` | Offload and profiling behavior switch on `exact_chunked_decoder`. | Backend behavior diverges depending on whether the model is treated as CLT-exact. |
| `circuit_tracer/replacement_model/replacement_model_transformerlens.py:514-530` | `decoder_provider` is only wired when `exact_chunked_decoder` is true. | PLT and CLT contexts do not share the same provider contract. |
| `tests/test_gemmascope2_chunked.py:37-81` and `tests/test_gemmascope2_chunked_gpu.py:20-85` | Current smoke coverage is centered on GemmaScope2 CLTs; the 1B GPU smoke uses `google/gemma-3-1b-pt` + `mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine`. | This is the current baseline to preserve while adding PLT parity. |
| `tests/test_attributions_gemma3_nnsight.py:548-581` | Gemma 3 coverage includes a CLT case (`270m`) and a 4B PLT case using `transcoder_all/width_16k_l0_small_affine`. | PLT smoke exists, but not yet for the requested GemmaScope2 width_262k family. |

### 2.2 Current interpretation

The code is not “CLT-only” in the absolute sense; rather, **exact/chunked
optimizations are encoded as CLT capabilities** and then reused opportunistically
elsewhere. That is why PLT support needs a first-class capability model instead of
more repo-name branching.

### 2.3 Detailed affected code paths

The highest-risk paths are below. These should be treated as the starting checklist
for implementation and review.

#### Sibling library: transcoder loading and provider APIs

| Path | Current behavior | Required PLT/CLT parity change |
|---|---|---|
| `circuit_tracer/transcoder/single_layer_transcoder.py:24-177` | `SingleLayerTranscoder` has a nominal lazy path, but its lazy keys assume cached internal names (`W_enc`, `W_dec`). | Add a GemmaScope2-native lazy reader for lowercase checkpoint keys (`w_enc`, `w_dec`, `threshold`, `b_enc`, `b_dec`) without materializing full width_262k tensors. |
| `circuit_tracer/transcoder/single_layer_transcoder.py:233-414` | `TranscoderSet.compute_attribution_components(...)` encodes all PLT layers, materializes active encoder rows, materializes active decoder rows, and reconstructs eagerly. | Add exact/chunked mode for PLTs: sparse activation matrix plus `chunked_decoder_state`, empty materialized decoder vectors, optional lazy encoder materialization, and chunked reconstruction. |
| `circuit_tracer/transcoder/single_layer_transcoder.py:519-584` | `load_gemma_scope_2_transcoder(...)` explicitly warns that lazy loading is unsupported and forces `lazy_encoder=False`, `lazy_decoder=False`. | Replace this forced eager path with a supported lazy GemmaScope2 PLT mode. Eager mode can remain available for small tests. |
| `circuit_tracer/transcoder/single_layer_transcoder.py:587-647` | `load_transcoder_set(...)` can load contiguous PLT layers, but does not expose exact/chunked capabilities on the returned set. | Add loader arguments and metadata for `exact_chunked_decoder`, `decoder_chunk_size`, cache bytes, checkpoint format, and architecture label. |
| `circuit_tracer/transcoder/cross_layer_transcoder.py:672-780` | CLT owns decoder block cache creation/clearing and cache fingerprinting. | Reuse or generalize this cache for any decoder provider. PLT cache keys can stay `(source_layer, chunk_id)`. |
| `circuit_tracer/transcoder/cross_layer_transcoder.py:896-986` | CLT `encode_sparse(...)` already supports sequential chunk/lazy-friendly Phase 0. | Mirror the memory discipline in `TranscoderSet`: avoid stacking dense `(n_pos, 262k)` activations across layers. |
| `circuit_tracer/transcoder/cross_layer_transcoder.py:1411-1465` | CLT decoder chunks have shape `(chunk_features, n_remaining_layers, d_model)`. | PLT decoder chunks should expose a compatibility shape `(chunk_features, 1, d_model)` or a provider method that maps source/output slots explicitly. |
| `circuit_tracer/transcoder/cross_layer_transcoder.py:1666-1740` | CLT exact path can skip materialized decoder vectors and optionally skip materialized encoder vectors. | Implement equivalent behavior for `TranscoderSet.compute_attribution_components(...)`. |

#### Sibling library: model/context/attribution runtime

| Path | Current behavior | Required PLT/CLT parity change |
|---|---|---|
| `circuit_tracer/replacement_model/replacement_model.py:72-122` | Factory already accepts `TranscoderSet | CrossLayerTranscoder`. | Keep public factory stable; feed it a capability-complete PLT `TranscoderSet`. |
| `circuit_tracer/replacement_model/replacement_model_nnsight.py:746-758` | Calls `compute_attribution_components(..., materialize_encoder_vecs=...)` only when `exact_chunked_decoder` is true. | Ensure PLT `TranscoderSet` supports this same signature when exact/chunked is enabled. |
| `circuit_tracer/replacement_model/replacement_model_nnsight.py:1051-1095` | Intervention/freezing path handles both 2D PLT decoder vectors and 3D CLT decoder vectors in non-chunked mode. | Keep non-chunked compatibility tests; exact/chunked PLT should avoid this materialized decoder path for width_262k. |
| `circuit_tracer/attribution/context_nnsight.py:230-248, 732-757, 985-1031` | Context already stores a decoder provider/cache, but replay assumes CLT output topology in the chunked path. | Add provider method such as `decoder_output_layers_for_source(source_layer, active_output_layers)`; CLT returns `>= source_layer`, PLT returns `source_layer` only. |
| `circuit_tracer/attribution/attribute_nnsight.py:1930-2808` | Refresh, row executor, row reduction, row-store cache control, and encoder residency are gated by `exact_chunked_decoder`. | Keep gates initially, but make `TranscoderSet.exact_chunked_decoder=True` mean PLT supports the same provider contract. |
| `circuit_tracer/attribution/attribute_nnsight.py:6964-7058` | Planner/probe path uses `ctx.materialize_encoder_vectors(...)` and chunked keys when exact mode is enabled. | Ensure PLT active encoder residency and lazy row materialization return rows matching the sparse activation ordering. |
| `circuit_tracer/attribution/attribute_nnsight.py:7826-8056` | Compact feature row-store is enabled by `compact_output and exact_chunked_decoder`. | Once PLT implements the provider contract, compact output should be enabled for PLT width_262k by default. |

#### Project harness: current repo adaptation points

| Path | Current behavior | Required adaptation |
|---|---|---|
| `trace_pipeline.py:56-58, 171-227` | Hardcoded `google/gemma-scope-2-1b-it`, CLT subfolder, and `google/gemma-3-1b-it`. | Add explicit model/transcoder config: architecture (`clt`/`plt`), repo ID, subfolder/template, model name, layer count/inference, exact/chunked knobs. |
| `explore_pipeline.py:26-27, 168-194` | Same 1B CLT assumptions for exploratory path. | Either parameterize or clearly mark CLT-only. Prefer sharing the same loader helper as `trace_pipeline.py`. |
| `src/nlp_research_project/exact_trace_bench/full_answer/runner.py:568-605, 787-793` | Shard runner only passes decoder/cache chunk knobs into `trace_pipeline.load_model(...)`. | Extend model-load knobs with model/transcoder family fields and record them in shard/token metadata. |
| `src/nlp_research_project/exact_trace_bench/cli.py:994-1109, 1310-1347` | Trace spec and launch commands carry many graph knobs but no model/transcoder architecture knobs. | Add opt-in CLI/config fields for PLT runs, or a benchmark-family manifest that runner consumes. Preserve existing CLT defaults. |
| `src/nlp_research_project/exact_trace_bench/full_answer/decoder_signature_cache.py:14-118` | Decoder-signature cache assumes GemmaScope2 CLT `params_layer_*.safetensors` with cross-layer `w_dec` shape. | Keep this as CLT-only unless/until PLT feature matching needs a separate PLT decoder-signature cache. Do not reuse blindly for PLTs. |

### 2.4 Data-shape contract to preserve

Shared sparse attribution shape:

```text
activation_matrix: sparse COO (source_layer, position, feature_id)
chunked_decoder_state:
  source_layers:      (nnz,)
  positions:          (nnz,)
  feature_ids:        (nnz,)
  activation_values:  (nnz,)
```

Architecture-specific decoder topology:

```text
CLT source layer s -> output layers s..L-1
PLT source layer s -> output layer s only
```

Compatibility target for provider chunks:

```text
CLT get_decoder_chunk(s, chunk): (chunk_features, L-s, d_model)
PLT get_decoder_chunk(s, chunk): (chunk_features, 1, d_model)
```

This contract lets Phase 3/4 row computation and compact output stay mostly
architecture-neutral while preserving the semantic difference between CLTs and
PLTs.

## 3. Proposed staged plan

### Stage 0 — Freeze the baseline

- Keep the current **Gemma 3 1B CLT** smoke/default path unchanged.
- Do not change current exact-trace defaults (`fp32`, stable row-L1 behavior).
- Record baseline repo SHAs before any serious run.

### Stage 1 — Make PLT/CLT capabilities explicit

- Replace repository-string inference with explicit transcoder capability
  detection at load time.
- Model capabilities separately from architecture label:
  - `supports_exact_chunked_decoder`
  - `supports_compact_row_store`
  - `supports_decoder_cache`
  - `supports_encoder_residency`
- Keep CLT and PLT cache formats readable, but stop using cache shape as the only
  source of truth for optimization eligibility.

Deliverables:

1. A small provider/capability surface used by attribution code. Initial fields:
   - `transcoder_architecture`: `"clt" | "plt"`,
   - `checkpoint_format`: e.g. `"gemmascope2-clt" | "gemmascope2-plt"`,
   - `exact_chunked_decoder`,
   - `decoder_chunk_size`,
   - `supports_decoder_chunk_cache`,
   - `supports_lazy_encoder_rows`,
   - `supports_lazy_decoder_chunks`.
2. Backward-compatible defaults so current CLT tests and harness runs do not need
   new arguments.

### Stage 2 — Unify the exact/chunked optimization surface

- Move shared exact-path logic behind capability checks instead of CLT-only
  gates.
- Preserve existing optimizations:
  - Phase-4 locality shaping,
  - refresh policy / row executor,
  - row reduction,
  - row-store cache control,
  - exact encoder residency.
- For PLTs, enable the same machinery where semantics match; where semantics do
  not match, fall back with explicit telemetry rather than silent divergence.

Implementation steps:

1. Implement GemmaScope2 PLT lazy readers.
   - Read encoder rows from `w_enc[:, feature_ids].T`.
   - Read decoder chunks from `w_dec[start:stop]`.
   - Keep thresholds/biases resident; these are small relative to width_262k
     encoder/decoder matrices.
2. Implement `TranscoderSet` exact/chunked mode.
   - Sequential per-layer sparse encode.
   - Chunked reconstruction that writes only to the same output layer.
   - No active decoder-vector materialization in exact/chunked mode.
   - Optional active encoder materialization controlled by existing residency
     knobs.
3. Generalize decoder replay topology in attribution contexts.
   - Replace CLT-only `output_layer >= source_layer` assumptions with a provider
     method.
   - Preserve CLT behavior exactly.
4. Reuse existing Phase 4/compact optimizations.
   - Keep `exact_chunked_decoder` as the compatibility gate for the first pass.
   - Add telemetry that reports `transcoder_architecture` so PLT/CLT results are
     distinguishable.

Do **not** rename `exact_chunked_decoder` in this implementation wave. It is
CLT-biased terminology, but changing the name would create avoidable churn. Add a
clearer alias later if needed.

### Stage 3 — Adapt the project harness

- Add explicit scenario families for:
  - current 1B CLT baseline,
  - GemmaScope2 width_262k PLTs for Gemma 3 1B/4B.
- Keep scenario metadata explicit about model family, transcoder kind, and exact
  capabilities.
- Preserve current fast-fixture defaults in the project docs and generated
  scenario conventions.

Harness implementation steps:

1. Add a shared loader config object or dict consumed by `trace_pipeline.load_model`:
   - `model_name`,
   - `transcoder_architecture`,
   - `repo_id`,
   - `clt_subfolder` or `plt_subfolder_template`,
   - `plt_width`, `plt_l0`, `plt_affine`,
   - `layer_count` or `infer_layers_from_hf_tree`,
   - exact/chunked/cache knobs.
2. Add PLT path construction for GemmaScope2:

   ```text
   transcoder_all/layer_{layer}_width_262k_l0_{small|big}{_affine}/params.safetensors
   ```

3. Record model/transcoder config in:
   - shard metadata,
   - per-token trace metadata,
   - experiment log entries for serious runs.
4. Keep decoder signature cache explicitly CLT-only unless a separate PLT cache is
   designed.

### Stage 4 — Validate with SLURM-backed tracing

- Use real SLURM jobs for any change that touches model loading, tracing, or
  exact-path semantics.
- Run one smoke per architecture family before promoting defaults.
- Compare compact outputs, frontier/row-store telemetry, and decoder-cache stats
  on the canonical prompt set.

Suggested real-run sequence:

1. **Synthetic/library parity only** on login node: no model loading.
2. **1B CLT baseline smoke** under SLURM: proves no regression.
3. **1B PLT width_262k one-token smoke** under SLURM:
   - compact output only,
   - conservative batch sizes,
   - decoder cache disabled first.
4. **1B PLT width_262k five-token benchmark** under SLURM:
   - use the existing five-token calibration selection if still desired,
   - capture Phase 0 counts/timing, peak memory, row-store size, cache stats.
5. **4B PLT width_262k one-token smoke** under SLURM only after 1B PLT memory is
   understood.
6. **4B PLT width_262k five-token benchmark** only if the one-token profile is
   tractable.

## 4. Testing strategy

### Login-safe

- config/dispatch unit tests,
- synthetic CLT/PLT load-save fixtures,
- scenario-generation and metadata tests,
- doc/index checks,
- no model downloads, no GPU use.

Concrete login-safe tests to add in `../circuit-tracer_chunked`:

1. Build tiny synthetic GemmaScope2 PLT safetensors with lowercase keys.
2. Assert lazy PLT encoder row reads match eager tensors.
3. Assert lazy PLT decoder chunks match eager `w_dec[start:stop]`.
4. Assert PLT chunked reconstruction equals existing eager PLT reconstruction on
   a tiny sparse activation matrix.
5. Assert chunked PLT attribution component output uses the same sparse feature
   ordering as eager mode.
6. Assert CLT tests still pass unchanged.

### GPU / SLURM only

- Gemma 3 1B CLT smoke (preserve current baseline),
- GemmaScope2 width_262k PLT smoke for Gemma 3 1B/4B,
- exact tracing/parity runs when runtime semantics change,
- full compact-output comparison on canonical prompts.

Real tracing tests should record:

- project repo SHA/dirty files,
- sibling repo SHA/dirty files,
- model name and transcoder artifact paths,
- exact/chunked capabilities detected at load time,
- Phase 0 active feature counts by layer/token,
- Phase 0 encode/reconstruction seconds,
- decoder chunk cache hits/misses/evictions,
- active encoder residency mode and bytes,
- peak CUDA allocation/reservation and SLURM MaxRSS,
- compact row-store bytes and Phase 4 timings.

## 5. Risks / open questions

- Is `exact_chunked_decoder` a true capability across both architectures, or a
  CLT-only specialization that needs a new shared abstraction?
- Which optimizations are genuinely PLT-compatible vs. CLT-only?
- Do the requested GemmaScope2 width_262k PLT checkpoints exist for both Gemma 3
  1B and 4B under stable repo IDs, or do we need a capability-driven loader that
  tolerates alternate paths?
- PLT `nnz` may be much larger than CLT `nnz`, so row-store and Phase 4 compute
  can dominate even after decoder memory is fixed.
- 4B width_262k PLTs may require smaller attribution batches, smaller decoder
  chunks, or a new Phase 0 microbatching path.
- Eager parity against real width_262k PLTs may be infeasible; use tiny synthetic
  parity plus real-run invariant checks.

## 6. Acceptance criteria

1. PLT and CLT both load through a capability-based exact-tracing path.
2. Existing exact/chunked optimizations still work where semantically valid.
3. Current 1B CLT baseline/defaults remain unchanged.
4. GemmaScope2 width_262k PLT support is added for Gemma 3 1B/4B.
5. Login-node validation stays model-free; SLURM is used for real tracing.

## 7. Ordered tasks

1. Inventory current CLT-only assumptions and loader/cache gates.
2. Add explicit capability metadata to transcoder loading and caching.
3. Refactor exact-path gating to use capabilities instead of CLT naming.
4. Update project harness scenario generation and docs.
5. Add synthetic parity tests for PLT/CLT dispatch.
6. Run SLURM smoke traces for 1B CLT and GemmaScope2 width_262k PLTs.
