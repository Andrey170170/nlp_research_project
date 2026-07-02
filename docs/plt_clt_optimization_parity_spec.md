# PLT/CLT Optimization Parity Spec

Status: Current implementation spec
Last updated: 2026-07-01

## 1. Problem statement

The sibling library `../circuit-tracer_chunked` currently exposes exact/chunked
tracing through CLT-shaped assumptions: load-time gates, cache layout, replay
topology, and runtime optimizations are all biased toward cross-layer transcoders.
PLT support exists, but it is not first-class and does not share a clean provider
model with CLTs.

We need a single parity plan that:

- preserves the current **Gemma 3 1B CLT** baseline semantically,
- makes the first-class abstraction architecture-neutral (`clt` vs `plt`),
- supports **GemmaScope2 width_262k PLTs** for Gemma 3 1B/4B,
- keeps exact/chunked semantics intact while allowing structure/compatibility churn
  in this worktree if it improves the API,
- keeps GPU/model-loading work off login nodes.

## 2. Scope / non-goals

### In scope

- Introduce a provider/capability model for exact/chunked transcoder usage.
- Refactor PLT/CLT exact-path gating to consume explicit architecture metadata.
- Keep CLT semantics stable while allowing naming/API cleanup.
- Add login-safe tests for provider contracts and harness configuration.
- Add SLURM-backed smoke coverage for the current CLT baseline and the requested
  PLT family.

### Non-goals

- Redesigning attribution math or changing exact-trace semantics.
- Supporting new transcoder families beyond `clt` and `plt`.
- Moving real model loading or tracing onto login nodes.
- Preserving CLT-biased public names if a cleaner abstraction is available.

## 3. Current findings: why the library is CLT-shaped

### 3.1 Code-path inventory

| Path | Current shape | Why it matters |
|---|---|---|
| `circuit_tracer/utils/hf_utils.py:121-211` | Loader dispatch infers CLT exact behavior from `repo_id`/`scan` strings and routes GemmaScope2 CLTs through `load_gemma_scope_2_clt(...)`; `exact_chunked_decoder` is effectively a CLT capability flag. | Architecture capability is conflated with repository naming, so exact/chunked eligibility is implicit. |
| `circuit_tracer/utils/caching.py:308-380` | Cache save/load has a dedicated GemmaScope2 CLT branch and serializes CLTs with exact-chunked assumptions. | Cache artifacts can encode CLT-only expectations, making PLT parity and cache reuse harder. |
| `circuit_tracer/transcoder/cross_layer_transcoder.py:72-126, 1874-1974` | `CrossLayerTranscoder` is documented and instantiated around CLT semantics; `load_gemma_scope_2_clt(...)` hard-sets `exact_chunked_decoder=True`. | The main optimized exact path is CLT-first by construction. |
| `circuit_tracer/attribution/attribute.py:251-275` | The public `attribute(...)` entrypoint rejects planner/override combinations unless the caller uses the NNSight compact exact path. | Public attribution is still keyed to a CLT-shaped compact path instead of a provider contract. |
| `circuit_tracer/attribution/attribute_nnsight.py:1622-1788, 1930-2808, 3084-3887, 6964-8056` | Phase-4 locality shaping, refresh policy, row executor, row reduction, row-store cache control, and exact encoder residency are gated on `compact_output and exact_chunked_decoder`. | Optimizations are not expressed as backend-neutral capabilities. |
| `circuit_tracer/attribution/attribute_transformerlens.py:214-269` | Offload and profiling behavior switch on `exact_chunked_decoder`. | Backend behavior diverges based on the old CLT-biased flag. |
| `circuit_tracer/replacement_model/replacement_model_transformerlens.py:514-530` | `decoder_provider` is only wired when `exact_chunked_decoder` is true. | PLT and CLT contexts do not share a single provider contract. |
| `tests/test_gemmascope2_chunked.py:37-81` and `tests/test_gemmascope2_chunked_gpu.py:20-85` | Current smoke coverage is centered on GemmaScope2 CLTs; the 1B GPU smoke uses `google/gemma-3-1b-pt` + `mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine`. | This is the current baseline to preserve while adding PLT parity. |
| `tests/test_attributions_gemma3_nnsight.py:548-581` | Gemma 3 coverage includes a CLT case (`270m`) and a 4B PLT case using `transcoder_all/width_16k_l0_small_affine`. | PLT smoke exists, but not yet for the requested GemmaScope2 width_262k family. |

### 3.2 Current interpretation

The code is not CLT-only in the absolute sense; rather, **exact/chunked
optimizations are encoded as CLT capabilities** and then reused opportunistically
elsewhere. This worktree should prefer a cleaner architecture-neutral abstraction
even if that causes local compatibility churn.

### 3.3 nnz nuance

`nnz` in this plan means **active encoder features**, not decoder fanout.

- CLTs may have more decoder fanout per active feature because one feature can map
  across multiple output layers.
- PLTs may still have equal or larger active-feature `nnz` because features are
  per-layer and are not cross-layer-compressed.

This is an empirical risk to measure, not an assertion about expected outcomes.

### 3.4 Data-shape contract to preserve

The target exact/chunked path should keep sparse attribution data explicit:

```text
activation_matrix:
  indices:        (3, nnz)  # source_layer, position, feature_id; torch COO order
  values:         (nnz,)
  shape:          (n_layers, n_pos, d_features)
  invariant:      coalesced; row order is the source of truth for active features

chunked_state:
  source_layers:   (nnz,)
  positions:       (nnz,)
  feature_ids:     (nnz,)
  activation_values: (nnz,)
  chunk_ids:       (nnz,) optional/cache-local convenience key
  invariant:       aligned 1:1 with activation_matrix.indices()/values()

decoder_chunk:
  chunk_id:        int
  feature_offset:  chunk_id * decoder_chunk_size
  weights:         (chunk_features, n_output_slots, d_model)
  output_layers:   (n_out,)
  invariant:       output_layers maps decoder_chunk[:, slot, :] to model layer

encoder_rows:
  source_layers:   (k,)
  feature_ids:     (k,)
  rows:            (k, d_model)
  invariant:       returned in exactly the requested row order
```

Output topology is architecture-specific:

```text
CLT source layer s -> output layers s..L-1
PLT source layer s -> output layer s only
```

Compatibility chunk shapes remain simple:

```text
CLT get_decoder_chunk(s, c): (chunk_features, L-s, d_model)
PLT get_decoder_chunk(s, c): (chunk_features, 1, d_model)
```

The runtime must not infer topology from tensor shape alone. It should use an
explicit source-layer/output-layer mapping supplied by the provider.

## 4. Proposed architecture-neutral provider model

### 4.1 Core types

Use `TranscoderArchitecture = Literal["clt", "plt"]` as the public architecture
label.

Introduce a capability dataclass, e.g. `TranscoderCapabilities`, with fields like:

- `architecture: TranscoderArchitecture`
- `checkpoint_format: str`
- `supports_exact_chunked_provider: bool`
- `supports_compact_row_store: bool`
- `supports_decoder_chunk_cache: bool`
- `supports_encoder_residency: bool`
- `supports_encoder_row_materialization: bool`
- `supports_lazy_decoder_chunks: bool`
- `supports_lazy_encoder_rows: bool`
- `decoder_output_topology: Literal["cross_layer", "same_layer"]`
- `default_decoder_chunk_size: int`
- `cross_batch_decoder_cache_bytes: int`
- `legacy_exact_chunked_decoder_name: bool` (temporary shim only; optional)

Introduce a provider contract, e.g. `ExactChunkedProvider`, with these
methods/properties:

```text
provider.architecture
provider.capabilities
provider.decoder_chunk_size
provider.d_transcoder
provider.d_model
provider.n_layers

provider.compute_attribution_components(..., materialize_encoder_vecs=...) -> dict
provider.get_decoder_chunk(source_layer, chunk_id, decoder_cache=None) -> Tensor
provider.decoder_output_layers_for_source(source_layer, active_output_layers=None)
  -> tuple[int, ...]
provider.decoder_output_slot(source_layer, output_layer) -> int
provider.materialize_encoder_rows(source_layers, feature_ids) -> Tensor
provider.create_decoder_block_cache(max_bytes=None, fingerprint=None) -> cache | None
provider.clear_decoder_block_cache(cache) -> None
provider.get_diagnostic_snapshot() -> dict[str, object]
```

### 4.2 Contract details

- `compute_attribution_components(...)` remains the Phase-0 entrypoint but should
  return provider-neutral keys: `activation_matrix`, `reconstruction`,
  `encoder_vecs`, `decoder_vecs`, `encoder_to_decoder_map`, `decoder_locations`,
  and, when exact/chunked is active, `chunked_decoder_state`.
- In exact/chunked mode `decoder_vecs`, `encoder_to_decoder_map`, and
  `decoder_locations` may be empty because decoder rows are replayed lazily from
  provider chunks.
- `get_decoder_chunk(...)` returns decoder weights for one source-layer/chunk
  pair. The caller must use `decoder_output_slot(...)`, not arithmetic like
  `output_layer - source_layer`, to select the correct slice.
- `materialize_encoder_rows(...)` should preserve row order from the sparse
  activation matrix.
- `decoder_output_layers_for_source(...)` replaces CLT-specific replay assumptions:
  CLT returns active output layers `>= source_layer`; PLT returns only
  `source_layer` when it is active.
- Decoder chunk cache keys should include enough provider identity to prevent CLT
  and PLT cache reuse: architecture, checkpoint fingerprint, source layer, chunk
  id, dtype, and chunk size.
- Diagnostics should surface architecture, checkpoint format, capability flags,
  chunk/cache settings, topology decisions, and fallback reasons.

### 4.3 Legacy naming policy

`exact_chunked_decoder` may remain as a temporary shim or compatibility alias, but
it should not be the primary public abstraction in new code paths.

Preferred new names:

- capability: `supports_exact_chunked_provider`,
- runtime gate: `exact_chunked_provider_enabled`,
- provider object: `decoder_provider` or `exact_provider`,
- architecture label: `transcoder_architecture`.

## 5. Implementation plan

### Phase 0 — Normalize the abstraction boundary

Goal: make the new provider model explicit before changing behavior.

Sibling library tasks:

- Add provider/capability types near the transcoder layer, preferably in a small
  module such as `circuit_tracer/transcoder/provider.py` to avoid circular imports.
- Define `TranscoderArchitecture`, `DecoderOutputTopology`, and the capability
  dataclass/protocol.
- Add adapter methods on existing CLT classes so they can satisfy the provider
  contract without changing CLT semantics.
- Add a helper such as `supports_exact_chunked_provider(transcoders)` so runtime
  gates do not need to know concrete classes.

Project harness tasks:

- Thread architecture/provider metadata through scenario config structures.
- Keep current CLT defaults unchanged at the CLI boundary for now.

Done when:

- code can ask one provider-neutral question to decide compact exact/chunked
  eligibility,
- CLT diagnostics report `architecture="clt"`, checkpoint format, chunk size, and
  cache settings,
- no PLT behavior has changed yet.

### Phase 1 — Convert CLT behavior to the new provider contract

Goal: preserve current CLT semantics while moving off CLT-biased gates.

Sibling library file-level tasks:

- `circuit_tracer/transcoder/cross_layer_transcoder.py`
  - implement the provider contract for CLTs,
  - expose `decoder_output_layers_for_source(...)`,
  - expose `decoder_output_slot(source_layer, output_layer)` as
    `output_layer - source_layer` with validation,
  - expose cache and diagnostics through the provider surface,
  - keep current exact/chunked CLT behavior identical.
- `circuit_tracer/utils/hf_utils.py`
  - return explicit architecture/capability metadata from loader selection,
  - stop inferring exact/chunked eligibility only from repo strings.
- `circuit_tracer/utils/caching.py`
  - include provider architecture/capability fingerprints in cache keys,
  - explicitly migrate or invalidate old cache artifacts rather than shaping the
    new API around historical cache layouts.
- `circuit_tracer/attribution/context_nnsight.py`
  - consume `decoder_output_layers_for_source(...)` instead of assuming cross-layer
    replay topology,
  - replace `decoder_chunk[:, output_layer - source_layer]` with
    `decoder_chunk[:, provider.decoder_output_slot(source_layer, output_layer)]`.
- `circuit_tracer/attribution/attribute_nnsight.py`
  - switch exact-path gates to capability checks,
  - rename internal metadata from CLT-specific names where practical while keeping
    the old field only as a compatibility/debug breadcrumb.

Done when:

- CLT unit/GPU smoke behavior is unchanged within existing tolerances,
- Phase-4 locality, planner, row-store, row-reduction, and encoder-residency gates
  read provider capabilities rather than `exact_chunked_decoder`,
- decoder cache fingerprints include architecture and checkpoint/provider identity.

### Phase 2 — Implement PLT provider support

Goal: make PLTs first-class under the same provider contract.

Implementation status (2026-07-01): completed in sibling library commits
`c211311`, `b47eb2a`, `e82ad4c`, `0e78ca0`, and `08ffcba` for login-safe
Phase-2 scope. The implemented path adds lazy GemmaScope2 PLT lowercase readers,
same-layer provider topology, exact/chunked PLT attribution components, HF/cache
provider metadata handling, TransformerLens provider integration, default-disabled
PLT decoder cache residency, and synthetic parity/loader tests. Real GemmaScope2
width_262k PLT tracing remains SLURM-only validation work and should be run after
Phase 3 exposes explicit project-side PLT scenarios.

Sibling library file-level tasks:

- `circuit_tracer/transcoder/single_layer_transcoder.py`
  - add lazy GemmaScope2 PLT readers for lowercase checkpoint keys,
  - read encoder rows from `w_enc[:, feature_ids].T`,
  - read decoder chunks from `w_dec[start:stop]`,
  - keep small tensors resident: `threshold`, `b_enc`, `b_dec`, optional affine
    skip,
  - implement exact/chunked `compute_attribution_components(...)` for PLTs,
  - support `get_decoder_chunk(...)` with same-layer topology,
  - support optional encoder-row materialization without eager full-matrix loads,
  - return empty materialized decoder rows in exact/chunked mode.
- `circuit_tracer/transcoder/single_layer_transcoder.py` and/or a new adapter
  module
  - return a PLT provider object that satisfies the shared contract.
- `circuit_tracer/replacement_model/replacement_model_nnsight.py`
  - call `compute_attribution_components(..., materialize_encoder_vecs=...)` based
    on provider capabilities rather than CLT-specific flags,
  - pass the provider into `AttributionContext` for both CLT and PLT exact/chunked
    modes.
- `circuit_tracer/replacement_model/replacement_model_transformerlens.py`
  - accept the provider contract directly,
  - remove the last CLT-only wiring assumptions around `decoder_provider`.
- `circuit_tracer/attribution/attribute_transformerlens.py`
  - switch offload/profiling decisions from `exact_chunked_decoder` to provider
    capabilities.

PLT implementation invariants:

- Phase 0 processes layers sequentially and does not stack dense
  `(n_layers, n_pos, 262k)` activations.
- The returned `chunked_decoder_state` row order matches
  `activation_matrix.indices()` after coalescing/filtering.
- PLT reconstruction writes only to the source/output layer.
- Width_262k real checkpoints never require eager materialization for normal exact
  tracing.

Done when:

- synthetic PLT exact/chunked reconstruction equals eager PLT reconstruction,
- lazy/eager encoder-row and decoder-chunk tests pass,
- NNSight compact exact tracing can build an `AttributionContext` for a PLT
  provider without materialized decoder vectors.

### Phase 3 — Update the harness to select architecture explicitly

Goal: make the project-side entrypoints architecture-aware and scenario-driven.

Project harness file-level tasks:

- `trace_pipeline.py`
  - accept explicit model/provider config: architecture, model name, repo id,
    revision, subfolder/template, layer count, chunk size, cache knobs,
  - route CLT and PLT through the same loader shape.
- `explore_pipeline.py`
  - either consume the same config or be explicitly CLT-only by declaration,
  - avoid duplicating loader logic.
- `src/nlp_research_project/exact_trace_bench/full_answer/runner.py`
  - pass architecture/provider metadata into shard execution,
  - record it in shard/token metadata.
- `src/nlp_research_project/exact_trace_bench/cli.py`
  - add explicit scenario/config fields for architecture and provider family,
  - preserve current CLT defaults.
- `src/nlp_research_project/exact_trace_bench/full_answer/decoder_signature_cache.py`
  - keep the current CLT cache path readable,
  - do not reuse it as the PLT cache unless a PLT-specific signature model is
    designed.

Suggested project-side config shape:

```text
TranscoderLoadConfig:
  model_name: str
  transcoder_architecture: "clt" | "plt"
  repo_id: str
  revision: str | None
  clt_subfolder: str | None
  plt_subfolder_template: str | None
  layer_count: int | "infer"
  feature_input_hook: str
  feature_output_hook: str
  lazy_encoder: bool
  lazy_decoder: bool
  decoder_chunk_size: int
  cross_batch_decoder_cache_bytes: int
```

GemmaScope2 PLT path template to support initially:

```text
transcoder_all/layer_{layer}_width_262k_l0_{small|big}{_affine}/params.safetensors
```

Done when:

- full-answer trace specs can request CLT or PLT explicitly,
- shard metadata and per-token trace metadata record architecture, repo/subfolder,
  layer count, chunk/cache settings, and detected capabilities,
- the existing fast CLT scenario remains the default scenario family.

### Phase 4 — Validate with login-safe tests and SLURM-backed smoke runs

Goal: prove provider parity without loading models on login nodes.

Sibling library test tasks:

- add synthetic PLT fixtures with lowercase keys,
- assert lazy PLT encoder-row reads match eager tensors,
- assert lazy PLT decoder chunks match eager slices,
- assert PLT chunked reconstruction matches tiny eager PLT reconstruction,
- assert sparse feature ordering is stable between eager and chunked modes,
- assert CLT tests still pass unchanged.

Project harness validation tasks:

- add config/dispatch tests for architecture selection,
- add scenario-generation tests for CLT and PLT families,
- add metadata assertions for recorded architecture/capability fields.

SLURM smoke sequence:

1. baseline 1B CLT smoke,
2. 1B PLT width_262k one-token smoke,
3. 1B PLT width_262k five-token benchmark,
4. 4B PLT width_262k one-token smoke only after the 1B profile is understood,
5. 4B PLT width_262k five-token benchmark only if memory is tractable.

## 6. Testing strategy

### Login-safe

- provider/capability unit tests,
- synthetic CLT/PLT load-save fixtures,
- sparse-state and chunked-topology tests,
- scenario-generation and metadata tests,
- doc/index checks,
- no model downloads, no GPU use.

Concrete login-safe assertions:

1. provider architecture is explicit and not inferred from repo naming alone,
2. PLT lazy row materialization preserves row order,
3. PLT lazy chunk materialization matches eager slices,
4. chunked state preserves sparse feature ordering,
5. CLT provider behavior is unchanged semantically,
6. the project harness records architecture/capability metadata.

### GPU / SLURM only

- current 1B CLT baseline smoke,
- GemmaScope2 width_262k PLT smoke for Gemma 3 1B/4B,
- exact tracing/parity runs when runtime semantics change,
- compact-output comparisons on canonical prompts.

Real tracing runs should record:

- project repo SHA/dirty files,
- sibling repo SHA/dirty files,
- live-vs-snapshot workspace status,
- model name and transcoder artifact paths,
- detected architecture and capability flags,
- phase 0 active-feature counts by layer/token,
- phase 0 encode/reconstruction timing,
- decoder-cache hits/misses/evictions,
- encoder-row residency mode and bytes,
- peak CUDA allocation/reservation and SLURM MaxRSS,
- compact row-store bytes and phase 4 timings.

## 7. Risks / open questions

- Which optimizations are genuinely PLT-compatible vs CLT-only?
- Do the requested GemmaScope2 width_262k PLT checkpoints exist for both Gemma 3
  1B and 4B under stable repo IDs?
- Can the old `exact_chunked_decoder` flag be removed cleanly, or should it remain
  as a temporary shim for one transition cycle?
- PLT `nnz` may be larger than CLT `nnz`. The concern is runtime pressure, not
  durable scratch output size: in-memory sparse state, optional encoder-row
  residency, Phase-3/4 row width, compact row-store temporary bytes, scratch/tmp
  I/O bandwidth, and walltime may dominate even after decoder materialization is
  fixed.
- 4B width_262k PLTs may require smaller batch sizes, smaller decoder chunks, or a
  new microbatching path.
- Eager parity against real width_262k PLTs may be infeasible; synthetic parity plus
  real-run invariant checks may be the right balance.

## 8. Acceptance criteria

1. CLT and PLT both load through an explicit provider/capability path.
2. `TranscoderArchitecture` is the primary architecture label; `exact_chunked_decoder`
   is only a shim/legacy alias if retained at all.
3. Existing exact/chunked optimizations still work where semantically valid.
4. The current 1B CLT baseline/defaults remain semantically unchanged.
5. GemmaScope2 width_262k PLT support exists for Gemma 3 1B/4B or the absence is
   clearly surfaced as a capability/load failure.
6. Login-node validation stays model-free; SLURM is used for real tracing.
7. Provider diagnostics and cache metadata make architecture/topology differences
   observable in logs and artifacts.

## 9. Ordered tasks

1. Add the provider/capability dataclasses and architecture enum.
2. Adapt CLT loaders/caches to emit the new provider contract.
3. Implement PLT lazy readers and exact/chunked provider support.
4. Update attribution/runtime code to consume provider topology and diagnostics.
5. Update project harness config, scenario generation, and metadata recording.
6. Add synthetic parity tests and harness unit tests.
7. Run SLURM smokes for the CLT baseline and the GemmaScope2 width_262k PLT path.
