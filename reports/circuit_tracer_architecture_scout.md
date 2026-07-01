# Sibling library architecture scout: `../circuit-tracer_chunked`

Date: 2026-06-30

Status: scouting/research only. No code changes.

## Package map

```text
circuit_tracer/
├── __init__.py                         # lazy public facade
├── __main__.py                         # CLI-ish graph file generation adapter
├── graph.py                            # graph object, persistence, pruning/influence math
├── attribution/
│   ├── attribute.py                    # public attribution dispatcher
│   ├── attribute_nnsight.py            # NNSight exact/chunked attribution mega-module
│   ├── attribute_transformerlens.py    # TL backend attribution
│   ├── context_nnsight.py              # NNSight attribution context/state machine
│   ├── context_transformerlens.py
│   ├── sparsification.py
│   └── targets.py
├── replacement_model/
│   ├── replacement_model.py            # base replacement model seam
│   ├── replacement_model_nnsight.py    # NNSight backend adapter/runtime behavior
│   └── replacement_model_transformerlens.py
├── transcoder/
│   ├── cross_layer_transcoder.py       # CLT math, cache, diagnostics, loading
│   ├── single_layer_transcoder.py
│   └── activation_functions.py
├── utils/
│   ├── hf_utils.py                     # URI/config/cache/HF loading helpers
│   ├── caching.py, disk_offload.py, telemetry.py, ...
│   └── MAPPING_INFO.md
└── frontend/
    ├── graph_models.py, feature_models.py, local_server.py, ...
    └── assets/
```

Likely domain concepts:

- `Graph`, logit targets, feature nodes, token nodes, graph pruning/influence.
- `ReplacementModel` backend abstraction: TransformerLens vs NNSight.
- `CrossLayerTranscoder`, decoder chunks, encoder vectors, sparse activations.
- `AttributionContext`, exact/chunked phases, row-store, replay validation.
- Prefix/full-sequence attribution sessions for repeated trajectory windows.
- HF model/transcoder URI resolution and cache loading.

## Highest-friction modules

### `circuit_tracer/attribution/attribute_nnsight.py`

This is the central architecture problem.

Observed outline:

- `attribute()` starts around line 7140 and takes a very wide keyword-only knob
  surface through line 7223.
- `_run_attribution()` spans roughly line 7634 through 12538.
- `_FileBackedFeatureRowStore` alone spans roughly line 829 through 1619.
- The module also contains prefix-view metadata validation, phase-4 scheduler
  configs/plans, row-store cache-control, exact encoder residency policy,
  dtype/precision policy, phase-0/phase-3 donor bundle loading, semantic
  descriptor capture, telemetry payload construction, and `FullSequenceWindowAttributionSession`.

Why it is shallow:

- The callable interface is nearly as complex as the implementation. Calling
  attribution means understanding a large flat bag of unrelated policy knobs.
- There is little locality: changing Phase 4 scheduling, row-store behavior,
  replay validation, prefix metadata, or debug capture all means entering the
  same mega-file and often the same `_run_attribution()` flow.
- It is hard for tests to target a stable module interface. Many tests currently
  reach helper functions or observe large side effects.

Deletion test:

- If a row-store helper, phase policy resolver, replay loader, or telemetry
  builder were deleted from this file, complexity would not vanish; it would
  simply move inside `_run_attribution()` or into the caller. That means these
  are real deepening opportunities.

### `circuit_tracer/attribution/context_nnsight.py`

Observed outline:

- `AttributionContext` spans roughly lines 58 through 1545.
- It mixes setup state, tensor staging, encoder residency, prefix-view context
  derivation, decoder cache lifecycle, diagnostic stats, feature attribution
  computation, residual caching, score computation, and batch execution.

Why it is shallow:

- `AttributionContext` sounds like a state holder, but it also owns several
  runtime behaviors. The module interface hides too little complexity because
  callers still need to reason about staging, cache lifetime, prefix views, and
  chunked batch execution together.

Deletion test:

- Deleting context helpers would not concentrate complexity; it would scatter
  cache/prefix/staging logic into attribution orchestration. These are separable
  modules behind a deeper context seam.

### `circuit_tracer/transcoder/cross_layer_transcoder.py`

Observed outline:

- `DecoderChunkCache` is defined in the same file as `CrossLayerTranscoder`.
- `CrossLayerTranscoder` spans roughly lines 72 through 1805.
- Loaders `load_clt`, `load_gemma_scope_2_clt`, and `_load_state_dict` follow
  the model class in the same module.
- The class mixes core encoding/decoding math, decoder block caching, diagnostics,
  trace logging, sparse membership fingerprints, and serialization/loading.

Why it is shallow:

- The transcoder module is both the math object and the runtime/cache/diagnostic
  adapter. That makes exact-chunked performance changes harder to isolate from
  core CLT behavior.

Deletion test:

- Cache and diagnostics are strong extraction candidates: deleting them should
  leave the transcoder math clearer rather than move complexity into callers.
  Loader behavior is also separable, but more likely to require compatibility care.

### `circuit_tracer/replacement_model/replacement_model_nnsight.py`

Observed outline:

- `NNSightReplacementModel` spans roughly lines 178 through 1404.
- It includes construction helpers, model configuration, activation fetching,
  tokenization, attribution setup, freeze/intervention logic, generation, and
  model hook location accessors.

Why it is shallow:

- The NNSight adapter module is not just an adapter. It also owns runtime
  intervention behavior and attribution setup policy. Backend-specific hook
  mapping is therefore coupled to user-facing operations.

Deletion test:

- If hook resolution or intervention behavior were removed, the surrounding
  backend adapter would become easier to reason about. Those seams are real,
  although they are less urgent than NNSight attribution.

### `circuit_tracer/graph.py`

Observed outline:

- `Graph` data/persistence object spans roughly lines 20 through 180.
- Influence, pruning, partial influence, and streaming partial influence math
  follow in the same module, with `compute_partial_feature_influences_streaming`
  spanning roughly lines 693 through 1200.

Why it is shallow:

- The `Graph` module is both a value object and an algorithm collection. This is
  not the worst problem, but it reduces locality for graph serialization versus
  graph math changes.

Deletion test:

- Graph math can likely move behind a sibling module without changing the data
  object. The public compatibility risk is manageable but real.

### `circuit_tracer/utils/hf_utils.py`

Why it matters:

- It mixes URI parsing, legacy aliases, config normalization, cache fallback,
  hub download, and model-kind dispatch. That makes the network/cache seam harder
  to test cheaply.

Deletion test:

- Pure parser/normalizer logic should be deletable from loader I/O and tested
  without HF/network access. This is a high-leverage smaller cleanup.

## Deepening candidates

### 1. NNSight attribution module split

Files/modules involved:

- `circuit_tracer/attribution/attribute_nnsight.py`
- `circuit_tracer/attribution/context_nnsight.py`
- Tests: `test_chunked_decoder_optimizations.py`, `test_full_sequence_window_session.py`,
  `test_prefix_view_metadata.py`, `test_phase3_replay_validation.py`,
  `test_phase3_frontier_buffer.py`, `test_phase4_frontier_buffer.py`,
  `test_attribute_nnsight_telemetry.py`

Problem:

- The current attribution module has low locality. It mixes the high-level
  attribution flow with phase policy, row storage, replay bundles, diagnostics,
  prefix views, and telemetry. The interface is too wide, and callers must pass
  or forward many knobs that belong to different domains.

Solution direction, without detailed new interfaces:

- Deepen around phase/runtime concepts rather than continuing to grow the flat
  mega-module. Leave the public attribution entrypoint as a compatibility facade
  while moving coherent behavior behind deeper modules.

Benefits:

- Locality: Phase 4 scheduling changes do not require reading replay loaders or
  prefix metadata code.
- Leverage: exact-trace bug fixes become easier to make and test in isolation.
- Testability: existing phase-specific tests can target smaller seams.

Deletion-test assessment:

- Strong. Many helper groups already behave like separate modules but live in one file.

Recommendation strength: **Strong**.

### 2. Exact-trace policy/config seam

Files/modules involved:

- `circuit_tracer/attribution/attribute.py`
- `circuit_tracer/attribution/attribute_nnsight.py`
- Harness: `src/nlp_research_project/exact_trace_bench/full_answer/runner.py`,
  `docs/knob_api_taxonomy.md`

Problem:

- The public attribution dispatcher exposes a large flat keyword surface. The
  NNSight backend adds even more hidden/internal override knobs. The harness must
  know too much about sibling internals and forward many options manually.

Solution direction, without detailed new interfaces:

- Group related policies behind deeper module seams while preserving the current
  public entrypoint as an adapter during migration.

Benefits:

- Locality: scheduler policy, row-store policy, dtype policy, replay/debug policy,
  and cache policy stop competing in one signature.
- Leverage: harness launch/scenario code can map experiment knobs to one stable
  attribution boundary instead of mirroring internals.
- Testability: policy resolution can be tested without running attribution.

Deletion-test assessment:

- Strong. Removing most individual policy resolvers from `attribute_nnsight.py`
  should concentrate behavior, not scatter it.

Recommendation strength: **Strong**.

### 3. Row-store and replay/prefix locality

Files/modules involved:

- `attribute_nnsight.py` (`_FileBackedFeatureRowStore`, donor bundle loaders,
  prefix-view validators, full-sequence session)
- `context_nnsight.py` (prefix view state, chunked feature computation, decoder cache)

Problem:

- Exact-trace correctness depends on subtle invariants: row-L1 denominator
  behavior, feature-row persistence, donor replay validation, prefix-view token
  metadata, and full-sequence reuse. Today these invariants are spread across a
  large orchestration module and a large context class.

Solution direction, without detailed new interfaces:

- Make row materialization/reuse and replay/prefix validation deeper modules with
  narrow responsibilities. Keep the attribution flow as their caller.

Benefits:

- Locality: row-store changes are not entangled with scheduler changes.
- Leverage: replay validation can become internal validation infrastructure with
  stable behavior.
- Testability: many invariants can remain login-safe unit tests.

Deletion-test assessment:

- Strong. These helpers are cohesive and already have targeted tests.

Recommendation strength: **Strong**.

### 4. Cross-layer transcoder runtime helpers

Files/modules involved:

- `circuit_tracer/transcoder/cross_layer_transcoder.py`
- Tests under `tests/transcoder/` and exact-chunked decoder tests.

Problem:

- Core CLT math is mixed with lazy decoder caches, diagnostic snapshots, trace
  logging, fingerprinting, and loading.

Solution direction, without detailed new interfaces:

- Keep the transcoder math object central, but move runtime/cache/diagnostic and
  loading concerns behind deeper adjacent modules.

Benefits:

- Locality: cache changes do not require editing the math class.
- Leverage: exact-chunked performance work becomes easier to reason about.
- Testability: cache behavior can be tested without full model loading.

Deletion-test assessment:

- Strong for cache/diagnostics; medium for loading.

Recommendation strength: **Worth exploring**.

### 5. NNSight replacement model adapter

Files/modules involved:

- `circuit_tracer/replacement_model/replacement_model_nnsight.py`
- Backend parity tests and intervention tests.

Problem:

- Construction, hook mapping, attribution setup, interventions, freeze logic, and
  generation all live in one backend adapter class.

Solution direction, without detailed new interfaces:

- Deepen backend runtime behaviors separately from model construction and hook
  location lookup.

Benefits:

- Locality: backend mapping changes become less likely to affect intervention or
  generation behavior.
- Leverage: parity tests can target the actual backend seam.
- Testability: hook/location resolution can be tested without invoking generation.

Deletion-test assessment:

- Medium-strong. The adapter is less pathological than attribution, but it has
  clear extractable responsibilities.

Recommendation strength: **Worth exploring**.

### 6. Graph value object versus graph algorithms

Files/modules involved:

- `circuit_tracer/graph.py`
- `tests/test_graph.py`, `tests/test_partial_influences.py`

Problem:

- Graph persistence and graph analysis algorithms are coupled in one module.

Solution direction, without detailed new interfaces:

- Keep the `Graph` value/persistence behavior stable while moving pruning and
  influence algorithms behind a deeper graph-analysis module.

Benefits:

- Locality: serialization compatibility and influence math can evolve separately.
- Leverage: graph-analysis algorithms become easier to test and optimize.

Deletion-test assessment:

- Medium-strong. More compatibility-sensitive than row-store/policy extraction.

Recommendation strength: **Worth exploring**.

### 7. HF config/loading adapter cleanup

Files/modules involved:

- `circuit_tracer/utils/hf_utils.py`
- `tests/utils/test_hf_utils.py`, `tests/utils/test_caching.py`

Problem:

- Pure normalization logic and I/O-heavy loading logic share one seam.

Solution direction, without detailed new interfaces:

- Separate pure URI/config normalization from cache/HF/network loading adapters.

Benefits:

- Locality: alias/config changes can be tested without remote I/O.
- Leverage: future model-kind support gets safer.
- Testability: parser and cache fallback behavior become cheap unit tests.

Deletion-test assessment:

- Strong for parsing/normalization; medium for loader behavior.

Recommendation strength: **Strong, smaller-scope**.

## Current tests to preserve as safety rails

Login-safe or mostly unit-level tests to inspect before implementation:

- `tests/test_chunked_decoder_optimizations.py`
- `tests/test_prefix_view_metadata.py`
- `tests/test_full_sequence_window_session.py`
- `tests/test_phase3_replay_validation.py`
- `tests/test_phase3_frontier_buffer.py`
- `tests/test_phase4_frontier_buffer.py`
- `tests/test_attribute_nnsight_telemetry.py`
- `tests/transcoder/test_cross_layer_transcoder.py`
- `tests/transcoder/test_cross_layer_transcoder_telemetry.py`
- `tests/test_graph.py`
- `tests/test_partial_influences.py`
- `tests/utils/test_hf_utils.py`
- `tests/utils/test_caching.py`

Backend/GPU/model-load tests exist, but should only be run in appropriate SLURM
contexts if they load models or require GPUs.

## Documentation state

Found:

- `README.md`
- `RESEARCH_USAGE.md`
- `CONTRIBUTING.md`
- `circuit_tracer/utils/MAPPING_INFO.md`
- frontend asset README

Not found:

- No `CONTEXT.md` or architecture glossary.
- No `docs/` directory in the sibling checkout.
- No obvious architecture decision log.

Implication:

- The architecture cleanup should likely create a small living architecture map
  or module guide once actual design starts. For this scouting pass, code and
  tests are the source of truth.

## Risks and unknowns

- The sibling repo is a fork with exact/chunked tracing changes. Architecture
  work must preserve current validated baseline behavior, especially fp32 exact
  trace dtype and row-L1 denominator behavior.
- The public `attribute()` signature likely has downstream callers beyond this
  harness. Compatibility adapters will matter.
- Heavy correctness/performance validation cannot be completed on a login node.
- Some debug/replay machinery may be intentionally internal infrastructure and
  should not become the ordinary workflow.
