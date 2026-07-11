# Sibling `circuit_tracer` code map

Status: Current-state map
Last updated: 2026-07-10

## Package and CLI

Unless noted, paths and commands in this page are relative to the sibling repo
`../circuit-tracer_chunked/`; run the listed commands from that checkout.

| Surface | Location |
|---|---|
| Package root | `circuit_tracer/` |
| Console script `circuit-tracer` | `circuit_tracer/__main__.py` |

## Main modules

| Area | Modules |
|---|---|
| Attribution entry points | `attribution/{attribute.py,attribute_nnsight.py,attribute_transformerlens.py,context_nnsight.py,context_transformerlens.py,targets.py,sparsification.py}` |
| NNSight phase runtime | `attribution/nnsight/phases/phase{0,1,2,3,4,5}.py` |
| NNSight support | `attribution/nnsight/{phase1_policy.py,phase4_policy.py,phase_support.py,replay.py,row_store.py,prefix_view.py,numerics.py,telemetry.py}` |
| Observability | `observability/{lifecycle.py,recorder.py,resources.py,human_logs.py,exception_export.py}` |
| Replacement models | `replacement_model/*` |
| Transcoders | `transcoder/{single_layer_transcoder.py,cross_layer_transcoder.py,loaders.py,decoder_cache.py,diagnostics.py,fingerprints.py}` |
| Utilities | `utils/*` |
| Graph | `graph.py` |
| Frontend | `frontend/*` |

## Public surface used by the project

| Export/API | Notes |
|---|---|
| `ReplacementModel` | Main integration entry point |
| `Graph` | Graph serialization and conversion |
| `attribute` / `attribute_phase0_stats` | Attribution entry points |
| `SparsificationConfig` | Sparsification configuration |
| `ReplacementModel.from_pretrained` | Main loader |
| `ReplacementModel.from_pretrained_and_transcoders` | Loader with transcoder inputs |
| Backend selection | `transformerlens` or `nnsight` |
| Loader helpers | `load_transcoder_from_hub`, `resolve_transcoder_paths`, `load_transcoder_set`, `load_clt`, `load_gemma_scope_2_clt` |
| Graph I/O | `Graph.to_pt`, `Graph.from_pt`, `create_graph_files` |
| Frontend | Graph frontend server |

## Phase map

| Phase | Current focus |
|---|---|
| Phase 0 | Setup, sparse activation discovery, sparsification, exact decoder setup, telemetry |
| Phase 1 | Forward and residual capture |
| Phase 2 | Inputs, target/logit objects |
| Phase 3 | Logit attribution, seed ranking, replay validation, donor/row/frontier prep |
| Phase 4 | Feature attribution, frontier, scheduler, reduction, refresh |
| Phase 5 | Packaging and graph finalization |

## Optimization/current capability map

| Area | Current items |
|---|---|
| Decoder caching | `DecoderChunkCache` |
| Attribution staging | `AttributionContext`, chunk spans, replay, cache fingerprints |
| Phase-4 knobs | `phase4_scheduler_mode`, `phase4_refresh_optimization`, `phase4_row_executor`, `phase4_row_reduction` |
| Phase-1 knobs | `phase1_trace_batch_policy`, `row_store_cache_control`, `exact_encoder_residency`, `exact_trace_internal_dtype` |
| Streaming influences | `graph.compute_partial_feature_influences_streaming` with stable row-L1 denominators |

## PLT / CLT boundary

| Transcoder family | Current implementation |
|---|---|
| PLT | `SingleLayerTranscoder` and `TranscoderSet` |
| CLT | `CrossLayerTranscoder` with shared-layer feature layout |
| Loader decision | `load_transcoder_from_hub()` reads `config.yaml` and returns the appropriate transcoder family |
| GemmaScope-2 CLT path | Uses the exact chunked decoder path automatically |
| CLT optimization knobs | `exact_chunked_decoder`, `decoder_chunk_size`, `cross_batch_decoder_cache_bytes` |

PLT optimization parity is active/future work. This map describes the current
boundary; it does not assert that every CLT optimization already applies to PLT.

## Tests commonly used for this area

- `tests/test_partial_influences.py`
- `tests/test_gemmascope2_chunked.py`
- `tests/test_phase0_stats.py`
- `tests/test_graph.py`
- `tests/test_cross_layer_transcoder.py`
- `tests/test_chunked_decoder_optimizations.py`
- `tests/test_double_pass_sparsification.py`

Common lightweight commands:

- `uv run ruff check circuit_tracer tests`
- `uv run pyright`
- `uv run circuit-tracer --help`
- `uv run pytest -q tests/test_partial_influences.py tests/test_gemmascope2_chunked.py`
- `uv run pytest -q tests/test_phase0_stats.py tests/test_graph.py tests/test_cross_layer_transcoder.py`
- `uv run pytest -q tests/test_chunked_decoder_optimizations.py tests/test_double_pass_sparsification.py`

GPU regression scripts under `scripts/slurm/` are SLURM-only.

## Current integration boundary notes

- The project imports sibling runtime pieces through `circuit_tracer` and direct
  imports such as `circuit_tracer.attribution.attribute_nnsight`.
- Project-local modules such as `trace_pipeline.py`, `trace_pipeline_chunked.py`,
  and `circuit_utils.py` are mapped in `project_code_map.md`, not this sibling
  map.
- `ReplacementModel` and transcoder loaders are the main stable integration points.
- PLT/CLT boundary: `SingleLayerTranscoder`/`TranscoderSet` cover PLT; `CrossLayerTranscoder` covers CLT.
- `load_transcoder_from_hub` reads `config.yaml` to choose transcoder kind.
- GemmaScope-2 CLTs use the exact chunked decoder path automatically.

## Current-state caveats for later debt audit

- `attribute_nnsight.py` remains the public compatibility and
  lifecycle-orchestration layer; phase algorithms and telemetry mechanics are
  decomposed into deep modules.
- `attribute()` exposes a wide knob surface, including NNSight-only and legacy
  compatibility settings.
- `Graph.logit_tokens` and some `Graph.from_pt()` tensor formats remain as
  compatibility surfaces.
- `phase4_refresh_prepared_chunk_cache_bytes` is experimental/retired.
- NNSight backend notes still describe it as more fragile and memory-sensitive
  than the TransformerLens path.
- `load_gemma_scope_2_transcoder()` does not currently support lazy loading;
  caching is the intended workaround.
