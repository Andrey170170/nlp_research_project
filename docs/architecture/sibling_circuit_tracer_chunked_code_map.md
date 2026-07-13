# Sibling `circuit_tracer` code map

Status: Current Phase C2 implementation map; immutable Granite validation pending
Last updated: 2026-07-13

Paths are relative to the sibling checkout `../circuit-tracer_chunked/`.

## Public tracing boundary

| Concept | Module |
|---|---|
| `trace_one`, `trace_batch`, `open_session` | `circuit_tracer/tracing/api.py` |
| `TraceRequest` and execution/semantic composition | `circuit_tracer/tracing/request.py` |
| Attribution problem | `circuit_tracer/tracing/problem.py` |
| Resolved plan and fingerprints | `circuit_tracer/tracing/{plan,planning}.py` |
| Trace/session results | `circuit_tracer/tracing/{result,session}.py` |
| Backend-neutral runner | `circuit_tracer/tracing/runner.py` |

The canonical request composes meaningful domain policies such as session,
decoder-cache, row-storage, replay, frontier-expansion, and observability
policies. Semantic and execution fingerprints are independent. Legacy kwargs,
signature reflection, and flat attribution entry points are absent.

## NNSight execution

| Owner | Modules |
|---|---|
| Thin backend coordinator | `attribution/nnsight/backend.py` |
| Readable Phase 0-5 orchestration | `attribution/nnsight/execution.py` |
| Validation and mechanism preparation | `attribution/nnsight/preparation.py` |
| Cleanup ownership | `attribution/nnsight/run_scope.py` |
| Domain phase operations | `attribution/nnsight/phases/phase{0,1,2,3,4,5}.py` |
| Context invariants and batch execution | `attribution/nnsight/context_state.py`, `batch_execution.py`, `attribution/context_nnsight.py` |
| Forward-session capability | `attribution/nnsight/forward_session.py` |
| Storage/replay mechanisms | `attribution/nnsight/{row_store,row_replay,tiled_rows,replay}.py` |
| Explicit policy resolvers | `phase1_policy.py`, `phase4_policy.py`, `session_controls.py`, `numerics.py` |

`AttributionExecution.run()` exposes the scientific order: Phase 0
preparation, forward tracing, active-feature/storage setup, seed attribution,
frontier expansion, and graph assembly. Policy resolution and terminal
lifecycle are outside that method.

## Supporting domains

| Area | Modules |
|---|---|
| Typed observations and run lifecycle | `observability/{events,lifecycle,run_scope}.py` |
| Incremental recording and resources | `observability/{recorder,resources}.py` |
| Human rendering and exception evidence | `observability/{human_logs,exception_export}.py` |
| Replacement-model setup | `replacement_model/{attribution_setup,model_adapter,nnsight_configuration,replacement_model_nnsight}.py` |
| Typed transcoder results/provider capabilities | `transcoder/{attribution_result,provider,diagnostics}.py` |
| PLT/CLT implementations | `transcoder/{single_layer_transcoder,cross_layer_transcoder}.py` |
| Graph algorithms and packaging | `graph.py` |

Tracing and storage code emit typed observations. Resource sampling, sink
schema/sequencing/flushing, terminal attachments, and human rendering belong to
`observability/`.

## Provider boundary

PLT and CLT variation enters through provider capabilities and typed
`AttributionComponents`, not model-family string branches or mapping-shaped
result shims. Model-specific tensor conventions are selected by replacement
model adapter capabilities.

## Removed surfaces

Phase C2 removes `attribution/attribute.py`,
`attribution/attribute_nnsight.py`, legacy request translators, reflected
signatures, private compatibility re-exports, and generic flat root routing.
There is one public runtime for single, batch, and session tracing.

## Validation

Architecture tests bound coordinator size/complexity, dependency direction,
meaningful context/result objects, stale imports, observability ownership,
provider dispatch, and terminal lifecycle behavior. The broad login-safe suite
must pass before snapshotting. Immutable `361_base` Gemma 3 1B CLT/PLT
full-retention and bounded runs remain the closure gate.
