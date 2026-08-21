# Pre-C2 Trace Contract

Status: Frozen characterization oracle
Frozen: 2026-07-13
Scope: Phase C2 sibling runtime and project tracing-path replacement

This record pins the behavior and artifacts that Phase C2 must preserve. It is
not an API-compatibility requirement. The legacy Python entry points listed
below must be removed, while the scientific and project artifact contracts
remain comparable.

## Granite Oracles

All oracles are immutable `361_base` Gemma 3 1B runs on one Granite H200 with
`exact_trace_internal_dtype=fp32`.

| Provider / mechanism | Run root |
|---|---|
| CLT A full-file reference | `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/baseline/phase_d/gemma3_1b_clt/phase-d-clt-20260712-03/granite_phase_d_clt_361_base_A_legacy_reference` |
| CLT D column-tiled | `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/baseline/phase_d/gemma3_1b_clt/phase-d-clt-de-20260712-06/granite_phase_d_clt_361_base_D_column_tiled_v1` |
| CLT E no-retention | `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/baseline/phase_d/gemma3_1b_clt/phase-d-clt-e-20260712-10/granite_phase_d_clt_361_base_E_none_recompute` |
| PLT A full-file reference | `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/baseline/phase_d/gemma3_1b_plt/phase-d-plt-20260712-03/granite_phase_d_plt_361_base_A_legacy_reference` |
| PLT D column-tiled | `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/baseline/phase_d/gemma3_1b_plt/phase-d-plt-de-20260712-12/granite_phase_d_plt_361_base_D_column_tiled_v1` |
| PLT E no-retention | `/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/granite/baseline/phase_d/gemma3_1b_plt/phase-d-plt-e-20260712-13/granite_phase_d_plt_361_base_E_none_recompute` |

The accepted Phase D bounded-mechanism comparisons have exact feature and edge
support. Weighted-edge Jaccard versus A is at least `0.9999999756` for CLT and
`0.9999999791` for PLT, with exact top-k overlap through 1,024. C2 must compare
against the same A artifacts and may not weaken this criterion without an
explicit scientific adjudication.

## Retained Project Artifacts

The project continues to own these paths and layouts:

- scenario envelope: `scenario.json`, `run.log`, `result.json`, `summary.json`,
  and `scenario_metrics.{json,csv}`;
- run metadata: `artifacts/run_config.json`, prompt metadata, and completion
  manifests;
- compact traces: per-step NPZ files and their graph metadata;
- crash-surviving incremental telemetry and final telemetry summaries;
- optional Phase 0, Phase 3, cross-cluster, anomaly, and semantic-descriptor
  sidecars;
- full-answer token graph/trace files, shard JSONL, and shard summary.

The sibling `TraceResult` becomes the source of graph data, semantic and
execution fingerprints, admission evidence, telemetry summary, and typed
failure evidence. Project code must not mutate sibling results to attach these
fields.

## Pre-Rewrite Callers

The frozen pre-C2 path is:

```text
scenario
  -> experiments/run_sparsification_experiment.py::build_command
  -> trace_pipeline_chunked.py
  -> extract_compact_chunked_attribution
  -> circuit_tracer.attribution.attribute_nnsight.attribute
  -> runtime legacy translation
  -> _attribute_impl
  -> _run_attribution
```

The full-answer runner independently imports `attribute_nnsight.attribute` and
`FullSequenceWindowAttributionSession`. Both callers must move atomically to
the canonical sibling tracing API.

## Deletion Gate

Phase C2 is not complete while production code contains any of these surfaces:

- sibling `attribute_nnsight.py`, `_attribute_impl`, `_run_attribution`, or
  `request_from_legacy`;
- `TraceRequest.legacy_kwargs` or reflected legacy signatures;
- generic flat `circuit_tracer.attribute(...)` routing;
- compatibility-only private helper re-exports;
- project dynamic imports of `circuit_tracer.attribution.attribute_nnsight`;
- the flat scenario-to-trace subprocess argument relay;
- a second project trace pipeline that bypasses the canonical request builder.

Tests may reference removed names only in explicit stale-reference assertions.
