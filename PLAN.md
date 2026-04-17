# Current Implementation Plan

## Problem

We now have a working fixed preflight Phase-4 batch planner and a sequence of
memory optimizations that preserve compact-graph outputs. The next immediate
work should do two things:

1. avoid paying planner overhead when the planner cannot possibly increase the
   Phase-4 feature batch size, and
2. add much richer profiling so we can see exactly where tracing time is spent
   before choosing the next speed optimization.

## Immediate changes

### 1. Smarter planner skip

Skip planner preflight entirely when there is no room to scale:

- if `planner_enabled` is false: current behavior
- if `feature_batch_size_max` is `None`: current behavior
- if `feature_batch_size_max <= initial/effective feature_batch_size`: skip planner
- log a clear reason in the run log and persist that the planner was skipped

Acceptance:

- no preflight probe work is run in no-headroom cases
- manifests / extraction distinguish:
  - planner disabled
  - planner enabled and skipped
  - planner enabled and executed

### 2. Rich tracing profiling

Add structured timing for:

- full run / completion / step end-to-end latency
- every attribution phase
- meaningful micro-operations in the fork:
  - trace invoke / forward setup
  - Phase 3 logit batches
  - Phase 4 feature batches
  - planner preflight batches
  - decoder / encoder loads
  - memmap row-store reads / writes / dense slice materialization
  - packaging / saving compact artifacts / saving raw graphs

## Implementation order

### Phase A — planner skip + planner status persistence

Files:

- `circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
- `nlp_research_project/trace_pipeline_chunked.py`
- `nlp_research_project/experiments/exact_trace_bench/extract.py`
- `nlp_research_project/experiments/extract_benchmark_index.py`

Tasks:

- add explicit planner skip condition for no-headroom cases
- persist planner mode/status fields
- expose effective / skipped planner state in extraction tables

### Phase B — fork event model + event sink

Files:

- `circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`
- `circuit-tracer_chunked/circuit_tracer/attribution/context_nnsight.py`
- `circuit-tracer_chunked/circuit_tracer/transcoder/cross_layer_transcoder.py`
- `circuit-tracer_chunked/circuit_tracer/utils/telemetry.py`

Tasks:

- define one scalar-only event schema
- emit structured timing events from meaningful hot-path blocks
- keep existing light counters/diagnostics, but route richer timing through a
  single event sink

### Phase C — completion-level telemetry artifacts

Files:

- `nlp_research_project/trace_pipeline_chunked.py`
- `nlp_research_project/trace_pipeline.py`

Tasks:

- add per-completion `telemetry.jsonl`
- add summarized per-step and per-completion latency blocks to `completion.json`
- add run-level summary fields that point to telemetry files

### Phase D — extraction + analysis integration

Files:

- `nlp_research_project/experiments/exact_trace_bench/extract.py`
- `nlp_research_project/experiments/extract_benchmark_index.py`
- `nlp_research_project/experiments/telemetry_gathering.py` (new)
- optional new profiling-focused extraction helpers if needed

Tasks:

- flatten summary timing fields into benchmark tables
- add a dedicated repo-local telemetry gathering/profiler script that:
  - reads `telemetry.jsonl` + completion manifests
  - emits flat CSV/JSON summaries
  - produces per-run / per-step / per-batch views
- optionally extract detailed event tables from `telemetry.jsonl`
- keep hot-path logging simple; do heavier aggregation offline

Current branch implementation notes:

- extraction now surfaces completion-level telemetry presence/path/count fields
  (including found/missing telemetry files and manifest-vs-step event counts)
- extraction now surfaces aggregated completion timing summaries (totals,
  averages, per-step rollups)
- extraction now surfaces aggregated attribution phase elapsed totals from
  completion timing summaries (with step-level fallback when needed)
- `experiments/telemetry_gathering.py` now emits:
  - `telemetry_runs.csv`
  - `telemetry_steps.csv`
  - `telemetry_batches.csv`
  - `telemetry_ops.csv`

## Output shape to target

Per completion:

- `completion.json` — human-usable summary
- `telemetry.jsonl` — detailed event stream

Per run / scenario tables:

- summary columns in `benchmark_enriched.csv`
- optional derived profiling tables for batch/event analysis
- dedicated telemetry-gathering outputs, e.g.:
  - `telemetry_runs.csv`
  - `telemetry_steps.csv`
  - `telemetry_batches.csv`
  - `telemetry_ops.csv`

## Success criteria

- planner skip works and is visible in artifacts
- we can answer, from structured data, all of the following:
  - where total step time goes
  - how much time is spent in each attribution phase
  - how much time is spent in decoder/encoder loads
  - how much time is spent in memmap row-store I/O
  - how much time is spent per Phase-3 / Phase-4 batch
  - how much time packaging and saving artifacts costs

## Known caveat

For the first pass, detailed profiling should default to compact scalar events
only. We should avoid tensor dumps, unnecessary `cuda.synchronize()` calls, and
unbounded per-op logging outside explicit profiling mode.
