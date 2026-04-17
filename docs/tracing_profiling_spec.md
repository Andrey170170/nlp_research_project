# Tracing Profiling Spec

## 1. Problem statement

Current exact-trace profiling is good enough for phase-level summaries, but it
is not sufficient to answer where time is really going inside the fork-native
exact chunked path. We need much richer visibility before choosing the next
speed optimization.

The profiling layer should let us inspect:

- end-to-end latency per run / completion / step,
- phase latency for every attribution step,
- per-batch latency for Phase 3 and Phase 4,
- meaningful micro-operations such as storage loads, memmap reads/writes,
  planner probes, packaging, and artifact writes.

## 2. Scope / non-goals

### Scope

- exact chunked NNSight path as the primary target
- compact output path first
- structured timing data for:
  - run
  - completion
  - step
  - phase
  - batch
  - meaningful micro-ops

### Non-goals

- no tensor payload dumps in profiling artifacts
- no always-on heavy tracing overhead in default runs
- no requirement to make raw/full-graph tracing fully symmetric on day one

## 3. Proposed approach

### 3.1 Event model

Use one scalar-only event schema for detailed telemetry.

Recommended event fields:

- `ts_ns` — monotonic timestamp
- `scope` — `run|completion|step|phase|batch|op`
- `name` — stable event name
- `phase` — optional (`phase0|phase1|phase2|phase3|phase4|packaging|save`)
- `step_index` — optional
- `batch_index` — optional
- `elapsed_ms` — event duration
- `count` — optional count for repeated operations
- `attrs` — compact scalar metadata dict only

Example operation names:

- `planner.preflight`
- `planner.phase3_probe_batch`
- `planner.phase4_probe_batch`
- `phase1.trace_invoke`
- `phase3.compute_batch`
- `phase4.compute_batch`
- `row_store.append_rows`
- `row_store.read_feature_rows`
- `row_store.materialize_dense_feature_slice`
- `decoder.load_chunk`
- `encoder.load_rows`
- `packaging.compact_output`
- `artifact.save_compact`
- `artifact.save_raw_graph`

### 3.2 Storage format

Use both:

1. **`telemetry.jsonl`** per completion for detailed event streams
2. **nested summary blocks in `completion.json`** for easy downstream use

Why both:

- JSONL is better for large detailed event sets and offline aggregation
- nested JSON summaries are easier to inspect directly and easier to surface in
  benchmark extraction tables

### 3.3 Data placement

#### Per run

Store in run/scenario summaries:

- total wall time
- model load time
- prompt load / setup time
- planner status (`disabled|skipped|executed`)
- planner chosen fixed feature batch size

#### Per completion

Store in `completion.json`:

- completion total latency
- step latency summary list
- per-phase totals aggregated across steps
- planner summary
- path to `telemetry.jsonl`

#### Per step

Store in step records inside `completion.json`:

- end-to-end step latency
- attribution latency
- token-generation latency
- artifact-save latency
- per-phase latency breakdown
- effective Phase-4 feature batch size

#### Per batch / op

Store only in `telemetry.jsonl`:

- Phase 3 batch timings
- Phase 4 batch timings
- planner probe batches
- memmap reads/writes/materialization
- encoder/decoder load operations
- packaging micro-ops

## 4. Best insertion points

### Fork

#### `circuit_tracer/attribution/attribute_nnsight.py`

Primary phase boundaries and high-level orchestration:

- precompute / setup
- forward pass
- input-vector build
- Phase 3 loop
- Phase 4 loop
- planner preflight
- compact packaging
- cleanup

#### `circuit_tracer/attribution/context_nnsight.py`

Primary batch-level insertion point:

- `compute_batch(...)`
- chunked replay sections inside Phase 3 / Phase 4

#### `circuit_tracer/transcoder/cross_layer_transcoder.py`

Primary storage/load insertion points:

- encoder row loads
- decoder chunk loads
- cache hits / misses / evictions / stores
- chunked reconstruction work

#### `_FileBackedFeatureRowStore` in `attribute_nnsight.py`

Primary row-store I/O insertion points:

- append rows
- read feature rows
- materialize dense feature slice

### Main repo

#### `trace_pipeline_chunked.py`

Step-level orchestration timing:

- attribution call
- token generation
- save compact artifact
- completion manifest write

#### `trace_pipeline.py`

Add equivalent summary timing where supported, even if planner stays unsupported
there.

#### Extraction

- `experiments/exact_trace_bench/extract.py`
- `experiments/extract_benchmark_index.py`
- `experiments/telemetry_gathering.py` (new)

These should surface summary timing fields and optionally consume detailed
JSONL-derived event tables later.

### 4.1 Telemetry gathering tool

Add a dedicated repo-local telemetry gathering script so profiling data is not
left as raw JSONL only.

Current implementation (main repo):

- script: `experiments/telemetry_gathering.py`
- CLI:

```bash
uv run python experiments/telemetry_gathering.py \
  --input-root /path/to/run_root \
  --output-dir experiments/extracted/<run_name>
```

- outputs:
  - `telemetry_runs.csv` (one row per completion-level summary)
  - `telemetry_steps.csv` (one row per traced step with manifest + telemetry step
    aggregates)
  - `telemetry_batches.csv` (batch-scope telemetry events)
  - `telemetry_ops.csv` (op-scope telemetry events)

Recommended responsibilities:

- discover `telemetry.jsonl` files under a run/artifact root
- join telemetry events with `completion.json`, `result.json`, and run metadata
- emit flat analysis-friendly outputs such as:
  - `telemetry_runs.csv`
  - `telemetry_steps.csv`
  - `telemetry_batches.csv`
  - `telemetry_ops.csv`
- compute summary rollups:
  - per-phase totals
  - hottest operations by total time
  - per-step end-to-end breakdowns
  - planner status / chosen batch-size outcomes

This should sit next to the existing extraction tooling, not replace it.
Benchmark extraction stays focused on benchmark rows; telemetry gathering is the
profiling-focused companion tool.

## 5. Planner-specific changes

Planner should skip itself entirely when there is no headroom to scale, e.g.:

- planner enabled, but `feature_batch_size_max <= initial/effective feature_batch_size`

Persist planner state explicitly as one of:

- `disabled`
- `skipped_no_headroom`
- `executed`

and record:

- initial feature batch size
- max feature batch size
- chosen effective feature batch size
- probe batch count

## 6. Acceptance criteria

- exact chunked compact runs produce `telemetry.jsonl` per completion in
  profiling mode
- `completion.json` contains summarized latency breakdowns
- extraction tables expose enough summary fields to compare:
  - per-step e2e latency
  - phase totals
  - batch-size planning outcome
  - row-store / decoder / encoder time aggregates
- planner skip state is visible in artifacts and extraction
- the telemetry gathering tool can produce flat CSV summaries from raw
  `telemetry.jsonl` without requiring manual ad hoc parsing

## 7. Risks and open questions

### Logging overhead

Risk:

- too many events can perturb timings

Mitigation:

- detailed event emission only in profiling mode
- scalar-only payloads
- no additional synchronization unless already required
- flush to JSONL at step/completion boundaries, not every op if avoidable

### Event volume

Risk:

- per-batch + per-op telemetry can become large

Mitigation:

- keep JSONL append-only and easy to post-process
- optionally sample some very hot repeated ops later if volume is too high

### Interpretation

Risk:

- current hypotheses about the bottleneck may be wrong

Mitigation:

- explicitly measure all meaningful operations first, then choose the next speed
  optimization from evidence rather than intuition.
