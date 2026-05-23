# Full-Answer / Multi-Token Tracing Harness Spec

Status: Current Track-2 design spec
Last updated: 2026-05-23

## 1. Problem statement

Single-token exact tracing is now organized behind the packaged
`exact-trace-bench` harness, but full-answer tracing is still not practical. A
full answer can contain many candidate token positions, and tracing every token
as a separate SLURM job would repeatedly load Gemma/GemmaScope and waste most of
the runtime budget.

Track 2 needs a harness that can:

1. freeze a generated answer trajectory once,
2. select token positions from that frozen trajectory,
3. build reproducible per-token trace specs,
4. pack those specs into cost-balanced SLURM shards,
5. run each shard from an immutable workspace snapshot while loading model/CLTs
   once per task,
6. aggregate compact graphs, target-token metrics, and timings.

The central safety guarantee from the current harness remains mandatory: SLURM
jobs should execute from a copied read-only workspace snapshot that includes both
this project repo and sibling `../circuit-tracer_chunked`, not from a mutable
live checkout.

## 2. Scope / non-goals

### In scope

1. A full-answer harness path under `src/nlp_research_project/exact_trace_bench/`.
2. Login-safe planning commands for token selection, trace-spec generation,
   sharding, and aggregation.
3. SLURM-only commands/jobs for trajectory freezing and exact token tracing.
4. First-class selected-token tracing modes:
   - explicit generated-token indices,
   - final-answer tokens,
   - numeric tokens,
   - high-surprisal tokens,
   - uniform every-k tokens.
5. Per-token compact graph artifacts and metadata.
6. Cost-balanced shard packing with a simple first cost model.
7. Tests for schema parsing, selection, sharding, and login-safe planning.

### Non-goals for the first implementation

1. Do not attempt all-token full-answer tracing as the first milestone.
2. Do not add baseline comparison/gating to full-answer traces initially.
3. Do not optimize Phase 3/4 internals here; Track 1 owns within-trace runtime
   optimization, mostly in `../circuit-tracer_chunked`.
4. Do not run model loading, generation, nnsight tracing, or exact attribution on
   login nodes.
5. Do not force existing single-token `run_sparsification_experiment.py` to own
   this workflow. Full-answer tracing has different artifacts and should use a
   separate runner.

## 3. Proposed approach

### 3.1 Package layout

Add a dedicated module family:

```text
src/nlp_research_project/exact_trace_bench/full_answer/
├── __init__.py
├── schemas.py
├── trajectory.py
├── selection.py
├── sharding.py
├── runner.py
├── aggregate.py
└── jobs.py
```

Add SLURM templates:

```text
slurm/exact_trace_bench/full_answer_prepare.ascend.sbatch
slurm/exact_trace_bench/full_answer_prepare.cardinal.sbatch
slurm/exact_trace_bench/full_answer_trace.ascend.sbatch
slurm/exact_trace_bench/full_answer_trace.cardinal.sbatch
```

Reuse existing harness pieces:

- `workspace.py` for immutable snapshots and import-path verification,
- `jobs.py` conventions for run metadata and sbatch command rendering,
- `config.py::base_trace_defaults()` for exact-trace defaults,
- `fixtures.py` for fixture/catalog resolution,
- existing compact graph comparison utilities only after per-token artifacts are
  stable.

### 3.2 Artifact model

#### `trajectory.json`

Produced only inside SLURM. It records a frozen generated answer and enough
provenance to reconstruct prefixes later.

Required fields:

```json
{
  "schema_version": 1,
  "trajectory_id": "828_base_greedy_20260523_001122",
  "fixture_name": "828_base",
  "fixture_kind": "base",
  "prepared_prompt_file": ".../prompt.txt",
  "prepared_prompt_meta_file": ".../fixture_meta.json",
  "prompt_text_hash": "sha256:...",
  "prompt_text": "optional, configurable",
  "prompt_token_ids": [1, 2, 3],
  "prompt_token_count": 3,
  "generation": {
    "temperature": 0.0,
    "max_new_tokens": 256,
    "seed": null,
    "stop_token_ids": []
  },
  "generated_tokens": [
    {
      "generated_index": 0,
      "absolute_token_position": 3,
      "token_id": 12345,
      "token_text": "42",
      "logprob": -0.02,
      "probability": 0.98,
      "rank": 1,
      "is_stop": false
    }
  ],
  "completion_text": "...",
  "stop_reason": "eos|max_new_tokens|stop_sequence",
  "timings": {},
  "provenance": {}
}
```

Index contract:

- `generated_index` is 0-based within generated tokens only.
- Prefix for generated token `i` is:

  ```text
  prompt_token_ids + generated_token_ids[:i]
  ```

- Target token for generated token `i` is:

  ```text
  generated_token_ids[i]
  ```

#### `trace_selection.json`

Produced on a login node from `trajectory.json`.

Required fields:

```json
{
  "schema_version": 1,
  "trajectory_id": "...",
  "selection_policy": {
    "explicit_indices": [3, 17],
    "include_final_answer": true,
    "include_numeric": true,
    "high_surprisal_top_k": 10,
    "uniform_every_k": 8
  },
  "selected_indices": [3, 8, 16, 17],
  "selection_reasons": {
    "3": ["explicit"],
    "8": ["uniform_every_k"],
    "17": ["explicit", "numeric"]
  }
}
```

Selection merges all requested policies, deduplicates indices, sorts ascending,
and records all reasons per token.

#### `trace_specs.jsonl`

One JSON object per selected generated token:

```json
{
  "schema_version": 1,
  "trace_id": "traj828_tok000017",
  "trajectory_id": "...",
  "generated_index": 17,
  "prefix_token_count": 140,
  "target_token_id": 12345,
  "target_token_text": "42",
  "target_mode": "frozen_target_only",
  "selection_reasons": ["numeric"],
  "graph_knobs": {
    "exact_trace_internal_dtype": "fp32",
    "max_feature_nodes": 8192,
    "max_edges": 20000
  },
  "estimated_cost": 140
}
```

Initial `estimated_cost` is `prefix_token_count`. Later revisions may use timing
history or active-feature estimates.

#### `shards.json`

Cost-balanced shard assignment:

```json
{
  "schema_version": 1,
  "trace_specs_file": "trace_specs.jsonl",
  "cost_model": "prefix_token_count_lpt_v1",
  "shards": [
    {
      "shard_id": 0,
      "estimated_cost_sum": 10240,
      "spec_indices": [17, 23, 41]
    }
  ]
}
```

Packing algorithm for v1: greedy longest-processing-time assignment by
`estimated_cost` into `--shard-count` bins.

#### Shard outputs

Suggested run layout:

```text
<run_root>/
├── trajectory.json
├── trace_selection.json
├── trace_specs.jsonl
├── shards.json
├── launch_metadata.json
├── shards/
│   └── shard_000/
│       ├── shard.json
│       ├── run.log
│       ├── trace_results.jsonl
│       └── token_000017/
│           ├── graph.npz
│           └── trace.json
├── aggregate.json
├── per_token_metrics.jsonl
└── per_token_metrics.csv
```

Per-token `trace.json` should include:

- trace id, trajectory id, generated index, target token id/text,
- graph path and compact graph summary,
- target probability/logprob/rank from the frozen trajectory or a fresh forward
  pass,
- timing breakdown,
- memory summary where available,
- exact knobs actually used,
- project and sibling library provenance,
- status/error fields.

### 3.3 CLI design

Add flat subcommands to the existing `exact-trace-bench` CLI.

SLURM-only trajectory generation:

```bash
uv run exact-trace-bench submit-full-answer-trajectory \
  --cluster ascend \
  --fixture 828_base \
  --max-new-tokens 256 \
  --temperature 0.0 \
  --immutable-workspace
```

Login-safe token selection and spec generation:

```bash
uv run exact-trace-bench build-full-answer-trace-specs \
  --trajectory /path/to/trajectory.json \
  --select final-answer \
  --select numeric \
  --every-k 8 \
  --indices 3,17,42 \
  --high-surprisal-top-k 10 \
  --output-dir experiments/generated/exact_trace_bench/full_answer/828_base
```

Login-safe sharding:

```bash
uv run exact-trace-bench build-full-answer-shards \
  --trace-specs .../trace_specs.jsonl \
  --shard-count 8 \
  --output .../shards.json
```

SLURM-only shard launch:

```bash
uv run exact-trace-bench launch-full-answer-shards \
  --cluster ascend \
  --trajectory .../trajectory.json \
  --trace-specs .../trace_specs.jsonl \
  --shards .../shards.json \
  --immutable-workspace \
  --run-name "828 selected-token trace"
```

Login-safe aggregation:

```bash
uv run exact-trace-bench aggregate-full-answer-shards \
  --run-root /fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/full_answer/<run_id>
```

### 3.4 SLURM job flow

#### Job 1 — freeze trajectory

1. Runs from immutable workspace snapshot.
2. Loads model/CLTs only inside SLURM.
3. Generates the answer without exact tracing.
4. Records prompt tokens, generated tokens, token text, probabilities/logprobs,
   ranks when available, timings, and provenance.
5. Writes `trajectory.json` to scratch.

#### Login-safe planning

1. Reads `trajectory.json`.
2. Resolves requested selection policies.
3. Writes `trace_selection.json` and `trace_specs.jsonl`.
4. Packs `shards.json`.

#### Job 2 — trace shards

1. SLURM array index selects `shard_id`.
2. Each task loads model/CLTs once.
3. For each assigned spec:
   - reconstruct prefix token ids from the frozen trajectory,
   - force the attribution target to the frozen target token,
   - write per-token compact graph and metadata,
   - clear transient GPU state between specs where practical,
   - append one result row to `trace_results.jsonl`.

The target behavior must be explicit: full-answer tracing should trace the frozen
target token, not whatever token would be selected by the ordinary top-logit path.

### 3.5 Library boundary

Project harness owns:

- trajectory artifact schema,
- selection policies,
- sharding,
- SLURM orchestration,
- aggregation,
- run metadata/provenance.

Sibling `../circuit-tracer_chunked` owns:

- exact attribution implementation,
- target-token forcing API,
- compact graph semantics.

Assumption for the first implementation: the sibling library can accept a forced
target token through its existing or lightly extended `attribution_targets` path.
If that path is not sufficient, add the smallest library change needed to run
exact attribution for a specified target token id at the current final position.

## 4. Acceptance criteria

### 4.1 Login-safe unit/contract tests

1. `trajectory.json`, `trace_selection.json`, `trace_specs.jsonl`, and
   `shards.json` schema helpers round-trip representative examples.
2. Token selection correctly handles explicit indices, final-answer tokens,
   numeric tokens, high-surprisal top-k, uniform every-k, deduplication, and
   reason merging.
3. Trace-spec generation uses the exact prefix/target index contract.
4. Sharding is deterministic and balances simple cost estimates.
5. CLI planning commands run on login nodes without importing/loading model code.
6. SLURM launch planning uses immutable workspace snapshots by default and points
   to snapshot-local scripts/specs.

### 4.2 First SLURM smoke

1. Freeze one deterministic trajectory for a canonical fixture, initially
   `828_base` or `361_base`.
2. Build selected-token specs for a small set, e.g. explicit index + every-k.
3. Run one shard with two or three token specs.
4. Each token output has `trace.json`, compact graph artifact, status, timings,
   target token id/text, and exact knob metadata.
5. Aggregation writes `aggregate.json`, `per_token_metrics.jsonl`, and
   `per_token_metrics.csv`.

### 4.3 Safety/provenance

1. Every SLURM job records project and sibling library branch/commit/dirty files.
2. Shard jobs run from copied read-only workspace snapshots.
3. Scratch output roots stay organized under existing cluster/tier conventions or
   an explicitly documented `full_answer` tier.
4. GPU/model work is never required for planning commands.

## 5. Risks and open questions

1. **Forced target support:** selected tokens, especially high-surprisal tokens,
   may not be top logits. The runner must force the frozen target token.
2. **Token span heuristics:** final-answer and numeric selections may require
   token-text heuristics first, then char/token alignment later.
3. **Shard balance:** prefix length is only a rough cost proxy; active feature
   count and Phase 3/4 behavior may dominate runtime.
4. **Memory drift in long shard tasks:** repeated exact traces in one process may
   leak or fragment GPU memory. Record memory and clear transient state between
   specs.
5. **Cross-cluster reuse:** to compare Ascend/Cardinal, freeze one trajectory and
   reuse the exact token ids/specs on both clusters where possible.
6. **Artifact compatibility:** existing extract/compare code assumes the older
   scenario-root layout. Full-answer aggregation should be separate first;
   compatibility adapters can come later.
7. **Trajectory probability/rank:** if generation code cannot reliably record
   logprob/rank, the trace runner may need a separate forward pass for selected
   tokens.
