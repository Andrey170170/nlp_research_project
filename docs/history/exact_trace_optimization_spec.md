# Within-Run Exact Trace Optimization and Benchmark Harness Spec

## 1. Problem statement

Exact single-trace runs are now feasible on OSC hardware, but they are still too expensive, too fragile, and too scattered across ad hoc scripts/config files to support fast optimization work.

The next phase of work is to make **single exact traces** more efficient while preserving graph correctness. Concretely, we need:

- a structured exact-trace benchmark/eval harness,
- better visibility into Phase 0 / Phase 3 / Phase 4 runtime and memory costs,
- lower host RAM usage,
- better-controlled VRAM usage,
- and a safe optimization sequence for Phase 4 scheduling and replay.

All benchmark outputs should live on scratch under `/fs/scratch/PAS3272/kopanev.1`, not in user-space, because artifacts are large and scratch is both faster and better suited to repeated benchmark runs.

This work supports **Direction 1: within-run trace optimization** from the midpoint report.

## 2. Scope and non-goals

### Scope

- Exact-mode, single-step benchmark runs for:
  - `828_base`
  - `361_base`
  - `94_base` as anomaly watch only
- Periodic long-eval runs for late-prefix fixtures:
  - especially `361_late`
- Structured orchestration for:
  - scenario generation,
  - SLURM job launch,
  - artifact layout,
  - metric extraction,
  - graph comparison,
  - summary generation
- Within-run optimization of:
  - Phase 0 materialization,
  - Phase 4 replay scheduling,
  - decoder cache usage,
  - CPU/GPU staging,
  - memory-heavy intermediate structures

### Non-goals

- No approximate tracing methods in this phase.
- No broad refactor of the scientific pipeline outside exact-trace benchmarking.
- No late-prefix prompts in the fast inner-loop harness.
- No aggressive Phase 4 queue heuristics until the `94_base` anomaly is understood.
- No assumption that decoder cache alone will solve the hard long-prompt regime.
- No always-on autonomous multi-agent research loop in the first harness version.

## 3. Current benchmark-backed observations

The extracted benchmark tables under `experiments/extracted/weekend_exact_chunked/` support the current midpoint conclusions and should be treated as the baseline for this spec.

### 3.1 Prompt set guidance

- **Inner-loop prompts:** `828_base`, `361_base`
- **Anomaly watch:** `94_base`
- **Long-eval only:** late-prefix fixtures, especially `361_late`

Rationale:

- `361_late` is currently too expensive for routine iteration.
- On Ascend Wave 2, `361_late` at `b128/c2048/cache0` used about **664 GiB RSS** and about **7408 s** in Phase 4.
- Even cache-enabled `361_late` runs remain multi-hour and very high-RAM.

### 3.2 Runtime / memory conclusions from current experiments

1. **Batch size is the primary runtime/VRAM knob.**
   - Example: on Ascend `828_base`, moving from `b128/c2048` to `b256/c4096` reduced Phase 4 time from about **3082 s** to about **2537 s**, while peak CUDA reserved rose from about **15.2 GiB** to about **17.9 GiB**.

2. **Chunk size is secondary.**
   - It matters, but less consistently than batch size.
   - It mostly changes decoder-load / replay efficiency, not the overall memory regime.

3. **Decoder cache can help, but is not the dominant lever in the hardest regime.**
   - On Ascend `361_base` at `b128/c2048`, moving from `cache0` to `cache12g` reduced Phase 4 from about **4869 s** to about **3821 s**.
   - On `361_late`, the same family improved Phase 4 from about **7408 s** to about **6860 s** (`cache12g`) / **6725 s** (`cache16g`), which is real but not transformative.

4. **Host RAM scales steeply with active feature count.**
   - This remains the main systems bottleneck in long-prompt exact tracing.
   - The current implementation still materializes very large host-side structures in exact mode.

5. **Throughput-oriented settings become unattractive on late-prefix stress cases.**
   - On Ascend `361_late`, `b256/c2048/cache12g` reduced Phase 4 only modestly relative to `b128/c2048/cache12g`, but peak CUDA reserved rose from about **27.7 GiB** to about **48.7 GiB**.
   - Therefore, late-prefix runs should stay out of the fast tuning loop.

### 3.3 Prompt 94 anomaly

`94_base` is a correctness/stability watchpoint, not a normal tuning prompt.

Current evidence:

- Same nominal config on both clusters: `b256/c4096/cache0`
- Phase 0 active features are nearly identical:
  - Ascend: about **3.371M**
  - Cardinal: about **3.370M**
- But Phase 4 runtime diverges drastically:
  - Ascend: about **329 s**
  - Cardinal: about **1836 s**

The midpoint analysis already indicates that Phase 4 selects drastically different node sets across hardware despite nearly identical Phase 0 state. Until this is explained, **any optimization that changes Phase 4 ordering, ranking, or accumulation behavior must be treated as high risk**.

Implication: prompt 94 is a blocker for aggressive queue-heuristic work, not a benchmark to optimize against in the same way as 828/361.

Current next-step plan for prompt 94:

- add a dedicated Phase-4 anomaly debug mode,
- keep baseline execution behavior unchanged,
- and collect shadow diagnostics for:
  - frontier refresh evolution,
  - cutoff/tie sensitivity,
  - deterministic tie-break comparison,
  - float64 shadow ranking comparison,
  - frozen-frontier overlap,
  - and environment fingerprinting.

See also: `docs/prompt94_anomaly_debug_spec.md`.

## 4. Proposed benchmark harness structure

The current exact benchmark flow is functional but spread across:

- scenario generators,
- SLURM scripts,
- tracing entrypoints,
- extraction scripts,
- and one-off comparison logic.

We should consolidate this into a dedicated package.

### 4.1 Proposed package layout

```text
experiments/exact_trace_bench/
  __init__.py
  fixtures.py            # canonical prompt sets and run tiers
  configs.py             # benchmark knobs and defaults
  scenarios.py           # scenario generation
  runner.py              # local orchestration entrypoints
  jobs.py                # SLURM-friendly job descriptions
  artifacts.py           # path/schema helpers
  metrics.py             # runtime/memory extraction
  compare_graphs.py      # graph + compact-output comparisons
  aggregate.py           # summary tables and merged outputs
  report.py              # markdown summary generation
  validation.py          # anomaly checks and invariants

scripts/exact_trace_bench/
  run_ascend.sbatch
  run_cardinal.sbatch
  compare_prompt94.sbatch
```

This package should become the default interface for exact benchmark work instead of continuing to add one-off files under `experiments/` and `scripts/`.

Implementation note: the new package should **copy/adapt** the proven extraction logic from the existing scripts in `experiments/` (for example `extract_runlog_metrics.py`, `extract_benchmark_index.py`, `extract_utils.py`) rather than importing those old modules directly. The goal is to reuse the parsing logic, not to create a long-term dependency from the new harness onto the current ad hoc extraction layer.

### 4.2 Canonical benchmark tiers

#### Tier A: fast inner-loop

- `828_base`
- `361_base`

Use for:

- knob sweeps,
- instrumentation validation,
- memory/runtime regressions,
- replay-scheduling experiments that are supposed to preserve semantics.

#### Tier B: anomaly watch

- `94_base`

Use for:

- cross-hardware checks,
- determinism/stability checks,
- validation after any Phase 4 scheduling or queueing changes.

#### Tier C: long-eval only

- `361_late`
- optionally other late fixtures later

Use for:

- periodic stress validation,
- checking whether a change meaningfully improves the real hard regime.

These should not be part of the default iteration loop.

### 4.3 Required harness outputs

Each run should emit a structured directory such as:

```text
 /fs/scratch/PAS3272/kopanev.1/exact_trace_bench/
  <run_id>/
    manifest.json
    config.json
    prompts.json
    raw/
    summaries/
    comparisons/
    logs/
```

Repo-local code should only store:

- harness source,
- small configs/templates,
- and optionally compact summary tables copied back from scratch.

Large raw benchmark artifacts should remain on scratch by default.

Minimum recorded metadata per run:

- prompt / fixture identity
- cluster / GPU type
- attribution batch size
- feature/logit batch sizes
- decoder chunk size
- decoder cache budget
- hidden exact-mode knobs
- active feature count
- per-phase timings
- peak RSS / peak CUDA allocated / peak CUDA reserved
- decoder load counts / seconds
- cache hits / misses / evictions

### 4.4 Required correctness artifacts

For benchmark runs, save enough raw compact-output data to compare graph semantics directly, not only the final top-edge `.npz` summary.

At minimum preserve:

- `selected_features`
- `feature_row_node_indices`
- `logit_row_node_indices`
- `feature_feature_edges`
- `logit_feature_edges`

This is necessary for:

- regression testing,
- cross-hardware comparison,
- prompt-94 investigation,
- and validating future scheduling changes.

### 4.5 Workspace-backed execution model

The harness should support **workspace-backed parallel runs**.

Rationale:

- `nnsight` executes directly against Python source.
- If code changes mid-run, active jobs can break or become invalid.
- Parallel optimization work is safer if each run points at an immutable workspace snapshot.

Recommended model:

- create a per-run or per-branch workspace under scratch,
- launch jobs from that frozen workspace,
- record the source commit / workspace path in the run manifest,
- and treat the workspace as immutable for the life of the job.

This should be the default for any multi-job or multi-researcher benchmark campaign.

### 4.6 Future extension: auto-research workdirs

Once the harness is stable, a lightweight “auto-research” layer is feasible.

Target model:

- multiple researchers/agents each get their own workspace copy,
- the eval harness inside that workspace is treated as read-only during runs,
- library code changes are isolated per workspace,
- jobs are launched to SLURM from that workspace,
- a cheap poller process checks job state,
- and the main researcher/agent resumes only when results are ready.

This is explicitly a **phase-2 extension after the harness works**, not part of the first implementation milestone.

## 5. Optimization roadmap

Optimization work should proceed in the following order.

### 5.1 Stage 1: harness and instrumentation first

Before changing semantics or scheduling behavior:

- centralize benchmark orchestration,
- standardize artifact layout,
- standardize metric extraction,
- and add the missing exact-mode knobs to the scenario/config layer.

This stage should also establish the scratch-backed output layout and frozen-workspace launch path.

This is required so later changes can be measured cleanly and rolled back safely.

### 5.2 Stage 2: expose and tune hidden exact-mode knobs

Expose the following through the benchmark harness:

- `chunked_feature_replay_window`
- `error_vector_prefetch_lookahead`
- `stage_encoder_vecs_on_cpu`
- `stage_error_vectors_on_cpu`
- `row_subchunk_size` if introduced later

These are the lowest-risk optimizations because they mostly surface existing behavior rather than invent new algorithms.

### 5.3 Stage 3: memory refactors

Target the largest structural bottlenecks next.

#### A. Lazy encoder-vector materialization

In exact chunked mode, avoid holding the full encoder-vector set in memory if on-demand loading is possible.

#### B. Streamed error-vector / reconstruction handling

Reduce Phase 0 peak materialization cost by avoiding unnecessary full-tensor residency.

#### C. Review large host-side structures

The current exact path in `attribute_nnsight.py` still allocates a very large `edge_matrix`. This is a likely major contributor to host RAM growth and should be treated as a first-class optimization target.

### 5.4 Stage 4: replay scheduling improvements that preserve frontier membership

Only after instrumentation and memory refactors:

- reorder work within a fixed frontier,
- improve decoder-chunk locality,
- batch more coherently by layer / chunk,
- decouple replay row-subchunk size from decoder chunk size.

Important rule: **preserve frontier membership**. Reordering already-selected work is much safer than changing the set of nodes chosen for expansion.

#### Current implementation order for Stage 4

The next planned Phase 4 speedup work should proceed in two steps:

1. **Fixed-frontier locality reordering first.**
   - Keep the selected `pending` node set exactly unchanged.
   - Reorder only execution order within that frontier.
   - Primary grouping should be by source layer, then decoder chunk / feature locality,
     then position.

2. **Conservative adaptive Phase 4 scaling second.**
   - After locality reordering lands and is benchmarked, add automatic scaling of
     the effective Phase 4 `feature_batch_size` based on observed memory headroom.
   - Keep this first autoscaling pass narrow: do not simultaneously autotune replay
     window, queue refresh policy, or row-subchunk size.

Rationale:

- the benchmark evidence so far suggests memory refactors are useful enablers but
  do not themselves produce the desired Phase 4 speedups,
- `feature_batch_size` remains the primary throughput lever,
- and preserving frontier membership is required until the prompt-94 anomaly story
  is better understood.

### 5.5 Stage 5: algorithmic queue heuristics

This includes things like:

- cost-aware node ordering,
- changing queue refresh policy,
- replacing full ranking with cheaper approximate ranking,
- or other Anthropic-inspired queue heuristics.

This stage is explicitly deferred until:

- the harness is in place,
- raw compact outputs are saved,
- and prompt 94 stability is better understood.

## 6. Acceptance criteria

This spec is complete when all of the following are true.

### Harness

- A dedicated `experiments/exact_trace_bench/` package exists.
- Exact benchmark jobs no longer depend on scattered ad hoc orchestration.
- `828_base` and `361_base` are the default fast benchmark prompts.
- `94_base` is explicitly tagged as anomaly watch.
- Late-prefix fixtures are explicitly tagged as long-eval-only.

### Metrics and artifacts

- Per-run outputs include standardized runtime and memory metrics.
- Per-run outputs include enough raw compact graph data for exact comparisons.
- Large outputs are written to `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/` by default.
- The new harness reuses existing extractor behavior by copying/adapting parsing logic from `experiments/extract_*.py`, not by importing those old modules directly.
- There is a one-command path to:
  - generate scenarios,
  - launch jobs,
  - aggregate results,
  - compare graphs.

### Execution model

- Benchmark runs can be launched from immutable workspace copies.
- Run manifests record the workspace path and source revision used for the job.

### Optimization workflow

- Hidden exact-mode knobs are configurable from the harness.
- Any Phase 4 scheduling change is tested on:
  - `828_base`
  - `361_base`
  - `94_base`
- Prompt 94 remains a required blocker check for Phase 4 changes.

## 7. Risks and open questions

### Risks

1. **Prompt 94 may reflect a correctness bug, not just floating-point sensitivity.**
2. **Changing Phase 4 ordering may silently change selected node sets.**
3. **Host RAM may remain dominant even after GPU-side improvements.**
4. **Late-prefix runs can distort priorities if treated as default tuning workloads.**
5. **Decoder-cache wins may be overestimated if evaluated without full host-RAM context.**
6. **Mutable source trees are unsafe for long `nnsight` jobs if files can change mid-run.**

### Open questions

1. Is the prompt-94 divergence caused by:
   - floating-point tie instability,
   - decoder ordering differences,
   - a chunked replay bug,
   - or another hardware-conditioned path?
2. Which hidden knobs already exist end-to-end, and which still require plumbing?
3. How much host RAM is attributable to:
   - `edge_matrix`,
   - encoder-vector materialization,
   - error-vector materialization,
   - and other exact-mode intermediates?
4. What is the strict correctness invariant for replay-scheduling changes:
   - exact selected-node match,
   - edge-level tolerance,
   - or both?
5. What is the minimum viable polling/orchestration layer for a later auto-research loop:
   - local poller script,
   - cheap always-on agent,
   - or scheduler-native notification?

## Near-term experiment order (1-2 weeks)

1. Create the benchmark harness skeleton and standardized artifact schema.
2. Add scratch-backed output roots and immutable-workspace launch support.
3. Move the current exact benchmark flow behind the new harness interface.
4. Copy/adapt existing extraction logic from `experiments/extract_*.py` into the new package.
5. Add raw compact-output preservation for graph comparison.
6. Add missing exact-mode knobs to config/scenario plumbing.
7. Re-run baseline inner-loop benchmarks on `828_base` and `361_base`.
8. Run prompt-94 compare as an explicit blocker check.
9. Sweep hidden knobs before changing algorithms.
10. Implement memory refactors.
11. Only then test replay-scheduling improvements that preserve frontier membership.
12. Defer queue-heuristic changes until the anomaly story is clearer.
