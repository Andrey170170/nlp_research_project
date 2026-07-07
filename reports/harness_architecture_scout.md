# Harness architecture scout: `nlp_research_project`

Date: 2026-06-30
R0 refresh: 2026-07-06, after PLT parity merge into project `main`
(`e020a34`; current plan/spec commit `45976be`).

Status: scouting/research only. No code changes.

## R0 post-merge delta note

- Rechecked the harness map after PLT parity work. The cleanup ordering is
  unchanged in substance: sibling cleanup should still come first; then split the
  runner adapter, CLI command families, and launch surfaces.
- Current high-friction spans: `full_answer/runner.py` is 1,203 lines with
  prefix/full-sequence helpers at 210-258, model-load knob resolution at
  573-595, and `run_real_shard()` at 712-1136; `cli.py` is 2,696 lines with
  `_cmd_download_transcoders()` at 891-931 and `build_parser()` at 1071-2686.
- PLT parity adds a supporting policy seam in `transcoder_config.py` (207 lines)
  and `transcoder_download.py` (126 lines): architecture/provider-family preset
  resolution, corrected hook defaults, identity validation, download allowlists,
  and provider metadata serialization. This is the natural project-side staging
  area for Phase B's pure governor resolver, not a reason to reorder Phase D.

Priority note: the sibling `circuit_tracer` library should be cleaned first. The
harness has several cleanup opportunities, but many are symptoms of the sibling
library exposing a very wide exact-trace interface.

## Package map

```text
src/nlp_research_project/exact_trace_bench/
├── cli.py                         # command router/parser for all harness modes
├── config.py                      # path and default trace config
├── fixtures.py                    # prompt fixture catalog and tiers
├── jobs.py                        # SLURM launch-plan rendering
├── presets.py                     # named preset submission wrapper
├── workspace.py                   # sibling/project snapshot + import verification
├── transcoder_config.py           # architecture/provider-family preset resolver
├── transcoder_download.py         # provider-aware download allowlists
├── baselines.py                   # baseline registry and validation
├── extract.py, graph_compare.py   # extraction and graph comparison helpers
├── scenarios/
│   ├── base.py, canonical.py, tier_api.py
│   └── wave0.py, waves.py
└── full_answer/
    ├── runner.py                  # shard runtime and sibling attribution calls
    ├── schemas.py, selection.py, sharding.py, trajectory.py
    ├── aggregate.py, audit.py, diagnostics.py
    ├── temporal.py, stability.py, roles.py, calibration.py
    └── ...
```

Adjacent legacy/launch areas:

- `experiments/run_sparsification_experiment.py`
- `slurm/exact_trace_bench/*.sbatch`
- Root scripts such as `trace_pipeline_chunked.py`, `circuit_utils.py`

Domain concepts:

- exact-trace scenario tiers: `fast`, `anomaly`, `long_eval`
- clusters: `ascend`, `cardinal`
- canonical prompt gates: `828_base`, `361_base`, `94_base`
- scenario waves and exact-trace knob taxonomy
- architecture/provider-family transcoder config, corrected hook defaults, and
  provider metadata passed to sibling loaders
- immutable project + sibling workspace snapshots
- full-answer trajectory generation, token selection, trace specs, shards,
  aggregation, audits, temporal/stability analysis
- compact/bucketed graph comparison and baseline registry

## Highest-friction modules

### `src/nlp_research_project/exact_trace_bench/full_answer/runner.py`

Observed outline:

- Contains shard selection/loading/dry-run helpers, prefix/full-sequence token
  reconstruction, prefix-view metadata, full-sequence cache/session preparation,
  target-logit forcing, runtime metadata, graph summary, model load knobs,
  attribution performance kwargs, debug sidecar serialization, real shard run,
  summary rows, and exception/trace payloads.
- Prefix/full-sequence metadata helpers span roughly lines 210 through 258.
- Model-load knob resolution spans roughly lines 573 through 595.
- `run_real_shard()` spans roughly lines 712 through 1136.
- Post-merge, the runner consumes architecture-aware config resolution and
  provider metadata rather than forwarding only CLT-oriented sibling knobs.

Why it is shallow:

- The runner is both the orchestrator and the adapter to sibling internals. It
  imports/uses sibling concepts like `FullSequenceWindowAttributionSession`,
  direct NNSight attribution kwargs, compact graph conversion, and bucketed graph
  saving.
- It has weak locality: changing sibling attribution knobs, sidecar payloads,
  session reuse, or shard metadata all touches one module.

Deletion test:

- If the sibling runtime adapter code were deleted from the runner, the core
  shard orchestration would become clearer. This is a real deepening opportunity,
  but the best implementation depends on the sibling cleanup.

### `src/nlp_research_project/exact_trace_bench/cli.py`

Observed outline:

- 30+ `_cmd_*` command handlers.
- 40+ `_cmd_*` command handlers.
- Full-answer trace-spec config handling starts around line 577.
- `_cmd_download_transcoders()` spans roughly lines 891 through 931.
- `build_parser()` spans roughly lines 1071 through 2686.
- Post-merge, the CLI owns provider-family, architecture, and hook propagation
  for trace-spec generation plus download/prefetch paths.

Why it is shallow:

- The CLI module is an interface module, but it is also a deep implementation
  module. Adding a command family or changing arguments requires navigating a
  very large file.

Deletion test:

- Moving command-family parser/handler groups out should concentrate complexity
  by domain, not scatter it. The CLI could become a shallow router.

### `experiments/run_sparsification_experiment.py` and SLURM templates

Why it matters:

- The harness appears to have a split-brain launch path: older experiment runner
  scripts remain connected to some SLURM templates, while newer full-answer work
  uses the packaged `exact-trace-bench` CLI.

Why it is shallow:

- Launch behavior is not behind one stable module. Experiment provenance and
  operational safety are spread across scripts, templates, CLI, and docs.

Deletion test:

- If legacy script logic can be deleted or turned into compatibility glue while
  the packaged CLI remains, launch complexity concentrates.

### `jobs.py` and `presets.py`

Observed outline:

- `jobs.py` renders several launch plans and owns defaults, resource profile
  resolution, output-root defaults, array validation, and template values.
- `presets.py` maps preset names to plan/submission behavior.

Why it is shallow:

- Launch planning, metadata policy, workspace selection, and concrete submission
  behavior are close together. This is less severe than the runner, but it is an
  important operational seam.

Deletion test:

- Rendering plans can be isolated from submission and preset policy. That should
  make launch behavior easier to test without SLURM.

### `workspace.py`

Observed outline:

- Owns ignore rules, git state collection, read-only chmod, uv source path
  parsing, tree copying, snapshot manifests, sibling root resolution, import path
  verification, snapshot creation, and launch workspace resolution.

Why it is mostly good but still broad:

- It is a useful safety module with a real seam. However, snapshot creation and
  import verification are distinct behaviors.

Deletion test:

- Splitting verification from snapshot management would improve locality without
  changing the conceptual module.

### `scenarios/base.py` and `scenarios/waves.py`

Why it matters:

- Scenario/wave modules contain a large policy/data surface for exact-trace run
  families and knob bundles.

Why it is shallow:

- Adding or comparing wave variants can require scanning tables and helper logic
  together. The module interface does not yet clearly separate knob taxonomy,
  cluster profile, and wave bundle concepts.

Deletion test:

- Some wave data can probably move behind deeper modules, but this is lower
  priority than runner/CLI cleanup.

## Deepening candidates

### 1. Full-answer runner runtime adapter

Files/modules involved:

- `src/nlp_research_project/exact_trace_bench/full_answer/runner.py`
- `src/nlp_research_project/exact_trace_bench/transcoder_config.py`
- sibling `circuit_tracer.attribution.attribute_nnsight`
- `trace_pipeline_chunked.py`, `circuit_utils.py` where imported by the runner

Problem:

- The runner mixes shard orchestration with direct sibling library runtime calls,
  graph saving, target forcing, metadata emission, debug sidecars, and error
  handling. It has too much knowledge of sibling internals.
- It now also consumes architecture-aware transcoder load config and provider
  metadata, making the eventual runtime adapter boundary more important.

Solution direction, without detailed new interfaces:

- Deepen the harness seam around “run one trace spec against the sibling runtime”
  while leaving shard orchestration as the high-level caller.

Benefits:

- Locality: changes to sibling attribution kwargs or session reuse stop touching
  the shard runner.
- Leverage: once the sibling library is cleaned, the harness can adapt in one
  place.
- Testability: fake trace-runtime and graph-sink behavior can be unit tested
  without model loading.

Deletion-test assessment:

- Strong, but best after the sibling attribution seam is improved.

Recommendation strength: **Strong**.

### 2. CLI command-family modules

Files/modules involved:

- `src/nlp_research_project/exact_trace_bench/cli.py`
- `src/nlp_research_project/exact_trace_bench/transcoder_config.py`
- `src/nlp_research_project/exact_trace_bench/transcoder_download.py`
- Tests: `tests/test_full_answer_cli.py`, package/import smoke tests.

Problem:

- One giant CLI file owns all parser groups and command handlers. The interface
  is broad by nature, but the implementation does not need to be in one module.
- Provider-family/architecture/hook propagation now appears in both trace-spec
  generation and download command paths.

Solution direction, without detailed new interfaces:

- Group command families behind deeper modules while keeping the installed
  `exact-trace-bench` entrypoint stable.
- Keep the architecture-aware transcoder resolver as shared policy rather than
  duplicating provider defaults inside CLI command handlers.

Benefits:

- Locality: full-answer, baseline, scenario, launch, and calibration commands can
  change independently.
- Locality: provider/hook default changes stay in the resolver and its tests.
- Leverage: adding a new experiment command becomes a local change.
- Testability: parser smoke tests can focus on command groups.

Deletion-test assessment:

- Medium-strong. Parser wiring will remain, but most implementation complexity
  can move out.

Recommendation strength: **Strong**.

### 3. Single launch path / legacy compatibility adapter

Files/modules involved:

- `experiments/run_sparsification_experiment.py`
- `slurm/exact_trace_bench/*.sbatch`
- `exact_trace_bench/jobs.py`, `presets.py`, `cli.py`

Problem:

- There are multiple launch surfaces. This increases provenance risk and makes
  architecture harder to change safely.

Solution direction, without detailed new interfaces:

- Consolidate ordinary launches through one packaged harness path; leave older
  scripts as explicit compatibility wrappers only if still needed.

Benefits:

- Locality: launch defaults, run metadata, and workspace safety live in one place.
- Leverage: fewer templates need updates for every exact-trace default change.
- Testability: launch-plan rendering can be validated without submitting jobs.

Deletion-test assessment:

- Strong if current templates can be redirected and old script logic becomes dead
  or compatibility-only.

Recommendation strength: **Strong**.

### 4. Launch plan policy versus rendering/submission

Files/modules involved:

- `jobs.py`
- `presets.py`

Problem:

- Defaults, resource profile mapping, output placement, template rendering, and
  submission behavior are close together.

Solution direction, without detailed new interfaces:

- Deepen launch planning so policy decisions can be tested independently from
  shell/SLURM execution.

Benefits:

- Locality: resource/output policy changes are easier to audit.
- Leverage: presets and direct launch plans share more behavior.
- Testability: login-safe tests can validate rendered commands and metadata.

Deletion-test assessment:

- Medium.

Recommendation strength: **Worth exploring**.

### 5. Workspace snapshot safety module cleanup

Files/modules involved:

- `workspace.py`
- Tests: `tests/test_exact_trace_bench_path_safety.py`

Problem:

- Snapshot creation, manifest handling, git state recording, sibling root
  discovery, and import verification share one module.

Solution direction, without detailed new interfaces:

- Keep a single workspace safety concept but separate snapshot management from
  import/path verification.

Benefits:

- Locality: import-path safety can be tested without copying workspaces.
- Leverage: snapshot manifests become easier to evolve.
- Testability: unit tests can isolate filesystem behavior.

Deletion-test assessment:

- Medium.

Recommendation strength: **Worth exploring**.

### 6. Scenario/wave policy locality

Files/modules involved:

- `scenarios/base.py`
- `scenarios/waves.py`
- `scenarios/tier_api.py`
- `docs/knob_api_taxonomy.md`

Problem:

- Wave bundles, knob taxonomy, and cluster/tier placement are adjacent but not
  cleanly separated.

Solution direction, without detailed new interfaces:

- Deepen scenario policy around stable domain concepts: tier placement, cluster
  profile, wave bundle, and knob family.

Benefits:

- Locality: adding a new wave or changing default tier placement requires less
  scanning.
- Leverage: docs and code can share a clearer taxonomy.
- Testability: generated scenario JSON can be snapshot-tested by family.

Deletion-test assessment:

- Medium.

Recommendation strength: **Worth exploring**.

## Documentation state

Useful current docs:

- `README.md`
- `AGENTS.md`
- `EXPERIMENTS.md`
- `docs/README.md`
- `docs/harness.md`
- `docs/knob_api_taxonomy.md`
- `docs/full_answer_harness_spec.md`
- `docs/exact_trace_sweep_campaign_spec.md`
- `docs/post_consolidation_cleanup_spec.md`
- `docs/current_project_roadmap.md`
- `experiments/logs/2026-05.jsonl`
- `experiments/logs/2026-06.jsonl`

Staleness risk:

- Several docs are specs or historical plans rather than current architecture.
  The architecture cleanup should reconcile docs against code and recent
  experiment logs rather than treating every doc as current.
- `docs/history/` already contains archived plans. More stale current-looking
  docs may need to move there later, but that is outside this scouting pass.

No obvious local convention found for ADRs or `CONTEXT.md`; that is fine. The
project's documentation convention is docs + experiment logs.

## Lightweight verification hooks for future work

Login-node-safe commands likely useful after implementation begins:

```bash
uv run ruff check .
uv run ty check .
uv run pytest tests/test_exact_trace_bench_package.py
uv run pytest tests/test_exact_trace_bench_path_safety.py
uv run pytest tests/test_exact_trace_bench_knob_taxonomy.py
uv run pytest tests/test_exact_trace_bench_baselines.py
uv run pytest tests/test_full_answer_cli.py
uv run pytest tests/test_full_answer_selection.py tests/test_full_answer_sharding.py tests/test_full_answer_audit.py tests/test_full_answer_runner_aggregate.py
uv run pytest tests/test_full_answer_temporal.py tests/test_full_answer_diagnostics.py tests/test_full_answer_stability.py
```

Avoid running model-loading traces, nnsight tracing, or GPU-heavy validation on
login nodes.

## Risks and unknowns

- Harness cleanup before sibling cleanup may bake in current sibling internals.
- Launch path changes are provenance-sensitive; preserve run metadata and
  workspace snapshot safety.
- Exact-trace defaults must preserve canonical fp32 internal dtype and current
  row-L1 denominator behavior.
- Documentation is useful but stale in places; every cleanup should include a
  small doc reconciliation step.
