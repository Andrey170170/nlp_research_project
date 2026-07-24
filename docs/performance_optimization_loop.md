# Exact-trace performance optimization loop

Status: isolated engineering workflow  
Branch: `perf/exact-trace-loop`

This loop measures candidate runtime changes without disturbing governor
calibration work. It runs the canonical exact-trace pipeline, compares the
resulting compact graph with frozen Granite H200 references, and fails when the
selected fidelity policy is exceeded. It does not fit, promote, or modify
governor profiles or launch defaults.

Keep changes and reports on the dedicated project and sibling worktrees:

```text
projects/worktrees/exact-trace-perf/
├── nlp_research_project/
└── circuit-tracer_chunked/
```

Both worktrees should be on `perf/exact-trace-loop`. The CLI records the branch,
commit, and dirty-file list for both repositories before execution and verifies
the same state after every case. A source-state change aborts the suite.

This is an explicit live-workspace exception to the ordinary immutable
experiment policy: `run_manifest.json` records `workspace_mode=live`, the
non-empty rationale that the isolated worktree itself is the candidate under
test, and `no_edits_during_run_enforced=true`. Do not edit either worktree while
a case is running.

## Suites

| Suite | Provider | Fixtures | Intended use |
|---|---|---|---|
| `clt-smoke` | Gemma 3 1B GemmaScope-2 CLT | `361_base` | shortest inner loop |
| `clt-pair` | Gemma 3 1B GemmaScope-2 CLT | `828_base`, `361_base` | broader CLT gate |
| `plt-hard` | Gemma 3 1B GemmaScope-2 PLT | `361_base` | longer, harder gate |
| `all` | both above | CLT pair plus PLT hard | release candidate gate |

List the fixed cases without loading a model:

```bash
uv run exact-trace-perf list
```

## Running inside an H200 job

The command is intentionally allocation-local: it does not submit or wrap a
Slurm job. Start or enter a Granite H200 allocation using the normal project
workflow, then run:

```bash
uv run exact-trace-perf run clt-smoke
```

The default fidelity policy is `bounded`. For timing evidence, the command
requires `SLURM_JOB_ID`, cluster `granite`, partition `rai-gpu-grn`, and exactly
one visible full H200 with at least 130,000 MiB reported by `nvidia-smi`.
`--dry-run` is login-safe and prints the underlying commands without creating
output:

```bash
uv run exact-trace-perf run all --dry-run
```

Use a stable, unique ID when comparing iterations:

```bash
uv run exact-trace-perf run clt-pair \
  --run-id perf-column-kernel-v2 \
  --fidelity bounded
```

The longer PLT checkpoint is:

```bash
uv run exact-trace-perf run plt-hard --run-id perf-column-kernel-v2-plt
```

`--allow-non-h200-test-only` exists only for focused automated tests. Results
produced with that override are not H200 performance evidence.

## Fidelity policies

The harness reuses `graph_compare.compare_artifact_dirs` through the existing
sparsification runner's baseline gate.

| Policy | Feature Jaccard | All-edge Jaccard | All-edge Top-256 | All-edge weighted | All-edge magnitude L1 | Token identity |
|---|---:|---:|---:|---:|---:|---:|
| `bounded` | >=0.98 | >=0.98 | >=0.98 | >=0.98 | <=0.02 | exact |
| `exact` | 1.0 | 1.0 | 1.0 | >=0.999999 | <=0.000001 | exact |

The policy applies to the worst aligned generation step, not only the overall
means. All-edge metrics include feature-to-feature and feature-to-logit compact
edges. All-edge magnitude normalized L1 deviation is
`sum(|candidate_magnitude-reference_magnitude|) / max(sum(candidate_magnitude), sum(reference_magnitude), 1e-12)`
over the union of labeled endpoints. Compact artifacts store absolute weights,
so this gate cannot establish internal sign parity.
The comparison must also align every completion and step; missing or
structurally incomplete comparisons fail independently of numeric thresholds.
Overall means remain in the artifacts for compatibility, while
`worst_step_evidence` identifies the completion and step responsible for each
gate value.

`bounded` is an engineering characterization budget for finding promising
performance changes. Passing it is not exact-semantics evidence and must not
promote a runtime mechanism, governor profile, calibration observation, or
launch default. Use `exact`, followed by the project’s separately reviewed
immutable gates, for any exact-semantics claim.

The frozen registry is
`experiments/baselines/exact_trace_performance_granite_20260709.json`. It points
to the corrected-hook `chpc-baseline-gemma-stack-20260709-03` H200 artifacts:

| Case | Baseline duration |
|---|---:|
| CLT `828_base` | 156.50s |
| CLT `361_base` | 105.07s |
| PLT `361_base` | 2805.70s |

These are engineering comparison anchors, not governor calibration observations.
Changing the registry is a reviewed baseline change and should not be folded
into an optimization patch.

The source run recorded a live project workspace at base commit `6a91f1b`
with only its Slurm launcher dirty, plus a clean sibling at `690fc04`; it did
not record an immutable snapshot. The registry pins SHA-256 digests for each
referenced `step_000.npz`, `completion.json`, and `result.json` and verifies
them before tracing, but those artifact digests do not make the historical
live workspace immutable. Treat this as a provenance limitation.

## Outputs

By default, reports are written outside the worktree under:

```text
/scratch/general/vast/$USER/nlp_research_project/exact_trace_bench/
  granite/sweep/performance_optimization/<run-id>/
```

Each run contains:

- `run_manifest.json`: suite, thresholds, baseline reference, Slurm/GPU
  provenance, content hashes for tracked and untracked source, both repository
  states, and timing-evidence eligibility;
- `configs/`: generated single-case scenario files;
- `candidates/`: ordinary sparsification runner logs, compact artifacts,
  results, scenario metrics, and baseline comparisons;
- `performance_report.json`: candidate and baseline duration, speedup, all
  worst-step parity metrics and evidence, and per-case/overall pass status.

`--allow-non-h200-test-only` is recorded as
`test_environment_override_used=true` and
`timing_evidence_eligible=false`; any timing from such a run is explicitly
non-evidence.

The CLI invokes `experiments/run_sparsification_experiment.py` rather than
adding a tracing path. It tails each `run.log` while the candidate is running
and prints one compact report row after comparison.

## Optimization cadence

1. Make one cohesive runtime change on the isolated branch.
2. Run `clt-smoke`.
3. If it is faster and passes, run `clt-pair`.
4. Run `plt-hard` for changes that may affect provider topology, batching,
   reduction order, decoder access, or memory behavior.
5. Use `all` before treating a change as a viable optimization candidate.
6. Keep only changes with a reproducible speedup and a passing parity report.

Custom Triton or CUDA operations belong in the sibling runtime, but only after
the project-side loop is green. Such changes still use this same CLI and frozen
registry; do not create a separate comparison implementation.
