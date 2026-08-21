# Single-forward batched-VJP implementation closeout

Status: implemented as an opt-in mode; GPU/model qualification pending
Date: 2026-08-08

## Outcome

The runtime now supports `single_forward_batched_vjp` without changing the
`duplicated_lanes` default. Logical attribution capacity remains independent of
physical forward width: the candidate retains one NNSight forward lane and
computes source-layer-grouped VJPs with `torch.autograd.grad(...,
is_grads_batched=True)`. Rows and deferred feature-VJP tape entries are restored
to canonical request order before returning to the shared attribution pipeline.

This is implementation evidence, not a performance or fidelity result. No
Gemma/GemmaScope model was loaded and no GPU job was launched in this closeout.

## Architecture and controls

- `BackwardPlan` and `BackwardExecutionTopology` own mode validation, logical
  capacity, physical lane count, planner capacity, and replay compatibility.
- `BackwardBatchEngine` isolates the established duplicated-lane engine from
  the new single-forward engine; the attribution context resolves the strategy
  once and contains no per-batch mode branch.
- An AST architecture test prevents mode conditionals from spreading outside
  the topology and engine owners.
- Phase 0 rejects missing or mismatched mode, capacity, forward-lane count, and
  internally inconsistent topology metadata.
- Phase-3 gradient-donor replay is rejected for this mode; the canonical native
  gradient path remains required until separately designed and proved.
- Feature-row influence selection was also regrouped into a typed
  `FeatureRowInfluencePolicy`, restoring the existing bounded-domain-object
  architecture gate.

## Failure and telemetry contract

Each batch records mode, logical capacity, physical lanes, source-layer group
count/max rows, autograd call count/time, cotangent total/peak bytes,
contraction time, memory deltas, and the explicit limitation that vmap fallback
visibility is currently PyTorch-runtime-warning-only. Failures identify the
source layer and one of `workspace_prepare`, `cotangent_build`, `autograd_grad`,
`contraction`, `row_reassembly`, or `tape_reassembly`. Error formatting and
telemetry failures cannot replace the primary exception. Partial workspaces and
previously staged tape entries are cleared on later-group failure.

## Login-safe validation

- Sibling focused/broader suite: 103 passed, 7 CUDA-dependent tests skipped.
- Project selection, launch-control, campaign, and preparation suite: 104
  passed.
- New engine tests prove mixed-source-layer batched VJPs against serial VJPs,
  canonical deferred-tape ordering, legacy default/factory behavior, failure
  serialization, and direct batched VJP over a tensor realized by an actual
  NNSight trace lifecycle.
- Ruff passes across the changed sibling and project Python surfaces.
- Targeted `ty` passes for the new topology, engine, workspace, observability,
  and Phase-0 hardening modules. The large project runner and existing Phase-2
  implementation retain pre-existing type-check debt outside this change.

## Next gate

Use
[`ls4_4b_batched_vjp_v1.json`](../experiments/performance_campaigns/ls4_4b_batched_vjp_v1.json)
in strict order:

1. preheated required-mode 4B/256 candidate transition probe;
2. matched 4B/256 control/candidate and strict comparison;
3. only if exact and resource-safe, repeat the probe/pair sequence at 4B/512;
4. only after measured Phase-1 HBM reduction and acceptable runtime at both
   points, rerun the frozen 4B/1,024 gate.

Stop on selection mismatch, PyTorch vectorization incompatibility, fidelity
failure, insufficient HBM reduction, or a projected runtime above the existing
two-hour operational ceiling. Do not promote the mode to a default from these
development points.

## Qualification launch update (2026-08-09)

The immutable preheated 4B/256 candidate transition probe is queued as Slurm
job `1757287`. Its frozen launch specification selects
`single_forward_batched_vjp`, one H200, 400 GiB host RAM, a one-hour walltime,
required `cuda_windowed`, and `measure_only` resource observation. The snapshot,
bundle hashes, scheduler request, and admission gates are recorded in the
append-only August experiment log.

Post-run mechanism validation now optionally fails closed on the effective
backward engine and forward-lane count in addition to the required feature-row
mode. Wrapper syntax, Ruff, diff checks, and 18 focused validator/package tests
pass. The queued immutable snapshot predates this defense-in-depth wrapper
addition, so its topology will be classified from the already-required
requested/effective execution artifacts before any next job is admitted.
