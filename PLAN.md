# Current Execution Plan — Phase-0 Replay Rollout (Project Repo)

## Status checklist

- [x] **Phase 1**: Phase-0 donor bundle capture artifacts wired in project pipeline.
- [x] **Phase 2**: Donor/replay validation metadata plumbed into manifests.
- [x] **Phase 3**: Replay-mode plumbing integrated in exact trace path.
- [x] **Phase 4**: Offline replay comparison helper added.
- [x] **Phase 5 (project-side)**: scenario/launch plumbing + extraction/index coverage + tests.

## Phase 5 outcome (high level)

- Scenario JSON generation now carries exact replay controls:
  `capture_phase0_donor_bundle`, `phase0_donor_bundle`,
  `phase0_replay_mode`, `phase0_donor_context_policy`.
- Exact launcher command building now forwards replay args only for
  `method=exact` (and omits them for `old_patch`).
- Extraction/index summaries now include:
  - Phase-0 donor bundle artifact presence/count/path/status metrics
  - Phase-0 replay mode/status/context policy/donor path
  - Validation warning count (latest + max)
  - Dtype round-trip loss (latest + any)

## Launch procedure (HPC-safe)

1. Generate scenario configs (cluster/tier as needed).
2. Run exact scenarios via SLURM-facing launcher with run metadata.
3. Extract benchmark tables after runs complete.

Reference command skeletons:

```bash
# 1) Build scenario files
uv run python -m experiments.exact_trace_bench build-scenarios \
  --cluster <ascend|cardinal> --all-tiers

# 2) Launch one generated scenario set (no GPU work on login node)
uv run python experiments/run_sparsification_experiment.py \
  --scenarios-file experiments/generated/exact_trace_bench/<scenario_file>.json \
  --output-root /fs/scratch/PAS3272/kopanev.1/exact_trace_bench/<cluster>/<tier>

# 3) Extract index tables
uv run python -m experiments.exact_trace_bench extract --input-root <scratch_root>
```

## Guardrails

- Do not run GPU/model-loading jobs outside SLURM allocations.
- Keep run placement under `{cluster}/{fast|anomaly|long_eval}` only.
- Record interpretation/provenance updates in `EXPERIMENTS.md`.
