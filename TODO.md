# TODO — main repo (cross-cluster investigation)

This workspace is for **cross-cluster investigation**, not optimization.

## Immediate tasks

1. Baseline status: verified within cluster on the synced merged baseline pair
   (`94e4283` project + `d1f3df3` library).

2. Launch status: submitted matched cross-cluster debug runs from immutable
   snapshot `workspace_20260422_001720_matched_cross_cluster_828_94_fp64` for:
   - `828_base` in `fast`
     - Ascend job `5024202`
     - Cardinal job `8694599`
   - `94_base` in `anomaly`
     - Ascend job `5024203`
     - Cardinal job `8694600`
   - matched config on both clusters:
     - `cross_cluster_debug=true`
     - `exact_trace_internal_dtype=fp64`
     - `decoder_chunk_size=2048`
     - `attribution/feature/logit_batch_size=128`

3. For each paired run, inspect checkpoints in order:
   - Phase 0 sparse setup
   - Phase 1 target logits
   - Phase 2 feature ordering
   - Phase 3 seed ranking
   - Phase 4 evolution
   - first pass completed on the matched single-step runs
   - current read: first structural divergence appears in Phase 0; first clearly
     downstream-meaningful divergence appears by Phase 3 seed ranking

4. Identify the **first meaningful divergence** between Ascend and Cardinal.
   - current answer: Phase 0 is the first structural divergence point; Phase 3
     is the first ranking/frontier divergence point

5. Decide whether the instability is injected before Phase 3 / Phase 4.
   - current answer: before Phase 3 / Phase 4, since Phase-0 feature sets and
     Phase-3 ranking/frontier hashes already diverge

## Questions to answer

- Is `828_base` stable across clusters on the synced baseline?
- On `94_base`, where does divergence start?
- Does Phase 3 still match closely while earlier checkpoints differ?
- Which checkpoint is the first one that explains the later drift?

## Current interpretation

- Within-cluster baseline consistency has been checked and looks good.
- The next task is cross-cluster diagnosis, not more same-cluster validation.
- Start with one healthy control (`828_base`) and one anomalous prompt
  (`94_base`).
- Single-step matched fp64 runs suggest `94_base` does not immediately collapse
  on Cardinal under the matched contract; to test later-step collapse hypotheses
  we likely need a deeper matched run.

## Rules for this workspace

- Do not use this workspace for optimization experiments.
- Keep canonical debug schema and interpretation here.
- Use normal run folders only:
  - `ascend/fast`
  - `ascend/anomaly`
  - `ascend/long_eval`
  - `cardinal/fast`
  - `cardinal/anomaly`
  - `cardinal/long_eval`
