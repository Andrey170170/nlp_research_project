# Weekend exact chunked benchmark plan

## 1. Problem statement

Measure how exact chunked tracing scales on Ascend (A100 40GB) and Cardinal (H100 90GB) as we vary attribution batch size, decoder chunk size, and cross-batch decoder cache budget. The goal is to identify the highest-throughput stable configuration on each cluster, plus the first cache budget that materially improves runtime.

## 2. Scope / non-goals

- In scope: throughput, memory, and stability benchmarking for exact chunked tracing.
- In scope: base prompts for first-wave screening, then late-prefix validation fixtures.
- In scope: GSM8K index 361 as the long stress prompt for continuity.
- In scope: one traced step per scenario.
- Non-goal: multi-step correctness analysis.
- Non-goal: split microbatch knob sweeps in the primary factorial.
- Non-goal: model/debug changes beyond benchmark plumbing and logging.

Primary factorial rule: keep `feature_batch_size = logit_batch_size = attribution_batch_size`; do not include split microbatch knobs in the main matrix.

Fixtures (6 total):
- Base prompts: 828 short, 94 medium, 361 long.
- Late-prefix fixtures: one near-complete deterministic prefix for each of 828 / 94 / 361, derived from a greedy completion and truncated to ~75-85% of the answer or ~2x original prompt tokens.

## 3. Proposed approach

1. Generate late-prefix fixtures first.
2. Run first-wave no-cache sweeps on base prompts only, one sweep on Ascend and one on Cardinal.
3. Wait for those results before selecting follow-up configs.
4. Select per-cluster shortlist: conservative + best-throughput config; add boundary/aggressive only if still interesting.
5. Run cache sweeps on stress cases using `361_base` and `361_late`.
6. Run broader late-prefix validation only on shortlisted configs.

### Matrix

#### A100 / Ascend

Wave 1 no-cache screen (2.5h walltime/job): base prompts only.

- b128 c2048 cache0
- b192 c2048 cache0
- b256 c2048 cache0
- b128 c4096 cache0
- b192 c4096 cache0
- b256 c4096 cache0
- b128 c8192 cache0
- optional boundary probe: b192 c8192 cache0

Wave 2 cache sweep (1.5h walltime/job): stress cases only.

- stress fixtures: `361_base` and `361_late`
- shortlist: conservative config + best-throughput no-cache config; optional boundary probe only if it remains interesting
- cache budgets: 0, 8 GiB, 12 GiB, 16 GiB

Late-prefix validation (2h walltime/job): shortlist only.

- fixtures: `828_late`, `94_late`, `361_late`
- include `361_late` in both shortlisted no-cache configs and shortlisted cache configs to separate prompt-length effects from cache benefits

#### H100 / Cardinal

Wave 1 no-cache screen (2h walltime/job): base prompts only.

- b128 c4096 cache0
- b256 c4096 cache0
- b384 c4096 cache0
- b128 c8192 cache0
- b256 c8192 cache0
- b384 c8192 cache0
- b256 c16384 cache0
- b384 c16384 cache0 (boundary probe)

Wave 2 cache sweep (1h walltime/job): stress cases only.

- stress fixtures: `361_base` and `361_late`
- shortlist: conservative config + best-throughput no-cache config; optional boundary/aggressive configs only if they survive and remain competitive
- cache budgets: 0, 8 GiB, 16 GiB, 24 GiB, 32 GiB

Late-prefix validation (1.5h walltime/job): shortlist only.

- fixtures: `828_late`, `94_late`, `361_late`
- include `361_late` in both shortlisted no-cache configs and shortlisted cache configs to separate prompt-length effects from cache benefits

### Execution notes

- First-wave no-cache jobs must use base prompts only.
- Late-prefix fixtures are a second-wave validation set, not part of the first-wave matrix.
- `361_late` is required in both shortlisted no-cache and shortlisted cache runs.
- Use prepared base and late-prefix fixtures interchangeably only within the appropriate wave.
- Record one scenario per job line item; do not batch multiple scenarios into a single traced step.
- Keep exact tracing path fixed across all runs to isolate scaling effects.

## 4. Acceptance criteria

For every scenario, record:

- status: success / failed / timeout / OOM
- total duration
- Phase 3 duration
- Phase 4 average batch duration and projected total
- peak CUDA allocated / reserved
- host RSS
- cache hits / misses / evictions
- prompt token count / initial input token count
- active feature counts

Benchmark outputs must support:

- highest-throughput stable config identified per cluster
- first clearly effective cache budget identified per cluster
- comparison of base vs late-prefix fixtures for memory use and runtime
- direct comparison across short / medium / long prompt lengths
- separation of prompt-length effects from cache benefits on `361_late`

## 5. Required code/setup changes

- Add fixed-prefix input support to the tracing harness so a scenario can start from a prepared prompt/prefix text file rather than always formatting a GSM8K question.
- Extend logging/manifests with `prompt_token_count`, `initial_input_token_count`, `generated_token_count`, and per-step prefix token count if cheap.
- Add a prefix-preparation utility plus a short Ascend SLURM job to generate deterministic completions for 828 / 94 / 361 and save truncated late-prefix fixtures with token counts.
- Add scenario support for prepared prefix fixtures so the matrix can mix base GSM8K questions and fixed-prefix cases cleanly.
- Add Cardinal sbatch support mirroring the exact sweep launcher used on Ascend.

## 6. Risks / open questions

- Exact cache-budget units must be normalized across cluster launchers before running sweeps.
- Late-prefix generation may need a fallback truncation rule if the deterministic completion is too short or unstable.
- Some boundary/aggressive probes may OOM before meaningful throughput data is collected.
- Need to confirm whether host RSS capture is available in the current manifests without extra sampling overhead.
- Shortlist rule: choose best-throughput config subject to stability/headroom, then add optional boundary/aggressive configs only if they still look competitive.
