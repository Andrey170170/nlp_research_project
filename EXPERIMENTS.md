# Experiments inventory

Status: Current compact index and interpretation summary
Last updated: 2026-05-20

This file is the readable front page for experiment provenance. It should stay
small enough to edit by hand.

Detailed historical narrative was archived to:

- `docs/history/experiment_log_legacy_2026-05-16.md`

Structured append-only event records now live under:

- `experiments/logs/`

For important future launches, baseline decisions, and reinterpretations:

1. update the compact index here when the result changes current interpretation,
2. append a structured record to `experiments/logs/YYYY-MM.jsonl`, and
3. preserve enough project + sibling-library provenance to reconstruct the run.

## Current baseline

| Item | Current value |
|---|---|
| Project workspace | `/users/PAS2119/andreykopanev/nlp_research_project` |
| Project branch / commit | `main` / `9314f30` (`Record post-consolidation cleanup plan`) |
| Sibling library workspace | `/users/PAS2119/andreykopanev/circuit-tracer_chunked` |
| Sibling branch / commit | `main` / `e91370a` (`Fix Phase-3 row replay effective state`) |
| Editable dependency path | `../circuit-tracer_chunked` |
| Canonical exact-trace dtype | `exact_trace_internal_dtype=fp32` |
| Canonical prompt tiers | `828_base`, `361_base` in `fast`; `94_base` in `anomaly` |
| Scratch root | `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench` |
| Run placement | cluster (`ascend`/`cardinal`) × tier (`fast`/`anomaly`/`long_eval`) |

Baseline preservation notes:

- `828_base` and `361_base` matched same-generation Ascend baselines exactly
  after consolidation.
- `94_base` matched the pre-consolidation optimization-control output exactly;
  mismatch against older Apr-20/21 references predates consolidation.
- The permanent row-L1 overflow fix is part of the validated baseline.
- Within a fixed `decoder_chunk_size`, cache-size changes were exact; changing
  `decoder_chunk_size` can cause small compact-output drift relative to the
  current `c2048` reference.

## Current interpretation

Track-A cross-cluster localization is mostly complete:

- the observed Ascend/Cardinal divergence is primarily driven by Phase-3 gradient
  differences,
- later exact-trace stages amplify those differences enough to affect compact
  circuit outputs,
- debug/replay machinery should remain available as internal validation tooling,
  but it should not dominate the ordinary user-facing workflow.

Clean/current toy parity follow-up (May 2026):

- Current exact-chunked outputs are semantically useful for continued research,
  but strict clean parity depends on NNSight trace/source batch semantics.
- Phase 3 is trace-batch sensitive in the underlying NNSight/replacement path;
  dense/non-chunk and exact-chunked runs agree at the same trace batch, while
  both can differ from singleton. Use singleton trace/batch settings for strict
  clean-like Phase-3 validation.
- Phase 4 is much more stable in same-feature micro-tests, but trace batch >1 can
  produce sparse row-specific differences. Treat batched Phase-4 rows as
  meaningful but not bit-exact clean parity evidence.
- The observed batch sensitivity is not attributed to the exact-chunked decoder
  implementation itself; it appears to be general NNSight/replacement behavior.

Near-term cleanup focus:

1. keep the current exact-trace baseline readable and reproducible,
2. keep docs and scenario defaults from drifting,
3. separate normal benchmark workflow from Track-A replay/debug tooling,
4. add lightweight tests before deeper harness or library refactors.

## Current run families

| Family | Meaning | Current status |
|---|---|---|
| `exact_trace_bench/ascend/fast` | Ascend quick validation/debug for normal prompts | Current |
| `exact_trace_bench/cardinal/fast` | Cardinal quick validation/debug for normal prompts | Current |
| `exact_trace_bench/{ascend,cardinal}/anomaly` | `94_base` anomaly/debug/parity work | Current |
| `exact_trace_bench/{ascend,cardinal}/long_eval` | Longer exact-bench evaluation tier | Current but SLURM-only |
| `workspace_snapshots/` | Immutable project + sibling-library launch snapshots | Current provenance mechanism |
| historical `matched_debug` artifacts | Old matched-debug campaign outputs/configs | Historical only; do not use as an ordinary bucket |

## Recent durable decisions

### 2026-05-20 — Sweep Wave 0/1/2A baseline and Phase-1 decision

Sweep campaign status:

- Wave 0 established the pinned cross-cluster baseline registry:
  `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/baselines/wave0-baseline-20260520-01.json`.
- Wave 1 locked stable resource settings for later waves:
  - Ascend fast: `batch=128`, `decoder_chunk_size=2048`, cache `0`.
  - Ascend anomaly: `batch=256`, `decoder_chunk_size=4096`, cache `0`.
  - Cardinal fast/anomaly: `decoder_chunk_size=4096`, cache `0`.
- Wave 2A Phase-1 trace-batch screening completed successfully after replacing
  Cardinal `c0811` node failures with reruns excluding that node.

Wave 2A effective result:

- 24/24 successful scenarios, all compared against Wave 0 baselines.
- Minimum feature/edge/weighted-edge Jaccard across effective results: `1.0`.
- `cap16` is the safer optional Phase-1 resource candidate, but the ordinary
  Wave 2B path keeps `phase1_trace_batch_policy=legacy` because Phase-1 caps did
  not produce a consistent cross-prompt speed win.
- Cardinal node `c0811` produced CUDA misaligned/illegal-access/CUBLAS failures
  and should be avoided or reported for exact-trace runs.

Structured record:

- `experiments/logs/2026-05.jsonl`

### 2026-05-15/16 — Phase-3 row capture/replay fix validated and committed

Project commit:

- `9314f30` (`Record post-consolidation cleanup plan`)

Sibling library commit:

- `e91370a` (`Fix Phase-3 row replay effective state`)

Validation summary:

- local CPU-only library tests passed:
  - `uv run pytest tests/test_attribute_nnsight_telemetry.py -q`
  - `uv run pytest tests/test_phase3_replay_validation.py -q`
  - `uv run pytest tests/test_partial_influences.py -q`
  - targeted `ruff check`
- Ascend SLURM validation completed for `828_base`, `361_base`, `94_base`, and
  the Phase-3 row donor replay smoke.
- Compact comparisons remained exact for canonical scenarios:
  `feature_jaccard=1.0`, `edge_jaccard=1.0`, `weighted_edge_jaccard=1.0`.
- Row donor replay smoke confirmed finite `float64` row denominators around the
  previous overflow boundary.

Relevant scratch roots:

- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/phase1-row-replay-fix-20260515-01`
- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/phase1-row-replay-fix-20260515-01`
- `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/phase1-row-replay-smoke-20260515-01`

Structured record:

- `experiments/logs/2026-05.jsonl`

### 2026-05-12 — Track-0B consolidation baseline validated

Project and sibling `main` were consolidated and validated as the working
baseline before post-consolidation cleanup.

Key references:

- same-generation post-consolidation references:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/fast/20260512_193309_605201_post-consolidation-ascend-validation`
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260512_193309_737038_post-consolidation-ascend-validation`
- pre-consolidation optimization `94_base` control:
  - `/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend/anomaly/20260512_201118_810555_pre-consolidation-optimization-94-control`

Structured record:

- `experiments/logs/2026-05.jsonl`

## Where to add future information

| Information type | Destination |
|---|---|
| Current baseline, interpretation, run-family meaning | This file |
| Append-only event/run records | `experiments/logs/YYYY-MM.jsonl` |
| Old long-form narratives | `docs/history/` |
| Current workflow policy | `AGENTS.md` |
| Current exact-bench harness usage | `docs/harness.md` and `experiments/exact_trace_bench/README.md` |
| Current cleanup plan | `docs/current_project_roadmap.md` |
| Durable design/spec tradeoffs | Current specs linked from `docs/README.md` |
