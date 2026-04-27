# Temporal Circuit Stability for LLM Reliability

Research project investigating whether the temporal stability of internal
attribution circuits during autoregressive generation can predict answer
correctness in math reasoning (GSM8K).

## Stack

- **Model**: Gemma-3-1B-IT with GemmaScope-2 cross-layer transcoders
- **Interpretability**: [circuit-tracer](https://github.com/safety-research/circuit-tracer)
  via the local editable fork at `../circuit-tracer_chunked`
- **Dataset**: GSM8K (math word problems)
- **Environment manager**: `uv` — all Python invocations must go through `uv run`
  or the `.venv` created by uv.

## HPC constraints

This repo runs on **Ohio Supercomputer Center (OSC)** nodes.
GPU work (model loading, tracing, inference) happens inside SLURM jobs — never
on login nodes.

**For CI / linting / local validation:**

- Only run lint checks (`ruff`, `mypy`) or lightweight unit tests.
- **Never** launch GPU-dependent code, model downloads, or heavy CPU work outside
  a SLURM job.
- If you need to run Python at all, use `uv run <script>` (or activate `.venv`).
- For exact chunked GemmaScope-2 tracing, prefer the fork's
  `cross_batch_decoder_cache_bytes` Phase-4 cache budget instead of re-explaining
  that local fork feature each session.

## Working-doc roles

- `AGENTS.md` = durable repo operating instructions and workflow conventions.
- `PLAN.md` = current scratch/working plan for the active task; do not treat it
  as the durable source of project policy.
- `EXPERIMENTS.md` = living experiment inventory **and investigation log** for
  experiment decisions, launches, findings, run-family meaning, and current
  interpretation notes.
- `docs/phase4_refresh_optimization_spec.md` = durable design/spec document for
  normalization stability and refresh-path optimization work.

When updating project guidance:

- put durable workflow rules in `AGENTS.md`,
- put current execution steps in `PLAN.md`,
- put observed results and run interpretation in `EXPERIMENTS.md`,
- put chosen implementation strategy/tradeoffs in the relevant spec under
  `docs/`.

When updating `EXPERIMENTS.md`:

- add dated entries for important investigations, baseline decisions, and result
  reinterpretations,
- record the relevant **project + sibling library provenance** when possible
  (workspace, branch, commit, and whether the run came from a live workspace or
  immutable snapshot),
- keep enough context that we can later reconstruct what happened and why a
  baseline or interpretation changed.

## Worktree strategy

- Primary workspace (`./`) is the canonical branch for **cross-cluster drift
  investigation**.
- Optimization work should happen in the sibling worktree at:
  - `../worktrees_opt/nlp_research_project`
- The project repo is never enough by itself: every workspace also depends on a
  sibling `circuit-tracer_chunked` checkout.
- Required pairings:
  - main project workspace: `./` ↔ main library workspace: `../circuit-tracer_chunked`
  - optimization project workspace: `../worktrees_opt/nlp_research_project` ↔
    optimization library workspace: `../worktrees_opt/circuit-tracer_chunked`
- Before any serious run, compare **both** the project and sibling library git
  state (branch, commit, dirty files) so we do not accidentally validate one repo
  against the wrong library checkout.
- SLURM launches from a workspace will import whichever sibling library exists at
  that relative path, so library state must be treated as part of the experiment
  definition.

Important:

- do **not** blindly recreate or wipe the optimization worktree,
- refresh it carefully from the validated main baseline,
- refresh the optimization **project+library pair together**, not one without the
  other,
- preserve any still-useful untracked content there (for example docs, fixture
  files, or generated scenario inputs needed to keep the worktree operational),
- only remove/replace files after checking whether they carry unique information.

## Two-track development split

### Track A — cross-cluster investigation

Use the main workspace for:

- paired Ascend/Cardinal debug launches,
- instability localization,
- validation of the canonical debug artifact schema,
- correctness-focused interpretation work.

Preferred prompt sequence:

1. `828_base` in normal `fast`
2. `94_base` in normal `anomaly`

Use:

- `cross_cluster_debug=true`
- `exact_trace_internal_dtype=fp64`

until a permanent overflow fix exists.

### Track B — optimization

Use the optimization worktree for:

1. permanent overflow fix,
2. upcast / RSS redesign,
3. remaining refresh-path speedups,
4. later replay-path or other secondary speed work.

For the permanent overflow fix, follow the chosen spec direction in
`docs/phase4_refresh_optimization_spec.md`:

- preferred first implementation: **scaled row-L1 computation** (or an
  equivalent exact stable normalization representation),
- goal: avoid raw row-abs-sum overflow without relying on fp32 collapse.

## Run placement convention

Keep scratch outputs organized only by:

- cluster: `ascend` / `cardinal`
- tier: `fast` / `anomaly` / `long_eval`

Do **not** introduce extra organizational buckets like `matched_debug` for
ordinary test/debug runs.

Instead, distinguish runs with:

- `run_id`
- `run_name`
- `run_description`
- `run_goal`
- scenario names

Examples:

- `828_base` quick validation/debug runs go under `ascend/fast` and
  `cardinal/fast`
- `94_base` instability/debug runs go under `ascend/anomaly` and
  `cardinal/anomaly`

## Repo layout

```
# ── Pipeline scripts (HPC) ──
trace_pipeline.py            # multi-prompt tracing (runs in SLURM GPU job)
trace_pipeline_chunked.py    # fork-native exact/chunked tracing entrypoint
scripts/trace_pipeline.sbatch        # SLURM script: 10 prompts × 3 completions
scripts/trace_exact_smoke.ascend.sbatch        # one-scenario exact smoke run
scripts/trace_exact_reference_overnight.ascend.sbatch # array of exact reference runs
evaluate.py                  # correctness evaluation (regex + GPT judge)
scripts/evaluate.sbatch      # SLURM script: CPU-only, calls OpenAI API
analyze.py                   # batch analysis with correct/incorrect comparison
scripts/analyze.sbatch       # SLURM script: CPU-only, 8 workers
circuit_utils.py             # shared utilities (sparsification, metrics, .npz I/O)

# ── Exploratory (archived) ──
explore_pipeline.py          # original single-prompt tracing (reference)
explore_pipeline.cardinal.sbatch
explore_analysis.py          # original single-completion analysis (reference)

# ── Docs ──
plan.md                      # weekly implementation plan
project_proposal.pdf         # original project proposal

# ── Data (on scratch, not in repo) ──
/fs/scratch/PAS3272/kopanev.1/traces/
  run_config.json            # tracing run configuration
  prompt_XXX/
    prompt_meta.json         # question, ground truth, GSM8K index
    completion_XXX/
      step_NNN.npz           # compact circuit data (~1-5 MB)
      step_NNN.pt            # raw attribution graph (~460 MB, optional)
      completion.json        # run manifest with per-step metadata
      evaluation.json        # correctness evaluation result
  analysis_summary.json      # aggregate metrics + H2 test results
  comparison_analysis.png    # correct vs incorrect temporal curves
```

## Key data structures

Each `step_NNN.pt` is a `circuit_tracer.Graph` saved via `.to_pt()`:

| Field | Shape | Description |
|---|---|---|
| `adjacency_matrix` | `(N, N)` float32 | Direct-effect matrix over all nodes |
| `active_features` | `(F, 3)` int64 | `[layer, position, feature_idx]` per feature |
| `activation_values` | `(F,)` bfloat16 | Scalar activation per feature |
| `input_tokens` | `(T,)` int64 | Vocab token IDs of the input |
| `logit_targets` | list | Target tokens for attribution |

Node ordering in adjacency matrix: `[features, errors, tokens, logits]`.

## Running

```bash
# 1. Trace: submit GPU job (10 prompts × 3 completions → scratch)
sbatch scripts/trace_pipeline.sbatch

# Exact chunked smoke / reference runs
sbatch scripts/trace_exact_smoke.ascend.sbatch
sbatch scripts/trace_exact_reference_overnight.ascend.sbatch

# 2. Evaluate: submit CPU job (needs OPENAI_KEY in .env)
sbatch scripts/evaluate.sbatch

# 3. Analyze: submit CPU job (reads .npz, produces plots + stats)
sbatch scripts/analyze.sbatch

# Local lint (safe on login node)
uv run ruff check .
uv run ty check .
```
