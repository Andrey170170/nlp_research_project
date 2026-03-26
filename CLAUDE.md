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
