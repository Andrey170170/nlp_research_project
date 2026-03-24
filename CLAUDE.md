# Temporal Circuit Stability for LLM Reliability

Research project investigating whether the temporal stability of internal
attribution circuits during autoregressive generation can predict answer
correctness in math reasoning (GSM8K).

## Stack

- **Model**: Gemma-3-1B-IT with GemmaScope-2 cross-layer transcoders
- **Interpretability**: [circuit-tracer](https://github.com/safety-research/circuit-tracer)
  (installed from git)
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

## Repo layout

```
explore_pipeline.py          # main tracing pipeline (runs in SLURM)
explore_pipeline.cardinal.sbatch  # SLURM submission script
explore_analysis.py          # exploratory analysis of traced graphs
plan.md                      # weekly implementation plan
project_proposal.pdf         # original project proposal
experiments/                 # (gitignored) traced graph artifacts
  traces/
    prompt_XXX/
      completion_XXX/
        step_NNN.pt          # per-step attribution graph
        step_NNN_meta.json   # per-step metadata
        completion.json      # run manifest
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
# Submit a tracing job
sbatch explore_pipeline.cardinal.sbatch

# Local lint (safe on login node)
uv run ruff check .
```
