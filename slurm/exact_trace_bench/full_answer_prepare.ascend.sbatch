#!/bin/bash
#SBATCH --job-name=full-answer-prep
#SBATCH --account=PAS3272
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10724
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$SLURM_SUBMIT_DIR}"
LIB_WORKSPACE_ROOT="${LIB_WORKSPACE_ROOT:-$WORKSPACE_ROOT/../circuit-tracer_chunked}"

cd "$WORKSPACE_ROOT"

PROMPT_PATH="${PROMPT_PATH:-}"
FIXTURE_CATALOG="${FIXTURE_CATALOG:-}"
FIXTURE_NAME="${FIXTURE_NAME:-}"
OUTPUT_TRAJECTORY="${OUTPUT_TRAJECTORY:?OUTPUT_TRAJECTORY is required}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:?MAX_NEW_TOKENS is required}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-}"
TRAJECTORY_ID="${TRAJECTORY_ID:-}"
INCLUDE_PROMPT_TEXT="${INCLUDE_PROMPT_TEXT:-0}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SLURM_SUBMIT_DIR/.venv}"
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(--output "$OUTPUT_TRAJECTORY" --max-new-tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE")
if [[ -n "$PROMPT_PATH" ]]; then ARGS+=(--prompt-path "$PROMPT_PATH"); fi
if [[ -n "$FIXTURE_CATALOG" ]]; then ARGS+=(--fixture-catalog "$FIXTURE_CATALOG"); fi
if [[ -n "$FIXTURE_NAME" ]]; then ARGS+=(--fixture-name "$FIXTURE_NAME"); fi
if [[ -n "$SEED" ]]; then ARGS+=(--seed "$SEED"); fi
if [[ -n "$TRAJECTORY_ID" ]]; then ARGS+=(--trajectory-id "$TRAJECTORY_ID"); fi
if [[ "$INCLUDE_PROMPT_TEXT" == "1" ]]; then ARGS+=(--include-prompt-text); fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Cluster: ascend"
echo "Start: $(date)"
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "Output trajectory: ${OUTPUT_TRAJECTORY}"
echo "Prompt path: ${PROMPT_PATH:-unset}"
echo "Fixture catalog: ${FIXTURE_CATALOG:-unset}"
echo "Fixture name: ${FIXTURE_NAME:-unset}"
echo "UV project env: ${UV_PROJECT_ENVIRONMENT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run --no-sync exact-trace-bench run-full-answer-trajectory "${ARGS[@]}"

echo "End: $(date)"
