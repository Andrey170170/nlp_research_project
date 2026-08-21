#!/bin/bash
#SBATCH --job-name=full-answer-prep
#SBATCH --account=marasovic
#SBATCH --partition=granite-gpu-guest
#SBATCH --qos=granite-gpu-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

: "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required}"
# shellcheck disable=SC1091
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/require_snapshot_workspace.sh"

cd "$WORKSPACE_ROOT"
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/chpc_env.sh"

PROMPT_PATH="${PROMPT_PATH:-}"
FIXTURE_CATALOG="${FIXTURE_CATALOG:-}"
FIXTURE_NAME="${FIXTURE_NAME:-}"
OUTPUT_TRAJECTORY="${OUTPUT_TRAJECTORY:?OUTPUT_TRAJECTORY is required}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_TRAJECTORY}"
SAMPLE_UNTIL_SUCCESS="${SAMPLE_UNTIL_SUCCESS:-0}"
COLLECT_ALL="${COLLECT_ALL:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:?MAX_NEW_TOKENS is required}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-}"
TRAJECTORY_ID="${TRAJECTORY_ID:-}"
INCLUDE_PROMPT_TEXT="${INCLUDE_PROMPT_TEXT:-0}"
EXPECTED_ANSWER="${EXPECTED_ANSWER:-}"
MAX_SUCCESS_TOKENS="${MAX_SUCCESS_TOKENS:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-}"
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
: "${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT must point to the external environment}"
export UV_PROJECT_ENVIRONMENT
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(--max-new-tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE")
if [[ -n "$PROMPT_PATH" ]]; then ARGS+=(--prompt-path "$PROMPT_PATH"); fi
if [[ -n "$FIXTURE_CATALOG" ]]; then ARGS+=(--fixture-catalog "$FIXTURE_CATALOG"); fi
if [[ -n "$FIXTURE_NAME" ]]; then ARGS+=(--fixture-name "$FIXTURE_NAME"); fi
if [[ "$INCLUDE_PROMPT_TEXT" == "1" ]]; then ARGS+=(--include-prompt-text); fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Cluster: granite"
echo "Start: $(date)"
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "Output trajectory: ${OUTPUT_TRAJECTORY}"
echo "Prompt path: ${PROMPT_PATH:-unset}"
echo "Fixture catalog: ${FIXTURE_CATALOG:-unset}"
echo "Fixture name: ${FIXTURE_NAME:-unset}"
echo "UV project env: ${UV_PROJECT_ENVIRONMENT}"
echo "UV cache dir: ${UV_CACHE_DIR}"
echo "HF home: ${HF_HOME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ "$SAMPLE_UNTIL_SUCCESS" == "1" ]]; then
  SAMPLE_ARGS=(--output-dir "$OUTPUT_DIR" --expected-answer "$EXPECTED_ANSWER" --max-success-tokens "$MAX_SUCCESS_TOKENS" --max-attempts "$MAX_ATTEMPTS")
  if [[ -n "$SEED" ]]; then SAMPLE_ARGS+=(--base-seed "$SEED"); fi
  if [[ -n "$TRAJECTORY_ID" ]]; then SAMPLE_ARGS+=(--trajectory-id-prefix "$TRAJECTORY_ID"); fi
  if [[ -n "$TIME_LIMIT_SECONDS" ]]; then SAMPLE_ARGS+=(--time-limit-seconds "$TIME_LIMIT_SECONDS"); fi
  if [[ "$COLLECT_ALL" == "1" ]]; then SAMPLE_ARGS+=(--collect-all); fi
  "$UV_PROJECT_ENVIRONMENT/bin/exact-trace-bench" sample-full-answer-trajectories "${ARGS[@]}" "${SAMPLE_ARGS[@]}"
else
  if [[ -n "$SEED" ]]; then ARGS+=(--seed "$SEED"); fi
  if [[ -n "$TRAJECTORY_ID" ]]; then ARGS+=(--trajectory-id "$TRAJECTORY_ID"); fi
  "$UV_PROJECT_ENVIRONMENT/bin/exact-trace-bench" run-full-answer-trajectory --output "$OUTPUT_TRAJECTORY" "${ARGS[@]}"
fi

echo "End: $(date)"
