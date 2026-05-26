#!/bin/bash
#SBATCH --job-name=full-answer-trace-quad
#SBATCH --account=PAS3272
#SBATCH --cluster=ascend
#SBATCH --partition=quad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=700G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$SLURM_SUBMIT_DIR}"
LIB_WORKSPACE_ROOT="${LIB_WORKSPACE_ROOT:-$WORKSPACE_ROOT/../circuit-tracer_chunked}"
cd "$WORKSPACE_ROOT"

TRAJECTORY_PATH="${TRAJECTORY_PATH:?TRAJECTORY_PATH is required}"
TRACE_SPECS_PATH="${TRACE_SPECS_PATH:?TRACE_SPECS_PATH is required}"
SHARDS_PATH="${SHARDS_PATH:?SHARDS_PATH is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
RUN_ID="${RUN_ID:-}"
RUN_NAME="${RUN_NAME:-}"
RUN_DESCRIPTION="${RUN_DESCRIPTION:-}"
RUN_GOAL="${RUN_GOAL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:?--run-id requires a value}"
      shift 2
      ;;
    --run-name)
      RUN_NAME="${2:?--run-name requires a value}"
      shift 2
      ;;
    --run-description)
      RUN_DESCRIPTION="${2:?--run-description requires a value}"
      shift 2
      ;;
    --run-goal)
      RUN_GOAL="${2:?--run-goal requires a value}"
      shift 2
      ;;
    *)
      echo "Unknown full-answer trace launch arg: $1" >&2
      exit 2
      ;;
  esac
done

export RUN_ID RUN_NAME RUN_DESCRIPTION RUN_GOAL

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SLURM_SUBMIT_DIR/.venv}"
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SHARD_ID}"
echo "Node: $(hostname)"
echo "Cluster: ascend"
echo "Partition: ${SLURM_JOB_PARTITION:-unset}"
echo "Start: $(date)"
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "Trajectory: ${TRAJECTORY_PATH}"
echo "Trace specs: ${TRACE_SPECS_PATH}"
echo "Shards: ${SHARDS_PATH}"
echo "Output root: ${OUTPUT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cmd=(uv run --no-sync exact-trace-bench run-full-answer-shard \
  --trajectory "$TRAJECTORY_PATH" \
  --trace-specs "$TRACE_SPECS_PATH" \
  --shards "$SHARDS_PATH" \
  --shard-id "$SHARD_ID" \
  --output-root "$OUTPUT_ROOT")
if [[ -n "$RUN_ID" ]]; then
  cmd+=(--run-id "$RUN_ID")
fi
if [[ -n "$RUN_NAME" ]]; then
  cmd+=(--run-name "$RUN_NAME")
fi
if [[ -n "$RUN_DESCRIPTION" ]]; then
  cmd+=(--run-description "$RUN_DESCRIPTION")
fi
if [[ -n "$RUN_GOAL" ]]; then
  cmd+=(--run-goal "$RUN_GOAL")
fi
"${cmd[@]}"

echo "End: $(date)"
