#!/bin/bash
#SBATCH --job-name=weekend-exact-long
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=700G
#SBATCH --time=04:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$SLURM_SUBMIT_DIR}"
LIB_WORKSPACE_ROOT="${LIB_WORKSPACE_ROOT:-$WORKSPACE_ROOT/../circuit-tracer_chunked}"

cd "$WORKSPACE_ROOT"

SCENARIOS_FILE="${SCENARIOS_FILE:-experiments/generated/exact_trace_bench/exact_trace_bench_long_eval_cardinal_scenarios.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/long_eval}"

RUN_ID=""
RUN_NAME=""
RUN_DESCRIPTION=""
RUN_GOAL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id|--run-name|--run-description|--run-goal)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for argument: $1" >&2
                exit 2
            fi
            case "$1" in
                --run-id) RUN_ID="$2" ;;
                --run-name) RUN_NAME="$2" ;;
                --run-description) RUN_DESCRIPTION="$2" ;;
                --run-goal) RUN_GOAL="$2" ;;
            esac
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

RUN_METADATA_ARGS=()
if [[ -n "$RUN_ID" ]]; then
    RUN_METADATA_ARGS+=(--run-id "$RUN_ID")
fi
if [[ -n "$RUN_NAME" ]]; then
    RUN_METADATA_ARGS+=(--run-name "$RUN_NAME")
fi
if [[ -n "$RUN_DESCRIPTION" ]]; then
    RUN_METADATA_ARGS+=(--run-description "$RUN_DESCRIPTION")
fi
if [[ -n "$RUN_GOAL" ]]; then
    RUN_METADATA_ARGS+=(--run-goal "$RUN_GOAL")
fi
BASELINE_ARGS=()
if [[ -n "${BASELINE_REGISTRY:-}" ]]; then
    BASELINE_ARGS+=(--baseline-registry "${BASELINE_REGISTRY}")
fi
if [[ "${FAIL_ON_BASELINE_MISSING:-0}" == "1" ]]; then
    BASELINE_ARGS+=(--fail-on-baseline-missing)
fi
if [[ "${FAIL_ON_VALIDATION_FAIL:-0}" == "1" ]]; then
    BASELINE_ARGS+=(--fail-on-validation-fail)
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SLURM_SUBMIT_DIR/.venv}"
if [[ -d "$LIB_WORKSPACE_ROOT" ]]; then
    export PYTHONPATH="$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Cluster: cardinal"
echo "Start: $(date)"
echo "Working directory: ${WORKSPACE_ROOT}"
echo "Scenarios file: ${SCENARIOS_FILE}"
echo "Output root: ${OUTPUT_ROOT}"
if [[ -n "$RUN_ID" ]]; then
    echo "Run ID: ${RUN_ID}"
fi
if [[ -n "$RUN_NAME" ]]; then
    echo "Run name: ${RUN_NAME}"
fi
if [[ -n "$RUN_DESCRIPTION" ]]; then
    echo "Run description: ${RUN_DESCRIPTION}"
fi
if [[ -n "$RUN_GOAL" ]]; then
    echo "Run goal: ${RUN_GOAL}"
fi
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "UV project env: ${UV_PROJECT_ENVIRONMENT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python -u experiments/run_sparsification_experiment.py \
    --scenarios-file "${SCENARIOS_FILE}" \
    --output-root "${OUTPUT_ROOT}" \
    --scenario-index "${SLURM_ARRAY_TASK_ID}" \
    "${BASELINE_ARGS[@]}" \
    "${RUN_METADATA_ARGS[@]}"

echo "End: $(date)"
