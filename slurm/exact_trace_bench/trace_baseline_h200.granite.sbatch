#!/bin/bash
#SBATCH --job-name=baseline-h200
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=400G
#SBATCH --time=05:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$SLURM_SUBMIT_DIR}"
LIB_WORKSPACE_ROOT="${LIB_WORKSPACE_ROOT:-$WORKSPACE_ROOT/../circuit-tracer_chunked}"

cd "$WORKSPACE_ROOT"
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/chpc_env.sh"

SCENARIOS_FILE="${SCENARIOS_FILE:-experiments/generated/exact_trace_bench/exact_trace_bench_baseline_gemma3_1b_clt_granite_scenarios.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/general/vast/${USER}/nlp_research_project/exact_trace_bench/granite/baseline/manual}"

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SLURM_SUBMIT_DIR/.venv}"
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Cluster: granite"
echo "Start: $(date)"
echo "Working directory: ${WORKSPACE_ROOT}"
echo "Scenarios file: ${SCENARIOS_FILE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run ID: ${RUN_ID:-unset}"
echo "Run name: ${RUN_NAME:-unset}"
echo "Run description: ${RUN_DESCRIPTION:-unset}"
echo "Run goal: ${RUN_GOAL:-unset}"
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "UV project env: ${UV_PROJECT_ENVIRONMENT}"
echo "UV cache dir: ${UV_CACHE_DIR}"
echo "HF home: ${HF_HOME}"
echo "HF hub cache: ${HF_HUB_CACHE}"
echo "Transformers cache: ${TRANSFORMERS_CACHE}"
echo "HF hub offline: ${HF_HUB_OFFLINE}"
echo "Transformers offline: ${TRANSFORMERS_OFFLINE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$UV_PROJECT_ENVIRONMENT/bin/python" -u experiments/run_sparsification_experiment.py \
    --scenarios-file "${SCENARIOS_FILE}" \
    --output-root "${OUTPUT_ROOT}" \
    --scenario-index "${SLURM_ARRAY_TASK_ID}" \
    "${RUN_METADATA_ARGS[@]}"

echo "End: $(date)"
