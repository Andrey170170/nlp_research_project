#!/bin/bash
#SBATCH --job-name=weekend-exact-361
#SBATCH --account=PAS3272
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10724
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

SCENARIOS_FILE="${SCENARIOS_FILE:-experiments/generated/weekend_exact_chunked_wave1_ascend_361_scenarios.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked/ascend/wave1}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Cluster: ascend"
echo "Start: $(date)"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo "Scenarios file: ${SCENARIOS_FILE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "CPU cores: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python -u experiments/run_sparsification_experiment.py \
    --scenarios-file "${SCENARIOS_FILE}" \
    --output-root "${OUTPUT_ROOT}" \
    --scenario-index "${SLURM_ARRAY_TASK_ID}"

echo "End: $(date)"
