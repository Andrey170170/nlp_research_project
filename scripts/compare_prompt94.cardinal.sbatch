#!/bin/bash
#SBATCH --job-name=compare-p94
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=350G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

SCENARIOS_FILE="${SCENARIOS_FILE:-experiments/generated/prompt94_compare_cardinal_scenarios.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked/cardinal/prompt94_compare}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Cluster: cardinal"
echo "Start: $(date)"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo "Scenarios file: ${SCENARIOS_FILE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python -u experiments/run_sparsification_experiment.py \
    --scenarios-file "${SCENARIOS_FILE}" \
    --output-root "${OUTPUT_ROOT}" \
    --scenario-index 0

echo "End: $(date)"
