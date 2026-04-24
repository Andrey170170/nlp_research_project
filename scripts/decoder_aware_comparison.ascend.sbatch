#!/bin/bash
#SBATCH --job-name=da-comparison
#SBATCH --account=PAS3272
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4027
#SBATCH --time=01:30:00
#SBATCH --array=0-1
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
#
# Decoder-aware vs top-K sparsification comparison
#
# Task 0 (array index 0): exact_compact  — compact path, top-K edge selection
# Task 1 (array index 1): exact_save_raw — full-graph path, decoder-aware edge
#                         selection (s_i = a_i^2 * ||D_i||_F^2) + saves .pt
#
# After both tasks finish, run the analysis:
#   uv run python experiments/analyze_decoder_aware_comparison.py \
#       --experiment-root $OUTPUT_ROOT
#
# To generate the scenarios file first (run locally, no GPU needed):
#   uv run python experiments/build_decoder_aware_comparison_configs.py \
#       --prompt-index 94
#
# To submit:
#   sbatch scripts/decoder_aware_comparison.ascend.sbatch
# Or override defaults:
#   SCENARIOS_FILE=... OUTPUT_ROOT=... sbatch scripts/decoder_aware_comparison.ascend.sbatch

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

SCENARIOS_FILE="${SCENARIOS_FILE:-experiments/generated/decoder_aware_comparison_scenarios.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/scratch/PAS3272/vinson1/decoder_aware_comparison}"

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
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python -u experiments/run_sparsification_experiment.py \
    --scenarios-file "${SCENARIOS_FILE}" \
    --output-root "${OUTPUT_ROOT}" \
    --scenario-index "${SLURM_ARRAY_TASK_ID}"

echo "End: $(date)"
