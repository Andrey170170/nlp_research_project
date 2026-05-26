#!/bin/bash
#SBATCH --job-name=feature-dist
#SBATCH --account=PAS3272
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10724
#SBATCH --time=01:30:00
#SBATCH --array=0-17
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

BATCHES_FILE="${BATCHES_FILE:-experiments/generated/feature_distribution_analysis_batches.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/scratch/PAS3272/kopanev.1/feature_distribution_analysis}"
DECODER_CHUNK_SIZE="${DECODER_CHUNK_SIZE:-256}"

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
echo "Batches file: ${BATCHES_FILE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Decoder chunk size: ${DECODER_CHUNK_SIZE}"
echo "CPU cores: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python -u experiments/run_feature_distribution_analysis.py \
    --batches-file "${BATCHES_FILE}" \
    --batch-index "${SLURM_ARRAY_TASK_ID}" \
    --output-root "${OUTPUT_ROOT}" \
    --decoder-chunk-size "${DECODER_CHUNK_SIZE}"

echo "End: $(date)"
