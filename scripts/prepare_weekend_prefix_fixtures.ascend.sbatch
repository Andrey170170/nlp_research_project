#!/bin/bash
#SBATCH --job-name=prep-weekend-prefix
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

cd "$SLURM_SUBMIT_DIR"

TARGET_SPEC_FILE="${TARGET_SPEC_FILE:-experiments/weekend_exact_chunked_fixture_targets.json}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/generated/weekend_exact_chunked_fixtures}"
DECODER_CHUNK_SIZE="${DECODER_CHUNK_SIZE:-256}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Cluster: ascend"
echo "Start: $(date)"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo "Target spec file: ${TARGET_SPEC_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python -u experiments/prepare_weekend_prefix_fixtures.py \
    --target-spec-file "${TARGET_SPEC_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --decoder-chunk-size "${DECODER_CHUNK_SIZE}"

echo "End: $(date)"
