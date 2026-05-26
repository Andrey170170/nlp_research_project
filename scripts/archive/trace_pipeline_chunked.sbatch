#!/bin/bash
#SBATCH --job-name=trace-pipeline-chunked
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=320G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

uv run python -u trace_pipeline_chunked.py \
    --prompts 10 \
    --completions 3 \
    --temperature 0.7 \
    --output-dir /fs/scratch/PAS3272/kopanev.1/traces_chunked \
    --save-raw \
    --max-feature-nodes 8192 \
    --max-edges 100000 \
    --decoder-chunk-size 256 \
    --verbose-attribution \
    --profile-attribution \
    --profile-log-interval 1

echo "End: $(date)"
