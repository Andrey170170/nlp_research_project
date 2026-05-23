#!/bin/bash
#SBATCH --job-name=trace-chunked-tests
#SBATCH --account=PAS3272
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10724
#SBATCH --time=04:00:00
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
echo "Cluster: ascend"
echo "Start: $(date)"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python -u tests/run_chunked_trace_matrix.py \
    --scenarios-file tests/chunked_trace_scenarios.json \
    --output-root /fs/scratch/PAS3272/kopanev.1/trace_chunked_test_matrix

echo "End: $(date)"
