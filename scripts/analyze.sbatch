#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --account=PAS3272
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Traces dir: ${TRACES_DIR:-/fs/scratch/PAS3272/kopanev.1/traces}"

uv run python analyze.py \
    --traces-dir "${TRACES_DIR:-/fs/scratch/PAS3272/kopanev.1/traces}" \
    --workers 8

echo "End: $(date)"
