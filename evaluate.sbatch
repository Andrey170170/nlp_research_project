#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --account=PAS3272
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Traces dir: ${TRACES_DIR:-/fs/scratch/PAS3272/kopanev.1/traces}"

uv run python evaluate.py \
    --traces-dir "${TRACES_DIR:-/fs/scratch/PAS3272/kopanev.1/traces}"

echo "End: $(date)"
