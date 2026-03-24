#!/bin/bash
#SBATCH --job-name=explore-pipeline
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# This project uses a uv-managed environment. Do not force OSC's PyTorch module
# unless you intend to replace the project's own torch install.
#
# OSC's GPU docs say many jobs need `module load cuda/<version>` to expose NVIDIA
# libraries. For this repo's uv-installed `torch 2.10.0+cu128`, leave CUDA
# modules unloaded unless you hit actual missing-library errors.
# Example fallback:
# module load cuda/12.3.0

source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

python explore_pipeline.py

echo "End: $(date)"
