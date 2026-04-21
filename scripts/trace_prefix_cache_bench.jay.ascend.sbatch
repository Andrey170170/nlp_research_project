#!/bin/bash
#SBATCH --job-name=prefix-cache-bench-jay
#SBATCH --account=PAS2136
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10724
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/baseline_exact}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_ROOT}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Cluster: ascend"
echo "Start: $(date)"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Baseline prefix-cache validation on GSM8K prompt 94.
# Deterministic (temperature=0), 20 steps. This run does NOT use the
# library-side cache yet — it only confirms that prefix features are
# invariant across steps (the premise the whole caching plan rests on).
uv run python -u -m prefix_caching.trace_pipeline_cached \
    --gsm8k-indices 94 \
    --completions 1 \
    --temperature 0 \
    --max-steps 20 \
    --attribution-batch-size 128 \
    --decoder-chunk-size 2048 \
    --output-dir "${OUTPUT_ROOT}"

echo "End: $(date)"
