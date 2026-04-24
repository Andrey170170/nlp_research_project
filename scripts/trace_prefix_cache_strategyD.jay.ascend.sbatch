#!/bin/bash
#SBATCH --job-name=prefix-cache-strategyD-jay
#SBATCH --account=PAS2136
#SBATCH --cluster=ascend
#SBATCH --partition=quad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=700G
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/strategy_d}"

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

# Strategy D run: persist the 12 GiB decoder-chunk cache across
# consecutive attribute() calls instead of rebuilding it each step.
# Decoder chunks depend only on frozen model weights, so this is safe
# by construction and targets the Phase-4 dominant cost.
#
# Compared to the forward_cache_v2 run (same config + prompt + step
# budget) we expect:
#   - step 0: same cost as baseline (cold cache)
#   - step 1+: lower Phase-4 time as decoder chunks hit instead of
#     rebuilding.  The decoder_chunk_session stdout line prints
#     bytes_resident + entries every step so the savings are auditable
#     even if cache_validation.json never flushes.
uv run python -u -m prefix_caching.trace_pipeline_cached \
    --gsm8k-indices 94 \
    --completions 1 \
    --temperature 0 \
    --max-steps 3 \
    --attribution-batch-size 128 \
    --feature-batch-size 128 \
    --logit-batch-size 128 \
    --decoder-chunk-size 2048 \
    --cross-batch-decoder-cache-bytes 12884901888 \
    --max-feature-nodes 8192 \
    --max-edges 20000 \
    --max-n-logits 3 \
    --desired-logit-prob 0.8 \
    --use-library-prefix-cache \
    --use-decoder-chunk-session \
    --output-dir "${OUTPUT_ROOT}"

echo "End: $(date)"
