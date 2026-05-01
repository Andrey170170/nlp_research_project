#!/bin/bash
#SBATCH --job-name=prefix-cache-kv-splice-debug-jay
#SBATCH --account=PAS2136
#SBATCH --cluster=ascend
#SBATCH --partition=quad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/kv_splice_debug}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_ROOT}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Cluster: ascend"
echo "Start: $(date)"
echo "Output root: ${OUTPUT_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------
# FAST-DEBUG VARIANT.  Goal is purely to surface the
# [KV_SPLICE_TRACEBACK_BEGIN]...[KV_SPLICE_TRACEBACK_END] block on
# stderr as fast as possible — we don't care about the attribution
# graph quality on this run.
#
# Differences from the full kv_splice sbatch:
#   - max-steps 2 (step 0 cold + step 1 splice attempt; step 1 is
#     where the splice fires).
#   - max-feature-nodes / max-edges shrunk so Phase 4 (feature
#     attribution) collapses to seconds instead of ~10 minutes.
#   - decoder-chunk-size 512, decoder cache budget shrunk.
#   - 30-minute wall-clock and 100 GiB RAM so the job queues in the
#     low-priority slot quickly.
#
# Expected outcome: total runtime well under 5 minutes; the splice
# fails on step 1 and dumps a full untruncated traceback to stderr.
# ---------------------------------------------------------------------
uv run python -u -m prefix_caching.trace_pipeline_cached \
    --gsm8k-indices 94 \
    --completions 1 \
    --temperature 0 \
    --max-steps 2 \
    --attribution-batch-size 32 \
    --feature-batch-size 32 \
    --logit-batch-size 32 \
    --decoder-chunk-size 512 \
    --cross-batch-decoder-cache-bytes 1073741824 \
    --max-feature-nodes 256 \
    --max-edges 1000 \
    --max-n-logits 3 \
    --desired-logit-prob 0.8 \
    --use-library-prefix-cache \
    --use-decoder-chunk-session \
    --profile-attribution \
    --output-dir "${OUTPUT_ROOT}"

echo "End: $(date)"
