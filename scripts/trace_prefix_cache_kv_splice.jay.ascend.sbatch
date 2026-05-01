#!/bin/bash
#SBATCH --job-name=prefix-cache-kv-splice-jay
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

OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/kv_splice}"

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

# KV-splice run: the library prefix cache now stores past_key_values
# alongside the MLP activations. On a partial-hit lookup the trace
# runs only over the new tokens with the cached KV fed back in,
# which should skip the bulk of Phase 0 forward work on consecutive
# steps.
#
# Expected log signals on success:
#   - Step 0 stdout includes phase0.setup.trace_done with splice_active=False
#     (cold start, no cache hit yet).
#   - Step 1+ stdout includes phase0.setup.kv_splice_done and a
#     phase0.setup.trace_done with splice_active=True and a much
#     lower elapsed_s than step 0.
#   - cache_validation.json shows reused_positions == cached_prefix_len
#     on steps after the first.
#
# If the splice fails at runtime the library logs
# phase0.setup.kv_splice_fallback and falls back to a full trace, so
# the run still produces correct (uncached-equivalent) output.
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
    --profile-attribution \
    --output-dir "${OUTPUT_ROOT}"

echo "End: $(date)"
