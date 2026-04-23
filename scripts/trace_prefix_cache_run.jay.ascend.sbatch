#!/bin/bash
#SBATCH --job-name=prefix-cache-run-jay
#SBATCH --account=PAS2136
#SBATCH --cluster=ascend
#SBATCH --partition=quad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/forward_cache}"

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

# Cached run on the same prompt 94 used for the baseline
# (scripts/trace_prefix_cache_bench.jay.ascend.sbatch).  Identical
# attribution config; only difference is --use-library-prefix-cache, which
# wires a circuit_tracer.attribution.PrefixActivationCache into every
# attribute() call.  The library populates the cache after each step and
# emits phase0.setup.prefix_cache_{lookup,store} trace events so we can
# later compare cached vs uncached step_NNN.npz artifacts for fidelity
# and measure hit rates.
#
# --time bumped to 08:00:00 (from 03:00:00) so the run can:
#   1. complete more than 5 steps for extended fidelity data
#   2. let the completion.json / cache_validation.json actually flush
#      (the 3 h cap killed the prior run mid step-5)
# The driver prints library_cache hit/miss/cached_prefix_len to stdout
# every step now, so even if this run also times out we still have the
# counters in the log.
uv run python -u -m prefix_caching.trace_pipeline_cached \
    --gsm8k-indices 94 \
    --completions 1 \
    --temperature 0 \
    --max-steps 20 \
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
    --output-dir "${OUTPUT_ROOT}"

echo "End: $(date)"
