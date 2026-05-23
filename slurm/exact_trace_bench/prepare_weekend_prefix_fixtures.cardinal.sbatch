#!/bin/bash
#SBATCH --job-name=prep-wave0-prefix
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=350G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$SLURM_SUBMIT_DIR}"
LIB_WORKSPACE_ROOT="${LIB_WORKSPACE_ROOT:-$WORKSPACE_ROOT/../circuit-tracer_chunked}"

cd "$WORKSPACE_ROOT"

TARGET_SPEC_FILE="${TARGET_SPEC_FILE:-experiments/exact_trace_wave0_fixture_targets.json}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/generated/exact_trace_wave0_fixtures}"
DECODER_CHUNK_SIZE="${DECODER_CHUNK_SIZE:-256}"
CROSS_BATCH_DECODER_CACHE_BYTES="${CROSS_BATCH_DECODER_CACHE_BYTES:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SLURM_SUBMIT_DIR/.venv}"
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CACHE_ARGS=()
if [[ -n "$CROSS_BATCH_DECODER_CACHE_BYTES" ]]; then
    CACHE_ARGS+=(--cross-batch-decoder-cache-bytes "$CROSS_BATCH_DECODER_CACHE_BYTES")
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Cluster: cardinal"
echo "Start: $(date)"
echo "Working directory: ${WORKSPACE_ROOT}"
echo "Target spec file: ${TARGET_SPEC_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Decoder chunk size: ${DECODER_CHUNK_SIZE}"
echo "Cross-batch decoder cache bytes: ${CROSS_BATCH_DECODER_CACHE_BYTES:-unset}"
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "UV project env: ${UV_PROJECT_ENVIRONMENT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run --no-sync python -u experiments/prepare_weekend_prefix_fixtures.py \
    --target-spec-file "${TARGET_SPEC_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --decoder-chunk-size "${DECODER_CHUNK_SIZE}" \
    "${CACHE_ARGS[@]}"

echo "End: $(date)"
