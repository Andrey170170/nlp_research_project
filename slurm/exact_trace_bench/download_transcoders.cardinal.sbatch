#!/bin/bash
#SBATCH --job-name=download-transcoders
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$SLURM_SUBMIT_DIR}"
LIB_WORKSPACE_ROOT="${LIB_WORKSPACE_ROOT:-$WORKSPACE_ROOT/../circuit-tracer_chunked}"
ENV_FILE="${ENV_FILE:-$WORKSPACE_ROOT/.env}"
PROVIDER_FAMILIES="${PROVIDER_FAMILIES:-gemmascope2-plt-1b-big-affine,gemmascope2-plt-4b-big-affine}"
TRANSCODER_CACHE_DIR="${TRANSCODER_CACHE_DIR:-/fs/scratch/PAS2836/kopanev.1/huggingface}"

cd "$WORKSPACE_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SLURM_SUBMIT_DIR/.venv}"
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Working directory: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "Provider families: ${PROVIDER_FAMILIES}"
echo "Env file present: $(if [[ -f "$ENV_FILE" ]]; then echo yes; else echo no; fi)"
if [[ -n "$TRANSCODER_CACHE_DIR" ]]; then
    echo "Transcoder cache dir: ${TRANSCODER_CACHE_DIR}"
fi

ARGS=(download-transcoders --env-file "$ENV_FILE")
IFS=',' read -r -a FAMILY_ARRAY <<< "$PROVIDER_FAMILIES"
for family in "${FAMILY_ARRAY[@]}"; do
    family="$(echo "$family" | xargs)"
    if [[ -n "$family" ]]; then
        ARGS+=(--provider-family "$family")
    fi
done
if [[ -n "$TRANSCODER_CACHE_DIR" ]]; then
    ARGS+=(--cache-dir "$TRANSCODER_CACHE_DIR")
fi

uv run --no-sync exact-trace-bench "${ARGS[@]}"

echo "End: $(date)"
