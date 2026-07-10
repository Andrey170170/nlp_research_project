#!/bin/bash
#SBATCH --job-name=download-transcoders
#SBATCH --account=marasovic
#SBATCH --partition=granite-guest
#SBATCH --qos=granite-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

: "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required}"
# shellcheck disable=SC1091
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/require_snapshot_workspace.sh"
ENV_FILE="${ENV_FILE:-$WORKSPACE_ROOT/.env}"
PROVIDER_FAMILIES="${PROVIDER_FAMILIES:-gemmascope2-plt-1b-big-affine,gemmascope2-plt-4b-big-affine}"

cd "$WORKSPACE_ROOT"
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/chpc_env.sh"

TRANSCODER_CACHE_DIR="${TRANSCODER_CACHE_DIR:-$HF_HOME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
: "${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT must point to the external environment}"
export UV_PROJECT_ENVIRONMENT
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Working directory: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "UV cache dir: ${UV_CACHE_DIR}"
echo "HF home: ${HF_HOME}"
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

"$UV_PROJECT_ENVIRONMENT/bin/exact-trace-bench" "${ARGS[@]}"

echo "End: $(date)"
