#!/bin/bash
#SBATCH --job-name=download-gemma-stack
#SBATCH --account=marasovic
#SBATCH --partition=granite-guest
#SBATCH --qos=granite-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

: "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required}"
# shellcheck disable=SC1091
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/require_snapshot_workspace.sh"
ENV_FILE="${ENV_FILE:-$WORKSPACE_ROOT/.env}"
MODEL_REPOS="${MODEL_REPOS:-google/gemma-3-1b-it,google/gemma-3-4b-it,google/gemma-3-12b-it}"
PROVIDER_FAMILIES="${PROVIDER_FAMILIES:-gemmascope2-clt-1b-medium-affine,gemmascope2-plt-1b-small,gemmascope2-plt-4b-small,gemmascope2-plt-12b-small}"

cd "$WORKSPACE_ROOT"
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/chpc_env.sh"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
: "${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT must point to the external environment}"
export UV_PROJECT_ENVIRONMENT
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Working directory: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "UV project env: ${UV_PROJECT_ENVIRONMENT}"
echo "UV cache dir: ${UV_CACHE_DIR}"
echo "HF home: ${HF_HOME}"
echo "Env file present: $(if [[ -f "$ENV_FILE" ]]; then echo yes; else echo no; fi)"
echo "Model repos: ${MODEL_REPOS}"
echo "Provider families: ${PROVIDER_FAMILIES}"

IFS=',' read -r -a MODEL_ARRAY <<< "$MODEL_REPOS"
for repo in "${MODEL_ARRAY[@]}"; do
    repo="$(echo "$repo" | xargs)"
    if [[ -z "$repo" ]]; then
        continue
    fi
    echo "Downloading model repo: ${repo}"
    "$UV_PROJECT_ENVIRONMENT/bin/python" - "$repo" "$HF_HOME" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
cache_dir = sys.argv[2]
snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
PY
done

ARGS=(download-transcoders --env-file "$ENV_FILE" --cache-dir "$HF_HOME")
IFS=',' read -r -a FAMILY_ARRAY <<< "$PROVIDER_FAMILIES"
for family in "${FAMILY_ARRAY[@]}"; do
    family="$(echo "$family" | xargs)"
    if [[ -n "$family" ]]; then
        ARGS+=(--provider-family "$family")
    fi
done
"$UV_PROJECT_ENVIRONMENT/bin/exact-trace-bench" "${ARGS[@]}"

echo "End: $(date)"
