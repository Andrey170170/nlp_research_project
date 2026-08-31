#!/bin/bash
#SBATCH --job-name=diagnose-nnsight-propagation
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

: "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required}"
: "${LIB_WORKSPACE_ROOT:?LIB_WORKSPACE_ROOT is required}"
: "${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT is required}"
: "${BEHAVIORAL_REPORT_PATH:?BEHAVIORAL_REPORT_PATH is required}"
: "${TRACE_SPECS_PATH:?TRACE_SPECS_PATH is required}"
: "${GRAPH_PATH:?GRAPH_PATH is required}"
: "${FRONTIER_PATH:?FRONTIER_PATH is required}"
: "${DIAGNOSTIC_OUTPUT_ROOT:?DIAGNOSTIC_OUTPUT_ROOT is required}"

# shellcheck disable=SC1091
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/require_snapshot_workspace.sh"
cd "$WORKSPACE_ROOT"
# shellcheck disable=SC1091
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/chpc_env.sh"

export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

for input_path in \
    "$BEHAVIORAL_REPORT_PATH" \
    "$TRACE_SPECS_PATH" \
    "$GRAPH_PATH" \
    "$(dirname "$GRAPH_PATH")/trace.json" \
    "$FRONTIER_PATH"; do
    if [[ ! -f "$input_path" || ! -r "$input_path" ]]; then
        echo "Diagnostic input must be a readable regular file: $input_path" >&2
        exit 2
    fi
done
if [[ -e "$DIAGNOSTIC_OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite diagnostic output: $DIAGNOSTIC_OUTPUT_ROOT" >&2
    exit 2
fi
mkdir -p "$DIAGNOSTIC_OUTPUT_ROOT"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "Diagnostic output: ${DIAGNOSTIC_OUTPUT_ROOT}"
nvidia-smi -L

# One process is intentional: this lane localizes a known failure and produces
# no repeatability, qualification, promotion, or scaling claim.
"$UV_PROJECT_ENVIRONMENT/bin/python" \
    "$WORKSPACE_ROOT/scripts/diagnose_nnsight_propagation.py" \
    --behavioral-report "$BEHAVIORAL_REPORT_PATH" \
    --trace-specs "$TRACE_SPECS_PATH" \
    --graph "$GRAPH_PATH" \
    --frontier "$FRONTIER_PATH" \
    --output "$DIAGNOSTIC_OUTPUT_ROOT/diagnostic-receipt.json"

echo "Diagnostic process complete; no qualification or promotion claim"
echo "End: $(date --iso-8601=seconds)"
