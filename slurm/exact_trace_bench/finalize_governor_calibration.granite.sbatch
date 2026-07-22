#!/bin/bash
# Submit with dependency after both arrays and one <task-job-id>::<scenario-root> argument per row.
# Example: sbatch --dependency=afterany:123:124 --export=ALL,WORKSPACE_ROOT=...,LIB_WORKSPACE_ROOT=...,BUNDLE_OUTPUT=... "$0" 123_0::/abs/row0 124_0::/abs/row9
#SBATCH --job-name=governor-cal-finalize
#SBATCH --account=marasovic
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

: "${WORKSPACE_ROOT:?WORKSPACE_ROOT must be an immutable project snapshot}"
: "${LIB_WORKSPACE_ROOT:?LIB_WORKSPACE_ROOT must be the matching library snapshot}"
: "${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT must point to the external environment}"
: "${BUNDLE_OUTPUT:?BUNDLE_OUTPUT must name the post-campaign response bundle}"
: "${BASELINE_REGISTRY:?BASELINE_REGISTRY must be the snapshot-pinned registry}"

# shellcheck disable=SC1091
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/require_snapshot_workspace.sh"
if [[ $# -eq 0 ]]; then
    echo "Pass one <task-job-id>::<absolute-scenario-root> argument per Wave C row" >&2
    exit 2
fi

export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORKSPACE_ROOT"

OBSERVATION_ARGS=()
if [[ -n "${HISTORICAL_OBSERVATION_MANIFEST:-}" ]]; then
    while IFS= read -r observation; do
        [[ -z "$observation" || "$observation" == \#* ]] && continue
        [[ "$observation" == /* ]] || {
            echo "Historical observation paths must be absolute: $observation" >&2
            exit 2
        }
        OBSERVATION_ARGS+=(--observation "$observation")
    done < "$HISTORICAL_OBSERVATION_MANIFEST"
fi

for source in "$@"; do
    job_id="${source%%::*}"
    root="${source#*::}"
    if [[ "$job_id" == "$source" || "$root" != /* ]]; then
        echo "Invalid source mapping, expected <task-job-id>::<absolute-root>: $source" >&2
        exit 2
    fi
    "$UV_PROJECT_ENVIRONMENT/bin/python" -m nlp_research_project.exact_trace_bench \
        finalize-calibration --job-id "$job_id" --root "$root" \
        --baseline-registry "$BASELINE_REGISTRY"
    OBSERVATION_ARGS+=(--observation "$root/calibration_observation.json")
done

"$UV_PROJECT_ENVIRONMENT/bin/python" -m nlp_research_project.exact_trace_bench \
    publish-response-bundle "${OBSERVATION_ARGS[@]}" --output "$BUNDLE_OUTPUT"
