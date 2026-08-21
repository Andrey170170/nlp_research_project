#!/bin/bash
#SBATCH --job-name=full-answer-sequence
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=600G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

: "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required}"
# shellcheck disable=SC1091
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/require_snapshot_workspace.sh"
cd "$WORKSPACE_ROOT"
# shellcheck disable=SC1091
source "$WORKSPACE_ROOT/slurm/exact_trace_bench/chpc_env.sh"

: "${LIB_WORKSPACE_ROOT:?LIB_WORKSPACE_ROOT is required}"
: "${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT is required}"
: "${PREPARED_SEQUENCE_PATH:?PREPARED_SEQUENCE_PATH is required}"
: "${SEQUENCE_STATE_ROOT:?SEQUENCE_STATE_ROOT is required}"
: "${PREHEAT_OUTPUT_ROOT:?PREHEAT_OUTPUT_ROOT is required}"
: "${PREHEAT_MODEL_ROOT:?PREHEAT_MODEL_ROOT is required}"
: "${PREHEAT_PROVIDER_ROOT:?PREHEAT_PROVIDER_ROOT is required}"
: "${EXPECTED_PROVIDER_FILE_COUNT:?EXPECTED_PROVIDER_FILE_COUNT is required}"
MIN_CUDA_TOTAL_GIB="${MIN_CUDA_TOTAL_GIB:-130}"

export UV_PROJECT_ENVIRONMENT
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

for input_path in \
    "$PREPARED_SEQUENCE_PATH" \
    "$PREHEAT_MODEL_ROOT" \
    "$PREHEAT_PROVIDER_ROOT"; do
    if [[ ! -e "$input_path" ]]; then
        echo "Required input does not exist: $input_path" >&2
        exit 2
    fi
done
for output_path in "$SEQUENCE_STATE_ROOT" "$PREHEAT_OUTPUT_ROOT"; do
    if [[ -e "$output_path" ]]; then
        echo "Refusing existing sequence output: $output_path" >&2
        exit 2
    fi
done

mapfile -t PREPARED_WORKLOAD_PATHS < <(
    "$UV_PROJECT_ENVIRONMENT/bin/exact-trace-perf" \
        inspect-prepared-campaign-sequence "$PREPARED_SEQUENCE_PATH" \
        --format prepared-paths
)
if [[ "${#PREPARED_WORKLOAD_PATHS[@]}" -eq 0 ]]; then
    echo "Prepared sequence has no workloads" >&2
    exit 2
fi
mapfile -t TRACE_OUTPUT_ROOTS < <(
    "$UV_PROJECT_ENVIRONMENT/bin/exact-trace-perf" \
        inspect-prepared-campaign-sequence "$PREPARED_SEQUENCE_PATH" \
        --format output-paths
)
"$UV_PROJECT_ENVIRONMENT/bin/python" - \
    "$SEQUENCE_STATE_ROOT" "$PREHEAT_OUTPUT_ROOT" "${TRACE_OUTPUT_ROOTS[@]}" <<'PY'
import sys
from pathlib import Path

paths = [Path(raw).resolve() for raw in sys.argv[1:]]
labels = ["sequence state", "preheat output"] + [
    f"trace output {index}" for index in range(len(paths) - 2)
]
for left_index, left in enumerate(paths):
    for right_index in range(left_index + 1, len(paths)):
        right = paths[right_index]
        if left == right or left in right.parents or right in left.parents:
            raise SystemExit(
                f"{labels[left_index]} root collides with {labels[right_index]} "
                f"root: {left} vs {right}"
            )
PY

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "Prepared sequence: ${PREPARED_SEQUENCE_PATH}"
echo "Sequence entries: ${#PREPARED_WORKLOAD_PATHS[@]}"
echo "Sequence state: ${SEQUENCE_STATE_ROOT}"
echo "Preheat output: ${PREHEAT_OUTPUT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF cache root: ${HF_HUB_CACHE}"
echo "HF offline: ${HF_HUB_OFFLINE}"

# Fail closed on every prepared launch, output collision, mechanism contract,
# and offline model resolution before warming caches or creating outputs.
"$UV_PROJECT_ENVIRONMENT/bin/exact-trace-perf" \
    run-prepared-campaign-sequence "$PREPARED_SEQUENCE_PATH" \
    --state-root "$SEQUENCE_STATE_ROOT" --validate-only
for prepared_path in "${PREPARED_WORKLOAD_PATHS[@]}"; do
    "$UV_PROJECT_ENVIRONMENT/bin/python" \
        "$WORKSPACE_ROOT/slurm/exact_trace_bench/validate_hf_cache.py" \
        --prepared-workload "$prepared_path" \
        --expected-model-root "$PREHEAT_MODEL_ROOT"
done
echo "Prepared sequence and offline cache preflight: passed"

mkdir -p "$PREHEAT_OUTPUT_ROOT"
nvidia-smi -L
"$UV_PROJECT_ENVIRONMENT/bin/python" - "$MIN_CUDA_TOTAL_GIB" "$PREHEAT_OUTPUT_ROOT/cuda_device.json" <<'PY'
import json
import socket
import sys
from pathlib import Path

import torch

minimum_gib = float(sys.argv[1])
output = Path(sys.argv[2])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
properties = torch.cuda.get_device_properties(0)
total_bytes = int(properties.total_memory)
payload = {
    "schema_version": 1,
    "hostname": socket.gethostname(),
    "device_name": properties.name,
    "total_memory_bytes": total_bytes,
    "total_memory_gib": total_bytes / (1024**3),
    "minimum_total_memory_gib": minimum_gib,
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload))
if total_bytes < minimum_gib * 1024**3:
    raise SystemExit("GPU capacity gate failed")
PY

shopt -s nullglob
provider_files=("$PREHEAT_PROVIDER_ROOT"/layer_*_width_262k_l0_small/params.safetensors)
shopt -u nullglob
if [[ "${#provider_files[@]}" -ne "$EXPECTED_PROVIDER_FILE_COUNT" ]]; then
    echo "Provider preheat set has ${#provider_files[@]} files; expected $EXPECTED_PROVIDER_FILE_COUNT" >&2
    exit 2
fi

echo "Preheat start: $(date --iso-8601=seconds)"
preheat_args=(--manifest-output "$PREHEAT_OUTPUT_ROOT/preheat_manifest.json")
if [[ -n "${PREHEAT_WORKERS:-}" ]]; then
    preheat_args+=(--workers "$PREHEAT_WORKERS")
fi
"$UV_PROJECT_ENVIRONMENT/bin/python" \
    "$WORKSPACE_ROOT/slurm/exact_trace_bench/preheat_file_cache.py" \
    "${preheat_args[@]}" \
    "$PREHEAT_MODEL_ROOT" \
    "${provider_files[@]}"
echo "Preheat end: $(date --iso-8601=seconds)"

"$UV_PROJECT_ENVIRONMENT/bin/exact-trace-perf" \
    run-prepared-campaign-sequence "$PREPARED_SEQUENCE_PATH" \
    --state-root "$SEQUENCE_STATE_ROOT"

echo "End: $(date --iso-8601=seconds)"
