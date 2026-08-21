#!/bin/bash
#SBATCH --job-name=full-answer-pair-preheated
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=400G
#SBATCH --time=02:00:00
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
: "${TRAJECTORY_PATH:?TRAJECTORY_PATH is required}"
: "${CONTROL_TRACE_SPECS_PATH:?CONTROL_TRACE_SPECS_PATH is required}"
: "${CONTROL_SHARDS_PATH:?CONTROL_SHARDS_PATH is required}"
: "${CANDIDATE_TRACE_SPECS_PATH:?CANDIDATE_TRACE_SPECS_PATH is required}"
: "${CANDIDATE_SHARDS_PATH:?CANDIDATE_SHARDS_PATH is required}"
: "${CONTROL_OUTPUT_ROOT:?CONTROL_OUTPUT_ROOT is required}"
: "${CANDIDATE_OUTPUT_ROOT:?CANDIDATE_OUTPUT_ROOT is required}"
: "${CONTROL_RUN_ID:?CONTROL_RUN_ID is required}"
: "${CANDIDATE_RUN_ID:?CANDIDATE_RUN_ID is required}"
: "${PAIR_OUTPUT_ROOT:?PAIR_OUTPUT_ROOT is required}"
: "${PREHEAT_MODEL_ROOT:?PREHEAT_MODEL_ROOT is required}"
: "${PREHEAT_PROVIDER_ROOT:?PREHEAT_PROVIDER_ROOT is required}"

MIN_CUDA_TOTAL_GIB="${MIN_CUDA_TOTAL_GIB:-130}"
EXPECTED_PROVIDER_FILE_COUNT="${EXPECTED_PROVIDER_FILE_COUNT:-34}"
RUN_CONTROL="${RUN_CONTROL:-1}"
RUN_CANDIDATE="${RUN_CANDIDATE:-1}"

for flag_name in RUN_CONTROL RUN_CANDIDATE; do
    if [[ "${!flag_name}" != "0" && "${!flag_name}" != "1" ]]; then
        echo "$flag_name must be 0 or 1" >&2
        exit 2
    fi
done
if [[ "$RUN_CONTROL" == "0" && "$RUN_CANDIDATE" == "0" ]]; then
    echo "At least one of RUN_CONTROL or RUN_CANDIDATE must be enabled" >&2
    exit 2
fi

for input_path in \
    "$TRAJECTORY_PATH" \
    "$CONTROL_TRACE_SPECS_PATH" \
    "$CONTROL_SHARDS_PATH" \
    "$CANDIDATE_TRACE_SPECS_PATH" \
    "$CANDIDATE_SHARDS_PATH" \
    "$PREHEAT_MODEL_ROOT" \
    "$PREHEAT_PROVIDER_ROOT"; do
    if [[ ! -e "$input_path" ]]; then
        echo "Required input does not exist: $input_path" >&2
        exit 2
    fi
done
if [[ -e "$PAIR_OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing output: $PAIR_OUTPUT_ROOT" >&2
    exit 2
fi
if [[ "$RUN_CONTROL" == "1" && -e "$CONTROL_OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing control output: $CONTROL_OUTPUT_ROOT" >&2
    exit 2
fi
if [[ "$RUN_CANDIDATE" == "1" && -e "$CANDIDATE_OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing candidate output: $CANDIDATE_OUTPUT_ROOT" >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "Trajectory: ${TRAJECTORY_PATH}"
echo "Control output: ${CONTROL_OUTPUT_ROOT}"
echo "Candidate output: ${CANDIDATE_OUTPUT_ROOT}"
echo "Pair output: ${PAIR_OUTPUT_ROOT}"
echo "Run control: ${RUN_CONTROL}"
echo "Run candidate: ${RUN_CANDIDATE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF cache root: ${HF_HUB_CACHE}"
echo "HF offline: ${HF_HUB_OFFLINE}"

cache_preflight_args=(--expected-model-root "$PREHEAT_MODEL_ROOT")
if [[ "$RUN_CONTROL" == "1" ]]; then
    cache_preflight_args+=(--trace-specs "$CONTROL_TRACE_SPECS_PATH")
fi
if [[ "$RUN_CANDIDATE" == "1" ]]; then
    cache_preflight_args+=(--trace-specs "$CANDIDATE_TRACE_SPECS_PATH")
fi
"$UV_PROJECT_ENVIRONMENT/bin/python" \
    "$WORKSPACE_ROOT/slurm/exact_trace_bench/validate_hf_cache.py" \
    "${cache_preflight_args[@]}"
echo "Offline model cache preflight: passed"

mkdir -p "$PAIR_OUTPUT_ROOT"
nvidia-smi -L

"$UV_PROJECT_ENVIRONMENT/bin/python" - "$MIN_CUDA_TOTAL_GIB" "$PAIR_OUTPUT_ROOT/cuda_device.json" <<'PY'
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
    "device_index": 0,
    "device_name": properties.name,
    "total_memory_bytes": total_bytes,
    "total_memory_gib": total_bytes / (1024**3),
    "minimum_total_memory_gib": minimum_gib,
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload))
if total_bytes < minimum_gib * 1024**3:
    raise SystemExit(
        f"GPU capacity gate failed: {total_bytes / (1024**3):.3f} GiB < {minimum_gib:.3f} GiB"
    )
PY

shopt -s nullglob
provider_files=("$PREHEAT_PROVIDER_ROOT"/layer_*_width_262k_l0_small/params.safetensors)
shopt -u nullglob
if [[ "${#provider_files[@]}" -ne "$EXPECTED_PROVIDER_FILE_COUNT" ]]; then
    echo "Provider preheat set has ${#provider_files[@]} files; expected $EXPECTED_PROVIDER_FILE_COUNT" >&2
    exit 2
fi

echo "Preheat start: $(date --iso-8601=seconds)"
preheat_args=(--manifest-output "$PAIR_OUTPUT_ROOT/preheat_manifest.json")
if [[ -n "${PREHEAT_WORKERS:-}" ]]; then
    preheat_args+=(--workers "$PREHEAT_WORKERS")
fi
"$UV_PROJECT_ENVIRONMENT/bin/python" "$WORKSPACE_ROOT/slurm/exact_trace_bench/preheat_file_cache.py" \
    "${preheat_args[@]}" \
    "$PREHEAT_MODEL_ROOT" \
    "${provider_files[@]}"
echo "Preheat end: $(date --iso-8601=seconds)"

GPU_MONITOR_DIR="$PAIR_OUTPUT_ROOT/gpu_monitor"
export GPU_MONITOR_DIR
mkdir -p "$GPU_MONITOR_DIR"
(
    cd "$GPU_MONITOR_DIR"
    exec /uufs/chpc.utah.edu/sys/installdir/chpcscripts/gpu/gpu-monitor.sh
) 2>"$GPU_MONITOR_DIR/gpu-monitor.stderr.log" &
GPU_MONITOR_PID=$!
cleanup_gpu_monitor() {
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup_gpu_monitor EXIT

run_arm() {
    local label="$1"
    local trace_specs="$2"
    local shards="$3"
    local output_root="$4"
    local run_id="$5"
    echo "$label start: $(date --iso-8601=seconds)"
    "$UV_PROJECT_ENVIRONMENT/bin/exact-trace-bench" run-full-answer-shard \
        --trajectory "$TRAJECTORY_PATH" \
        --trace-specs "$trace_specs" \
        --shards "$shards" \
        --shard-id 0 \
        --output-root "$output_root" \
        --run-id "$run_id" \
        --run-name "$label preheated full-H200 comparison" \
        --run-description "Matched 4B/512 exact-trace arm after deterministic model/provider page-cache preheat." \
        --run-goal "Measure full-trace runtime and fidelity without cold file-cache or MIG-allocation confounds."
    echo "$label end: $(date --iso-8601=seconds)"
}

validate_arm() {
    local label="$1"
    local output_root="$2"
    local expected_mode="$3"
    "$UV_PROJECT_ENVIRONMENT/bin/python" \
        "$WORKSPACE_ROOT/slurm/exact_trace_bench/validate_full_answer_feature_row_mode.py" \
        --label "$label" \
        --output-root "$output_root" \
        --expected-mode "$expected_mode"
}

if [[ "$RUN_CONTROL" == "1" ]]; then
    run_arm "control" "$CONTROL_TRACE_SPECS_PATH" "$CONTROL_SHARDS_PATH" \
        "$CONTROL_OUTPUT_ROOT" "$CONTROL_RUN_ID"
    validate_arm "control" "$CONTROL_OUTPUT_ROOT" "cpu_exact"
fi
if [[ "$RUN_CANDIDATE" == "1" ]]; then
    run_arm "candidate" "$CANDIDATE_TRACE_SPECS_PATH" "$CANDIDATE_SHARDS_PATH" \
        "$CANDIDATE_OUTPUT_ROOT" "$CANDIDATE_RUN_ID"
    validate_arm "candidate" "$CANDIDATE_OUTPUT_ROOT" "cuda_windowed"
fi

echo "End: $(date --iso-8601=seconds)"
