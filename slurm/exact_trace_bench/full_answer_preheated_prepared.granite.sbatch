#!/bin/bash
#SBATCH --job-name=full-answer-preheated
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=600G
#SBATCH --time=08:00:00
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
: "${PREPARED_WORKLOAD_PATH:?PREPARED_WORKLOAD_PATH is required}"
: "${TRACE_OUTPUT_ROOT:?TRACE_OUTPUT_ROOT is required}"
: "${PROBE_OUTPUT_ROOT:?PROBE_OUTPUT_ROOT is required}"
: "${PREHEAT_MODEL_ROOT:?PREHEAT_MODEL_ROOT is required}"
: "${PREHEAT_PROVIDER_ROOT:?PREHEAT_PROVIDER_ROOT is required}"

: "${EXPECTED_PROVIDER_FILE_COUNT:?EXPECTED_PROVIDER_FILE_COUNT is required}"
MIN_CUDA_TOTAL_GIB="${MIN_CUDA_TOTAL_GIB:-130}"

# The selected mechanism resolver is snapshot code. Establish the immutable
# project/library import path before invoking any Python module from that snapshot;
# the external environment may still point at an older editable checkout.
export UV_PROJECT_ENVIRONMENT
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PREPARED_WORKLOAD_INPUT="$PREPARED_WORKLOAD_PATH"
if [[ -d "$PREPARED_WORKLOAD_PATH" ]]; then
    PREPARED_WORKLOAD_PATH="${PREPARED_WORKLOAD_PATH%/}/prepared_workload.json"
fi
if [[ ! -f "$PREPARED_WORKLOAD_PATH" || ! -r "$PREPARED_WORKLOAD_PATH" ]]; then
    echo "Prepared workload must resolve to a readable regular file: $PREPARED_WORKLOAD_PATH" >&2
    exit 2
fi

mapfile -t PREPARED_EXPECTATIONS < <(
    "$UV_PROJECT_ENVIRONMENT/bin/python" -m \
        nlp_research_project.exact_trace_bench.full_answer.prepared_expectations \
        "$PREPARED_WORKLOAD_PATH" --format lines
)
if [[ "${#PREPARED_EXPECTATIONS[@]}" -ne 16 ]]; then
    echo "Prepared workload did not yield sixteen execution expectations" >&2
    exit 2
fi
reconcile_expectation() {
    local name="$1"
    local supplied="$2"
    local prepared="$3"
    if [[ -n "$supplied" && "$supplied" != "$prepared" ]]; then
        echo "$name=$supplied conflicts with prepared value $prepared" >&2
        exit 2
    fi
    printf '%s' "$prepared"
}
EXPECTED_MODE="$(reconcile_expectation EXPECTED_MODE "${EXPECTED_MODE:-}" "${PREPARED_EXPECTATIONS[0]}")"
EXPECTED_BACKWARD_ENGINE_MODE="$(reconcile_expectation EXPECTED_BACKWARD_ENGINE_MODE "${EXPECTED_BACKWARD_ENGINE_MODE:-}" "${PREPARED_EXPECTATIONS[1]}")"
EXPECTED_FORWARD_GRAPH_MODE="$(reconcile_expectation EXPECTED_FORWARD_GRAPH_MODE "${EXPECTED_FORWARD_GRAPH_MODE:-}" "${PREPARED_EXPECTATIONS[2]}")"
EXPECTED_VJP_KERNEL_MODE="$(reconcile_expectation EXPECTED_VJP_KERNEL_MODE "${EXPECTED_VJP_KERNEL_MODE:-}" "${PREPARED_EXPECTATIONS[3]}")"
EXPECTED_FORWARD_LANE_COUNT="$(reconcile_expectation EXPECTED_FORWARD_LANE_COUNT "${EXPECTED_FORWARD_LANE_COUNT:-}" "${PREPARED_EXPECTATIONS[4]}")"
EXPECTED_SESSION_CAPACITY="$(reconcile_expectation EXPECTED_SESSION_CAPACITY "${EXPECTED_SESSION_CAPACITY:-}" "${PREPARED_EXPECTATIONS[5]}")"
EXPECTED_BACKWARD_BATCH_CAPACITY="$(reconcile_expectation EXPECTED_BACKWARD_BATCH_CAPACITY "${EXPECTED_BACKWARD_BATCH_CAPACITY:-}" "${PREPARED_EXPECTATIONS[6]}")"
EXPECTED_PHASE1_TRACE_BATCH_SIZE_MAX="$(reconcile_expectation EXPECTED_PHASE1_TRACE_BATCH_SIZE_MAX "${EXPECTED_PHASE1_TRACE_BATCH_SIZE_MAX:-}" "${PREPARED_EXPECTATIONS[7]}")"
EXPECTED_PHASE1_EFFECTIVE_TRACE_BATCH_SIZE="$(reconcile_expectation EXPECTED_PHASE1_EFFECTIVE_TRACE_BATCH_SIZE "${EXPECTED_PHASE1_EFFECTIVE_TRACE_BATCH_SIZE:-}" "${PREPARED_EXPECTATIONS[8]}")"
EXPECTED_PHASE3_BATCH_SIZE="$(reconcile_expectation EXPECTED_PHASE3_BATCH_SIZE "${EXPECTED_PHASE3_BATCH_SIZE:-}" "${PREPARED_EXPECTATIONS[9]}")"
EXPECTED_PHASE4_BATCH_SIZE="$(reconcile_expectation EXPECTED_PHASE4_BATCH_SIZE "${EXPECTED_PHASE4_BATCH_SIZE:-}" "${PREPARED_EXPECTATIONS[10]}")"
REQUIRE_FULL_COMPLETION="$(reconcile_expectation REQUIRE_FULL_COMPLETION "${REQUIRE_FULL_COMPLETION:-}" "${PREPARED_EXPECTATIONS[11]}")"
EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY="$(reconcile_expectation EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY "${EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY:-}" "${PREPARED_EXPECTATIONS[12]}")"
EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY_REQUIREMENT="$(reconcile_expectation EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY_REQUIREMENT "${EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY_REQUIREMENT:-}" "${PREPARED_EXPECTATIONS[13]}")"
EXPECTED_DECODER_ACTIVE_ROW_MAX_BYTES="$(reconcile_expectation EXPECTED_DECODER_ACTIVE_ROW_MAX_BYTES "${EXPECTED_DECODER_ACTIVE_ROW_MAX_BYTES:-}" "${PREPARED_EXPECTATIONS[14]}")"
EXPECTED_DECODER_ACTIVE_ROW_SAFETY_MARGIN_BYTES="$(reconcile_expectation EXPECTED_DECODER_ACTIVE_ROW_SAFETY_MARGIN_BYTES "${EXPECTED_DECODER_ACTIVE_ROW_SAFETY_MARGIN_BYTES:-}" "${PREPARED_EXPECTATIONS[15]}")"

for input_path in \
    "$PREHEAT_MODEL_ROOT" \
    "$PREHEAT_PROVIDER_ROOT"; do
    if [[ ! -e "$input_path" ]]; then
        echo "Required input does not exist: $input_path" >&2
        exit 2
    fi
done
if [[ -e "$TRACE_OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite trace output: $TRACE_OUTPUT_ROOT" >&2
    exit 2
fi
if [[ -e "$PROBE_OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite probe output: $PROBE_OUTPUT_ROOT" >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "Prepared workload input: ${PREPARED_WORKLOAD_INPUT}"
echo "Prepared workload file: ${PREPARED_WORKLOAD_PATH}"
echo "Trace output: ${TRACE_OUTPUT_ROOT}"
echo "Probe output: ${PROBE_OUTPUT_ROOT}"
echo "Expected mode: ${EXPECTED_MODE}"
echo "Expected backward engine: ${EXPECTED_BACKWARD_ENGINE_MODE}"
echo "Expected forward graph: ${EXPECTED_FORWARD_GRAPH_MODE}"
echo "Expected VJP kernel: ${EXPECTED_VJP_KERNEL_MODE}"
echo "Expected forward lanes: ${EXPECTED_FORWARD_LANE_COUNT}"
echo "Expected session capacity: ${EXPECTED_SESSION_CAPACITY}"
echo "Expected backward capacity: ${EXPECTED_BACKWARD_BATCH_CAPACITY}"
echo "Expected Phase-1 configured maximum: ${EXPECTED_PHASE1_TRACE_BATCH_SIZE_MAX}"
echo "Expected Phase-1 actual trace width: ${EXPECTED_PHASE1_EFFECTIVE_TRACE_BATCH_SIZE}"
echo "Expected Phase-3 batch maximum: ${EXPECTED_PHASE3_BATCH_SIZE}"
echo "Expected Phase-4 batch maximum: ${EXPECTED_PHASE4_BATCH_SIZE}"
echo "Require full completion: ${REQUIRE_FULL_COMPLETION}"
echo "Expected decoder active-row residency: ${EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY}"
echo "Expected decoder active-row requirement: ${EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY_REQUIREMENT}"
echo "Expected decoder active-row ceiling bytes: ${EXPECTED_DECODER_ACTIVE_ROW_MAX_BYTES}"
echo "Expected decoder active-row safety margin bytes: ${EXPECTED_DECODER_ACTIVE_ROW_SAFETY_MARGIN_BYTES}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF cache root: ${HF_HUB_CACHE}"
echo "HF offline: ${HF_HUB_OFFLINE}"

"$UV_PROJECT_ENVIRONMENT/bin/exact-trace-perf" \
    run-prepared-campaign-workload "$PREPARED_WORKLOAD_PATH" --validate-only
echo "Prepared workload preflight: passed"

"$UV_PROJECT_ENVIRONMENT/bin/python" \
    "$WORKSPACE_ROOT/slurm/exact_trace_bench/validate_hf_cache.py" \
    --prepared-workload "$PREPARED_WORKLOAD_PATH" \
    --expected-model-root "$PREHEAT_MODEL_ROOT"
echo "Offline model cache preflight: passed"

mkdir -p "$PROBE_OUTPUT_ROOT"
nvidia-smi -L

"$UV_PROJECT_ENVIRONMENT/bin/python" -m \
    nlp_research_project.exact_trace_bench.full_answer.runtime_environment \
    --output "$PROBE_OUTPUT_ROOT/runtime_environment.json"

"$UV_PROJECT_ENVIRONMENT/bin/python" - "$MIN_CUDA_TOTAL_GIB" "$PROBE_OUTPUT_ROOT/cuda_device.json" <<'PY'
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
preheat_args=(--manifest-output "$PROBE_OUTPUT_ROOT/preheat_manifest.json")
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
    run-prepared-campaign-workload "$PREPARED_WORKLOAD_PATH"

validation_args=(
    --label "prepared-run" \
    --output-root "$TRACE_OUTPUT_ROOT" \
    --expected-mode "$EXPECTED_MODE"
    --expected-backward-engine-mode "$EXPECTED_BACKWARD_ENGINE_MODE"
    --expected-forward-graph-mode "$EXPECTED_FORWARD_GRAPH_MODE"
    --expected-vjp-kernel-mode "$EXPECTED_VJP_KERNEL_MODE"
    --expected-forward-lane-count "$EXPECTED_FORWARD_LANE_COUNT"
    --expected-session-capacity "$EXPECTED_SESSION_CAPACITY"
    --expected-backward-batch-capacity "$EXPECTED_BACKWARD_BATCH_CAPACITY"
    --expected-phase1-trace-batch-size-max "$EXPECTED_PHASE1_TRACE_BATCH_SIZE_MAX"
    --expected-phase1-effective-trace-batch-size "$EXPECTED_PHASE1_EFFECTIVE_TRACE_BATCH_SIZE"
    --expected-phase3-batch-size "$EXPECTED_PHASE3_BATCH_SIZE"
    --expected-phase4-batch-size "$EXPECTED_PHASE4_BATCH_SIZE"
    --expected-decoder-active-row-residency "$EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY"
    --expected-decoder-active-row-residency-requirement "$EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY_REQUIREMENT"
    --expected-decoder-active-row-max-bytes "$EXPECTED_DECODER_ACTIVE_ROW_MAX_BYTES"
    --expected-decoder-active-row-safety-margin-bytes "$EXPECTED_DECODER_ACTIVE_ROW_SAFETY_MARGIN_BYTES"
)
REQUIRE_PHASE4_DEVICE_TIMING="${REQUIRE_PHASE4_DEVICE_TIMING:-0}"
REQUIRE_UNAMBIGUOUS_PER_DEVICE_GPU_EVIDENCE="${REQUIRE_UNAMBIGUOUS_PER_DEVICE_GPU_EVIDENCE:-0}"
REQUIRE_TYPED_ONLY_GRAPH="${REQUIRE_TYPED_ONLY_GRAPH:-0}"

if [[ "$REQUIRE_PHASE4_DEVICE_TIMING" == "1" ]]; then
    validation_args+=(--require-phase4-device-timing)
    if [[ -n "${EXPECTED_PHASE4_DEVICE_TIMING_BACKEND:-}" ]]; then
        validation_args+=(
            --expected-phase4-device-timing-backend
            "$EXPECTED_PHASE4_DEVICE_TIMING_BACKEND"
        )
    fi
elif [[ "$REQUIRE_PHASE4_DEVICE_TIMING" != "0" ]]; then
    echo "REQUIRE_PHASE4_DEVICE_TIMING must be 0 or 1: $REQUIRE_PHASE4_DEVICE_TIMING" >&2
    exit 2
fi
if [[ "$REQUIRE_UNAMBIGUOUS_PER_DEVICE_GPU_EVIDENCE" == "1" ]]; then
    validation_args+=(--require-unambiguous-per-device-gpu-evidence)
elif [[ "$REQUIRE_UNAMBIGUOUS_PER_DEVICE_GPU_EVIDENCE" != "0" ]]; then
    echo "REQUIRE_UNAMBIGUOUS_PER_DEVICE_GPU_EVIDENCE must be 0 or 1: $REQUIRE_UNAMBIGUOUS_PER_DEVICE_GPU_EVIDENCE" >&2
    exit 2
fi
if [[ -n "${EXPECTED_GRAPH_FEATURE_COUNT:-}" ]]; then
    validation_args+=(--expected-graph-feature-count "$EXPECTED_GRAPH_FEATURE_COUNT")
fi
if [[ "$REQUIRE_TYPED_ONLY_GRAPH" == "1" ]]; then
    validation_args+=(--require-typed-only-graph)
elif [[ "$REQUIRE_TYPED_ONLY_GRAPH" != "0" ]]; then
    echo "REQUIRE_TYPED_ONLY_GRAPH must be 0 or 1: $REQUIRE_TYPED_ONLY_GRAPH" >&2
    exit 2
fi
if [[ -n "${EXPECTED_TYPED_SCHEMA_VERSION:-}" ]]; then
    validation_args+=(--expected-graph-schema-version "$EXPECTED_TYPED_SCHEMA_VERSION")
fi
if [[ -n "${EXPECTED_TYPED_SAVE_FORMAT:-}" ]]; then
    validation_args+=(--expected-graph-save-format "$EXPECTED_TYPED_SAVE_FORMAT")
fi
if [[ -n "${EXPECTED_TYPED_RETENTION_POLICY_ID:-}" ]]; then
    validation_args+=(--expected-graph-retention-policy-id "$EXPECTED_TYPED_RETENTION_POLICY_ID")
fi
if [[ -n "${EXPECTED_TYPED_RETENTION_POLICY_FINGERPRINT:-}" ]]; then
    validation_args+=(
        --expected-graph-retention-policy-fingerprint
        "$EXPECTED_TYPED_RETENTION_POLICY_FINGERPRINT"
    )
fi
if [[ "$EXPECTED_VJP_KERNEL_MODE" == "autograd_batched" ]]; then
    validation_args+=(--require-structured-vjp-evidence)
fi
if [[ "$REQUIRE_FULL_COMPLETION" == "1" ]]; then
    validation_args+=(--require-full-completion)
elif [[ "$REQUIRE_FULL_COMPLETION" != "0" ]]; then
    echo "REQUIRE_FULL_COMPLETION must be 0 or 1: $REQUIRE_FULL_COMPLETION" >&2
    exit 2
fi
"$UV_PROJECT_ENVIRONMENT/bin/python" \
    "$WORKSPACE_ROOT/slurm/exact_trace_bench/validate_full_answer_feature_row_mode.py" \
    "${validation_args[@]}" \
    > "$PROBE_OUTPUT_ROOT/mechanism_validation.json"

echo "End: $(date --iso-8601=seconds)"
