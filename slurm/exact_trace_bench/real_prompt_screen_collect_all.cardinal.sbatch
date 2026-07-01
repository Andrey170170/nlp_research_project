#!/bin/bash
#SBATCH --job-name=real-prompt-screen
#SBATCH --account=PAS2836
#SBATCH --cluster=cardinal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$SLURM_SUBMIT_DIR}"
LIB_WORKSPACE_ROOT="${LIB_WORKSPACE_ROOT:-$WORKSPACE_ROOT/../circuit-tracer_chunked}"

TARGET_SPEC_FILE="${TARGET_SPEC_FILE:?TARGET_SPEC_FILE is required}"
FIXTURE_CATALOG="${FIXTURE_CATALOG:?FIXTURE_CATALOG is required}"
SAMPLE_ROOT="${SAMPLE_ROOT:?SAMPLE_ROOT is required}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
TEMPERATURE="${TEMPERATURE:-0.8}"
BASE_SEED="${BASE_SEED:-3000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-16}"
MAX_SUCCESS_TOKENS="${MAX_SUCCESS_TOKENS:-768}"
COLLECT_ALL="${COLLECT_ALL:-1}"
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-}"

cd "$WORKSPACE_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SLURM_SUBMIT_DIR/.venv}"
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

eval "$(python - "$TARGET_SPEC_FILE" "$FIXTURE_CATALOG" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import re
import shlex
import sys
from pathlib import Path

target_spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
catalog = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
task_id = int(sys.argv[3])
indices = target_spec["gsm8k_indices"]
gsm8k_index = int(indices[task_id])
fixture_name = f"{gsm8k_index}_base"
fixtures = [
    row
    for row in catalog.get("fixtures", [])
    if row.get("fixture_name") == fixture_name and row.get("fixture_kind") == "base"
]
if len(fixtures) != 1:
    raise SystemExit(f"expected one base fixture {fixture_name!r}, found {len(fixtures)}")
ground_truth = str(fixtures[0].get("ground_truth_answer", ""))
match = re.search(r"####\s*([^\n]+)", ground_truth)
if not match:
    raise SystemExit(f"could not parse final answer for {fixture_name}")
expected_answer = match.group(1).replace(",", "").strip()
for key, value in {
    "GSM8K_INDEX": str(gsm8k_index),
    "FIXTURE_NAME": fixture_name,
    "EXPECTED_ANSWER": expected_answer,
}.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

OUTPUT_DIR="$SAMPLE_ROOT/$FIXTURE_NAME"
mkdir -p "$OUTPUT_DIR"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Cluster: cardinal"
echo "Start: $(date)"
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Library workspace root: ${LIB_WORKSPACE_ROOT}"
echo "Target spec: ${TARGET_SPEC_FILE}"
echo "Fixture catalog: ${FIXTURE_CATALOG}"
echo "Fixture: ${FIXTURE_NAME}"
echo "GSM8K index: ${GSM8K_INDEX}"
echo "Expected answer: ${EXPECTED_ANSWER}"
echo "Output dir: ${OUTPUT_DIR}"
echo "UV project env: ${UV_PROJECT_ENVIRONMENT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ARGS=(
  --fixture-catalog "$FIXTURE_CATALOG"
  --fixture-name "$FIXTURE_NAME"
  --output-dir "$OUTPUT_DIR"
  --trajectory-id-prefix "$FIXTURE_NAME"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --base-seed "$BASE_SEED"
  --expected-answer "$EXPECTED_ANSWER"
  --max-success-tokens "$MAX_SUCCESS_TOKENS"
  --max-attempts "$MAX_ATTEMPTS"
)
if [[ "$COLLECT_ALL" == "1" ]]; then ARGS+=(--collect-all); fi
if [[ -n "$TIME_LIMIT_SECONDS" ]]; then ARGS+=(--time-limit-seconds "$TIME_LIMIT_SECONDS"); fi

uv run --no-sync exact-trace-bench sample-full-answer-trajectories "${ARGS[@]}"

echo "End: $(date)"
