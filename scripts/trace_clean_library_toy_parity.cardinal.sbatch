#!/bin/bash
#SBATCH --job-name=clean-toy-parity
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
CLEAN_LIB_ROOT="${CLEAN_LIB_ROOT:-/users/PAS2119/andreykopanev/worktrees_opt/upstream/circuit-tracer}"
PROMPT_FILE="${PROMPT_FILE:-experiments/generated/toy_clean_library_parity/dax_wug_prompt.txt}"
PROMPT_META_FILE="${PROMPT_META_FILE:-experiments/generated/toy_clean_library_parity/dax_wug_prompt_meta.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/cardinal/fast/clean-library-toy-parity}"

cd "$WORKSPACE_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$WORKSPACE_ROOT/.venv}"
export PYTHONPATH="$CLEAN_LIB_ROOT:$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Cluster: cardinal"
echo "Start: $(date)"
echo "Working directory: ${WORKSPACE_ROOT}"
echo "Clean library root: ${CLEAN_LIB_ROOT}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "UV project env: ${UV_PROJECT_ENVIRONMENT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python -u scripts/run_clean_library_toy_parity.py \
    --clean-library-root "$CLEAN_LIB_ROOT" \
    --prompt-file "$PROMPT_FILE" \
    --prompt-meta-file "$PROMPT_META_FILE" \
    --output-dir "$OUTPUT_ROOT/artifacts" \
    --max-feature-nodes 8192 \
    --max-edges 20000 \
    --attribution-batch-size 128 \
    --max-n-logits 3 \
    --desired-logit-prob 0.8 \
    --temperature 0.0 \
    --verbose-attribution

echo "End: $(date)"
