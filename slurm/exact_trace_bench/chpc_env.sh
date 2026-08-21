#!/bin/bash

# Shared CHPC cache defaults for Granite jobs. Keep large caches off home and
# make imports deterministic when home config directories are not writable.
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-/scratch/general/vast/${USER}/nlp_research_project}"
EXACT_TRACE_HF_CACHE_ROOT="${EXACT_TRACE_HF_CACHE_ROOT:-${PROJECT_CACHE_ROOT}/huggingface}"
EXACT_TRACE_HF_OFFLINE="${EXACT_TRACE_HF_OFFLINE:-1}"

if [[ "$EXACT_TRACE_HF_CACHE_ROOT" != /* ]]; then
    echo "EXACT_TRACE_HF_CACHE_ROOT must be an absolute path: $EXACT_TRACE_HF_CACHE_ROOT" >&2
    return 2 2>/dev/null || exit 2
fi
if [[ "$EXACT_TRACE_HF_OFFLINE" != "0" && "$EXACT_TRACE_HF_OFFLINE" != "1" ]]; then
    echo "EXACT_TRACE_HF_OFFLINE must be 0 or 1: $EXACT_TRACE_HF_OFFLINE" >&2
    return 2 2>/dev/null || exit 2
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_CACHE_ROOT}/uv-cache}"
# Do not inherit these independently through `sbatch --export=ALL`: one stale
# value is enough to make a gated model appear absent and trigger a network
# lookup. EXACT_TRACE_HF_CACHE_ROOT is the only supported cache-path override.
export EXACT_TRACE_HF_CACHE_ROOT
export HF_HOME="$EXACT_TRACE_HF_CACHE_ROOT"
export HF_HUB_CACHE="$EXACT_TRACE_HF_CACHE_ROOT"
export TRANSFORMERS_CACHE="$EXACT_TRACE_HF_CACHE_ROOT"
export EXACT_TRACE_HF_OFFLINE
export HF_HUB_OFFLINE="$EXACT_TRACE_HF_OFFLINE"
export TRANSFORMERS_OFFLINE="$EXACT_TRACE_HF_OFFLINE"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_CACHE_ROOT}/matplotlib}"
export TMPDIR="${TMPDIR:-${PROJECT_CACHE_ROOT}/tmp}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR" "$TMPDIR"
