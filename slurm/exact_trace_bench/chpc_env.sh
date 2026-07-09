#!/bin/bash

# Shared CHPC cache defaults for Granite jobs. Keep large caches off home and
# make imports deterministic when home config directories are not writable.
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-/scratch/general/vast/${USER}/nlp_research_project}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_CACHE_ROOT}/uv-cache}"
export HF_HOME="${HF_HOME:-${PROJECT_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_CACHE_ROOT}/matplotlib}"
export TMPDIR="${TMPDIR:-${PROJECT_CACHE_ROOT}/tmp}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR" "$TMPDIR"
