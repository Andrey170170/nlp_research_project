#!/bin/bash
set -euo pipefail

uv run python -m experiments.exact_trace_bench submit-preset --preset full-all "$@"
