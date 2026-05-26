#!/bin/bash
set -euo pipefail

uv run exact-trace-bench submit-preset --preset fast-all "$@"
