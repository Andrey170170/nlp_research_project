from __future__ import annotations

import argparse
from pathlib import Path

from nlp_research_project.exact_trace_bench.config import DEFAULT_GENERATED_DIR
from nlp_research_project.exact_trace_bench.scenarios.phase4_static_coalescing import (
    build_phase4_static_coalescing_config,
    write_phase4_static_coalescing_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Granite H200 static Phase-4 coalescing validation."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--model-cache-root", type=Path, default=None)
    parser.add_argument("--baseline-registry", type=Path, default=None)
    args = parser.parse_args()
    kwargs = {
        key: value
        for key, value in {
            "scratch_root": args.scratch_root,
            "model_cache_root": args.model_cache_root,
            "baseline_registry": args.baseline_registry,
        }.items()
        if value is not None
    }
    payload = build_phase4_static_coalescing_config(**kwargs)
    print(write_phase4_static_coalescing_config(payload, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
