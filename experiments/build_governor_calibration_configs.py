from __future__ import annotations

import argparse
from pathlib import Path

from nlp_research_project.exact_trace_bench.config import DEFAULT_GENERATED_DIR
from nlp_research_project.exact_trace_bench.scenarios.governor_calibration import (
    GOVERNOR_CALIBRATION_BASELINE_REGISTRY,
    write_all_governor_calibration_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the staged Granite H200 governor calibration matrix."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--model-cache-root", type=Path, default=None)
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=GOVERNOR_CALIBRATION_BASELINE_REGISTRY,
    )
    args = parser.parse_args()
    kwargs = {
        "output_dir": args.output_dir,
        "model_cache_root": args.model_cache_root,
        "baseline_registry": args.baseline_registry,
    }
    if args.scratch_root is not None:
        kwargs["scratch_root"] = args.scratch_root
    for path in write_all_governor_calibration_configs(**kwargs):
        print(path)


if __name__ == "__main__":
    main()
