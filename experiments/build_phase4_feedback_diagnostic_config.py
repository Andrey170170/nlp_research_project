from __future__ import annotations

import argparse
from pathlib import Path

from nlp_research_project.exact_trace_bench.config import DEFAULT_GENERATED_DIR
from nlp_research_project.exact_trace_bench.scenarios.phase4_feedback_diagnostic import (
    PHASE4_FEEDBACK_DIAGNOSTIC_BASELINE_REGISTRY,
    build_phase4_feedback_diagnostic_config,
    write_phase4_feedback_diagnostic_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the bounded Granite H200 Phase-4 feedback diagnostic."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--model-cache-root", type=Path, default=None)
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=PHASE4_FEEDBACK_DIAGNOSTIC_BASELINE_REGISTRY,
    )
    args = parser.parse_args()
    kwargs = {
        "model_cache_root": args.model_cache_root,
        "baseline_registry": args.baseline_registry,
    }
    if args.scratch_root is not None:
        kwargs["scratch_root"] = args.scratch_root
    payload = build_phase4_feedback_diagnostic_config(**kwargs)
    print(
        write_phase4_feedback_diagnostic_config(
            payload,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
