from __future__ import annotations

import argparse
from pathlib import Path

from nlp_research_project.exact_trace_bench.phase4_static_validation import (
    validate_phase4_static_coalescing_run,
    write_phase4_static_validation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate a completed static Phase-4 coalescing matrix."
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--weighted-edge-jaccard-min", type=float, default=0.999999)
    args = parser.parse_args()

    report = validate_phase4_static_coalescing_run(
        args.run_root,
        weighted_edge_jaccard_min=args.weighted_edge_jaccard_min,
    )
    output_path = write_phase4_static_validation_report(args.run_root, report)
    print(output_path)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
