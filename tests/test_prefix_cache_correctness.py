"""Offline correctness check for prefix cache validation results.

Reads ``cache_validation.json`` produced by
``prefix_caching.trace_pipeline_cached`` and asserts that prefix features
matched at every step.  Runs on a login node — no GPU required.

Usage::

    uv run python tests/test_prefix_cache_correctness.py \\
        --results-dir /fs/scratch/PAS3272/kopanev.1/prefix_cache_bench/prompt_000/completion_000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_cache_validation(results_dir: Path) -> bool:
    cache_path = results_dir / "cache_validation.json"
    if not cache_path.exists():
        print(f"ERROR: {cache_path} not found")
        return False

    data = json.loads(cache_path.read_text())
    steps = data["steps"]
    n_steps = len(steps)

    print(f"Checking {n_steps} steps from {cache_path}")
    print(f"  Temperature: {data['temperature']}")
    print()

    all_passed = True
    total_cached_features = 0
    total_matched_features = 0

    for step in steps:
        idx = step["step_index"]
        comparison = step.get("comparison")

        if comparison is None:
            # First step has no cache to compare against.
            print(f"  Step {idx:03d}: first step (no comparison) — "
                  f"features={step['total_active_features']}")
            continue

        pos_rate = comparison["position_match_rate"]
        feat_rate = comparison["feature_match_rate"]
        cached_pos = comparison["cached_positions"]
        matched_pos = comparison["matched_positions"]
        cached_feat = comparison["cached_feature_count"]
        matched_feat = comparison["matched_feature_count"]

        total_cached_features += cached_feat
        total_matched_features += matched_feat

        passed = pos_rate == 1.0

        status = "PASS" if passed else "FAIL"
        print(
            f"  Step {idx:03d}: {status} — "
            f"positions {matched_pos}/{cached_pos} "
            f"features {matched_feat}/{cached_feat} "
            f"({feat_rate:.4f})"
        )

        if not passed:
            all_passed = False
            mismatched = cached_pos - matched_pos
            print(f"           {mismatched} positions had different features!")

    print()

    if total_cached_features > 0:
        overall_rate = total_matched_features / total_cached_features
        print(f"Overall feature match rate: {overall_rate:.6f} "
              f"({total_matched_features}/{total_cached_features})")
        redundancy = total_cached_features / (
            total_cached_features + sum(
                s["total_active_features"] - s.get("comparison", {}).get("cached_feature_count", 0)
                for s in steps if s.get("comparison") is not None
            )
        )
        print(f"Estimated redundant work per step: {redundancy:.1%}")

    print()
    if all_passed:
        print("RESULT: ALL STEPS PASSED — prefix features are invariant across steps")
    else:
        print("RESULT: SOME STEPS FAILED — prefix features changed unexpectedly")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check prefix cache validation results"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Path to completion directory containing cache_validation.json",
    )
    args = parser.parse_args()

    passed = check_cache_validation(Path(args.results_dir))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
