"""Compare cached vs uncached circuit-tracer runs produced by
``prefix_caching.trace_pipeline_cached``.

Reads ``step_NNN.npz`` files from a baseline directory and a cached
directory, pairs them by step index, and reports:

- **Feature Jaccard** per step: |A ∩ B| / |A ∪ B| over the
  ``feature_ids`` rows ``(layer, position, feature_idx)``.  With the
  library's PrefixActivationCache in place, we expect this to be close
  to 1.0 because the pre-sparsification activations are byte-identical
  and sparsification is re-run each step.
- **Edge set Jaccard** per step: the set of retained (row, col)
  positions after pruning, which is a stricter version of graph equality
  than feature Jaccard.
- **Edge-weight max |Δ|**: for edges that exist in both graphs, the
  largest absolute weight difference.
- **Attribution runtime delta** (per step) from ``cache_validation.json``.

Runs on a login node (CPU only).  No GPU required.

Usage::

    uv run python -m prefix_caching.compare_cached_vs_uncached \\
        --baseline /fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/baseline_exact \\
        --cached   /fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/forward_cache \\
        --prompt   prompt_000 \\
        --completion completion_000 \\
        --output /fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/analysis/fidelity_report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from circuit_utils import load_compact


@dataclass
class StepComparison:
    step_index: int
    baseline_feature_count: int
    cached_feature_count: int
    feature_jaccard: float
    baseline_edge_count: int
    cached_edge_count: int
    edge_set_jaccard: float
    edge_weight_max_abs_delta: float | None
    baseline_attribution_seconds: float | None
    cached_attribution_seconds: float | None
    speedup_vs_baseline: float | None


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _feature_rows_as_tuples(feature_ids: np.ndarray) -> set[tuple[int, int, int]]:
    return {
        (int(row[0]), int(row[1]), int(row[2]))
        for row in feature_ids
    }


def _edge_dict(
    row_idx: np.ndarray, col_idx: np.ndarray, weights: np.ndarray
) -> dict[tuple[int, int], float]:
    return {
        (int(r), int(c)): float(w)
        for r, c, w in zip(row_idx, col_idx, weights)
    }


def compare_step(baseline_path: Path, cached_path: Path) -> StepComparison:
    b = load_compact(baseline_path)
    c = load_compact(cached_path)
    assert b.step_idx == c.step_idx, (
        f"step index mismatch: baseline={b.step_idx} cached={c.step_idx}"
    )

    b_features = _feature_rows_as_tuples(b.feature_ids)
    c_features = _feature_rows_as_tuples(c.feature_ids)

    b_edges = _edge_dict(b.row_idx, b.col_idx, b.weights)
    c_edges = _edge_dict(c.row_idx, c.col_idx, c.weights)
    shared = set(b_edges) & set(c_edges)

    max_abs_delta: float | None
    if shared:
        deltas = [abs(b_edges[k] - c_edges[k]) for k in shared]
        max_abs_delta = float(max(deltas))
    else:
        max_abs_delta = None

    return StepComparison(
        step_index=b.step_idx,
        baseline_feature_count=len(b_features),
        cached_feature_count=len(c_features),
        feature_jaccard=_jaccard(b_features, c_features),
        baseline_edge_count=len(b_edges),
        cached_edge_count=len(c_edges),
        edge_set_jaccard=_jaccard(set(b_edges), set(c_edges)),
        edge_weight_max_abs_delta=max_abs_delta,
        baseline_attribution_seconds=None,
        cached_attribution_seconds=None,
        speedup_vs_baseline=None,
    )


def _attribution_seconds_by_step(
    cache_validation_path: Path,
) -> dict[int, float]:
    if not cache_validation_path.exists():
        return {}
    data = json.loads(cache_validation_path.read_text())
    result: dict[int, float] = {}
    for step in data.get("steps", []):
        idx = step.get("step_index")
        seconds = step.get("attribution_seconds")
        if idx is not None and seconds is not None:
            result[int(idx)] = float(seconds)
    return result


def _list_step_files(completion_dir: Path) -> list[Path]:
    return sorted(completion_dir.glob("step_*.npz"))


def compare_directories(
    baseline_root: Path,
    cached_root: Path,
    prompt_id: str,
    completion_id: str,
) -> dict[str, Any]:
    b_dir = baseline_root / prompt_id / completion_id
    c_dir = cached_root / prompt_id / completion_id

    if not b_dir.is_dir():
        raise FileNotFoundError(f"baseline completion dir missing: {b_dir}")
    if not c_dir.is_dir():
        raise FileNotFoundError(f"cached completion dir missing: {c_dir}")

    b_steps = {int(p.stem.split("_")[1]): p for p in _list_step_files(b_dir)}
    c_steps = {int(p.stem.split("_")[1]): p for p in _list_step_files(c_dir)}
    shared_indices = sorted(set(b_steps) & set(c_steps))
    missing_in_cached = sorted(set(b_steps) - set(c_steps))
    missing_in_baseline = sorted(set(c_steps) - set(b_steps))

    b_timing = _attribution_seconds_by_step(b_dir / "cache_validation.json")
    c_timing = _attribution_seconds_by_step(c_dir / "cache_validation.json")

    rows: list[StepComparison] = []
    for idx in shared_indices:
        row = compare_step(b_steps[idx], c_steps[idx])
        row.baseline_attribution_seconds = b_timing.get(idx)
        row.cached_attribution_seconds = c_timing.get(idx)
        if (
            row.baseline_attribution_seconds is not None
            and row.cached_attribution_seconds
            and row.cached_attribution_seconds > 0
        ):
            row.speedup_vs_baseline = (
                row.baseline_attribution_seconds
                / row.cached_attribution_seconds
            )
        rows.append(row)

    summary = {
        "prompt_id": prompt_id,
        "completion_id": completion_id,
        "baseline_root": str(baseline_root),
        "cached_root": str(cached_root),
        "shared_step_count": len(shared_indices),
        "missing_in_cached": missing_in_cached,
        "missing_in_baseline": missing_in_baseline,
        "mean_feature_jaccard": (
            float(np.mean([r.feature_jaccard for r in rows])) if rows else None
        ),
        "min_feature_jaccard": (
            float(np.min([r.feature_jaccard for r in rows])) if rows else None
        ),
        "mean_edge_set_jaccard": (
            float(np.mean([r.edge_set_jaccard for r in rows])) if rows else None
        ),
        "max_edge_weight_abs_delta": (
            float(
                np.nanmax(
                    [
                        r.edge_weight_max_abs_delta
                        for r in rows
                        if r.edge_weight_max_abs_delta is not None
                    ]
                )
            )
            if any(r.edge_weight_max_abs_delta is not None for r in rows)
            else None
        ),
        "mean_speedup_vs_baseline": (
            float(
                np.mean(
                    [
                        r.speedup_vs_baseline
                        for r in rows
                        if r.speedup_vs_baseline is not None
                    ]
                )
            )
            if any(r.speedup_vs_baseline is not None for r in rows)
            else None
        ),
        "steps": [asdict(r) for r in rows],
    }
    return summary


def _print_human(summary: dict[str, Any]) -> None:
    print(
        f"\nComparison: {summary['prompt_id']}/{summary['completion_id']}"
    )
    print(f"  Shared step count: {summary['shared_step_count']}")
    if summary["missing_in_cached"]:
        print(f"  Missing in cached: {summary['missing_in_cached']}")
    if summary["missing_in_baseline"]:
        print(f"  Missing in baseline: {summary['missing_in_baseline']}")
    mfj = summary["mean_feature_jaccard"]
    mej = summary["mean_edge_set_jaccard"]
    mfjmin = summary["min_feature_jaccard"]
    print(
        f"  Feature Jaccard    mean={mfj:.6f} "
        f"min={mfjmin:.6f}"
        if mfj is not None
        else "  Feature Jaccard unavailable (no shared steps)"
    )
    if mej is not None:
        print(f"  Edge-set Jaccard   mean={mej:.6f}")
    if summary["max_edge_weight_abs_delta"] is not None:
        print(
            f"  Edge-weight |Δ|    max={summary['max_edge_weight_abs_delta']:.6e}"
        )
    if summary["mean_speedup_vs_baseline"] is not None:
        print(
            f"  Attribution speedup (mean): "
            f"{summary['mean_speedup_vs_baseline']:.3f}×"
        )
    print()
    print("  Per-step breakdown:")
    print(
        f"    {'step':>5}  "
        f"{'feat_J':>9}  {'edge_J':>9}  "
        f"{'n_feat_b':>9}  {'n_feat_c':>9}  "
        f"{'sec_b':>8}  {'sec_c':>8}  {'spd':>6}"
    )
    for row in summary["steps"]:
        sec_b = row["baseline_attribution_seconds"]
        sec_c = row["cached_attribution_seconds"]
        spd = row["speedup_vs_baseline"]
        print(
            f"    {row['step_index']:>5}  "
            f"{row['feature_jaccard']:>9.6f}  "
            f"{row['edge_set_jaccard']:>9.6f}  "
            f"{row['baseline_feature_count']:>9d}  "
            f"{row['cached_feature_count']:>9d}  "
            f"{(f'{sec_b:>8.1f}' if sec_b is not None else '     n/a'):>8}  "
            f"{(f'{sec_c:>8.1f}' if sec_c is not None else '     n/a'):>8}  "
            f"{(f'{spd:>6.2f}x' if spd is not None else '   n/a'):>6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cached vs uncached circuit-tracer outputs"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Root of uncached baseline output (e.g. .../baseline_exact)",
    )
    parser.add_argument(
        "--cached",
        required=True,
        help="Root of cached output (e.g. .../forward_cache)",
    )
    parser.add_argument(
        "--prompt",
        default="prompt_000",
        help="Prompt directory name (default: prompt_000)",
    )
    parser.add_argument(
        "--completion",
        default="completion_000",
        help="Completion directory name (default: completion_000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the JSON summary",
    )
    args = parser.parse_args()

    summary = compare_directories(
        baseline_root=Path(args.baseline),
        cached_root=Path(args.cached),
        prompt_id=args.prompt,
        completion_id=args.completion,
    )

    _print_human(summary)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary to {out}")


if __name__ == "__main__":
    main()
