"""Reconstruct a ``cache_validation.json`` for the cached run from surviving
artifacts when the completion JSON was not flushed (e.g. SLURM TIMEOUT).

The driver ``trace_pipeline_cached.py`` normally writes
``completion.json`` / ``cache_validation.json`` at the end of each
completion.  If the process is killed mid-completion, those files do not
exist, but the following do survive on scratch / disk:

* ``step_NNN.npz`` files for every step that finished
* ``run_config.json`` at the output root
* the SLURM stdout log, which prints one line per step with the
  observational metrics

This script merges those surviving artifacts into a best-effort
``cache_validation_reconstructed.json`` that captures *everything we can
still derive*, with each metric explicitly labelled as:

* ``"from_log"``  – parsed out of stdout
* ``"from_npz"``  – computed from saved trace graphs
* ``"from_run_config"`` – copied from run_config.json
* ``"lost"``       – would have been in the flushed JSON, no longer
                     recoverable (e.g. the library's ``hit_count``
                     counter that lived in a Python object)

CPU-only, safe on the login node.

Usage::

    uv run python -m prefix_caching.reconstruct_diagnostics \\
        --baseline-log logs/slurm-prefix-cache-bench-jay-5011889.out \\
        --cached-log   logs/slurm-prefix-cache-run-jay-5022818.out \\
        --cached-root  /fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/forward_cache \\
        --fidelity     /fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/analysis/fidelity_report.json \\
        --output       /fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/analysis/cache_validation_reconstructed.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

STEP_PATTERN = re.compile(
    r"Step (?P<idx>\d+): "
    r"(?:first step \(no cache yet\) features=(?P<feat0>\d+) attrib=(?P<t0>[\d.]+)s"
    r"|cached_pos=(?P<cpos>\d+) matched_pos=(?P<mpos>\d+) "
    r"pos_rate=(?P<pr>[\d.]+) feat_rate=(?P<fr>[\d.]+) attrib=(?P<t>[\d.]+)s)"
)


def parse_step_lines(log_path: Path) -> list[dict[str, Any]]:
    """Extract per-step metrics from a driver stdout log."""
    steps: list[dict[str, Any]] = []
    for line in log_path.read_text().splitlines():
        m = STEP_PATTERN.search(line)
        if not m:
            continue
        idx = int(m.group("idx"))
        if m.group("feat0") is not None:
            steps.append(
                {
                    "step_index": idx,
                    "is_first_step": True,
                    "driver_feature_count": int(m.group("feat0")),
                    "driver_cached_positions": 0,
                    "driver_matched_positions": 0,
                    "driver_position_match_rate": None,
                    "driver_feature_match_rate": None,
                    "attribution_seconds": float(m.group("t0")),
                }
            )
        else:
            steps.append(
                {
                    "step_index": idx,
                    "is_first_step": False,
                    "driver_feature_count": None,
                    "driver_cached_positions": int(m.group("cpos")),
                    "driver_matched_positions": int(m.group("mpos")),
                    "driver_position_match_rate": float(m.group("pr")),
                    "driver_feature_match_rate": float(m.group("fr")),
                    "attribution_seconds": float(m.group("t")),
                }
            )
    return steps


def _step_by_index(steps: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    for s in steps:
        if s["step_index"] == idx:
            return s
    return None


def _extract_job_header(log_path: Path) -> dict[str, Any]:
    head: dict[str, str | None] = {
        "job_id": None,
        "node": None,
        "cluster": None,
        "start": None,
        "gpu": None,
        "vram": None,
    }
    patterns = {
        "job_id": re.compile(r"^Job ID: (.+)$"),
        "node": re.compile(r"^Node: (.+)$"),
        "cluster": re.compile(r"^Cluster: (.+)$"),
        "start": re.compile(r"^Start: (.+)$"),
        "gpu": re.compile(r"^GPU: (.+)$"),
        "vram": re.compile(r"^VRAM: (.+)$"),
    }
    for line in log_path.read_text().splitlines()[:25]:
        for key, pat in patterns.items():
            m = pat.match(line)
            if m and head[key] is None:
                head[key] = m.group(1)
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--cached-log", type=Path, required=True)
    parser.add_argument("--cached-root", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_steps = parse_step_lines(args.baseline_log)
    cached_steps = parse_step_lines(args.cached_log)
    baseline_header = _extract_job_header(args.baseline_log)
    cached_header = _extract_job_header(args.cached_log)

    run_config_path = args.cached_root / "run_config.json"
    run_config = json.loads(run_config_path.read_text())

    fidelity = json.loads(args.fidelity.read_text())
    fidelity_steps = {s["step_index"]: s for s in fidelity.get("steps", [])}

    merged_steps: list[dict[str, Any]] = []
    shared_indices = sorted(
        {s["step_index"] for s in baseline_steps}
        & {s["step_index"] for s in cached_steps}
    )
    for idx in shared_indices:
        b = _step_by_index(baseline_steps, idx)
        c = _step_by_index(cached_steps, idx)
        f = fidelity_steps.get(idx, {})

        b_sec = b["attribution_seconds"] if b else None
        c_sec = c["attribution_seconds"] if c else None
        overhead_sec = None
        if b_sec is not None and c_sec is not None:
            overhead_sec = round(c_sec - b_sec, 2)

        merged_steps.append(
            {
                "step_index": idx,
                "fidelity_from_npz": {
                    "feature_jaccard": f.get("feature_jaccard"),
                    "edge_set_jaccard": f.get("edge_set_jaccard"),
                    "edge_weight_max_abs_delta": f.get("edge_weight_max_abs_delta"),
                    "baseline_feature_count": f.get("baseline_feature_count"),
                    "cached_feature_count": f.get("cached_feature_count"),
                    "baseline_edge_count": f.get("baseline_edge_count"),
                    "cached_edge_count": f.get("cached_edge_count"),
                },
                "timing_from_log": {
                    "baseline_attribution_seconds": b_sec,
                    "cached_attribution_seconds": c_sec,
                    "cached_minus_baseline_seconds": overhead_sec,
                },
                "observational_from_log": {
                    "cached_positions": c["driver_cached_positions"] if c else None,
                    "matched_positions": c["driver_matched_positions"] if c else None,
                    "position_match_rate": c["driver_position_match_rate"] if c else None,
                    "feature_match_rate": c["driver_feature_match_rate"] if c else None,
                    "note": (
                        "These are driver-side observational metrics comparing "
                        "this step's active features to the previous step's. "
                        "They are NOT the library PrefixActivationCache internal "
                        "counters."
                    ),
                },
                "library_cache_counters": {
                    "status": "lost",
                    "hit_count": None,
                    "miss_count": None,
                    "cached_prefix_len": None,
                    "note": (
                        "Library counters live in a Python object and are only "
                        "flushed when completion.json writes at end of "
                        "completion. SLURM killed the process mid step-5 on "
                        "timeout, so these were never written. Can be recovered "
                        "in future runs by adding per-step stdout logging in "
                        "the library."
                    ),
                },
            }
        )

    def _mean(values: list[float]) -> float | None:
        clean = [v for v in values if v is not None]
        if not clean:
            return None
        return round(sum(clean) / len(clean), 4)

    summary = {
        "experiment": "prefix_cache_validation (reconstructed)",
        "reconstruction_note": (
            "completion.json / cache_validation.json were not flushed because "
            "SLURM killed the cached run on TIMEOUT mid step-5. This JSON was "
            "rebuilt from surviving artifacts: stdout logs, step_NNN.npz "
            "files, run_config.json, and the fidelity_report.json produced "
            "by compare_cached_vs_uncached. Fields are tagged with their "
            "source."
        ),
        "run_config_from_run_config_json": run_config,
        "baseline_job_from_log": baseline_header,
        "cached_job_from_log": cached_header,
        "shared_step_count": len(shared_indices),
        "aggregate": {
            "mean_feature_jaccard": fidelity.get("mean_feature_jaccard"),
            "min_feature_jaccard": fidelity.get("min_feature_jaccard"),
            "mean_edge_set_jaccard": fidelity.get("mean_edge_set_jaccard"),
            "max_edge_weight_abs_delta": fidelity.get("max_edge_weight_abs_delta"),
            "mean_baseline_attribution_seconds": _mean(
                [s["attribution_seconds"] for s in baseline_steps]
            ),
            "mean_cached_attribution_seconds": _mean(
                [s["attribution_seconds"] for s in cached_steps]
            ),
            "mean_cached_minus_baseline_seconds": _mean(
                [
                    s["timing_from_log"]["cached_minus_baseline_seconds"]
                    for s in merged_steps
                    if s["timing_from_log"]["cached_minus_baseline_seconds"] is not None
                ]
            ),
        },
        "steps": merged_steps,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print(f"Wrote reconstructed cache_validation to {args.output}")
    print()
    print(f"  Shared steps: {len(shared_indices)}")
    print(f"  Mean feature Jaccard: {summary['aggregate']['mean_feature_jaccard']}")
    print(f"  Max edge weight |Δ|:  {summary['aggregate']['max_edge_weight_abs_delta']}")
    print(
        f"  Mean baseline attribution: "
        f"{summary['aggregate']['mean_baseline_attribution_seconds']}s"
    )
    print(
        f"  Mean cached attribution:   "
        f"{summary['aggregate']['mean_cached_attribution_seconds']}s"
    )
    print(
        f"  Mean cached - baseline:    "
        f"{summary['aggregate']['mean_cached_minus_baseline_seconds']}s"
    )


if __name__ == "__main__":
    main()
