from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze import analyze_single_completion, load_completion_steps
from circuit_utils import StepData


def feature_jaccard(a: StepData, b: StepData) -> float:
    features_a = {tuple(row.tolist()) for row in a.feature_ids}
    features_b = {tuple(row.tolist()) for row in b.feature_ids}
    union = features_a | features_b
    if not union:
        return float("nan")
    return len(features_a & features_b) / len(union)


def edge_maps(step: StepData) -> dict[tuple[int, int], float]:
    return {
        (int(row), int(col)): float(abs(weight))
        for row, col, weight in zip(
            step.row_idx, step.col_idx, step.weights, strict=True
        )
    }


def edge_jaccard(a: StepData, b: StepData) -> float:
    edges_a = set(edge_maps(a))
    edges_b = set(edge_maps(b))
    union = edges_a | edges_b
    if not union:
        return float("nan")
    return len(edges_a & edges_b) / len(union)


def weighted_edge_jaccard(a: StepData, b: StepData) -> float:
    map_a = edge_maps(a)
    map_b = edge_maps(b)
    keys = set(map_a) | set(map_b)
    if not keys:
        return float("nan")
    num = sum(min(map_a.get(key, 0.0), map_b.get(key, 0.0)) for key in keys)
    den = sum(max(map_a.get(key, 0.0), map_b.get(key, 0.0)) for key in keys)
    return num / den if den > 0.0 else float("nan")


def edge_mass_ratio(approx: StepData, exact: StepData) -> float:
    exact_mass = float(np.abs(exact.weights).sum())
    if exact_mass == 0.0:
        return float("nan")
    return float(np.abs(approx.weights).sum()) / exact_mass


def safe_corr(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    if mask.sum() < 2:
        return None
    return float(np.corrcoef(arr_a[mask], arr_b[mask])[0, 1])


def load_scenario_dir(scenario_dir: Path) -> dict[str, Any] | None:
    scenario_path = scenario_dir / "scenario.json"
    result_path = scenario_dir / "result.json"
    if not scenario_path.exists() or not result_path.exists():
        return None

    scenario = json.loads(scenario_path.read_text())
    result = json.loads(result_path.read_text())
    artifacts_dir = scenario_dir / "artifacts"
    completion_dirs = sorted(artifacts_dir.glob("prompt_*/completion_*"))

    temporal = []
    steps_by_completion: dict[str, list[StepData]] = {}
    for completion_dir in completion_dirs:
        completion_key = f"{completion_dir.parent.name}/{completion_dir.name}"
        steps = load_completion_steps(
            completion_dir, workers=1, max_edges=scenario["max_edges"]
        )
        steps_by_completion[completion_key] = steps
        temporal_summary = analyze_single_completion(
            completion_dir, workers=1, max_edges=scenario["max_edges"]
        )
        if temporal_summary is not None:
            temporal.append({"completion_key": completion_key, **temporal_summary})

    return {
        "scenario": scenario,
        "result": result,
        "steps_by_completion": steps_by_completion,
        "temporal": temporal,
    }


def compare_to_exact(entries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in entries:
        prompt_idx = int(entry["scenario"]["gsm8k_indices"][0])
        grouped[prompt_idx][entry["scenario"]["method"]] = entry

    comparisons = []
    for prompt_idx, methods in sorted(grouped.items()):
        exact_entry = methods.get("exact")
        if exact_entry is None:
            continue

        for method_name, approx_entry in methods.items():
            if method_name == "exact":
                continue

            per_completion = []
            for completion_key, exact_steps in exact_entry[
                "steps_by_completion"
            ].items():
                approx_steps = approx_entry["steps_by_completion"].get(completion_key)
                if not approx_steps:
                    continue
                n_steps = min(len(exact_steps), len(approx_steps))
                if n_steps == 0:
                    continue

                feature_scores = []
                edge_scores = []
                weighted_scores = []
                mass_ratios = []
                for idx in range(n_steps):
                    exact_step = exact_steps[idx]
                    approx_step = approx_steps[idx]
                    feature_scores.append(feature_jaccard(exact_step, approx_step))
                    edge_scores.append(edge_jaccard(exact_step, approx_step))
                    weighted_scores.append(
                        weighted_edge_jaccard(exact_step, approx_step)
                    )
                    mass_ratios.append(edge_mass_ratio(approx_step, exact_step))

                exact_temporal = next(
                    (
                        item
                        for item in exact_entry["temporal"]
                        if item["completion_key"] == completion_key
                    ),
                    None,
                )
                approx_temporal = next(
                    (
                        item
                        for item in approx_entry["temporal"]
                        if item["completion_key"] == completion_key
                    ),
                    None,
                )

                per_completion.append(
                    {
                        "completion_key": completion_key,
                        "n_aligned_steps": n_steps,
                        "mean_feature_jaccard_vs_exact": float(
                            np.nanmean(feature_scores)
                        ),
                        "mean_edge_jaccard_vs_exact": float(np.nanmean(edge_scores)),
                        "mean_weighted_edge_jaccard_vs_exact": float(
                            np.nanmean(weighted_scores)
                        ),
                        "mean_edge_mass_ratio_vs_exact": float(np.nanmean(mass_ratios)),
                        "temporal_weighted_jaccard_corr": (
                            safe_corr(
                                approx_temporal["wjaccard_curve"],
                                exact_temporal["wjaccard_curve"],
                            )
                            if exact_temporal and approx_temporal
                            else None
                        ),
                        "temporal_feature_jaccard_corr": (
                            safe_corr(
                                approx_temporal["feature_jaccard_curve"],
                                exact_temporal["feature_jaccard_curve"],
                            )
                            if exact_temporal and approx_temporal
                            else None
                        ),
                    }
                )

            if per_completion:
                comparisons.append(
                    {
                        "prompt_index": prompt_idx,
                        "method": method_name,
                        "comparisons": per_completion,
                        "mean_feature_jaccard_vs_exact": float(
                            np.mean(
                                [
                                    item["mean_feature_jaccard_vs_exact"]
                                    for item in per_completion
                                ]
                            )
                        ),
                        "mean_edge_jaccard_vs_exact": float(
                            np.mean(
                                [
                                    item["mean_edge_jaccard_vs_exact"]
                                    for item in per_completion
                                ]
                            )
                        ),
                        "mean_weighted_edge_jaccard_vs_exact": float(
                            np.mean(
                                [
                                    item["mean_weighted_edge_jaccard_vs_exact"]
                                    for item in per_completion
                                ]
                            )
                        ),
                    }
                )

    return {
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze calibration/main sparsification experiment outputs"
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Root directory containing scenario subdirectories written by run_sparsification_experiment.py",
    )
    args = parser.parse_args()

    entries = []
    for scenario_dir in sorted(args.experiment_root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        loaded = load_scenario_dir(scenario_dir)
        if loaded is not None:
            entries.append(loaded)

    summary = {
        "experiment_root": str(args.experiment_root),
        "scenario_count": len(entries),
        "scenarios": [
            {
                "name": entry["scenario"]["name"],
                "stage": entry["scenario"].get("stage"),
                "method": entry["scenario"]["method"],
                "gsm8k_indices": entry["scenario"]["gsm8k_indices"],
                "status": entry["result"]["status"],
                "duration_seconds": entry["result"].get("duration_seconds"),
                "temporal": entry["temporal"],
            }
            for entry in entries
        ],
        "exact_comparisons": compare_to_exact(entries),
    }
    summary_path = args.experiment_root / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote analysis summary to {summary_path}")


if __name__ == "__main__":
    main()
