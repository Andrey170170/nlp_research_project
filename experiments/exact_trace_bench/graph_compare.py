from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from circuit_utils import StepData, load_compact


def _feature_set(step: StepData) -> set[tuple[int, int, int]]:
    return {tuple(row.tolist()) for row in step.feature_ids}


def _feature_label(step: StepData, feature_index: int) -> tuple[int, int, int]:
    return tuple(step.feature_ids[feature_index].tolist())


def _edge_map(step: StepData) -> dict[tuple[object, object], float]:
    edge_map: dict[tuple[object, object], float] = {}
    for row, col, weight in zip(step.row_idx, step.col_idx, step.weights, strict=True):
        row_value = int(row)
        if row_value >= step.n_features:
            continue
        source_label = _feature_label(step, int(col))
        target_label: object = ("feature",) + _feature_label(step, row_value)
        edge_map[(target_label, source_label)] = float(abs(weight))
    return edge_map


def _jaccard(a: set[Any], b: set[Any]) -> float:
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


def _weighted_edge_jaccard(
    edge_map_a: dict[tuple[object, object], float],
    edge_map_b: dict[tuple[object, object], float],
) -> float:
    keys = set(edge_map_a) | set(edge_map_b)
    if not keys:
        return float("nan")
    num = sum(min(edge_map_a.get(key, 0.0), edge_map_b.get(key, 0.0)) for key in keys)
    den = sum(max(edge_map_a.get(key, 0.0), edge_map_b.get(key, 0.0)) for key in keys)
    return num / den if den > 0.0 else float("nan")


def _feature_layer_histogram(features: set[tuple[int, int, int]]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for layer, _position, _feature in features:
        counts[layer] += 1
    return dict(sorted(counts.items()))


def compare_step_pair(step_a: StepData, step_b: StepData) -> dict[str, Any]:
    features_a = _feature_set(step_a)
    features_b = _feature_set(step_b)
    shared_features = features_a & features_b
    edges_a = _edge_map(step_a)
    edges_b = _edge_map(step_b)

    shared_by_layer = _feature_layer_histogram(shared_features)

    return {
        "step_index_a": step_a.step_idx,
        "step_index_b": step_b.step_idx,
        "n_features_a": len(features_a),
        "n_features_b": len(features_b),
        "n_features_shared": len(shared_features),
        "feature_jaccard": _jaccard(features_a, features_b),
        "n_edges_a": len(edges_a),
        "n_edges_b": len(edges_b),
        "edge_jaccard": _jaccard(set(edges_a), set(edges_b)),
        "weighted_edge_jaccard": _weighted_edge_jaccard(edges_a, edges_b),
        "n_logit_rows_a": len(
            {int(row) for row in step_a.row_idx if int(row) >= step_a.n_features}
        ),
        "n_logit_rows_b": len(
            {int(row) for row in step_b.row_idx if int(row) >= step_b.n_features}
        ),
        "feature_layers_a": _feature_layer_histogram(features_a),
        "feature_layers_b": _feature_layer_histogram(features_b),
        "feature_layers_shared": shared_by_layer,
    }


def _load_completion_steps(completion_dir: Path) -> list[StepData]:
    return [load_compact(path) for path in sorted(completion_dir.glob("step_*.npz"))]


def _completion_key(artifacts_dir: Path, completion_dir: Path) -> str:
    return str(completion_dir.relative_to(artifacts_dir))


def compare_artifact_dirs(
    left_artifacts: Path, right_artifacts: Path
) -> dict[str, Any]:
    left_completions = {
        _completion_key(left_artifacts, path): path
        for path in sorted(left_artifacts.glob("prompt_*/completion_*"))
    }
    right_completions = {
        _completion_key(right_artifacts, path): path
        for path in sorted(right_artifacts.glob("prompt_*/completion_*"))
    }
    shared_completion_keys = sorted(set(left_completions) & set(right_completions))

    completion_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []

    for completion_key in shared_completion_keys:
        steps_a = {
            step.step_idx: step
            for step in _load_completion_steps(left_completions[completion_key])
        }
        steps_b = {
            step.step_idx: step
            for step in _load_completion_steps(right_completions[completion_key])
        }
        shared_step_indices = sorted(set(steps_a) & set(steps_b))
        n_aligned_steps = len(shared_step_indices)
        if n_aligned_steps == 0:
            continue

        step_rows = [
            compare_step_pair(steps_a[step_idx], steps_b[step_idx])
            for step_idx in shared_step_indices
        ]
        all_step_rows.extend(
            [{"completion_key": completion_key, **row} for row in step_rows]
        )

        completion_rows.append(
            {
                "completion_key": completion_key,
                "n_steps_aligned": n_aligned_steps,
                "mean_feature_jaccard": float(
                    np.nanmean([row["feature_jaccard"] for row in step_rows])
                ),
                "mean_edge_jaccard": float(
                    np.nanmean([row["edge_jaccard"] for row in step_rows])
                ),
                "mean_weighted_edge_jaccard": float(
                    np.nanmean([row["weighted_edge_jaccard"] for row in step_rows])
                ),
                "mean_shared_features": float(
                    np.nanmean([row["n_features_shared"] for row in step_rows])
                ),
            }
        )

    summary = {
        "left_artifacts": str(left_artifacts),
        "right_artifacts": str(right_artifacts),
        "shared_completion_count": len(shared_completion_keys),
        "left_only_completion_count": len(
            set(left_completions) - set(right_completions)
        ),
        "right_only_completion_count": len(
            set(right_completions) - set(left_completions)
        ),
        "completion_comparisons": completion_rows,
        "step_comparisons": all_step_rows,
    }

    if completion_rows:
        summary["overall_mean_feature_jaccard"] = float(
            np.nanmean([row["mean_feature_jaccard"] for row in completion_rows])
        )
        summary["overall_mean_edge_jaccard"] = float(
            np.nanmean([row["mean_edge_jaccard"] for row in completion_rows])
        )
        summary["overall_mean_weighted_edge_jaccard"] = float(
            np.nanmean([row["mean_weighted_edge_jaccard"] for row in completion_rows])
        )

    return summary
