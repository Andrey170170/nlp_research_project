from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from circuit_utils import StepData

DEFAULT_EDGE_TOP_KS = (64, 128, 256, 512, 1024)
DEFAULT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)


def _feature_set(step: "StepData") -> set[tuple[int, int, int]]:
    return {tuple(row.tolist()) for row in step.feature_ids}


def _feature_label(step: "StepData", feature_index: int) -> tuple[int, int, int]:
    return tuple(step.feature_ids[feature_index].tolist())


def _edge_map(step: "StepData") -> dict[tuple[object, object], float]:
    edge_map: dict[tuple[object, object], float] = {}
    for row, col, weight in zip(step.row_idx, step.col_idx, step.weights):
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


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left_centered = left64 - float(left64.mean())
    right_centered = right64 - float(right64.mean())
    denominator = float(
        np.sqrt(
            np.sum(left_centered * left_centered)
            * np.sum(right_centered * right_centered)
        )
    )
    if denominator == 0.0:
        return None
    return float(np.sum(left_centered * right_centered) / denominator)


def _quantile_summary(values: np.ndarray) -> dict[str, float] | None:
    values64 = np.asarray(values, dtype=np.float64)
    if values64.size == 0:
        return None
    quantiles = np.quantile(values64, DEFAULT_QUANTILES)
    return {
        f"q{int(q * 100):02d}" if q < 1.0 else "q100": float(value)
        for q, value in zip(DEFAULT_QUANTILES, quantiles)
    }


def _topk_edge_overlap(
    left_edges: dict[tuple[object, object], float],
    right_edges: dict[tuple[object, object], float],
    *,
    ks: tuple[int, ...] = DEFAULT_EDGE_TOP_KS,
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    left_sorted = sorted(left_edges, key=lambda key: (-left_edges[key], repr(key)))
    right_sorted = sorted(right_edges, key=lambda key: (-right_edges[key], repr(key)))
    for k in ks:
        k_eff = min(k, len(left_sorted), len(right_sorted))
        if k_eff <= 0:
            result[str(k)] = {
                "k_effective": 0,
                "shared_count": 0,
                "overlap_fraction_of_k": None,
                "jaccard": None,
            }
            continue
        left_top = set(left_sorted[:k_eff])
        right_top = set(right_sorted[:k_eff])
        shared = left_top & right_top
        result[str(k)] = {
            "k_effective": k_eff,
            "shared_count": len(shared),
            "overlap_fraction_of_k": len(shared) / k_eff,
            "jaccard": _jaccard(left_top, right_top),
        }
    return result


def _feature_layer_histogram(features: set[tuple[int, int, int]]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for layer, _position, _feature in features:
        counts[layer] += 1
    return dict(sorted(counts.items()))


def _feature_position_bucket_histogram(
    features: set[tuple[int, int, int]], *, bucket_size: int = 10
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _layer, position, _feature in features:
        bucket_start = (int(position) // bucket_size) * bucket_size
        counts[f"{bucket_start}-{bucket_start + bucket_size - 1}"] += 1
    return dict(sorted(counts.items()))


def _feature_support_decomposition(
    features_a: set[tuple[int, int, int]], features_b: set[tuple[int, int, int]]
) -> dict[str, Any]:
    shared = features_a & features_b
    left_unique = features_a - shared
    right_unique = features_b - shared
    return {
        "shared_count": len(shared),
        "left_unique_count": len(left_unique),
        "right_unique_count": len(right_unique),
        "shared_by_layer": _feature_layer_histogram(shared),
        "left_unique_by_layer": _feature_layer_histogram(left_unique),
        "right_unique_by_layer": _feature_layer_histogram(right_unique),
        "shared_by_position_bucket": _feature_position_bucket_histogram(shared),
        "left_unique_by_position_bucket": _feature_position_bucket_histogram(
            left_unique
        ),
        "right_unique_by_position_bucket": _feature_position_bucket_histogram(
            right_unique
        ),
    }


def _classify_feature(
    label: tuple[int, int, int], shared_features: set[tuple[int, int, int]]
) -> str:
    return "shared" if label in shared_features else "unique"


def _all_edge_map(step: "StepData") -> dict[tuple[object, object], float]:
    edge_map: dict[tuple[object, object], float] = {}
    for row, col, weight in zip(step.row_idx, step.col_idx, step.weights):
        row_value = int(row)
        col_value = int(col)
        if not 0 <= col_value < step.n_features:
            continue
        source_label: object = ("feature",) + _feature_label(step, col_value)
        if row_value < step.n_features:
            target_label: object = ("feature",) + _feature_label(step, row_value)
        else:
            target_label = ("logit", row_value - step.n_features)
        edge_map[(target_label, source_label)] = float(abs(weight))
    return edge_map


def _edge_class_maps(
    step: "StepData", shared_features: set[tuple[int, int, int]]
) -> dict[str, dict[tuple[object, object], float]]:
    class_maps: dict[str, dict[tuple[object, object], float]] = {
        "shared_to_shared": {},
        "shared_to_unique": {},
        "unique_to_shared": {},
        "unique_to_unique": {},
        "shared_to_logit": {},
        "unique_to_logit": {},
    }
    for row, col, weight in zip(step.row_idx, step.col_idx, step.weights):
        row_value = int(row)
        col_value = int(col)
        if not 0 <= col_value < step.n_features:
            continue
        source_feature = _feature_label(step, col_value)
        source_class = _classify_feature(source_feature, shared_features)
        source_label: object = ("feature",) + source_feature
        if row_value < step.n_features:
            target_feature = _feature_label(step, row_value)
            target_class = _classify_feature(target_feature, shared_features)
            target_label: object = ("feature",) + target_feature
            edge_class = f"{source_class}_to_{target_class}"
        else:
            target_label = ("logit", row_value - step.n_features)
            edge_class = f"{source_class}_to_logit"
        class_maps.setdefault(edge_class, {})[(target_label, source_label)] = float(
            abs(weight)
        )
    return class_maps


def _edge_class_summary(
    class_maps: dict[str, dict[tuple[object, object], float]],
) -> dict[str, dict[str, float | int | None]]:
    total_mass = sum(sum(edge_map.values()) for edge_map in class_maps.values())
    return {
        edge_class: {
            "edge_count": len(edge_map),
            "mass": float(sum(edge_map.values())),
            "mass_fraction": (
                float(sum(edge_map.values()) / total_mass) if total_mass > 0.0 else None
            ),
        }
        for edge_class, edge_map in sorted(class_maps.items())
    }


def _shared_endpoint_edge_stability(
    left_edges: dict[tuple[object, object], float],
    right_edges: dict[tuple[object, object], float],
) -> dict[str, Any]:
    common_edges = sorted(set(left_edges) & set(right_edges), key=repr)
    left_common = np.asarray(
        [left_edges[key] for key in common_edges], dtype=np.float64
    )
    right_common = np.asarray(
        [right_edges[key] for key in common_edges], dtype=np.float64
    )
    abs_delta = np.abs(left_common - right_common)
    relative_denominator = np.maximum(
        np.maximum(np.abs(left_common), np.abs(right_common)),
        1e-12,
    )
    relative_delta = abs_delta / relative_denominator if abs_delta.size else abs_delta
    return {
        "left_edge_count": len(left_edges),
        "right_edge_count": len(right_edges),
        "common_edge_count": len(common_edges),
        "edge_jaccard": _jaccard(set(left_edges), set(right_edges)),
        "weighted_edge_jaccard": _weighted_edge_jaccard(left_edges, right_edges),
        "common_edge_weight_pearson": _safe_pearson(left_common, right_common),
        "common_edge_abs_delta_quantiles": _quantile_summary(abs_delta),
        "common_edge_relative_delta_quantiles": _quantile_summary(relative_delta),
        "topk_overlap": _topk_edge_overlap(left_edges, right_edges),
    }


def compare_step_pair(step_a: "StepData", step_b: "StepData") -> dict[str, Any]:
    features_a = _feature_set(step_a)
    features_b = _feature_set(step_b)
    shared_features = features_a & features_b
    edges_a = _edge_map(step_a)
    edges_b = _edge_map(step_b)
    all_edges_a = _all_edge_map(step_a)
    all_edges_b = _all_edge_map(step_b)
    edge_class_maps_a = _edge_class_maps(step_a, shared_features)
    edge_class_maps_b = _edge_class_maps(step_b, shared_features)

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
        "feature_support_decomposition": _feature_support_decomposition(
            features_a, features_b
        ),
        "edge_class_decomposition_a": _edge_class_summary(edge_class_maps_a),
        "edge_class_decomposition_b": _edge_class_summary(edge_class_maps_b),
        "shared_endpoint_edge_stability": _shared_endpoint_edge_stability(
            edge_class_maps_a.get("shared_to_shared", {}),
            edge_class_maps_b.get("shared_to_shared", {}),
        ),
        "all_edge_weighted_jaccard": _weighted_edge_jaccard(all_edges_a, all_edges_b),
    }


def _load_completion_steps(completion_dir: Path) -> list["StepData"]:
    from circuit_utils import load_compact

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
