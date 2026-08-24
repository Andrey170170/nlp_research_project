from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nlp_research_project.exact_trace_bench.compact_io import StepData
    from nlp_research_project.circuit_stability_analysis.signed_graph import SignedGraph

DEFAULT_EDGE_TOP_KS = (64, 128, 256, 512, 1024)
DEFAULT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
COMPACT_STEP_RE = re.compile(r"step_\d+\.npz\Z")
LEGACY_ALL_EDGE_SCOPE = "legacy_feature_source_edges_only"
LEGACY_ALL_EDGE_BUCKETS = ("feature<-feature", "logit<-feature")
TYPED_BUCKET_SCOPE = "typed_bucket_edges_only"
HISTORICAL_RETENTION_POLICY_ID = "historical_bucket_top_p_unstable_argsort_v0"
HISTORICAL_RETENTION_ALGORITHM_VERSION = "absolute_mass_top_p_then_cap_unstable_ties_v0"
HISTORICAL_RETENTION_ALGORITHM = {
    "algorithm": HISTORICAL_RETENTION_ALGORITHM_VERSION,
    "input_cast": "float32_before_abs",
    "nonzero_rule": "float32_abs_not_equal_zero",
    "mass_accumulator": "float64",
    "sort": "torch_argsort_descending_tie_order_unspecified",
    "top_p_boundary": "smallest_prefix_with_cumulative_mass_greater_or_equal",
    "cap_application": "after_top_p_prefix_selection",
    "retained_order": "torch_argsort_descending_tie_order_unspecified",
    "persisted_weight": "signed_float32",
}


@dataclass(frozen=True)
class _LoadedCompact:
    graph: Any
    historical_reference: bool


@dataclass(frozen=True)
class _CanonicalBucket:
    rows: np.ndarray
    cols: np.ndarray
    weights: np.ndarray


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


def _jaccard(a: set[Any], b: set[Any]) -> float | None:
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def _weighted_edge_jaccard(
    edge_map_a: dict[Any, float],
    edge_map_b: dict[Any, float],
) -> float | None:
    keys = set(edge_map_a) | set(edge_map_b)
    if not keys:
        return None
    num = sum(min(edge_map_a.get(key, 0.0), edge_map_b.get(key, 0.0)) for key in keys)
    den = sum(max(edge_map_a.get(key, 0.0), edge_map_b.get(key, 0.0)) for key in keys)
    return num / den if den > 0.0 else None


def _finite_mean(values: list[float | int | None]) -> float | None:
    finite_values = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.mean(finite_values)) if finite_values else None


def _finite_min(values: list[float | int | None]) -> float | None:
    finite_values = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return min(finite_values) if finite_values else None


def _finite_max(values: list[float | int | None]) -> float | None:
    finite_values = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return max(finite_values) if finite_values else None


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
    left_edges: dict[Any, float],
    right_edges: dict[Any, float],
    *,
    ks: tuple[int, ...] = DEFAULT_EDGE_TOP_KS,
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    left_sorted = sorted(left_edges, key=lambda key: (-left_edges[key], repr(key)))
    right_sorted = sorted(right_edges, key=lambda key: (-right_edges[key], repr(key)))
    for k in ks:
        left_k = min(k, len(left_sorted))
        right_k = min(k, len(right_sorted))
        left_top = set(left_sorted[:left_k])
        right_top = set(right_sorted[:right_k])
        shared = left_top & right_top
        union = left_top | right_top
        denominator = max(left_k, right_k)
        result[str(k)] = {
            "k_effective": min(left_k, right_k),
            "left_k_effective": left_k,
            "right_k_effective": right_k,
            "shared_count": len(shared),
            "overlap_fraction_of_k": (
                len(shared) / denominator if denominator else 1.0
            ),
            "jaccard": len(shared) / len(union) if union else 1.0,
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


def _historical_all_edge_map(step: "StepData") -> dict[tuple[object, object], float]:
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


def _historical_all_edge_signed_map(
    step: "StepData",
) -> dict[tuple[object, object], float]:
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
        edge_map[(target_label, source_label)] = float(weight)
    return edge_map


def _shared_edge_sign_agreement(
    left_edges: dict[Any, float],
    right_edges: dict[Any, float],
) -> float | None:
    shared = set(left_edges) & set(right_edges)
    if not shared:
        return None
    matching = sum(
        np.sign(left_edges[key]) == np.sign(right_edges[key]) for key in shared
    )
    return float(matching / len(shared))


def _normalized_l1_deviation(
    left_edges: dict[Any, float],
    right_edges: dict[Any, float],
) -> float:
    keys = set(left_edges) | set(right_edges)
    numerator = sum(
        abs(left_edges.get(key, 0.0) - right_edges.get(key, 0.0)) for key in keys
    )
    denominator = max(
        sum(abs(value) for value in left_edges.values()),
        sum(abs(value) for value in right_edges.values()),
        1e-12,
    )
    return numerator / denominator


def _validate_bucket_graph(graph: "SignedGraph") -> None:
    names = tuple(graph.bucket_names)
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate typed bucket names in {graph.path}")
    lengths = {
        len(graph.bucket_row_idx),
        len(graph.bucket_col_idx),
        len(graph.bucket_weights),
        len(graph.bucket_ids),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"typed bucket COO arrays have mismatched lengths in {graph.path}"
        )
    ids = np.asarray(graph.bucket_ids, dtype=np.int64)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= len(names)):
        raise ValueError(f"typed bucket id is outside bucket_names in {graph.path}")
    if not np.isfinite(graph.bucket_weights).all():
        raise ValueError(f"non-finite typed bucket weight in {graph.path}")


def _canonical_bucket_map(graph: "SignedGraph", bucket_name: str) -> _CanonicalBucket:
    """Canonicalize one COO bucket without retaining every bucket in memory."""
    if bucket_name not in graph.bucket_names:
        return _CanonicalBucket(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    bucket_id = graph.bucket_names.index(bucket_name)
    mask = np.asarray(graph.bucket_ids) == bucket_id
    rows = np.asarray(graph.bucket_row_idx[mask], dtype=np.int64)
    cols = np.asarray(graph.bucket_col_idx[mask], dtype=np.int64)
    weights = np.asarray(graph.bucket_weights[mask], dtype=np.float64)
    if not weights.size:
        return _CanonicalBucket(rows, cols, weights)
    # Sorting the signed value after the primary row/column keys gives COO
    # duplicates a deterministic reduction order as well as making entry order
    # irrelevant.
    order = np.lexsort((weights, cols, rows))
    rows = rows[order]
    cols = cols[order]
    weights = weights[order]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero((rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])) + 1,
        )
    )
    reduced = np.add.reduceat(weights, starts)
    return _CanonicalBucket(rows[starts], cols[starts], reduced)


def _bucket_metric_report(
    left: _CanonicalBucket, right: _CanonicalBucket
) -> dict[str, Any]:
    key_dtype = np.dtype([("row", np.int64), ("col", np.int64)])
    left_keys = np.empty(left.rows.size, dtype=key_dtype)
    right_keys = np.empty(right.rows.size, dtype=key_dtype)
    left_keys["row"], left_keys["col"] = left.rows, left.cols
    right_keys["row"], right_keys["col"] = right.rows, right.cols
    _shared, left_idx, right_idx = np.intersect1d(
        left_keys,
        right_keys,
        assume_unique=True,
        return_indices=True,
    )
    shared_count = int(left_idx.size)
    union_count = int(left.weights.size + right.weights.size - shared_count)
    left_abs = np.abs(left.weights)
    right_abs = np.abs(right.weights)
    left_total = float(left_abs.sum(dtype=np.float64))
    right_total = float(right_abs.sum(dtype=np.float64))
    shared_min = float(
        np.minimum(left_abs[left_idx], right_abs[right_idx]).sum(dtype=np.float64)
    )
    weighted_denominator = left_total + right_total - shared_min
    left_unique = left_total - float(left_abs[left_idx].sum(dtype=np.float64))
    right_unique = right_total - float(right_abs[right_idx].sum(dtype=np.float64))
    magnitude_delta = float(
        np.abs(left_abs[left_idx] - right_abs[right_idx]).sum(dtype=np.float64)
    )
    signed_delta = float(
        np.abs(left.weights[left_idx] - right.weights[right_idx]).sum(dtype=np.float64)
    )
    exact = bool(
        np.array_equal(left.rows, right.rows)
        and np.array_equal(left.cols, right.cols)
        and np.array_equal(left.weights, right.weights)
    )

    def top_candidates(bucket: _CanonicalBucket) -> dict[tuple[int, int], float]:
        order = np.lexsort((bucket.cols, bucket.rows, -np.abs(bucket.weights)))[
            : max(DEFAULT_EDGE_TOP_KS)
        ]
        return {
            (int(bucket.rows[index]), int(bucket.cols[index])): float(
                abs(bucket.weights[index])
            )
            for index in order
        }

    denominator = max(left_total, right_total, 1e-12)
    return {
        "edge_count_a": int(left.weights.size),
        "edge_count_b": int(right.weights.size),
        "shared_edge_count": shared_count,
        "union_edge_count": union_count,
        "support_jaccard": shared_count / union_count if union_count else None,
        "weighted_jaccard": (
            shared_min / weighted_denominator if weighted_denominator > 0.0 else None
        ),
        "normalized_l1_deviation": (magnitude_delta + left_unique + right_unique)
        / denominator,
        "shared_sign_agreement": (
            float(
                np.mean(
                    np.sign(left.weights[left_idx]) == np.sign(right.weights[right_idx])
                )
            )
            if shared_count
            else None
        ),
        "signed_normalized_l1_deviation": (signed_delta + left_unique + right_unique)
        / denominator,
        "topk_overlap": _topk_edge_overlap(top_candidates(left), top_candidates(right)),
        "exact": exact,
        "classification": "strict_exact" if exact else "non_exact",
    }


def _validate_current_domain_compatibility(
    left_graph: "SignedGraph", right_graph: "SignedGraph"
) -> None:
    """Fail closed when local COO identities refer to different typed domains."""

    domain_arrays = (
        ("token_ids", left_graph.token_ids, right_graph.token_ids),
        (
            "logit_token_ids",
            left_graph.logit_token_ids,
            right_graph.logit_token_ids,
        ),
    )
    for name, left, right in domain_arrays:
        if (
            left is None
            or right is None
            or not np.array_equal(np.asarray(left), np.asarray(right))
        ):
            raise ValueError(f"typed graph {name} domain mismatch")
    left_error_shape = left_graph.error_node_shape
    right_error_shape = right_graph.error_node_shape
    if (
        left_error_shape is None
        or right_error_shape is None
        or tuple(left_error_shape) != tuple(right_error_shape)
    ):
        raise ValueError("typed graph error_node_shape domain mismatch")


def _validate_historical_current_domain_compatibility(
    historical_graph: Any, current_graph: "SignedGraph"
) -> None:
    """Admit historical domains only when their semantic identities are known."""

    historical_logit_ids = historical_graph.logit_token_ids
    if historical_logit_ids is None or np.any(
        np.asarray(historical_logit_ids, dtype=np.int64) == -1
    ):
        raise ValueError("historical logit_token_ids comparison scope is unavailable")
    _validate_current_domain_compatibility(historical_graph, current_graph)


def _typed_bucket_comparison(
    left_graph: "SignedGraph | None", right_graph: "SignedGraph | None"
) -> dict[str, Any]:
    left_available = bool(left_graph is not None and left_graph.bucket_names)
    right_available = bool(right_graph is not None and right_graph.bucket_names)
    base: dict[str, Any] = {
        "scope": TYPED_BUCKET_SCOPE,
        "left_available": left_available,
        "right_available": right_available,
        "comparable": left_available and right_available,
    }
    if not left_available or not right_available:
        base["classification"] = (
            "not_available" if left_available == right_available else "incomplete"
        )
        base["buckets"] = {}
        return base

    assert left_graph is not None
    assert right_graph is not None
    _validate_bucket_graph(left_graph)
    _validate_bucket_graph(right_graph)
    left_names = set(left_graph.bucket_names)
    right_names = set(right_graph.bucket_names)
    bucket_names = sorted(left_names | right_names)
    bucket_reports: dict[str, dict[str, Any]] = {}
    for name in bucket_names:
        bucket_reports[name] = _bucket_metric_report(
            _canonical_bucket_map(left_graph, name),
            _canonical_bucket_map(right_graph, name),
        )
    exact = left_names == right_names and all(
        bool(report["exact"]) for report in bucket_reports.values()
    )
    shared_edges = sum(
        int(report["shared_edge_count"]) for report in bucket_reports.values()
    )
    union_edges = sum(
        int(report["union_edge_count"]) for report in bucket_reports.values()
    )
    aggregate = {
        "edge_count_a": sum(
            int(report["edge_count_a"]) for report in bucket_reports.values()
        ),
        "edge_count_b": sum(
            int(report["edge_count_b"]) for report in bucket_reports.values()
        ),
        "shared_edge_count": shared_edges,
        "union_edge_count": union_edges,
        "support_jaccard": shared_edges / union_edges if union_edges else None,
        "exact": exact,
        "classification": "strict_exact" if exact else "non_exact",
    }
    base.update(
        {
            "classification": "strict_exact" if exact else "non_exact",
            "bucket_count_a": len(left_names),
            "bucket_count_b": len(right_names),
            "shared_bucket_count": len(left_names & right_names),
            "left_only_buckets": sorted(left_names - right_names),
            "right_only_buckets": sorted(right_names - left_names),
            "non_exact_buckets": [
                name for name in bucket_names if not bucket_reports[name]["exact"]
            ],
            "aggregate": aggregate,
            "buckets": bucket_reports,
        }
    )
    return base


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


def _compare_historical_step_pair(
    step_a: "StepData",
    step_b: "StepData",
    *,
    bucket_graph_a: "SignedGraph | None" = None,
    bucket_graph_b: "SignedGraph | None" = None,
) -> dict[str, Any]:
    features_a = _feature_set(step_a)
    features_b = _feature_set(step_b)
    shared_features = features_a & features_b
    edges_a = _edge_map(step_a)
    edges_b = _edge_map(step_b)
    all_edges_a = _historical_all_edge_map(step_a)
    all_edges_b = _historical_all_edge_map(step_b)
    signed_all_edges_a = _historical_all_edge_signed_map(step_a)
    signed_all_edges_b = _historical_all_edge_signed_map(step_b)
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
        "topk_edge_overlap": _topk_edge_overlap(edges_a, edges_b),
        "all_edge_jaccard": _jaccard(set(all_edges_a), set(all_edges_b)),
        "all_edge_weighted_jaccard": _weighted_edge_jaccard(all_edges_a, all_edges_b),
        "all_edge_topk_overlap": _topk_edge_overlap(all_edges_a, all_edges_b),
        "all_edge_normalized_l1_deviation": (
            _normalized_l1_deviation(
                all_edges_a,
                all_edges_b,
            )
        ),
        "all_edge_shared_sign_agreement": _shared_edge_sign_agreement(
            signed_all_edges_a,
            signed_all_edges_b,
        ),
        "all_edge_signed_normalized_l1_deviation": _normalized_l1_deviation(
            signed_all_edges_a,
            signed_all_edges_b,
        ),
        "all_edge_scope": LEGACY_ALL_EDGE_SCOPE,
        "all_edge_included_buckets": list(LEGACY_ALL_EDGE_BUCKETS),
        "all_edge_includes_typed_buckets": False,
        "typed_bucket_comparison": _typed_bucket_comparison(
            bucket_graph_a, bucket_graph_b
        ),
        "target_token_match": float(step_a.token_text == step_b.token_text),
        "target_token_a": step_a.token_text,
        "target_token_b": step_b.token_text,
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
    }


def compare_historical_step_pair(
    step_a: "StepData", step_b: "StepData"
) -> dict[str, Any]:
    """Compare two explicitly historical global-projection steps."""
    return _compare_historical_step_pair(step_a, step_b)


def _load_compact_for_compare(
    path: Path, *, historical_reference: bool
) -> _LoadedCompact:
    if historical_reference:
        from nlp_research_project.exact_trace_bench.compact_io import (
            load_historical_compact_graph,
        )
        from nlp_research_project.exact_trace_bench.typed_compact_graph import (
            CANONICAL_BUCKET_NAMES,
            DEFAULT_RETENTION_POLICY_ID,
            get_retention_policy,
        )

        compact = load_historical_compact_graph(path)
        if compact.bucket_row_idx is None:
            raise ValueError(f"historical reference lacks typed bucket arrays: {path}")
        policy = get_retention_policy(DEFAULT_RETENTION_POLICY_ID)
        if compact.bucket_names != CANONICAL_BUCKET_NAMES or any(
            compact.bucket_metadata.get(name, {}).get("policy")
            != policy.buckets[name].to_json()
            for name in CANONICAL_BUCKET_NAMES
        ):
            raise ValueError(
                f"historical reference does not implement {policy.policy_id}: {path}"
            )
        return _LoadedCompact(graph=compact, historical_reference=True)

    from nlp_research_project.exact_trace_bench.typed_compact_graph import (
        load_typed_compact_graph,
    )

    return _LoadedCompact(
        graph=load_typed_compact_graph(path), historical_reference=False
    )


def _step_view(loaded: _LoadedCompact) -> Any:
    return loaded.graph.step if loaded.historical_reference else loaded.graph


def _bucket_view(loaded: _LoadedCompact) -> Any:
    return loaded.graph


def _historical_retention_policy_fingerprint(graph: Any) -> str:
    payload = {
        "policy_id": HISTORICAL_RETENTION_POLICY_ID,
        "algorithm": HISTORICAL_RETENTION_ALGORITHM,
        "buckets": {
            name: graph.bucket_metadata[name]["policy"] for name in graph.bucket_names
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fingerprint_report(left: _LoadedCompact, right: _LoadedCompact) -> dict[str, Any]:
    from nlp_research_project.exact_trace_bench.typed_compact_graph import (
        DEFAULT_RETENTION_POLICY_ID,
        get_retention_policy,
    )

    policy = get_retention_policy(DEFAULT_RETENTION_POLICY_ID)
    right_graph = right.graph
    if right.historical_reference:
        raise ValueError("the candidate graph may not use the historical adapter")
    fields = (
        "content_fingerprint",
        "graph_fingerprint",
        "target_fingerprint",
        "provider_fingerprint",
        "trace_fingerprint",
        "step_fingerprint",
    )
    missing = [name for name in fields if not getattr(right_graph, name, None)]
    if missing:
        raise ValueError(f"candidate graph lacks required fingerprints: {missing}")
    if right_graph.retention_policy_id != policy.policy_id:
        raise ValueError("candidate graph retention policy id is not typed_top_p_v1")
    if right_graph.retention_policy_fingerprint != policy.fingerprint:
        raise ValueError("candidate graph retention policy fingerprint mismatch")
    report: dict[str, Any] = {
        "candidate_schema_version": right_graph.schema_version,
        "candidate_retention_policy_id": right_graph.retention_policy_id,
        "candidate_retention_policy_fingerprint": (
            right_graph.retention_policy_fingerprint
        ),
        "candidate_current_policy_valid": True,
        "historical_reference": left.historical_reference,
        "policy_compatible": True,
    }
    if left.historical_reference:
        report.update(
            {
                "reference_policy_identity": HISTORICAL_RETENTION_POLICY_ID,
                "reference_retention_policy_id": HISTORICAL_RETENTION_POLICY_ID,
                "reference_retention_algorithm_version": (
                    HISTORICAL_RETENTION_ALGORITHM_VERSION
                ),
                "reference_retention_policy_fingerprint": (
                    _historical_retention_policy_fingerprint(left.graph)
                ),
                "reference_bucket_rules_match_candidate": True,
                "fingerprint_comparisons_available": False,
                "policy_compatible": False,
            }
        )
    else:
        left_graph = left.graph
        if not all(getattr(left_graph, name, None) for name in fields):
            raise ValueError("reference graph lacks required fingerprints")
        report["fingerprint_comparisons_available"] = True
        report["policy_compatible"] = bool(
            left_graph.retention_policy_id == right_graph.retention_policy_id
            and left_graph.retention_policy_fingerprint
            == right_graph.retention_policy_fingerprint
        )
        for name in fields:
            report[f"{name}_match"] = bool(
                getattr(left_graph, name) == getattr(right_graph, name)
            )
    return report


def _compare_loaded_pair(left: _LoadedCompact, right: _LoadedCompact) -> dict[str, Any]:
    left_step = _step_view(left)
    right_step = _step_view(right)
    if left.historical_reference:
        _validate_historical_current_domain_compatibility(left.graph, right.graph)
    else:
        _validate_current_domain_compatibility(left.graph, right.graph)
    features_a = _feature_set(left_step)
    features_b = _feature_set(right_step)
    typed = _typed_bucket_comparison(_bucket_view(left), _bucket_view(right))
    if not typed.get("comparable"):
        raise ValueError("all six typed edge buckets are required for comparison")
    fingerprints = _fingerprint_report(left, right)
    row: dict[str, Any] = {
        "step_index_a": int(left_step.step_idx),
        "step_index_b": int(right_step.step_idx),
        "n_features_a": len(features_a),
        "n_features_b": len(features_b),
        "n_features_shared": len(features_a & features_b),
        "feature_jaccard": _jaccard(features_a, features_b),
        "target_token_match": float(left_step.token_text == right_step.token_text),
        "target_token_a": left_step.token_text,
        "target_token_b": right_step.token_text,
        "feature_support_decomposition": _feature_support_decomposition(
            features_a, features_b
        ),
        "typed_bucket_comparison": typed,
        "fingerprints": fingerprints,
        "policy_compatible": fingerprints["policy_compatible"],
    }
    for name, report in typed["buckets"].items():
        prefix = f"bucket_{name.replace('<-', '_').replace('-', '_')}"
        for metric in (
            "support_jaccard",
            "weighted_jaccard",
            "normalized_l1_deviation",
            "shared_sign_agreement",
            "signed_normalized_l1_deviation",
        ):
            row[f"{prefix}_{metric}"] = report[metric]
        row[f"{prefix}_exact"] = float(bool(report["exact"]))
        row[f"{prefix}_top256_jaccard"] = report["topk_overlap"]["256"]["jaccard"]
    return row


def compare_compact_paths(
    left_path: Path,
    right_path: Path,
    *,
    historical_reference: bool = False,
) -> dict[str, Any]:
    """Compare typed graphs, optionally naming the left side as historical."""
    left = _load_compact_for_compare(
        Path(left_path), historical_reference=historical_reference
    )
    right = _load_compact_for_compare(Path(right_path), historical_reference=False)
    return _compare_loaded_pair(left, right)


def _load_completion_steps(
    completion_dir: Path, *, historical_reference: bool
) -> list[_LoadedCompact]:
    return [
        _load_compact_for_compare(path, historical_reference=historical_reference)
        for path in sorted(completion_dir.glob("step_*.npz"))
        if COMPACT_STEP_RE.fullmatch(path.name)
    ]


def _comparison_set_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    typed = [row["typed_bucket_comparison"] for row in rows]
    comparable = [row for row in typed if row["comparable"]]
    if not typed or not comparable:
        classification = "not_available"
    elif len(comparable) != len(typed):
        classification = "incomplete"
    elif all(row["classification"] == "strict_exact" for row in comparable):
        classification = "strict_exact"
    else:
        classification = "non_exact"
    return {
        "scope": TYPED_BUCKET_SCOPE,
        "classification": classification,
        "step_count": len(typed),
        "comparable_step_count": len(comparable),
        "strict_exact_step_count": sum(
            row["classification"] == "strict_exact" for row in comparable
        ),
        "non_exact_buckets": sorted(
            {
                bucket
                for row in comparable
                for bucket in row.get("non_exact_buckets", [])
            }
        ),
    }


def _completion_key(artifacts_dir: Path, completion_dir: Path) -> str:
    return str(completion_dir.relative_to(artifacts_dir))


def compare_artifact_dirs(
    left_artifacts: Path,
    right_artifacts: Path,
    *,
    historical_reference: bool = False,
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
            _step_view(loaded).step_idx: loaded
            for loaded in _load_completion_steps(
                left_completions[completion_key],
                historical_reference=historical_reference,
            )
        }
        steps_b = {
            _step_view(loaded).step_idx: loaded
            for loaded in _load_completion_steps(
                right_completions[completion_key], historical_reference=False
            )
        }
        shared_step_indices = sorted(set(steps_a) & set(steps_b))
        n_aligned_steps = len(shared_step_indices)
        step_rows = [
            _compare_loaded_pair(steps_a[step_idx], steps_b[step_idx])
            for step_idx in shared_step_indices
        ]
        all_step_rows.extend(
            [{"completion_key": completion_key, **row} for row in step_rows]
        )

        completion_rows.append(
            {
                "completion_key": completion_key,
                "n_steps_left": len(steps_a),
                "n_steps_right": len(steps_b),
                "n_steps_aligned": n_aligned_steps,
                "left_only_step_count": len(set(steps_a) - set(steps_b)),
                "right_only_step_count": len(set(steps_b) - set(steps_a)),
                "mean_feature_jaccard": _finite_mean(
                    [row["feature_jaccard"] for row in step_rows]
                ),
                "mean_target_token_match": _finite_mean(
                    [row["target_token_match"] for row in step_rows]
                ),
                "mean_shared_features": _finite_mean(
                    [row["n_features_shared"] for row in step_rows]
                ),
                "policy_compatible": all(row["policy_compatible"] for row in step_rows),
                "typed_bucket_comparison": _comparison_set_summary(step_rows),
            }
        )

    summary: dict[str, Any] = {
        "left_artifacts": str(left_artifacts),
        "right_artifacts": str(right_artifacts),
        "shared_completion_count": len(shared_completion_keys),
        "left_only_completion_count": len(
            set(left_completions) - set(right_completions)
        ),
        "right_only_completion_count": len(
            set(right_completions) - set(left_completions)
        ),
        "aligned_completion_count": sum(
            row["n_steps_aligned"] > 0 for row in completion_rows
        ),
        "aligned_step_count": sum(row["n_steps_aligned"] for row in completion_rows),
        "comparison_complete": bool(completion_rows)
        and not (set(left_completions) ^ set(right_completions))
        and all(
            row["n_steps_aligned"] > 0
            and row["left_only_step_count"] == 0
            and row["right_only_step_count"] == 0
            for row in completion_rows
        ),
        "historical_reference": historical_reference,
        "policy_compatible": bool(all_step_rows)
        and all(row["policy_compatible"] for row in all_step_rows),
        "candidate_current_policy_valid": bool(all_step_rows)
        and all(
            row["fingerprints"].get("candidate_current_policy_valid") is True
            for row in all_step_rows
        ),
        "reference_bucket_rules_match_candidate": (
            bool(all_step_rows)
            and all(
                row["fingerprints"].get("reference_bucket_rules_match_candidate")
                is True
                for row in all_step_rows
            )
            if historical_reference
            else None
        ),
        "completion_comparisons": completion_rows,
        "step_comparisons": all_step_rows,
        "typed_bucket_comparison": _comparison_set_summary(all_step_rows),
    }
    summary["overall_typed_bucket_classification"] = summary["typed_bucket_comparison"][
        "classification"
    ]

    if completion_rows:
        summary["overall_mean_feature_jaccard"] = _finite_mean(
            [row["mean_feature_jaccard"] for row in completion_rows]
        )
        summary["overall_mean_target_token_match"] = _finite_mean(
            [row["mean_target_token_match"] for row in completion_rows]
        )
    worst_metric_sources = {
        "worst_step_feature_jaccard": "feature_jaccard",
        "worst_step_target_token_match": "target_token_match",
    }
    bucket_metric_suffixes = (
        "support_jaccard",
        "weighted_jaccard",
        "top256_jaccard",
        "shared_sign_agreement",
        "exact",
        "normalized_l1_deviation",
        "signed_normalized_l1_deviation",
    )
    from nlp_research_project.exact_trace_bench.typed_compact_graph import (
        CANONICAL_BUCKET_NAMES,
    )

    for bucket_name in CANONICAL_BUCKET_NAMES:
        prefix = f"bucket_{bucket_name.replace('<-', '_').replace('-', '_')}"
        for suffix in bucket_metric_suffixes:
            row_key = f"{prefix}_{suffix}"
            summary_key = f"worst_step_{row_key}"
            worst_metric_sources[summary_key] = row_key
            summary[f"overall_mean_{row_key}"] = _finite_mean(
                [row[row_key] for row in all_step_rows]
            )
    for summary_key, row_key in worst_metric_sources.items():
        values = [row[row_key] for row in all_step_rows]
        summary[summary_key] = (
            _finite_max(values)
            if row_key.endswith("normalized_l1_deviation")
            else _finite_min(values)
        )

    summary["worst_step_evidence"] = {}
    for summary_key, row_key in worst_metric_sources.items():
        worst_value = summary[summary_key]
        matching_rows = []
        for row in all_step_rows:
            value = row[row_key]
            if worst_value is not None and value == worst_value:
                matching_rows.append(
                    {
                        "completion_key": row["completion_key"],
                        "step_index_a": row["step_index_a"],
                        "step_index_b": row["step_index_b"],
                        "value": value,
                    }
                )
        summary["worst_step_evidence"][summary_key] = matching_rows

    return summary
