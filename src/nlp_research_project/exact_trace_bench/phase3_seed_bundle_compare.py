from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TOP_KS = (64, 128, 256, 512, 1024)
DEFAULT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)


def _feature_tuples(features: np.ndarray) -> list[tuple[int, int, int]]:
    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError("active_features must have shape (N, 3)")
    return [(int(row[0]), int(row[1]), int(row[2])) for row in features.tolist()]


def _safe_jaccard(
    left: set[tuple[int, int, int]], right: set[tuple[int, int, int]]
) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


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


def _rankdata(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    if values64.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(values64, kind="mergesort")
    ranks = np.empty(values64.size, dtype=np.float64)
    sorted_values = values64[order]
    start = 0
    while start < values64.size:
        end = start + 1
        while end < values64.size and sorted_values[end] == sorted_values[start]:
            end += 1
        # Use average 1-indexed ranks for ties, matching standard Spearman behavior.
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    return _safe_pearson(_rankdata(left), _rankdata(right))


def _quantile_summary(values: np.ndarray) -> dict[str, float] | None:
    values64 = np.asarray(values, dtype=np.float64)
    if values64.size == 0:
        return None
    quantiles = np.quantile(values64, DEFAULT_QUANTILES)
    return {
        f"q{int(q * 100):02d}" if q < 1.0 else "q100": float(value)
        for q, value in zip(DEFAULT_QUANTILES, quantiles)
    }


def _top_labels_by_score(
    labels: list[tuple[int, int, int]],
    scores: np.ndarray,
    k: int,
    *,
    largest_abs: bool = True,
) -> set[tuple[int, int, int]]:
    if k <= 0 or not labels:
        return set()
    score_values = (
        np.abs(scores) if largest_abs else np.asarray(scores, dtype=np.float64)
    )
    k_eff = min(k, len(labels))
    order = np.argsort(-score_values, kind="mergesort")[:k_eff]
    return {labels[int(idx)] for idx in order.tolist()}


def _topk_overlap_summary(
    left_labels: list[tuple[int, int, int]],
    left_scores: np.ndarray,
    right_labels: list[tuple[int, int, int]],
    right_scores: np.ndarray,
    *,
    ks: tuple[int, ...] = DEFAULT_TOP_KS,
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for k in ks:
        k_eff = min(k, len(left_labels), len(right_labels))
        if k_eff <= 0:
            result[str(k)] = {
                "k_effective": 0,
                "left_count": 0,
                "right_count": 0,
                "shared_count": 0,
                "overlap_fraction_of_k": None,
                "jaccard": None,
            }
            continue
        left_top = _top_labels_by_score(left_labels, left_scores, k_eff)
        right_top = _top_labels_by_score(right_labels, right_scores, k_eff)
        shared = left_top & right_top
        result[str(k)] = {
            "k_effective": k_eff,
            "left_count": len(left_top),
            "right_count": len(right_top),
            "shared_count": len(shared),
            "overlap_fraction_of_k": len(shared) / k_eff,
            "jaccard": _safe_jaccard(left_top, right_top),
        }
    return result


def _rank_map(
    labels: list[tuple[int, int, int]], scores: np.ndarray
) -> dict[tuple[int, int, int], int]:
    if not labels:
        return {}
    order = np.argsort(-np.abs(np.asarray(scores, dtype=np.float64)), kind="mergesort")
    return {labels[int(idx)]: rank for rank, idx in enumerate(order.tolist(), start=1)}


def _aligned_shared_values(
    left_labels: list[tuple[int, int, int]],
    left_values: np.ndarray,
    right_labels: list[tuple[int, int, int]],
    right_values: np.ndarray,
    shared_support: set[tuple[int, int, int]],
) -> tuple[list[tuple[int, int, int]], np.ndarray, np.ndarray]:
    left_by_label = {
        label: float(left_values[idx]) for idx, label in enumerate(left_labels)
    }
    right_by_label = {
        label: float(right_values[idx]) for idx, label in enumerate(right_labels)
    }
    shared_labels = sorted(shared_support)
    return (
        shared_labels,
        np.asarray([left_by_label[label] for label in shared_labels], dtype=np.float64),
        np.asarray(
            [right_by_label[label] for label in shared_labels], dtype=np.float64
        ),
    )


def _shared_score_stability(
    left_labels: list[tuple[int, int, int]],
    left_values: np.ndarray,
    right_labels: list[tuple[int, int, int]],
    right_values: np.ndarray,
    shared_support: set[tuple[int, int, int]],
) -> dict[str, Any]:
    shared_labels, left_shared, right_shared = _aligned_shared_values(
        left_labels, left_values, right_labels, right_values, shared_support
    )
    abs_delta = np.abs(left_shared - right_shared)
    relative_denominator = np.maximum(
        np.maximum(np.abs(left_shared), np.abs(right_shared)),
        1e-12,
    )
    relative_delta = abs_delta / relative_denominator if abs_delta.size else abs_delta

    left_ranks = _rank_map(left_labels, left_values)
    right_ranks = _rank_map(right_labels, right_values)
    rank_deltas = np.asarray(
        [abs(left_ranks[label] - right_ranks[label]) for label in shared_labels],
        dtype=np.float64,
    )
    sign_agreement = None
    if shared_labels:
        left_signs = np.sign(left_shared)
        right_signs = np.sign(right_shared)
        sign_agreement = float(np.mean(left_signs == right_signs))

    return {
        "shared_count": len(shared_labels),
        "pearson": _safe_pearson(left_shared, right_shared),
        "spearman": _safe_spearman(left_shared, right_shared),
        "sign_agreement_fraction": sign_agreement,
        "abs_delta_quantiles": _quantile_summary(abs_delta),
        "relative_delta_quantiles": _quantile_summary(relative_delta),
        "abs_rank_delta_quantiles": _quantile_summary(rank_deltas),
        "topk_overlap": _topk_overlap_summary(
            left_labels, left_values, right_labels, right_values
        ),
    }


def _resolve_seed_bundle_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"Phase-3 seed bundle path does not exist: {resolved}")
    matches = sorted(resolved.glob("step_*_phase3_seed_bundle.npz"))
    if not matches:
        raise FileNotFoundError(f"No phase3 seed bundle found under: {resolved}")
    if len(matches) > 1:
        raise ValueError(
            f"Expected exactly one phase3 seed bundle under {resolved}, found {len(matches)}"
        )
    return matches[0]


def load_phase3_seed_bundle(path: Path) -> dict[str, Any]:
    bundle_path = _resolve_seed_bundle_path(path)
    payload = np.load(bundle_path, allow_pickle=False)
    return {
        "path": str(bundle_path),
        "active_features": payload["active_features"],
        "activation_values": payload["activation_values"],
        "seed_feature_influences": payload["seed_feature_influences"],
        "frontier_pre_locality": payload["frontier_pre_locality"],
        "frontier_post_locality": payload["frontier_post_locality"],
        "queue_size": int(payload["queue_size"]),
        "actual_max_feature_nodes": int(payload["actual_max_feature_nodes"]),
        "total_active_features": int(payload["total_active_features"]),
        "status": str(payload["status"].item()),
        "planner_compute_dtype": str(payload["planner_compute_dtype"].item()),
        "influence_compute_dtype": str(payload["influence_compute_dtype"].item()),
    }


def _frontier_label_set(
    frontier_indices: np.ndarray,
    feature_labels: list[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    result: set[tuple[int, int, int]] = set()
    for raw_idx in frontier_indices.tolist():
        idx = int(raw_idx)
        if 0 <= idx < len(feature_labels):
            result.add(feature_labels[idx])
    return result


def _support_mass_split(
    feature_labels: list[tuple[int, int, int]],
    influences: np.ndarray,
    shared_support: set[tuple[int, int, int]],
) -> dict[str, float | int | None]:
    abs_influences = np.abs(np.asarray(influences, dtype=np.float64))
    if abs_influences.shape[0] != len(feature_labels):
        raise ValueError(
            "seed_feature_influences length must match active_features rows"
        )
    shared_mask = np.asarray(
        [label in shared_support for label in feature_labels], dtype=bool
    )
    total_mass = float(abs_influences.sum())
    shared_mass = float(abs_influences[shared_mask].sum())
    unique_mass = float(abs_influences[~shared_mask].sum())
    return {
        "feature_count": int(len(feature_labels)),
        "shared_feature_count": int(shared_mask.sum()),
        "unique_feature_count": int((~shared_mask).sum()),
        "total_abs_influence_mass": total_mass,
        "shared_abs_influence_mass": shared_mass,
        "unique_abs_influence_mass": unique_mass,
        "shared_abs_influence_fraction": (
            shared_mass / total_mass if total_mass > 0.0 else None
        ),
        "unique_abs_influence_fraction": (
            unique_mass / total_mass if total_mass > 0.0 else None
        ),
    }


def _top_unique_features(
    feature_labels: list[tuple[int, int, int]],
    influences: np.ndarray,
    shared_support: set[tuple[int, int, int]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    abs_influences = np.abs(np.asarray(influences, dtype=np.float64))
    unique_rows = [
        (idx, label, float(influences[idx]), float(abs_influences[idx]))
        for idx, label in enumerate(feature_labels)
        if label not in shared_support
    ]
    unique_rows.sort(key=lambda row: (-row[3], row[1]))
    ranks = _rank_map(feature_labels, influences)
    return [
        {
            "row_index": int(idx),
            "feature": [int(label[0]), int(label[1]), int(label[2])],
            "seed_rank": int(ranks[label]),
            "seed_influence": influence,
            "abs_seed_influence": abs_influence,
        }
        for idx, label, influence, abs_influence in unique_rows[:limit]
    ]


def _frontier_overlap_summary(
    left_frontier: set[tuple[int, int, int]],
    right_frontier: set[tuple[int, int, int]],
    shared_support: set[tuple[int, int, int]],
) -> dict[str, Any]:
    overall = _safe_jaccard(left_frontier, right_frontier)
    shared_only = _safe_jaccard(
        left_frontier & shared_support,
        right_frontier & shared_support,
    )
    improvement = None
    if overall is not None and shared_only is not None:
        improvement = shared_only - overall
    return {
        "left_count": len(left_frontier),
        "right_count": len(right_frontier),
        "shared_count": len(left_frontier & right_frontier),
        "jaccard": overall,
        "shared_support_only_jaccard": shared_only,
        "shared_support_improvement": improvement,
    }


def _numeric_summary_value(summary: dict[str, float | int | None], key: str) -> float:
    value = summary[key]
    return float(value) if value is not None else 0.0


def _frontier_rank_drift_summary(
    left_frontier_indices: np.ndarray,
    right_frontier_indices: np.ndarray,
    left_labels: list[tuple[int, int, int]],
    right_labels: list[tuple[int, int, int]],
    shared_support: set[tuple[int, int, int]],
) -> dict[str, Any]:
    left_ranks = {
        left_labels[int(idx)]: rank
        for rank, idx in enumerate(left_frontier_indices.tolist(), start=1)
        if 0 <= int(idx) < len(left_labels) and left_labels[int(idx)] in shared_support
    }
    right_ranks = {
        right_labels[int(idx)]: rank
        for rank, idx in enumerate(right_frontier_indices.tolist(), start=1)
        if 0 <= int(idx) < len(right_labels)
        and right_labels[int(idx)] in shared_support
    }
    shared_frontier = sorted(set(left_ranks) & set(right_ranks))
    rank_deltas = np.asarray(
        [abs(left_ranks[label] - right_ranks[label]) for label in shared_frontier],
        dtype=np.float64,
    )
    return {
        "shared_frontier_count": len(shared_frontier),
        "left_shared_frontier_count": len(left_ranks),
        "right_shared_frontier_count": len(right_ranks),
        "abs_rank_delta_quantiles": _quantile_summary(rank_deltas),
    }


def _interpret_phase3_causality(summary: dict[str, Any]) -> str:
    post = summary["frontier_post_locality"]
    pre = summary["frontier_pre_locality"]
    left_mass = summary["left_support_mass_split"]["shared_abs_influence_fraction"]
    right_mass = summary["right_support_mass_split"]["shared_abs_influence_fraction"]
    shared_mass_floor = min(
        float(left_mass) if left_mass is not None else 0.0,
        float(right_mass) if right_mass is not None else 0.0,
    )
    overall_post = post["jaccard"]
    shared_post = post["shared_support_only_jaccard"]
    overall_pre = pre["jaccard"]
    shared_pre = pre["shared_support_only_jaccard"]

    if (
        shared_post is not None
        and overall_post is not None
        and shared_post >= 0.95
        and shared_post >= overall_post + 0.25
        and shared_mass_floor >= 0.8
    ):
        return "phase3_mostly_explained_by_phase0_support_split"
    if (
        shared_post is not None
        and overall_post is not None
        and shared_post >= overall_post + 0.1
    ) or (
        shared_pre is not None
        and overall_pre is not None
        and shared_pre >= overall_pre + 0.1
    ):
        return "phase3_partially_explained_by_phase0_support_split"
    return "phase3_mismatch_persists_on_shared_support"


def compare_phase3_seed_bundles(left: Path, right: Path) -> dict[str, Any]:
    left_bundle = load_phase3_seed_bundle(left)
    right_bundle = load_phase3_seed_bundle(right)

    left_labels = _feature_tuples(
        np.asarray(left_bundle["active_features"], dtype=np.int64)
    )
    right_labels = _feature_tuples(
        np.asarray(right_bundle["active_features"], dtype=np.int64)
    )

    left_support = set(left_labels)
    right_support = set(right_labels)
    shared_support = left_support & right_support
    union_support = left_support | right_support

    left_pre = _frontier_label_set(
        np.asarray(left_bundle["frontier_pre_locality"], dtype=np.int64),
        left_labels,
    )
    right_pre = _frontier_label_set(
        np.asarray(right_bundle["frontier_pre_locality"], dtype=np.int64),
        right_labels,
    )
    left_post = _frontier_label_set(
        np.asarray(left_bundle["frontier_post_locality"], dtype=np.int64),
        left_labels,
    )
    right_post = _frontier_label_set(
        np.asarray(right_bundle["frontier_post_locality"], dtype=np.int64),
        right_labels,
    )

    left_mass_split = _support_mass_split(
        left_labels,
        np.asarray(left_bundle["seed_feature_influences"], dtype=np.float64),
        shared_support,
    )
    right_mass_split = _support_mass_split(
        right_labels,
        np.asarray(right_bundle["seed_feature_influences"], dtype=np.float64),
        shared_support,
    )

    activation_stability = _shared_score_stability(
        left_labels,
        np.asarray(left_bundle["activation_values"], dtype=np.float64),
        right_labels,
        np.asarray(right_bundle["activation_values"], dtype=np.float64),
        shared_support,
    )
    seed_influence_stability = _shared_score_stability(
        left_labels,
        np.asarray(left_bundle["seed_feature_influences"], dtype=np.float64),
        right_labels,
        np.asarray(right_bundle["seed_feature_influences"], dtype=np.float64),
        shared_support,
    )

    frontier_pre = _frontier_overlap_summary(left_pre, right_pre, shared_support)
    frontier_pre["rank_drift"] = _frontier_rank_drift_summary(
        np.asarray(left_bundle["frontier_pre_locality"], dtype=np.int64),
        np.asarray(right_bundle["frontier_pre_locality"], dtype=np.int64),
        left_labels,
        right_labels,
        shared_support,
    )
    frontier_post = _frontier_overlap_summary(left_post, right_post, shared_support)
    frontier_post["rank_drift"] = _frontier_rank_drift_summary(
        np.asarray(left_bundle["frontier_post_locality"], dtype=np.int64),
        np.asarray(right_bundle["frontier_post_locality"], dtype=np.int64),
        left_labels,
        right_labels,
        shared_support,
    )

    left_influences = np.asarray(
        left_bundle["seed_feature_influences"], dtype=np.float64
    )
    right_influences = np.asarray(
        right_bundle["seed_feature_influences"], dtype=np.float64
    )

    summary: dict[str, Any] = {
        "left_path": left_bundle["path"],
        "right_path": right_bundle["path"],
        "left_status": left_bundle["status"],
        "right_status": right_bundle["status"],
        "support": {
            "left_feature_count": len(left_support),
            "right_feature_count": len(right_support),
            "shared_feature_count": len(shared_support),
            "left_unique_feature_count": len(left_support - shared_support),
            "right_unique_feature_count": len(right_support - shared_support),
            "feature_jaccard": (
                len(shared_support) / len(union_support) if union_support else None
            ),
        },
        "left_support_mass_split": left_mass_split,
        "right_support_mass_split": right_mass_split,
        "shared_support_score_stability": {
            "activation_values": activation_stability,
            "seed_feature_influences": seed_influence_stability,
        },
        "unique_support_details": {
            "left_top_unique_by_abs_influence": _top_unique_features(
                left_labels, left_influences, shared_support
            ),
            "right_top_unique_by_abs_influence": _top_unique_features(
                right_labels, right_influences, shared_support
            ),
            "left_unique_mass_fraction": left_mass_split[
                "unique_abs_influence_fraction"
            ],
            "right_unique_mass_fraction": right_mass_split[
                "unique_abs_influence_fraction"
            ],
            "left_shared_to_unique_mass_ratio": _safe_ratio(
                _numeric_summary_value(left_mass_split, "shared_abs_influence_mass"),
                _numeric_summary_value(left_mass_split, "unique_abs_influence_mass"),
            ),
            "right_shared_to_unique_mass_ratio": _safe_ratio(
                _numeric_summary_value(right_mass_split, "shared_abs_influence_mass"),
                _numeric_summary_value(right_mass_split, "unique_abs_influence_mass"),
            ),
        },
        "frontier_pre_locality": frontier_pre,
        "frontier_post_locality": frontier_post,
    }
    summary["interpretation"] = _interpret_phase3_causality(summary)
    return summary


def compare_phase3_seed_bundles_to_json(
    left: Path,
    right: Path,
    *,
    output_json: Path | None = None,
) -> dict[str, Any]:
    result = compare_phase3_seed_bundles(left, right)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
