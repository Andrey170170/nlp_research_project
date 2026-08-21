from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..graph_compare import compare_compact_paths
from ..io_utils import ensure_dir, write_json

DEFAULT_TOP_KS = (128, 512, 1024, 4096, 8192, 16384)
DEFAULT_NEAR_CUTOFF_FRACTIONS = (0.001, 0.01, 0.05)
DEFAULT_QUANTILES = (0.0, 0.5, 0.9, 0.99, 1.0)


def _scalar(value: np.ndarray) -> Any:
    item = value.item() if getattr(value, "shape", ()) == () else value
    if isinstance(item, np.generic):
        return item.item()
    return item


def _row_keys(rows: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).reshape(
        -1
    )


def _jaccard_from_keys(left: np.ndarray, right: np.ndarray) -> float:
    left_unique = np.unique(left)
    right_unique = np.unique(right)
    union = np.union1d(left_unique, right_unique).size
    if union == 0:
        return float("nan")
    return float(
        np.intersect1d(left_unique, right_unique, assume_unique=True).size / union
    )


def _intersect_row_indices(
    left_rows: np.ndarray, right_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    left_keys = _row_keys(left_rows)
    right_keys = _row_keys(right_rows)
    _common, left_idx, right_idx = np.intersect1d(
        left_keys,
        right_keys,
        assume_unique=False,
        return_indices=True,
    )
    return left_idx.astype(np.int64), right_idx.astype(np.int64)


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left_centered = left64 - float(left64.mean())
    right_centered = right64 - float(right64.mean())
    denom = float(
        np.sqrt(
            np.sum(left_centered * left_centered)
            * np.sum(right_centered * right_centered)
        )
    )
    if denom == 0.0:
        return None
    return float(np.sum(left_centered * right_centered) / denom)


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    values64 = np.asarray(values, dtype=np.float64)
    if values64.size == 0:
        return None
    quantiles = np.quantile(values64, DEFAULT_QUANTILES)
    return {
        f"q{int(q * 100):02d}" if q < 1.0 else "q100": float(value)
        for q, value in zip(DEFAULT_QUANTILES, quantiles)
    }


def _value_delta_summary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    abs_delta = np.abs(left64 - right64)
    denom = np.maximum(np.maximum(np.abs(left64), np.abs(right64)), 1e-12)
    return {
        "count": int(left64.size),
        "pearson": _safe_pearson(left64, right64),
        "abs_delta_quantiles": _quantiles(abs_delta),
        "relative_delta_quantiles": _quantiles(abs_delta / denom),
    }


def _histogram(values: np.ndarray, *, minlength: int | None = None) -> list[int]:
    if values.size == 0:
        return [] if minlength is None else [0] * int(minlength)
    effective_minlength = int(values.max()) + 1 if minlength is None else int(minlength)
    return [
        int(value)
        for value in np.bincount(values.astype(np.int64), minlength=effective_minlength)
    ]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(int(k), int(values.size))
    if k_eff <= 0:
        return np.empty(0, dtype=np.int64)
    if k_eff == values.size:
        return np.argsort(values, kind="stable")[::-1]
    candidate = np.argpartition(values, -k_eff)[-k_eff:]
    return candidate[np.argsort(values[candidate], kind="stable")[::-1]]


def _topk_overlap(
    left_rows: np.ndarray,
    left_scores: np.ndarray,
    right_rows: np.ndarray,
    right_scores: np.ndarray,
    *,
    ks: tuple[int, ...],
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for k in ks:
        left_idx = _topk_indices(left_scores, k)
        right_idx = _topk_indices(right_scores, k)
        k_eff = min(int(left_idx.size), int(right_idx.size))
        if k_eff == 0:
            result[str(k)] = {
                "k_effective": 0,
                "shared_count": 0,
                "overlap_fraction_of_k": None,
                "jaccard": None,
            }
            continue
        left_keys = _row_keys(left_rows[left_idx[:k_eff]])
        right_keys = _row_keys(right_rows[right_idx[:k_eff]])
        shared = np.intersect1d(
            np.unique(left_keys), np.unique(right_keys), assume_unique=True
        ).size
        union = np.union1d(np.unique(left_keys), np.unique(right_keys)).size
        result[str(k)] = {
            "k_effective": int(k_eff),
            "shared_count": int(shared),
            "overlap_fraction_of_k": float(shared / k_eff),
            "jaccard": float(shared / union) if union else None,
        }
    return result


def _cutoff_summary(scores: np.ndarray, k: int) -> dict[str, Any]:
    scores64 = np.asarray(scores, dtype=np.float64)
    if scores64.size == 0:
        return {"k_effective": 0}
    ordered = np.sort(scores64)[::-1]
    k_eff = min(int(k), int(ordered.size))
    cutoff = float(ordered[k_eff - 1])
    next_score = float(ordered[k_eff]) if k_eff < ordered.size else None
    gap = None if next_score is None else float(cutoff - next_score)
    abs_cutoff = max(abs(cutoff), 1e-12)
    return {
        "k_effective": int(k_eff),
        "cutoff_score": cutoff,
        "next_score": next_score,
        "cutoff_gap": gap,
        "relative_cutoff_gap": None if gap is None else float(gap / abs_cutoff),
        "near_cutoff_counts": {
            str(frac): int(
                np.sum(np.abs(scores64 - cutoff) <= abs_cutoff * float(frac))
            )
            for frac in DEFAULT_NEAR_CUTOFF_FRACTIONS
        },
    }


def _phase0_summary(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, Any]:
    left_features = np.asarray(left["active_features"], dtype=np.int64)
    right_features = np.asarray(right["active_features"], dtype=np.int64)
    left_idx, right_idx = _intersect_row_indices(left_features, right_features)
    left_keys = _row_keys(left_features)
    right_keys = _row_keys(right_features)
    shared = int(left_idx.size)
    left_count = int(left_features.shape[0])
    right_count = int(right_features.shape[0])
    layer_minlength = (
        max(int(left_features[:, 0].max()), int(right_features[:, 0].max())) + 1
    )
    position_minlength = (
        max(int(left_features[:, 1].max()), int(right_features[:, 1].max())) + 1
    )
    return {
        "left_active_feature_count": left_count,
        "right_active_feature_count": right_count,
        "shared_active_feature_count": shared,
        "left_only_active_feature_count": left_count - shared,
        "right_only_active_feature_count": right_count - shared,
        "active_feature_jaccard": _jaccard_from_keys(left_keys, right_keys),
        "membership_hash_equal": bool(
            _scalar(
                left.get("active_feature_membership_hash_canonical", np.asarray(None))
            )
            == _scalar(
                right.get("active_feature_membership_hash_canonical", np.asarray(None))
            )
        ),
        "value_hash_equal": bool(
            _scalar(left.get("active_feature_values_hash", np.asarray(None)))
            == _scalar(right.get("active_feature_values_hash", np.asarray(None)))
        ),
        "layer_counts_left": _histogram(left_features[:, 0], minlength=layer_minlength),
        "layer_counts_right": _histogram(
            right_features[:, 0], minlength=layer_minlength
        ),
        "position_counts_left": _histogram(
            left_features[:, 1], minlength=position_minlength
        ),
        "position_counts_right": _histogram(
            right_features[:, 1], minlength=position_minlength
        ),
        "shared_activation_value_delta": _value_delta_summary(
            np.asarray(left["activation_values"])[left_idx],
            np.asarray(right["activation_values"])[right_idx],
        ),
        "target_logit_hash_equal": bool(
            _scalar(left.get("target_logit_hash", np.asarray(None)))
            == _scalar(right.get("target_logit_hash", np.asarray(None)))
        ),
    }


def _frontier_overlap(
    left_bundle: dict[str, np.ndarray], right_bundle: dict[str, np.ndarray], key: str
) -> dict[str, Any] | None:
    if key not in left_bundle or key not in right_bundle:
        return None
    left_active = np.asarray(left_bundle["active_features"], dtype=np.int64)
    right_active = np.asarray(right_bundle["active_features"], dtype=np.int64)
    left_frontier = np.asarray(left_bundle[key], dtype=np.int64)
    right_frontier = np.asarray(right_bundle[key], dtype=np.int64)
    left_rows = left_active[left_frontier]
    right_rows = right_active[right_frontier]
    left_keys = _row_keys(left_rows)
    right_keys = _row_keys(right_rows)
    shared = int(
        np.intersect1d(
            np.unique(left_keys), np.unique(right_keys), assume_unique=True
        ).size
    )
    union = int(np.union1d(np.unique(left_keys), np.unique(right_keys)).size)
    return {
        "left_count": int(left_frontier.size),
        "right_count": int(right_frontier.size),
        "shared_count": shared,
        "jaccard": float(shared / union) if union else float("nan"),
    }


def _phase3_summary(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, Any]:
    left_features = np.asarray(left["active_features"], dtype=np.int64)
    right_features = np.asarray(right["active_features"], dtype=np.int64)
    left_scores = np.asarray(left["seed_feature_influences"], dtype=np.float64)
    right_scores = np.asarray(right["seed_feature_influences"], dtype=np.float64)
    left_idx, right_idx = _intersect_row_indices(left_features, right_features)
    max_nodes = min(
        int(
            _scalar(left.get("actual_max_feature_nodes", np.asarray(left_scores.size)))
        ),
        int(
            _scalar(
                right.get("actual_max_feature_nodes", np.asarray(right_scores.size))
            )
        ),
    )
    return {
        "left_total_active_features": int(left_features.shape[0]),
        "right_total_active_features": int(right_features.shape[0]),
        "shared_active_features": int(left_idx.size),
        "shared_seed_influence_delta": _value_delta_summary(
            left_scores[left_idx],
            right_scores[right_idx],
        ),
        "topk_overlap": _topk_overlap(
            left_features,
            left_scores,
            right_features,
            right_scores,
            ks=DEFAULT_TOP_KS,
        ),
        "left_cutoff": _cutoff_summary(left_scores, max_nodes),
        "right_cutoff": _cutoff_summary(right_scores, max_nodes),
        "frontier_pre_locality_overlap": _frontier_overlap(
            left, right, "frontier_pre_locality"
        ),
        "frontier_post_locality_overlap": _frontier_overlap(
            left, right, "frontier_post_locality"
        ),
    }


def _graph_summary(
    left_token_dir: Path, right_token_dir: Path
) -> dict[str, Any] | None:
    left_graph = left_token_dir / "graph.npz"
    right_graph = right_token_dir / "graph.npz"
    if not left_graph.exists() or not right_graph.exists():
        return None
    return compare_compact_paths(left_graph, right_graph)


def compare_token_stability(
    left_token_dir: Path,
    right_token_dir: Path,
    *,
    output_json: Path | None = None,
) -> dict[str, Any]:
    left_token_dir = Path(left_token_dir)
    right_token_dir = Path(right_token_dir)
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis_kind": "full_answer_amplification_stability_compare",
        "left_token_dir": str(left_token_dir),
        "right_token_dir": str(right_token_dir),
    }

    left_phase0_path = left_token_dir / "phase0_donor_bundle.npz"
    right_phase0_path = right_token_dir / "phase0_donor_bundle.npz"
    if left_phase0_path.exists() and right_phase0_path.exists():
        result["phase0"] = _phase0_summary(
            _load_npz(left_phase0_path),
            _load_npz(right_phase0_path),
        )

    left_phase3_path = left_token_dir / "phase3_seed_bundle.npz"
    right_phase3_path = right_token_dir / "phase3_seed_bundle.npz"
    if left_phase3_path.exists() and right_phase3_path.exists():
        result["phase3_seed"] = _phase3_summary(
            _load_npz(left_phase3_path),
            _load_npz(right_phase3_path),
        )

    graph = _graph_summary(left_token_dir, right_token_dir)
    if graph is not None:
        result["compact_graph"] = graph

    if output_json is not None:
        ensure_dir(output_json.parent)
        write_json(output_json, result)
    return result
