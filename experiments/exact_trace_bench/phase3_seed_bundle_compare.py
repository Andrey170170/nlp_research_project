from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


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
        "frontier_pre_locality": {
            "left_count": len(left_pre),
            "right_count": len(right_pre),
            "jaccard": _safe_jaccard(left_pre, right_pre),
            "shared_support_only_jaccard": _safe_jaccard(
                left_pre & shared_support,
                right_pre & shared_support,
            ),
        },
        "frontier_post_locality": {
            "left_count": len(left_post),
            "right_count": len(right_post),
            "jaccard": _safe_jaccard(left_post, right_post),
            "shared_support_only_jaccard": _safe_jaccard(
                left_post & shared_support,
                right_post & shared_support,
            ),
        },
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
