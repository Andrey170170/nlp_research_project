from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _feature_tuples(features: np.ndarray) -> list[tuple[int, int, int]]:
    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError("candidate_features must have shape (N, 3)")
    return [(int(row[0]), int(row[1]), int(row[2])) for row in features.tolist()]


def _resolve_descriptor_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"Semantic descriptor path does not exist: {resolved}")
    matches = sorted(resolved.glob("step_*_feature_semantic_descriptors.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No semantic descriptor artifact found under: {resolved}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Expected exactly one semantic descriptor artifact under {resolved}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _scalar_str(payload: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    return str(payload[key].item()) if key in payload else default


def _scalar_int(payload: np.lib.npyio.NpzFile, key: str, default: int = 0) -> int:
    return int(payload[key].item()) if key in payload else default


def load_semantic_descriptors(path: Path) -> dict[str, Any]:
    descriptor_path = _resolve_descriptor_path(path)
    payload = np.load(descriptor_path, allow_pickle=False)
    required = [
        "candidate_features",
        "semantic_sketch",
        "seed_influence",
        "activation_value",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Descriptor artifact missing required arrays: {missing}")
    features = np.asarray(payload["candidate_features"], dtype=np.int64)
    sketches = np.asarray(payload["semantic_sketch"], dtype=np.float64)
    if sketches.ndim != 2 or sketches.shape[0] != features.shape[0]:
        raise ValueError(
            "semantic_sketch must have shape (M, D) aligned to candidate_features"
        )
    return {
        "path": str(descriptor_path),
        "status": _scalar_str(payload, "status"),
        "descriptor_version": _scalar_str(payload, "descriptor_version"),
        "descriptor_kind": _scalar_str(payload, "descriptor_kind"),
        "descriptor_dim": _scalar_int(payload, "descriptor_dim", sketches.shape[1]),
        "semantic_descriptor_top_k": _scalar_int(payload, "semantic_descriptor_top_k"),
        "candidate_count": _scalar_int(payload, "candidate_count", features.shape[0]),
        "total_active_features": _scalar_int(payload, "total_active_features"),
        "candidate_features": features,
        "candidate_labels": _feature_tuples(features),
        "semantic_sketch": sketches,
        "seed_influence": np.asarray(payload["seed_influence"], dtype=np.float64),
        "activation_value": np.asarray(payload["activation_value"], dtype=np.float64),
        "is_selected_phase4": np.asarray(
            payload["is_selected_phase4"] if "is_selected_phase4" in payload else [],
            dtype=bool,
        ),
    }


def _safe_jaccard(
    left: set[tuple[int, int, int]], right: set[tuple[int, int, int]]
) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    denominator = float(
        np.sqrt(
            np.sum(left_centered * left_centered)
            * np.sum(right_centered * right_centered)
        )
    )
    if denominator == 0.0:
        return None
    return float(np.sum(left_centered * right_centered) / denominator)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def _descriptor_rows_by_label(
    bundle: dict[str, Any],
) -> dict[tuple[int, int, int], int]:
    return {label: idx for idx, label in enumerate(bundle["candidate_labels"])}


def _shared_score_summary(
    left: dict[str, Any],
    right: dict[str, Any],
    shared_labels: set[tuple[int, int, int]],
) -> dict[str, Any]:
    left_rows = _descriptor_rows_by_label(left)
    right_rows = _descriptor_rows_by_label(right)
    ordered = sorted(shared_labels)
    left_seed = np.asarray(
        [left["seed_influence"][left_rows[label]] for label in ordered]
    )
    right_seed = np.asarray(
        [right["seed_influence"][right_rows[label]] for label in ordered]
    )
    left_activation = np.asarray(
        [left["activation_value"][left_rows[label]] for label in ordered]
    )
    right_activation = np.asarray(
        [right["activation_value"][right_rows[label]] for label in ordered]
    )
    return {
        "shared_candidate_count": len(ordered),
        "seed_influence_pearson": _safe_pearson(left_seed, right_seed),
        "activation_value_pearson": _safe_pearson(left_activation, right_activation),
        "seed_influence_abs_delta_mean": (
            float(np.mean(np.abs(left_seed - right_seed))) if ordered else None
        ),
        "activation_value_abs_delta_mean": (
            float(np.mean(np.abs(left_activation - right_activation)))
            if ordered
            else None
        ),
    }


def _candidate_record(bundle: dict[str, Any], row_idx: int) -> dict[str, Any]:
    feature = bundle["candidate_labels"][row_idx]
    return {
        "feature": [int(feature[0]), int(feature[1]), int(feature[2])],
        "row_index": int(row_idx),
        "seed_influence": float(bundle["seed_influence"][row_idx]),
        "abs_seed_influence": float(abs(bundle["seed_influence"][row_idx])),
        "activation_value": float(bundle["activation_value"][row_idx]),
    }


def _top_unmatched_records(
    bundle: dict[str, Any], labels: set[tuple[int, int, int]], *, limit: int
) -> list[dict[str, Any]]:
    rows_by_label = _descriptor_rows_by_label(bundle)
    rows = [rows_by_label[label] for label in labels if label in rows_by_label]
    rows.sort(
        key=lambda idx: (
            -abs(float(bundle["seed_influence"][idx])),
            bundle["candidate_labels"][idx],
        )
    )
    return [_candidate_record(bundle, idx) for idx in rows[:limit]]


def _semantic_matches(
    left: dict[str, Any],
    right: dict[str, Any],
    left_only: set[tuple[int, int, int]],
    right_only: set[tuple[int, int, int]],
    *,
    position_window: int,
    similarity_threshold: float,
    limit: int,
) -> list[dict[str, Any]]:
    left_rows = _descriptor_rows_by_label(left)
    right_rows = _descriptor_rows_by_label(right)
    right_candidates_by_layer: dict[int, list[tuple[int, tuple[int, int, int]]]] = {}
    for label in right_only:
        right_candidates_by_layer.setdefault(label[0], []).append(
            (right_rows[label], label)
        )

    matches: list[dict[str, Any]] = []
    ordered_left = sorted(
        [label for label in left_only if label in left_rows],
        key=lambda label: (
            -abs(float(left["seed_influence"][left_rows[label]])),
            label,
        ),
    )
    for left_label in ordered_left:
        left_row = left_rows[left_label]
        best: tuple[float, int, tuple[int, int, int]] | None = None
        for right_row, right_label in right_candidates_by_layer.get(left_label[0], []):
            if abs(right_label[1] - left_label[1]) > position_window:
                continue
            similarity = _cosine_similarity(
                left["semantic_sketch"][left_row], right["semantic_sketch"][right_row]
            )
            if similarity is None:
                continue
            if best is None or similarity > best[0]:
                best = (similarity, right_row, right_label)
        if best is None:
            continue
        similarity, right_row, right_label = best
        is_high_confidence = similarity >= similarity_threshold
        matches.append(
            {
                "left_feature": [
                    int(left_label[0]),
                    int(left_label[1]),
                    int(left_label[2]),
                ],
                "right_feature": [
                    int(right_label[0]),
                    int(right_label[1]),
                    int(right_label[2]),
                ],
                "left_row_index": int(left_row),
                "right_row_index": int(right_row),
                "descriptor_cosine": similarity,
                "high_confidence": bool(is_high_confidence),
                "left_abs_seed_influence": float(abs(left["seed_influence"][left_row])),
                "right_abs_seed_influence": float(
                    abs(right["seed_influence"][right_row])
                ),
                "activation_abs_delta": float(
                    abs(
                        left["activation_value"][left_row]
                        - right["activation_value"][right_row]
                    )
                ),
            }
        )
        if len(matches) >= limit:
            break
    return matches


def _coverage_summary(
    matches: list[dict[str, Any]], total_left_mass: float, total_right_mass: float
) -> dict[str, Any]:
    high_confidence = [match for match in matches if match["high_confidence"]]
    covered_left_mass = sum(
        float(match["left_abs_seed_influence"]) for match in high_confidence
    )
    covered_right_mass = sum(
        float(match["right_abs_seed_influence"]) for match in high_confidence
    )
    return {
        "matched_count": len(matches),
        "high_confidence_count": len(high_confidence),
        "left_high_confidence_mass": covered_left_mass,
        "right_high_confidence_mass": covered_right_mass,
        "left_high_confidence_mass_fraction": (
            covered_left_mass / total_left_mass if total_left_mass > 0.0 else None
        ),
        "right_high_confidence_mass_fraction": (
            covered_right_mass / total_right_mass if total_right_mass > 0.0 else None
        ),
    }


def _interpret(summary: dict[str, Any]) -> str:
    descriptor_kind = summary["descriptor_metadata"]["left_descriptor_kind"]
    if not descriptor_kind:
        return "insufficient_descriptor_coverage"
    support = summary["candidate_support"]
    if support["feature_jaccard"] is not None and support["feature_jaccard"] >= 0.95:
        return "exact_id_stable"
    shared_scores = summary["shared_candidate_scores"]
    if (
        shared_scores["seed_influence_pearson"] is not None
        and shared_scores["seed_influence_pearson"] < 0.5
    ):
        return "shared_support_scores_unstable"
    coverage = summary["semantic_substitute_coverage"]
    left_fraction = coverage["left_high_confidence_mass_fraction"] or 0.0
    right_fraction = coverage["right_high_confidence_mass_fraction"] or 0.0
    if min(left_fraction, right_fraction) >= 0.8:
        return "semantic_substitutes_explain_mismatch"
    return "unique_features_semantically_unmatched"


def compare_semantic_feature_descriptors(
    left_descriptor: Path,
    right_descriptor: Path,
    *,
    position_window: int = 0,
    similarity_threshold: float = 0.95,
    top_unmatched_limit: int = 20,
    match_limit: int = 200,
) -> dict[str, Any]:
    left = load_semantic_descriptors(left_descriptor)
    right = load_semantic_descriptors(right_descriptor)
    left_support = set(left["candidate_labels"])
    right_support = set(right["candidate_labels"])
    shared = left_support & right_support
    left_only = left_support - shared
    right_only = right_support - shared
    left_only_mass = sum(
        abs(float(left["seed_influence"][_descriptor_rows_by_label(left)[label]]))
        for label in left_only
    )
    right_only_mass = sum(
        abs(float(right["seed_influence"][_descriptor_rows_by_label(right)[label]]))
        for label in right_only
    )

    matches = _semantic_matches(
        left,
        right,
        left_only,
        right_only,
        position_window=position_window,
        similarity_threshold=similarity_threshold,
        limit=match_limit,
    )
    summary: dict[str, Any] = {
        "left_path": left["path"],
        "right_path": right["path"],
        "descriptor_metadata": {
            "left_descriptor_kind": left["descriptor_kind"],
            "right_descriptor_kind": right["descriptor_kind"],
            "left_descriptor_dim": left["descriptor_dim"],
            "right_descriptor_dim": right["descriptor_dim"],
        },
        "candidate_support": {
            "left_candidate_count": len(left_support),
            "right_candidate_count": len(right_support),
            "shared_candidate_count": len(shared),
            "left_only_candidate_count": len(left_only),
            "right_only_candidate_count": len(right_only),
            "feature_jaccard": _safe_jaccard(left_support, right_support),
        },
        "shared_candidate_scores": _shared_score_summary(left, right, shared),
        "unmatched_candidates": {
            "left_top_unmatched_by_abs_influence": _top_unmatched_records(
                left, left_only, limit=top_unmatched_limit
            ),
            "right_top_unmatched_by_abs_influence": _top_unmatched_records(
                right, right_only, limit=top_unmatched_limit
            ),
            "left_unmatched_abs_seed_influence_mass": float(left_only_mass),
            "right_unmatched_abs_seed_influence_mass": float(right_only_mass),
        },
        "semantic_matches": matches,
        "semantic_substitute_coverage": _coverage_summary(
            matches, left_only_mass, right_only_mass
        ),
        "matching_parameters": {
            "position_window": int(position_window),
            "similarity_threshold": float(similarity_threshold),
            "top_unmatched_limit": int(top_unmatched_limit),
            "match_limit": int(match_limit),
        },
    }
    summary["interpretation"] = _interpret(summary)
    return summary


def compare_semantic_feature_descriptors_to_json(
    left_descriptor: Path,
    right_descriptor: Path,
    *,
    output_json: Path | None = None,
    position_window: int = 0,
    similarity_threshold: float = 0.95,
) -> dict[str, Any]:
    result = compare_semantic_feature_descriptors(
        left_descriptor,
        right_descriptor,
        position_window=position_window,
        similarity_threshold=similarity_threshold,
    )
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
