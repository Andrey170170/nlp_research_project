"""Exact and numeric comparison of captured trace diagnostic bundles.

This module deliberately compares *pairs* without assigning control/candidate
roles.  A caller may state the experimental isolation that makes a causal claim
valid; otherwise the report remains a neutral pairwise observation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import ensure_dir, write_json

NEUTRAL_CLAIM_BOUNDARY = (
    "Neutral pairwise diagnostic only: no control/candidate roles or causal "
    "isolation are assumed."
)

BUNDLE_FILES = {
    "phase0_donor": "phase0_donor_bundle.npz",
    "phase3_gradient": "phase3_gradient_bundle.npz",
    "phase3_row": "phase3_row_bundle.npz",
    "phase3_seed": "phase3_seed_bundle.npz",
}

_REQUIRED_ARRAYS = {
    "phase0_donor": (
        "input_tokens",
        "target_token_ids",
        "target_probabilities",
        "target_logits",
        "active_features",
        "activation_values",
    ),
    "phase3_gradient": ("target_token_ids", "gradients"),
    "phase3_row": (
        "target_token_ids",
        "phase3_feature_rows",
        "feature_abs_sums",
        "error_abs_sums",
        "token_abs_sums",
        "row_abs_sums",
    ),
    "phase3_seed": (
        "active_features",
        "activation_values",
        "seed_feature_influences",
        "frontier_pre_locality",
        "frontier_post_locality",
    ),
}


class DiagnosticBundleError(ValueError):
    """Raised when a token directory cannot support a sound comparison."""


def _load_required_bundles(token_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    token_dir = Path(token_dir)
    missing_files = [
        filename
        for filename in BUNDLE_FILES.values()
        if not (token_dir / filename).is_file()
    ]
    if missing_files:
        raise DiagnosticBundleError(
            f"token directory {token_dir} is missing required diagnostic bundles: "
            + ", ".join(missing_files)
        )

    bundles: dict[str, dict[str, np.ndarray]] = {}
    missing_arrays: list[str] = []
    for bundle_name, filename in BUNDLE_FILES.items():
        path = token_dir / filename
        try:
            with np.load(path, allow_pickle=False) as archive:
                bundle = {key: np.asarray(archive[key]) for key in archive.files}
        except (OSError, ValueError) as error:
            raise DiagnosticBundleError(
                f"could not safely load diagnostic bundle {path}: {error}"
            ) from error
        bundles[bundle_name] = bundle
        missing_arrays.extend(
            f"{filename}:{key}"
            for key in _REQUIRED_ARRAYS[bundle_name]
            if key not in bundle
        )
    if missing_arrays:
        raise DiagnosticBundleError(
            "required diagnostic arrays are missing: " + ", ".join(missing_arrays)
        )
    return bundles


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _value_exact_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    if left.dtype.kind in "fc" and right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _nonfinite_kind(values: np.ndarray) -> np.ndarray:
    kinds = np.zeros(values.shape, dtype=np.int8)
    kinds[np.isnan(values)] = 1
    kinds[np.isposinf(values)] = 2
    kinds[np.isneginf(values)] = 3
    return kinds


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator > 0.0:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else None


def _numeric_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any] | None:
    if left.shape != right.shape:
        return None
    if not (
        np.issubdtype(left.dtype, np.number)
        and np.issubdtype(right.dtype, np.number)
        and not np.issubdtype(left.dtype, np.complexfloating)
        and not np.issubdtype(right.dtype, np.complexfloating)
    ):
        return None

    left64 = left.astype(np.float64, copy=False)
    right64 = right.astype(np.float64, copy=False)
    finite = np.isfinite(left64) & np.isfinite(right64)
    left_nonfinite = ~np.isfinite(left64)
    right_nonfinite = ~np.isfinite(right64)
    nonfinite_mismatch = int(
        np.count_nonzero(
            (left_nonfinite != right_nonfinite)
            | (
                (left_nonfinite & right_nonfinite)
                & (_nonfinite_kind(left64) != _nonfinite_kind(right64))
            )
        )
    )

    left_finite = left64[finite]
    right_finite = right64[finite]
    error = np.abs(left_finite - right_finite)
    l1_error = float(error.sum(dtype=np.float64))
    left_l1 = float(np.abs(left_finite).sum(dtype=np.float64))
    right_l1 = float(np.abs(right_finite).sum(dtype=np.float64))
    joint_max = max(
        float(np.max(np.abs(left_finite))) if left_finite.size else 0.0,
        float(np.max(np.abs(right_finite))) if right_finite.size else 0.0,
    )
    max_error = float(error.max()) if error.size else 0.0
    if error.size:
        flat_finite_indices = np.flatnonzero(finite)
        max_flat_index = int(flat_finite_indices[int(np.argmax(error))])
        max_index = [
            int(index) for index in np.unravel_index(max_flat_index, left.shape)
        ]
        max_left = float(left64.flat[max_flat_index])
        max_right = float(right64.flat[max_flat_index])
    else:
        max_index = None
        max_left = None
        max_right = None

    left_scale = float(np.max(np.abs(left_finite))) if left_finite.size else 0.0
    right_scale = float(np.max(np.abs(right_finite))) if right_finite.size else 0.0
    if left_scale > 0.0 and right_scale > 0.0:
        left_scaled = left_finite / left_scale
        right_scaled = right_finite / right_scale
        cosine = float(
            np.dot(left_scaled, right_scaled)
            / (np.linalg.norm(left_scaled) * np.linalg.norm(right_scaled))
        )
        cosine = max(-1.0, min(1.0, cosine))
    elif left_scale == 0.0 and right_scale == 0.0:
        cosine = 1.0
    else:
        cosine = None

    return {
        "element_count": int(left.size),
        "finite_pair_count": int(np.count_nonzero(finite)),
        "nonfinite_pair_count": int(np.count_nonzero(left_nonfinite & right_nonfinite)),
        "nonfinite_mismatch_count": nonfinite_mismatch,
        "allclose": bool(
            np.allclose(left64, right64, rtol=rtol, atol=atol, equal_nan=True)
        ),
        "rtol": rtol,
        "atol": atol,
        "l1_error": l1_error,
        "mean_absolute_error": float(error.mean()) if error.size else 0.0,
        "root_mean_square_error": (
            max_error * float(np.sqrt(np.mean(np.square(error / max_error))))
            if max_error > 0.0
            else 0.0
        ),
        "max_absolute_error": max_error,
        "max_error_index": max_index,
        "max_error_left_value": max_left,
        "max_error_right_value": max_right,
        "left_reference_normalized_l1": _safe_ratio(l1_error, left_l1),
        "symmetric_normalized_l1": _safe_ratio(2.0 * l1_error, left_l1 + right_l1),
        "joint_max_normalized_linf": _safe_ratio(max_error, joint_max),
        "cosine_similarity": cosine,
    }


def _compare_array(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left_hash = _array_sha256(left)
    right_hash = _array_sha256(right)
    return {
        "left": {
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "sha256": left_hash,
        },
        "right": {
            "shape": list(right.shape),
            "dtype": str(right.dtype),
            "sha256": right_hash,
        },
        "shape_equal": left.shape == right.shape,
        "dtype_equal": left.dtype == right.dtype,
        "raw_exact_equal": left_hash == right_hash,
        "value_exact_equal": _value_exact_equal(left, right),
        "numeric": _numeric_metrics(left, right, rtol=rtol, atol=atol),
    }


def _feature_support_metrics(
    left: np.ndarray, right: np.ndarray
) -> dict[str, Any] | None:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != right.shape[1:]:
        return None
    left_keys = {tuple(int(item) for item in row) for row in left}
    right_keys = {tuple(int(item) for item in row) for row in right}
    union = left_keys | right_keys
    return {
        "left_count": len(left_keys),
        "right_count": len(right_keys),
        "shared_count": len(left_keys & right_keys),
        "left_unique_count": len(left_keys - right_keys),
        "right_unique_count": len(right_keys - left_keys),
        "jaccard": 1.0 if not union else len(left_keys & right_keys) / len(union),
    }


def _array_changed(
    comparisons: dict[str, dict[str, dict[str, Any]]],
    bundle: str,
    key: str,
) -> bool:
    comparison = comparisons[bundle][key]
    return not comparison["raw_exact_equal"]


def _earliest_divergence(
    comparisons: dict[str, dict[str, dict[str, Any]]],
    left_bundles: dict[str, dict[str, np.ndarray]],
    right_bundles: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    checkpoints: list[tuple[str, str, str]] = [
        ("alignment_mismatch", "phase0_donor", "input_tokens"),
        ("alignment_mismatch", "phase0_donor", "target_token_ids"),
        ("phase0_active_feature_support", "phase0_donor", "active_features"),
    ]
    left_raw = left_bundles["phase0_donor"].get("activation_values_raw_uint16")
    right_raw = right_bundles["phase0_donor"].get("activation_values_raw_uint16")
    if (
        left_raw is not None
        and right_raw is not None
        and left_raw.size > 0
        and right_raw.size > 0
    ):
        checkpoints.append(
            (
                "phase0_activation_values",
                "phase0_donor",
                "activation_values_raw_uint16",
            )
        )
    checkpoints.extend(
        [
            ("phase0_activation_values", "phase0_donor", "activation_values"),
            ("phase1_target_state", "phase0_donor", "target_logits"),
            ("phase1_target_state", "phase0_donor", "target_probabilities"),
        ]
    )
    for key in ("layer_mask", "batch_call_indices"):
        if key in comparisons["phase3_gradient"]:
            checkpoints.append(("phase3_gradient_layout", "phase3_gradient", key))
    checkpoints.extend(
        [
            ("phase3_gradients", "phase3_gradient", "gradients"),
            ("phase3_feature_rows", "phase3_row", "phase3_feature_rows"),
        ]
    )
    for classification, key in (
        (
            "phase3_error_column_signed_rows_by_layer",
            "phase3_error_rows_by_layer",
        ),
        ("phase3_token_column_signed_rows", "phase3_token_rows"),
    ):
        if key in comparisons["phase3_row"]:
            checkpoints.append((classification, "phase3_row", key))
    checkpoints.extend(
        [
            (
                "phase3_feature_column_l1_contribution",
                "phase3_row",
                "feature_abs_sums",
            ),
            (
                "phase3_error_column_l1_contribution",
                "phase3_row",
                "error_abs_sums",
            ),
            (
                "phase3_token_column_l1_contribution",
                "phase3_row",
                "token_abs_sums",
            ),
            (
                "phase3_total_row_l1_denominator",
                "phase3_row",
                "row_abs_sums",
            ),
            ("phase3_seed_support", "phase3_seed", "active_features"),
            (
                "phase3_seed_activation_values",
                "phase3_seed",
                "activation_values",
            ),
            (
                "phase3_seed_influences",
                "phase3_seed",
                "seed_feature_influences",
            ),
            (
                "phase3_frontier_pre_locality",
                "phase3_seed",
                "frontier_pre_locality",
            ),
            (
                "phase3_frontier_post_locality",
                "phase3_seed",
                "frontier_post_locality",
            ),
        ]
    )
    for classification, bundle, key in checkpoints:
        if _array_changed(comparisons, bundle, key):
            return {
                "classification": classification,
                "bundle": bundle,
                "array": key,
                "basis": "first raw-exact mismatch in declared checkpoint order",
            }

    for bundle_name in BUNDLE_FILES:
        for key in sorted(comparisons[bundle_name]):
            if _array_changed(comparisons, bundle_name, key):
                return {
                    "classification": "diagnostic_metadata_only",
                    "bundle": bundle_name,
                    "array": key,
                    "basis": "primary captured tensors match; ancillary data differs",
                }
        left_only = set(left_bundles[bundle_name]) - set(right_bundles[bundle_name])
        right_only = set(right_bundles[bundle_name]) - set(left_bundles[bundle_name])
        if left_only or right_only:
            return {
                "classification": "diagnostic_metadata_only",
                "bundle": bundle_name,
                "array": min(left_only | right_only),
                "basis": "primary captured tensors match; array presence differs",
            }
    return {
        "classification": "none_exact",
        "bundle": None,
        "array": None,
        "basis": "all arrays in all required diagnostic bundles are raw-exact",
    }


def compare_diagnostic_bundle_dirs(
    left_token_dir: Path,
    right_token_dir: Path,
    *,
    claim_boundary: str | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Compare the four required diagnostic bundles in two token directories."""

    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")
    left_token_dir = Path(left_token_dir)
    right_token_dir = Path(right_token_dir)
    left_bundles = _load_required_bundles(left_token_dir)
    right_bundles = _load_required_bundles(right_token_dir)

    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    bundle_reports: dict[str, Any] = {}
    for bundle_name, filename in BUNDLE_FILES.items():
        left = left_bundles[bundle_name]
        right = right_bundles[bundle_name]
        left_keys = set(left)
        right_keys = set(right)
        common_keys = sorted(left_keys & right_keys)
        array_comparisons = {
            key: _compare_array(left[key], right[key], rtol=rtol, atol=atol)
            for key in common_keys
        }
        comparisons[bundle_name] = array_comparisons
        if "active_features" in left and "active_features" in right:
            support = _feature_support_metrics(
                left["active_features"], right["active_features"]
            )
        else:
            support = None
        left_path = left_token_dir / filename
        right_path = right_token_dir / filename
        left_file_hash = _file_sha256(left_path)
        right_file_hash = _file_sha256(right_path)
        bundle_reports[bundle_name] = {
            "filename": filename,
            "left_file_sha256": left_file_hash,
            "right_file_sha256": right_file_hash,
            "file_byte_exact": left_file_hash == right_file_hash,
            "left_only_arrays": sorted(left_keys - right_keys),
            "right_only_arrays": sorted(right_keys - left_keys),
            "all_array_names_equal": left_keys == right_keys,
            "all_common_arrays_raw_exact": all(
                item["raw_exact_equal"] for item in array_comparisons.values()
            ),
            "feature_support": support,
            "arrays": array_comparisons,
        }

    all_names_equal = all(
        report["all_array_names_equal"] for report in bundle_reports.values()
    )
    all_arrays_exact = all(
        report["all_common_arrays_raw_exact"] for report in bundle_reports.values()
    )
    boundary = (
        claim_boundary.strip() if claim_boundary and claim_boundary.strip() else None
    )
    return {
        "schema_version": 1,
        "analysis_kind": "exact_trace_required_diagnostic_bundle_pair",
        "left_token_dir": str(left_token_dir),
        "right_token_dir": str(right_token_dir),
        "claim_boundary": {
            "source": "caller_supplied" if boundary is not None else "neutral_default",
            "statement": boundary or NEUTRAL_CLAIM_BOUNDARY,
            "limitation": (
                "Earliest means earliest among these persisted checkpoints; it "
                "does not localize unobserved operations between checkpoints."
            ),
        },
        "required_artifacts_complete": True,
        "summary": {
            "all_bundle_array_names_equal": all_names_equal,
            "all_arrays_raw_exact": all_names_equal and all_arrays_exact,
            "all_bundle_files_byte_exact": all(
                report["file_byte_exact"] for report in bundle_reports.values()
            ),
        },
        "earliest_divergence": _earliest_divergence(
            comparisons, left_bundles, right_bundles
        ),
        "bundles": bundle_reports,
    }


def compare_diagnostic_bundle_dirs_to_json(
    left_token_dir: Path,
    right_token_dir: Path,
    *,
    output_json: Path,
    claim_boundary: str | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict[str, Any]:
    result = compare_diagnostic_bundle_dirs(
        left_token_dir,
        right_token_dir,
        claim_boundary=claim_boundary,
        rtol=rtol,
        atol=atol,
    )
    ensure_dir(output_json.parent)
    write_json(output_json, result)
    return result
