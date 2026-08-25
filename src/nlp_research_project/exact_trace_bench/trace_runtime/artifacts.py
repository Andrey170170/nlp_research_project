"""Serialization of project-owned exact-trace debug and replay artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ArtifactSaver = Callable[[Mapping[str, Any], Path], None]
ArtifactValidator = Callable[[Mapping[str, np.ndarray]], None]

_SEMANTIC_DESCRIPTOR_TRANSIENT_POLICY_ID = "bounded_seed_frontier_handoff_v1"
_SEMANTIC_DESCRIPTOR_TRANSIENT_MAX_BYTES = 64 * 1024 * 1024
_SEMANTIC_DESCRIPTOR_TRANSIENT_ARRAY_COUNT = 3
_SEMANTIC_DESCRIPTOR_TRANSIENT_RECEIPT_FIELDS = (
    "semantic_descriptor_transient_policy_id",
    "semantic_descriptor_transient_max_bytes",
    "semantic_descriptor_transient_required_bytes",
    "semantic_descriptor_transient_admitted",
    "semantic_descriptor_transient_released",
    "semantic_descriptor_transient_array_count",
)


@dataclass(frozen=True)
class CaptureArtifactCodec:
    """Schema-owned serializer for one diagnostic capture artifact."""

    name: str
    saver: ArtifactSaver
    required_fields: tuple[str, ...]
    nonempty_fields: tuple[str, ...]
    validator: ArtifactValidator


class CaptureArtifactContractError(RuntimeError):
    """A successful trace did not durably persist every requested capture."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        missing = ", ".join(str(name) for name in report.get("missing", []))
        failed = ", ".join(
            str(item.get("name"))
            for item in report.get("failed", [])
            if isinstance(item, Mapping)
        )
        details = []
        if missing:
            details.append(f"missing={missing}")
        if failed:
            details.append(f"failed={failed}")
        super().__init__(
            "trace completed without requested diagnostic capture sidecars"
            + (f": {'; '.join(details)}" if details else "")
        )


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        unsupported = {
            dtype
            for dtype in (
                torch.bfloat16,
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None),
            )
            if dtype is not None
        }
        if tensor.dtype in unsupported:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _optional_str_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray("")
    if isinstance(value, (list, tuple)):
        return np.asarray([str(item) for item in value])
    return np.asarray(str(value))


def save_phase3_seed_bundle(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        active_features=_to_numpy_array(payload.get("active_features", [])),
        activation_values=_to_numpy_array(payload.get("activation_values", [])),
        seed_feature_influences=_to_numpy_array(
            payload.get("seed_feature_influences", [])
        ),
        frontier_pre_locality=_to_numpy_array(payload.get("frontier_pre_locality", [])),
        frontier_post_locality=_to_numpy_array(
            payload.get("frontier_post_locality", [])
        ),
        queue_size=np.array(payload.get("queue_size", 0), dtype=np.int64),
        actual_max_feature_nodes=np.array(
            payload.get("actual_max_feature_nodes", 0), dtype=np.int64
        ),
        total_active_features=np.array(
            payload.get("total_active_features", 0), dtype=np.int64
        ),
        status=np.array(
            "" if payload.get("status") is None else str(payload["status"])
        ),
        planner_compute_dtype=np.array(
            ""
            if payload.get("planner_compute_dtype") is None
            else str(payload["planner_compute_dtype"])
        ),
        influence_compute_dtype=np.array(
            ""
            if payload.get("influence_compute_dtype") is None
            else str(payload["influence_compute_dtype"])
        ),
    )


def _phase3_common(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "schema_version": np.array(payload.get("schema_version", 1), dtype=np.int64),
        "status": _optional_str_array(payload.get("status")),
        "capture_kind": _optional_str_array(payload.get("capture_kind")),
        "target_token_ids": _to_numpy_array(payload.get("target_token_ids", [])),
        "target_probabilities": _to_numpy_array(
            payload.get("target_probabilities", [])
        ),
        "target_token_ids_hash": _optional_str_array(
            payload.get("target_token_ids_hash")
        ),
        "target_probability_hash": _optional_str_array(
            payload.get("target_probability_hash")
        ),
        "active_feature_count": np.array(
            payload.get("active_feature_count", 0), dtype=np.int64
        ),
        "active_features_hash": _optional_str_array(
            payload.get("active_features_hash")
        ),
        "activation_values_hash": _optional_str_array(
            payload.get("activation_values_hash")
        ),
    }


def save_phase3_gradient_bundle(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        **_phase3_common(payload),
        gradient_batch_representation=_optional_str_array(
            payload.get("gradient_batch_representation")
        ),
        canonical_target_width=np.array(
            payload.get("canonical_target_width", 0), dtype=np.int64
        ),
        gradients=_to_numpy_array(payload.get("gradients", [])),
        layer_mask=_to_numpy_array(payload.get("layer_mask", [])),
        batch_call_indices=_to_numpy_array(payload.get("batch_call_indices", [])),
        per_layer_abs_sum=_to_numpy_array(payload.get("per_layer_abs_sum", [])),
        per_layer_max_abs=_to_numpy_array(payload.get("per_layer_max_abs", [])),
        per_layer_nonfinite_count=_to_numpy_array(
            payload.get("per_layer_nonfinite_count", [])
        ),
        per_layer_hashes=_optional_str_array(payload.get("per_layer_hashes", [])),
        gradient_hash=_optional_str_array(payload.get("gradient_hash")),
    )


def save_phase3_row_bundle(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        **_phase3_common(payload),
        phase3_feature_rows=_to_numpy_array(payload.get("phase3_feature_rows", [])),
        row_abs_sums=_to_numpy_array(payload.get("row_abs_sums", [])),
        feature_abs_sums=_to_numpy_array(payload.get("feature_abs_sums", [])),
        error_abs_sums=_to_numpy_array(payload.get("error_abs_sums", [])),
        token_abs_sums=_to_numpy_array(payload.get("token_abs_sums", [])),
        nonfeature_row_layout=_optional_str_array(payload.get("nonfeature_row_layout")),
        phase3_error_rows_by_layer=_to_numpy_array(
            payload.get("phase3_error_rows_by_layer", [])
        ),
        phase3_token_rows=_to_numpy_array(payload.get("phase3_token_rows", [])),
        total_active_features=np.array(
            payload.get("total_active_features", 0), dtype=np.int64
        ),
        error_column_count=np.array(
            payload.get("error_column_count", 0), dtype=np.int64
        ),
        token_column_count=np.array(
            payload.get("token_column_count", 0), dtype=np.int64
        ),
        row_hash=_optional_str_array(payload.get("row_hash")),
        row_abs_sum_hash=_optional_str_array(payload.get("row_abs_sum_hash")),
        error_rows_hash=_optional_str_array(payload.get("error_rows_hash")),
        token_rows_hash=_optional_str_array(payload.get("token_rows_hash")),
    )


def save_feature_semantic_descriptors(payload: Mapping[str, Any], path: Path) -> None:
    transient_fields = sorted(
        str(name) for name in payload if str(name).startswith("_transient_")
    )
    if transient_fields:
        raise ValueError(
            "feature semantic descriptor transients must be finalized before "
            f"persistence: {transient_fields}"
        )
    _validate_semantic_descriptor_transient_receipt(payload)
    array_fields = (
        "candidate_features",
        "candidate_row_indices",
        "activation_value",
        "seed_influence",
        "seed_rank",
        "is_top_seed",
        "is_frontier_pre",
        "frontier_pre_rank",
        "is_frontier_post",
        "frontier_post_rank",
        "is_selected_phase4",
        "phase4_selected_rank",
        "semantic_sketch",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        **{field: _to_numpy_array(payload.get(field, [])) for field in array_fields},
        status=np.array(
            "" if payload.get("status") is None else str(payload["status"])
        ),
        descriptor_version=np.array(
            ""
            if payload.get("descriptor_version") is None
            else str(payload["descriptor_version"])
        ),
        descriptor_kind=np.array(
            ""
            if payload.get("descriptor_kind") is None
            else str(payload["descriptor_kind"])
        ),
        descriptor_dim=np.array(payload.get("descriptor_dim", 0), dtype=np.int64),
        semantic_descriptor_top_k=np.array(
            payload.get("semantic_descriptor_top_k", 0), dtype=np.int64
        ),
        candidate_bound_kind=np.array(
            ""
            if payload.get("candidate_bound_kind") is None
            else str(payload["candidate_bound_kind"])
        ),
        semantic_descriptor_control_limit=np.array(
            payload.get("semantic_descriptor_control_limit", 0), dtype=np.int64
        ),
        candidate_count=np.array(payload.get("candidate_count", 0), dtype=np.int64),
        total_active_features=np.array(
            payload.get("total_active_features", 0), dtype=np.int64
        ),
        phase4_selection_available=np.array(
            bool(payload.get("phase4_selection_available", False))
        ),
        seed_influence_available=np.array(
            bool(payload.get("seed_influence_available", False))
        ),
        semantic_descriptor_transient_policy_id=np.array(
            _text(payload["semantic_descriptor_transient_policy_id"])
        ),
        semantic_descriptor_transient_max_bytes=np.array(
            _strict_int_scalar(
                payload["semantic_descriptor_transient_max_bytes"],
                "semantic_descriptor_transient_max_bytes",
            ),
            dtype=np.int64,
        ),
        semantic_descriptor_transient_required_bytes=np.array(
            _strict_int_scalar(
                payload["semantic_descriptor_transient_required_bytes"],
                "semantic_descriptor_transient_required_bytes",
            ),
            dtype=np.int64,
        ),
        semantic_descriptor_transient_admitted=np.array(
            _strict_bool_scalar(
                payload["semantic_descriptor_transient_admitted"],
                "semantic_descriptor_transient_admitted",
            )
        ),
        semantic_descriptor_transient_released=np.array(
            _strict_bool_scalar(
                payload["semantic_descriptor_transient_released"],
                "semantic_descriptor_transient_released",
            )
        ),
        semantic_descriptor_transient_array_count=np.array(
            _strict_int_scalar(
                payload["semantic_descriptor_transient_array_count"],
                "semantic_descriptor_transient_array_count",
            ),
            dtype=np.int64,
        ),
    )


def _normalize_dtype_name(value: Any) -> str:
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if value is None:
        return ""
    return str(value).replace("torch.", "").strip()


def _to_optional_json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _activation_values(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, str, np.ndarray]:
    value = payload.get("activation_values", [])
    dtype_name = _normalize_dtype_name(payload.get("activation_values_dtype"))
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        resolved_dtype = dtype_name or _normalize_dtype_name(tensor.dtype)
        if tensor.dtype == torch.bfloat16:
            return (
                tensor.to(dtype=torch.float32).numpy(),
                resolved_dtype or "bfloat16",
                tensor.view(torch.uint16).numpy().astype(np.uint16, copy=False),
            )
        return _to_numpy_array(tensor), resolved_dtype, np.empty(0, dtype=np.uint16)
    values = _to_numpy_array(value)
    resolved_dtype = dtype_name or str(values.dtype)
    raw = payload.get("activation_values_raw_uint16")
    if raw is None:
        return values, resolved_dtype, np.empty(0, dtype=np.uint16)
    return values, resolved_dtype, _to_numpy_array(raw).astype(np.uint16, copy=False)


def save_phase0_donor_bundle(payload: Mapping[str, Any], path: Path) -> None:
    activation_values, activation_dtype, activation_raw = _activation_values(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        active_features=_to_numpy_array(payload.get("active_features", [])),
        activation_values=activation_values,
        activation_values_dtype=np.array(activation_dtype),
        activation_values_raw_uint16=activation_raw,
        activation_matrix_shape=_to_numpy_array(
            payload.get("activation_matrix_shape", [])
        ),
        active_feature_count=np.array(
            payload.get("active_feature_count", 0), dtype=np.int64
        ),
        active_feature_membership_hash_raw_order=np.array(
            _to_optional_json_text(
                payload.get("active_feature_membership_hash_raw_order")
            )
        ),
        active_feature_membership_hash_canonical=np.array(
            _to_optional_json_text(
                payload.get("active_feature_membership_hash_canonical")
            )
        ),
        active_feature_values_hash=np.array(
            _to_optional_json_text(payload.get("active_feature_values_hash"))
        ),
        active_feature_layer_counts=_to_numpy_array(
            payload.get("active_feature_layer_counts", [])
        ),
        input_tokens=_to_numpy_array(payload.get("input_tokens", [])),
        input_token_count=np.array(payload.get("input_token_count", 0), dtype=np.int64),
        input_tokens_hash=np.array(
            _to_optional_json_text(payload.get("input_tokens_hash"))
        ),
        target_token_ids=_to_numpy_array(payload.get("target_token_ids", [])),
        target_count=np.array(payload.get("target_count", 0), dtype=np.int64),
        target_token_ids_hash=np.array(
            _to_optional_json_text(payload.get("target_token_ids_hash"))
        ),
        target_probabilities=_to_numpy_array(payload.get("target_probabilities", [])),
        target_probability_hash=np.array(
            _to_optional_json_text(payload.get("target_probability_hash"))
        ),
        target_logits=_to_numpy_array(payload.get("target_logits", [])),
        target_logit_hash=np.array(
            _to_optional_json_text(payload.get("target_logit_hash"))
        ),
        clt_constants_hash=np.array(
            _to_optional_json_text(payload.get("clt_constants_hash"))
        ),
        provenance=np.array(_to_optional_json_text(payload.get("provenance"))),
        prompt_metadata=np.array(
            _to_optional_json_text(payload.get("prompt_metadata"))
        ),
        target_metadata=np.array(
            _to_optional_json_text(payload.get("target_metadata"))
        ),
        schema_version=np.array(int(payload.get("schema_version", 0)), dtype=np.int64),
        replay_kind=np.array(
            "" if payload.get("replay_kind") is None else str(payload["replay_kind"])
        ),
        replayed_effective_state=np.array(
            bool(payload.get("replayed_effective_state", False))
        ),
        phase0_replay_mode=np.array(
            ""
            if payload.get("phase0_replay_mode") is None
            else str(payload["phase0_replay_mode"])
        ),
        status=np.array(
            "" if payload.get("status") is None else str(payload["status"])
        ),
    )


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"expected a scalar, got shape {array.shape}")
    return array.item()


def _text(value: Any) -> str:
    return str(_scalar(value)).strip()


def _int(value: Any) -> int:
    scalar = _scalar(value)
    if isinstance(scalar, bool):
        raise ValueError("expected an integer, got bool")
    return int(scalar)


def _strict_int_scalar(value: Any, label: str) -> int:
    scalar = _scalar(value)
    if isinstance(scalar, bool) or not isinstance(scalar, (int, np.integer)):
        raise TypeError(f"{label} must be an integer scalar")
    return int(scalar)


def _strict_bool_scalar(value: Any, label: str) -> bool:
    scalar = _scalar(value)
    if not isinstance(scalar, (bool, np.bool_)):
        raise TypeError(f"{label} must be a boolean scalar")
    return bool(scalar)


def _blake2s(raw: bytes) -> str:
    return hashlib.blake2s(raw, digest_size=8).hexdigest()


def _hash_index_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype=np.int64)
    return _blake2s(array.tobytes())


def _hash_float_array(value: Any, *, dtype: np.dtype[Any]) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype=dtype)
    return _blake2s(array.tobytes())


def _hash_raw_array(value: Any) -> str:
    return _blake2s(np.ascontiguousarray(np.asarray(value)).view(np.uint8).tobytes())


def _require_hash(
    payload: Mapping[str, np.ndarray],
    field: str,
    computed: str,
) -> None:
    stored = _text(payload[field])
    if not stored:
        raise ValueError(f"missing {field}")
    if stored != computed:
        raise ValueError(
            f"{field} mismatch within capture (stored={stored}, computed={computed})"
        )


def _require_finite(payload: Mapping[str, np.ndarray], *fields: str) -> None:
    for field in fields:
        values = np.asarray(payload[field])
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{field} must be numeric")
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"{field} contains nonfinite values")


def _validate_phase0_capture(payload: Mapping[str, np.ndarray]) -> None:
    if _int(payload["schema_version"]) != 1:
        raise ValueError("schema_version must be 1")
    if _text(payload["replay_kind"]) != "phase0_active_features_v1":
        raise ValueError("replay_kind must be 'phase0_active_features_v1'")
    active_features = np.asarray(payload["active_features"])
    activation_values = np.asarray(payload["activation_values"]).reshape(-1)
    matrix_shape = np.asarray(payload["activation_matrix_shape"]).reshape(-1)
    if active_features.ndim != 2 or active_features.shape[1] != 3:
        raise ValueError("active_features must have shape [N, 3]")
    if matrix_shape.shape != (3,):
        raise ValueError("activation_matrix_shape must have length 3")
    row_count = int(active_features.shape[0])
    if activation_values.size != row_count:
        raise ValueError("activation_values length does not match active_features")
    if _int(payload["active_feature_count"]) != row_count:
        raise ValueError("active_feature_count does not match active_features")

    indices = np.ascontiguousarray(active_features.T, dtype=np.int64)
    shape_bytes = np.ascontiguousarray(matrix_shape, dtype=np.int64).tobytes()
    raw_hasher = hashlib.blake2s(digest_size=8)
    raw_hasher.update(shape_bytes)
    if indices.size:
        raw_hasher.update(indices.tobytes())
    else:
        raw_hasher.update(b"empty")
    _require_hash(
        payload,
        "active_feature_membership_hash_raw_order",
        raw_hasher.hexdigest(),
    )

    canonical_hasher = hashlib.blake2s(digest_size=8)
    canonical_hasher.update(shape_bytes)
    if indices.size:
        _, n_positions, n_features = (int(value) for value in matrix_shape)
        flat = (
            indices[0] * n_positions * n_features + indices[1] * n_features + indices[2]
        )
        canonical_hasher.update(np.sort(flat).astype(np.int64, copy=False).tobytes())
    else:
        canonical_hasher.update(b"empty")
    _require_hash(
        payload,
        "active_feature_membership_hash_canonical",
        canonical_hasher.hexdigest(),
    )

    activation_dtype = _text(payload["activation_values_dtype"]).replace("torch.", "")
    raw_uint16 = np.asarray(payload.get("activation_values_raw_uint16", [])).reshape(-1)
    activation_hash_input: np.ndarray | None = activation_values
    if activation_dtype == "bfloat16":
        activation_hash_input = raw_uint16 if raw_uint16.size == row_count else None
    if activation_hash_input is not None:
        _require_hash(
            payload,
            "active_feature_values_hash",
            _hash_raw_array(activation_hash_input),
        )

    input_tokens = np.asarray(payload["input_tokens"]).reshape(-1)
    target_ids = np.asarray(payload["target_token_ids"]).reshape(-1)
    if _int(payload["input_token_count"]) != input_tokens.size:
        raise ValueError("input_token_count does not match input_tokens")
    if _int(payload["target_count"]) != target_ids.size:
        raise ValueError("target_count does not match target_token_ids")
    _require_hash(payload, "input_tokens_hash", _hash_index_array(input_tokens))
    if target_ids.size:
        _require_hash(
            payload,
            "target_token_ids_hash",
            _hash_index_array(target_ids),
        )


def _validate_phase3_seed_capture(payload: Mapping[str, np.ndarray]) -> None:
    active_features = np.asarray(payload["active_features"])
    activation_values = np.asarray(payload["activation_values"]).reshape(-1)
    influences = np.asarray(payload["seed_feature_influences"]).reshape(-1)
    if active_features.ndim != 2 or active_features.shape[1] != 3:
        raise ValueError("active_features must have shape [N, 3]")
    expected = int(active_features.shape[0])
    if activation_values.size != expected or influences.size != expected:
        raise ValueError(
            "activation_values and seed_feature_influences must match active_features"
        )
    for field in ("frontier_pre_locality", "frontier_post_locality"):
        frontier = np.asarray(payload[field]).reshape(-1)
        if frontier.size and (
            bool((frontier < 0).any()) or bool((frontier >= expected).any())
        ):
            raise ValueError(f"{field} contains an out-of-range feature index")
    _require_finite(payload, "activation_values", "seed_feature_influences")


def _validate_phase3_common_capture(payload: Mapping[str, np.ndarray]) -> int:
    target_ids = np.asarray(payload["target_token_ids"]).reshape(-1)
    probabilities = np.asarray(payload["target_probabilities"]).reshape(-1)
    if target_ids.size == 0 or probabilities.size != target_ids.size:
        raise ValueError(
            "target_probabilities must be nonempty and match target_token_ids"
        )
    _require_finite(payload, "target_probabilities")
    _require_hash(payload, "target_token_ids_hash", _hash_index_array(target_ids))
    _require_hash(
        payload,
        "target_probability_hash",
        _hash_float_array(probabilities, dtype=np.dtype(np.float64)),
    )
    return int(target_ids.size)


def _validate_phase3_gradient_capture(payload: Mapping[str, np.ndarray]) -> None:
    schema_version = _int(payload["schema_version"])
    if schema_version not in {1, 2}:
        raise ValueError("schema_version must be 1 or 2")
    if _text(payload["capture_kind"]) != f"phase3_gradient_bundle_v{schema_version}":
        raise ValueError("capture_kind does not match schema_version")
    target_count = _validate_phase3_common_capture(payload)
    gradients = np.asarray(payload["gradients"])
    if gradients.ndim != 4:
        raise ValueError(
            "gradients must have shape [layers, batch, positions, d_model]"
        )
    if schema_version == 2:
        if (
            _text(payload["gradient_batch_representation"])
            != "canonical_target_width_v1"
        ):
            raise ValueError("unexpected gradient_batch_representation")
        if _int(payload["canonical_target_width"]) != target_count:
            raise ValueError("canonical_target_width does not match target_token_ids")
        if int(gradients.shape[1]) != target_count:
            raise ValueError(
                "gradient batch width does not match canonical target width"
            )
    elif int(gradients.shape[1]) < target_count:
        raise ValueError("gradient batch width is smaller than target count")

    layer_count = int(gradients.shape[0])
    layer_mask = np.asarray(payload["layer_mask"]).reshape(-1)
    abs_sums = np.asarray(payload["per_layer_abs_sum"], dtype=np.float64).reshape(-1)
    max_abs = np.asarray(payload["per_layer_max_abs"], dtype=np.float64).reshape(-1)
    nonfinite = np.asarray(payload["per_layer_nonfinite_count"]).reshape(-1)
    layer_hashes = np.asarray(payload["per_layer_hashes"]).reshape(-1)
    for field, values in (
        ("layer_mask", layer_mask),
        ("per_layer_abs_sum", abs_sums),
        ("per_layer_max_abs", max_abs),
        ("per_layer_nonfinite_count", nonfinite),
        ("per_layer_hashes", layer_hashes),
    ):
        if values.size != layer_count:
            raise ValueError(f"{field} length does not match gradient layer count")
    _require_finite(payload, "gradients", "per_layer_abs_sum", "per_layer_max_abs")
    computed_abs_sums = np.abs(gradients).sum(axis=(1, 2, 3), dtype=np.float64)
    computed_max_abs = np.asarray(
        [float(np.abs(layer).max()) if layer.size else 0.0 for layer in gradients],
        dtype=np.float64,
    )
    computed_nonfinite = (~np.isfinite(gradients)).sum(axis=(1, 2, 3))
    if not np.array_equal(abs_sums, computed_abs_sums):
        raise ValueError("per_layer_abs_sum does not match gradients")
    if not np.array_equal(max_abs, computed_max_abs):
        raise ValueError("per_layer_max_abs does not match gradients")
    if not np.array_equal(nonfinite.astype(np.int64), computed_nonfinite):
        raise ValueError("per_layer_nonfinite_count does not match gradients")
    for index, layer in enumerate(gradients):
        computed = _hash_float_array(layer, dtype=np.dtype(np.float32))
        if str(layer_hashes[index]) != computed:
            raise ValueError(f"per_layer_hashes[{index}] mismatch within capture")
    _require_hash(
        payload,
        "gradient_hash",
        _hash_float_array(gradients, dtype=np.dtype(np.float32)),
    )


def _row_abs_sums(values: np.ndarray) -> np.ndarray:
    return np.abs(values).sum(axis=1, dtype=np.float64)


def _validate_phase3_row_capture(payload: Mapping[str, np.ndarray]) -> None:
    schema_version = _int(payload["schema_version"])
    _validate_phase3_row_payload(payload)
    target_count = _validate_phase3_common_capture(payload)
    feature_rows = np.asarray(payload["phase3_feature_rows"])
    row_abs_sums = np.asarray(payload["row_abs_sums"], dtype=np.float64).reshape(-1)
    feature_abs_sums = np.asarray(
        payload["feature_abs_sums"], dtype=np.float64
    ).reshape(-1)
    error_abs_sums = np.asarray(payload["error_abs_sums"], dtype=np.float64).reshape(-1)
    token_abs_sums = np.asarray(payload["token_abs_sums"], dtype=np.float64).reshape(-1)
    if feature_rows.ndim != 2 or int(feature_rows.shape[0]) != target_count:
        raise ValueError("phase3_feature_rows must have one row per target")
    if int(feature_rows.shape[1]) != _int(payload["total_active_features"]):
        raise ValueError("total_active_features does not match phase3_feature_rows")
    if _int(payload["active_feature_count"]) != int(feature_rows.shape[1]):
        raise ValueError("active_feature_count does not match phase3_feature_rows")
    for field, values in (
        ("row_abs_sums", row_abs_sums),
        ("feature_abs_sums", feature_abs_sums),
        ("error_abs_sums", error_abs_sums),
        ("token_abs_sums", token_abs_sums),
    ):
        if values.size != target_count:
            raise ValueError(f"{field} must have one value per target")
    _require_finite(
        payload,
        "phase3_feature_rows",
        "row_abs_sums",
        "feature_abs_sums",
        "error_abs_sums",
        "token_abs_sums",
    )
    if not np.allclose(
        feature_abs_sums,
        _row_abs_sums(feature_rows),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("feature_abs_sums do not match phase3_feature_rows")
    if not np.allclose(
        row_abs_sums,
        feature_abs_sums + error_abs_sums + token_abs_sums,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("row_abs_sums do not match feature/error/token split sums")
    _require_hash(
        payload,
        "row_hash",
        _hash_float_array(feature_rows, dtype=np.dtype(np.float32)),
    )
    _require_hash(
        payload,
        "row_abs_sum_hash",
        _hash_float_array(row_abs_sums, dtype=np.dtype(np.float64)),
    )

    if schema_version == 2:
        error_rows = np.asarray(payload["phase3_error_rows_by_layer"])
        token_rows = np.asarray(payload["phase3_token_rows"])
        _require_finite(payload, "phase3_error_rows_by_layer", "phase3_token_rows")
        computed_error_abs = _row_abs_sums(error_rows.reshape(target_count, -1))
        computed_token_abs = _row_abs_sums(token_rows)
        if not np.allclose(
            error_abs_sums,
            computed_error_abs,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("error_abs_sums do not match phase3_error_rows_by_layer")
        if not np.allclose(
            token_abs_sums,
            computed_token_abs,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("token_abs_sums do not match phase3_token_rows")
        _require_hash(payload, "error_rows_hash", _hash_raw_array(error_rows))
        _require_hash(payload, "token_rows_hash", _hash_raw_array(token_rows))


def _validate_semantic_descriptor_capture(
    payload: Mapping[str, np.ndarray],
) -> None:
    _validate_semantic_descriptor_transient_receipt(payload)
    candidates = np.asarray(payload["candidate_features"])
    indices = np.asarray(payload["candidate_row_indices"]).reshape(-1)
    sketch = np.asarray(payload["semantic_sketch"])
    if candidates.ndim not in {1, 2}:
        raise ValueError("candidate_features must be rank 1 or 2")
    candidate_count = int(candidates.shape[0]) if candidates.ndim else 0
    if indices.size != candidate_count:
        raise ValueError("candidate_row_indices does not match candidate_features")
    if sketch.ndim != 2 or int(sketch.shape[0]) != candidate_count:
        raise ValueError("semantic_sketch must have one row per candidate")
    _require_finite(payload, "semantic_sketch")


def _validate_semantic_descriptor_transient_receipt(
    payload: Mapping[str, Any],
) -> None:
    missing = [
        field for field in _SEMANTIC_DESCRIPTOR_TRANSIENT_RECEIPT_FIELDS if field not in payload
    ]
    if missing:
        raise ValueError(
            "feature semantic descriptor payload missing bounded transient receipt "
            f"fields: {missing}"
        )
    policy_id = _text(payload["semantic_descriptor_transient_policy_id"])
    max_bytes = _strict_int_scalar(
        payload["semantic_descriptor_transient_max_bytes"],
        "semantic_descriptor_transient_max_bytes",
    )
    required_bytes = _strict_int_scalar(
        payload["semantic_descriptor_transient_required_bytes"],
        "semantic_descriptor_transient_required_bytes",
    )
    admitted = _strict_bool_scalar(
        payload["semantic_descriptor_transient_admitted"],
        "semantic_descriptor_transient_admitted",
    )
    released = _strict_bool_scalar(
        payload["semantic_descriptor_transient_released"],
        "semantic_descriptor_transient_released",
    )
    array_count = _strict_int_scalar(
        payload["semantic_descriptor_transient_array_count"],
        "semantic_descriptor_transient_array_count",
    )
    if policy_id != _SEMANTIC_DESCRIPTOR_TRANSIENT_POLICY_ID:
        raise ValueError("unsupported semantic descriptor transient policy_id")
    if max_bytes != _SEMANTIC_DESCRIPTOR_TRANSIENT_MAX_BYTES:
        raise ValueError("semantic descriptor transient max_bytes must be exactly 64 MiB")
    if required_bytes < 0 or required_bytes > max_bytes:
        raise ValueError("semantic descriptor transient required_bytes exceeds its bound")
    if not admitted or not released:
        raise ValueError("semantic descriptor transient receipt must be admitted and released")
    if array_count != _SEMANTIC_DESCRIPTOR_TRANSIENT_ARRAY_COUNT:
        raise ValueError("semantic descriptor transient array_count must be exactly 3")


CAPTURE_ARTIFACT_CODECS: tuple[CaptureArtifactCodec, ...] = (
    CaptureArtifactCodec(
        "phase0_donor_bundle",
        save_phase0_donor_bundle,
        required_fields=(
            "schema_version",
            "replay_kind",
            "status",
            "active_features",
            "activation_values",
            "activation_values_dtype",
            "activation_matrix_shape",
            "active_feature_count",
            "active_feature_membership_hash_raw_order",
            "active_feature_membership_hash_canonical",
            "active_feature_values_hash",
            "active_feature_layer_counts",
            "input_tokens",
            "input_token_count",
            "input_tokens_hash",
            "target_token_ids",
            "target_count",
            "target_token_ids_hash",
        ),
        nonempty_fields=(
            "activation_matrix_shape",
            "input_tokens",
            "target_token_ids",
        ),
        validator=_validate_phase0_capture,
    ),
    CaptureArtifactCodec(
        "phase3_seed_bundle",
        save_phase3_seed_bundle,
        required_fields=(
            "status",
            "active_features",
            "activation_values",
            "seed_feature_influences",
            "frontier_pre_locality",
            "frontier_post_locality",
            "queue_size",
            "actual_max_feature_nodes",
            "total_active_features",
            "planner_compute_dtype",
            "influence_compute_dtype",
        ),
        nonempty_fields=(
            "active_features",
            "activation_values",
        ),
        validator=_validate_phase3_seed_capture,
    ),
    CaptureArtifactCodec(
        "phase3_gradient_bundle",
        save_phase3_gradient_bundle,
        required_fields=(
            "schema_version",
            "status",
            "capture_kind",
            "gradient_batch_representation",
            "canonical_target_width",
            "target_token_ids",
            "target_probabilities",
            "target_token_ids_hash",
            "target_probability_hash",
            "active_feature_count",
            "active_features_hash",
            "activation_values_hash",
            "gradients",
            "layer_mask",
            "batch_call_indices",
            "per_layer_abs_sum",
            "per_layer_max_abs",
            "per_layer_nonfinite_count",
            "per_layer_hashes",
            "gradient_hash",
        ),
        nonempty_fields=(
            "target_token_ids",
            "target_probabilities",
        ),
        validator=_validate_phase3_gradient_capture,
    ),
    CaptureArtifactCodec(
        "phase3_row_bundle",
        save_phase3_row_bundle,
        required_fields=(
            "schema_version",
            "status",
            "capture_kind",
            "target_token_ids",
            "target_probabilities",
            "target_token_ids_hash",
            "target_probability_hash",
            "active_feature_count",
            "active_features_hash",
            "activation_values_hash",
            "phase3_feature_rows",
            "row_abs_sums",
            "feature_abs_sums",
            "error_abs_sums",
            "token_abs_sums",
            "total_active_features",
            "error_column_count",
            "token_column_count",
            "row_hash",
            "row_abs_sum_hash",
        ),
        nonempty_fields=(
            "target_token_ids",
            "target_probabilities",
        ),
        validator=_validate_phase3_row_capture,
    ),
    CaptureArtifactCodec(
        "feature_semantic_descriptors",
        save_feature_semantic_descriptors,
        required_fields=(
            "status",
            "descriptor_version",
            "descriptor_kind",
            "candidate_features",
            "candidate_row_indices",
            "semantic_sketch",
            *_SEMANTIC_DESCRIPTOR_TRANSIENT_RECEIPT_FIELDS,
        ),
        nonempty_fields=(),
        validator=_validate_semantic_descriptor_capture,
    ),
)


def _value_size(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, np.ndarray):
        return int(value.size)
    if isinstance(value, (str, bytes, list, tuple, Mapping)):
        return len(value)
    return 1 if value is not None else 0


def _validate_capture_payload(
    codec: CaptureArtifactCodec, payload: Mapping[str, Any]
) -> None:
    missing = [name for name in codec.required_fields if name not in payload]
    empty = [
        name
        for name in codec.nonempty_fields
        if name in payload and _value_size(payload[name]) == 0
    ]
    status = payload.get("status")
    try:
        status_text = _text(status)
    except (TypeError, ValueError):
        status_text = ""
    if missing or empty or not status_text:
        details = []
        if missing:
            details.append(f"missing required fields {missing!r}")
        if empty:
            details.append(f"empty required fields {empty!r}")
        if not status_text:
            details.append("missing non-empty status")
        raise ValueError(f"invalid {codec.name} payload: {'; '.join(details)}")
    if codec.name == "phase3_row_bundle":
        _validate_phase3_row_payload(payload)


def load_and_validate_capture_artifact(
    name: str,
    path: Path,
) -> dict[str, np.ndarray]:
    """Reopen one persisted capture through its schema-owned strict validator."""

    codecs = {codec.name: codec for codec in CAPTURE_ARTIFACT_CODECS}
    codec = codecs.get(name)
    if codec is None:
        raise ValueError(f"no schema codec registered for {name!r}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            payload = {field: np.asarray(archive[field]) for field in archive.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(
            f"could not safely load {name} artifact {path}: {error}"
        ) from error
    _validate_capture_payload(codec, payload)
    codec.validator(payload)
    return payload


def _validate_phase3_row_payload(payload: Mapping[str, Any]) -> None:
    """Validate the versioned raw-nonfeature extension before persistence."""
    schema_version = int(payload.get("schema_version", 0))
    capture_kind = str(payload.get("capture_kind", ""))
    if schema_version not in {1, 2}:
        raise ValueError(
            "invalid phase3_row_bundle payload: schema_version must be 1 or 2"
        )
    expected_kind = f"phase3_row_bundle_v{schema_version}"
    if capture_kind != expected_kind:
        raise ValueError(
            "invalid phase3_row_bundle payload: capture_kind must match "
            f"schema_version ({expected_kind!r})"
        )
    if schema_version == 1:
        return

    required_v2 = (
        "nonfeature_row_layout",
        "phase3_error_rows_by_layer",
        "phase3_token_rows",
        "error_rows_hash",
        "token_rows_hash",
    )
    missing = [name for name in required_v2 if name not in payload]
    if missing:
        raise ValueError(
            f"invalid phase3_row_bundle payload: missing version-2 fields {missing!r}"
        )
    if payload["nonfeature_row_layout"] != "target_layer_position_v1":
        raise ValueError(
            "invalid phase3_row_bundle payload: unexpected nonfeature_row_layout"
        )

    error_rows = _to_numpy_array(payload["phase3_error_rows_by_layer"])
    token_rows = _to_numpy_array(payload["phase3_token_rows"])
    target_ids = _to_numpy_array(payload.get("target_token_ids", [])).reshape(-1)
    if error_rows.ndim != 3:
        raise ValueError(
            "invalid phase3_row_bundle payload: phase3_error_rows_by_layer "
            "must have shape [targets, layers, positions]"
        )
    if token_rows.ndim != 2:
        raise ValueError(
            "invalid phase3_row_bundle payload: phase3_token_rows must have "
            "shape [targets, positions]"
        )
    target_count = int(target_ids.size)
    if (
        int(error_rows.shape[0]) != target_count
        or int(token_rows.shape[0]) != target_count
    ):
        raise ValueError(
            "invalid phase3_row_bundle payload: raw nonfeature target dimension "
            "does not match target_token_ids"
        )
    if int(error_rows.shape[2]) != int(token_rows.shape[1]):
        raise ValueError(
            "invalid phase3_row_bundle payload: error and token position "
            "dimensions do not match"
        )
    error_column_count = int(payload.get("error_column_count", -1))
    token_column_count = int(payload.get("token_column_count", -1))
    if error_column_count != int(error_rows.shape[1] * error_rows.shape[2]):
        raise ValueError(
            "invalid phase3_row_bundle payload: error_column_count does not "
            "match raw error rows"
        )
    if token_column_count != int(token_rows.shape[1]):
        raise ValueError(
            "invalid phase3_row_bundle payload: token_column_count does not "
            "match raw token rows"
        )
    if not np.issubdtype(error_rows.dtype, np.floating) or not np.issubdtype(
        token_rows.dtype, np.floating
    ):
        raise ValueError(
            "invalid phase3_row_bundle payload: raw nonfeature rows must be floating point"
        )


def write_capture_artifacts(
    *,
    requested: Mapping[str, bool],
    payloads: Mapping[str, Any],
    path_for: Callable[[str], Path],
) -> dict[str, Any]:
    """Atomically persist requested captures with their schema-specific codecs.

    The returned report is JSON-safe and deliberately records every contract
    category.  Callers must persist it before invoking
    :func:`require_complete_capture_artifacts` so an artifact failure cannot be
    mistaken for a successful probe with no capture request.
    """

    requested_names = sorted(name for name, enabled in requested.items() if enabled)
    codecs = {codec.name: codec for codec in CAPTURE_ARTIFACT_CODECS}
    written: list[str] = []
    paths: dict[str, str] = {}
    payload_statuses: dict[str, str] = {}
    missing: list[str] = []
    failed: list[dict[str, str]] = []

    for name in requested_names:
        codec = codecs.get(name)
        payload = payloads.get(name)
        path = path_for(name)
        if codec is None:
            failed.append(
                {
                    "name": name,
                    "path": str(path),
                    "error_type": "UnknownCaptureArtifact",
                    "message": f"no schema codec registered for {name}",
                }
            )
            continue
        if not isinstance(payload, Mapping):
            missing.append(name)
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
        try:
            _validate_capture_payload(codec, payload)
            codec.saver(payload, temporary)
            if not temporary.is_file():
                raise OSError(f"capture codec did not create {temporary}")
            if temporary.stat().st_size == 0:
                raise OSError(f"capture codec created an empty file: {temporary}")
            os.replace(temporary, path)
        except Exception as error:  # noqa: BLE001 - report arbitrary codec failures
            if temporary.exists():
                temporary.unlink()
            failed.append(
                {
                    "name": name,
                    "path": str(path),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue

        written.append(name)
        paths[name] = str(path)
        payload_statuses[name] = str(payload.get("status", "captured"))

    return {
        "schema_version": 1,
        "requested": requested_names,
        "written": written,
        "missing": missing,
        "failed": failed,
        "paths": paths,
        "payload_statuses": payload_statuses,
        "complete": not missing and not failed,
    }


def require_complete_capture_artifacts(report: Mapping[str, Any]) -> None:
    """Fail closed after a successful trace if requested evidence was lost."""

    if not bool(report.get("complete", False)):
        raise CaptureArtifactContractError(report)


def legacy_capture_artifact_status(report: Mapping[str, Any]) -> dict[str, str]:
    """Retain the established flat summary for existing result consumers."""

    statuses = {
        str(name): str(status)
        for name, status in report.get("payload_statuses", {}).items()
    }
    statuses.update(
        {str(name): "missing_payload" for name in report.get("missing", [])}
    )
    for failure in report.get("failed", []):
        if isinstance(failure, Mapping):
            statuses[str(failure.get("name"))] = (
                f"save_failed:{failure.get('error_type', 'Exception')}"
            )
    return statuses
