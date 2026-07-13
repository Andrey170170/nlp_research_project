"""Serialization of project-owned exact-trace debug and replay artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


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


def save_phase3_seed_bundle(payload: dict[str, Any], path: Path) -> None:
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
        status=np.array("" if payload.get("status") is None else str(payload["status"])),
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


def _phase3_common(payload: dict[str, Any]) -> dict[str, np.ndarray]:
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


def save_phase3_gradient_bundle(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        **_phase3_common(payload),
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


def save_phase3_row_bundle(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        **_phase3_common(payload),
        phase3_feature_rows=_to_numpy_array(payload.get("phase3_feature_rows", [])),
        row_abs_sums=_to_numpy_array(payload.get("row_abs_sums", [])),
        feature_abs_sums=_to_numpy_array(payload.get("feature_abs_sums", [])),
        error_abs_sums=_to_numpy_array(payload.get("error_abs_sums", [])),
        token_abs_sums=_to_numpy_array(payload.get("token_abs_sums", [])),
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
    )


def save_feature_semantic_descriptors(payload: dict[str, Any], path: Path) -> None:
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
        status=np.array("" if payload.get("status") is None else str(payload["status"])),
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


def _activation_values(payload: dict[str, Any]) -> tuple[np.ndarray, str, np.ndarray]:
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


def save_phase0_donor_bundle(payload: dict[str, Any], path: Path) -> None:
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
        prompt_metadata=np.array(_to_optional_json_text(payload.get("prompt_metadata"))),
        target_metadata=np.array(_to_optional_json_text(payload.get("target_metadata"))),
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
        status=np.array("" if payload.get("status") is None else str(payload["status"])),
    )
