"""Canonical typed-only compact graph serialization.

This module is the serialization seam for new exact-trace artifacts.  It owns
retention policy selection, typed endpoint encoding, deterministic
fingerprinting, atomic persistence, and fail-closed loading.  Historical
legacy and mixed artifacts deliberately remain behind ``compact_io``'s
explicit historical adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 2
COMPACT_SAVE_FORMAT = "typed_compact_graph_v2"
DEFAULT_RETENTION_POLICY_ID = "typed_top_p_v1"
FEATURE_ID_BASE = 1_000_000
CANONICAL_BUCKET_NAMES = (
    "feature<-feature",
    "feature<-error",
    "feature<-token",
    "logit<-feature",
    "logit<-error",
    "logit<-token",
)
LEGACY_EDGE_FIELDS = frozenset({"row_idx", "col_idx", "weights"})
_STEP_PATH_RE = re.compile(r"(?:token_|step_)(\d+)")
RETENTION_ALGORITHM_CONTRACT = {
    "algorithm": "absolute_mass_top_p_then_cap_v1",
    "input_cast": "float32_before_abs",
    "nonzero_rule": "float32_abs_not_equal_zero",
    "mass_accumulator": "float64",
    "sort": "descending_abs_stable_original_flat_index_ties",
    "top_p_boundary": "smallest_prefix_with_cumulative_mass_greater_or_equal",
    "cap_application": "after_top_p_prefix_selection",
    "cap_bound_flag": "true_only_when_cap_prevents_requested_top_p_prefix",
    "retained_order": "descending_abs_stable_original_flat_index_ties",
    "persisted_weight": "signed_float32",
}


@dataclass(frozen=True)
class BucketRetentionRule:
    top_p: float
    cap: int | None

    def to_json(self) -> dict[str, float | int | None]:
        return {"top_p": self.top_p, "cap": self.cap}


@dataclass(frozen=True)
class TypedEdgeRetentionPolicy:
    policy_id: str
    buckets: Mapping[str, BucketRetentionRule]
    algorithm: Mapping[str, str]

    @property
    def fingerprint(self) -> str:
        return _fingerprint_json(self.to_json())

    def to_json(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "algorithm": dict(self.algorithm),
            "buckets": {
                name: self.buckets[name].to_json() for name in CANONICAL_BUCKET_NAMES
            },
        }


@dataclass(frozen=True)
class ArtifactProvenance:
    """Opaque identities hashed into the canonical artifact.

    Each value must be JSON-compatible after normalization.  Keeping these as
    one parameter preserves a small build interface while ensuring callers
    state the four scientifically distinct identities.
    """

    provider: Mapping[str, Any]
    trace: Mapping[str, Any]
    target: Mapping[str, Any]
    step: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("provider", "trace", "target", "step"):
            identity = getattr(self, name)
            if not isinstance(identity, Mapping) or not identity:
                raise ValueError(f"artifact provenance {name} identity is required")
            if not _has_meaningful_identity_value(identity):
                raise ValueError(
                    f"artifact provenance {name} identity has no meaningful value"
                )
        for key in ("semantic_fingerprint", "execution_fingerprint"):
            value = self.trace.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"artifact provenance trace.{key} is required")
        if not any(
            key in self.target
            for key in (
                "token_id",
                "target_token_id",
                "logit_token_ids",
                "token_text",
                "target_token_text",
            )
        ):
            raise ValueError("artifact provenance target identity lacks a target key")
        if not any(key in self.step for key in ("step_idx", "generated_index")):
            raise ValueError("artifact provenance step identity lacks a step key")


@dataclass(frozen=True)
class TypedCompactGraph:
    path: Path | None
    schema_version: int
    compact_save_format: str
    step_idx: int
    token_text: str
    logprob: float | None
    feature_ids: np.ndarray
    bucket_row_idx: np.ndarray
    bucket_col_idx: np.ndarray
    bucket_weights: np.ndarray
    bucket_ids: np.ndarray
    bucket_names: tuple[str, ...]
    bucket_metadata: dict[str, dict[str, Any]]
    error_node_shape: tuple[int, int]
    token_ids: np.ndarray
    logit_token_ids: np.ndarray
    retention_policy_id: str
    retention_policy_fingerprint: str
    content_fingerprint: str
    graph_fingerprint: str
    target_fingerprint: str
    provider_fingerprint: str
    trace_fingerprint: str
    step_fingerprint: str

    @property
    def n_features(self) -> int:
        return int(self.feature_ids.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.bucket_weights.size)


_POLICIES = {
    DEFAULT_RETENTION_POLICY_ID: TypedEdgeRetentionPolicy(
        policy_id=DEFAULT_RETENTION_POLICY_ID,
        algorithm=RETENTION_ALGORITHM_CONTRACT,
        buckets={
            "feature<-feature": BucketRetentionRule(0.95, 1_000_000),
            "feature<-error": BucketRetentionRule(0.99, 16_000),
            "feature<-token": BucketRetentionRule(0.95, 250_000),
            "logit<-feature": BucketRetentionRule(1.0, None),
            "logit<-error": BucketRetentionRule(1.0, None),
            "logit<-token": BucketRetentionRule(1.0, None),
        },
    )
}


_SERIALIZED_FIELDS = frozenset(
    {
        "schema_version",
        "compact_save_format",
        "step_idx",
        "token_text",
        "logprob",
        "feature_ids",
        "bucket_row_idx",
        "bucket_col_idx",
        "bucket_weights",
        "bucket_ids",
        "bucket_names",
        "bucket_metadata_json",
        "error_node_shape",
        "token_ids",
        "logit_token_ids",
        "retention_policy_id",
        "retention_policy_json",
        "retention_policy_fingerprint",
        "content_fingerprint",
        "graph_fingerprint",
        "target_fingerprint",
        "provider_fingerprint",
        "trace_fingerprint",
        "step_fingerprint",
    }
)


def get_retention_policy(
    policy_id: str = DEFAULT_RETENTION_POLICY_ID,
) -> TypedEdgeRetentionPolicy:
    """Return one admitted, immutable typed-edge retention policy."""

    try:
        return _POLICIES[policy_id]
    except KeyError as exc:
        raise ValueError(f"unknown typed edge retention policy: {policy_id!r}") from exc


def build_typed_compact_graph(
    compact_result: Mapping[str, Any],
    step_idx: int,
    *,
    provenance: ArtifactProvenance,
    token_text: str = "",
    logprob: float | None = None,
    retention_policy_id: str = DEFAULT_RETENTION_POLICY_ID,
) -> TypedCompactGraph:
    """Build a canonical typed-only graph from one compact trace result."""

    if step_idx < 0:
        raise ValueError("step_idx must be non-negative")
    policy = get_retention_policy(retention_policy_id)
    active_features = _cpu_int_tensor(compact_result, "active_features", ndim=2)
    if active_features.shape[1] != 3:
        raise ValueError("active_features must have shape (n_features, 3)")
    selected_features = _cpu_int_tensor(compact_result, "selected_features", ndim=1)
    feature_rows = _cpu_int_tensor(compact_result, "feature_row_node_indices", ndim=1)
    logit_rows = _cpu_int_tensor(compact_result, "logit_row_node_indices", ndim=1)
    if selected_features.numel() and (
        int(selected_features.min()) < 0
        or int(selected_features.max()) >= active_features.shape[0]
    ):
        raise ValueError("selected_features is outside active_features")
    if len(set(selected_features.tolist())) != int(selected_features.numel()):
        raise ValueError("selected_features must contain unique indices")
    selected_feature_ids = active_features[selected_features]
    feature_ids = selected_feature_ids.numpy().astype(np.int64, copy=True)

    input_tokens = _cpu_int_tensor(compact_result, "input_tokens", ndim=1)
    token_ids = input_tokens.numpy().astype(np.int64, copy=True)
    if token_ids.size == 0:
        raise ValueError("typed compact graphs require a non-empty token domain")
    n_pos = int(token_ids.size)
    n_error = _nonnegative_int(compact_result.get("n_error_nodes", 0), "n_error_nodes")
    n_token = _nonnegative_int(
        compact_result.get("n_token_nodes", n_pos), "n_token_nodes"
    )
    if n_token != n_pos:
        raise ValueError("n_token_nodes must match len(input_tokens)")
    if n_error % n_pos:
        raise ValueError("n_error_nodes must be divisible by the token domain")
    error_node_shape = (n_error // n_pos, n_pos)
    n_feature_layers = error_node_shape[0]
    if n_feature_layers <= 0:
        raise ValueError(
            "typed compact graphs require a non-empty feature-layer domain"
        )
    if active_features.numel() and (
        int(active_features.min()) < 0
        or int(active_features[:, 0].max()) >= n_feature_layers
        or int(active_features[:, 1].max()) >= n_pos
        or int(active_features[:, 2].max()) >= FEATURE_ID_BASE
    ):
        raise ValueError(
            "active_features is outside declared layer, position, or feature domains"
        )
    logit_token_ids = np.asarray(
        [
            getattr(target, "vocab_idx", -1)
            for target in compact_result["logit_targets"]
        ],
        dtype=np.int64,
    )
    if logit_token_ids.size == 0 or int(logit_token_ids.min()) < 0:
        raise ValueError("logit_token_ids must contain a non-empty non-negative domain")
    _validate_provenance_bindings(
        provenance,
        compact_result=compact_result,
        step_idx=step_idx,
        token_text=token_text,
        logit_token_ids=logit_token_ids,
    )
    if feature_rows.numel() and (
        int(feature_rows.min()) < 0
        or int(feature_rows.max()) >= active_features.shape[0]
    ):
        raise ValueError("feature_row_node_indices is outside active_features")
    if len(set(feature_rows.tolist())) != int(feature_rows.numel()):
        raise ValueError("feature_row_node_indices must contain unique indices")
    if not set(feature_rows.tolist()).issubset(set(selected_features.tolist())):
        raise ValueError(
            "feature_row_node_indices must refer to persisted selected_features"
        )
    if logit_rows.numel() and int(logit_rows.min()) < 0:
        raise ValueError("logit_row_node_indices contains a negative index")
    if len(set(logit_rows.tolist())) != int(logit_rows.numel()):
        raise ValueError("logit_row_node_indices must contain unique indices")
    if int(logit_rows.numel()) != int(logit_token_ids.size):
        raise ValueError("logit row count must match logit_targets")

    def encode_feature_ids(values: torch.Tensor) -> torch.Tensor:
        return (
            values[:, 0].to(torch.int64) * n_pos * FEATURE_ID_BASE
            + values[:, 1].to(torch.int64) * FEATURE_ID_BASE
            + values[:, 2].to(torch.int64)
        )

    expected_shapes = {
        "feature<-feature": (
            int(feature_rows.numel()),
            int(selected_features.numel()),
        ),
        "feature<-error": (int(feature_rows.numel()), n_error),
        "feature<-token": (int(feature_rows.numel()), n_token),
        "logit<-feature": (
            int(logit_rows.numel()),
            int(selected_features.numel()),
        ),
        "logit<-error": (int(logit_rows.numel()), n_error),
        "logit<-token": (int(logit_rows.numel()), n_token),
    }
    matrix_keys = {
        "feature<-feature": "feature_feature_edges",
        "feature<-error": "feature_error_edges",
        "feature<-token": "feature_token_edges",
        "logit<-feature": "logit_feature_edges",
        "logit<-error": "logit_error_edges",
        "logit<-token": "logit_token_edges",
    }
    matrices: dict[str, torch.Tensor] = {}
    for name in CANONICAL_BUCKET_NAMES:
        matrix = _cpu_float_matrix(compact_result[matrix_keys[name]], name)
        if tuple(matrix.shape) != expected_shapes[name]:
            raise ValueError(
                f"{name} shape {tuple(matrix.shape)} does not match declared "
                f"domains {expected_shapes[name]}"
            )
        matrices[name] = matrix

    bucket_specs = (
        (
            "feature<-feature",
            matrices["feature<-feature"],
            feature_rows,
            selected_feature_ids,
        ),
        ("feature<-error", matrices["feature<-error"], feature_rows, None),
        ("feature<-token", matrices["feature<-token"], feature_rows, None),
        (
            "logit<-feature",
            matrices["logit<-feature"],
            logit_rows,
            selected_feature_ids,
        ),
        ("logit<-error", matrices["logit<-error"], logit_rows, None),
        ("logit<-token", matrices["logit<-token"], logit_rows, None),
    )
    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_bucket_ids: list[np.ndarray] = []
    metadata: dict[str, dict[str, Any]] = {}
    for bucket_id, (name, raw_matrix, row_nodes, feature_cols) in enumerate(
        bucket_specs
    ):
        matrix = raw_matrix
        if int(row_nodes.numel()) != int(matrix.shape[0]):
            raise ValueError(f"{name} row count does not match its row-node domain")
        if name.endswith("<-feature"):
            assert feature_cols is not None
            if int(feature_cols.shape[0]) != int(matrix.shape[1]):
                raise ValueError(
                    f"{name} column count does not match selected_features"
                )
        keep, cap_bound = _retained_indices(matrix, policy.buckets[name])
        flat = matrix.reshape(-1).float()
        flat_abs = flat.abs().double()
        raw_edge_count = int(torch.count_nonzero(flat_abs).item())
        raw_abs_mass = float(flat_abs.sum().item())
        retained_abs_mass = float(flat_abs[keep].sum().item()) if keep.numel() else 0.0
        n_cols = int(matrix.shape[1])
        local_rows = keep // n_cols
        local_cols = keep % n_cols
        if name.startswith("feature<-"):
            if local_rows.numel() and (
                int(row_nodes[local_rows].min()) < 0
                or int(row_nodes[local_rows].max()) >= active_features.shape[0]
            ):
                raise ValueError(f"{name} row nodes are outside active_features")
            rows = encode_feature_ids(active_features[row_nodes[local_rows]])
        else:
            rows = local_rows.to(torch.int64)
        if name.endswith("<-feature"):
            assert feature_cols is not None
            cols = encode_feature_ids(feature_cols[local_cols])
        else:
            cols = local_cols.to(torch.int64)
        weights = flat[keep].float()
        all_rows.append(rows.numpy().astype(np.int64, copy=False))
        all_cols.append(cols.numpy().astype(np.int64, copy=False))
        all_weights.append(weights.numpy().astype(np.float32, copy=False))
        all_bucket_ids.append(np.full(int(keep.numel()), bucket_id, dtype=np.int16))
        metadata[name] = {
            "bucket": name,
            "raw_edge_count": raw_edge_count,
            "raw_abs_mass": raw_abs_mass,
            "retained_edge_count": int(keep.numel()),
            "retained_abs_mass": retained_abs_mass,
            "retained_fraction": (
                retained_abs_mass / raw_abs_mass if raw_abs_mass else None
            ),
            "cap_bound_before_top_p": cap_bound,
            "policy": policy.buckets[name].to_json(),
            "weights_signed": True,
        }

    graph = TypedCompactGraph(
        path=None,
        schema_version=SCHEMA_VERSION,
        compact_save_format=COMPACT_SAVE_FORMAT,
        step_idx=step_idx,
        token_text=token_text,
        logprob=logprob,
        feature_ids=feature_ids,
        bucket_row_idx=np.concatenate(all_rows),
        bucket_col_idx=np.concatenate(all_cols),
        bucket_weights=np.concatenate(all_weights),
        bucket_ids=np.concatenate(all_bucket_ids),
        bucket_names=CANONICAL_BUCKET_NAMES,
        bucket_metadata=metadata,
        error_node_shape=error_node_shape,
        token_ids=token_ids,
        logit_token_ids=logit_token_ids,
        retention_policy_id=policy.policy_id,
        retention_policy_fingerprint=policy.fingerprint,
        content_fingerprint="",
        graph_fingerprint="",
        target_fingerprint=_fingerprint_json(provenance.target),
        provider_fingerprint=_fingerprint_json(provenance.provider),
        trace_fingerprint=_fingerprint_json(provenance.trace),
        step_fingerprint=_fingerprint_json(provenance.step),
    )
    content_fingerprint = _content_fingerprint(graph)
    graph = _replace_fingerprints(graph, content_fingerprint)
    _validate_graph(graph)
    return graph


def save_typed_compact_graph(graph: TypedCompactGraph, path: Path) -> None:
    """Validate and atomically persist one canonical typed-only artifact."""

    _validate_graph(graph)
    expected = _content_fingerprint(graph)
    if graph.content_fingerprint != expected or graph.graph_fingerprint != expected:
        raise ValueError("typed compact graph content fingerprint is stale")
    payload = _serialized_payload(graph)
    graph_path = Path(path)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = graph_path.with_name(f".{graph_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, graph_path)
    finally:
        temporary.unlink(missing_ok=True)


def load_typed_compact_graph(
    path: Path,
    *,
    expected_step_idx: int | None = None,
    validate_step_path: bool = True,
) -> TypedCompactGraph:
    """Strictly load a canonical v2 artifact, rejecting legacy or schema drift."""

    graph_path = Path(path)
    with np.load(str(graph_path), allow_pickle=False) as data:
        fields = set(data.files)
        if fields & LEGACY_EDGE_FIELDS:
            raise ValueError("canonical typed graph contains legacy edge fields")
        missing = sorted(_SERIALIZED_FIELDS - fields)
        unknown = sorted(fields - _SERIALIZED_FIELDS)
        if missing or unknown:
            raise ValueError(
                "canonical typed graph fields do not match schema v2: "
                f"missing={missing}, unknown={unknown}"
            )
        schema_version = _integer_scalar(data, "schema_version")
        save_format = _string_scalar(data, "compact_save_format")
        step_idx = _integer_scalar(data, "step_idx")
        logprob_value = float(_scalar(data, "logprob"))
        error_shape_values = _integer_array(data, "error_node_shape", ndim=1)
        if error_shape_values.shape != (2,):
            raise ValueError("error_node_shape must contain exactly two dimensions")
        graph = TypedCompactGraph(
            path=graph_path,
            schema_version=schema_version,
            compact_save_format=save_format,
            step_idx=step_idx,
            token_text=_string_scalar(data, "token_text"),
            logprob=None if np.isnan(logprob_value) else logprob_value,
            feature_ids=_integer_array(data, "feature_ids", ndim=2),
            bucket_row_idx=_integer_array(data, "bucket_row_idx", ndim=1),
            bucket_col_idx=_integer_array(data, "bucket_col_idx", ndim=1),
            bucket_weights=_float_array(data, "bucket_weights", ndim=1),
            bucket_ids=_integer_array(data, "bucket_ids", ndim=1),
            bucket_names=tuple(
                str(value)
                for value in _string_array(data, "bucket_names", ndim=1).tolist()
            ),
            bucket_metadata=_metadata_from_json(
                _string_scalar(data, "bucket_metadata_json")
            ),
            error_node_shape=(
                int(error_shape_values[0]),
                int(error_shape_values[1]),
            ),
            token_ids=_integer_array(data, "token_ids", ndim=1),
            logit_token_ids=_integer_array(data, "logit_token_ids", ndim=1),
            retention_policy_id=_string_scalar(data, "retention_policy_id"),
            retention_policy_fingerprint=_string_scalar(
                data, "retention_policy_fingerprint"
            ),
            content_fingerprint=_string_scalar(data, "content_fingerprint"),
            graph_fingerprint=_string_scalar(data, "graph_fingerprint"),
            target_fingerprint=_string_scalar(data, "target_fingerprint"),
            provider_fingerprint=_string_scalar(data, "provider_fingerprint"),
            trace_fingerprint=_string_scalar(data, "trace_fingerprint"),
            step_fingerprint=_string_scalar(data, "step_fingerprint"),
        )
        policy_json = _json_from_string(
            _string_scalar(data, "retention_policy_json"),
            field="retention_policy_json",
        )
    _validate_graph(graph)
    policy = get_retention_policy(graph.retention_policy_id)
    if policy_json != policy.to_json():
        raise ValueError("retention_policy_json does not match the admitted policy")
    if graph.retention_policy_fingerprint != policy.fingerprint:
        raise ValueError("retention policy fingerprint mismatch")
    expected_content = _content_fingerprint(graph)
    if graph.content_fingerprint != expected_content:
        raise ValueError("content fingerprint mismatch")
    if graph.graph_fingerprint != expected_content:
        raise ValueError("graph fingerprint mismatch")
    required_step = expected_step_idx
    if required_step is None and validate_step_path:
        required_step = _step_index_from_path(graph_path)
    if required_step is not None and graph.step_idx != required_step:
        raise ValueError(
            f"graph step_idx {graph.step_idx} does not match expected step "
            f"{required_step}: {graph_path}"
        )
    return graph


def _retained_indices(
    values: torch.Tensor, rule: BucketRetentionRule
) -> tuple[torch.Tensor, bool]:
    flat_abs = values.float().abs().reshape(-1)
    nonzero = torch.where(flat_abs != 0)[0]
    if nonzero.numel() == 0:
        return nonzero, False
    nonzero_values = flat_abs[nonzero]
    order = torch.argsort(nonzero_values, descending=True, stable=True)
    desired_count = int(nonzero.numel())
    if rule.top_p < 1.0:
        sorted_values = nonzero_values[order].double()
        cumulative = torch.cumsum(sorted_values, dim=0) / sorted_values.sum()
        desired_count = (
            int(
                torch.searchsorted(
                    cumulative, torch.tensor(rule.top_p, dtype=cumulative.dtype)
                ).item()
            )
            + 1
        )
    retained_count = desired_count
    cap_bound = rule.cap is not None and rule.cap < desired_count
    if rule.cap is not None:
        retained_count = min(retained_count, rule.cap)
    return nonzero[order[:retained_count]], cap_bound


def _serialized_payload(graph: TypedCompactGraph) -> dict[str, np.ndarray]:
    policy = get_retention_policy(graph.retention_policy_id)
    return {
        "schema_version": np.array(graph.schema_version, dtype=np.int32),
        "compact_save_format": np.array(graph.compact_save_format),
        "step_idx": np.array(graph.step_idx, dtype=np.int32),
        "token_text": np.array(graph.token_text),
        "logprob": np.array(graph.logprob if graph.logprob is not None else np.nan),
        "feature_ids": graph.feature_ids,
        "bucket_row_idx": graph.bucket_row_idx,
        "bucket_col_idx": graph.bucket_col_idx,
        "bucket_weights": graph.bucket_weights,
        "bucket_ids": graph.bucket_ids,
        "bucket_names": np.asarray(graph.bucket_names),
        "bucket_metadata_json": np.array(_canonical_json(graph.bucket_metadata)),
        "error_node_shape": np.asarray(graph.error_node_shape, dtype=np.int32),
        "token_ids": graph.token_ids,
        "logit_token_ids": graph.logit_token_ids,
        "retention_policy_id": np.array(graph.retention_policy_id),
        "retention_policy_json": np.array(_canonical_json(policy.to_json())),
        "retention_policy_fingerprint": np.array(graph.retention_policy_fingerprint),
        "content_fingerprint": np.array(graph.content_fingerprint),
        "graph_fingerprint": np.array(graph.graph_fingerprint),
        "target_fingerprint": np.array(graph.target_fingerprint),
        "provider_fingerprint": np.array(graph.provider_fingerprint),
        "trace_fingerprint": np.array(graph.trace_fingerprint),
        "step_fingerprint": np.array(graph.step_fingerprint),
    }


def _content_fingerprint(graph: TypedCompactGraph) -> str:
    digest = hashlib.sha256()
    scalars = {
        "schema_version": graph.schema_version,
        "compact_save_format": graph.compact_save_format,
        "step_idx": graph.step_idx,
        "token_text": graph.token_text,
        "logprob": graph.logprob,
        "bucket_names": list(graph.bucket_names),
        "bucket_metadata": graph.bucket_metadata,
        "error_node_shape": list(graph.error_node_shape),
        "retention_policy_id": graph.retention_policy_id,
        "retention_policy_fingerprint": graph.retention_policy_fingerprint,
        "target_fingerprint": graph.target_fingerprint,
        "provider_fingerprint": graph.provider_fingerprint,
        "trace_fingerprint": graph.trace_fingerprint,
        "step_fingerprint": graph.step_fingerprint,
    }
    digest.update(_canonical_json(scalars).encode())
    for name, array in (
        ("feature_ids", graph.feature_ids),
        ("bucket_row_idx", graph.bucket_row_idx),
        ("bucket_col_idx", graph.bucket_col_idx),
        ("bucket_weights", graph.bucket_weights),
        ("bucket_ids", graph.bucket_ids),
        ("token_ids", graph.token_ids),
        ("logit_token_ids", graph.logit_token_ids),
    ):
        canonical = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(str(canonical.dtype).encode())
        digest.update(_canonical_json(list(canonical.shape)).encode())
        digest.update(canonical.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _replace_fingerprints(
    graph: TypedCompactGraph, content_fingerprint: str
) -> TypedCompactGraph:
    values = {name: getattr(graph, name) for name in graph.__dataclass_fields__}
    values["content_fingerprint"] = content_fingerprint
    values["graph_fingerprint"] = content_fingerprint
    return TypedCompactGraph(**values)


def _validate_graph(graph: TypedCompactGraph) -> None:
    if graph.schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if graph.compact_save_format != COMPACT_SAVE_FORMAT:
        raise ValueError(f"compact_save_format must be {COMPACT_SAVE_FORMAT!r}")
    if graph.step_idx < 0:
        raise ValueError("step_idx must be non-negative")
    if graph.logprob is not None and not np.isfinite(graph.logprob):
        raise ValueError("logprob must be finite or None")
    if graph.feature_ids.ndim != 2 or graph.feature_ids.shape[1:] != (3,):
        raise ValueError("feature_ids must have shape (n_features, 3)")
    if not np.issubdtype(graph.feature_ids.dtype, np.integer):
        raise ValueError("feature_ids must be integers")
    if graph.feature_ids.size and int(graph.feature_ids.min()) < 0:
        raise ValueError("feature_ids contains a negative identity")
    if len({tuple(row) for row in graph.feature_ids.tolist()}) != graph.n_features:
        raise ValueError("feature_ids must contain unique identities")
    if graph.bucket_names != CANONICAL_BUCKET_NAMES:
        raise ValueError("bucket_names must be the canonical ordered six buckets")
    lengths = {
        len(graph.bucket_row_idx),
        len(graph.bucket_col_idx),
        len(graph.bucket_weights),
        len(graph.bucket_ids),
    }
    if len(lengths) != 1:
        raise ValueError("typed bucket COO arrays have mismatched lengths")
    for name, values in (
        ("bucket_row_idx", graph.bucket_row_idx),
        ("bucket_col_idx", graph.bucket_col_idx),
        ("bucket_ids", graph.bucket_ids),
        ("token_ids", graph.token_ids),
        ("logit_token_ids", graph.logit_token_ids),
    ):
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
    if graph.token_ids.size == 0 or int(graph.token_ids.min()) < 0:
        raise ValueError("token_ids must contain a non-empty non-negative domain")
    if graph.logit_token_ids.size == 0 or int(graph.logit_token_ids.min()) < 0:
        raise ValueError("logit_token_ids must contain a non-empty non-negative domain")
    if graph.bucket_weights.ndim != 1 or not np.issubdtype(
        graph.bucket_weights.dtype, np.floating
    ):
        raise ValueError("bucket_weights must be a one-dimensional float array")
    if not np.isfinite(graph.bucket_weights).all():
        raise ValueError("bucket_weights contains a non-finite value")
    if np.any(graph.bucket_weights == 0):
        raise ValueError("bucket_weights must contain only retained nonzero edges")
    if graph.bucket_row_idx.size and int(graph.bucket_row_idx.min()) < 0:
        raise ValueError("bucket_row_idx contains a negative identity")
    if graph.bucket_col_idx.size and int(graph.bucket_col_idx.min()) < 0:
        raise ValueError("bucket_col_idx contains a negative identity")
    if graph.bucket_ids.size and (
        int(graph.bucket_ids.min()) < 0
        or int(graph.bucket_ids.max()) >= len(CANONICAL_BUCKET_NAMES)
    ):
        raise ValueError("bucket_ids is outside bucket_names")
    if graph.bucket_ids.size:
        coordinates = np.column_stack(
            (graph.bucket_ids, graph.bucket_row_idx, graph.bucket_col_idx)
        )
        if len(np.unique(coordinates, axis=0)) != len(coordinates):
            raise ValueError("typed bucket COO contains duplicate coordinates")
    if len(graph.error_node_shape) != 2 or min(graph.error_node_shape) < 0:
        raise ValueError("error_node_shape must contain two non-negative dimensions")
    if graph.error_node_shape[1] != len(graph.token_ids):
        raise ValueError("error_node_shape position dimension must match token_ids")
    if graph.error_node_shape[0] <= 0:
        raise ValueError("error_node_shape must declare a feature-layer domain")
    policy = get_retention_policy(graph.retention_policy_id)
    if graph.retention_policy_fingerprint != policy.fingerprint:
        raise ValueError("retention policy fingerprint mismatch")
    if set(graph.bucket_metadata) != set(CANONICAL_BUCKET_NAMES):
        raise ValueError("bucket metadata must cover exactly the canonical buckets")
    _validate_metadata(graph, policy)
    _validate_endpoint_domains(graph)
    for field in (
        "content_fingerprint",
        "graph_fingerprint",
        "target_fingerprint",
        "provider_fingerprint",
        "trace_fingerprint",
        "step_fingerprint",
    ):
        value = getattr(graph, field)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise ValueError(f"{field} must be a sha256 fingerprint")


def _validate_metadata(
    graph: TypedCompactGraph, policy: TypedEdgeRetentionPolicy
) -> None:
    required = {
        "bucket",
        "raw_edge_count",
        "raw_abs_mass",
        "retained_edge_count",
        "retained_abs_mass",
        "retained_fraction",
        "cap_bound_before_top_p",
        "policy",
        "weights_signed",
    }
    for bucket_id, name in enumerate(CANONICAL_BUCKET_NAMES):
        row = graph.bucket_metadata[name]
        if set(row) != required:
            raise ValueError(f"bucket metadata fields do not match schema for {name}")
        if row["bucket"] != name or row["policy"] != policy.buckets[name].to_json():
            raise ValueError(f"bucket metadata policy identity mismatch for {name}")
        if row["weights_signed"] is not True:
            raise ValueError(f"bucket weights must be signed for {name}")
        if not isinstance(row["cap_bound_before_top_p"], bool):
            raise ValueError(f"cap-bound metadata must be bool for {name}")
        raw_count = _metadata_count(row, "raw_edge_count", name)
        retained_count = _metadata_count(row, "retained_edge_count", name)
        rule = policy.buckets[name]
        if rule.cap is not None and retained_count > rule.cap:
            raise ValueError(f"retained edge count exceeds policy cap for {name}")
        raw_mass = _metadata_mass(row, "raw_abs_mass", name)
        retained_mass = _metadata_mass(row, "retained_abs_mass", name)
        if (raw_count == 0) != (raw_mass == 0):
            raise ValueError(f"raw edge count and mass zero-state disagree for {name}")
        if (retained_count == 0) != (retained_mass == 0):
            raise ValueError(
                f"retained edge count and mass zero-state disagree for {name}"
            )
        actual_count = int(np.count_nonzero(graph.bucket_ids == bucket_id))
        if retained_count != actual_count or raw_count < retained_count:
            raise ValueError(f"bucket edge counts are inconsistent for {name}")
        actual_mass = float(
            np.abs(graph.bucket_weights[graph.bucket_ids == bucket_id])
            .astype(np.float64)
            .sum()
        )
        if retained_mass > raw_mass + max(1e-9, raw_mass * 1e-9):
            raise ValueError(f"retained mass exceeds raw mass for {name}")
        if not np.isclose(retained_mass, actual_mass, rtol=1e-6, atol=1e-8):
            raise ValueError(f"retained mass does not match edges for {name}")
        fraction = row["retained_fraction"]
        if raw_mass == 0:
            if retained_mass != 0 or fraction is not None:
                raise ValueError(f"zero-mass retained fraction is invalid for {name}")
        elif not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise ValueError(f"retained fraction must be numeric for {name}")
        elif not np.isclose(
            float(fraction), retained_mass / raw_mass, rtol=1e-9, atol=1e-12
        ):
            raise ValueError(f"retained fraction is inconsistent for {name}")
        if raw_mass > 0:
            fraction_value = float(fraction)
            cap_bound = row["cap_bound_before_top_p"]
            if cap_bound:
                if rule.cap is None or retained_count != rule.cap:
                    raise ValueError(f"cap-bound metadata is impossible for {name}")
                if fraction_value >= rule.top_p:
                    raise ValueError(
                        f"cap-bound metadata reached requested top-p for {name}"
                    )
            elif fraction_value + 1e-12 < rule.top_p:
                raise ValueError(
                    f"retained fraction misses top-p without a bound cap for {name}"
                )


def _validate_endpoint_domains(graph: TypedCompactGraph) -> None:
    n_pos = len(graph.token_ids)
    if n_pos <= 0:
        raise ValueError("typed feature endpoints require a token domain")
    if graph.feature_ids.size and (
        int(graph.feature_ids[:, 0].max()) >= graph.error_node_shape[0]
        or int(graph.feature_ids[:, 1].max()) >= n_pos
        or int(graph.feature_ids[:, 2].max()) >= FEATURE_ID_BASE
    ):
        raise ValueError("feature_ids is outside the typed identity domain")
    encoded_features = set(
        (
            graph.feature_ids[:, 0] * n_pos * FEATURE_ID_BASE
            + graph.feature_ids[:, 1] * FEATURE_ID_BASE
            + graph.feature_ids[:, 2]
        ).tolist()
    )
    error_count = int(np.prod(graph.error_node_shape, dtype=np.int64))
    for bucket_id, name in enumerate(CANONICAL_BUCKET_NAMES):
        mask = graph.bucket_ids == bucket_id
        rows = graph.bucket_row_idx[mask]
        cols = graph.bucket_col_idx[mask]
        if name.startswith("feature<-") and any(
            int(value) not in encoded_features for value in rows
        ):
            raise ValueError(f"{name} target is not a selected feature")
        if (
            name.startswith("logit<-")
            and rows.size
            and int(rows.max()) >= len(graph.logit_token_ids)
        ):
            raise ValueError(f"{name} target is outside logit_token_ids")
        if name.endswith("<-feature") and any(
            int(value) not in encoded_features for value in cols
        ):
            raise ValueError(f"{name} source is not a selected feature")
        if name.endswith("<-error") and cols.size and int(cols.max()) >= error_count:
            raise ValueError(f"{name} source is outside error_node_shape")
        if name.endswith("<-token") and cols.size and int(cols.max()) >= n_pos:
            raise ValueError(f"{name} source is outside token_ids")


def _cpu_int_tensor(payload: Mapping[str, Any], key: str, *, ndim: int) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{key} must be a {ndim}-dimensional tensor")
    if value.dtype == torch.bool or torch.is_floating_point(value):
        raise ValueError(f"{key} must be an integer tensor")
    return value.to(dtype=torch.int64).detach().cpu()


def _cpu_float_matrix(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} edges must be a two-dimensional tensor")
    matrix = value.detach().cpu()
    if not torch.is_floating_point(matrix):
        raise ValueError(f"{name} edges must be floating point")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} edges contain a non-finite value")
    return matrix.to(dtype=torch.float32)


def _validate_provenance_bindings(
    provenance: ArtifactProvenance,
    *,
    compact_result: Mapping[str, Any],
    step_idx: int,
    token_text: str,
    logit_token_ids: np.ndarray,
) -> None:
    for key in ("semantic_fingerprint", "execution_fingerprint"):
        compact_fingerprint = compact_result.get(key)
        if not isinstance(compact_fingerprint, str) or not compact_fingerprint:
            raise ValueError(f"compact_result.{key} is required for provenance binding")
        if provenance.trace[key] != compact_fingerprint:
            raise ValueError(
                f"artifact provenance trace.{key} does not match compact_result"
            )

    for key in ("step_idx", "generated_index"):
        if key in provenance.step and provenance.step[key] != step_idx:
            raise ValueError(f"artifact provenance step.{key} does not match step_idx")

    target = provenance.target
    for key in ("token_text", "target_token_text"):
        if key in target:
            supplied_text = target[key]
            if not isinstance(supplied_text, str):
                raise ValueError(f"artifact provenance target.{key} must be a string")
            if supplied_text != token_text:
                raise ValueError(
                    f"artifact provenance target.{key} does not match token_text"
                )
    expected_targets = [int(value) for value in logit_token_ids.tolist()]
    if "logit_token_ids" in target:
        supplied = target["logit_token_ids"]
        if not isinstance(supplied, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in supplied
        ):
            raise ValueError("artifact provenance target.logit_token_ids must be ints")
        if list(supplied) != expected_targets:
            raise ValueError(
                "artifact provenance target.logit_token_ids does not match targets"
            )
    for key in ("token_id", "target_token_id"):
        if key in target:
            supplied = target[key]
            if isinstance(supplied, bool) or not isinstance(supplied, int):
                raise ValueError(f"artifact provenance target.{key} must be an int")
            if supplied not in expected_targets:
                raise ValueError(
                    f"artifact provenance target.{key} does not match targets"
                )


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative int")
    return value


def _metadata_count(row: Mapping[str, Any], field: str, bucket: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative int for {bucket}")
    return value


def _metadata_mass(row: Mapping[str, Any], field: str, bucket: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric for {bucket}")
    converted = float(value)
    if not np.isfinite(converted) or converted < 0:
        raise ValueError(f"{field} must be finite and non-negative for {bucket}")
    return converted


def _scalar(data: Any, field: str) -> np.ndarray:
    value = np.asarray(data[field])
    if value.ndim != 0:
        raise ValueError(f"{field} must be a scalar")
    return value


def _integer_scalar(data: Any, field: str) -> int:
    value = _scalar(data, field)
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{field} must be an integer scalar")
    return int(value)


def _string_scalar(data: Any, field: str) -> str:
    value = _scalar(data, field)
    if value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be a string scalar")
    return str(value)


def _integer_array(data: Any, field: str, *, ndim: int) -> np.ndarray:
    value = np.asarray(data[field])
    if value.ndim != ndim or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{field} must be a {ndim}-dimensional integer array")
    return value.copy()


def _float_array(data: Any, field: str, *, ndim: int) -> np.ndarray:
    value = np.asarray(data[field])
    if value.ndim != ndim or not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{field} must be a {ndim}-dimensional floating array")
    return value.copy()


def _string_array(data: Any, field: str, *, ndim: int) -> np.ndarray:
    value = np.asarray(data[field])
    if value.ndim != ndim or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be a {ndim}-dimensional string array")
    return value.copy()


def _metadata_from_json(raw: str) -> dict[str, dict[str, Any]]:
    value = _json_from_string(raw, field="bucket_metadata_json")
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(row, dict) for name, row in value.items()
    ):
        raise ValueError("bucket_metadata_json must contain an object of objects")
    return value


def _json_from_string(raw: str, *, field: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must contain valid JSON") from exc


def _step_index_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = _STEP_PATH_RE.fullmatch(Path(part).stem)
        if match:
            return int(match.group(1))
    return None


def _fingerprint_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(value).encode()).hexdigest()}"


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _has_meaningful_identity_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_meaningful_identity_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_meaningful_identity_value(item) for item in value)
    return value is not None and value != ""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(
                "fingerprint identities must not contain non-finite floats"
            )
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(
        f"fingerprint identity is not JSON-compatible: {type(value).__name__}"
    )
