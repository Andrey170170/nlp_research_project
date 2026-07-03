from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

FEATURE_ID_BASE = 1_000_000


@dataclass(frozen=True)
class FeatureEndpoint:
    layer: int
    position: int
    feature_id: int


@dataclass(frozen=True)
class SignedBucketEdge:
    bucket: str
    row_id: int
    col_id: int
    weight: float


@dataclass(frozen=True)
class SignedGraph:
    path: Path
    step_idx: int
    token_text: str
    logprob: float | None
    feature_ids: np.ndarray
    token_ids: np.ndarray | None
    logit_token_ids: np.ndarray | None
    error_node_shape: tuple[int, ...] | None
    bucket_names: tuple[str, ...]
    bucket_metadata: dict[str, dict[str, Any]]
    bucket_row_idx: np.ndarray
    bucket_col_idx: np.ndarray
    bucket_weights: np.ndarray
    bucket_ids: np.ndarray

    def bucket_edge_arrays(
        self, bucket_name: str, *, max_edges: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if bucket_name not in self.bucket_names:
            empty_int = np.asarray([], dtype=np.int64)
            empty_float = np.asarray([], dtype=np.float64)
            return empty_int, empty_int, empty_float
        bucket_id = self.bucket_names.index(bucket_name)
        mask = self.bucket_ids == bucket_id
        rows = np.asarray(self.bucket_row_idx[mask], dtype=np.int64)
        cols = np.asarray(self.bucket_col_idx[mask], dtype=np.int64)
        weights = np.asarray(self.bucket_weights[mask], dtype=np.float64)
        if max_edges is not None and max_edges > 0 and weights.size > max_edges:
            keep = np.argpartition(np.abs(weights), -max_edges)[-max_edges:]
            keep = keep[np.argsort(-np.abs(weights[keep]))]
            rows = rows[keep]
            cols = cols[keep]
            weights = weights[keep]
        return rows, cols, weights

    def iter_bucket_edges(self, bucket_name: str) -> Iterator[SignedBucketEdge]:
        rows, cols, weights = self.bucket_edge_arrays(bucket_name)
        for row, col, weight in zip(rows, cols, weights):
            yield SignedBucketEdge(bucket_name, int(row), int(col), float(weight))


def encode_feature_endpoint(
    layer: int, position: int, feature_id: int, n_pos: int
) -> int:
    return (
        int(layer) * (int(n_pos) * FEATURE_ID_BASE)
        + int(position) * FEATURE_ID_BASE
        + int(feature_id)
    )


def decode_feature_endpoint(encoded: int, n_pos: int) -> FeatureEndpoint:
    encoded = int(encoded)
    block = int(n_pos) * FEATURE_ID_BASE
    layer = encoded // block
    rem = encoded % block
    position = rem // FEATURE_ID_BASE
    feature_id = rem % FEATURE_ID_BASE
    return FeatureEndpoint(layer=layer, position=position, feature_id=feature_id)


def graph_path_index(path: Path) -> int | None:
    match = re.search(r"token_(\d{6})", str(path))
    return int(match.group(1)) if match else None


def _metadata(data: Any) -> dict[str, dict[str, Any]]:
    if "bucket_metadata_json" not in data.files:
        return {}
    rows = json.loads(str(data["bucket_metadata_json"]))
    return {str(row.get("bucket", "")): row for row in rows}


def load_signed_graph(
    path: str | Path, *, validate_step_path: bool = True
) -> SignedGraph:
    graph_path = Path(path)
    with np.load(str(graph_path), allow_pickle=False) as data:
        step_idx = int(data["step_idx"]) if "step_idx" in data.files else -1
        path_idx = graph_path_index(graph_path)
        if validate_step_path and path_idx is not None and step_idx != path_idx:
            raise ValueError(
                f"graph step_idx {step_idx} does not match path token index {path_idx}: {graph_path}"
            )
        logprob = float(data["logprob"]) if "logprob" in data.files else float("nan")
        names = tuple(str(x) for x in data.get("bucket_names", np.asarray([])).tolist())
        return SignedGraph(
            path=graph_path,
            step_idx=step_idx,
            token_text=str(data["token_text"]) if "token_text" in data.files else "",
            logprob=None if np.isnan(logprob) else logprob,
            feature_ids=data["feature_ids"]
            if "feature_ids" in data.files
            else np.empty((0, 3), dtype=np.int64),
            token_ids=data["token_ids"] if "token_ids" in data.files else None,
            logit_token_ids=data["logit_token_ids"]
            if "logit_token_ids" in data.files
            else None,
            error_node_shape=tuple(int(x) for x in data["error_node_shape"])
            if "error_node_shape" in data.files
            else None,
            bucket_names=names,
            bucket_metadata=_metadata(data),
            bucket_row_idx=data["bucket_row_idx"]
            if "bucket_row_idx" in data.files
            else np.asarray([], dtype=np.int64),
            bucket_col_idx=data["bucket_col_idx"]
            if "bucket_col_idx" in data.files
            else np.asarray([], dtype=np.int64),
            bucket_weights=data["bucket_weights"]
            if "bucket_weights" in data.files
            else np.asarray([], dtype=np.float32),
            bucket_ids=data["bucket_ids"]
            if "bucket_ids" in data.files
            else np.asarray([], dtype=np.int16),
        )


def infer_n_pos_source(graph: SignedGraph) -> tuple[int, str]:
    if graph.error_node_shape and len(graph.error_node_shape) >= 2:
        return int(graph.error_node_shape[-1]), "error_node_shape"
    if (
        graph.feature_ids.size
        and graph.feature_ids.ndim == 2
        and graph.feature_ids.shape[1] >= 2
    ):
        return int(np.max(graph.feature_ids[:, 1])) + 1, "feature_ids_fallback"
    return max(1, graph.step_idx + 1), "step_idx_fallback"


def infer_n_pos(graph: SignedGraph) -> int:
    n_pos, source = infer_n_pos_source(graph)
    if source != "error_node_shape":
        warnings.warn(
            f"inferring n_pos for {graph.path} from {source}; decoded feature "
            "positions may be underestimated if retained feature_ids do not cover "
            "the full sequence",
            RuntimeWarning,
            stacklevel=2,
        )
    return n_pos
