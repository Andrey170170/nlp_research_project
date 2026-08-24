from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    load_typed_compact_graph,
)

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


def load_signed_graph(
    path: str | Path, *, validate_step_path: bool = True
) -> SignedGraph:
    graph_path = Path(path)
    compact = load_typed_compact_graph(
        graph_path, validate_step_path=validate_step_path
    )
    return SignedGraph(
        path=graph_path,
        step_idx=compact.step_idx,
        token_text=compact.token_text,
        logprob=compact.logprob,
        feature_ids=compact.feature_ids,
        token_ids=compact.token_ids,
        logit_token_ids=compact.logit_token_ids,
        error_node_shape=compact.error_node_shape,
        bucket_names=compact.bucket_names,
        bucket_metadata=compact.bucket_metadata,
        bucket_row_idx=compact.bucket_row_idx,
        bucket_col_idx=compact.bucket_col_idx,
        bucket_weights=compact.bucket_weights,
        bucket_ids=compact.bucket_ids,
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
