from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    ArtifactProvenance,
    build_typed_compact_graph,
    save_typed_compact_graph,
)


def write_typed_graph(
    path: Path,
    *,
    step_idx: int = 0,
    feature_ids: list[tuple[int, int, int]] | None = None,
    token_ids: list[int] | None = None,
    logit_token_ids: list[int] | None = None,
    error_layer_count: int | None = None,
    token_text: str = "A",
    bucket_values: dict[str, list[list[float]]] | None = None,
) -> None:
    features = feature_ids or [(0, 0, 4), (1, 1, 9)]
    tokens = token_ids or [10, 11]
    logit_tokens = logit_token_ids or [42]
    count = len(features)
    token_count = len(tokens)
    minimum_feature_layers = max(
        2, max(layer for layer, _position, _feature in features) + 1
    )
    feature_layer_count = error_layer_count or minimum_feature_layers
    if feature_layer_count < minimum_feature_layers:
        raise ValueError("error_layer_count cannot exclude a feature layer")
    error_count = feature_layer_count * token_count
    logit_count = len(logit_tokens)
    values: dict[str, Any] = {
        "feature<-feature": torch.eye(count, dtype=torch.float32),
        "feature<-error": torch.ones((count, error_count), dtype=torch.float32),
        "feature<-token": torch.ones((count, token_count), dtype=torch.float32),
        "logit<-feature": torch.ones((logit_count, count), dtype=torch.float32),
        "logit<-error": torch.ones((logit_count, error_count), dtype=torch.float32),
        "logit<-token": torch.ones((logit_count, token_count), dtype=torch.float32),
    }
    for name, matrix in (bucket_values or {}).items():
        tensor = torch.tensor(matrix, dtype=torch.float32)
        if name.endswith("<-error") and tensor.shape[1] == token_count:
            tensor = tensor.repeat(1, feature_layer_count)
        values[name] = tensor
    compact_result = {
        "active_features": torch.tensor(features, dtype=torch.int64),
        "selected_features": torch.arange(count, dtype=torch.int64),
        "feature_row_node_indices": torch.arange(count, dtype=torch.int64),
        "logit_row_node_indices": torch.arange(logit_count, dtype=torch.int64),
        "feature_feature_edges": values["feature<-feature"],
        "feature_error_edges": values["feature<-error"],
        "feature_token_edges": values["feature<-token"],
        "logit_feature_edges": values["logit<-feature"],
        "logit_error_edges": values["logit<-error"],
        "logit_token_edges": values["logit<-token"],
        "n_error_nodes": error_count,
        "n_token_nodes": token_count,
        "input_tokens": torch.tensor(tokens, dtype=torch.int64),
        "logit_targets": [
            SimpleNamespace(vocab_idx=token_id) for token_id in logit_tokens
        ],
        "semantic_fingerprint": "semantic-v1",
        "execution_fingerprint": "execution-v1",
    }
    graph = build_typed_compact_graph(
        compact_result,
        step_idx,
        token_text=token_text,
        logprob=-0.25,
        provenance=ArtifactProvenance(
            provider={"family": "fixture"},
            trace={
                "semantics": "fixture-v1",
                "semantic_fingerprint": "semantic-v1",
                "execution_fingerprint": "execution-v1",
            },
            target={"token_text": token_text},
            step={"step_idx": step_idx},
        ),
    )
    save_typed_compact_graph(graph, path)


def write_historical_tie_cutoff_graph(
    path: Path,
    *,
    canonical_source: Path,
    token_ids: list[int] | None = None,
    logit_token_ids: list[int] | None = None,
    error_node_shape: tuple[int, int] | None = None,
) -> None:
    """Write one mixed-v1 graph with a valid historical tie-cutoff choice."""
    with np.load(canonical_source, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]).copy() for name in data.files}

    bucket_names = tuple(str(value) for value in payload["bucket_names"].tolist())
    feature_bucket_id = bucket_names.index("feature<-feature")
    feature_edges = np.flatnonzero(payload["bucket_ids"] == feature_bucket_id)
    if len(feature_edges) != 2:
        raise ValueError("tie-cutoff fixture requires two retained feature edges")

    first_feature = int(payload["feature_ids"][0, 2])
    layer, position, feature = (int(value) for value in payload["feature_ids"][1])
    n_pos = len(payload["token_ids"])
    second_feature = (
        layer * n_pos * 1_000_000 + position * 1_000_000 + feature
    )
    tail_index = int(feature_edges[1])
    payload["bucket_row_idx"][tail_index] = second_feature
    payload["bucket_col_idx"][tail_index] = first_feature

    canonical_metadata = json.loads(str(payload["bucket_metadata_json"]))
    historical_metadata = [
        {
            "bucket": name,
            "raw_total_abs_mass": canonical_metadata[name]["raw_abs_mass"],
            "retained_abs_mass": canonical_metadata[name]["retained_abs_mass"],
            "retained_fraction": canonical_metadata[name]["retained_fraction"],
            "raw_nnz": canonical_metadata[name]["raw_edge_count"],
            "retained_nnz": canonical_metadata[name]["retained_edge_count"],
            "policy": canonical_metadata[name]["policy"],
            "weights_signed": True,
        }
        for name in bucket_names
    ]
    historical_payload = {
        "row_idx": np.asarray([0], dtype=np.int32),
        "col_idx": np.asarray([0], dtype=np.int32),
        "weights": np.asarray([1.0], dtype=np.float32),
        "feature_ids": payload["feature_ids"],
        "token_text": payload["token_text"],
        "logprob": payload["logprob"],
        "n_features": np.asarray(len(payload["feature_ids"]), dtype=np.int32),
        "step_idx": payload["step_idx"],
        "compact_save_format": np.asarray("typed_bucketed"),
        "bucket_row_idx": payload["bucket_row_idx"],
        "bucket_col_idx": payload["bucket_col_idx"],
        "bucket_weights": payload["bucket_weights"],
        "bucket_ids": payload["bucket_ids"],
        "bucket_names": payload["bucket_names"],
        "bucket_metadata_json": np.asarray(json.dumps(historical_metadata)),
        "error_node_shape": (
            payload["error_node_shape"]
            if error_node_shape is None
            else np.asarray(error_node_shape, dtype=np.int32)
        ),
        "token_ids": (
            payload["token_ids"]
            if token_ids is None
            else np.asarray(token_ids, dtype=np.int64)
        ),
        "logit_token_ids": (
            payload["logit_token_ids"]
            if logit_token_ids is None
            else np.asarray(logit_token_ids, dtype=np.int64)
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **historical_payload)
