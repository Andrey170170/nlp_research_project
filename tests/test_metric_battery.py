from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nlp_research_project.exact_trace_bench.full_answer.metric_battery import (
    metric_battery_rows,
    rbo_ext,
    soft_feature_matching_rows,
    top_p_core_rows,
    union_support_similarity_rows,
    weighted_jaccard,
)
from nlp_research_project.exact_trace_bench.full_answer.temporal import GraphSnapshot


class FakeDecoderStore:
    def cosine_matrix(
        self, layer: int, left_feature_ids: list[int], right_feature_ids: list[int]
    ) -> np.ndarray:
        values = np.zeros(
            (len(left_feature_ids), len(right_feature_ids)), dtype=np.float32
        )
        for i, left in enumerate(left_feature_ids):
            for j, right in enumerate(right_feature_ids):
                if left == right:
                    values[i, j] = 1.0
                elif {left, right} == {1, 2}:
                    values[i, j] = 0.85
                else:
                    values[i, j] = 0.10
        return values


def _encode_feature_feature_id(
    layer: int, position: int, feature_idx: int, *, n_pos: int = 2
) -> int:
    return layer * n_pos * 1_000_000 + position * 1_000_000 + feature_idx


def _snapshot(
    index: int,
    features: set[tuple[int, int, int]],
    all_edges: dict[tuple[object, object], float],
    bucket_edges: dict[str, dict[tuple[object, object], float]] | None = None,
) -> GraphSnapshot:
    return GraphSnapshot(
        generated_index=index,
        token_text=f"tok{index}",
        features=features,
        edges={k: v for k, v in all_edges.items() if k[0] != ("logit", 0)},
        all_edges=all_edges,
        bucket_edges=bucket_edges or {},
        bucket_metadata={"feature<-feature": {"error_node_shape": [1, 2]}},
        error_node_shape=(1, 2),
    )


def _metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], Any]:
    return {
        (str(row["bucket"]), str(row["metric"]), str(row["params_json"])): row["value"]
        for row in rows
    }


def test_weighted_jaccard_edge_cases() -> None:
    assert weighted_jaccard({}, {}) is None
    assert weighted_jaccard({"a": 1.0}, {"a": 1.0}) == pytest.approx(1.0)
    assert weighted_jaccard({"a": 1.0}, {"b": 1.0}) == pytest.approx(0.0)
    assert weighted_jaccard(
        {"a": 1.0, "b": 3.0}, {"a": 2.0, "b": 1.0}
    ) == pytest.approx(2 / 5)


def test_rbo_ext_hand_computed_cases() -> None:
    assert rbo_ext(["a", "b", "c"], ["a", "b", "c"], p=0.9) == pytest.approx(1.0)
    assert rbo_ext(["a", "b"], ["c", "d"], p=0.9) == pytest.approx(0.0)
    assert rbo_ext(["a", "b"], ["b", "a"], p=0.5) == pytest.approx(0.5)


def test_top_p_core_boundary_includes_crossing_item() -> None:
    rows = top_p_core_rows(
        {"a": 0.6, "b": 0.4},
        {"a": 0.5, "b": 0.5},
        bucket="unit",
        thresholds=[0.5, 0.95],
    )
    lookup = _metric_lookup(rows)
    assert lookup[("unit", "top_p_core_size_a", '{"p":0.5}')] == 1
    assert lookup[("unit", "top_p_core_size_b", '{"p":0.5}')] == 1
    assert lookup[("unit", "top_p_core_size_a", '{"p":0.95}')] == 2


def test_union_support_similarity_reports_cosine_and_spearman() -> None:
    rows = union_support_similarity_rows(
        {"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 4.0}, bucket="unit"
    )
    lookup = _metric_lookup(rows)
    assert lookup[("unit", "union_raw_weight_cosine", "{}")] == pytest.approx(1.0)
    assert lookup[("unit", "intersection_spearman", "{}")] == pytest.approx(1.0)
    assert lookup[("unit", "intersection_mass_fraction_a", "{}")] == pytest.approx(1.0)


def test_soft_feature_matching_uses_decoder_cosine_greedily() -> None:
    rows = soft_feature_matching_rows(
        {(0, 1): 2.0, (0, 9): 1.0},
        {(0, 2): 3.0, (0, 9): 1.0},
        decoder_store=FakeDecoderStore(),  # type: ignore[arg-type]
        thresholds=[0.8, 0.9],
    )
    lookup = _metric_lookup(rows)
    assert (
        lookup[
            ("positionless_feature_nodes", "decoder_soft_match_count", '{"tau":0.8}')
        ]
        == 2
    )
    assert (
        lookup[
            ("positionless_feature_nodes", "decoder_soft_match_count", '{"tau":0.9}')
        ]
        == 1
    )
    assert lookup[
        (
            "positionless_feature_nodes",
            "decoder_soft_matched_mass_fraction_a",
            '{"tau":0.8}',
        )
    ] == pytest.approx(1.0)


def test_metric_battery_rows_cover_edges_nodes_flows_and_soft_matching() -> None:
    left_ff_source = _encode_feature_feature_id(0, 0, 1)
    left_ff_target = _encode_feature_feature_id(1, 1, 2)
    right_ff_source = _encode_feature_feature_id(0, 1, 2)
    right_ff_target = _encode_feature_feature_id(1, 0, 2)
    left = _snapshot(
        0,
        {(0, 0, 1), (1, 1, 2)},
        {
            (("feature", 1, 1, 2), ("feature", 0, 0, 1)): 2.0,
            (("logit", 0), ("feature", 1, 1, 2)): 1.0,
        },
        {"feature<-feature": {(left_ff_target, left_ff_source): 2.0}},
    )
    right = _snapshot(
        1,
        {(0, 1, 2), (1, 0, 2)},
        {
            (("feature", 1, 0, 2), ("feature", 0, 1, 2)): 3.0,
            (("logit", 0), ("feature", 1, 0, 2)): 1.0,
        },
        {"feature<-feature": {(right_ff_target, right_ff_source): 3.0}},
    )

    rows = metric_battery_rows(
        left,
        right,
        decoder_store=FakeDecoderStore(),  # type: ignore[arg-type]
        derived_ks=[1],
        top_p_thresholds=[0.8],
        rbo_persistence=[0.9],
        soft_thresholds=[0.8],
    )
    keys = {(row["bucket"], row["metric"]) for row in rows}
    assert ("all_edges", "derived_k_weighted_jaccard") in keys
    assert ("feature<-feature", "top_p_core_weighted_jaccard") in keys
    assert ("positionless_feature_nodes", "rbo_ext") in keys
    assert ("feature_feature_layer_flow", "total_variation_distance") in keys
    assert ("positionless_feature_nodes", "decoder_soft_weighted_jaccard") in keys
