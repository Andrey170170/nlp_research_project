from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nlp_research_project.exact_trace_bench.full_answer.diagnostics import (
    diagnose_full_answer_stability,
)
from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    CANONICAL_BUCKET_NAMES,
)
from typed_graph_fixtures import write_typed_graph


def _write_token_dir(path: Path, *, features: np.ndarray, scores: np.ndarray) -> None:
    path.mkdir()
    (path / "trace.json").write_text(
        json.dumps(
            {
                "target_position": 4,
                "generated_index": 1,
                "target_token_id": 99,
                "target_token_text": " token",
                "prefix_hash": "top_level_ignored",
                "prefix_view_metadata": {
                    "prefix_token_ids_sha256": "abc",
                    "input_token_ids_sha256": "def",
                },
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        path / "phase0_donor_bundle.npz",
        active_features=features,
        activation_values=np.ones(features.shape[0], dtype=np.float32),
    )
    order = np.argsort(np.abs(scores))[::-1]
    np.savez_compressed(
        path / "phase3_seed_bundle.npz",
        active_features=features,
        seed_feature_influences=scores.astype(np.float32),
        frontier_pre_locality=order[:2].astype(np.int64),
        frontier_post_locality=order[:1].astype(np.int64),
    )
    write_typed_graph(
        path / "graph.npz",
        feature_ids=[tuple(int(value) for value in row) for row in features],
        token_ids=list(range(32)),
        logit_token_ids=[99],
        token_text=" token",
    )


def test_diagnose_full_answer_stability_reports_rich_metrics(tmp_path: Path) -> None:
    left_features = np.asarray([[0, 0, 1], [0, 1, 2], [1, 11, 3]], dtype=np.int64)
    right_features = np.asarray([[0, 0, 1], [1, 11, 3], [2, 20, 4]], dtype=np.int64)
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_token_dir(left, features=left_features, scores=np.asarray([10.0, 5.0, 1.0]))
    _write_token_dir(right, features=right_features, scores=np.asarray([9.0, 4.0, 2.0]))

    output = tmp_path / "nested" / "diagnostics.json"
    result = diagnose_full_answer_stability(left, right, output_json=output)

    assert output.exists()
    assert result["artifact_capabilities"] == {
        "trace_json": True,
        "phase0_donor_bundle": True,
        "phase3_seed_bundle": True,
        "compact_graph": True,
    }
    assert result["alignment_checks"]["target_position_equal"] is True
    assert result["alignment_checks"]["left_prefix_token_ids_sha256"] == "abc"
    assert result["alignment_checks"]["left_input_token_ids_sha256"] == "def"
    assert result["summary"]["compact_feature_count_jaccard"] == 0.5
    assert result["summary"]["phase0_active_jaccard"] == 0.5
    assert all(
        result["summary"][
            f"bucket_{name.replace('<-', '_').replace('-', '_')}_support_jaccard"
        ]
        is not None
        for name in CANONICAL_BUCKET_NAMES
    )
    assert all(
        result["summary"][
            f"bucket_{name.replace('<-', '_').replace('-', '_')}_weighted_jaccard"
        ]
        is not None
        for name in CANONICAL_BUCKET_NAMES
    )
    assert "weighted_edge_jaccard" not in result["summary"]
    assert (
        "all six typed buckets" in result["summary"]["graph_node_mass_aggregation_rule"]
    )
    assert result["summary"]["phase3_active_weighted_jaccard"] is not None
    assert "32" in result["rank_overlap_curves"]["phase3_abs_seed"]
    assert "by_layer" in result["grouped_decompositions"]
    assert "layer_position" in result["feature_grouping"]
    assert result["feature_grouping"]["layer_position"]["shared_group_count"] == 2
    assert result["top_disagreements"]["left_unique"][0]["key"] == [0, 1, 2]


def test_diagnose_full_answer_stability_rejects_legacy_compact_graphs(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for path in (left / "graph.npz", right / "graph.npz"):
        np.savez_compressed(
            path,
            feature_ids=np.asarray([[0, 0, 1]], dtype=np.int64),
            row_idx=np.asarray([0], dtype=np.int64),
            col_idx=np.asarray([0], dtype=np.int64),
            weights=np.asarray([1.0], dtype=np.float32),
        )

    with pytest.raises(ValueError, match="legacy edge fields"):
        diagnose_full_answer_stability(left, right)


def test_diagnose_full_answer_stability_handles_missing_artifacts(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    result = diagnose_full_answer_stability(left, right)

    assert result["artifact_capabilities"]["compact_graph"] is False
    assert result["summary"] == {}
