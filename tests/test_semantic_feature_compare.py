from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from nlp_research_project.exact_trace_bench.semantic_feature_compare import (  # noqa: E402
    compare_semantic_feature_descriptors,
)


def _write_descriptor(
    path: Path, *, features, sketches, influences, activations
) -> None:
    np.savez_compressed(
        path,
        candidate_features=np.asarray(features, dtype=np.int64),
        candidate_row_indices=np.arange(len(features), dtype=np.int64),
        activation_value=np.asarray(activations, dtype=np.float32),
        seed_influence=np.asarray(influences, dtype=np.float64),
        seed_rank=np.arange(len(features), dtype=np.int64),
        is_top_seed=np.ones(len(features), dtype=bool),
        is_frontier_pre=np.zeros(len(features), dtype=bool),
        frontier_pre_rank=np.full(len(features), -1, dtype=np.int64),
        is_frontier_post=np.zeros(len(features), dtype=bool),
        frontier_post_rank=np.full(len(features), -1, dtype=np.int64),
        is_selected_phase4=np.zeros(len(features), dtype=bool),
        phase4_selected_rank=np.full(len(features), -1, dtype=np.int64),
        semantic_sketch=np.asarray(sketches, dtype=np.float32),
        status=np.asarray("captured"),
        descriptor_version=np.asarray("v1"),
        descriptor_kind=np.asarray("test_descriptor"),
        descriptor_dim=np.asarray(len(sketches[0]), dtype=np.int64),
        semantic_descriptor_top_k=np.asarray(8, dtype=np.int64),
        candidate_count=np.asarray(len(features), dtype=np.int64),
        total_active_features=np.asarray(len(features), dtype=np.int64),
        phase4_selection_available=np.asarray(False),
        seed_influence_available=np.asarray(True),
    )


def test_compare_semantic_feature_descriptors_matches_substitutes(tmp_path) -> None:
    left = tmp_path / "left_feature_semantic_descriptors.npz"
    right = tmp_path / "right_feature_semantic_descriptors.npz"

    _write_descriptor(
        left,
        features=[(0, 0, 1), (0, 0, 2), (1, 3, 10)],
        sketches=[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        influences=[0.9, 0.5, 0.4],
        activations=[1.0, 0.5, 0.25],
    )
    _write_descriptor(
        right,
        features=[(0, 0, 1), (0, 0, 4), (1, 3, 11)],
        sketches=[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        influences=[0.88, 0.5, 0.4],
        activations=[1.1, 0.55, 0.20],
    )

    result = compare_semantic_feature_descriptors(
        left, right, similarity_threshold=0.99
    )

    assert result["candidate_support"]["shared_candidate_count"] == 1
    assert result["candidate_support"]["left_only_candidate_count"] == 2
    assert result["semantic_substitute_coverage"]["high_confidence_count"] == 2
    assert (
        result["semantic_substitute_coverage"]["left_high_confidence_mass_fraction"]
        == 1.0
    )
    assert result["interpretation"] == "semantic_substitutes_explain_mismatch"


def test_compare_semantic_feature_descriptors_reports_unmatched_uniques(
    tmp_path,
) -> None:
    left = tmp_path / "left_feature_semantic_descriptors.npz"
    right = tmp_path / "right_feature_semantic_descriptors.npz"

    _write_descriptor(
        left,
        features=[(0, 0, 1), (0, 0, 2)],
        sketches=[[1.0, 0.0], [0.0, 1.0]],
        influences=[0.9, 0.5],
        activations=[1.0, 0.5],
    )
    _write_descriptor(
        right,
        features=[(0, 0, 1), (0, 0, 4)],
        sketches=[[1.0, 0.0], [1.0, 0.0]],
        influences=[0.88, 0.5],
        activations=[1.1, 0.55],
    )

    result = compare_semantic_feature_descriptors(
        left, right, similarity_threshold=0.99
    )

    assert result["semantic_substitute_coverage"]["high_confidence_count"] == 0
    assert result["unmatched_candidates"]["left_top_unmatched_by_abs_influence"][0][
        "feature"
    ] == [0, 0, 2]
    assert result["interpretation"] == "unique_features_semantically_unmatched"
