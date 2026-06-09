from __future__ import annotations

from pathlib import Path

import numpy as np

from nlp_research_project.exact_trace_bench.full_answer.stability import (
    compare_token_stability,
)


def _write_phase0(path: Path, features: np.ndarray, values: np.ndarray) -> None:
    np.savez_compressed(
        path,
        active_features=features.astype(np.int64),
        activation_values=values.astype(np.float32),
        active_feature_count=np.asarray(features.shape[0], dtype=np.int64),
        active_feature_membership_hash_canonical=np.asarray("hash"),
        active_feature_values_hash=np.asarray("values"),
        target_logit_hash=np.asarray("target"),
    )


def _write_phase3(path: Path, features: np.ndarray, scores: np.ndarray) -> None:
    order = np.argsort(scores)[::-1]
    np.savez_compressed(
        path,
        active_features=features.astype(np.int64),
        activation_values=np.ones(features.shape[0], dtype=np.float32),
        seed_feature_influences=scores.astype(np.float32),
        frontier_pre_locality=order[:2].astype(np.int64),
        frontier_post_locality=order[:2].astype(np.int64),
        queue_size=np.asarray(2, dtype=np.int64),
        actual_max_feature_nodes=np.asarray(2, dtype=np.int64),
        total_active_features=np.asarray(features.shape[0], dtype=np.int64),
        status=np.asarray("captured"),
        planner_compute_dtype=np.asarray("float32"),
        influence_compute_dtype=np.asarray("float32"),
    )


def test_compare_token_stability_summarizes_phase0_and_phase3(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    left_features = np.asarray([[0, 0, 1], [0, 1, 2], [1, 0, 3]], dtype=np.int64)
    right_features = np.asarray([[0, 0, 1], [1, 0, 3], [1, 1, 4]], dtype=np.int64)

    _write_phase0(
        left / "phase0_donor_bundle.npz", left_features, np.asarray([1.0, 2.0, 3.0])
    )
    _write_phase0(
        right / "phase0_donor_bundle.npz", right_features, np.asarray([1.1, 3.1, 4.0])
    )
    _write_phase3(
        left / "phase3_seed_bundle.npz", left_features, np.asarray([0.9, 0.8, 0.1])
    )
    _write_phase3(
        right / "phase3_seed_bundle.npz", right_features, np.asarray([0.95, 0.2, 0.7])
    )

    output_json = tmp_path / "comparison.json"
    result = compare_token_stability(left, right, output_json=output_json)

    assert output_json.exists()
    assert result["phase0"]["shared_active_feature_count"] == 2
    assert result["phase0"]["left_only_active_feature_count"] == 1
    assert result["phase0"]["right_only_active_feature_count"] == 1
    assert result["phase0"]["active_feature_jaccard"] == 0.5
    assert result["phase3_seed"]["topk_overlap"]["128"]["shared_count"] == 2
    assert result["phase3_seed"]["frontier_pre_locality_overlap"]["jaccard"] == 1 / 3
