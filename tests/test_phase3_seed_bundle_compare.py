from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiments.exact_trace_bench.phase3_seed_bundle_compare import (  # noqa: E402
    compare_phase3_seed_bundles,
)


def _write_bundle(path, *, active_features, activation_values, influences, pre, post):
    np.savez_compressed(
        path,
        active_features=np.asarray(active_features, dtype=np.int64),
        activation_values=np.asarray(activation_values, dtype=np.float32),
        seed_feature_influences=np.asarray(influences, dtype=np.float64),
        frontier_pre_locality=np.asarray(pre, dtype=np.int64),
        frontier_post_locality=np.asarray(post, dtype=np.int64),
        queue_size=np.asarray(2, dtype=np.int64),
        actual_max_feature_nodes=np.asarray(4, dtype=np.int64),
        total_active_features=np.asarray(len(active_features), dtype=np.int64),
        status=np.asarray("captured"),
        planner_compute_dtype=np.asarray("float64"),
        influence_compute_dtype=np.asarray("float64"),
    )


def test_compare_phase3_seed_bundles_reports_shared_support_decomposition(
    tmp_path,
) -> None:
    left = tmp_path / "left_phase3_seed_bundle.npz"
    right = tmp_path / "right_phase3_seed_bundle.npz"

    _write_bundle(
        left,
        active_features=[(0, 0, 1), (0, 0, 2), (0, 0, 3)],
        activation_values=[1.0, 0.5, 0.25],
        influences=[0.80, 0.15, 0.05],
        pre=[0, 2],
        post=[0, 2],
    )
    _write_bundle(
        right,
        active_features=[(0, 0, 1), (0, 0, 2), (0, 0, 4)],
        activation_values=[1.0, 0.5, 0.25],
        influences=[0.79, 0.16, 0.05],
        pre=[0, 2],
        post=[0, 2],
    )

    result = compare_phase3_seed_bundles(left, right)

    assert result["support"]["shared_feature_count"] == 2
    assert result["support"]["left_unique_feature_count"] == 1
    assert result["support"]["right_unique_feature_count"] == 1
    assert result["support"]["feature_jaccard"] == 0.5

    assert result["left_support_mass_split"][
        "shared_abs_influence_fraction"
    ] == pytest.approx(0.95)
    assert result["right_support_mass_split"][
        "shared_abs_influence_fraction"
    ] == pytest.approx(0.95)

    assert result["frontier_post_locality"]["jaccard"] == (1 / 3)
    assert result["frontier_post_locality"]["shared_support_only_jaccard"] == 1.0
    assert result["frontier_post_locality"][
        "shared_support_improvement"
    ] == pytest.approx(2 / 3)
    assert (
        result["shared_support_score_stability"]["seed_feature_influences"][
            "shared_count"
        ]
        == 2
    )
    assert (
        result["shared_support_score_stability"]["seed_feature_influences"][
            "topk_overlap"
        ]["64"]["shared_count"]
        == 2
    )
    assert result["unique_support_details"]["left_top_unique_by_abs_influence"][0][
        "feature"
    ] == [0, 0, 3]
    assert result["interpretation"] == "phase3_mostly_explained_by_phase0_support_split"


def test_compare_phase3_seed_bundles_reports_shared_rank_reordering(tmp_path) -> None:
    left = tmp_path / "left_phase3_seed_bundle.npz"
    right = tmp_path / "right_phase3_seed_bundle.npz"
    features = [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]

    _write_bundle(
        left,
        active_features=features,
        activation_values=[4.0, 3.0, 2.0, 1.0],
        influences=[0.4, 0.3, 0.2, 0.1],
        pre=[0, 1, 2, 3],
        post=[0, 1, 2, 3],
    )
    _write_bundle(
        right,
        active_features=features,
        activation_values=[1.0, 2.0, 3.0, 4.0],
        influences=[0.1, 0.2, 0.3, 0.4],
        pre=[3, 2, 1, 0],
        post=[3, 2, 1, 0],
    )

    result = compare_phase3_seed_bundles(left, right)
    influence_stability = result["shared_support_score_stability"][
        "seed_feature_influences"
    ]
    post_rank_drift = result["frontier_post_locality"]["rank_drift"]

    assert influence_stability["shared_count"] == 4
    assert influence_stability["spearman"] == pytest.approx(-1.0)
    assert influence_stability["topk_overlap"]["64"]["overlap_fraction_of_k"] == 1.0
    assert post_rank_drift["shared_frontier_count"] == 4
    assert post_rank_drift["abs_rank_delta_quantiles"]["q100"] == pytest.approx(3.0)
