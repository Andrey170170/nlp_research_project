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
    assert result["interpretation"] == "phase3_mostly_explained_by_phase0_support_split"
