from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from nlp_research_project.exact_trace_bench.graph_compare import (  # noqa: E402
    compare_artifact_dirs,
    compare_step_pair,
)


@dataclass
class SimpleStep:
    step_idx: int
    row_idx: np.ndarray
    col_idx: np.ndarray
    weights: np.ndarray
    feature_ids: np.ndarray
    token_text: str
    logprob: float | None
    n_features: int


def _step(*, feature_ids, rows, cols, weights) -> SimpleStep:
    return SimpleStep(
        step_idx=0,
        row_idx=np.asarray(rows, dtype=np.int32),
        col_idx=np.asarray(cols, dtype=np.int32),
        weights=np.asarray(weights, dtype=np.float32),
        feature_ids=np.asarray(feature_ids, dtype=np.int64),
        token_text="Let",
        logprob=-0.1,
        n_features=len(feature_ids),
    )


def test_compare_step_pair_reports_shared_unique_edge_decomposition() -> None:
    left = _step(
        feature_ids=[(0, 0, 1), (0, 0, 2), (0, 0, 3)],
        rows=[1, 2, 3],
        cols=[0, 0, 0],
        weights=[0.4, 0.2, 0.4],
    )
    right = _step(
        feature_ids=[(0, 0, 1), (0, 0, 2), (0, 0, 4)],
        rows=[1, 2, 3],
        cols=[0, 1, 2],
        weights=[0.5, 0.2, 0.3],
    )

    result = compare_step_pair(cast(Any, left), cast(Any, right))

    assert result["feature_support_decomposition"]["shared_count"] == 2
    assert result["feature_support_decomposition"]["left_unique_count"] == 1
    assert result["feature_support_decomposition"]["right_unique_count"] == 1

    left_classes = result["edge_class_decomposition_a"]
    right_classes = result["edge_class_decomposition_b"]
    assert left_classes["shared_to_shared"]["edge_count"] == 1
    assert left_classes["shared_to_unique"]["edge_count"] == 1
    assert left_classes["shared_to_logit"]["edge_count"] == 1
    assert right_classes["shared_to_shared"]["edge_count"] == 1
    assert right_classes["shared_to_unique"]["edge_count"] == 1
    assert right_classes["unique_to_logit"]["edge_count"] == 1
    assert left_classes["shared_to_logit"]["mass_fraction"] == pytest.approx(0.4)

    shared_edge_stability = result["shared_endpoint_edge_stability"]
    assert shared_edge_stability["common_edge_count"] == 1
    assert shared_edge_stability["edge_jaccard"] == 1.0
    assert shared_edge_stability["weighted_edge_jaccard"] == pytest.approx(0.8)
    assert shared_edge_stability["topk_overlap"]["64"]["shared_count"] == 1

    assert result["all_edge_weighted_jaccard"] < 1.0


def test_topk_overlap_penalizes_edges_present_on_only_one_side() -> None:
    left = _step(
        feature_ids=[(0, 0, 1), (0, 0, 2), (0, 0, 3)],
        rows=[1],
        cols=[0],
        weights=[1.0],
    )
    right = _step(
        feature_ids=[(0, 0, 1), (0, 0, 2), (0, 0, 3)],
        rows=[1, 2],
        cols=[0, 0],
        weights=[1.0, 0.5],
    )

    result = compare_step_pair(cast(Any, left), cast(Any, right))
    top256 = result["topk_edge_overlap"]["256"]

    assert top256["left_k_effective"] == 1
    assert top256["right_k_effective"] == 2
    assert top256["jaccard"] == 0.5


def test_topk_overlap_defines_both_empty_as_identical() -> None:
    left = _step(feature_ids=[(0, 0, 1)], rows=[], cols=[], weights=[])
    right = _step(feature_ids=[(0, 0, 1)], rows=[], cols=[], weights=[])

    result = compare_step_pair(cast(Any, left), cast(Any, right))

    assert result["topk_edge_overlap"]["256"]["jaccard"] == 1.0


def _write_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), payload=np.asarray([1], dtype=np.int32))


def test_compare_artifact_dirs_ignores_auxiliary_step_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left_completion = tmp_path / "left" / "prompt_000" / "completion_000"
    right_completion = tmp_path / "right" / "prompt_000" / "completion_000"
    _write_npz(left_completion / "step_000.npz")
    _write_npz(right_completion / "step_000.npz")
    _write_npz(left_completion / "step_000_feature_semantic_descriptors.npz")
    _write_npz(right_completion / "step_000_phase3_seed_bundle.npz")

    loaded_names: list[str] = []

    def load_compact(path: Path) -> SimpleStep:
        loaded_names.append(path.name)
        return _step(
            feature_ids=[(0, 0, 1)],
            rows=[0],
            cols=[0],
            weights=[1.0],
        )

    from nlp_research_project.exact_trace_bench import compact_io

    monkeypatch.setattr(compact_io, "load_compact", load_compact)

    result = compare_artifact_dirs(tmp_path / "left", tmp_path / "right")

    assert loaded_names == ["step_000.npz", "step_000.npz"]
    assert result["shared_completion_count"] == 1
    assert result["aligned_completion_count"] == 1
    assert result["aligned_step_count"] == 1
    assert result["comparison_complete"] is True
    assert len(result["step_comparisons"]) == 1
    assert result["step_comparisons"][0]["feature_jaccard"] == 1.0
    assert result["completion_comparisons"][0]["mean_top256_edge_jaccard"] == 1.0
    assert result["overall_mean_top256_edge_jaccard"] == 1.0
    assert result["step_comparisons"][0]["topk_edge_overlap"]["256"]["jaccard"] == 1.0


def test_compare_artifact_dirs_reports_incomplete_step_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left_completion = tmp_path / "left" / "prompt_000" / "completion_000"
    right_completion = tmp_path / "right" / "prompt_000" / "completion_000"
    _write_npz(left_completion / "step_000.npz")
    _write_npz(left_completion / "step_001.npz")
    _write_npz(right_completion / "step_000.npz")

    from nlp_research_project.exact_trace_bench import compact_io

    monkeypatch.setattr(
        compact_io,
        "load_compact",
        lambda path: SimpleStep(
            **{
                **_step(
                    feature_ids=[(0, 0, 1)],
                    rows=[0],
                    cols=[0],
                    weights=[1.0],
                ).__dict__,
                "step_idx": int(path.stem.split("_")[1]),
            }
        ),
    )

    result = compare_artifact_dirs(tmp_path / "left", tmp_path / "right")

    assert result["aligned_step_count"] == 1
    assert result["comparison_complete"] is False
    assert result["completion_comparisons"][0]["left_only_step_count"] == 1
