from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from nlp_research_project.exact_trace_bench import phase0_replay_matrix_compare as matrix_compare  # noqa: E402


def _mkdirs(tmp_path: Path) -> dict[str, Path]:
    names = (
        "ascend_baseline",
        "cardinal_baseline",
        "ascend_self_replay",
        "cardinal_self_replay",
        "ascend_with_cardinal",
        "cardinal_with_ascend",
    )
    roots = {name: tmp_path / name for name in names}
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    return roots


def test_compare_phase0_replay_matrix_reports_cross_swap_movement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _mkdirs(tmp_path)

    compact_scores = {
        ("ascend_with_cardinal", "ascend_baseline"): (0.30, 0.40),
        ("ascend_with_cardinal", "cardinal_baseline"): (0.80, 0.90),
        ("cardinal_with_ascend", "cardinal_baseline"): (0.25, 0.35),
        ("cardinal_with_ascend", "ascend_baseline"): (0.70, 0.75),
    }
    phase3_scores = {
        ("ascend_with_cardinal", "ascend_baseline"): (0.20, 0.30, 0.60, 0.10),
        ("ascend_with_cardinal", "cardinal_baseline"): (0.90, 0.95, 0.98, 0.80),
        ("cardinal_with_ascend", "cardinal_baseline"): (0.22, 0.25, 0.55, 0.12),
        ("cardinal_with_ascend", "ascend_baseline"): (0.88, 0.92, 0.97, 0.79),
    }

    def fake_compare_artifact_dirs(left: Path, right: Path):
        feature, weighted = compact_scores.get((left.name, right.name), (1.0, 1.0))
        return {
            "overall_mean_feature_jaccard": feature,
            "overall_mean_weighted_edge_jaccard": weighted,
        }

    def fake_compare_phase3_artifact_dirs(left: Path, right: Path):
        support, pearson, top1024, frontier = phase3_scores.get(
            (left.name, right.name),
            (1.0, 1.0, 1.0, 1.0),
        )
        return {
            "overall_mean_support_jaccard": support,
            "overall_mean_seed_influence_pearson": pearson,
            "overall_mean_seed_top1024_overlap": top1024,
            "overall_mean_frontier_post_jaccard": frontier,
        }

    monkeypatch.setattr(
        matrix_compare,
        "compare_artifact_dirs",
        fake_compare_artifact_dirs,
    )
    monkeypatch.setattr(
        matrix_compare,
        "_compare_phase3_artifact_dirs",
        fake_compare_phase3_artifact_dirs,
    )
    monkeypatch.setattr(
        matrix_compare,
        "_compare_semantic_artifact_dirs",
        lambda _left, _right: {"status": "not_available"},
    )

    result = matrix_compare.compare_phase0_replay_matrix(
        roots["ascend_baseline"],
        roots["cardinal_baseline"],
        roots["ascend_self_replay"],
        roots["cardinal_self_replay"],
        roots["ascend_with_cardinal"],
        roots["cardinal_with_ascend"],
    )

    ascend_move = result["movement_scores"]["ascend_with_cardinal"]
    assert ascend_move["compact_feature_jaccard"]["movement_score"] == pytest.approx(
        0.5
    )
    assert ascend_move["compact_weighted_edge_jaccard"][
        "movement_score"
    ] == pytest.approx(0.5)
    assert ascend_move["phase3_support_jaccard"]["movement_score"] == pytest.approx(0.7)
    assert ascend_move["phase3_seed_influence_pearson"][
        "movement_score"
    ] == pytest.approx(0.65)
    assert ascend_move["phase3_frontier_post_jaccard"][
        "movement_score"
    ] == pytest.approx(0.7)


def test_compare_phase0_replay_matrix_self_replay_gate_uses_strict_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _mkdirs(tmp_path)

    replay_manifest = (
        roots["ascend_self_replay"]
        / "prompt_000"
        / "completion_000"
        / "completion.json"
    )
    replay_manifest.parent.mkdir(parents=True, exist_ok=True)
    replay_manifest.write_text(
        json.dumps(
            {
                "phase0_replay_dtype_roundtrip_loss": True,
                "steps": [{"phase0_replay_dtype_roundtrip_loss": True}],
            }
        ),
        encoding="utf-8",
    )

    def fake_compare_artifact_dirs(left: Path, right: Path):
        if left.name == "ascend_baseline" and right.name == "ascend_self_replay":
            return {
                "overall_mean_feature_jaccard": 1.0,
                "overall_mean_weighted_edge_jaccard": 0.998,
            }
        return {
            "overall_mean_feature_jaccard": 1.0,
            "overall_mean_weighted_edge_jaccard": 1.0,
        }

    def fake_compare_phase3_artifact_dirs(left: Path, right: Path):
        if left.name == "ascend_baseline" and right.name == "ascend_self_replay":
            return {
                "overall_mean_support_jaccard": 1.0,
                "overall_mean_seed_influence_pearson": 0.99995,
                "overall_mean_seed_top1024_overlap": 1.0,
                "overall_mean_frontier_post_jaccard": 1.0,
            }
        return {
            "overall_mean_support_jaccard": 1.0,
            "overall_mean_seed_influence_pearson": 1.0,
            "overall_mean_seed_top1024_overlap": 1.0,
            "overall_mean_frontier_post_jaccard": 1.0,
        }

    monkeypatch.setattr(
        matrix_compare,
        "compare_artifact_dirs",
        fake_compare_artifact_dirs,
    )
    monkeypatch.setattr(
        matrix_compare,
        "_compare_phase3_artifact_dirs",
        fake_compare_phase3_artifact_dirs,
    )

    result = matrix_compare.compare_phase0_replay_matrix(
        roots["ascend_baseline"],
        roots["cardinal_baseline"],
        roots["ascend_self_replay"],
        roots["cardinal_self_replay"],
        roots["ascend_with_cardinal"],
        roots["cardinal_with_ascend"],
        include_semantic=False,
    )

    ascend_gate = result["self_replay_gate"]["ascend_self_replay"]
    assert ascend_gate["pass"] is False
    assert ascend_gate["checks"]["compact_weighted_edge_jaccard"]["pass"] is False
    assert ascend_gate["dtype_roundtrip_loss_detected"] is True
    assert any(
        "strict self-replay thresholds" in warning
        for warning in ascend_gate["warnings"]
    )

    cardinal_gate = result["self_replay_gate"]["cardinal_self_replay"]
    assert cardinal_gate["pass"] is True
    assert cardinal_gate["dtype_roundtrip_loss_detected"] is False
