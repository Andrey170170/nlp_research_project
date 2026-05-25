from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from nlp_research_project.exact_trace_bench.full_answer.temporal import (
    analyze_full_answer_temporal,
)
from nlp_research_project.exact_trace_bench.full_answer.temporal_plots import (
    plot_full_answer_temporal,
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


def _step(index: int, feature_ids: list[tuple[int, int, int]]) -> SimpleStep:
    # Feature 0 is stable; feature 1 varies. Include one feature edge and one
    # logit edge so the all-edge view exercises logit targets.
    return SimpleStep(
        step_idx=index,
        row_idx=np.asarray([1, len(feature_ids)], dtype=np.int32),
        col_idx=np.asarray([0, 0], dtype=np.int32),
        weights=np.asarray([1.0, 2.0 + index], dtype=np.float32),
        feature_ids=np.asarray(feature_ids, dtype=np.int64),
        token_text=f"tok{index}",
        logprob=None,
        n_features=len(feature_ids),
    )


def _write_graph(path: Path, step: SimpleStep) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row_idx=step.row_idx,
        col_idx=step.col_idx,
        weights=step.weights,
        feature_ids=step.feature_ids,
        token_text=np.array(step.token_text),
        logprob=np.array(np.nan),
        n_features=np.array(step.n_features, dtype=np.int32),
        step_idx=np.array(step.step_idx, dtype=np.int32),
    )


def test_temporal_analyzer_adjacent_and_rolling_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    steps = {
        0: _step(0, [(0, 0, 1), (0, 0, 2)]),
        1: _step(1, [(0, 0, 1), (0, 0, 3)]),
        2: _step(2, [(0, 0, 1), (0, 0, 3)]),
    }
    for index, step in steps.items():
        _write_graph(
            run_root / "shards" / "shard_000" / f"token_{index:06d}" / "graph.npz",
            step,
        )

    out_dir = tmp_path / "out"
    summary = analyze_full_answer_temporal(
        run_root=run_root, output_dir=out_dir, windows=[2], lags=[1, 2]
    )

    assert summary["token_count"] == 3
    assert summary["missing_indices"] == []
    adjacent = [
        json.loads(line)
        for line in (out_dir / "adjacent_pairs.jsonl").read_text().splitlines()
    ]
    assert len(adjacent) == 2
    assert adjacent[0]["feature_jaccard"] == pytest.approx(1 / 3)
    assert adjacent[0]["features_entered"] == 1
    assert adjacent[0]["features_exited"] == 1
    assert adjacent[0]["all_edge_weighted_jaccard"] < 1.0
    assert adjacent[0]["positionless_feature_jaccard"] == pytest.approx(1 / 3)
    assert "all_edge_top128_weighted_jaccard" in adjacent[0]
    assert "all_edge_core80_shared_mass_fraction_a" in adjacent[0]
    assert "layer_flow_weighted_jaccard" in adjacent[0]
    assert adjacent[1]["feature_jaccard"] == 1.0

    rolling = [
        json.loads(line)
        for line in (out_dir / "rolling_windows.jsonl").read_text().splitlines()
    ]
    assert len(rolling) == 2
    assert rolling[0]["feature_union_size"] == 3
    assert rolling[0]["positionless_feature_union_size"] == 3
    assert "all_edge_persistence50_mass_fraction" in rolling[0]
    assert rolling[0]["feature_intersection_core_size"] == 1
    assert rolling[1]["feature_intersection_core_size"] == 2
    assert rolling[1]["feature_union_entered"] == 0
    assert rolling[1]["feature_union_exited"] == 1

    lag_rows = [
        json.loads(line)
        for line in (out_dir / "lag_pairs.jsonl").read_text().splitlines()
    ]
    assert [row["lag"] for row in lag_rows] == [1, 1, 2]

    for name in (
        "token_timeline.jsonl",
        "cumulative_core.jsonl",
        "layer_flow_by_token.jsonl",
    ):
        assert (out_dir / name).exists()
        assert (out_dir / name).read_text().strip()
    timeline = [
        json.loads(line)
        for line in (out_dir / "token_timeline.jsonl").read_text().splitlines()
    ]
    assert timeline[0]["phase_bin"] == "early"
    cumulative = [
        json.loads(line)
        for line in (out_dir / "cumulative_core.jsonl").read_text().splitlines()
    ]
    assert "positionless_feature_persistence100_core_size" in cumulative[-1]
    layer_flow = [
        json.loads(line)
        for line in (out_dir / "layer_flow_by_token.jsonl").read_text().splitlines()
    ]
    assert "mass_fraction" in layer_flow[0]
    assert "global_core_summary" in summary


def test_temporal_analyzer_rejects_step_path_mismatch(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    graph_path = run_root / "shards" / "shard_000" / "token_000000" / "graph.npz"
    _write_graph(graph_path, _step(7, [(0, 0, 1), (0, 0, 2)]))

    with pytest.raises(ValueError, match="does not match path index"):
        analyze_full_answer_temporal(
            run_root=run_root,
            output_dir=tmp_path / "out",
            windows=[2],
            lags=[1],
        )


def test_temporal_plots_write_manifest_and_pngs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    for index, step in {
        0: _step(0, [(0, 0, 1), (0, 0, 2)]),
        1: _step(1, [(0, 0, 1), (0, 0, 3)]),
        2: _step(2, [(0, 0, 1), (0, 0, 3)]),
    }.items():
        _write_graph(
            run_root / "shards" / "shard_000" / f"token_{index:06d}" / "graph.npz",
            step,
        )
    analysis_dir = tmp_path / "analysis"
    analyze_full_answer_temporal(
        run_root=run_root, output_dir=analysis_dir, windows=[2], lags=[1, 2]
    )

    plot_dir = tmp_path / "plots"
    manifest = plot_full_answer_temporal(
        analysis_dir=analysis_dir,
        output_dir=plot_dir,
    )

    manifest_path = plot_dir / "plot_manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text())["analysis_dir"] == str(analysis_dir)
    expected = {
        "adjacent_jaccards.png",
        "adjacent_churn_rates.png",
        "weighted_churn_mass.png",
        "lag_jaccards.png",
        "rolling_core_sizes.png",
        "rolling_union_churn.png",
        "edge_core_stability.png",
        "positionless_feature_reuse.png",
        "layer_flow_stability.png",
        "layer_flow_heatmaps.png",
        "global_core_churn.png",
        "answer_phase_timeline.png",
    }
    assert {Path(path).name for path in manifest["generated_files"]} == expected
    for name in expected:
        path = plot_dir / name
        assert path.exists()
        assert path.stat().st_size > 0
