from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from nlp_research_project.circuit_stability_analysis.signed_graph import (
    decode_feature_endpoint,
)
from nlp_research_project.exact_trace_bench.compact_io import (
    CANONICAL_TYPED_BUCKET_NAMES,
)
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


def _encode_feature_feature_id(
    layer: int, position: int, feature_idx: int, *, n_pos: int = 2
) -> int:
    return layer * n_pos * 1_000_000 + position * 1_000_000 + feature_idx


def _write_bucketed_graph(
    path: Path,
    step: SimpleStep,
    weight: float,
    *,
    row_id: int | None = None,
    col_id: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(CANONICAL_TYPED_BUCKET_NAMES)
    feature_bucket = names.index("feature<-feature")
    source_id = (
        col_id
        if col_id is not None
        else _encode_feature_feature_id(*step.feature_ids[0], n_pos=2)
    )
    source = decode_feature_endpoint(source_id, 2)
    target_id = row_id if row_id is not None else step.step_idx
    target = decode_feature_endpoint(target_id, 2)
    persisted_feature_ids = np.unique(
        np.concatenate(
            [
                step.feature_ids,
                np.asarray([[source.layer, source.position, source.feature_id]]),
                np.asarray([[target.layer, target.position, target.feature_id]]),
            ]
        ),
        axis=0,
    )
    metadata = [
        {
            "bucket": name,
            "raw_total_abs_mass": weight + 1.0 if name == "feature<-feature" else 0.0,
            "retained_abs_mass": weight if name == "feature<-feature" else 0.0,
            "retained_fraction": (
                weight / (weight + 1.0) if name == "feature<-feature" else None
            ),
            "raw_nnz": 2 if name == "feature<-feature" else 0,
            "retained_nnz": 1 if name == "feature<-feature" else 0,
            "policy": {"top_p": 0.95, "cap": 1000000},
            "weights_signed": True,
        }
        for name in names
    ]
    np.savez_compressed(
        path,
        row_idx=step.row_idx,
        col_idx=step.col_idx,
        weights=step.weights,
        feature_ids=persisted_feature_ids,
        token_text=np.array(step.token_text),
        logprob=np.array(np.nan),
        n_features=np.array(len(persisted_feature_ids), dtype=np.int32),
        step_idx=np.array(step.step_idx, dtype=np.int32),
        compact_save_format=np.array("typed_bucketed"),
        bucket_row_idx=np.asarray([target_id], dtype=np.int64),
        bucket_col_idx=np.asarray([source_id], dtype=np.int64),
        bucket_weights=np.asarray([weight], dtype=np.float32),
        bucket_ids=np.asarray([feature_bucket], dtype=np.int16),
        bucket_names=np.asarray(names),
        bucket_metadata_json=np.array(json.dumps(metadata)),
        error_node_shape=np.asarray([1, 2], dtype=np.int32),
        token_ids=np.asarray([1, 2], dtype=np.int64),
        logit_token_ids=np.asarray([3], dtype=np.int64),
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


def test_temporal_analyzer_reads_bucket_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    for index, weight in [(0, 2.0), (1, 3.0)]:
        _write_bucketed_graph(
            run_root / "shards" / "shard_000" / f"token_{index:06d}" / "graph.npz",
            _step(index, [(0, 0, 1), (0, 0, 2)]),
            weight,
        )

    out_dir = tmp_path / "out"
    analyze_full_answer_temporal(
        run_root=run_root, output_dir=out_dir, windows=[2], lags=[1]
    )
    adjacent = [
        json.loads(line)
        for line in (out_dir / "adjacent_pairs.jsonl").read_text().splitlines()
    ]
    row = adjacent[0]
    assert row["bucket_feature_feature_edge_count_a"] == 1
    assert row["bucket_feature_feature_jaccard"] == 0.0
    assert row["bucket_feature_feature_weighted_jaccard"] == 0.0
    assert row["bucket_feature_feature_retained_nnz_a"] == 1
    assert row["bucket_feature_feature_raw_total_abs_mass_b"] == 4.0


def test_temporal_analyzer_collapses_feature_feature_bucket_flows(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source_a = _encode_feature_feature_id(1, 0, 10)
    target_a = _encode_feature_feature_id(2, 1, 20)
    source_b = _encode_feature_feature_id(1, 1, 10)
    target_b = _encode_feature_feature_id(2, 0, 20)
    source_c = _encode_feature_feature_id(3, 0, 30)
    for index, row_id, col_id, weight in [
        (0, target_a, source_a, 2.0),
        (1, target_b, source_b, 3.0),
        (2, target_b, source_c, 5.0),
    ]:
        _write_bucketed_graph(
            run_root / "shards" / "shard_000" / f"token_{index:06d}" / "graph.npz",
            _step(index, [(0, 0, 1), (0, 0, 2)]),
            weight,
            row_id=row_id,
            col_id=col_id,
        )

    out_dir = tmp_path / "out"
    summary = analyze_full_answer_temporal(
        run_root=run_root, output_dir=out_dir, windows=[2], lags=[1, 2]
    )
    adjacent = [
        json.loads(line)
        for line in (out_dir / "adjacent_pairs.jsonl").read_text().splitlines()
    ]
    assert adjacent[0]["feature_feature_layer_flow_weighted_jaccard"] == pytest.approx(
        2 / 3
    )
    assert adjacent[0][
        "feature_feature_positionless_flow_weighted_jaccard"
    ] == pytest.approx(2 / 3)
    assert adjacent[1]["feature_feature_layer_flow_weighted_jaccard"] == 0.0
    assert adjacent[1]["feature_feature_positionless_flow_weighted_jaccard"] == 0.0
    assert summary["adjacent_summary"][
        "mean_feature_feature_layer_flow_weighted_jaccard"
    ] == pytest.approx(1 / 3)

    lag_rows = [
        json.loads(line)
        for line in (out_dir / "lag_pairs.jsonl").read_text().splitlines()
    ]
    lag_two = [row for row in lag_rows if row["lag"] == 2][0]
    assert lag_two["feature_feature_layer_flow_weighted_jaccard"] == 0.0


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
