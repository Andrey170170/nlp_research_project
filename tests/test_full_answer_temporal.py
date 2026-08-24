from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench.full_answer.temporal import (
    _load_snapshot,
    analyze_full_answer_temporal,
)
from nlp_research_project.exact_trace_bench.full_answer.temporal_plots import (
    plot_full_answer_temporal,
)
from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    load_typed_compact_graph,
)
from typed_graph_fixtures import write_typed_graph


def _graph_path(run_root: Path, index: int) -> Path:
    return run_root / "shards" / "shard_000" / f"token_{index:06d}" / "graph.npz"


def _domain_shift_bucket_values(
    *, n_pos: int, logit_token_ids: list[int]
) -> dict[str, list[list[float]]]:
    feature_count = 2
    error_count = 2 * n_pos
    logit_count = len(logit_token_ids)

    def zeros(rows: int, columns: int) -> list[list[float]]:
        return [[0.0 for _column in range(columns)] for _row in range(rows)]

    values = {
        "feature<-feature": zeros(feature_count, feature_count),
        "feature<-error": zeros(feature_count, error_count),
        "feature<-token": zeros(feature_count, n_pos),
        "logit<-feature": zeros(logit_count, feature_count),
        "logit<-error": zeros(logit_count, error_count),
        "logit<-token": zeros(logit_count, n_pos),
    }
    error_layer_one_position_one = n_pos + 1
    logit_42_row = logit_token_ids.index(42)
    values["feature<-feature"][1][0] = 1.0
    values["feature<-error"][1][error_layer_one_position_one] = 2.0
    values["feature<-token"][1][1] = 3.0
    values["logit<-feature"][logit_42_row][0] = 4.0
    values["logit<-error"][logit_42_row][error_layer_one_position_one] = 5.0
    values["logit<-token"][logit_42_row][1] = 6.0
    return values


def test_temporal_analyzer_emits_explicit_bucket_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_typed_graph(
        _graph_path(run_root, 0),
        feature_ids=[(0, 0, 4), (1, 1, 9)],
    )
    write_typed_graph(
        _graph_path(run_root, 1),
        step_idx=1,
        feature_ids=[(0, 0, 4), (1, 1, 10)],
        bucket_values={"feature<-error": [[1.0, 0.0], [0.0, -2.0]]},
    )
    write_typed_graph(
        _graph_path(run_root, 2),
        step_idx=2,
        feature_ids=[(0, 0, 4), (1, 1, 10)],
        bucket_values={"feature<-error": [[1.0, 0.0], [0.0, -2.0]]},
    )

    out_dir = tmp_path / "out"
    summary = analyze_full_answer_temporal(
        run_root=run_root, output_dir=out_dir, windows=[2], lags=[1, 2]
    )
    assert summary["token_count"] == 3
    adjacent = [
        json.loads(line)
        for line in (out_dir / "adjacent_pairs.jsonl").read_text().splitlines()
    ]
    first = adjacent[0]
    assert first["feature_jaccard"] == pytest.approx(1 / 3)
    assert "bucket_feature_feature_weighted_jaccard" in first
    assert "bucket_feature_error_core80_shared_mass_fraction_a" in first
    assert "bucket_logit_token_top128_weighted_jaccard" in first
    assert first["bucket_feature_feature_raw_edge_count_a"] == 2
    assert first["bucket_feature_feature_raw_abs_mass_a"] == pytest.approx(2.0)
    assert first["bucket_feature_feature_retained_edge_count_a"] == 2
    assert first["bucket_feature_feature_retained_abs_mass_a"] == pytest.approx(2.0)
    assert first["bucket_feature_feature_retained_fraction_a"] == pytest.approx(1.0)
    assert first["bucket_feature_feature_cap_bound_before_top_p_a"] is False
    assert "bucket_feature_feature_raw_total_abs_mass_a" not in first
    assert "bucket_feature_feature_raw_nnz_a" not in first
    assert not any(key.startswith("all_edge") for key in first)

    rolling = [
        json.loads(line)
        for line in (out_dir / "rolling_windows.jsonl").read_text().splitlines()
    ]
    assert rolling[0]["bucket_feature_feature_union_size"] > 0
    assert "bucket_feature_feature_persistence80_mass_fraction" in rolling[0]


def test_temporal_analyzer_rejects_step_path_mismatch(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_typed_graph(_graph_path(run_root, 0), step_idx=7)
    with pytest.raises(ValueError, match="does not match expected step 0"):
        analyze_full_answer_temporal(
            run_root=run_root,
            output_dir=tmp_path / "out",
            windows=[2],
            lags=[1],
        )


def test_temporal_edges_are_semantic_across_changing_domains(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    features = [(0, 0, 4), (1, 1, 9)]
    left_path = _graph_path(run_root, 0)
    right_path = _graph_path(run_root, 1)
    write_typed_graph(
        left_path,
        feature_ids=features,
        token_ids=[10, 11],
        logit_token_ids=[42],
        bucket_values=_domain_shift_bucket_values(n_pos=2, logit_token_ids=[42]),
    )
    write_typed_graph(
        right_path,
        step_idx=1,
        feature_ids=features,
        token_ids=[10, 11, 12],
        logit_token_ids=[7, 42],
        bucket_values=_domain_shift_bucket_values(
            n_pos=3,
            logit_token_ids=[7, 42],
        ),
    )

    left_graph = load_typed_compact_graph(left_path)
    right_graph = load_typed_compact_graph(right_path)
    assert left_graph.error_node_shape == (2, 2)
    assert right_graph.error_node_shape == (2, 3)
    assert left_graph.logit_token_ids.tolist() == [42]
    assert right_graph.logit_token_ids.tolist() == [7, 42]

    feature_4 = ("feature", 0, 0, 4)
    feature_9 = ("feature", 1, 1, 9)
    error_1_1 = ("error", 1, 1, 11)
    token_1 = ("token", 1, 11)
    logit_42 = ("logit", 42)
    expected = {
        "feature<-feature": {(feature_9, feature_4): 1.0},
        "feature<-error": {(feature_9, error_1_1): 2.0},
        "feature<-token": {(feature_9, token_1): 3.0},
        "logit<-feature": {(logit_42, feature_4): 4.0},
        "logit<-error": {(logit_42, error_1_1): 5.0},
        "logit<-token": {(logit_42, token_1): 6.0},
    }
    assert _load_snapshot(left_path).bucket_edges == expected
    assert _load_snapshot(right_path).bucket_edges == expected

    out_dir = tmp_path / "out"
    analyze_full_answer_temporal(
        run_root=run_root,
        output_dir=out_dir,
        windows=[2],
        lags=[1],
    )
    adjacent = json.loads((out_dir / "adjacent_pairs.jsonl").read_text().strip())
    for bucket in (
        "feature_feature",
        "feature_error",
        "feature_token",
        "logit_feature",
        "logit_error",
        "logit_token",
    ):
        assert adjacent[f"bucket_{bucket}_jaccard"] == 1.0
        assert adjacent[f"bucket_{bucket}_weighted_jaccard"] == 1.0
        assert adjacent[f"bucket_{bucket}_entered"] == 0
        assert adjacent[f"bucket_{bucket}_exited"] == 0
        assert adjacent[f"bucket_{bucket}_stayed"] == 1


def test_temporal_plots_write_manifest_and_pngs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    for index in range(3):
        write_typed_graph(_graph_path(run_root, index), step_idx=index)
    analysis_dir = tmp_path / "analysis"
    analyze_full_answer_temporal(
        run_root=run_root, output_dir=analysis_dir, windows=[2], lags=[1, 2]
    )
    plot_dir = tmp_path / "plots"
    manifest = plot_full_answer_temporal(
        analysis_dir=analysis_dir,
        output_dir=plot_dir,
    )
    assert (plot_dir / "plot_manifest.json").exists()
    assert len(manifest["generated_files"]) == 12
    assert all(Path(path).stat().st_size > 0 for path in manifest["generated_files"])
