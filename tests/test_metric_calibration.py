from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nlp_research_project.exact_trace_bench.full_answer.calibration import (
    build_scorecard,
    load_calibration_pairs,
    position_band_for_index,
    rebucket_metric_calibration,
    run_metric_calibration,
)
from nlp_research_project.exact_trace_bench.full_answer.calibration_plots import (
    plot_metric_calibration,
)


def _write_graph(
    run_root: Path,
    index: int,
    features: list[tuple[int, int, int]],
    weights: list[float] | None = None,
) -> Path:
    path = run_root / "shards" / "shard_000" / f"token_{index:06d}" / "graph.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = weights or [1.0, 2.0]
    np.savez_compressed(
        path,
        row_idx=np.asarray([1, len(features)], dtype=np.int64),
        col_idx=np.asarray([0, 0], dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float32),
        feature_ids=np.asarray(features, dtype=np.int64),
        token_text=np.asarray(f"tok{index}"),
        logprob=np.asarray(np.nan),
        n_features=np.asarray(len(features), dtype=np.int64),
        step_idx=np.asarray(index, dtype=np.int64),
    )
    return path


def test_position_band_for_index_matches_plan_boundaries() -> None:
    assert position_band_for_index(0, 100) == "pct_00_20"
    assert position_band_for_index(19, 100) == "pct_00_20"
    assert position_band_for_index(20, 100) == "pct_20_40"
    assert position_band_for_index(40, 100) == "pct_40_60"
    assert position_band_for_index(60, 100) == "pct_60_80"
    assert position_band_for_index(80, 100) == "pct_80_100"
    assert position_band_for_index(99, 100) == "pct_80_100"


def test_rebucket_metric_calibration_uses_classification_token_counts(
    tmp_path: Path,
) -> None:
    analysis_dir = tmp_path / "old"
    output_dir = tmp_path / "rebucketed"
    analysis_dir.mkdir()
    rows = [
        {
            "pair_id": "p1",
            "pair_category": "noise",
            "sub_category": "identity",
            "trajectory": "traj_correct",
            "prompt_id": "prompt_a",
            "answer_label": "correct",
            "generated_index": 80,
            "position_band": "late",
            "bucket": "all_edges",
            "metric": "similarity_metric",
            "params_json": "{}",
            "value": 1.0,
        },
        {
            "pair_id": "p2",
            "pair_category": "null",
            "sub_category": "same_prompt_correct_wrong_same_role",
            "trajectory": "prompt_a:correct_vs_wrong:reasoning_text",
            "prompt_id": "prompt_a",
            "generated_index": 20,
            "position_band": "mid",
            "bucket": "all_edges",
            "metric": "similarity_metric",
            "params_json": "{}",
            "value": 0.1,
        },
    ]
    (analysis_dir / "metric_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (analysis_dir / "pair_manifest_resolved.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    (analysis_dir / "calibration_summary.json").write_text("{}", encoding="utf-8")
    catalog = tmp_path / "role_classification_catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "name": "traj_correct",
                        "prompt_id": "prompt_a",
                        "label": "correct",
                        "token_count": 100,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = rebucket_metric_calibration(
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        classification_catalog=catalog,
        write_parquet=False,
    )

    rebucketed = [
        json.loads(line)
        for line in (output_dir / "metric_rows.jsonl").read_text().splitlines()
    ]
    assert summary["position_band_scheme"] == "generated_token_quintile_pct20_v1"
    assert rebucketed[0]["position_band"] == "pct_80_100"
    assert rebucketed[1]["position_band"] == "pct_20_40"


def test_load_calibration_pairs_expands_trajectory_and_relative_pairs(
    tmp_path: Path,
) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    for i in range(5):
        _write_graph(left_root, i, [(0, i, 1), (0, i, 2)])
        _write_graph(right_root, i, [(0, i, 3), (0, i, 4)])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "trajectories": [
                    {
                        "name": "traj_left",
                        "run_root": str(left_root),
                        "lags": [1, 2],
                        "max_pairs_per_lag": 2,
                    }
                ],
                "matched_relative_pairs": [
                    {
                        "name": "null_lr",
                        "left_run_root": str(left_root),
                        "right_run_root": str(right_root),
                        "sample_count": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pairs = load_calibration_pairs(manifest_path)

    assert len([pair for pair in pairs if pair.pair_category == "temporal"]) == 4
    assert len([pair for pair in pairs if pair.pair_category == "null"]) == 3
    assert {pair.sub_category for pair in pairs if pair.lag == 1} == {"adjacent"}


def test_build_scorecard_handles_similarity_and_distance_metrics() -> None:
    rows = []
    for category, value in [("null", 0.1), ("temporal", 0.5), ("noise", 0.9)]:
        rows.append(
            {
                "pair_category": category,
                "sub_category": "adjacent" if category == "temporal" else None,
                "lag": 1 if category == "temporal" else None,
                "bucket": "unit",
                "metric": "weighted_jaccard",
                "params_json": "{}",
                "position_band": "mid",
                "value": value,
            }
        )
    for category, value in [("null", 0.9), ("temporal", 0.5), ("noise", 0.1)]:
        rows.append(
            {
                "pair_category": category,
                "sub_category": "adjacent" if category == "temporal" else None,
                "lag": 1 if category == "temporal" else None,
                "bucket": "unit",
                "metric": "normalized_l1_distance",
                "params_json": "{}",
                "position_band": "mid",
                "value": value,
            }
        )

    scorecard = build_scorecard(rows)
    by_metric = {row["metric"]: row for row in scorecard}

    assert by_metric["weighted_jaccard"]["metric_direction"] == "higher_is_more_similar"
    assert by_metric["weighted_jaccard"]["separation_score"] == 0.5
    assert (
        by_metric["normalized_l1_distance"]["metric_direction"]
        == "lower_is_more_similar"
    )
    assert by_metric["normalized_l1_distance"]["separation_score"] == 0.5


def test_run_metric_calibration_writes_rows_and_scorecard(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    graph0 = _write_graph(run_root, 0, [(0, 0, 1), (0, 0, 2)])
    graph1 = _write_graph(run_root, 1, [(0, 1, 1), (0, 1, 3)])
    graph2 = _write_graph(run_root, 2, [(0, 2, 8), (0, 2, 9)])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": "noise_pair",
                        "left_graph": str(graph0),
                        "right_graph": str(graph0),
                        "pair_category": "noise",
                        "position_band": "mid",
                    },
                    {
                        "pair_id": "temporal_pair",
                        "left_graph": str(graph0),
                        "right_graph": str(graph1),
                        "pair_category": "temporal",
                        "sub_category": "adjacent",
                        "position_band": "mid",
                        "lag": 1,
                    },
                    {
                        "pair_id": "null_pair",
                        "left_graph": str(graph0),
                        "right_graph": str(graph2),
                        "pair_category": "null",
                        "position_band": "mid",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    summary = run_metric_calibration(
        manifest_path=manifest_path,
        output_dir=out_dir,
        write_parquet=False,
    )

    assert summary["pair_count"] == 3
    assert summary["metric_row_count"] > 0
    assert summary["decoder_soft_matching_enabled"] is False
    assert summary["decoder_cosine_device"] == "cpu"
    assert (out_dir / "metric_rows.jsonl").exists()
    assert (out_dir / "metric_rows.csv").exists()
    assert (out_dir / "scorecard.csv").exists()
    assert (out_dir / "pair_rows" / "noise_pair.jsonl").exists()
    assert (out_dir / "pair_context" / "noise_pair.json").exists()
    resolved = json.loads((out_dir / "pair_manifest_resolved.json").read_text())
    assert {row["pair_id"] for row in resolved} == {
        "noise_pair",
        "temporal_pair",
        "null_pair",
    }

    resumed = run_metric_calibration(
        manifest_path=manifest_path,
        output_dir=out_dir,
        write_parquet=False,
    )
    assert {row["status"] for row in resumed["checkpoint_results"]} == {
        "skipped_existing"
    }


def test_run_metric_calibration_parallel_workers(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    graph0 = _write_graph(run_root, 0, [(0, 0, 1), (0, 0, 2)])
    graph1 = _write_graph(run_root, 1, [(0, 1, 1), (0, 1, 3)])
    graph2 = _write_graph(run_root, 2, [(0, 2, 8), (0, 2, 9)])
    manifest_path = tmp_path / "manifest_parallel.json"
    manifest_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": "parallel_noise",
                        "left_graph": str(graph0),
                        "right_graph": str(graph0),
                        "pair_category": "noise",
                        "position_band": "mid",
                    },
                    {
                        "pair_id": "parallel_temporal",
                        "left_graph": str(graph0),
                        "right_graph": str(graph1),
                        "pair_category": "temporal",
                        "sub_category": "adjacent",
                        "position_band": "mid",
                        "lag": 1,
                    },
                    {
                        "pair_id": "parallel_null",
                        "left_graph": str(graph0),
                        "right_graph": str(graph2),
                        "pair_category": "null",
                        "position_band": "mid",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = run_metric_calibration(
        manifest_path=manifest_path,
        output_dir=tmp_path / "out_parallel",
        write_parquet=False,
        workers=2,
    )

    assert summary["workers"] == 2
    assert summary["pair_count"] == 3
    assert summary["metric_row_count"] > 0
    assert {row["status"] for row in summary["checkpoint_results"]} == {"completed"}


def test_plot_metric_calibration_writes_manifest_plots_and_report(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    graph0 = _write_graph(run_root, 0, [(0, 0, 1), (0, 0, 2)])
    graph1 = _write_graph(run_root, 1, [(0, 1, 1), (0, 1, 3)])
    graph2 = _write_graph(run_root, 2, [(0, 2, 8), (0, 2, 9)])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "left_graph": str(graph0),
                        "right_graph": str(graph0),
                        "pair_category": "noise",
                        "position_band": "mid",
                    },
                    {
                        "left_graph": str(graph0),
                        "right_graph": str(graph1),
                        "pair_category": "temporal",
                        "sub_category": "adjacent",
                        "position_band": "mid",
                        "lag": 1,
                    },
                    {
                        "left_graph": str(graph0),
                        "right_graph": str(graph2),
                        "pair_category": "null",
                        "position_band": "mid",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    analysis_dir = tmp_path / "analysis"
    run_metric_calibration(
        manifest_path=manifest_path,
        output_dir=analysis_dir,
        write_parquet=False,
    )

    plot_dir = tmp_path / "plots"
    manifest = plot_metric_calibration(analysis_dir=analysis_dir, output_dir=plot_dir)

    assert (plot_dir / "plot_manifest.json").exists()
    assert (plot_dir / "metric_calibration_findings.md").exists()
    assert manifest["generated_files"]
    for path in manifest["generated_files"]:
        assert Path(path).exists()
