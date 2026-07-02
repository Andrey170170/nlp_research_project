from __future__ import annotations

import csv
import json
from pathlib import Path

from nlp_research_project.circuit_stability_analysis.summaries import summarize_metrics


def test_summaries_group_means_and_delta(tmp_path: Path) -> None:
    rows = [
        {
            "prompt_id": "p1",
            "pair_category": "temporal",
            "pair_subcategory": "adjacent",
            "label_a": "correct",
            "label_b": "correct",
            "region_a": "r",
            "region_b": "r",
            "analysis_group_a": "a",
            "analysis_group_b": "a",
            "metric_version": "core_v1",
            "role_version": "rv",
            "cluster_version": "cv",
            "generated_index": "1",
            "global_frac": "0.25",
            "feature_node_weighted_jaccard": "0.8",
            "decode_errors_a": "0",
        },
        {
            "prompt_id": "p1",
            "pair_category": "temporal",
            "pair_subcategory": "adjacent",
            "label_a": "wrong",
            "label_b": "wrong",
            "region_a": "r",
            "region_b": "r",
            "analysis_group_a": "a",
            "analysis_group_b": "a",
            "metric_version": "core_v1",
            "role_version": "rv",
            "cluster_version": "cv",
            "generated_index": "2",
            "global_frac": "0.75",
            "feature_node_weighted_jaccard": "0.3",
            "decode_errors_a": "1",
        },
        {
            "prompt_id": "p2",
            "pair_category": "temporal",
            "pair_subcategory": "adjacent",
            "label_a": "correct",
            "label_b": "correct",
            "region_a": "r",
            "region_b": "r",
            "analysis_group_a": "a",
            "analysis_group_b": "a",
            "metric_version": "core_v1",
            "role_version": "rv",
            "cluster_version": "cv",
            "generated_index": "3",
            "global_frac": "0.50",
            "feature_node_weighted_jaccard": "0.6",
            "decode_errors_a": "0",
        },
        {
            "prompt_id": "p2",
            "pair_category": "temporal",
            "pair_subcategory": "adjacent",
            "label_a": "wrong",
            "label_b": "wrong",
            "region_a": "r",
            "region_b": "r",
            "analysis_group_a": "a",
            "analysis_group_b": "a",
            "metric_version": "core_v1",
            "role_version": "rv",
            "cluster_version": "cv",
            "generated_index": "4",
            "global_frac": "0.90",
            "feature_node_weighted_jaccard": "0.5",
            "decode_errors_a": "1",
        },
    ]
    p = tmp_path / "metrics.csv"
    with p.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=sorted(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize_metrics(p, tmp_path / "out")

    summary = list(
        csv.DictReader((tmp_path / "out" / "prompt_metric_summary.csv").open())
    )
    corr = [r for r in summary if r["label_pair"] == "correct|correct"]
    assert len(corr) == 2
    assert corr[0]["pair_count"] == "1"
    assert float(corr[0]["mean_feature_node_weighted_jaccard"]) in {0.8, 0.6}
    assert all("mean_generated_index" not in r for r in summary)
    assert all("mean_global_frac" not in r for r in summary)

    agg = list(
        csv.DictReader((tmp_path / "out" / "aggregate_metric_summary.csv").open())
    )
    agg_corr = next(r for r in agg if r["label_pair"] == "correct|correct")
    assert agg_corr["prompt_count"] == "2"
    assert agg_corr["total_pair_count"] == "2"
    assert abs(float(agg_corr["mean_feature_node_weighted_jaccard"]) - 0.7) < 1e-9

    delta = list(csv.DictReader((tmp_path / "out" / "correct_wrong_delta.csv").open()))
    assert len(delta) == 2
    agg_delta = list(
        csv.DictReader((tmp_path / "out" / "aggregate_correct_wrong_delta.csv").open())
    )
    assert len(agg_delta) == 1
    assert agg_delta[0]["prompt_count"] == "2"
    assert abs(float(agg_delta[0]["delta_feature_node_weighted_jaccard"]) - 0.3) < 1e-9

    audit = json.loads((tmp_path / "out" / "summary_audit.json").read_text())
    assert audit["row_count"] == 4
