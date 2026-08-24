from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from nlp_research_project.circuit_stability_analysis.cli import main
from nlp_research_project.circuit_stability_analysis.metrics import run_metrics
from typed_graph_fixtures import write_typed_graph


def _graph(path: Path, fid: int, *, token_id: int, logit_id: int) -> None:
    write_typed_graph(
        path,
        feature_ids=[(1, 0, fid), (1, 0, 99)],
        token_ids=[token_id, 999],
        logit_token_ids=[logit_id],
        token_text="x",
        bucket_values={
            "feature<-feature": [[0.0, 1.0], [0.0, 0.0]],
            "feature<-error": [[0.0, 0.0], [0.0, 0.0]],
            "feature<-token": [[1.0, 0.0], [0.0, 0.0]],
            "logit<-feature": [[1.0, 0.0]],
            "logit<-error": [[0.0, 0.0]],
            "logit<-token": [[0.0, 0.0]],
        },
    )


def test_metrics_keep_graph_step_path_validation(tmp_path: Path) -> None:
    graph = tmp_path / "token_000001" / "graph.npz"
    _graph(graph, 10, token_id=7, logit_id=70)
    manifest = {
        "metadata": {},
        "pairs": [
            {
                "pair_id": "p0",
                "pair_category": "temporal",
                "pair_subcategory": "adjacent",
                "prompt_id": "p",
                "label_a": "correct",
                "label_b": "correct",
                "graph_path_a": str(graph),
                "graph_path_b": str(graph),
            }
        ],
    }
    mp = tmp_path / "pairs.json"
    mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="step_idx"):
        run_metrics(mp, tmp_path / "metrics.csv")


def test_metrics_exact_cluster_and_cli(tmp_path: Path) -> None:
    g1 = tmp_path / "g1.npz"
    g2 = tmp_path / "g2.npz"
    _graph(g1, 10, token_id=7, logit_id=70)
    _graph(g2, 10, token_id=8, logit_id=80)
    manifest = {
        "metadata": {"role_version": "rv", "cluster_version": "cv"},
        "pairs": [
            {
                "pair_id": "p0",
                "pair_category": "temporal",
                "pair_subcategory": "adjacent",
                "prompt_id": "p",
                "label_a": "correct",
                "label_b": "correct",
                "graph_path_a": str(g1),
                "graph_path_b": str(g2),
            },
        ],
    }
    mp = tmp_path / "pairs.json"
    mp.write_text(json.dumps(manifest))
    cdir = tmp_path / "clusters"
    cdir.mkdir()
    with (cdir / "feature_cluster_assignments.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["layer", "feature_id", "cluster_id"])
        w.writeheader()
        w.writerows([{"layer": 1, "feature_id": 99, "cluster_id": "D"}])
    (cdir / "cluster_manifest.json").write_text(
        json.dumps({"cluster_version": "clusters_v1"})
    )

    out = tmp_path / "metrics.csv"
    run_metrics(mp, out, cluster_manifest=cdir / "cluster_manifest.json")
    rows = list(csv.DictReader(out.open()))
    assert float(rows[0]["feature_node_weighted_jaccard"]) == 1.0
    assert float(rows[0]["feature_token_weighted_jaccard"]) == 0.0
    assert float(rows[0]["logit_feature_weighted_jaccard"]) == 0.0
    assert float(rows[0]["cluster_node_weighted_jaccard"]) == 1.0
    assert rows[0]["decoder_soft_available"] == "false"
    assert rows[0]["decoder_soft_jaccard_0.70"] == ""

    out_blank = tmp_path / "metrics_blank.csv"
    run_metrics(mp, out_blank)
    blank_rows = list(csv.DictReader(out_blank.open()))
    assert blank_rows[0]["cluster_node_weighted_jaccard"] == ""

    class FakeDecoderSignatureStore:
        def __init__(self, cache_dir: Path) -> None:
            self.cache_dir = cache_dir

        def cosine_matrix(self, layer: int, left_feature_ids, right_feature_ids):
            mat = np.zeros(
                (len(left_feature_ids), len(right_feature_ids)), dtype=np.float32
            )
            for i in range(min(len(left_feature_ids), len(right_feature_ids))):
                mat[i, i] = 0.85
            return mat

    import nlp_research_project.circuit_stability_analysis.metrics as metrics_module

    metrics_module.DecoderSignatureStore = FakeDecoderSignatureStore
    out_soft = tmp_path / "metrics_soft.csv"
    run_metrics(mp, out_soft, decoder_cache_dir=tmp_path / "decoder_cache")
    soft_rows = list(csv.DictReader(out_soft.open()))
    assert soft_rows[0]["decoder_soft_available"] == "true"
    assert float(soft_rows[0]["decoder_soft_matched_mass_0.80"]) > 0
    assert float(soft_rows[0]["decoder_soft_jaccard_0.80"]) > 0

    main(
        [
            "run-metrics",
            "--pair-manifest",
            str(mp),
            "--output",
            str(tmp_path / "cli.csv"),
            "--max-pairs",
            "1",
        ]
    )
    assert len(list(csv.DictReader((tmp_path / "cli.csv").open()))) == 1
