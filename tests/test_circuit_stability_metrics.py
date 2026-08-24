from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from nlp_research_project.circuit_stability_analysis.cli import main
from nlp_research_project.circuit_stability_analysis.metrics import run_metrics
from nlp_research_project.circuit_stability_analysis.signed_graph import (
    encode_feature_endpoint,
)
from nlp_research_project.exact_trace_bench.compact_io import (
    CANONICAL_TYPED_BUCKET_NAMES,
)


def _fid(fid: int) -> int:
    return encode_feature_endpoint(1, 0, fid, 2)


def _graph(path: Path, fid: int, *, token_id: int, logit_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(CANONICAL_TYPED_BUCKET_NAMES)
    name_id = {name: index for index, name in enumerate(names)}
    edges = [
        (name_id["feature<-token"], _fid(fid), 0, 1.0),
        (name_id["logit<-feature"], 0, _fid(fid), 1.0),
        (name_id["feature<-feature"], _fid(fid), _fid(99), 1.0),
    ]
    counts = {name: 0 for name in names}
    masses = {name: 0.0 for name in names}
    for bucket_id, _row, _col, _weight in edges:
        counts[names[bucket_id]] += 1
        masses[names[bucket_id]] += abs(_weight)
    np.savez_compressed(
        path,
        step_idx=np.array(0, dtype=np.int32),
        token_text=np.array("x"),
        logprob=np.array(np.nan),
        n_features=np.array(2, dtype=np.int32),
        row_idx=np.asarray([], dtype=np.int32),
        col_idx=np.asarray([], dtype=np.int32),
        weights=np.asarray([], dtype=np.float32),
        compact_save_format=np.asarray("typed_bucketed"),
        token_ids=np.asarray([token_id, 999]),
        logit_token_ids=np.asarray([logit_id]),
        feature_ids=np.asarray([[1, 0, fid], [1, 0, 99]]),
        error_node_shape=np.asarray([1, 2]),
        bucket_names=np.asarray(names),
        bucket_ids=np.asarray([e[0] for e in edges]),
        bucket_row_idx=np.asarray([e[1] for e in edges]),
        bucket_col_idx=np.asarray([e[2] for e in edges]),
        bucket_weights=np.asarray([e[3] for e in edges], dtype=np.float32),
        bucket_metadata_json=np.asarray(
            json.dumps(
                [
                    {
                        "bucket": name,
                        "raw_total_abs_mass": masses[name],
                        "retained_abs_mass": masses[name],
                        "retained_fraction": 1.0 if masses[name] else None,
                        "raw_nnz": counts[name],
                        "retained_nnz": counts[name],
                        "policy": {"top_p": 1.0, "cap": None},
                        "weights_signed": True,
                    }
                    for name in names
                ]
            )
        ),
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
