from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from nlp_research_project.circuit_stability_analysis.cli import main
from nlp_research_project.circuit_stability_analysis.clustering import cluster_features
from nlp_research_project.circuit_stability_analysis.profiles import (
    build_feature_profiles,
    load_feature_profile_bundle,
)
from nlp_research_project.circuit_stability_analysis.signed_graph import (
    decode_feature_endpoint,
    encode_feature_endpoint,
    load_signed_graph,
)
from nlp_research_project.exact_trace_bench.compact_io import (
    CANONICAL_TYPED_BUCKET_NAMES,
    StepData,
    save_compact,
)


def _write_graph(
    path: Path, step: int, edges: list[tuple[str, int, int, float]], *, n_pos: int = 4
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(CANONICAL_TYPED_BUCKET_NAMES)
    name_id = {n: i for i, n in enumerate(names)}
    selected = {
        decode_feature_endpoint(endpoint, n_pos)
        for name, row, col, _weight in edges
        for endpoint in (
            ([row] if name.startswith("feature<-") else [])
            + ([col] if name.endswith("<-feature") else [])
        )
    }
    feature_ids = np.asarray(
        sorted((value.layer, value.position, value.feature_id) for value in selected),
        dtype=np.int64,
    ).reshape(-1, 3)
    counts = {name: sum(edge[0] == name for edge in edges) for name in names}
    masses = {
        name: sum(abs(edge[3]) for edge in edges if edge[0] == name) for name in names
    }
    np.savez_compressed(
        path,
        step_idx=np.array(step, dtype=np.int32),
        token_text=np.array(f" tok{step}"),
        logprob=np.array(np.nan),
        n_features=np.asarray(len(feature_ids), dtype=np.int32),
        feature_ids=feature_ids,
        row_idx=np.asarray([], dtype=np.int32),
        col_idx=np.asarray([], dtype=np.int32),
        weights=np.asarray([], dtype=np.float32),
        compact_save_format=np.asarray("typed_bucketed"),
        token_ids=np.asarray([100, 101, 102, 103], dtype=np.int64),
        logit_token_ids=np.asarray([200, 201], dtype=np.int64),
        error_node_shape=np.asarray([1, n_pos], dtype=np.int32),
        bucket_names=np.asarray(names),
        bucket_ids=np.asarray([name_id[e[0]] for e in edges], dtype=np.int16),
        bucket_row_idx=np.asarray([e[1] for e in edges], dtype=np.int64),
        bucket_col_idx=np.asarray([e[2] for e in edges], dtype=np.int64),
        bucket_weights=np.asarray([e[3] for e in edges], dtype=np.float32),
        bucket_metadata_json=np.array(
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


def _fid(fid: int, pos: int = 1) -> int:
    return encode_feature_endpoint(1, pos, fid, 4)


def _csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_signed_graph_preserves_signs_and_endpoint_roundtrip(tmp_path: Path) -> None:
    enc = encode_feature_endpoint(2, 3, 456, 8)
    assert decode_feature_endpoint(enc, 8) == decode_feature_endpoint(enc, 8).__class__(
        2, 3, 456
    )
    graph_path = tmp_path / "token_000000" / "graph.npz"
    _write_graph(
        graph_path,
        0,
        [("logit<-feature", 0, _fid(10), -2.5), ("feature<-token", _fid(10), 1, 1.25)],
    )
    graph = load_signed_graph(graph_path)
    weights = [e.weight for e in graph.iter_bucket_edges("logit<-feature")]
    assert weights == [-2.5]


def test_signed_graph_rejects_legacy_compact_without_typed_buckets(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "token_000000" / "graph.npz"
    save_compact(
        StepData(
            step_idx=0,
            row_idx=np.asarray([], dtype=np.int32),
            col_idx=np.asarray([], dtype=np.int32),
            weights=np.asarray([], dtype=np.float32),
            feature_ids=np.asarray([], dtype=np.int64).reshape(0, 3),
            token_text="x",
            logprob=None,
            n_features=0,
        ),
        graph_path,
    )

    with pytest.raises(ValueError, match="requires typed bucket arrays"):
        load_signed_graph(graph_path)


def test_profile_extraction_outputs_signed_sparse_views(tmp_path: Path) -> None:
    g0 = tmp_path / "token_000000" / "graph.npz"
    g1 = tmp_path / "token_000001" / "graph.npz"
    _write_graph(
        g0,
        0,
        [
            ("feature<-token", _fid(10), 0, 1.0),
            ("logit<-feature", 0, _fid(10), -3.0),
            ("feature<-feature", _fid(10), _fid(11), 2.0),
        ],
    )
    _write_graph(
        g1,
        1,
        [
            ("feature<-token", _fid(10), 1, 2.0),
            ("logit<-feature", 1, _fid(10), 4.0),
            ("feature<-feature", _fid(12), _fid(10), -1.0),
        ],
    )
    tokens = tmp_path / "tokens.csv"
    rows = [
        {
            "wave": "w",
            "prompt_id": "p",
            "label": "correct",
            "trajectory_name": "t",
            "generated_index": str(i),
            "graph_path": str(p),
        }
        for i, p in enumerate([g0, g1])
    ]
    _csv(tokens, rows)
    roles = tmp_path / "roles.csv"
    _csv(roles, [{**r, "region": "x"} for r in rows])
    bundle = build_feature_profiles(
        tokens, roles, tmp_path / "out", observation_top_k_per_graph=2
    )
    assert (tmp_path / "out" / "graph_profiles" / "feature_profile_index.csv").exists()
    idx10 = next(i for i, r in enumerate(bundle.index) if int(r["feature_id"]) == 10)
    assert int(bundle.index[idx10]["obs_count"]) == 2
    assert float(bundle.index[idx10]["total_abs_mass"]) > 0
    loaded = load_feature_profile_bundle(tmp_path / "out" / "graph_profiles")
    assert loaded.matrices["output"].shape[0] == len(bundle.index)
    assert loaded.matrices["output"].data.min() < 0
    audit = json.loads(
        (tmp_path / "out" / "graph_profiles" / "graph_profile_audit.json").read_text()
    )
    assert audit["bucket_sign_mode"] == "signed_observed"
    assert audit["bucket_weight_stats"]["logit<-feature"]["negative_count"] == 1


def test_within_layer_clustering_separates_opposite_output_profiles(
    tmp_path: Path,
) -> None:
    graphs = []
    for step, (fid, out_w) in enumerate([(10, 2.0), (11, 3.0), (12, -2.0)]):
        p = tmp_path / f"token_{step:06d}" / "graph.npz"
        graphs.append(p)
        _write_graph(
            p,
            step,
            [
                ("feature<-token", _fid(fid), 0, 1.0),
                ("logit<-feature", 0, _fid(fid), out_w),
                ("feature<-feature", _fid(fid), _fid(99), 1.0),
            ],
        )
    tokens = tmp_path / "tokens.csv"
    rows = [
        {
            "wave": "w",
            "prompt_id": "p",
            "label": "correct",
            "trajectory_name": "t",
            "generated_index": str(i),
            "graph_path": str(p),
        }
        for i, p in enumerate(graphs)
    ]
    _csv(tokens, rows)
    roles = tmp_path / "roles.csv"
    _csv(roles, [{**r, "region": "x"} for r in rows])
    build_feature_profiles(tokens, roles, tmp_path / "out")
    cluster_features(
        tmp_path / "out" / "graph_profiles",
        tmp_path / "out",
        neighbors=2,
        clusters_per_layer=2,
        min_affinity=0.01,
        seed=1,
    )
    with (
        tmp_path
        / "out"
        / "feature_clusters"
        / "clusters_v1"
        / "feature_cluster_assignments.csv"
    ).open(newline="") as h:
        by_fid = {int(r["feature_id"]): r for r in csv.DictReader(h)}
    assert by_fid[10]["cluster_id"] == by_fid[11]["cluster_id"]
    assert by_fid[12]["cluster_id"] != by_fid[10]["cluster_id"]
    manifest = json.loads(
        (
            tmp_path
            / "out"
            / "feature_clusters"
            / "clusters_v1"
            / "cluster_manifest.json"
        ).read_text()
    )
    assert manifest["bucket_sign_mode"] == "signed_observed"
    assert manifest["neighbors"] == 2


def test_cli_smoke_extract_profiles_and_cluster(tmp_path: Path) -> None:
    g = tmp_path / "token_000000" / "graph.npz"
    _write_graph(
        g,
        0,
        [
            ("feature<-token", _fid(10), 0, 1.0),
            ("logit<-feature", 0, _fid(10), 1.0),
            ("feature<-feature", _fid(10), _fid(11), 1.0),
        ],
    )
    tokens = tmp_path / "tokens.csv"
    rows = [
        {
            "wave": "w",
            "prompt_id": "p",
            "label": "correct",
            "trajectory_name": "t",
            "generated_index": "0",
            "graph_path": str(g),
        }
    ]
    _csv(tokens, rows)
    roles = tmp_path / "roles.csv"
    _csv(roles, [{**rows[0], "region": "x"}])
    main(
        [
            "extract-profiles",
            "--tokens",
            str(tokens),
            "--roles",
            str(roles),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-graphs",
            "1",
        ]
    )
    main(
        [
            "cluster-features",
            "--profiles-dir",
            str(tmp_path / "out" / "graph_profiles"),
            "--output-dir",
            str(tmp_path / "out"),
            "--neighbors",
            "1",
        ]
    )
    assert (
        tmp_path / "out" / "feature_clusters" / "clusters_v1" / "cluster_manifest.json"
    ).exists()
