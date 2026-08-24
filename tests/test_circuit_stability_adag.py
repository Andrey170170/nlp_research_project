from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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
from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    ArtifactProvenance,
    build_typed_compact_graph,
    save_typed_compact_graph,
)


def _write_graph(
    path: Path, step: int, edges: list[tuple[str, int, int, float]], *, n_pos: int = 4
) -> None:
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
    encoded_to_index = {
        encode_feature_endpoint(*row, n_pos): index
        for index, row in enumerate(feature_ids.tolist())
    }
    feature_layer_count = max(2, int(feature_ids[:, 0].max()) + 1)
    error_count = feature_layer_count * n_pos
    matrices = {
        "feature<-feature": torch.zeros((len(feature_ids), len(feature_ids))),
        "feature<-error": torch.zeros((len(feature_ids), error_count)),
        "feature<-token": torch.zeros((len(feature_ids), n_pos)),
        "logit<-feature": torch.zeros((2, len(feature_ids))),
        "logit<-error": torch.zeros((2, error_count)),
        "logit<-token": torch.zeros((2, n_pos)),
    }
    for name, row, col, weight in edges:
        row_index = encoded_to_index[row] if name.startswith("feature<-") else row
        col_index = encoded_to_index[col] if name.endswith("<-feature") else col
        matrices[name][row_index, col_index] = weight
    compact = {
        "active_features": torch.tensor(feature_ids),
        "selected_features": torch.arange(len(feature_ids)),
        "feature_row_node_indices": torch.arange(len(feature_ids)),
        "logit_row_node_indices": torch.arange(2),
        "feature_feature_edges": matrices["feature<-feature"],
        "feature_error_edges": matrices["feature<-error"],
        "feature_token_edges": matrices["feature<-token"],
        "logit_feature_edges": matrices["logit<-feature"],
        "logit_error_edges": matrices["logit<-error"],
        "logit_token_edges": matrices["logit<-token"],
        "n_error_nodes": error_count,
        "n_token_nodes": n_pos,
        "input_tokens": torch.arange(100, 100 + n_pos),
        "logit_targets": [SimpleNamespace(vocab_idx=200 + i) for i in range(2)],
        "semantic_fingerprint": "adag-semantic-v1",
        "execution_fingerprint": "adag-execution-v1",
    }
    graph = build_typed_compact_graph(
        compact,
        step,
        token_text=f" tok{step}",
        provenance=ArtifactProvenance(
            provider={"provider_id": "adag-fixture"},
            trace={
                "semantic_fingerprint": "adag-semantic-v1",
                "execution_fingerprint": "adag-execution-v1",
            },
            target={"token_text": f" tok{step}"},
            step={"step_idx": step},
        ),
    )
    save_typed_compact_graph(graph, path)


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
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            step_idx=np.asarray(0, dtype=np.int32),
            row_idx=np.asarray([], dtype=np.int32),
            col_idx=np.asarray([], dtype=np.int32),
            weights=np.asarray([], dtype=np.float32),
            feature_ids=np.asarray([], dtype=np.int64).reshape(0, 3),
            token_text=np.asarray("x"),
            logprob=np.asarray(np.nan),
            n_features=np.asarray(0, dtype=np.int32),
        )

    with pytest.raises(ValueError, match="contains legacy edge fields"):
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
                ("feature<-feature", _fid(fid), _fid(fid), 1.0),
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
