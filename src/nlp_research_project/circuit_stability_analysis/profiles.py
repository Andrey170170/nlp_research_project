from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .signed_graph import decode_feature_endpoint, infer_n_pos, load_signed_graph

VIEWS = ("input", "output", "f2f_layer", "f2f_position")


@dataclass(frozen=True)
class ProfileBundle:
    profiles_dir: Path
    index: list[dict[str, Any]]
    matrices: dict[str, sparse.csr_matrix]
    vocab: dict[str, list[str]]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("wave", ""),
        row.get("prompt_id", ""),
        row.get("label", ""),
        row.get("generated_index", ""),
    )


def _validate_role_join(
    token_rows: list[dict[str, str]], role_rows: list[dict[str, str]]
) -> dict[str, int]:
    role_keys = {_key(row) for row in role_rows}
    token_keys = [_key(row) for row in token_rows]
    missing = [key for key in token_keys if key not in role_keys]
    if missing:
        examples = ", ".join(repr(key) for key in missing[:5])
        raise ValueError(
            "roles table is missing token keys required for profile extraction: "
            f"{examples}"
        )
    return {
        "role_rows": len(role_rows),
        "token_rows_checked": len(token_rows),
        "matched_token_rows": len(token_rows) - len(missing),
    }


def _rel_bin(delta: int) -> str:
    if delta == 0:
        return "same"
    if delta > 0:
        return "future"
    d = abs(delta)
    return "prev_1_2" if d <= 2 else "prev_3_8" if d <= 8 else "prev_9_plus"


def _add(
    vecs: dict[str, dict[tuple[int, int], float]],
    vocabs: dict[str, dict[str, int]],
    view: str,
    fkey: tuple[int, int],
    coord: str,
    weight: float,
) -> None:
    col = vocabs[view].setdefault(coord, len(vocabs[view]))
    vecs[view][(fkey[0], fkey[1], col)] += float(weight)


def _update_bucket_weight_stats(stats: dict[str, dict[str, float]], graph: Any) -> None:
    for bucket_id, bucket in enumerate(graph.bucket_names):
        weights = graph.bucket_weights[graph.bucket_ids == bucket_id]
        bucket_stats = stats.setdefault(
            bucket,
            {
                "count": 0,
                "negative_count": 0,
                "positive_count": 0,
                "zero_count": 0,
                "min_weight": float("inf"),
                "max_weight": float("-inf"),
                "sum_abs_weight": 0.0,
            },
        )
        if weights.size == 0:
            continue
        bucket_stats["count"] += int(weights.size)
        bucket_stats["negative_count"] += int(np.sum(weights < 0))
        bucket_stats["positive_count"] += int(np.sum(weights > 0))
        bucket_stats["zero_count"] += int(np.sum(weights == 0))
        bucket_stats["min_weight"] = min(
            float(bucket_stats["min_weight"]), float(np.min(weights))
        )
        bucket_stats["max_weight"] = max(
            float(bucket_stats["max_weight"]), float(np.max(weights))
        )
        bucket_stats["sum_abs_weight"] += float(
            np.sum(np.abs(weights).astype(np.float64))
        )


def build_feature_profiles(
    tokens: Path,
    roles: Path,
    output_dir: Path,
    *,
    max_graphs: int | None = None,
    observation_top_k_per_graph: int = 0,
    max_edges_per_bucket: int | None = None,
) -> ProfileBundle:
    graph_dir = output_dir / "graph_profiles"
    graph_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(tokens)
    if max_graphs is not None:
        rows = rows[:max_graphs]
    role_join = _validate_role_join(rows, _read_csv(roles))
    vocabs: dict[str, dict[str, int]] = {v: {} for v in VIEWS}
    acc: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    obs: dict[tuple[int, int], set[tuple[str, str, str, str]]] = defaultdict(set)
    vecs: dict[str, dict[tuple[int, int, int], float]] = {
        v: defaultdict(float) for v in VIEWS
    }
    audit = {
        "graphs_seen": 0,
        "missing_graphs": 0,
        "decode_errors": 0,
        "profile_version": "adag_lite_profiles_v1",
        "tokens_path": str(tokens),
        "roles_path": str(roles),
        "max_graphs": max_graphs,
        "max_edges_per_bucket": max_edges_per_bucket,
        "observation_top_k_per_graph": observation_top_k_per_graph,
        **role_join,
    }
    bucket_weight_stats: dict[str, dict[str, float]] = {}
    obs_handle = (graph_dir / "feature_observations.jsonl").open("w")
    try:
        for row in rows:
            gpath = Path(row.get("graph_path", ""))
            if not gpath.exists():
                audit["missing_graphs"] += 1
                continue
            graph = load_signed_graph(gpath)
            audit["graphs_seen"] += 1
            _update_bucket_weight_stats(bucket_weight_stats, graph)
            n_pos = infer_n_pos(graph)
            token_key = _key(row)
            graph_feature_mass: dict[tuple[int, int], float] = defaultdict(float)
            rows_arr, cols_arr, weights_arr = graph.bucket_edge_arrays(
                "feature<-token", max_edges=max_edges_per_bucket
            )
            for row_id, col_id, weight in zip(rows_arr, cols_arr, weights_arr):
                try:
                    tgt = decode_feature_endpoint(int(row_id), n_pos)
                except Exception:
                    audit["decode_errors"] += 1
                    continue
                fkey = (tgt.layer, tgt.feature_id)
                obs[fkey].add(token_key)
                src = int(col_id)
                tok_id = (
                    int(graph.token_ids[src])
                    if graph.token_ids is not None and 0 <= src < len(graph.token_ids)
                    else src
                )
                _add(
                    vecs,
                    vocabs,
                    "input",
                    fkey,
                    f"rel={_rel_bin(src - tgt.position)}|tokbin={tok_id % 1024}",
                    float(weight),
                )
                acc[fkey]["input_abs_mass"] += abs(float(weight))
                acc[fkey]["total_abs_mass"] += abs(float(weight))
                graph_feature_mass[fkey] += abs(float(weight))
            rows_arr, cols_arr, weights_arr = graph.bucket_edge_arrays(
                "logit<-feature", max_edges=max_edges_per_bucket
            )
            for row_id, col_id, weight in zip(rows_arr, cols_arr, weights_arr):
                try:
                    src = decode_feature_endpoint(int(col_id), n_pos)
                except Exception:
                    audit["decode_errors"] += 1
                    continue
                fkey = (src.layer, src.feature_id)
                obs[fkey].add(token_key)
                logit = (
                    int(graph.logit_token_ids[int(row_id)])
                    if graph.logit_token_ids is not None
                    and 0 <= int(row_id) < len(graph.logit_token_ids)
                    else int(row_id)
                )
                _add(vecs, vocabs, "output", fkey, f"logit={logit}", float(weight))
                acc[fkey]["output_abs_mass"] += abs(float(weight))
                acc[fkey]["total_abs_mass"] += abs(float(weight))
                graph_feature_mass[fkey] += abs(float(weight))
            rows_arr, cols_arr, weights_arr = graph.bucket_edge_arrays(
                "feature<-feature", max_edges=max_edges_per_bucket
            )
            for row_id, col_id, weight in zip(rows_arr, cols_arr, weights_arr):
                try:
                    tgt = decode_feature_endpoint(int(row_id), n_pos)
                    src = decode_feature_endpoint(int(col_id), n_pos)
                except Exception:
                    audit["decode_errors"] += 1
                    continue
                for ep, direction, other in ((tgt, "in", src), (src, "out", tgt)):
                    fkey = (ep.layer, ep.feature_id)
                    obs[fkey].add(token_key)
                    _add(
                        vecs,
                        vocabs,
                        "f2f_layer",
                        fkey,
                        f"{direction}|self={ep.layer}|other={other.layer}",
                        float(weight),
                    )
                    _add(
                        vecs,
                        vocabs,
                        "f2f_position",
                        fkey,
                        f"{direction}|{_rel_bin(other.position - ep.position)}",
                        float(weight),
                    )
                    acc[fkey]["f2f_layer_abs_mass"] += abs(float(weight))
                    acc[fkey]["f2f_position_abs_mass"] += abs(float(weight))
                    acc[fkey]["total_abs_mass"] += abs(float(weight))
                    graph_feature_mass[fkey] += abs(float(weight))
            if observation_top_k_per_graph:
                for (layer, fid), mass in sorted(
                    graph_feature_mass.items(), key=lambda x: -x[1]
                )[:observation_top_k_per_graph]:
                    obs_handle.write(
                        json.dumps(
                            {
                                "token_key": token_key,
                                "layer": layer,
                                "feature_id": fid,
                                "abs_mass": mass,
                                "graph_path": str(gpath),
                            }
                        )
                        + "\n"
                    )
    finally:
        obs_handle.close()
    for values in bucket_weight_stats.values():
        if values["count"] == 0:
            values["min_weight"] = 0.0
            values["max_weight"] = 0.0
    audit["bucket_weight_stats"] = bucket_weight_stats
    audit["bucket_sign_mode"] = (
        "signed_observed"
        if any(v["negative_count"] > 0 for v in bucket_weight_stats.values())
        else "nonnegative_observed"
    )
    features = sorted(acc)
    row_of = {f: i for i, f in enumerate(features)}
    index = []
    for layer, fid in features:
        item = {
            "layer": layer,
            "feature_id": fid,
            "obs_count": len(obs[(layer, fid)]),
            **acc[(layer, fid)],
        }
        index.append(item)
    matrices = {}
    for view in VIEWS:
        rr = []
        cc = []
        dd = []
        for (layer, fid, col), val in vecs[view].items():
            rr.append(row_of[(layer, fid)])
            cc.append(col)
            dd.append(val)
        mat = sparse.csr_matrix(
            (dd, (rr, cc)), shape=(len(features), len(vocabs[view])), dtype=np.float64
        )
        norm = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
        norm[norm == 0] = 1.0
        matrices[view] = sparse.diags(1.0 / norm).dot(mat).tocsr()
    _write_bundle(
        graph_dir,
        index,
        matrices,
        {
            k: [x for x, _ in sorted(v.items(), key=lambda kv: kv[1])]
            for k, v in vocabs.items()
        },
        audit,
    )
    return ProfileBundle(
        graph_dir,
        index,
        matrices,
        {
            k: [x for x, _ in sorted(v.items(), key=lambda kv: kv[1])]
            for k, v in vocabs.items()
        },
    )


def _write_bundle(
    path: Path,
    index: list[dict[str, Any]],
    matrices: dict[str, sparse.csr_matrix],
    vocab: dict[str, list[str]],
    audit: dict[str, Any],
) -> None:
    fields = [
        "layer",
        "feature_id",
        "obs_count",
        "total_abs_mass",
        "input_abs_mass",
        "output_abs_mass",
        "f2f_layer_abs_mass",
        "f2f_position_abs_mass",
    ]
    with (path / "feature_profile_index.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows({f: r.get(f, 0) for f in fields} for r in index)
    arrays = {}
    for name, mat in matrices.items():
        arrays[f"{name}_data"] = mat.data
        arrays[f"{name}_indices"] = mat.indices
        arrays[f"{name}_indptr"] = mat.indptr
        arrays[f"{name}_shape"] = np.asarray(mat.shape)
    np.savez_compressed(path / "feature_profile_vectors.npz", **arrays)
    (path / "feature_profile_vocab.json").write_text(
        json.dumps(vocab, indent=2, sort_keys=True)
    )
    (path / "graph_profile_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )


def load_feature_profile_bundle(profiles_dir: Path) -> ProfileBundle:
    p = Path(profiles_dir)
    if (p / "graph_profiles").exists():
        p = p / "graph_profiles"
    index = _read_csv(p / "feature_profile_index.csv")
    data = np.load(p / "feature_profile_vectors.npz", allow_pickle=False)
    matrices = {
        v: sparse.csr_matrix(
            (data[f"{v}_data"], data[f"{v}_indices"], data[f"{v}_indptr"]),
            shape=tuple(data[f"{v}_shape"]),
        ).tocsr()
        for v in VIEWS
    }
    vocab = json.loads((p / "feature_profile_vocab.json").read_text())
    return ProfileBundle(p, index, matrices, vocab)
