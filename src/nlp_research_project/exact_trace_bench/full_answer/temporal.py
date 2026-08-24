from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np

from ..graph_compare import (
    _jaccard,
    _weighted_edge_jaccard,
)
from ..io_utils import ensure_dir, write_json, write_jsonl
from ..typed_compact_graph import (
    FEATURE_ID_BASE,
    TypedCompactGraph,
    load_typed_compact_graph,
)

DEFAULT_WINDOWS = (5, 10, 25)
DEFAULT_LAGS = (1, 2, 4, 8, 16, 32)
EDGE_TOP_KS = (128, 512, 1024)
MASS_CORE_THRESHOLDS = (0.50, 0.80, 0.95)


@dataclass(frozen=True)
class GraphSnapshot:
    generated_index: int
    token_text: str
    features: set[tuple[int, int, int]]
    bucket_edges: dict[str, dict[tuple[object, object], float]]
    bucket_metadata: dict[str, dict[str, Any]]
    error_node_shape: tuple[int, ...] | None = None

    @property
    def positionless_features(self) -> set[tuple[int, int]]:
        return {(layer, feature_id) for layer, _position, feature_id in self.features}

    @cached_property
    def feature_feature_collapsed_flows(
        self,
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int, int, int], float]]:
        return _feature_feature_collapsed_flows(self)


def _graph_path_index(path: Path) -> int:
    return int(path.parent.name.removeprefix("token_"))


def discover_graph_paths(
    run_root: Path, *, max_tokens: int | None = None
) -> list[Path]:
    paths = sorted(
        (run_root / "shards").glob("shard_*/token_*/graph.npz"),
        key=_graph_path_index,
    )
    if max_tokens is not None:
        paths = paths[:max_tokens]
    return paths


def _load_snapshot(path: Path) -> GraphSnapshot:
    graph = load_typed_compact_graph(path)
    generated_index = int(graph.step_idx)
    bucket_edges, bucket_metadata = _load_bucket_edges(graph)
    return GraphSnapshot(
        generated_index=generated_index,
        token_text=graph.token_text,
        features={
            (int(layer), int(position), int(feature_id))
            for layer, position, feature_id in graph.feature_ids.tolist()
        },
        bucket_edges=bucket_edges,
        bucket_metadata=bucket_metadata,
        error_node_shape=graph.error_node_shape,
    )


def _load_bucket_edges(
    graph: TypedCompactGraph,
) -> tuple[dict[str, dict[tuple[object, object], float]], dict[str, dict[str, Any]]]:
    names = list(graph.bucket_names)
    metadata = graph.bucket_metadata
    out = {name: {} for name in names}
    n_pos = len(graph.token_ids)
    feature_endpoints = {
        int(layer) * n_pos * FEATURE_ID_BASE
        + int(position) * FEATURE_ID_BASE
        + int(feature_id): (
            "feature",
            int(layer),
            int(position),
            int(feature_id),
        )
        for layer, position, feature_id in graph.feature_ids.tolist()
    }

    def feature_endpoint(encoded: int) -> tuple[object, ...]:
        try:
            return feature_endpoints[encoded]
        except KeyError as exc:
            raise ValueError(f"unknown persisted feature endpoint: {encoded}") from exc

    def error_endpoint(index: int) -> tuple[object, ...]:
        layer, position = divmod(index, n_pos)
        return ("error", layer, position, int(graph.token_ids[position]))

    def token_endpoint(position: int) -> tuple[object, ...]:
        return ("token", position, int(graph.token_ids[position]))

    def logit_endpoint(index: int) -> tuple[object, ...]:
        return ("logit", int(graph.logit_token_ids[index]))

    def canonical_edge(
        name: str, row: int, col: int
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        if name == "feature<-feature":
            return feature_endpoint(row), feature_endpoint(col)
        if name == "feature<-error":
            return feature_endpoint(row), error_endpoint(col)
        if name == "feature<-token":
            return feature_endpoint(row), token_endpoint(col)
        if name == "logit<-feature":
            return logit_endpoint(row), feature_endpoint(col)
        if name == "logit<-error":
            return logit_endpoint(row), error_endpoint(col)
        if name == "logit<-token":
            return logit_endpoint(row), token_endpoint(col)
        raise ValueError(f"unsupported typed bucket: {name!r}")

    for row, col, weight, bucket_id in zip(
        graph.bucket_row_idx,
        graph.bucket_col_idx,
        graph.bucket_weights,
        graph.bucket_ids,
    ):
        name = names[int(bucket_id)]
        edge = canonical_edge(name, int(row), int(col))
        out[name][edge] = out[name].get(edge, 0.0) + float(abs(weight))
    return out, metadata


def _sanitize_bucket(name: str) -> str:
    return name.replace("<-", "_").replace("-", "_")


def _bucket_pair_metrics(a: GraphSnapshot, b: GraphSnapshot) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name in sorted(set(a.bucket_edges) | set(b.bucket_edges)):
        prefix = f"bucket_{_sanitize_bucket(name)}"
        ea = a.bucket_edges.get(name, {})
        eb = b.bucket_edges.get(name, {})
        row[f"{prefix}_edge_count_a"] = len(ea)
        row[f"{prefix}_edge_count_b"] = len(eb)
        row[f"{prefix}_jaccard"] = _jaccard(set(ea), set(eb))
        row[f"{prefix}_weighted_jaccard"] = _weighted_edge_jaccard(ea, eb)
        row.update(_churn(set(ea), set(eb), prefix=prefix))
        row.update(_mass_churn(ea, eb, prefix=prefix))
        row.update(_topk_metrics(ea, eb, prefix=prefix))
        row.update(_mass_core_metrics(ea, eb, prefix=prefix))
        for side, snap in (("a", a), ("b", b)):
            meta = snap.bucket_metadata.get(name, {})
            for key in (
                "raw_edge_count",
                "raw_abs_mass",
                "retained_edge_count",
                "retained_abs_mass",
                "retained_fraction",
                "cap_bound_before_top_p",
            ):
                if key in meta:
                    row[f"{prefix}_{key}_{side}"] = meta[key]
    return row


def _churn(a: set[Any], b: set[Any], *, prefix: str) -> dict[str, int | float]:
    stayed = a & b
    entered = b - a
    exited = a - b
    denom = len(a | b)
    return {
        f"{prefix}_entered": len(entered),
        f"{prefix}_exited": len(exited),
        f"{prefix}_stayed": len(stayed),
        f"{prefix}_entered_rate": len(entered) / denom if denom else float("nan"),
        f"{prefix}_exited_rate": len(exited) / denom if denom else float("nan"),
        f"{prefix}_stayed_rate": len(stayed) / denom if denom else float("nan"),
    }


def _mass_churn(
    a: dict[tuple[object, object], float],
    b: dict[tuple[object, object], float],
    *,
    prefix: str,
) -> dict[str, float]:
    keys = set(a) | set(b)
    mass_stayed = sum(min(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    mass_entered = sum(max(b.get(key, 0.0) - a.get(key, 0.0), 0.0) for key in keys)
    mass_exited = sum(max(a.get(key, 0.0) - b.get(key, 0.0), 0.0) for key in keys)
    total_mass = mass_stayed + mass_entered + mass_exited
    return {
        f"{prefix}_mass_entered": float(mass_entered),
        f"{prefix}_mass_exited": float(mass_exited),
        f"{prefix}_mass_stayed": float(mass_stayed),
        f"{prefix}_total_mass": float(total_mass),
    }


def _mass(edge_map: dict[tuple[object, object], float]) -> float:
    return float(sum(edge_map.values()))


def _weighted_jaccard_keys(
    a: dict[Any, float],
    b: dict[Any, float],
    keys: set[Any],
) -> float:
    if not keys:
        return float("nan")
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return (
        float(sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys) / den)
        if den
        else float("nan")
    )


def _topk_metrics(
    a: dict[tuple[object, object], float],
    b: dict[tuple[object, object], float],
    *,
    prefix: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    total_a = _mass(a)
    total_b = _mass(b)
    a_sorted = sorted(a, key=lambda k: (-a[k], repr(k)))
    b_sorted = sorted(b, key=lambda k: (-b[k], repr(k)))
    for k in EDGE_TOP_KS:
        ak = set(a_sorted[: min(k, len(a_sorted))])
        bk = set(b_sorted[: min(k, len(b_sorted))])
        shared = ak & bk
        topk_prefix = f"{prefix}_top{k}"
        denom = min(k, len(a_sorted), len(b_sorted))
        row[f"{topk_prefix}_jaccard"] = _jaccard(ak, bk)
        row[f"{topk_prefix}_overlap_fraction"] = len(shared) / denom if denom else None
        row[f"{topk_prefix}_weighted_jaccard"] = _weighted_jaccard_keys(a, b, ak | bk)
        row[f"{topk_prefix}_mass_fraction_a"] = (
            sum(a[e] for e in ak) / total_a if total_a else None
        )
        row[f"{topk_prefix}_mass_fraction_b"] = (
            sum(b[e] for e in bk) / total_b if total_b else None
        )
        row[f"{topk_prefix}_count_a"] = len(ak)
        row[f"{topk_prefix}_count_b"] = len(bk)
    return row


def _mass_core_metrics(
    a: dict[tuple[object, object], float],
    b: dict[tuple[object, object], float],
    *,
    prefix: str,
) -> dict[str, Any]:
    def core(
        edge_map: dict[tuple[object, object], float], threshold: float
    ) -> set[tuple[object, object]]:
        total = _mass(edge_map)
        running = 0.0
        out: set[tuple[object, object]] = set()
        for key in sorted(edge_map, key=lambda k: (-edge_map[k], repr(k))):
            if total and running / total >= threshold:
                break
            out.add(key)
            running += edge_map[key]
        return out

    row: dict[str, Any] = {}
    total_a = _mass(a)
    total_b = _mass(b)
    for threshold in MASS_CORE_THRESHOLDS:
        name = f"{prefix}_core{int(threshold * 100):02d}"
        ca = core(a, threshold)
        cb = core(b, threshold)
        shared = ca & cb
        row[f"{name}_size_a"] = len(ca)
        row[f"{name}_size_b"] = len(cb)
        row[f"{name}_jaccard"] = _jaccard(ca, cb)
        row[f"{name}_weighted_jaccard"] = _weighted_jaccard_keys(a, b, ca | cb)
        row[f"{name}_shared_mass_fraction_a"] = (
            sum(a[e] for e in shared) / total_a if total_a else None
        )
        row[f"{name}_shared_mass_fraction_b"] = (
            sum(b[e] for e in shared) / total_b if total_b else None
        )
    return row


def _layer_flow(
    edge_map: dict[tuple[object, object], float],
) -> dict[tuple[int, str, int | None], float]:
    flows: dict[tuple[int, str, int | None], float] = {}
    for target, source in edge_map:
        if not (
            isinstance(source, tuple) and len(source) >= 4 and source[0] == "feature"
        ):
            continue
        source_layer = int(cast(Any, source[1]))
        if isinstance(target, tuple) and target and target[0] == "feature":
            key = (source_layer, "feature", int(cast(Any, target[1])))
        else:
            key = (source_layer, "logit", None)
        flows[key] = flows.get(key, 0.0) + edge_map[(target, source)]
    return flows


def _feature_feature_collapsed_flows(
    snap: GraphSnapshot,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int, int, int], float]]:
    edges = snap.bucket_edges.get("feature<-feature")
    layer_flow: dict[tuple[int, int], float] = {}
    positionless_flow: dict[tuple[int, int, int, int], float] = {}
    if not edges:
        return layer_flow, positionless_flow
    for target, source in edges:
        if not (
            isinstance(target, tuple)
            and len(target) == 4
            and target[0] == "feature"
            and isinstance(source, tuple)
            and len(source) == 4
            and source[0] == "feature"
        ):
            raise ValueError("feature<-feature edge has non-feature endpoint")
        target_layer = int(cast(Any, target[1]))
        target_feature = int(cast(Any, target[3]))
        source_layer = int(cast(Any, source[1]))
        source_feature = int(cast(Any, source[3]))
        mass = edges[(target, source)]
        layer_key = (source_layer, target_layer)
        positionless_key = (source_layer, source_feature, target_layer, target_feature)
        layer_flow[layer_key] = layer_flow.get(layer_key, 0.0) + mass
        positionless_flow[positionless_key] = (
            positionless_flow.get(positionless_key, 0.0) + mass
        )
    return layer_flow, positionless_flow


def _l1_distance(a: dict[Any, float], b: dict[Any, float]) -> float:
    return float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)))


def pair_metrics(a: GraphSnapshot, b: GraphSnapshot) -> dict[str, Any]:
    posa = a.positionless_features
    posb = b.positionless_features
    shifted = posa & posb
    ff_layer_flow_a, ff_positionless_flow_a = a.feature_feature_collapsed_flows
    ff_layer_flow_b, ff_positionless_flow_b = b.feature_feature_collapsed_flows
    row: dict[str, Any] = {
        "generated_index_a": a.generated_index,
        "generated_index_b": b.generated_index,
        "token_text_a": a.token_text,
        "token_text_b": b.token_text,
        "feature_jaccard": _jaccard(a.features, b.features),
        "feature_count_a": len(a.features),
        "feature_count_b": len(b.features),
        "positionless_feature_jaccard": _jaccard(posa, posb),
        "positionless_feature_count_a": len(posa),
        "positionless_feature_count_b": len(posb),
        "shifted_position_reuse_count": len(
            shifted
            - {
                (layer, feature_id)
                for layer, _p, feature_id in (a.features & b.features)
            }
        ),
        "shifted_position_reuse_fraction_a": len(shifted) / len(posa) if posa else None,
        "shifted_position_reuse_fraction_b": len(shifted) / len(posb) if posb else None,
        "bucket_feature_feature_layer_flow_weighted_jaccard": _weighted_jaccard_keys(
            ff_layer_flow_a,
            ff_layer_flow_b,
            set(ff_layer_flow_a) | set(ff_layer_flow_b),
        ),
        "bucket_feature_feature_positionless_flow_weighted_jaccard": _weighted_jaccard_keys(
            ff_positionless_flow_a,
            ff_positionless_flow_b,
            set(ff_positionless_flow_a) | set(ff_positionless_flow_b),
        ),
    }
    row.update(_churn(a.features, b.features, prefix="features"))
    row.update(_churn(posa, posb, prefix="positionless_features"))
    row.update(_bucket_pair_metrics(a, b))
    return row


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    vals = [
        value
        for row in rows
        if row.get(key) is not None and not np.isnan(value := float(row[key]))
    ]
    return float(np.mean(vals)) if vals else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "pair_count": len(rows),
        "mean_feature_jaccard": _mean(rows, "feature_jaccard"),
        "mean_bucket_feature_feature_layer_flow_weighted_jaccard": _mean(
            rows, "bucket_feature_feature_layer_flow_weighted_jaccard"
        ),
        "mean_bucket_feature_feature_positionless_flow_weighted_jaccard": _mean(
            rows, "bucket_feature_feature_positionless_flow_weighted_jaccard"
        ),
    }
    bucket_prefixes = sorted(
        {
            key.removesuffix("_weighted_jaccard")
            for row in rows
            for key in row
            if key.startswith("bucket_")
            and key.endswith("_weighted_jaccard")
            and "_top" not in key
            and "_core" not in key
            and "_flow_" not in key
        }
    )
    for prefix in bucket_prefixes:
        summary[f"mean_{prefix}_jaccard"] = _mean(rows, f"{prefix}_jaccard")
        summary[f"mean_{prefix}_weighted_jaccard"] = _mean(
            rows, f"{prefix}_weighted_jaccard"
        )
    return summary


def rolling_window_rows(
    snapshots: list[GraphSnapshot], windows: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: dict[int, dict[str, set[Any]]] = {}
    for window in windows:
        for end in range(window - 1, len(snapshots)):
            chunk = snapshots[end - window + 1 : end + 1]
            feature_counts = Counter(f for snap in chunk for f in snap.features)
            positionless_counts = Counter(
                f for snap in chunk for f in snap.positionless_features
            )
            threshold = int(np.ceil(window * 0.8))
            threshold50 = int(np.ceil(window * 0.5))
            feature_union = set(feature_counts)
            positionless_union = set(positionless_counts)
            feature_intersection = {k for k, v in feature_counts.items() if v == window}
            positionless_intersection = {
                k for k, v in positionless_counts.items() if v == window
            }
            current = {
                "feature_union": feature_union,
                "positionless_union": positionless_union,
                "feature_intersection": feature_intersection,
                "positionless_intersection": positionless_intersection,
            }
            row: dict[str, Any] = {
                "window": window,
                "end_generated_index": snapshots[end].generated_index,
                "start_generated_index": snapshots[end - window + 1].generated_index,
                "feature_union_size": len(feature_union),
                "feature_intersection_core_size": len(feature_intersection),
                "feature_persistence50_core_size": sum(
                    v >= threshold50 for v in feature_counts.values()
                ),
                "feature_persistence80_core_size": sum(
                    v >= threshold for v in feature_counts.values()
                ),
                "feature_persistence100_core_size": len(feature_intersection),
                "positionless_feature_union_size": len(positionless_union),
                "positionless_feature_intersection_core_size": len(
                    positionless_intersection
                ),
                "positionless_feature_persistence50_core_size": sum(
                    v >= threshold50 for v in positionless_counts.values()
                ),
                "positionless_feature_persistence80_core_size": sum(
                    v >= threshold for v in positionless_counts.values()
                ),
                "positionless_feature_persistence100_core_size": len(
                    positionless_intersection
                ),
            }
            bucket_names = sorted(
                {name for snap in chunk for name in snap.bucket_edges}
            )
            for bucket_name in bucket_names:
                prefix = f"bucket_{_sanitize_bucket(bucket_name)}"
                edge_counts = Counter(
                    edge
                    for snap in chunk
                    for edge in snap.bucket_edges.get(bucket_name, {})
                )
                edge_mass: Counter[Any] = Counter()
                total_mass = 0.0
                for snap in chunk:
                    edges = snap.bucket_edges.get(bucket_name, {})
                    edge_mass.update(edges)
                    total_mass += sum(edges.values())
                edge_union = set(edge_counts)
                edge_intersection = {
                    edge for edge, count in edge_counts.items() if count == window
                }
                current[f"{prefix}_union"] = edge_union
                current[f"{prefix}_intersection"] = edge_intersection
                row[f"{prefix}_union_size"] = len(edge_union)
                row[f"{prefix}_intersection_core_size"] = len(edge_intersection)
                for label, cutoff in (
                    ("50", threshold50),
                    ("80", threshold),
                    ("100", window),
                ):
                    row[f"{prefix}_persistence{label}_core_size"] = sum(
                        count >= cutoff for count in edge_counts.values()
                    )
                    row[f"{prefix}_persistence{label}_mass_fraction"] = (
                        sum(
                            mass
                            for edge, mass in edge_mass.items()
                            if edge_counts[edge] >= cutoff
                        )
                        / total_mass
                        if total_mass
                        else None
                    )
            if window in previous:
                prev = previous[window]
                row.update(
                    _churn(prev["feature_union"], feature_union, prefix="feature_union")
                )
                row.update(
                    _churn(
                        prev["positionless_union"],
                        positionless_union,
                        prefix="positionless_feature_union",
                    )
                )
                row.update(
                    _churn(
                        prev["feature_intersection"],
                        feature_intersection,
                        prefix="feature_intersection",
                    )
                )
                for key, value in current.items():
                    if key.startswith("bucket_") and key in prev:
                        row.update(_churn(prev[key], value, prefix=key))
            previous[window] = current
            rows.append(row)
    return rows


def token_timeline_rows(snapshots: list[GraphSnapshot]) -> list[dict[str, Any]]:
    denom = max(len(snapshots) - 1, 1)
    rows = []
    for i, snap in enumerate(snapshots):
        frac = i / denom
        rows.append(
            {
                "generated_index": snap.generated_index,
                "token_text": snap.token_text,
                "phase_fraction": frac,
                "phase_bin": "early"
                if frac < 1 / 3
                else "middle"
                if frac < 2 / 3
                else "late",
                "is_punctuation": bool(
                    snap.token_text.strip() in {".", ",", ";", ":", "!", "?"}
                ),
                "contains_newline": "\n" in snap.token_text,
            }
        )
    return rows


def cumulative_core_rows(
    snapshots: list[GraphSnapshot],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_counts: Counter[Any] = Counter()
    positionless_counts: Counter[Any] = Counter()
    bucket_counts: dict[str, Counter[Any]] = {}
    bucket_mass: dict[str, Counter[Any]] = {}
    bucket_total_mass: Counter[str] = Counter()
    for i, snap in enumerate(snapshots, start=1):
        feature_counts.update(snap.features)
        positionless_counts.update(snap.positionless_features)
        for bucket_name, edges in snap.bucket_edges.items():
            bucket_counts.setdefault(bucket_name, Counter()).update(edges)
            bucket_mass.setdefault(bucket_name, Counter()).update(edges)
            bucket_total_mass[bucket_name] += sum(edges.values())
        thresholds = {
            "50": int(np.ceil(i * 0.5)),
            "80": int(np.ceil(i * 0.8)),
            "100": i,
        }
        row: dict[str, Any] = {
            "generated_index": snap.generated_index,
            "prefix_length": i,
            "feature_union_size": len(feature_counts),
            "positionless_feature_union_size": len(positionless_counts),
        }
        for name, threshold in thresholds.items():
            row[f"feature_persistence{name}_core_size"] = sum(
                v >= threshold for v in feature_counts.values()
            )
            row[f"positionless_feature_persistence{name}_core_size"] = sum(
                v >= threshold for v in positionless_counts.values()
            )
            for bucket_name, edge_counts in sorted(bucket_counts.items()):
                prefix = f"bucket_{_sanitize_bucket(bucket_name)}"
                row[f"{prefix}_union_size"] = len(edge_counts)
                row[f"{prefix}_persistence{name}_core_size"] = sum(
                    count >= threshold for count in edge_counts.values()
                )
                total_mass = bucket_total_mass[bucket_name]
                row[f"{prefix}_persistence{name}_mass_fraction"] = (
                    sum(
                        mass
                        for edge, mass in bucket_mass[bucket_name].items()
                        if edge_counts[edge] >= threshold
                    )
                    / total_mass
                    if total_mass
                    else None
                )
        rows.append(row)
    return rows, (rows[-1] if rows else {})


def layer_flow_rows(snapshots: list[GraphSnapshot]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        flows, _positionless = snap.feature_feature_collapsed_flows
        total = sum(flows.values())
        for (source_layer, target_layer), mass in sorted(flows.items()):
            rows.append(
                {
                    "generated_index": snap.generated_index,
                    "token_text": snap.token_text,
                    "source_layer": source_layer,
                    "bucket": "feature<-feature",
                    "target_layer": target_layer,
                    "mass": mass,
                    "mass_fraction": mass / total if total else None,
                }
            )
    return rows


def analyze_full_answer_temporal(
    *,
    run_root: Path,
    output_dir: Path,
    windows: list[int],
    lags: list[int],
    max_tokens: int | None = None,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    snapshots = [
        _load_snapshot(path)
        for path in discover_graph_paths(run_root, max_tokens=max_tokens)
    ]
    snapshots.sort(key=lambda snap: snap.generated_index)
    indices = [snap.generated_index for snap in snapshots]
    expected = set(range(indices[0], indices[-1] + 1)) if indices else set()
    missing = sorted(expected - set(indices))

    adjacent = [pair_metrics(a, b) for a, b in zip(snapshots, snapshots[1:])]
    lag_rows: list[dict[str, Any]] = []
    lag_summaries: dict[str, Any] = {}
    by_index = {snap.generated_index: snap for snap in snapshots}
    for lag in lags:
        rows = [
            pair_metrics(by_index[i], by_index[i + lag]) | {"lag": lag}
            for i in indices
            if i + lag in by_index
        ]
        lag_rows.extend(rows)
        lag_summaries[str(lag)] = _summary(rows)
    rolling = rolling_window_rows(snapshots, windows)
    timeline = token_timeline_rows(snapshots)
    cumulative, global_core_summary = cumulative_core_rows(snapshots)
    layer_flows = layer_flow_rows(snapshots)
    summary = {
        "run_root": str(run_root),
        "token_count": len(snapshots),
        "generated_index_min": min(indices) if indices else None,
        "generated_index_max": max(indices) if indices else None,
        "missing_indices": missing,
        "adjacent_summary": _summary(adjacent),
        "lag_summaries": lag_summaries,
        "rolling_window_summary": {
            str(w): {"row_count": sum(r["window"] == w for r in rolling)}
            for w in windows
        },
        "global_core_summary": global_core_summary,
    }
    write_jsonl(output_dir / "adjacent_pairs.jsonl", adjacent)
    write_jsonl(output_dir / "lag_pairs.jsonl", lag_rows)
    write_jsonl(output_dir / "rolling_windows.jsonl", rolling)
    write_jsonl(output_dir / "token_timeline.jsonl", timeline)
    write_jsonl(output_dir / "cumulative_core.jsonl", cumulative)
    write_jsonl(output_dir / "layer_flow_by_token.jsonl", layer_flows)
    write_json(output_dir / "temporal_summary.json", summary)
    return summary
