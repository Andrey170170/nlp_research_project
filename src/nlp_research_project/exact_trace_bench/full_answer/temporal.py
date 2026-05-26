from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, cast

import numpy as np

from ..graph_compare import (
    _all_edge_map,
    _edge_map,
    _feature_set,
    _jaccard,
    _weighted_edge_jaccard,
)
from ..io_utils import ensure_dir, write_json, write_jsonl

if TYPE_CHECKING:
    from circuit_utils import StepData

DEFAULT_WINDOWS = (5, 10, 25)
DEFAULT_LAGS = (1, 2, 4, 8, 16, 32)
EDGE_TOP_KS = (128, 512, 1024)
MASS_CORE_THRESHOLDS = (0.50, 0.80, 0.95)


@dataclass(frozen=True)
class GraphSnapshot:
    generated_index: int
    token_text: str
    features: set[tuple[int, int, int]]
    edges: dict[tuple[object, object], float]
    all_edges: dict[tuple[object, object], float]

    @property
    def positionless_features(self) -> set[tuple[int, int]]:
        return {(layer, feature_id) for layer, _position, feature_id in self.features}


@dataclass(frozen=True)
class CompactStep:
    step_idx: int
    row_idx: np.ndarray
    col_idx: np.ndarray
    weights: np.ndarray
    feature_ids: np.ndarray
    token_text: str
    logprob: float | None
    n_features: int


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
    data = np.load(str(path), allow_pickle=False)
    logprob = float(data["logprob"])
    step = CompactStep(
        step_idx=int(data["step_idx"]),
        row_idx=data["row_idx"],
        col_idx=data["col_idx"],
        weights=data["weights"],
        feature_ids=data["feature_ids"],
        token_text=str(data["token_text"]),
        logprob=logprob if not np.isnan(logprob) else None,
        n_features=int(data["n_features"]),
    )
    generated_index = int(step.step_idx)
    path_index = _graph_path_index(path)
    if generated_index != path_index:
        raise ValueError(
            f"graph step_idx {generated_index} does not match path index {path_index}: {path}"
        )
    step_for_compare = cast("StepData", step)
    return GraphSnapshot(
        generated_index=generated_index,
        token_text=str(getattr(step, "token_text", "")),
        features=_feature_set(step_for_compare),
        edges=_edge_map(step_for_compare),
        all_edges=_all_edge_map(step_for_compare),
    )


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
    a: dict[tuple[object, object], float], b: dict[tuple[object, object], float]
) -> dict[str, float]:
    keys = set(a) | set(b)
    mass_stayed = sum(min(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    mass_entered = sum(max(b.get(key, 0.0) - a.get(key, 0.0), 0.0) for key in keys)
    mass_exited = sum(max(a.get(key, 0.0) - b.get(key, 0.0), 0.0) for key in keys)
    total_mass = mass_stayed + mass_entered + mass_exited
    return {
        "mass_entered": float(mass_entered),
        "mass_exited": float(mass_exited),
        "mass_stayed": float(mass_stayed),
        "total_mass": float(total_mass),
        "all_edges_mass_entered": float(mass_entered),
        "all_edges_mass_exited": float(mass_exited),
        "all_edges_mass_stayed": float(mass_stayed),
        "all_edges_total_mass": float(total_mass),
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
    a: dict[tuple[object, object], float], b: dict[tuple[object, object], float]
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
        prefix = f"all_edge_top{k}"
        denom = min(k, len(a_sorted), len(b_sorted))
        row[f"{prefix}_jaccard"] = _jaccard(ak, bk)
        row[f"{prefix}_overlap_fraction"] = len(shared) / denom if denom else None
        row[f"{prefix}_weighted_jaccard"] = _weighted_jaccard_keys(a, b, ak | bk)
        row[f"{prefix}_mass_fraction_a"] = (
            sum(a[e] for e in ak) / total_a if total_a else None
        )
        row[f"{prefix}_mass_fraction_b"] = (
            sum(b[e] for e in bk) / total_b if total_b else None
        )
        row[f"{prefix}_count_a"] = len(ak)
        row[f"{prefix}_count_b"] = len(bk)
    return row


def _mass_core_metrics(
    a: dict[tuple[object, object], float], b: dict[tuple[object, object], float]
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
        name = f"all_edge_core{int(threshold * 100):02d}"
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


def _l1_distance(a: dict[Any, float], b: dict[Any, float]) -> float:
    return float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)))


def pair_metrics(a: GraphSnapshot, b: GraphSnapshot) -> dict[str, Any]:
    posa = a.positionless_features
    posb = b.positionless_features
    shifted = posa & posb
    flow_a = _layer_flow(a.all_edges)
    flow_b = _layer_flow(b.all_edges)
    flow_mass_a = float(sum(flow_a.values()))
    flow_mass_b = float(sum(flow_b.values()))
    logit_a = sum(v for (_sl, kind, _tl), v in flow_a.items() if kind == "logit")
    logit_b = sum(v for (_sl, kind, _tl), v in flow_b.items() if kind == "logit")
    row: dict[str, Any] = {
        "generated_index_a": a.generated_index,
        "generated_index_b": b.generated_index,
        "token_text_a": a.token_text,
        "token_text_b": b.token_text,
        "feature_jaccard": _jaccard(a.features, b.features),
        "edge_jaccard": _jaccard(set(a.edges), set(b.edges)),
        "weighted_edge_jaccard": _weighted_edge_jaccard(a.edges, b.edges),
        "all_edge_weighted_jaccard": _weighted_edge_jaccard(a.all_edges, b.all_edges),
        "feature_count_a": len(a.features),
        "feature_count_b": len(b.features),
        "edge_count_a": len(a.edges),
        "edge_count_b": len(b.edges),
        "all_edge_count_a": len(a.all_edges),
        "all_edge_count_b": len(b.all_edges),
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
        "layer_flow_weighted_jaccard": _weighted_jaccard_keys(
            flow_a, flow_b, set(flow_a) | set(flow_b)
        ),
        "layer_flow_l1_distance": _l1_distance(flow_a, flow_b),
        "layer_flow_logit_mass_fraction_a": logit_a / flow_mass_a
        if flow_mass_a
        else None,
        "layer_flow_logit_mass_fraction_b": logit_b / flow_mass_b
        if flow_mass_b
        else None,
    }
    row.update(_churn(a.features, b.features, prefix="features"))
    row.update(_churn(posa, posb, prefix="positionless_features"))
    row.update(_churn(set(a.all_edges), set(b.all_edges), prefix="all_edges"))
    row.update(_mass_churn(a.all_edges, b.all_edges))
    row.update(_topk_metrics(a.all_edges, b.all_edges))
    row.update(_mass_core_metrics(a.all_edges, b.all_edges))
    return row


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.nanmean(vals)) if vals else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pair_count": len(rows),
        "mean_feature_jaccard": _mean(rows, "feature_jaccard"),
        "mean_edge_jaccard": _mean(rows, "edge_jaccard"),
        "mean_weighted_edge_jaccard": _mean(rows, "weighted_edge_jaccard"),
        "mean_all_edge_weighted_jaccard": _mean(rows, "all_edge_weighted_jaccard"),
    }


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
            edge_counts = Counter(e for snap in chunk for e in snap.all_edges)
            threshold = int(np.ceil(window * 0.8))
            threshold50 = int(np.ceil(window * 0.5))
            feature_union = set(feature_counts)
            positionless_union = set(positionless_counts)
            edge_union = set(edge_counts)
            feature_intersection = {k for k, v in feature_counts.items() if v == window}
            positionless_intersection = {
                k for k, v in positionless_counts.items() if v == window
            }
            edge_intersection = {k for k, v in edge_counts.items() if v == window}
            total_edge_mass = sum(sum(snap.all_edges.values()) for snap in chunk)
            edge_mass_by_count = Counter()
            for snap in chunk:
                edge_mass_by_count.update(snap.all_edges)
            current = {
                "feature_union": feature_union,
                "positionless_union": positionless_union,
                "edge_union": edge_union,
                "feature_intersection": feature_intersection,
                "positionless_intersection": positionless_intersection,
                "edge_intersection": edge_intersection,
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
                "all_edge_union_size": len(edge_union),
                "all_edge_intersection_core_size": len(edge_intersection),
                "all_edge_persistence50_core_size": sum(
                    v >= threshold50 for v in edge_counts.values()
                ),
                "all_edge_persistence80_core_size": sum(
                    v >= threshold for v in edge_counts.values()
                ),
                "all_edge_persistence100_core_size": len(edge_intersection),
                "all_edge_persistence50_mass_fraction": sum(
                    mass
                    for edge, mass in edge_mass_by_count.items()
                    if edge_counts[edge] >= threshold50
                )
                / total_edge_mass
                if total_edge_mass
                else None,
                "all_edge_persistence80_mass_fraction": sum(
                    mass
                    for edge, mass in edge_mass_by_count.items()
                    if edge_counts[edge] >= threshold
                )
                / total_edge_mass
                if total_edge_mass
                else None,
                "all_edge_persistence100_mass_fraction": sum(
                    mass
                    for edge, mass in edge_mass_by_count.items()
                    if edge_counts[edge] == window
                )
                / total_edge_mass
                if total_edge_mass
                else None,
            }
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
                    _churn(prev["edge_union"], edge_union, prefix="all_edge_union")
                )
                row.update(
                    _churn(
                        prev["feature_intersection"],
                        feature_intersection,
                        prefix="feature_intersection",
                    )
                )
                row.update(
                    _churn(
                        prev["edge_intersection"],
                        edge_intersection,
                        prefix="all_edge_intersection",
                    )
                )
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
    edge_counts: Counter[Any] = Counter()
    edge_mass: Counter[Any] = Counter()
    total_edge_mass = 0.0
    for i, snap in enumerate(snapshots, start=1):
        feature_counts.update(snap.features)
        positionless_counts.update(snap.positionless_features)
        edge_counts.update(snap.all_edges)
        edge_mass.update(snap.all_edges)
        total_edge_mass += sum(snap.all_edges.values())
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
            "all_edge_union_size": len(edge_counts),
        }
        for name, threshold in thresholds.items():
            row[f"feature_persistence{name}_core_size"] = sum(
                v >= threshold for v in feature_counts.values()
            )
            row[f"positionless_feature_persistence{name}_core_size"] = sum(
                v >= threshold for v in positionless_counts.values()
            )
            row[f"all_edge_persistence{name}_core_size"] = sum(
                v >= threshold for v in edge_counts.values()
            )
            row[f"all_edge_persistence{name}_mass_fraction"] = (
                sum(m for e, m in edge_mass.items() if edge_counts[e] >= threshold)
                / total_edge_mass
                if total_edge_mass
                else None
            )
        rows.append(row)
    return rows, (rows[-1] if rows else {})


def layer_flow_rows(snapshots: list[GraphSnapshot]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        flows = _layer_flow(snap.all_edges)
        total = sum(flows.values())
        for (source_layer, target_kind, target_layer), mass in sorted(flows.items()):
            rows.append(
                {
                    "generated_index": snap.generated_index,
                    "token_text": snap.token_text,
                    "source_layer": source_layer,
                    "target_kind": target_kind,
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
