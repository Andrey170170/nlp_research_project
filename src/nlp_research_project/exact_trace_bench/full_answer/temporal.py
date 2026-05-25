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


@dataclass(frozen=True)
class GraphSnapshot:
    generated_index: int
    token_text: str
    features: set[tuple[int, int, int]]
    edges: dict[tuple[object, object], float]
    all_edges: dict[tuple[object, object], float]


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


def pair_metrics(a: GraphSnapshot, b: GraphSnapshot) -> dict[str, Any]:
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
    }
    row.update(_churn(a.features, b.features, prefix="features"))
    row.update(_churn(set(a.all_edges), set(b.all_edges), prefix="all_edges"))
    row.update(_mass_churn(a.all_edges, b.all_edges))
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
            edge_counts = Counter(e for snap in chunk for e in snap.all_edges)
            threshold = int(np.ceil(window * 0.8))
            feature_union = set(feature_counts)
            edge_union = set(edge_counts)
            feature_intersection = {k for k, v in feature_counts.items() if v == window}
            edge_intersection = {k for k, v in edge_counts.items() if v == window}
            current = {
                "feature_union": feature_union,
                "edge_union": edge_union,
                "feature_intersection": feature_intersection,
                "edge_intersection": edge_intersection,
            }
            row: dict[str, Any] = {
                "window": window,
                "end_generated_index": snapshots[end].generated_index,
                "start_generated_index": snapshots[end - window + 1].generated_index,
                "feature_union_size": len(feature_union),
                "feature_intersection_core_size": len(feature_intersection),
                "feature_persistence80_core_size": sum(
                    v >= threshold for v in feature_counts.values()
                ),
                "all_edge_union_size": len(edge_union),
                "all_edge_intersection_core_size": len(edge_intersection),
                "all_edge_persistence80_core_size": sum(
                    v >= threshold for v in edge_counts.values()
                ),
            }
            if window in previous:
                prev = previous[window]
                row.update(
                    _churn(prev["feature_union"], feature_union, prefix="feature_union")
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
    }
    write_jsonl(output_dir / "adjacent_pairs.jsonl", adjacent)
    write_jsonl(output_dir / "lag_pairs.jsonl", lag_rows)
    write_jsonl(output_dir / "rolling_windows.jsonl", rolling)
    write_json(output_dir / "temporal_summary.json", summary)
    return summary
