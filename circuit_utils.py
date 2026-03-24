"""
Shared utilities for temporal circuit stability analysis.

Provides graph sparsification, compact serialisation, and temporal
metric computation used by both the tracing pipeline and analysis scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# ── configuration ────────────────────────────────────────────────────
STABLE_CORE_WINDOW = 10  # W for stable-core detection
STABLE_CORE_PERSISTENCE = 0.8  # p threshold
MAX_EDGES = 10_000  # default cap on retained edges per step


# ── per-step compact data ────────────────────────────────────────────


@dataclass
class StepData:
    """Compact representation of a single generation step's circuit."""

    step_idx: int
    # Sparse edge representation (COO format)
    row_idx: np.ndarray  # int32
    col_idx: np.ndarray  # int32
    weights: np.ndarray  # float32, normalised to sum=1
    # Feature identity
    feature_ids: np.ndarray  # (F, 3) int64: [layer, position, feature_idx]
    # Generation metadata
    token_text: str
    logprob: float | None
    n_features: int


# ── graph helpers ────────────────────────────────────────────────────


def extract_circuit_subgraph(
    adj: torch.Tensor, n_features: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract feature→feature and feature→logit edges from full adjacency.

    Error and token (embed) nodes carry extreme gradient magnitudes (up to
    1e38, plus NaN/Inf) that swamp the real circuit.  Restricting to
    feature-only edges gives a clean, interpretable subgraph.

    Returns (feat_block, logit_block, logit_start_row).
    """
    logit_start = adj.shape[0] - 2  # circuit-tracer puts logit nodes last
    feat_block = adj[:n_features, :n_features]
    logit_block = adj[logit_start:, :n_features]
    return feat_block, logit_block, logit_start


def sparsify_edges(
    adj: torch.Tensor,
    n_features: int,
    max_edges: int = MAX_EDGES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep top-K feature-circuit edges by absolute weight, normalised.

    Returns (row_idx, col_idx, weights) as numpy arrays — ready for
    compact serialisation.  Row/col indices are in the *original*
    adjacency matrix coordinate system.
    """
    feat_block, logit_block, logit_start = extract_circuit_subgraph(adj, n_features)

    ff_flat = feat_block.abs().float().view(-1)
    fl_flat = logit_block.abs().float().view(-1)
    combined = torch.cat([ff_flat, fl_flat])

    k = min(max_edges, int((combined != 0).sum().item()))
    if k == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    topk_vals, topk_idx = torch.topk(combined, k, sorted=False)

    topk64 = topk_vals.double()
    kept_mass = float(topk64.sum().item())
    if kept_mass == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    ff_size = ff_flat.numel()
    n_feat_cols = feat_block.shape[1]
    n_logit_cols = logit_block.shape[1]

    rows: list[int] = []
    cols: list[int] = []
    in_ff = topk_idx < ff_size

    # Feature→feature indices
    ff_idx = topk_idx[in_ff]
    rows.extend((ff_idx // n_feat_cols).tolist())
    cols.extend((ff_idx % n_feat_cols).tolist())

    # Feature→logit indices (offset into logit block)
    fl_idx = topk_idx[~in_ff] - ff_size
    rows.extend((fl_idx // n_logit_cols + logit_start).tolist())
    cols.extend((fl_idx % n_logit_cols).tolist())

    norm_weights = (topk64 / kept_mass).float().numpy()
    return (
        np.array(rows, dtype=np.int32),
        np.array(cols, dtype=np.int32),
        norm_weights,
    )


# ── compact serialisation ────────────────────────────────────────────


def save_compact(step: StepData, path: Path) -> None:
    """Save a StepData to a compressed .npz file (~1-5 MB)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        row_idx=step.row_idx,
        col_idx=step.col_idx,
        weights=step.weights,
        feature_ids=step.feature_ids,
        token_text=np.array(step.token_text),
        logprob=np.array(step.logprob if step.logprob is not None else np.nan),
        n_features=np.array(step.n_features, dtype=np.int32),
        step_idx=np.array(step.step_idx, dtype=np.int32),
    )


def load_compact(path: Path) -> StepData:
    """Load a StepData from a .npz file."""
    data = np.load(str(path), allow_pickle=False)
    lp = float(data["logprob"])
    return StepData(
        step_idx=int(data["step_idx"]),
        row_idx=data["row_idx"],
        col_idx=data["col_idx"],
        weights=data["weights"],
        feature_ids=data["feature_ids"],
        token_text=str(data["token_text"]),
        logprob=lp if not np.isnan(lp) else None,
        n_features=int(data["n_features"]),
    )


def step_from_pt(
    pt_path: Path, step_idx: int, *, max_edges: int = MAX_EDGES
) -> StepData:
    """Load a raw .pt graph and convert to compact StepData."""
    graph = torch.load(pt_path, map_location="cpu", weights_only=False)
    adj = graph["adjacency_matrix"]
    af = graph["active_features"]  # (F, 3): layer, pos, feat_idx
    n_features = af.shape[0]

    row_idx, col_idx, weights = sparsify_edges(adj, n_features, max_edges=max_edges)

    feature_ids = af.numpy().astype(np.int64)
    del graph, adj

    return StepData(
        step_idx=step_idx,
        row_idx=row_idx,
        col_idx=col_idx,
        weights=weights,
        feature_ids=feature_ids,
        token_text="",
        logprob=None,
        n_features=n_features,
    )


# ── feature helpers ──────────────────────────────────────────────────


def feature_set_from_ids(feature_ids: np.ndarray) -> set[tuple[int, int]]:
    """Extract position-agnostic (layer, feature_idx) set from feature_ids array."""
    return {(int(row[0]), int(row[2])) for row in feature_ids}


def edge_dict_from_coo(
    row_idx: np.ndarray, col_idx: np.ndarray, weights: np.ndarray
) -> dict[tuple[int, int], float]:
    """Convert COO arrays to {(row, col): weight} dict for Jaccard computation."""
    return {(int(r), int(c)): float(w) for r, c, w in zip(row_idx, col_idx, weights)}


# ── temporal metrics ─────────────────────────────────────────────────


def unweighted_jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 1.0


def weighted_jaccard(
    e1: dict[tuple[int, int], float],
    e2: dict[tuple[int, int], float],
) -> float:
    """wJacc = sum min(w1, w2) / sum max(w1, w2) over union of edges."""
    all_edges = set(e1) | set(e2)
    if not all_edges:
        return 1.0
    num = sum(min(e1.get(e, 0.0), e2.get(e, 0.0)) for e in all_edges)
    den = sum(max(e1.get(e, 0.0), e2.get(e, 0.0)) for e in all_edges)
    return num / den if den else 1.0


def compute_temporal_metrics(
    steps: list[StepData],
    *,
    stable_core_window: int = STABLE_CORE_WINDOW,
    stable_core_persistence: float = STABLE_CORE_PERSISTENCE,
) -> dict[str, list]:
    """Compute all temporal metrics over an ordered list of StepData.

    Returns a dict of parallel lists keyed by metric name.
    """
    edge_wjaccard: list[float] = []
    edge_jaccard: list[float] = []
    feature_jaccard: list[float] = []
    core_sizes: list[float] = []
    core_masses: list[float] = []
    edge_history: list[set[tuple[int, int]]] = []

    prev_edges: dict[tuple[int, int], float] | None = None
    prev_features: set[tuple[int, int]] | None = None

    for sd in steps:
        edges = edge_dict_from_coo(sd.row_idx, sd.col_idx, sd.weights)
        edge_set = set(edges.keys())
        features = feature_set_from_ids(sd.feature_ids)

        edge_history.append(edge_set)

        # Consecutive overlap
        if prev_edges is not None and prev_features is not None:
            edge_wjaccard.append(weighted_jaccard(prev_edges, edges))
            edge_jaccard.append(unweighted_jaccard(set(prev_edges.keys()), edge_set))
            feature_jaccard.append(unweighted_jaccard(prev_features, features))
        else:
            edge_wjaccard.append(float("nan"))
            edge_jaccard.append(float("nan"))
            feature_jaccard.append(float("nan"))

        prev_edges = edges
        prev_features = features

        # Stable core
        W = stable_core_window
        p = stable_core_persistence
        if len(edge_history) >= W:
            window = edge_history[-W:]
            all_edges_in_window = set().union(*window)
            core = {
                e
                for e in all_edges_in_window
                if sum(1 for s in window if e in s) / W >= p
            }
            core_sizes.append(len(core))
            core_masses.append(sum(edges.get(e, 0.0) for e in core))
        else:
            core_sizes.append(float("nan"))
            core_masses.append(float("nan"))

    return {
        "edge_wjaccard": edge_wjaccard,
        "edge_jaccard": edge_jaccard,
        "feature_jaccard": feature_jaccard,
        "core_sizes": core_sizes,
        "core_masses": core_masses,
    }
