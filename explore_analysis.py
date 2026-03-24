"""
Exploratory analysis of traced attribution graphs.

Computes temporal stability metrics from per-step .pt graph files
and produces summary statistics + plots to sanity-check whether
the temporal circuit stability hypothesis is viable.

Runs on CPU (safe for login nodes with --workers 1).
Use --workers N on a compute node to parallelise graph loading.

Usage:
    uv run python explore_analysis.py [COMPLETION_DIR] [--workers N]

Default: experiments/traces/prompt_000/completion_000, workers=1
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# ── configuration ────────────────────────────────────────────────────
ALPHA = 0.95  # cumulative-mass coverage for sparsification
STABLE_CORE_WINDOW = 10  # W for stable-core detection
STABLE_CORE_PERSISTENCE = 0.8  # p threshold
END_OF_TURN_ID = 106  # Gemma-3 <end_of_turn>


# ── per-step extracted data ──────────────────────────────────────────


@dataclass
class StepData:
    step_idx: int
    edges: dict[tuple[int, int], float]
    feature_set: set[tuple[int, int]]
    n_active_features: int
    n_edges_retained: int
    total_edge_mass: float


# ── graph helpers ────────────────────────────────────────────────────


MAX_EDGES = 10_000  # cap on retained edges per step (memory/speed guard)


def _extract_circuit_subgraph(
    adj: torch.Tensor, n_features: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract the feature circuit: feature→feature and feature→logit edges.

    Error and token (embed) nodes carry extreme gradient magnitudes (up to
    1e38, plus NaN/Inf) that swamp the real circuit.  Restricting to
    feature-only edges gives a clean, interpretable subgraph.

    Returns a (n_features + n_logits, n_features) submatrix where rows are
    target nodes [features, logits] and columns are source feature nodes.
    Original row/col indices are preserved in the returned edge keys.
    """
    # feature→feature block: adj[:n_feat, :n_feat]
    # logit→feature block:   adj[-2:, :n_feat]  (last 2 nodes are logits)
    # We only keep target rows that are features or logits
    logit_start = adj.shape[0] - 2  # circuit-tracer puts logit nodes last
    feat_block = adj[:n_features, :n_features]
    logit_block = adj[logit_start:, :n_features]
    return feat_block, logit_block, logit_start


def sparsify_edges(
    adj: torch.Tensor,
    n_features: int,
    max_edges: int = MAX_EDGES,
) -> dict[tuple[int, int], float]:
    """Keep top-K feature-circuit edges by absolute weight, normalised to sum to 1.

    Only considers feature→feature and feature→logit edges.  Error and token
    nodes are excluded — they carry extreme gradient magnitudes (NaN/Inf,
    values up to 1e38) that are reconstruction artefacts, not circuit signal.
    """
    feat_block, logit_block, logit_start = _extract_circuit_subgraph(adj, n_features)

    # Flatten both blocks, topk across the union
    ff_flat = feat_block.abs().float().view(-1)
    fl_flat = logit_block.abs().float().view(-1)
    combined = torch.cat([ff_flat, fl_flat])

    k = min(max_edges, int((combined != 0).sum().item()))
    if k == 0:
        return {}

    topk_vals, topk_idx = torch.topk(combined, k, sorted=False)

    # Normalise in float64
    topk64 = topk_vals.double()
    kept_mass = float(topk64.sum().item())
    if kept_mass == 0:
        return {}

    # Map flat indices back to original (row, col) in full adjacency matrix
    ff_size = ff_flat.numel()
    n_feat_cols = feat_block.shape[1]
    n_logit_cols = logit_block.shape[1]

    rows_list: list[int] = []
    cols_list: list[int] = []
    in_ff = topk_idx < ff_size
    # Feature→feature indices
    ff_idx = topk_idx[in_ff]
    rows_list.extend((ff_idx // n_feat_cols).tolist())
    cols_list.extend((ff_idx % n_feat_cols).tolist())
    # Feature→logit indices (offset into logit block)
    fl_idx = topk_idx[~in_ff] - ff_size
    rows_list.extend((fl_idx // n_logit_cols + logit_start).tolist())
    cols_list.extend((fl_idx % n_logit_cols).tolist())

    vals = (topk64 / kept_mass).tolist()
    return {(r, c): v for r, c, v in zip(rows_list, cols_list, vals)}


def active_feature_set(graph_data: dict) -> set[tuple[int, int]]:
    """Extract set of (layer, feature_idx) tuples — position-agnostic."""
    af = graph_data["active_features"]  # (F, 3): layer, pos, feat_idx
    return {(int(row[0].item()), int(row[2].item())) for row in af}


def _process_single_step(
    pt_path: Path, step_idx: int, *, max_edges: int = MAX_EDGES
) -> StepData:
    """Load a .pt graph, sparsify, extract features. Thread-safe."""
    graph = torch.load(pt_path, map_location="cpu", weights_only=False)
    adj = graph["adjacency_matrix"]
    af_set = active_feature_set(graph)
    n_features = graph["active_features"].shape[0]
    # Total mass of the feature circuit (excl. error/token nodes)
    feat_block, logit_block, _ = _extract_circuit_subgraph(adj, n_features)
    circuit_vals = torch.cat([feat_block.abs().view(-1), logit_block.abs().view(-1)])
    total_mass = float(circuit_vals.double().sum().item())
    edges = sparsify_edges(adj, n_features, max_edges=max_edges)
    del graph, adj  # free ~460 MB immediately
    return StepData(
        step_idx=step_idx,
        edges=edges,
        feature_set=af_set,
        n_active_features=len(af_set),
        n_edges_retained=len(edges),
        total_edge_mass=total_mass,
    )


# ── temporal metrics ─────────────────────────────────────────────────


def unweighted_jaccard(s1: set[tuple[int, int]], s2: set[tuple[int, int]]) -> float:
    if not s1 and not s2:
        return 1.0
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 1.0


def weighted_jaccard(
    e1: dict[tuple[int, int], float],
    e2: dict[tuple[int, int], float],
) -> float:
    """wJacc = sum min(m1, m2) / sum max(m1, m2)  over union of edges."""
    all_edges = set(e1) | set(e2)
    if not all_edges:
        return 1.0
    num = sum(min(e1.get(e, 0.0), e2.get(e, 0.0)) for e in all_edges)
    den = sum(max(e1.get(e, 0.0), e2.get(e, 0.0)) for e in all_edges)
    return num / den if den else 1.0


# ── main analysis ────────────────────────────────────────────────────


def analyse_completion(
    completion_dir: Path, *, workers: int = 1, max_edges: int = MAX_EDGES
) -> None:
    manifest_path = completion_dir / "completion.json"
    if not manifest_path.exists():
        print(f"No completion.json in {completion_dir}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    steps = manifest["steps"]
    n_steps = len(steps)

    # Find the meaningful generation boundary (first <end_of_turn>)
    meaningful_end = n_steps
    for s in steps:
        if s["next_token_id"] == END_OF_TURN_ID:
            meaningful_end = s["step_index"]
            break

    print(f"Completion: {completion_dir}")
    print(f"Total steps traced: {n_steps}")
    print(f"Meaningful steps (before <end_of_turn>): {meaningful_end}")
    print(
        f"Completion text (first 500 chars): {manifest.get('completion_text', '')[:500]}"
    )
    print()

    # ── phase 1: load + sparsify all steps (parallelisable) ──────────
    step_range = range(min(meaningful_end + 1, n_steps))
    pt_paths = []
    for idx in step_range:
        p = completion_dir / f"step_{idx:03d}.pt"
        if p.exists():
            pt_paths.append((p, idx))
        else:
            print(f"  Warning: {p} not found, skipping")

    print(f"Loading {len(pt_paths)} graphs (workers={workers})...")
    step_data_map: dict[int, StepData] = {}

    if workers <= 1:
        # Sequential — minimal memory footprint
        for pt_path, idx in tqdm(pt_paths, desc="Loading graphs", unit="step"):
            step_data_map[idx] = _process_single_step(pt_path, idx, max_edges=max_edges)
    else:
        # Parallel — trades memory for speed (~460 MB per in-flight graph)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_single_step, pt_path, idx, max_edges=max_edges
                ): idx
                for pt_path, idx in pt_paths
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Loading graphs ({workers}w)",
                unit="step",
            ):
                sd = fut.result()
                step_data_map[sd.step_idx] = sd

    ordered_indices = sorted(step_data_map.keys())

    # ── phase 2: sequential temporal metrics ─────────────────────────
    step_indices: list[int] = []
    n_active_features: list[int] = []
    n_edges_retained: list[int] = []
    total_edge_mass: list[float] = []
    logprobs: list[float | None] = []
    token_texts: list[str] = []

    edge_wjaccard: list[float] = []
    edge_jaccard: list[float] = []
    feature_jaccard: list[float] = []

    edge_history: list[set[tuple[int, int]]] = []
    core_sizes: list[float] = []
    core_masses: list[float] = []

    prev: StepData | None = None

    for step_idx in tqdm(ordered_indices, desc="Computing metrics", unit="step"):
        sd = step_data_map[step_idx]

        step_indices.append(step_idx)
        n_active_features.append(sd.n_active_features)
        n_edges_retained.append(sd.n_edges_retained)
        total_edge_mass.append(sd.total_edge_mass)

        lp = steps[step_idx]["next_token_logprob"] if step_idx < len(steps) else None
        logprobs.append(lp)
        token_texts.append(
            steps[step_idx]["next_token_text"] if step_idx < len(steps) else ""
        )

        edge_set = set(sd.edges.keys())
        edge_history.append(edge_set)

        # Consecutive overlap
        if prev is not None:
            edge_wjaccard.append(weighted_jaccard(prev.edges, sd.edges))
            edge_jaccard.append(unweighted_jaccard(set(prev.edges.keys()), edge_set))
            feature_jaccard.append(unweighted_jaccard(prev.feature_set, sd.feature_set))
        else:
            edge_wjaccard.append(float("nan"))
            edge_jaccard.append(float("nan"))
            feature_jaccard.append(float("nan"))

        prev = sd

        # Stable core (sliding window)
        W = STABLE_CORE_WINDOW
        p = STABLE_CORE_PERSISTENCE
        if len(edge_history) >= W:
            window = edge_history[-W:]
            all_edges_in_window = set().union(*window)
            core = set()
            for e in all_edges_in_window:
                freq = sum(1 for s_set in window if e in s_set) / W
                if freq >= p:
                    core.add(e)
            core_sizes.append(len(core))
            core_mass = sum(sd.edges.get(e, 0.0) for e in core)
            core_masses.append(core_mass)
        else:
            core_sizes.append(float("nan"))
            core_masses.append(float("nan"))

    # Free the big map
    del step_data_map

    # ── summary statistics ───────────────────────────────────────────
    valid_wj = [v for v in edge_wjaccard if not np.isnan(v)]
    valid_ej = [v for v in edge_jaccard if not np.isnan(v)]
    valid_fj = [v for v in feature_jaccard if not np.isnan(v)]
    valid_cores = [v for v in core_sizes if not np.isnan(v)]
    valid_core_mass = [v for v in core_masses if not np.isnan(v)]
    valid_lp = [v for v in logprobs if v is not None]

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Steps analysed: {len(step_indices)} (meaningful: {meaningful_end})")
    print(
        f"Active features per step: mean={np.mean(n_active_features):.0f}, "
        f"std={np.std(n_active_features):.0f}"
    )
    print(
        f"Retained edges per step (alpha={ALPHA}): mean={np.mean(n_edges_retained):.0f}, "
        f"std={np.std(n_edges_retained):.0f}"
    )
    print()
    print("Step-to-step overlap (consecutive pairs):")
    if valid_wj:
        print(
            f"  Weighted Jaccard (edges):   mean={np.mean(valid_wj):.4f}, "
            f"std={np.std(valid_wj):.4f}, min={np.min(valid_wj):.4f}, max={np.max(valid_wj):.4f}"
        )
    else:
        print("  Weighted Jaccard (edges):   NO VALID DATA")
    if valid_ej:
        print(
            f"  Unweighted Jaccard (edges): mean={np.mean(valid_ej):.4f}, "
            f"std={np.std(valid_ej):.4f}"
        )
    if valid_fj:
        print(
            f"  Feature Jaccard:            mean={np.mean(valid_fj):.4f}, "
            f"std={np.std(valid_fj):.4f}"
        )
    if valid_wj:
        print(f"  Churn (1 - wJacc):          mean={1 - np.mean(valid_wj):.4f}")
    print()
    if valid_cores:
        print(f"Stable core (W={STABLE_CORE_WINDOW}, p={STABLE_CORE_PERSISTENCE}):")
        print(
            f"  Core size:  mean={np.mean(valid_cores):.0f}, "
            f"std={np.std(valid_cores):.0f}, "
            f"min={np.min(valid_cores):.0f}, max={np.max(valid_cores):.0f}"
        )
        print(
            f"  Core mass fraction: mean={np.mean(valid_core_mass):.4f}, "
            f"std={np.std(valid_core_mass):.4f}"
        )
    print()
    if valid_lp:
        print(
            f"Log-probabilities: mean={np.mean(valid_lp):.4f}, "
            f"std={np.std(valid_lp):.4f}"
        )
    print()

    # ── key signal check ─────────────────────────────────────────────
    print("=" * 60)
    print("HYPOTHESIS VIABILITY CHECK")
    print("=" * 60)
    print()

    mean_wj = float(np.mean(valid_wj)) if valid_wj else float("nan")
    mean_core = float(np.mean(valid_cores)) if valid_cores else 0.0
    print("H1 — Temporal structure exists?")
    if not valid_wj:
        print("  INSUFFICIENT DATA (no valid edge overlap values)")
    elif mean_wj > 0.01 and mean_wj < 0.99:
        print(
            f"  YES: wJaccard overlap mean={mean_wj:.4f} (neither trivial nor perfect)"
        )
        print(f"  Stable core exists with mean size={mean_core:.0f} edges")
    else:
        print(f"  UNCLEAR: wJaccard overlap mean={mean_wj:.4f}")
    print()

    # Check if overlap changes over time (early vs late)
    mid = len(valid_wj) // 2
    if mid > 5:
        early_wj = np.mean(valid_wj[:mid])
        late_wj = np.mean(valid_wj[mid:])
        print(f"  Early-half wJacc: {early_wj:.4f}")
        print(f"  Late-half wJacc:  {late_wj:.4f}")
        if late_wj > early_wj:
            print("  -> Circuits stabilise over generation (promising for H2)")
        else:
            print("  -> Circuits do NOT stabilise monotonically")
    print()

    # Correlation: logprob vs overlap
    if len(valid_lp) > 10 and len(valid_wj) > 10:
        lps = [logprobs[i] for i in range(1, len(logprobs)) if logprobs[i] is not None]
        wjs = valid_wj[: len(lps)]
        if len(lps) == len(wjs) and len(lps) > 5:
            corr = float(
                np.corrcoef(np.array(lps, dtype=np.float64), np.array(wjs))[0, 1]
            )
            print(f"Correlation(logprob, wJaccard): {corr:.4f}")
            if abs(corr) < 0.5:
                print(
                    "  -> Low correlation: temporal features may carry independent signal (good for H2)"
                )
            else:
                print(
                    "  -> Moderate/high correlation: temporal features may be confounded with confidence"
                )

    print()

    # ── plots ────────────────────────────────────────────────────────
    out_dir = completion_dir / "analysis_1"
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Temporal Circuit Analysis — {completion_dir.name}", fontsize=14)
    steps_arr = np.array(step_indices)

    # 1. Overlap metrics over time
    ax = axes[0, 0]
    ax.plot(steps_arr[1:], valid_wj, label="Weighted Jaccard", alpha=0.8)
    ax.plot(steps_arr[1:], valid_ej, label="Unweighted Jaccard", alpha=0.8)
    ax.plot(steps_arr[1:], valid_fj, label="Feature Jaccard", alpha=0.6, linestyle="--")
    ax.axvline(meaningful_end, color="red", linestyle=":", label="<end_of_turn>")
    ax.set_xlabel("Step")
    ax.set_ylabel("Jaccard overlap")
    ax.set_title("Step-to-step overlap")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)

    # 2. Churn over time
    ax = axes[0, 1]
    churn = [1 - v for v in valid_wj]
    ax.plot(steps_arr[1:], churn, color="tomato", alpha=0.8)
    ax.axvline(meaningful_end, color="red", linestyle=":", label="<end_of_turn>")
    ax.set_xlabel("Step")
    ax.set_ylabel("Churn (1 - wJacc)")
    ax.set_title("Edge churn")
    ax.set_ylim(-0.02, 1.02)

    # 3. Active features & retained edges
    ax = axes[1, 0]
    ax.plot(steps_arr, n_active_features, label="Active features", alpha=0.8)
    ax.plot(steps_arr, n_edges_retained, label=f"Retained edges (α={ALPHA})", alpha=0.8)
    ax.axvline(meaningful_end, color="red", linestyle=":", label="<end_of_turn>")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.set_title("Graph size over time")
    ax.legend(fontsize=8)

    # 4. Stable core size
    ax = axes[1, 1]
    valid_core_steps = steps_arr[STABLE_CORE_WINDOW - 1 :]
    if len(valid_cores) > 0:
        ax.plot(
            valid_core_steps[: len(valid_cores)], valid_cores, color="teal", alpha=0.8
        )
        ax.axvline(meaningful_end, color="red", linestyle=":", label="<end_of_turn>")
    ax.set_xlabel("Step")
    ax.set_ylabel("Core size (edges)")
    ax.set_title(f"Stable core (W={STABLE_CORE_WINDOW}, p={STABLE_CORE_PERSISTENCE})")

    # 5. Log-probabilities
    ax = axes[2, 0]
    valid_lp_steps = [
        step_indices[i] for i in range(len(logprobs)) if logprobs[i] is not None
    ]
    if valid_lp:
        ax.plot(
            valid_lp_steps,
            valid_lp,
            color="purple",
            alpha=0.7,
            marker=".",
            markersize=3,
        )
        ax.axvline(meaningful_end, color="red", linestyle=":", label="<end_of_turn>")
    ax.set_xlabel("Step")
    ax.set_ylabel("Log-probability")
    ax.set_title("Next-token log-probability")

    # 6. Scatter: logprob vs wJaccard (independence check)
    ax = axes[2, 1]
    if len(valid_lp) > 10 and len(valid_wj) > 10:
        lps_plot = [
            logprobs[i] for i in range(1, len(logprobs)) if logprobs[i] is not None
        ]
        wjs_plot = valid_wj[: len(lps_plot)]
        ax.scatter(lps_plot, wjs_plot, alpha=0.4, s=10, color="steelblue")
        ax.set_xlabel("Log-probability")
        ax.set_ylabel("Weighted Jaccard")
        ax.set_title("Independence check: logprob vs overlap")

    plt.tight_layout()
    plot_path = out_dir / "temporal_analysis.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Saved plots to {plot_path}")

    # ── save numeric results ─────────────────────────────────────────
    results = {
        "completion_dir": str(completion_dir),
        "n_steps_analysed": len(step_indices),
        "meaningful_end": meaningful_end,
        "alpha": ALPHA,
        "stable_core_window": STABLE_CORE_WINDOW,
        "stable_core_persistence": STABLE_CORE_PERSISTENCE,
        "summary": {
            "mean_weighted_jaccard": float(np.mean(valid_wj)) if valid_wj else None,
            "std_weighted_jaccard": float(np.std(valid_wj)) if valid_wj else None,
            "mean_unweighted_jaccard": float(np.mean(valid_ej)) if valid_ej else None,
            "mean_feature_jaccard": float(np.mean(valid_fj)) if valid_fj else None,
            "mean_churn": float(1 - np.mean(valid_wj)) if valid_wj else None,
            "mean_core_size": float(np.mean(valid_cores)) if valid_cores else None,
            "mean_core_mass_fraction": float(np.mean(valid_core_mass))
            if valid_core_mass
            else None,
            "mean_logprob": float(np.mean(valid_lp)) if valid_lp else None,
        },
        "per_step": {
            "step_index": step_indices,
            "n_active_features": n_active_features,
            "n_edges_retained": n_edges_retained,
            "edge_wjaccard": [
                float(v) if not np.isnan(v) else None for v in edge_wjaccard
            ],
            "edge_jaccard": [
                float(v) if not np.isnan(v) else None for v in edge_jaccard
            ],
            "feature_jaccard": [
                float(v) if not np.isnan(v) else None for v in feature_jaccard
            ],
            "core_size": [float(v) if not np.isnan(v) else None for v in core_sizes],
            "core_mass_fraction": [
                float(v) if not np.isnan(v) else None for v in core_masses
            ],
            "logprob": [float(v) if v is not None else None for v in logprobs],
            "token_text": token_texts,
        },
    }
    results_path = out_dir / "temporal_metrics.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal circuit analysis")
    parser.add_argument(
        "completion_dir",
        nargs="?",
        default="experiments/traces/prompt_000/completion_000",
        help="Path to completion directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for graph loading (1=sequential, safe for login nodes)",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=MAX_EDGES,
        help=f"Cap on retained edges per step (default: {MAX_EDGES})",
    )
    args = parser.parse_args()

    analyse_completion(
        Path(args.completion_dir), workers=args.workers, max_edges=args.max_edges
    )
