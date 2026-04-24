"""Analyze decoder-aware vs compact top-K sparsification comparison.

Expected experiment layout (produced by run_sparsification_experiment.py):
  <experiment_root>/
    exact_compact_<p>/
      artifacts/
        prompt_000/completion_000/
          step_000.npz ...
      run.log
      scenario.json
    exact_save_raw_<p>/
      artifacts/
        prompt_000/completion_000/
          step_000.npz   ← decoder-aware edge sets
          step_000.pt    ← raw Graph, used for independent top-K re-check
      run.log
      scenario.json

The script produces:
  <experiment_root>/decoder_aware_comparison_report.json
  <experiment_root>/decoder_aware_comparison.png

Usage:
    uv run python experiments/analyze_decoder_aware_comparison.py \\
        --experiment-root /path/to/experiment_root
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repo root is importable when run as a script
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from circuit_utils import StepData, load_compact, step_from_pt


# ---------------------------------------------------------------------------
# Per-layer feature counting (key for detecting layer collapse)
# ---------------------------------------------------------------------------

def features_per_layer(step: StepData) -> dict[int, int]:
    """Count unique features retained per model layer for a single step."""
    if step.feature_ids is None or step.feature_ids.shape[0] == 0:
        return {}
    # feature_ids shape: (F, 3) with [layer, position, feature_idx]
    layers = step.feature_ids[:, 0]
    counts: dict[int, int] = defaultdict(int)
    for layer in layers.tolist():
        counts[int(layer)] += 1
    return dict(counts)


def layer_coverage(steps: list[StepData]) -> dict[int, float]:
    """Mean features-per-layer averaged over steps (proxy for layer diversity)."""
    if not steps:
        return {}
    all_counts: dict[int, list[int]] = defaultdict(list)
    for step in steps:
        for layer, count in features_per_layer(step).items():
            all_counts[layer].append(count)
    return {layer: float(np.mean(counts)) for layer, counts in sorted(all_counts.items())}


def layer_presence_rate(steps: list[StepData]) -> dict[int, float]:
    """Fraction of steps in which each layer has at least one retained feature."""
    if not steps:
        return {}
    n = len(steps)
    present: dict[int, int] = defaultdict(int)
    for step in steps:
        for layer in features_per_layer(step):
            present[layer] += 1
    return {layer: present[layer] / n for layer in sorted(present)}


# ---------------------------------------------------------------------------
# Edge mass and Jaccard helpers (simplified, operating on StepData)
# ---------------------------------------------------------------------------

def _feature_set(step: StepData) -> frozenset[tuple[int, int, int]]:
    if step.feature_ids is None or step.feature_ids.shape[0] == 0:
        return frozenset()
    return frozenset(tuple(row) for row in step.feature_ids.tolist())


def _edge_set(step: StepData) -> frozenset[tuple[int, int]]:
    if len(step.row_idx) == 0:
        return frozenset()
    return frozenset(zip(step.row_idx.tolist(), step.col_idx.tolist()))


def feature_jaccard(a: StepData, b: StepData) -> float:
    sa, sb = _feature_set(a), _feature_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def edge_jaccard(a: StepData, b: StepData) -> float:
    ea, eb = _edge_set(a), _edge_set(b)
    if not ea and not eb:
        return 1.0
    return len(ea & eb) / len(ea | eb)


def edge_mass_retained(approx: StepData) -> float:
    """Sum of retained edge weights (should sum to 1.0; lower = edges dropped)."""
    if len(approx.weights) == 0:
        return 0.0
    return float(np.sum(approx.weights))


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_npz_steps(completion_dir: Path) -> list[StepData]:
    files = sorted(completion_dir.glob("step_*.npz"))
    return [load_compact(p) for p in files]


def _load_pt_topk_steps(
    completion_dir: Path,
    max_edges: int = 20000,
) -> list[StepData]:
    """Re-load .pt files with activation_values=None → pure top-K selection."""
    files = sorted(completion_dir.glob("step_*.pt"))
    steps = []
    for pt_file in files:
        idx = int(pt_file.stem.split("_")[1])
        # step_from_pt uses circuit_utils.sparsify_edges;
        # .pt files may or may not have activation_values.
        # We force top-K by calling with a patched graph dict:
        graph_dict = _load_graph_dict_no_activations(pt_file)
        steps.append(_step_from_dict_topk(graph_dict, idx, max_edges=max_edges))
    return steps


def _load_graph_dict_no_activations(pt_file: Path) -> dict[str, Any]:
    import torch
    graph_dict = torch.load(pt_file, map_location="cpu", weights_only=False)
    # Remove activation_values so sparsify_edges falls back to top-K
    if isinstance(graph_dict, dict):
        graph_dict.pop("activation_values", None)
    return graph_dict


def _step_from_dict_topk(
    graph_dict: dict[str, Any],
    step_idx: int,
    max_edges: int,
) -> StepData:
    """Apply step_from_pt logic but with activation_values explicitly absent."""
    from circuit_utils import sparsify_edges, StepData
    import numpy as np
    import torch

    adj = graph_dict.get("adjacency_matrix")
    af = graph_dict.get("active_features")
    if adj is None or af is None:
        return StepData(
            step_idx=step_idx,
            row_idx=np.empty(0, dtype=np.int32),
            col_idx=np.empty(0, dtype=np.int32),
            weights=np.empty(0, dtype=np.float32),
            feature_ids=np.empty((0, 3), dtype=np.int64),
            token_text="",
            logprob=None,
            n_features=0,
        )
    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj)
    if not isinstance(af, torch.Tensor):
        af = torch.tensor(af)
    n_features = int(af.shape[0])
    # activation_values=None → top-K path
    row_idx, col_idx, weights = sparsify_edges(
        adj, n_features, max_edges=max_edges, activation_values=None
    )
    return StepData(
        step_idx=step_idx,
        row_idx=row_idx,
        col_idx=col_idx,
        weights=weights,
        feature_ids=af.cpu().numpy().astype(np.int64),
        token_text=graph_dict.get("token_text", ""),
        logprob=graph_dict.get("logprob"),
        n_features=n_features,
    )


# ---------------------------------------------------------------------------
# Runtime / memory extraction (re-uses patterns from run_sparsification_experiment)
# ---------------------------------------------------------------------------

import re

_PHASE4_RE = re.compile(r"Feature attributions completed in (?P<seconds>[\d.]+)")
_MEMORY_RE = re.compile(
    r"peak.*?rss=(?P<rss>[\d.]+|n/a)\s+GiB.*?cuda_alloc=(?P<cuda_alloc>[\d.]+|n/a)\s+GiB",
    re.IGNORECASE,
)


def _extract_runtime(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    phase4_match = _PHASE4_RE.search(text)
    mem_match = _MEMORY_RE.search(text)
    result: dict[str, Any] = {}
    if phase4_match:
        result["phase4_duration_seconds"] = float(phase4_match.group("seconds"))
    if mem_match:
        rss = mem_match.group("rss")
        cuda = mem_match.group("cuda_alloc")
        result["peak_rss_gib"] = None if rss == "n/a" else float(rss)
        result["peak_cuda_allocated_gib"] = None if cuda == "n/a" else float(cuda)
    return result


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def _find_completion_dirs(scenario_dir: Path) -> list[Path]:
    artifacts = scenario_dir / "artifacts"
    return sorted(artifacts.glob("prompt_*/completion_*")) if artifacts.exists() else []


def analyze_pair(
    compact_dir: Path,
    save_raw_dir: Path,
    max_edges: int = 20000,
) -> dict[str, Any]:
    """Compare compact (top-K) vs save_raw (decoder-aware) scenarios."""
    compact_comps = _find_completion_dirs(compact_dir)
    save_raw_comps = _find_completion_dirs(save_raw_dir)

    compact_runtime = _extract_runtime(compact_dir / "run.log")
    save_raw_runtime = _extract_runtime(save_raw_dir / "run.log")

    per_completion: list[dict[str, Any]] = []

    for comp_path in compact_comps:
        rel = comp_path.relative_to(compact_dir / "artifacts")
        matching_raw = save_raw_dir / "artifacts" / rel
        if not matching_raw.exists():
            continue

        topk_steps = _load_npz_steps(comp_path)
        da_steps = _load_npz_steps(matching_raw)

        has_pt = bool(list(matching_raw.glob("step_*.pt")))
        pt_topk_steps = (
            _load_pt_topk_steps(matching_raw, max_edges=max_edges)
            if has_pt else []
        )

        n = min(len(topk_steps), len(da_steps))
        if n == 0:
            continue

        feat_jaccards, edge_jaccards, mass_retained_da = [], [], []
        topk_layer_counts: list[dict[int, int]] = []
        da_layer_counts: list[dict[int, int]] = []

        for i in range(n):
            tk = topk_steps[i]
            da = da_steps[i]
            feat_jaccards.append(feature_jaccard(tk, da))
            edge_jaccards.append(edge_jaccard(tk, da))
            mass_retained_da.append(edge_mass_retained(da))
            topk_layer_counts.append(features_per_layer(tk))
            da_layer_counts.append(features_per_layer(da))

        # Layer coverage summary
        topk_cov = layer_coverage(topk_steps[:n])
        da_cov = layer_coverage(da_steps[:n])
        topk_pres = layer_presence_rate(topk_steps[:n])
        da_pres = layer_presence_rate(da_steps[:n])

        # Layer collapse: layers that are present in DA but absent from top-K
        all_layers = sorted(set(topk_cov) | set(da_cov))
        max_layer = max(all_layers) if all_layers else 0
        late_layers = [l for l in all_layers if l >= max_layer * 0.6]
        late_topk_mean = float(np.mean([topk_cov.get(l, 0) for l in late_layers])) if late_layers else 0.0
        late_da_mean = float(np.mean([da_cov.get(l, 0) for l in late_layers])) if late_layers else 0.0
        collapse_avoidance = late_da_mean - late_topk_mean  # positive = DA preserves more late-layer features

        # pt cross-check
        pt_check: dict[str, Any] = {}
        if pt_topk_steps:
            n2 = min(len(topk_steps), len(pt_topk_steps))
            pt_feat_jaccards = [
                feature_jaccard(topk_steps[i], pt_topk_steps[i])
                for i in range(n2)
            ]
            pt_check = {
                "n_pt_steps": n2,
                "mean_feature_jaccard_compact_vs_pt_topk": float(np.nanmean(pt_feat_jaccards)),
                "note": "Should be ~1.0 if both paths produce identical top-K feature sets",
            }

        per_completion.append({
            "completion_key": str(rel),
            "n_steps": n,
            "mean_feature_jaccard_topk_vs_da": float(np.nanmean(feat_jaccards)),
            "mean_edge_jaccard_topk_vs_da": float(np.nanmean(edge_jaccards)),
            "layer_coverage_topk": topk_cov,
            "layer_coverage_da": da_cov,
            "layer_presence_rate_topk": topk_pres,
            "layer_presence_rate_da": da_pres,
            "late_layer_mean_features_topk": late_topk_mean,
            "late_layer_mean_features_da": late_da_mean,
            "late_layer_collapse_avoidance": collapse_avoidance,
            "pt_cross_check": pt_check,
        })

    return {
        "compact_scenario": compact_dir.name,
        "save_raw_scenario": save_raw_dir.name,
        "compact_runtime": compact_runtime,
        "save_raw_runtime": save_raw_runtime,
        "runtime_overhead_seconds": (
            (save_raw_runtime.get("phase4_duration_seconds") or 0)
            - (compact_runtime.get("phase4_duration_seconds") or 0)
            if save_raw_runtime.get("phase4_duration_seconds") is not None
            and compact_runtime.get("phase4_duration_seconds") is not None
            else None
        ),
        "per_completion": per_completion,
    }


def _plot_comparison(analysis: dict[str, Any], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    completions = analysis.get("per_completion", [])
    if not completions:
        return

    comp = completions[0]  # first completion
    tkcov = comp["layer_coverage_topk"]
    dacov = comp["layer_coverage_da"]
    all_layers = sorted(set(tkcov) | set(dacov))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-layer mean feature count
    ax = axes[0]
    tk_vals = [tkcov.get(l, 0.0) for l in all_layers]
    da_vals = [dacov.get(l, 0.0) for l in all_layers]
    x = np.arange(len(all_layers))
    w = 0.35
    ax.bar(x - w / 2, tk_vals, w, label="compact top-K", alpha=0.8, color="steelblue")
    ax.bar(x + w / 2, da_vals, w, label="decoder-aware", alpha=0.8, color="darkorange")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([str(all_layers[i]) for i in range(0, len(all_layers), 4)], rotation=45)
    ax.set_xlabel("Model Layer")
    ax.set_ylabel("Mean Features Retained (across steps)")
    ax.set_title("Per-Layer Feature Retention\n(layer collapse visible as missing bars)")
    ax.legend()

    # Right: summary comparison table
    ax2 = axes[1]
    ax2.axis("off")
    rows = [
        ["Metric", "Top-K compact", "Decoder-aware"],
        [
            "Late-layer mean feat.",
            f"{comp['late_layer_mean_features_topk']:.2f}",
            f"{comp['late_layer_mean_features_da']:.2f}",
        ],
        [
            "Late-layer collapse Δ",
            "—",
            f"{comp['late_layer_collapse_avoidance']:+.2f}",
        ],
        [
            "Mean feat. Jaccard",
            "—",
            f"{comp['mean_feature_jaccard_topk_vs_da']:.3f}",
        ],
        [
            "Mean edge Jaccard",
            "—",
            f"{comp['mean_edge_jaccard_topk_vs_da']:.3f}",
        ],
        [
            "Phase-4 duration (s)",
            str(analysis["compact_runtime"].get("phase4_duration_seconds", "n/a")),
            str(analysis["save_raw_runtime"].get("phase4_duration_seconds", "n/a")),
        ],
        [
            "Peak CUDA alloc (GiB)",
            str(analysis["compact_runtime"].get("peak_cuda_allocated_gib", "n/a")),
            str(analysis["save_raw_runtime"].get("peak_cuda_allocated_gib", "n/a")),
        ],
    ]
    table = ax2.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax2.set_title("Summary Comparison", pad=20)

    fig.suptitle(
        f"Decoder-aware vs Top-K sparsification\n"
        f"{analysis['compact_scenario']} vs {analysis['save_raw_scenario']}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze decoder-aware vs top-K sparsification comparison"
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Root directory containing exact_compact_<p> and exact_save_raw_<p> subdirs",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=20000,
        help="Edge budget used when re-loading .pt files for cross-check (default: 20000)",
    )
    args = parser.parse_args()

    root = args.experiment_root
    # Find matching compact / save_raw pair
    compact_dirs = sorted(root.glob("exact_compact_*"))
    save_raw_dirs = sorted(root.glob("exact_save_raw_*"))

    if not compact_dirs:
        print(f"ERROR: no exact_compact_* directories found under {root}")
        sys.exit(1)
    if not save_raw_dirs:
        print(f"ERROR: no exact_save_raw_* directories found under {root}")
        sys.exit(1)

    # Pair by prompt suffix
    pairs: list[tuple[Path, Path]] = []
    for cd in compact_dirs:
        suffix = cd.name.replace("exact_compact_", "")
        matching = [d for d in save_raw_dirs if d.name.endswith(suffix)]
        if matching:
            pairs.append((cd, matching[0]))

    if not pairs:
        print("ERROR: could not find matching compact/save_raw pairs")
        sys.exit(1)

    all_analyses = []
    for compact_dir, save_raw_dir in pairs:
        print(f"Analyzing pair: {compact_dir.name} vs {save_raw_dir.name}")
        analysis = analyze_pair(compact_dir, save_raw_dir, max_edges=args.max_edges)
        all_analyses.append(analysis)

        for comp in analysis["per_completion"]:
            cav = comp["late_layer_collapse_avoidance"]
            direction = "better" if cav > 0 else "worse"
            print(
                f"  [{comp['completion_key']}] "
                f"late-layer collapse avoidance: {cav:+.2f} ({direction} for decoder-aware)  |  "
                f"feat Jaccard: {comp['mean_feature_jaccard_topk_vs_da']:.3f}  "
                f"edge Jaccard: {comp['mean_edge_jaccard_topk_vs_da']:.3f}"
            )

        plot_path = root / f"decoder_aware_comparison_{pairs.index((compact_dir, save_raw_dir)):02d}.png"
        _plot_comparison(analysis, plot_path)

    report = {"experiment_root": str(root), "analyses": all_analyses}
    report_path = root / "decoder_aware_comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {report_path}")


if __name__ == "__main__":
    main()
