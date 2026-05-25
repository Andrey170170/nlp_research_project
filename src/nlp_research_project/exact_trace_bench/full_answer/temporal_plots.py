from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from ..io_utils import ensure_dir, iter_jsonl, read_json, write_json


def _rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path)) if path.exists() else []


def _values(rows: Sequence[dict[str, Any]], key: str) -> list[float | None]:
    return [None if row.get(key) is None else float(row[key]) for row in rows]


def _plot_lines(
    ax: Any,
    rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    y_keys: Sequence[str],
    title: str,
    ylabel: str,
) -> None:
    xs = _values(rows, x_key)
    for key in y_keys:
        ax.plot(
            xs, _values(rows, key), marker="o", linewidth=1.5, markersize=3, label=key
        )
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_adjacent_jaccards(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_lines(
        ax,
        rows,
        x_key="generated_index_b",
        y_keys=(
            "feature_jaccard",
            "edge_jaccard",
            "weighted_edge_jaccard",
            "all_edge_weighted_jaccard",
        ),
        title="Adjacent-token Jaccards",
        ylabel="Jaccard",
    )
    return _save(fig, output_dir / "adjacent_jaccards.png")


def _plot_adjacent_churn(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    _plot_lines(
        axes[0],
        rows,
        x_key="generated_index_b",
        y_keys=(
            "features_entered_rate",
            "features_exited_rate",
            "features_stayed_rate",
        ),
        title="Feature churn rates",
        ylabel="Rate",
    )
    _plot_lines(
        axes[1],
        rows,
        x_key="generated_index_b",
        y_keys=(
            "all_edges_entered_rate",
            "all_edges_exited_rate",
            "all_edges_stayed_rate",
        ),
        title="All-edge churn rates",
        ylabel="Rate",
    )
    return _save(fig, output_dir / "adjacent_churn_rates.png")


def _plot_weighted_churn(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_lines(
        ax,
        rows,
        x_key="generated_index_b",
        y_keys=("mass_entered", "mass_exited", "mass_stayed"),
        title="Weighted all-edge churn mass",
        ylabel="Mass",
    )
    return _save(fig, output_dir / "weighted_churn_mass.png")


def _plot_lag_jaccards(summary: dict[str, Any], output_dir: Path) -> Path:
    lag_summaries = summary.get("lag_summaries", {})
    rows = [
        dict(payload, lag=int(lag))
        for lag, payload in sorted(lag_summaries.items(), key=lambda item: int(item[0]))
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_lines(
        ax,
        rows,
        x_key="lag",
        y_keys=(
            "mean_feature_jaccard",
            "mean_edge_jaccard",
            "mean_weighted_edge_jaccard",
            "mean_all_edge_weighted_jaccard",
        ),
        title="Mean Jaccard by lag",
        ylabel="Mean Jaccard",
    )
    return _save(fig, output_dir / "lag_jaccards.png")


def _by_window(rows: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["window"]), []).append(row)
    return grouped


def _plot_rolling_core_sizes(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for window, window_rows in _by_window(rows).items():
        xs = _values(window_rows, "end_generated_index")
        axes[0].plot(
            xs,
            _values(window_rows, "feature_intersection_core_size"),
            marker="o",
            label=f"w{window} intersection",
        )
        axes[0].plot(
            xs,
            _values(window_rows, "feature_persistence80_core_size"),
            marker="o",
            label=f"w{window} persistence80",
        )
        axes[1].plot(
            xs,
            _values(window_rows, "all_edge_intersection_core_size"),
            marker="o",
            label=f"w{window} intersection",
        )
        axes[1].plot(
            xs,
            _values(window_rows, "all_edge_persistence80_core_size"),
            marker="o",
            label=f"w{window} persistence80",
        )
    for ax, title in zip(
        axes, ("Feature rolling core sizes", "All-edge rolling core sizes")
    ):
        ax.set_title(title)
        ax.set_xlabel("end_generated_index")
        ax.set_ylabel("Core size")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
    return _save(fig, output_dir / "rolling_core_sizes.png")


def _plot_rolling_union_churn(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for window, window_rows in _by_window(rows).items():
        xs = _values(window_rows, "end_generated_index")
        for ax, prefix in ((axes[0], "feature_union"), (axes[1], "all_edge_union")):
            for suffix in ("entered_rate", "exited_rate", "stayed_rate"):
                key = f"{prefix}_{suffix}"
                ax.plot(
                    xs,
                    _values(window_rows, key),
                    marker="o",
                    label=f"w{window} {suffix}",
                )
    for ax, title in zip(
        axes,
        ("Feature union rolling churn", "All-edge union rolling churn"),
    ):
        ax.set_title(title)
        ax.set_xlabel("end_generated_index")
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
    return _save(fig, output_dir / "rolling_union_churn.png")


def plot_full_answer_temporal(
    *, analysis_dir: Path, output_dir: Path
) -> dict[str, Any]:
    ensure_dir(output_dir)
    summary = read_json(analysis_dir / "temporal_summary.json")
    adjacent = _rows(analysis_dir / "adjacent_pairs.jsonl")
    rolling = _rows(analysis_dir / "rolling_windows.jsonl")

    generated = [
        _plot_adjacent_jaccards(adjacent, output_dir),
        _plot_adjacent_churn(adjacent, output_dir),
        _plot_weighted_churn(adjacent, output_dir),
        _plot_lag_jaccards(summary, output_dir),
        _plot_rolling_core_sizes(rolling, output_dir),
        _plot_rolling_union_churn(rolling, output_dir),
    ]
    manifest = {
        "analysis_dir": str(analysis_dir),
        "generated_files": [str(path) for path in generated],
    }
    write_json(output_dir / "plot_manifest.json", manifest)
    return manifest
