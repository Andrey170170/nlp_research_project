from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_BENCHMARK_CSV = Path(
    "experiments/extracted/weekend_exact_chunked/benchmark_enriched.csv"
)
DEFAULT_FEATURE_CSV = Path(
    "experiments/extracted/feature_distribution_analysis/feature_distribution_prompts.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/figures/optimization_benchmarks")

PROMPT_ORDER = [94, 361, 828]
CLUSTER_ORDER = ["ascend", "cardinal"]
BATCH_ORDER = [128, 192, 256, 384]
CHUNK_ORDER = [2048, 4096, 8192, 16384]
PROMPT_COLORS = {94: "#d62728", 361: "#1f77b4", 828: "#2ca02c"}
BATCH_COLORS = {128: "#1f77b4", 192: "#ff7f0e", 256: "#2ca02c", 384: "#d62728"}
CLUSTER_COLORS = {"ascend": "#1f77b4", "cardinal": "#ff7f0e"}
MARKERS = {2048: "o", 4096: "s", 8192: "^", 16384: "D"}


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(x) < 2:
        return None
    coeffs = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = coeffs[0] * xs + coeffs[1]
    return xs, ys


def _annotate_points(ax: plt.Axes, xs, ys, labels, fontsize: int = 7) -> None:
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(
            label, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=fontsize
        )


def load_tables(
    benchmark_csv: Path, feature_csv: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bench = pd.read_csv(benchmark_csv)
    feat = pd.read_csv(feature_csv)

    bench_bool_cols = [
        "is_special_case",
        "slurm_any_ram_oom",
        "slurm_any_cuda_oom",
        "slurm_any_timeout",
    ]
    for col in bench_bool_cols:
        if col in bench.columns:
            bench[col] = bench[col].astype(str).str.lower().eq("true")

    numeric_cols = [
        "duration_seconds",
        "gsm8k_index",
        "attribution_batch_size",
        "decoder_chunk_size",
        "decoder_cache_gib",
        "prompt_token_count",
        "initial_input_token_count",
        "max_active_features",
        "resource_snapshot_rss_gib",
        "resource_snapshot_cuda_peak_reserved_gib",
        "phase4_feature_attribution_seconds",
        "phase0_encode_seconds",
        "phase0_reconstruction_seconds",
    ]
    for col in numeric_cols:
        if col in bench.columns:
            bench[col] = pd.to_numeric(bench[col], errors="coerce")

    feat_numeric_cols = [
        "gsm8k_index",
        "token_count",
        "prompt_token_count",
        "total_active_features",
        "active_features_per_token",
        "phase0_encode_seconds",
        "phase0_reconstruction_seconds",
        "duration_seconds",
        "resource_after_rss_gib",
        "resource_after_cuda_peak_reserved_gib",
    ]
    for col in feat_numeric_cols:
        if col in feat.columns:
            feat[col] = pd.to_numeric(feat[col], errors="coerce")

    return bench, feat


def plot_phase0_feature_distribution_scaling(feat: pd.DataFrame, out_dir: Path) -> None:
    data = feat[feat["status"] == "success"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    panels = [
        (
            "token_count",
            "total_active_features",
            "Prompt tokens",
            "Active features",
            "Phase 0: prompt length vs active features",
        ),
        (
            "total_active_features",
            "duration_seconds",
            "Active features",
            "Phase 0 duration (s)",
            "Phase 0: active features vs runtime",
        ),
        (
            "total_active_features",
            "resource_after_cuda_peak_reserved_gib",
            "Active features",
            "Peak VRAM after prompt (GiB)",
            "Phase 0: active features vs peak VRAM",
        ),
    ]

    for ax, (x_col, y_col, xlab, ylab, title) in zip(axes, panels):
        sub = data[[x_col, y_col]].dropna()
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        ax.scatter(x, y, alpha=0.7, s=18, color="#1f77b4")
        fit = _fit_line(x, y)
        if fit is not None:
            ax.plot(fit[0], fit[1], color="#d62728", linewidth=1.5)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    fig.suptitle("Feature-distribution analysis: Phase-0 scaling", fontsize=14)
    _save(fig, out_dir, "phase0_feature_distribution_scaling")


def plot_exact_feature_scaling(bench: pd.DataFrame, out_dir: Path) -> None:
    data = bench[(bench["status"] == "success") & (~bench["is_special_case"])].copy()
    data["runtime_minutes"] = data["duration_seconds"] / 60.0
    data["active_features_m"] = data["max_active_features"] / 1_000_000.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [
        (
            "runtime_minutes",
            "Total runtime (min)",
            "Exact tracing: active features vs total runtime",
        ),
        (
            "resource_snapshot_rss_gib",
            "Final host RSS (GiB)",
            "Exact tracing: active features vs host RAM",
        ),
        (
            "resource_snapshot_cuda_peak_reserved_gib",
            "Peak VRAM reserved (GiB)",
            "Exact tracing: active features vs peak VRAM",
        ),
    ]

    for ax, (y_col, ylab, title) in zip(axes, panels):
        for cluster in CLUSTER_ORDER:
            sub = data[data["cluster"] == cluster].dropna(
                subset=["active_features_m", y_col]
            )
            ax.scatter(
                sub["active_features_m"],
                sub[y_col],
                label=cluster,
                alpha=0.8,
                s=36,
                color=CLUSTER_COLORS[cluster],
            )
        fit = _fit_line(
            data["active_features_m"].dropna().to_numpy(dtype=float),
            data[y_col].dropna().to_numpy(dtype=float),
        )
        if fit is not None:
            ax.plot(fit[0], fit[1], color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Active features (millions)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False)
    fig.suptitle(
        "Exact tracing scaling on successful runs (excluding prompt94_compare)",
        fontsize=14,
    )
    _save(fig, out_dir, "exact_feature_scaling")


def _plot_wave1_grid(
    data: pd.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    out_dir: Path,
    stem: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
    for row_idx, cluster in enumerate(CLUSTER_ORDER):
        for col_idx, prompt_id in enumerate(PROMPT_ORDER):
            ax = axes[row_idx, col_idx]
            sub = data[
                (data["cluster"] == cluster) & (data["gsm8k_index"] == prompt_id)
            ].copy()
            for batch in BATCH_ORDER:
                batch_sub = sub[sub["attribution_batch_size"] == batch].sort_values(
                    "decoder_chunk_size"
                )
                if batch_sub.empty:
                    continue
                ax.plot(
                    batch_sub["decoder_chunk_size"],
                    batch_sub[value_col],
                    marker="o",
                    linewidth=1.6,
                    label=f"b{batch}",
                    color=BATCH_COLORS.get(batch, None),
                )
            ax.set_title(f"{cluster} · prompt {prompt_id}")
            ax.set_xlabel("Decoder chunk size")
            ax.set_ylabel(ylabel)
            ax.set_xticks(
                [
                    c
                    for c in CHUNK_ORDER
                    if c in sub["decoder_chunk_size"].dropna().unique()
                ]
            )
            ax.grid(alpha=0.25)
    legend_batches = sorted(
        {int(v) for v in data["attribution_batch_size"].dropna().unique().tolist()}
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=BATCH_COLORS.get(batch, "#444444"),
            marker="o",
            linewidth=1.6,
            label=f"b{batch}",
        )
        for batch in legend_batches
    ]
    if handles:
        labels = [str(h.get_label()) for h in handles]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def plot_wave1_config_figures(bench: pd.DataFrame, out_dir: Path) -> None:
    data = bench[
        (bench["status"] == "success")
        & (~bench["is_special_case"])
        & (bench["stage"].astype(str).str.contains("wave1"))
    ].copy()
    data["runtime_minutes"] = data["duration_seconds"] / 60.0
    cluster_style = {
        "ascend": {"linestyle": "-", "marker": "o"},
        "cardinal": {"linestyle": "--", "marker": "s"},
    }

    def _plot_overlay(value_col: str, ylabel: str, title: str, stem: str) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=False)
        value_max = data[value_col].dropna().max()
        for col_idx, prompt_id in enumerate(PROMPT_ORDER):
            ax = axes[col_idx]
            sub = data[data["gsm8k_index"] == prompt_id].copy()
            for cluster in CLUSTER_ORDER:
                for batch in sorted(
                    sub.loc[sub["cluster"] == cluster, "attribution_batch_size"]
                    .dropna()
                    .unique()
                ):
                    batch_sub = sub[
                        (sub["cluster"] == cluster)
                        & (sub["attribution_batch_size"] == batch)
                    ].sort_values("decoder_chunk_size")
                    if batch_sub.empty:
                        continue
                    style = cluster_style[cluster]
                    ax.plot(
                        batch_sub["decoder_chunk_size"],
                        batch_sub[value_col],
                        color=BATCH_COLORS.get(int(batch), "#444444"),
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        linewidth=1.8,
                        alpha=0.95,
                    )
            ax.set_title(f"prompt {prompt_id}")
            ax.set_xlabel("Decoder chunk size")
            ax.set_ylabel(ylabel)
            ax.set_xticks(
                [
                    c
                    for c in CHUNK_ORDER
                    if c in sub["decoder_chunk_size"].dropna().unique()
                ]
            )
            ax.set_ylim(bottom=0, top=float(value_max) * 1.08)
            ax.grid(alpha=0.25)

        batch_handles = [
            Line2D(
                [0],
                [0],
                color=BATCH_COLORS.get(batch, "#444444"),
                marker="o",
                linewidth=1.8,
                label=f"batch {batch}",
            )
            for batch in sorted(
                {
                    int(v)
                    for v in data["attribution_batch_size"].dropna().unique().tolist()
                }
            )
        ]
        cluster_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=cluster_style[c]["linestyle"],
                marker=cluster_style[c]["marker"],
                linewidth=1.8,
                label=c,
            )
            for c in CLUSTER_ORDER
        ]
        handles = batch_handles + cluster_handles
        fig.legend(
            handles,
            [str(h.get_label()) for h in handles],
            loc="upper center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, 1.04),
        )
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        _save(fig, out_dir, stem)

    _plot_overlay(
        "runtime_minutes",
        "Runtime (min)",
        "Wave-1 successful runs: runtime by config (Ascend and Cardinal overlaid by prompt)",
        "wave1_runtime_by_config",
    )
    _plot_overlay(
        "resource_snapshot_cuda_peak_reserved_gib",
        "Peak VRAM reserved (GiB)",
        "Wave-1 successful runs: peak VRAM by config (Ascend and Cardinal overlaid by prompt)",
        "wave1_peak_vram_by_config",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False, sharey=False)
    x_max = data["resource_snapshot_cuda_peak_reserved_gib"].dropna().max()
    y_max = data["runtime_minutes"].dropna().max()
    for col_idx, prompt_id in enumerate(PROMPT_ORDER):
        ax = axes[col_idx]
        sub = data[data["gsm8k_index"] == prompt_id].copy()
        for cluster in CLUSTER_ORDER:
            cluster_sub = sub[sub["cluster"] == cluster]
            ax.scatter(
                cluster_sub["resource_snapshot_cuda_peak_reserved_gib"],
                cluster_sub["runtime_minutes"],
                c=[
                    BATCH_COLORS.get(int(v), "#444444")
                    for v in cluster_sub["attribution_batch_size"]
                ],
                marker=cluster_style[cluster]["marker"],
                s=70,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.4,
            )
        ax.set_title(f"prompt {prompt_id}")
        ax.set_xlabel("Peak VRAM reserved (GiB)")
        ax.set_ylabel("Runtime (min)")
        ax.set_xlim(left=0, right=float(x_max) * 1.08)
        ax.set_ylim(bottom=0, top=float(y_max) * 1.08)
        ax.grid(alpha=0.25)

    batch_handles = [
        Line2D(
            [0],
            [0],
            color=BATCH_COLORS.get(batch, "#444444"),
            marker="o",
            linewidth=0,
            markersize=8,
            label=f"batch {batch}",
        )
        for batch in sorted(
            {int(v) for v in data["attribution_batch_size"].dropna().unique().tolist()}
        )
    ]
    cluster_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=cluster_style[c]["marker"],
            linewidth=0,
            markersize=8,
            label=c,
        )
        for c in CLUSTER_ORDER
    ]
    handles = batch_handles + cluster_handles
    fig.legend(
        handles,
        [str(h.get_label()) for h in handles],
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle("Wave-1 successful runs: runtime/VRAM tradeoff by prompt", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "wave1_runtime_vram_tradeoff")


def plot_wave2_cache_sweeps(bench: pd.DataFrame, out_dir: Path) -> None:
    data = bench[
        (bench["stage"].astype(str).str.contains("wave2"))
        & (bench["cluster"] == "ascend")
        & (~bench["is_special_case"])
        & (bench["gsm8k_index"] == 361)
        & (bench["fixture_name"].isin(["361_base", "361_late"]))
    ].copy()
    if data.empty:
        return

    data["runtime_minutes"] = data["duration_seconds"] / 60.0
    data["config_label"] = data.apply(
        lambda r: (
            f"b{int(r['attribution_batch_size'])}/c{int(r['decoder_chunk_size'])}"
        ),
        axis=1,
    )
    config_colors = {
        label: BATCH_COLORS.get(
            int(
                data.loc[data["config_label"] == label, "attribution_batch_size"].iloc[
                    0
                ]
            ),
            "#444444",
        )
        for label in sorted(data["config_label"].unique())
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col", sharey="row")
    fixture_order = ["361_base", "361_late"]
    success_runtime_max = (
        data.loc[data["status"] == "success", "runtime_minutes"].dropna().max()
    )
    if pd.isna(success_runtime_max):
        success_runtime_max = 1.0
    vram_max = (
        data.loc[
            data["status"] == "success", "resource_snapshot_cuda_peak_reserved_gib"
        ]
        .dropna()
        .max()
    )
    if pd.isna(vram_max):
        vram_max = 1.0

    for col_idx, fixture in enumerate(fixture_order):
        fixture_data = data[data["fixture_name"] == fixture].copy()
        for config_label in sorted(fixture_data["config_label"].unique()):
            sub = fixture_data[
                fixture_data["config_label"] == config_label
            ].sort_values("decoder_cache_gib")
            succ = sub[sub["status"] == "success"]
            oom = sub[sub["status"] != "success"]

            if not succ.empty:
                axes[0, col_idx].plot(
                    succ["decoder_cache_gib"],
                    succ["runtime_minutes"],
                    marker="o",
                    linewidth=1.8,
                    color=config_colors[config_label],
                )
                axes[1, col_idx].plot(
                    succ["decoder_cache_gib"],
                    succ["resource_snapshot_cuda_peak_reserved_gib"],
                    marker="o",
                    linewidth=1.8,
                    color=config_colors[config_label],
                )

            if not oom.empty:
                axes[0, col_idx].scatter(
                    oom["decoder_cache_gib"],
                    np.full(len(oom), success_runtime_max * 1.05),
                    marker="x",
                    s=80,
                    linewidths=2,
                    color=config_colors[config_label],
                )

        axes[0, col_idx].set_title(fixture)
        axes[0, col_idx].set_ylabel("Runtime (min)")
        axes[0, col_idx].set_ylim(bottom=0, top=success_runtime_max * 1.12)
        axes[0, col_idx].set_xlabel("Decoder cache budget (GiB)")
        axes[0, col_idx].tick_params(axis="x", labelbottom=True)
        axes[0, col_idx].tick_params(axis="y", labelleft=True)
        axes[0, col_idx].grid(alpha=0.25)

        axes[1, col_idx].set_title(fixture)
        axes[1, col_idx].set_ylabel("Peak VRAM reserved (GiB)")
        axes[1, col_idx].set_xlabel("Decoder cache budget (GiB)")
        axes[1, col_idx].set_ylim(bottom=0, top=vram_max * 1.12)
        axes[1, col_idx].tick_params(axis="y", labelleft=True)
        axes[1, col_idx].grid(alpha=0.25)

    handles = [
        Line2D(
            [0], [0], color=config_colors[label], marker="o", linewidth=1.8, label=label
        )
        for label in sorted(config_colors)
    ]
    handles.append(
        Line2D(
            [0], [0], color="black", marker="x", linewidth=0, markersize=8, label="OOM"
        )
    )
    labels = [str(h.get_label()) for h in handles]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Wave-2 cache sweeps on prompt 361 (Ascend)", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "wave2_cache_sweeps")


def plot_cluster_shared_config_comparison(bench: pd.DataFrame, out_dir: Path) -> None:
    data = bench[
        (bench["stage"].astype(str).str.contains("wave1"))
        & (bench["status"] == "success")
        & (~bench["is_special_case"])
    ].copy()
    key_cols = [
        "gsm8k_index",
        "attribution_batch_size",
        "decoder_chunk_size",
        "decoder_cache_gib",
    ]
    grouped = data.groupby(key_cols)

    pairs = []
    for key, group in grouped:
        if set(group["cluster"]) != {"ascend", "cardinal"}:
            continue
        a = group[group["cluster"] == "ascend"].iloc[0]
        c = group[group["cluster"] == "cardinal"].iloc[0]
        pairs.append(
            {
                "gsm8k_index": key[0],
                "batch": key[1],
                "chunk": key[2],
                "ascend_runtime_min": a["duration_seconds"] / 60.0,
                "cardinal_runtime_min": c["duration_seconds"] / 60.0,
                "ascend_vram_gib": a["resource_snapshot_cuda_peak_reserved_gib"],
                "cardinal_vram_gib": c["resource_snapshot_cuda_peak_reserved_gib"],
            }
        )
    pair_df = pd.DataFrame(pairs)

    pair_df["config_label"] = pair_df.apply(
        lambda r: f"b{int(r['batch'])}\nc{int(r['chunk'])}", axis=1
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    runtime_max = max(
        pair_df["ascend_runtime_min"].max(), pair_df["cardinal_runtime_min"].max()
    )
    vram_max = max(pair_df["ascend_vram_gib"].max(), pair_df["cardinal_vram_gib"].max())
    for col_idx, prompt_id in enumerate(PROMPT_ORDER):
        sub = pair_df[pair_df["gsm8k_index"] == prompt_id].sort_values(
            ["batch", "chunk"]
        )
        if sub.empty:
            continue
        x = np.arange(len(sub))
        width = 0.34
        axes[0, col_idx].bar(
            x - width / 2,
            sub["ascend_runtime_min"],
            width=width,
            color=CLUSTER_COLORS["ascend"],
            label="ascend",
        )
        axes[0, col_idx].bar(
            x + width / 2,
            sub["cardinal_runtime_min"],
            width=width,
            color=CLUSTER_COLORS["cardinal"],
            label="cardinal",
        )
        axes[1, col_idx].bar(
            x - width / 2,
            sub["ascend_vram_gib"],
            width=width,
            color=CLUSTER_COLORS["ascend"],
            label="ascend",
        )
        axes[1, col_idx].bar(
            x + width / 2,
            sub["cardinal_vram_gib"],
            width=width,
            color=CLUSTER_COLORS["cardinal"],
            label="cardinal",
        )

        for row_idx in range(2):
            axes[row_idx, col_idx].set_xticks(x)
            axes[row_idx, col_idx].set_xticklabels(sub["config_label"], rotation=0)
            axes[row_idx, col_idx].grid(axis="y", alpha=0.25)
            axes[row_idx, col_idx].set_title(f"prompt {prompt_id}")

        axes[0, col_idx].set_ylabel("Runtime (min)")
        axes[1, col_idx].set_ylabel("Peak VRAM reserved (GiB)")
        axes[0, col_idx].set_ylim(bottom=0, top=float(runtime_max) * 1.12)
        axes[1, col_idx].set_ylim(bottom=0, top=float(vram_max) * 1.12)
    handles = [
        Line2D([0], [0], color=CLUSTER_COLORS[c], linewidth=8, label=c)
        for c in CLUSTER_ORDER
    ]
    fig.legend(
        handles,
        [str(h.get_label()) for h in handles],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Wave-1 cluster comparison on matched successful configs", fontsize=14)
    _save(fig, out_dir, "cluster_shared_config_comparison")


def plot_cardinal_headroom_gain(bench: pd.DataFrame, out_dir: Path) -> None:
    data = bench[
        (bench["cluster"] == "cardinal")
        & (bench["stage"].astype(str).str.contains("wave1"))
        & (bench["status"] == "success")
        & (~bench["is_special_case"])
    ].copy()
    ascend_keys = {
        (
            int(r["gsm8k_index"]),
            int(r["attribution_batch_size"]),
            int(r["decoder_chunk_size"]),
        )
        for _, r in bench[
            (bench["cluster"] == "ascend")
            & (bench["stage"].astype(str).str.contains("wave1"))
            & (~bench["is_special_case"])
        ]
        .dropna(subset=["gsm8k_index", "attribution_batch_size", "decoder_chunk_size"])
        .iterrows()
    }

    rows = []
    for prompt_id in PROMPT_ORDER:
        prompt_sub = data[data["gsm8k_index"] == prompt_id].copy()
        if prompt_sub.empty:
            continue
        prompt_sub["shared_with_ascend"] = prompt_sub.apply(
            lambda r: (
                (
                    int(r["gsm8k_index"]),
                    int(r["attribution_batch_size"]),
                    int(r["decoder_chunk_size"]),
                )
                in ascend_keys
            ),
            axis=1,
        )
        shared = prompt_sub[prompt_sub["shared_with_ascend"]]
        unique = prompt_sub[~prompt_sub["shared_with_ascend"]]
        if shared.empty or unique.empty:
            continue
        best_shared = shared.loc[shared["duration_seconds"].idxmin()]
        best_unique = unique.loc[unique["duration_seconds"].idxmin()]
        rows.append(
            {
                "prompt_id": prompt_id,
                "shared_runtime_min": best_shared["duration_seconds"] / 60.0,
                "unique_runtime_min": best_unique["duration_seconds"] / 60.0,
                "shared_label": f"b{int(best_shared['attribution_batch_size'])}/c{int(best_shared['decoder_chunk_size'])}",
                "unique_label": f"b{int(best_unique['attribution_batch_size'])}/c{int(best_unique['decoder_chunk_size'])}",
                "shared_vram": best_shared["resource_snapshot_cuda_peak_reserved_gib"],
                "unique_vram": best_unique["resource_snapshot_cuda_peak_reserved_gib"],
            }
        )
    headroom = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(headroom))
    width = 0.34
    ax.bar(
        x - width / 2,
        headroom["shared_runtime_min"],
        width=width,
        color="#1f77b4",
        label="Best shared config",
    )
    ax.bar(
        x + width / 2,
        headroom["unique_runtime_min"],
        width=width,
        color="#ff7f0e",
        label="Best Cardinal-only config",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"prompt {int(p)}" for p in headroom["prompt_id"]])
    ax.set_ylabel("Runtime (min)")
    ax.set_title("Cardinal headroom gain from configs unavailable on Ascend")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for xpos, (_, row) in enumerate(headroom.iterrows()):
        gain = 100.0 * (1.0 - row["unique_runtime_min"] / row["shared_runtime_min"])
        ax.text(
            float(xpos),
            max(row["shared_runtime_min"], row["unique_runtime_min"]) + 0.8,
            f"{gain:.1f}% faster\n{row['shared_label']} → {row['unique_label']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _save(fig, out_dir, "cardinal_headroom_gain")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report-ready graphs for circuit_tracer optimization benchmarks"
    )
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    bench, feat = load_tables(args.benchmark_csv, args.feature_csv)
    plot_phase0_feature_distribution_scaling(feat, args.output_dir)
    plot_exact_feature_scaling(bench, args.output_dir)
    plot_wave1_config_figures(bench, args.output_dir)
    plot_wave2_cache_sweeps(bench, args.output_dir)
    plot_cluster_shared_config_comparison(bench, args.output_dir)
    plot_cardinal_headroom_gain(bench, args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
