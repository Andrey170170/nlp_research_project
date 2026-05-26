from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("experiments/figures/presentation")


def build_early_memory_table() -> pd.DataFrame:
    """Historical early memory/offload comparison.

    These rows come from the optimization extract table and are intentionally
    treated as a narrative systems comparison, not a strict microbenchmark.
    All rows are Ascend, b128/c2048-era exact runs.
    """

    return pd.DataFrame(
        [
            {
                "prompt": "828_base",
                "condition": "Historical baseline",
                "rss_gib": 189.48,
                "cuda_peak_gib": 9.07,
                "duration_s": 3355.86,
                "source": "unnamed historical b128/c2048 row",
            },
            {
                "prompt": "828_base",
                "condition": "Lazy/offload smoke",
                "rss_gib": 181.41,
                "cuda_peak_gib": 8.08,
                "duration_s": 3481.12,
                "source": "lazy encoder smoke ascend",
            },
            {
                "prompt": "828_base",
                "condition": "Row-store / offload",
                "rss_gib": 137.11,
                "cuda_peak_gib": 8.08,
                "duration_s": 3415.99,
                "source": "memmap row-store fast ascend",
            },
            {
                "prompt": "361_base",
                "condition": "Historical baseline",
                "rss_gib": 329.18,
                "cuda_peak_gib": 14.35,
                "duration_s": 5085.54,
                "source": "unnamed historical b128/c2048 row",
            },
            {
                "prompt": "361_base",
                "condition": "Lazy/offload smoke",
                "rss_gib": 313.55,
                "cuda_peak_gib": 12.53,
                "duration_s": 4902.77,
                "source": "lazy encoder smoke ascend",
            },
            {
                "prompt": "361_base",
                "condition": "Row-store / offload",
                "rss_gib": 236.07,
                "cuda_peak_gib": 12.53,
                "duration_s": 5403.03,
                "source": "memmap row-store fast ascend",
            },
        ]
    )


def build_postfix_phase4_table() -> pd.DataFrame:
    """Post-overflow-fix Phase 4 memory/runtime comparison.

    All rows are Ascend fp32 b256/c4096/cache0 single-step runs on the
    optimization track. Values are from EXPERIMENTS.md / extracted benchmark
    rows. Use for directionality; some rows include debug/telemetry changes.
    """

    return pd.DataFrame(
        [
            {
                "prompt": "828_base",
                "condition": "Post-fix baseline",
                "phase4_s": 542.90,
                "completion_s": 808.97,
                "rss_gib": 259.01,
                "cuda_peak_gib": 22.60,
                "source": "overflow fix fast ascend fp32",
            },
            {
                "prompt": "828_base",
                "condition": "RSS cleanup",
                "phase4_s": 432.78,
                "completion_s": 684.25,
                "rss_gib": 133.30,
                "cuda_peak_gib": 22.60,
                "source": "rss validation fast ascend fp32",
            },
            {
                "prompt": "828_base",
                "condition": "Hybrid row-store",
                "phase4_s": 434.24,
                "completion_s": 686.40,
                "rss_gib": 133.29,
                "cuda_peak_gib": 22.60,
                "source": "phase4 hybrid rowstore rerun fp32",
            },
            {
                "prompt": "828_base",
                "condition": "Locality v2",
                "phase4_s": 392.82,
                "completion_s": 646.57,
                "rss_gib": 141.99,
                "cuda_peak_gib": 22.60,
                "source": "phase4 locality validation v2 fp32",
            },
            {
                "prompt": "828_base",
                "condition": "Planner V1",
                "phase4_s": 314.78,
                "completion_s": 563.91,
                "rss_gib": 142.05,
                "cuda_peak_gib": 22.60,
                "source": "phase4 planner v1 fp32",
            },
            {
                "prompt": "361_base",
                "condition": "Post-fix baseline",
                "phase4_s": 1722.67,
                "completion_s": 2130.47,
                "rss_gib": 333.97,
                "cuda_peak_gib": 29.20,
                "source": "overflow fix fast ascend fp32",
            },
            {
                "prompt": "361_base",
                "condition": "RSS cleanup",
                "phase4_s": 609.18,
                "completion_s": 972.95,
                "rss_gib": 229.07,
                "cuda_peak_gib": 29.20,
                "source": "rss validation fast ascend fp32",
            },
            {
                "prompt": "361_base",
                "condition": "Hybrid row-store",
                "phase4_s": 674.39,
                "completion_s": 1365.45,
                "rss_gib": 229.08,
                "cuda_peak_gib": 29.20,
                "source": "phase4 hybrid rowstore rerun fp32",
            },
            {
                "prompt": "361_base",
                "condition": "Locality v2",
                "phase4_s": 587.04,
                "completion_s": 948.87,
                "rss_gib": 241.62,
                "cuda_peak_gib": 29.20,
                "source": "phase4 locality validation v2 fp32",
            },
            {
                "prompt": "361_base",
                "condition": "Planner V1",
                "phase4_s": 580.01,
                "completion_s": 957.78,
                "rss_gib": 241.58,
                "cuda_peak_gib": 29.20,
                "source": "phase4 planner v1 fp32",
            },
        ]
    )


def _bar_labels(ax: plt.Axes, bars, fmt: str = "{:.0f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(1.0, height * 0.015),
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )


def plot_early_memory(df: pd.DataFrame) -> None:
    prompt_order = ["828_base", "361_base"]
    condition_order = [
        "Historical baseline",
        "Lazy/offload smoke",
        "Row-store / offload",
    ]
    colors = {
        "Historical baseline": "#7f7f7f",
        "Lazy/offload smoke": "#1f77b4",
        "Row-store / offload": "#2ca02c",
    }

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    x = np.arange(len(prompt_order))
    width = 0.25

    for idx, cond in enumerate(condition_order):
        sub = df[df["condition"] == cond].set_index("prompt").loc[prompt_order]
        offset = (idx - 1) * width
        bars0 = axes[0].bar(
            x + offset, sub["rss_gib"], width=width, color=colors[cond], label=cond
        )
        bars1 = axes[1].bar(
            x + offset,
            sub["cuda_peak_gib"],
            width=width,
            color=colors[cond],
            label=cond,
        )
        _bar_labels(axes[0], bars0, "{:.0f}")
        _bar_labels(axes[1], bars1, "{:.1f}")

    axes[0].set_title("Host RAM: major early win")
    axes[0].set_ylabel("Process RSS snapshot (GiB)")
    axes[1].set_title("GPU peak: only modest early reduction")
    axes[1].set_ylabel("CUDA peak allocated (GiB)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_order)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Early memory work mostly reduced host RAM, not the core VRAM ceiling", y=1.02
    )
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "memory_early_offload_comparison.png", dpi=220, bbox_inches="tight"
    )
    fig.savefig(OUT_DIR / "memory_early_offload_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def plot_postfix_phase4(df: pd.DataFrame) -> None:
    prompt_order = ["828_base", "361_base"]
    condition_order = [
        "Post-fix baseline",
        "RSS cleanup",
        "Hybrid row-store",
        "Locality v2",
        "Planner V1",
    ]
    colors = {
        "Post-fix baseline": "#d62728",
        "RSS cleanup": "#ff7f0e",
        "Hybrid row-store": "#9467bd",
        "Locality v2": "#1f77b4",
        "Planner V1": "#2ca02c",
    }

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharex=True)
    x = np.arange(len(prompt_order))
    width = 0.15

    for idx, cond in enumerate(condition_order):
        sub = df[df["condition"] == cond].set_index("prompt").loc[prompt_order]
        offset = (idx - 2) * width
        bars0 = axes[0].bar(
            x + offset,
            sub["phase4_s"] / 60.0,
            width=width,
            color=colors[cond],
            label=cond,
        )
        bars1 = axes[1].bar(
            x + offset, sub["rss_gib"], width=width, color=colors[cond], label=cond
        )
        _bar_labels(axes[0], bars0, "{:.1f}")
        _bar_labels(axes[1], bars1, "{:.0f}")

    axes[0].set_title("Phase 4 wall time")
    axes[0].set_ylabel("Minutes")
    axes[1].set_title("Host RSS snapshot")
    axes[1].set_ylabel("GiB")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_order)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "After overflow fix: memory cleanup stabilized RAM; Phase 4 remains the runtime target",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "memory_postfix_phase4_progress.png", dpi=220, bbox_inches="tight"
    )
    fig.savefig(OUT_DIR / "memory_postfix_phase4_progress.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    early = build_early_memory_table()
    postfix = build_postfix_phase4_table()
    early.to_csv(OUT_DIR / "memory_early_offload_comparison_data.csv", index=False)
    postfix.to_csv(OUT_DIR / "memory_postfix_phase4_progress_data.csv", index=False)
    plot_early_memory(early)
    plot_postfix_phase4(postfix)
    print(f"Wrote figures and data to {OUT_DIR}")


if __name__ == "__main__":
    main()
