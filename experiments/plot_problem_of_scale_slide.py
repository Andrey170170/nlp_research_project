from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_CSV = Path(
    "experiments/extracted/feature_distribution_analysis/feature_distribution_prompts.csv"
)
BENCHMARK_CSV = Path(
    "experiments/extracted/weekend_exact_chunked/benchmark_enriched.csv"
)
OUT_DIR = Path("experiments/figures/presentation")


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    slope, intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]
    return float(slope), float(intercept), float(corr)


def main() -> None:
    feat = pd.read_csv(FEATURE_CSV)
    bench = pd.read_csv(BENCHMARK_CSV)

    feat = feat[feat["status"].eq("success")].copy()
    for col in ["token_count", "total_active_features"]:
        feat[col] = pd.to_numeric(feat[col], errors="coerce")

    for col in [
        "max_active_features",
        "resource_snapshot_rss_gib",
        "duration_seconds",
        "is_special_case",
    ]:
        if col in bench.columns:
            if col == "is_special_case":
                bench[col] = bench[col].astype(str).str.lower().eq("true")
            else:
                bench[col] = pd.to_numeric(bench[col], errors="coerce")
    bench = bench[(bench["status"].eq("success")) & (~bench["is_special_case"])].copy()
    bench["active_features_m"] = bench["max_active_features"] / 1_000_000.0

    feature_slope, feature_intercept, feature_corr = _fit(
        feat["token_count"].to_numpy(float),
        feat["total_active_features"].to_numpy(float),
    )
    rss_slope, rss_intercept, rss_corr = _fit(
        bench["active_features_m"].to_numpy(float),
        bench["resource_snapshot_rss_gib"].to_numpy(float),
    )

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.titlesize": 18,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # Panel A: prompt length -> active feature count.
    ax = axes[0]
    x = feat["token_count"].to_numpy(float)
    y_m = feat["total_active_features"].to_numpy(float) / 1_000_000.0
    ax.scatter(x, y_m, s=24, alpha=0.62, color="#1f77b4", edgecolor="none")
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ys_m = (feature_slope * xs + feature_intercept) / 1_000_000.0
    ax.plot(xs, ys_m, color="#d62728", linewidth=2.4)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Active sparse features (millions)")
    ax.set_title("Prompt length almost linearly sets trace size")
    ax.grid(alpha=0.25)
    ax.text(
        0.04,
        0.96,
        f"360 GSM8K prompts\n$r={feature_corr:.3f}$\n≈ {feature_slope / 1000:.1f}k features/token",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#cccccc",
        },
    )

    # Panel B: active feature count -> host RAM.
    ax = axes[1]
    colors = {"ascend": "#1f77b4", "cardinal": "#ff7f0e"}
    for cluster, sub in bench.groupby("cluster"):
        ax.scatter(
            sub["active_features_m"],
            sub["resource_snapshot_rss_gib"],
            s=46,
            alpha=0.78,
            color=colors.get(cluster, "#777777"),
            label=cluster,
            edgecolor="white",
            linewidth=0.5,
        )
    x2 = bench["active_features_m"].to_numpy(float)
    xs = np.linspace(np.nanmin(x2), np.nanmax(x2), 200)
    ys = rss_slope * xs + rss_intercept
    ax.plot(xs, ys, color="black", linestyle="--", linewidth=2.0)
    ax.set_xlabel("Active sparse features (millions)")
    ax.set_ylabel("Final host RSS (GiB)")
    ax.set_title("Every extra million features costs tens of GiB RAM")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.04,
        0.78,
        f"successful exact runs\n$r={rss_corr:.3f}$\n≈ {rss_slope:.1f} GiB / 1M features",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#cccccc",
        },
    )

    fig.suptitle(
        "Why exact attribution tracing does not scale like a normal forward pass"
    )
    fig.text(
        0.5,
        0.005,
        "More prompt tokens → more active sparse features → larger exact graph state → higher RAM/VRAM pressure",
        ha="center",
        fontsize=12,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "problem_of_scale_summary.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / "problem_of_scale_summary.svg", bbox_inches="tight")

    print(f"feature_slope_per_token={feature_slope:.3f}")
    print(f"feature_corr={feature_corr:.6f}")
    print(f"rss_slope_gib_per_million_features={rss_slope:.3f}")
    print(f"rss_corr={rss_corr:.6f}")
    print(f"wrote={OUT_DIR / 'problem_of_scale_summary.png'}")


if __name__ == "__main__":
    main()
