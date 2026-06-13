from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..io_utils import ensure_dir, write_json


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _finite_metric_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out[out["value"].notna()].copy()


def _selected_scorecard(scorecard: pd.DataFrame, *, limit: int = 8) -> pd.DataFrame:
    if scorecard.empty:
        return scorecard
    out = scorecard.copy()
    out["passes_calibration"] = (
        out["passes_calibration"].astype(str).str.lower() == "true"
    )
    out["abs_separation_delta"] = (
        pd.to_numeric(out["separation_score"], errors="coerce") - 0.5
    ).abs()
    out = out.sort_values(
        ["passes_calibration", "abs_separation_delta", "bucket", "metric"],
        ascending=[False, True, True, True],
    )
    return out.head(limit)


def _plot_distributions(
    metric_rows: pd.DataFrame, scorecard: pd.DataFrame, output_dir: Path
) -> Path | None:
    selected = _selected_scorecard(scorecard, limit=6)
    if selected.empty:
        return None
    fig, axes = plt.subplots(
        len(selected), 1, figsize=(10, max(3, 2.2 * len(selected)))
    )
    if len(selected) == 1:
        axes = [axes]
    for ax, (_idx, card) in zip(axes, selected.iterrows()):
        mask = (
            (metric_rows["bucket"] == card["bucket"])
            & (metric_rows["metric"] == card["metric"])
            & (metric_rows["params_json"] == card["params_json"])
            & (metric_rows["position_band"].fillna("unknown") == card["position_band"])
        )
        subset = metric_rows[mask]
        categories = ["null", "temporal", "noise"]
        values = [
            subset.loc[subset["pair_category"] == cat, "value"].dropna()
            for cat in categories
        ]
        ax.boxplot(values, tick_labels=categories, showmeans=True)
        ax.set_title(
            f"{card['bucket']} / {card['metric']} / {card['params_json']} / {card['position_band']}"
        )
        ax.set_ylabel("value")
        ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, output_dir / "metric_calibration_distributions.png")


def _plot_derived_k_curves(metric_rows: pd.DataFrame, output_dir: Path) -> Path | None:
    subset = metric_rows[metric_rows["metric"] == "derived_k_weighted_jaccard"].copy()
    if subset.empty:
        return None
    subset["K"] = subset["params_json"].str.extract(r'"K":(\d+)').astype(float)
    subset = subset[subset["K"].notna()]
    if subset.empty:
        return None
    buckets = list(subset["bucket"].drop_duplicates().head(4))
    fig, axes = plt.subplots(len(buckets), 1, figsize=(10, max(3, 3 * len(buckets))))
    if len(buckets) == 1:
        axes = [axes]
    for ax, bucket in zip(axes, buckets):
        by_bucket = subset[subset["bucket"] == bucket]
        grouped = (
            by_bucket.groupby(["pair_category", "K"], dropna=False)["value"]
            .mean()
            .reset_index()
        )
        for category, group in grouped.groupby("pair_category"):
            group = group.sort_values("K")
            ax.plot(group["K"], group["value"], marker="o", label=str(category))
        ax.set_xscale("log", base=2)
        ax.set_title(f"Derived-K weighted Jaccard: {bucket}")
        ax.set_xlabel("K")
        ax.set_ylabel("mean value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
    return _save(fig, output_dir / "derived_k_weighted_jaccard_curves.png")


def _plot_lag_decay(
    metric_rows: pd.DataFrame, scorecard: pd.DataFrame, output_dir: Path
) -> Path | None:
    selected = _selected_scorecard(scorecard, limit=6)
    temporal = metric_rows[
        (metric_rows["pair_category"] == "temporal") & metric_rows["lag"].notna()
    ].copy()
    if selected.empty or temporal.empty:
        return None
    temporal["lag"] = pd.to_numeric(temporal["lag"], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for _idx, card in selected.iterrows():
        mask = (
            (temporal["bucket"] == card["bucket"])
            & (temporal["metric"] == card["metric"])
            & (temporal["params_json"] == card["params_json"])
        )
        subset = temporal[mask]
        if subset.empty:
            continue
        grouped = subset.groupby("lag")["value"].mean().reset_index().sort_values("lag")
        ax.plot(
            grouped["lag"],
            grouped["value"],
            marker="o",
            label=f"{card['bucket']}:{card['metric']}:{card['params_json']}",
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return None
    ax.set_title("Temporal lag-decay curves")
    ax.set_xlabel("lag")
    ax.set_ylabel("mean value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="x-small")
    return _save(fig, output_dir / "lag_decay_curves.png")


def _plot_scorecard_passes(scorecard: pd.DataFrame, output_dir: Path) -> Path | None:
    if scorecard.empty:
        return None
    data = scorecard.copy()
    data["passes_calibration"] = (
        data["passes_calibration"].astype(str).str.lower() == "true"
    )
    grouped = (
        data.groupby("bucket")["passes_calibration"].sum().sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped.plot(kind="bar", ax=ax)
    ax.set_title("Passing scorecard rows by bucket")
    ax.set_xlabel("bucket")
    ax.set_ylabel("pass count")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, output_dir / "scorecard_pass_counts.png")


def write_metric_calibration_report(*, analysis_dir: Path, output_path: Path) -> Path:
    scorecard = _read_csv(analysis_dir / "scorecard.csv")
    selected = _selected_scorecard(scorecard, limit=20)
    pass_count = int(
        (scorecard["passes_calibration"].astype(str).str.lower() == "true").sum()
    )
    lines = [
        "# Metric calibration findings template",
        "",
        f"Analysis dir: `{analysis_dir}`",
        f"Scorecard rows: {len(scorecard)}",
        f"Passing rows: {pass_count}",
        "",
        "## Top scorecard rows",
        "",
        "| bucket | metric | params | band | pass | separation | null mean | temporal mean | noise mean |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _idx, row in selected.iterrows():
        lines.append(
            "| {bucket} | {metric} | `{params}` | {band} | {passes} | {sep} | {null} | {temporal} | {noise} |".format(
                bucket=row.get("bucket", ""),
                metric=row.get("metric", ""),
                params=row.get("params_json", "{}"),
                band=row.get("position_band", ""),
                passes=row.get("passes_calibration", ""),
                sep=row.get("separation_score", ""),
                null=row.get("null_mean", ""),
                temporal=row.get("temporal_adjacent_mean", ""),
                noise=row.get("noise_mean", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Notes to fill after inspection",
            "",
            "- Surviving metric families:",
            "- Buckets without surviving metrics:",
            "- Position-band caveats:",
            "- Decision for EXPERIMENTS.md:",
        ]
    )
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def plot_metric_calibration(*, analysis_dir: Path, output_dir: Path) -> dict[str, Any]:
    ensure_dir(output_dir)
    metric_rows = _finite_metric_rows(_read_csv(analysis_dir / "metric_rows.csv"))
    scorecard = _read_csv(analysis_dir / "scorecard.csv")
    generated: list[str] = []
    for path in (
        _plot_distributions(metric_rows, scorecard, output_dir),
        _plot_derived_k_curves(metric_rows, output_dir),
        _plot_lag_decay(metric_rows, scorecard, output_dir),
        _plot_scorecard_passes(scorecard, output_dir),
    ):
        if path is not None:
            generated.append(str(path))
    report_path = write_metric_calibration_report(
        analysis_dir=analysis_dir,
        output_path=output_dir / "metric_calibration_findings.md",
    )
    generated.append(str(report_path))
    manifest = {
        "analysis_dir": str(analysis_dir),
        "output_dir": str(output_dir),
        "generated_files": generated,
    }
    write_json(output_dir / "plot_manifest.json", manifest)
    return manifest
