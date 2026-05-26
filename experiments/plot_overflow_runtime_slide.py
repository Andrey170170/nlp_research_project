from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("experiments/figures/presentation")
SCRATCH = Path("/fs/scratch/PAS3272/kopanev.1/exact_trace_bench/ascend")


RUNS = [
    # Pre-fix true fp32 / raw row-L1 path.
    {
        "prompt": "94_base",
        "condition": "Pre-fix fp32",
        "path": SCRATCH
        / "anomaly/20260421_124700_true-fp32-norm-debug-ascend/ascend_anomaly_94_base_b256_c4096_cache0g/artifacts/prompt_000/completion_000/completion.json",
        "notes": "phase4_anomaly_debug=true; b256/c4096",
    },
    {
        "prompt": "828_base",
        "condition": "Pre-fix fp32",
        "path": SCRATCH
        / "fast/20260421_124700_true-fp32-norm-debug-ascend/ascend_fast_828_base_b128_c2048_cache0g/artifacts/prompt_000/completion_000/completion.json",
        "notes": "phase4_anomaly_debug=true; b128/c2048",
    },
    {
        "prompt": "361_base",
        "condition": "Pre-fix fp32",
        "path": SCRATCH
        / "fast/20260421_124700_true-fp32-norm-361-baseline-ascend/ascend_fast_361_base_b128_c2048_cache0g/artifacts/prompt_000/completion_000/completion.json",
        "notes": "non-debug baseline; b128/c2048",
    },
    # fp64 workaround / raw row-L1 but higher precision.
    {
        "prompt": "94_base",
        "condition": "fp64 workaround",
        "path": SCRATCH
        / "anomaly/20260421_124700_true-fp64-norm-debug-ascend/ascend_anomaly_94_base_b256_c4096_cache0g/artifacts/prompt_000/completion_000/completion.json",
        "notes": "phase4_anomaly_debug=true; b256/c4096",
    },
    {
        "prompt": "828_base",
        "condition": "fp64 workaround",
        "path": SCRATCH
        / "fast/20260421_124700_true-fp64-norm-debug-ascend/ascend_fast_828_base_b128_c2048_cache0g/artifacts/prompt_000/completion_000/completion.json",
        "notes": "phase4_anomaly_debug=true; b128/c2048",
    },
    {
        "prompt": "361_base",
        "condition": "fp64 workaround",
        "path": SCRATCH
        / "fast/20260421_124700_true-fp64-norm-361-baseline-ascend/ascend_fast_361_base_b128_c2048_cache0g/artifacts/prompt_000/completion_000/completion.json",
        "notes": "non-debug baseline; b128/c2048",
    },
    # Permanent overflow fix, fp32 default.
    {
        "prompt": "94_base",
        "condition": "Permanent fix fp32",
        "path": SCRATCH
        / "anomaly/ascend_anomaly_94_base_overflow_fix_fp32_b256_c4096_cache0/artifacts/prompt_000/completion_000/completion.json",
        "notes": "stable denominator; b256/c4096",
    },
    {
        "prompt": "828_base",
        "condition": "Permanent fix fp32",
        "path": SCRATCH
        / "fast/ascend_fast_828_base_overflow_fix_fp32_b256_c4096_cache0/artifacts/prompt_000/completion_000/completion.json",
        "notes": "stable denominator; b256/c4096",
    },
    {
        "prompt": "361_base",
        "condition": "Permanent fix fp32",
        "path": SCRATCH
        / "fast/ascend_fast_361_base_overflow_fix_fp32_b256_c4096_cache0/artifacts/prompt_000/completion_000/completion.json",
        "notes": "stable denominator; b256/c4096",
    },
]

POST_FIX_MATCHED_RUNS = [
    {
        "prompt": "94_base",
        "condition": "Permanent fix fp32",
        "path": SCRATCH
        / "anomaly/ascend_anomaly_94_base_overflow_fix_fp32_b256_c4096_cache0/artifacts/prompt_000/completion_000/completion.json",
    },
    {
        "prompt": "94_base",
        "condition": "Permanent fix fp64",
        "path": SCRATCH
        / "anomaly/ascend_anomaly_94_base_overflow_fix_fp64_b256_c4096_cache0/artifacts/prompt_000/completion_000/completion.json",
    },
    {
        "prompt": "828_base",
        "condition": "Permanent fix fp32",
        "path": SCRATCH
        / "fast/ascend_fast_828_base_overflow_fix_fp32_b256_c4096_cache0/artifacts/prompt_000/completion_000/completion.json",
    },
    {
        "prompt": "828_base",
        "condition": "Permanent fix fp64",
        "path": SCRATCH
        / "fast/ascend_fast_828_base_overflow_fix_fp64_b256_c4096_cache0/artifacts/prompt_000/completion_000/completion.json",
    },
    {
        "prompt": "361_base",
        "condition": "Permanent fix fp32",
        "path": SCRATCH
        / "fast/ascend_fast_361_base_overflow_fix_fp32_b256_c4096_cache0/artifacts/prompt_000/completion_000/completion.json",
    },
    {
        "prompt": "361_base",
        "condition": "Permanent fix fp64",
        "path": SCRATCH
        / "fast/ascend_fast_361_base_overflow_fix_fp64_b256_c4096_cache0/artifacts/prompt_000/completion_000/completion.json",
    },
]


def _phase4_seconds(payload: dict) -> float:
    timing = payload.get("timing_summary", {})
    wall = timing.get("attribution_phase_wall_clock_elapsed_seconds_total", {})
    if "phase4" in wall:
        return float(wall["phase4"])
    steps = payload.get("steps", [])
    if steps:
        step_wall = steps[0].get("attribution_phase_wall_clock_elapsed_seconds", {})
        if "phase4" in step_wall:
            return float(step_wall["phase4"])
    return float("nan")


def _completion_seconds(payload: dict) -> float:
    timing = payload.get("timing_summary", {})
    if "completion_end_to_end_seconds" in timing:
        return float(timing["completion_end_to_end_seconds"])
    return float(payload.get("duration_seconds", "nan"))


def _first_step(payload: dict) -> dict:
    steps = payload.get("steps", [])
    return steps[0] if steps else {}


def build_table(run_specs: list[dict[str, object]] = RUNS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in run_specs:
        path = Path(str(spec["path"]))
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        step = _first_step(payload)
        diag = step.get("transcoder_diagnostics", {}) if isinstance(step, dict) else {}
        resource = payload.get("resource_snapshot", {})
        rows.append(
            {
                "prompt": spec["prompt"],
                "condition": spec["condition"],
                "completion_seconds": _completion_seconds(payload),
                "phase4_seconds": _phase4_seconds(payload),
                "edges_retained": step.get("n_edges_retained"),
                "active_features": step.get("n_active_features"),
                "decoder_load_count": diag.get("decoder_load_count"),
                "rss_gib": resource.get("rss_gib"),
                "exact_trace_internal_dtype": payload.get("exact_trace_internal_dtype"),
                "phase4_anomaly_debug": payload.get("phase4_anomaly_debug"),
                "attribution_batch_size": payload.get("attribution_batch_size"),
                "decoder_chunk_size": payload.get("decoder_chunk_size"),
                "notes": spec.get("notes", ""),
                "source_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def plot_runtime(df: pd.DataFrame) -> None:
    prompt_order = ["94_base", "828_base", "361_base"]
    condition_order = ["Pre-fix fp32", "fp64 workaround", "Permanent fix fp32"]
    colors = {
        "Pre-fix fp32": "#d62728",
        "fp64 workaround": "#9467bd",
        "Permanent fix fp32": "#2ca02c",
    }

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.titlesize": 16,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), sharex=True)

    x = np.arange(len(prompt_order))
    width = 0.24
    for cond_idx, cond in enumerate(condition_order):
        sub = df[df["condition"] == cond].set_index("prompt").loc[prompt_order]
        offset = (cond_idx - 1) * width
        axes[0].bar(
            x + offset,
            sub["completion_seconds"] / 60.0,
            width=width,
            label=cond,
            color=colors[cond],
        )
        bars = axes[1].bar(
            x + offset,
            sub["phase4_seconds"] / 60.0,
            width=width,
            label=cond,
            color=colors[cond],
        )
        for bar, edges in zip(bars, sub["edges_retained"].tolist(), strict=False):
            if pd.notna(edges):
                label = "8k" if int(edges) == 8192 else "20k"
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.7,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )

    axes[0].set_title("End-to-end completion time")
    axes[0].set_ylabel("Minutes")
    axes[1].set_title("Phase 4 wall-clock time")
    axes[1].set_ylabel("Minutes")
    axes[1].text(
        0.99,
        0.98,
        "edge labels: retained edges\n8k = collapsed/truncated\n20k = full configured cap",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#dddddd",
        },
    )

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_order)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Overflow investigation: fp32 bug → fp64 workaround → permanent fp32 fix"
    )
    fig.text(
        0.5,
        0.01,
        "Historical comparison; configs/debug flags differ across some runs. Use for narrative direction, not strict microbenchmark speedup.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_DIR / "overflow_runtime_historical.png", dpi=240, bbox_inches="tight"
    )
    fig.savefig(OUT_DIR / "overflow_runtime_historical.svg", bbox_inches="tight")
    plt.close(fig)


def plot_decoder_loads(df: pd.DataFrame) -> None:
    prompt_order = ["94_base", "828_base", "361_base"]
    condition_order = ["Pre-fix fp32", "fp64 workaround", "Permanent fix fp32"]
    colors = {
        "Pre-fix fp32": "#d62728",
        "fp64 workaround": "#9467bd",
        "Permanent fix fp32": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(prompt_order))
    width = 0.24
    for cond_idx, cond in enumerate(condition_order):
        sub = df[df["condition"] == cond].set_index("prompt").loc[prompt_order]
        offset = (cond_idx - 1) * width
        ax.bar(
            x + offset,
            sub["decoder_load_count"],
            width=width,
            label=cond,
            color=colors[cond],
        )
    ax.set_title("Decoder replay/load count during overflow investigation")
    ax.set_ylabel("Decoder load count")
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_order)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_DIR / "overflow_decoder_loads_historical.png", dpi=240, bbox_inches="tight"
    )
    fig.savefig(OUT_DIR / "overflow_decoder_loads_historical.svg", bbox_inches="tight")
    plt.close(fig)


def plot_post_fix_matched(df: pd.DataFrame) -> None:
    prompt_order = ["94_base", "828_base", "361_base"]
    condition_order = ["Permanent fix fp32", "Permanent fix fp64"]
    colors = {"Permanent fix fp32": "#2ca02c", "Permanent fix fp64": "#9467bd"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    x = np.arange(len(prompt_order))
    width = 0.34
    for cond_idx, cond in enumerate(condition_order):
        sub = df[df["condition"] == cond].set_index("prompt").loc[prompt_order]
        offset = (cond_idx - 0.5) * width
        axes[0].bar(
            x + offset,
            sub["completion_seconds"] / 60.0,
            width=width,
            color=colors[cond],
            label=cond,
        )
        axes[1].bar(
            x + offset,
            sub["phase4_seconds"] / 60.0,
            width=width,
            color=colors[cond],
            label=cond,
        )
    axes[0].set_title("End-to-end completion time")
    axes[1].set_title("Phase 4 wall-clock time")
    for ax in axes:
        ax.set_ylabel("Minutes")
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_order)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Post-fix matched validation: fp32 keeps fp64 outputs, runs faster")
    fig.text(
        0.5,
        0.01,
        "All bars use the permanent overflow-fix code path with b256/c4096/cache0; compact fp32/fp64 artifacts matched exactly in validation.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(
        OUT_DIR / "overflow_postfix_fp32_fp64_matched.png", dpi=240, bbox_inches="tight"
    )
    fig.savefig(OUT_DIR / "overflow_postfix_fp32_fp64_matched.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build_table()
    post_df = build_table(POST_FIX_MATCHED_RUNS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "overflow_runtime_historical_data.csv", index=False)
    post_df.to_csv(OUT_DIR / "overflow_postfix_fp32_fp64_matched_data.csv", index=False)
    plot_runtime(df)
    plot_decoder_loads(df)
    plot_post_fix_matched(post_df)
    print(df.to_string(index=False))
    print(f"wrote={OUT_DIR / 'overflow_runtime_historical.png'}")
    print(f"wrote={OUT_DIR / 'overflow_decoder_loads_historical.png'}")
    print(f"wrote={OUT_DIR / 'overflow_postfix_fp32_fp64_matched.png'}")
    print(f"wrote={OUT_DIR / 'overflow_runtime_historical_data.csv'}")
    print(f"wrote={OUT_DIR / 'overflow_postfix_fp32_fp64_matched_data.csv'}")


if __name__ == "__main__":
    main()
