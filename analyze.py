"""
Batch analysis of traced completions with correct/incorrect comparison.

Reads compact .npz step files and evaluation.json per completion,
computes temporal metrics, and produces aggregate comparison plots.

Usage:
    python analyze.py --traces-dir /fs/scratch/PAS3272/kopanev.1/traces [--workers N]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

from circuit_utils import (
    STABLE_CORE_PERSISTENCE,
    STABLE_CORE_WINDOW,
    StepData,
    compute_temporal_metrics,
    load_compact,
    step_from_pt,
)

END_OF_TURN_ID = 106


def is_complete_completion(completion_dir: Path) -> bool:
    """Return True if a traced completion finished and has a manifest."""
    return (completion_dir / "completion.json").exists()


def load_completion_steps(
    completion_dir: Path,
    *,
    workers: int = 1,
    max_edges: int = 10_000,
) -> list[StepData]:
    """Load all steps for a completion, preferring .npz over .pt."""
    npz_files = sorted(completion_dir.glob("step_*.npz"))
    pt_files = sorted(completion_dir.glob("step_*.pt"))

    def _load_npz(path_idx: tuple[Path, int]) -> StepData:
        return load_compact(path_idx[0])

    def _load_pt(path_idx: tuple[Path, int]) -> StepData:
        return step_from_pt(path_idx[0], path_idx[1], max_edges=max_edges)

    if npz_files:
        paths = [(p, int(p.stem.split("_")[1])) for p in npz_files]
        loader = _load_npz
    elif pt_files:
        paths = [(p, int(p.stem.split("_")[1])) for p in pt_files]
        loader = _load_pt
    else:
        return []

    # Find meaningful end from manifest
    manifest_path = completion_dir / "completion.json"
    meaningful_end = len(paths)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for s in manifest.get("steps", []):
            if s.get("next_token_id") == END_OF_TURN_ID:
                meaningful_end = s["step_index"] + 1
                break

    paths = [(p, idx) for p, idx in paths if idx < meaningful_end]

    results: dict[int, StepData] = {}
    if workers <= 1:
        for p, idx in paths:
            results[idx] = loader((p, idx))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(loader, (p, idx)): idx for p, idx in paths}
            for fut in as_completed(futures):
                sd = fut.result()
                results[sd.step_idx] = sd

    return [results[k] for k in sorted(results.keys())]


def analyze_single_completion(
    completion_dir: Path,
    *,
    workers: int = 1,
    max_edges: int = 10_000,
) -> dict | None:
    """Compute temporal metrics for one completion. Returns summary dict."""
    manifest_path = completion_dir / "completion.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    steps = load_completion_steps(completion_dir, workers=workers, max_edges=max_edges)
    if len(steps) < 2:
        return None

    metrics = compute_temporal_metrics(steps)
    step_indices = [sd.step_idx for sd in steps]
    step_meta = {s.get("step_index"): s for s in manifest.get("steps", [])}

    meaningful_end = len(step_indices)
    for idx, sd in enumerate(steps):
        meta = step_meta.get(sd.step_idx, {})
        if meta.get("next_token_id") == END_OF_TURN_ID:
            meaningful_end = idx + 1
            break

    # Filter NaN
    valid_wj = [v for v in metrics["edge_wjaccard"] if not np.isnan(v)]
    valid_fj = [v for v in metrics["feature_jaccard"] if not np.isnan(v)]
    valid_cores = [v for v in metrics["core_sizes"] if not np.isnan(v)]
    valid_core_mass = [v for v in metrics["core_masses"] if not np.isnan(v)]

    return {
        "completion_dir": str(completion_dir),
        "completion_text": manifest.get("completion_text", ""),
        "meaningful_end": meaningful_end,
        "n_steps": len(steps),
        "mean_wjaccard": float(np.mean(valid_wj)) if valid_wj else None,
        "std_wjaccard": float(np.std(valid_wj)) if valid_wj else None,
        "mean_feature_jaccard": float(np.mean(valid_fj)) if valid_fj else None,
        "mean_core_size": float(np.mean(valid_cores)) if valid_cores else None,
        "mean_core_mass": float(np.mean(valid_core_mass)) if valid_core_mass else None,
        "mean_churn": float(1 - np.mean(valid_wj)) if valid_wj else None,
        "step_index": step_indices,
        "n_active_features": [sd.n_features for sd in steps],
        "n_edges_retained": [len(sd.weights) for sd in steps],
        "logprob": [sd.logprob for sd in steps],
        "token_text": [sd.token_text for sd in steps],
        "wjaccard_curve": metrics["edge_wjaccard"],
        "edge_jaccard_curve": metrics["edge_jaccard"],
        "feature_jaccard_curve": metrics["feature_jaccard"],
        "core_size_curve": metrics["core_sizes"],
        "core_mass_curve": metrics["core_masses"],
        "churn_curve": [1 - v for v in metrics["edge_wjaccard"]],
    }


def _json_ready(values: list[float | str | None]) -> list[float | str | None]:
    out: list[float | str | None] = []
    for v in values:
        if isinstance(v, float) and np.isnan(v):
            out.append(None)
        else:
            out.append(v)
    return out


def _write_completion_artifacts(completion_dir: Path, result: dict) -> None:
    """Write per-completion plots/metrics like explore_analysis.py."""
    out_dir = completion_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    step_indices = np.array(result["step_index"])
    wj = np.array(result["wjaccard_curve"], dtype=float)
    ej = np.array(result["edge_jaccard_curve"], dtype=float)
    fj = np.array(result["feature_jaccard_curve"], dtype=float)
    churn = np.array(result["churn_curve"], dtype=float)
    core = np.array(result["core_size_curve"], dtype=float)
    core_mass = np.array(result["core_mass_curve"], dtype=float)
    logprob = np.array(
        [np.nan if v is None else float(v) for v in result["logprob"]], dtype=float
    )
    meaningful_end = result["meaningful_end"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f"Temporal Circuit Analysis — {completion_dir.name}", fontsize=14)

    ax = axes[0, 0]
    if len(step_indices) > 1:
        ax.plot(step_indices[1:], wj[1:], label="Weighted Jaccard", alpha=0.8)
        ax.plot(step_indices[1:], ej[1:], label="Unweighted Jaccard", alpha=0.8)
        ax.plot(
            step_indices[1:],
            fj[1:],
            label="Feature Jaccard",
            alpha=0.6,
            linestyle="--",
        )
    ax.axvline(meaningful_end - 1, color="red", linestyle=":", label="<end_of_turn>")
    ax.set_xlabel("Step")
    ax.set_ylabel("Jaccard overlap")
    ax.set_title("Step-to-step overlap")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)

    ax = axes[0, 1]
    if len(step_indices) > 1:
        ax.plot(step_indices[1:], churn[1:], color="tomato", alpha=0.8)
    ax.axvline(meaningful_end - 1, color="red", linestyle=":", label="<end_of_turn>")
    ax.set_xlabel("Step")
    ax.set_ylabel("Churn (1 - wJacc)")
    ax.set_title("Edge churn")
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1, 0]
    ax.plot(
        step_indices, result["n_active_features"], label="Active features", alpha=0.8
    )
    ax.plot(
        step_indices,
        result["n_edges_retained"],
        label="Retained edges",
        alpha=0.8,
    )
    ax.axvline(meaningful_end - 1, color="red", linestyle=":", label="<end_of_turn>")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.set_title("Graph size over time")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(step_indices, core, color="teal", alpha=0.8)
    ax.axvline(meaningful_end - 1, color="red", linestyle=":", label="<end_of_turn>")
    ax.set_xlabel("Step")
    ax.set_ylabel("Core size (edges)")
    ax.set_title(f"Stable core (W={STABLE_CORE_WINDOW}, p={STABLE_CORE_PERSISTENCE})")

    ax = axes[2, 0]
    ax.plot(step_indices, core_mass, color="darkcyan", alpha=0.8)
    ax.axvline(meaningful_end - 1, color="red", linestyle=":", label="<end_of_turn>")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mass fraction")
    ax.set_title("Stable core mass fraction")
    ax.set_ylim(-0.02, 1.02)

    ax = axes[2, 1]
    valid = ~np.isnan(logprob)
    if valid.any():
        ax.plot(
            step_indices[valid],
            logprob[valid],
            color="purple",
            alpha=0.7,
            marker=".",
            markersize=3,
        )
    ax.axvline(meaningful_end - 1, color="red", linestyle=":", label="<end_of_turn>")
    ax.set_xlabel("Step")
    ax.set_ylabel("Log-probability")
    ax.set_title("Next-token log-probability")

    ax = axes[3, 0]
    valid_pair = (
        (~np.isnan(logprob[1:])) & (~np.isnan(wj[1:])) if len(step_indices) > 1 else []
    )
    if len(step_indices) > 1 and np.any(valid_pair):
        ax.scatter(logprob[1:][valid_pair], wj[1:][valid_pair], alpha=0.4, s=10)
    ax.set_xlabel("Log-probability")
    ax.set_ylabel("Weighted Jaccard")
    ax.set_title("Independence check: logprob vs overlap")

    axes[3, 1].axis("off")

    plt.tight_layout()
    plot_path = out_dir / "temporal_analysis.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)

    metrics_path = out_dir / "temporal_metrics.json"
    metrics = {
        "completion_dir": result["completion_dir"],
        "n_steps_analysed": result["n_steps"],
        "meaningful_end": result["meaningful_end"],
        "stable_core_window": STABLE_CORE_WINDOW,
        "stable_core_persistence": STABLE_CORE_PERSISTENCE,
        "summary": {
            "mean_weighted_jaccard": result["mean_wjaccard"],
            "std_weighted_jaccard": result["std_wjaccard"],
            "mean_feature_jaccard": result["mean_feature_jaccard"],
            "mean_churn": result["mean_churn"],
            "mean_core_size": result["mean_core_size"],
            "mean_core_mass_fraction": result["mean_core_mass"],
        },
        "per_step": {
            "step_index": result["step_index"],
            "n_active_features": result["n_active_features"],
            "n_edges_retained": result["n_edges_retained"],
            "edge_wjaccard": _json_ready(result["wjaccard_curve"]),
            "edge_jaccard": _json_ready(result["edge_jaccard_curve"]),
            "feature_jaccard": _json_ready(result["feature_jaccard_curve"]),
            "core_size": _json_ready(result["core_size_curve"]),
            "core_mass_fraction": _json_ready(result["core_mass_curve"]),
            "logprob": _json_ready(result["logprob"]),
            "token_text": result["token_text"],
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))


def run_analysis(args: argparse.Namespace) -> None:
    traces_dir = Path(args.traces_dir)
    if not traces_dir.exists():
        print(f"Traces directory not found: {traces_dir}")
        sys.exit(1)

    prompt_dirs = sorted(traces_dir.glob("prompt_*"))
    if not prompt_dirs:
        print(f"No prompt directories in {traces_dir}")
        sys.exit(1)

    # Collect all completions with their correctness labels
    correct_results: list[dict] = []
    incorrect_results: list[dict] = []
    unlabeled_results: list[dict] = []

    for prompt_dir in tqdm(prompt_dirs, desc="Prompts"):
        completion_dirs = sorted(prompt_dir.glob("completion_*"))

        for comp_dir in tqdm(
            completion_dirs,
            desc=f"  {prompt_dir.name}",
            leave=False,
        ):
            if not is_complete_completion(comp_dir):
                tqdm.write(
                    f"Warning: skipping {comp_dir} "
                    "(missing completion.json; run likely interrupted)"
                )
                continue

            result = analyze_single_completion(
                comp_dir, workers=args.workers, max_edges=args.max_edges
            )
            if result is None:
                continue

            result["prompt"] = prompt_dir.name
            result["completion"] = comp_dir.name
            _write_completion_artifacts(comp_dir, result)

            # Check evaluation
            eval_path = comp_dir / "evaluation.json"
            if eval_path.exists():
                evaluation = json.loads(eval_path.read_text())
                result["correct"] = evaluation.get("judge_correct", None)
                if result["correct"] is True:
                    correct_results.append(result)
                elif result["correct"] is False:
                    incorrect_results.append(result)
                else:
                    unlabeled_results.append(result)
            else:
                result["correct"] = None
                unlabeled_results.append(result)

    all_results = correct_results + incorrect_results + unlabeled_results
    print(f"\nAnalyzed {len(all_results)} completions:")
    print(f"  Correct: {len(correct_results)}")
    print(f"  Incorrect: {len(incorrect_results)}")
    print(f"  Unlabeled: {len(unlabeled_results)}")

    if not all_results:
        print("No completions found!")
        return

    # Aggregate statistics
    summary: dict = {"n_completions": len(all_results)}

    for label, group in [
        ("correct", correct_results),
        ("incorrect", incorrect_results),
        ("all", all_results),
    ]:
        if not group:
            summary[label] = None
            continue

        wj_means = [r["mean_wjaccard"] for r in group if r["mean_wjaccard"] is not None]
        churn_means = [r["mean_churn"] for r in group if r["mean_churn"] is not None]
        core_means = [
            r["mean_core_size"] for r in group if r["mean_core_size"] is not None
        ]
        core_mass_means = [
            r["mean_core_mass"] for r in group if r["mean_core_mass"] is not None
        ]

        summary[label] = {
            "n": len(group),
            "mean_wjaccard": float(np.mean(wj_means)) if wj_means else None,
            "std_wjaccard": float(np.std(wj_means)) if wj_means else None,
            "mean_churn": float(np.mean(churn_means)) if churn_means else None,
            "mean_core_size": float(np.mean(core_means)) if core_means else None,
            "mean_core_mass_fraction": float(np.mean(core_mass_means))
            if core_mass_means
            else None,
        }

    # Statistical test: correct vs incorrect
    if correct_results and incorrect_results:
        correct_wj = [
            r["mean_wjaccard"]
            for r in correct_results
            if r["mean_wjaccard"] is not None
        ]
        incorrect_wj = [
            r["mean_wjaccard"]
            for r in incorrect_results
            if r["mean_wjaccard"] is not None
        ]
        if len(correct_wj) >= 2 and len(incorrect_wj) >= 2:
            stat, pval = stats.mannwhitneyu(
                correct_wj, incorrect_wj, alternative="two-sided"
            )
            summary["h2_test"] = {
                "test": "Mann-Whitney U",
                "statistic": float(stat),
                "p_value": float(pval),
                "correct_mean_wj": float(np.mean(correct_wj)),
                "incorrect_mean_wj": float(np.mean(incorrect_wj)),
                "significant_005": pval < 0.05,
            }
            print("\nH2 Test (Mann-Whitney U):")
            print(f"  Correct mean wJaccard: {np.mean(correct_wj):.4f}")
            print(f"  Incorrect mean wJaccard: {np.mean(incorrect_wj):.4f}")
            print(f"  p-value: {pval:.4f}")
            print(f"  Significant (p<0.05): {pval < 0.05}")

    # Save summary
    out_path = traces_dir / "analysis_summary.json"

    # Strip curves from saved summary (keep it compact)
    save_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if not k.endswith("_curve")}
        save_results.append(sr)
    summary["per_completion"] = save_results

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {out_path}")

    # ── Plots ────────────────────────────────────────────────────────
    _plot_comparison(traces_dir, correct_results, incorrect_results, all_results)


def _pad_curves(curves: list[list[float]], fill: float = float("nan")) -> np.ndarray:
    """Pad variable-length curves to the same length, return 2D array."""
    if not curves:
        return np.empty((0, 0))
    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), fill)
    for i, c in enumerate(curves):
        padded[i, : len(c)] = c
    return padded


def _plot_comparison(
    traces_dir: Path,
    correct: list[dict],
    incorrect: list[dict],
    all_results: list[dict],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Temporal Circuit Stability: Correct vs Incorrect", fontsize=14)

    for label, group, color in [
        ("Correct", correct, "steelblue"),
        ("Incorrect", incorrect, "tomato"),
    ]:
        if not group:
            continue

        # wJaccard curves
        wj_curves = _pad_curves([r["wjaccard_curve"] for r in group])
        mean_wj = np.nanmean(wj_curves, axis=0)
        std_wj = np.nanstd(wj_curves, axis=0)
        steps = np.arange(len(mean_wj))

        ax = axes[0, 0]
        ax.plot(
            steps, mean_wj, color=color, label=f"{label} (n={len(group)})", alpha=0.9
        )
        ax.fill_between(
            steps, mean_wj - std_wj, mean_wj + std_wj, color=color, alpha=0.15
        )

        # Churn curves
        churn_curves = _pad_curves([r["churn_curve"] for r in group])
        mean_churn = np.nanmean(churn_curves, axis=0)
        std_churn = np.nanstd(churn_curves, axis=0)

        ax = axes[0, 1]
        ax.plot(steps, mean_churn, color=color, label=label, alpha=0.9)
        ax.fill_between(
            steps,
            mean_churn - std_churn,
            mean_churn + std_churn,
            color=color,
            alpha=0.15,
        )

        # Core size curves
        core_curves = _pad_curves([r["core_size_curve"] for r in group])
        mean_core = np.nanmean(core_curves, axis=0)
        std_core = np.nanstd(core_curves, axis=0)

        ax = axes[1, 0]
        ax.plot(steps, mean_core, color=color, label=label, alpha=0.9)
        ax.fill_between(
            steps, mean_core - std_core, mean_core + std_core, color=color, alpha=0.15
        )

        # Core mass curves
        core_mass_curves = _pad_curves([r["core_mass_curve"] for r in group])
        mean_core_mass = np.nanmean(core_mass_curves, axis=0)
        std_core_mass = np.nanstd(core_mass_curves, axis=0)

        ax = axes[1, 1]
        ax.plot(steps, mean_core_mass, color=color, label=label, alpha=0.9)
        ax.fill_between(
            steps,
            mean_core_mass - std_core_mass,
            mean_core_mass + std_core_mass,
            color=color,
            alpha=0.15,
        )

    # Labels
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Weighted Jaccard")
    axes[0, 0].set_title("Edge Overlap (wJaccard)")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_ylim(-0.02, 1.02)

    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Churn (1 - wJacc)")
    axes[0, 1].set_title("Edge Churn")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_ylim(-0.02, 1.02)

    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Core Size (edges)")
    axes[1, 0].set_title(
        f"Stable Core (W={STABLE_CORE_WINDOW}, p={STABLE_CORE_PERSISTENCE})"
    )
    axes[1, 0].legend(fontsize=9)

    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Mass fraction")
    axes[1, 1].set_title("Stable core mass fraction")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_ylim(-0.02, 1.02)

    # Box plot: mean wJaccard correct vs incorrect
    ax = axes[2, 0]
    box_data = []
    box_labels = []
    if correct:
        wj_c = [r["mean_wjaccard"] for r in correct if r["mean_wjaccard"] is not None]
        if wj_c:
            box_data.append(wj_c)
            box_labels.append(f"Correct\n(n={len(wj_c)})")
    if incorrect:
        wj_i = [r["mean_wjaccard"] for r in incorrect if r["mean_wjaccard"] is not None]
        if wj_i:
            box_data.append(wj_i)
            box_labels.append(f"Incorrect\n(n={len(wj_i)})")
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ["steelblue", "tomato"][: len(box_data)]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
    ax.set_ylabel("Mean wJaccard")
    ax.set_title("H2: Stability vs Correctness")

    axes[2, 1].axis("off")

    plt.tight_layout()
    plot_path = traces_dir / "comparison_analysis.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch temporal circuit analysis")
    parser.add_argument(
        "--traces-dir",
        required=True,
        help="Directory containing prompt_*/completion_* traces",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel workers for step loading"
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=10_000,
        help="Edge cap (only used for .pt files)",
    )
    args = parser.parse_args()
    run_analysis(args)
