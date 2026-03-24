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
    steps = load_completion_steps(completion_dir, workers=workers, max_edges=max_edges)
    if len(steps) < 2:
        return None

    metrics = compute_temporal_metrics(steps)

    # Filter NaN
    valid_wj = [v for v in metrics["edge_wjaccard"] if not np.isnan(v)]
    valid_fj = [v for v in metrics["feature_jaccard"] if not np.isnan(v)]
    valid_cores = [v for v in metrics["core_sizes"] if not np.isnan(v)]

    return {
        "n_steps": len(steps),
        "mean_wjaccard": float(np.mean(valid_wj)) if valid_wj else None,
        "std_wjaccard": float(np.std(valid_wj)) if valid_wj else None,
        "mean_feature_jaccard": float(np.mean(valid_fj)) if valid_fj else None,
        "mean_core_size": float(np.mean(valid_cores)) if valid_cores else None,
        "mean_churn": float(1 - np.mean(valid_wj)) if valid_wj else None,
        "wjaccard_curve": metrics["edge_wjaccard"],
        "feature_jaccard_curve": metrics["feature_jaccard"],
        "core_size_curve": metrics["core_sizes"],
        "churn_curve": [1 - v for v in metrics["edge_wjaccard"]],
    }


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
            result = analyze_single_completion(
                comp_dir, workers=args.workers, max_edges=args.max_edges
            )
            if result is None:
                continue

            result["prompt"] = prompt_dir.name
            result["completion"] = comp_dir.name

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

        summary[label] = {
            "n": len(group),
            "mean_wjaccard": float(np.mean(wj_means)) if wj_means else None,
            "std_wjaccard": float(np.std(wj_means)) if wj_means else None,
            "mean_churn": float(np.mean(churn_means)) if churn_means else None,
            "mean_core_size": float(np.mean(core_means)) if core_means else None,
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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

    # Box plot: mean wJaccard correct vs incorrect
    ax = axes[1, 1]
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
