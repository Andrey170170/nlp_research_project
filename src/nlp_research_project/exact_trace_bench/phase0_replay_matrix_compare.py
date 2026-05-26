from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from .graph_compare import compare_artifact_dirs
from .io_utils import read_json
from .phase3_seed_bundle_compare import compare_phase3_seed_bundles
from .semantic_feature_compare import compare_semantic_feature_descriptors

PHASE3_BUNDLE_PATTERN = "step_*_phase3_seed_bundle.npz"
SEMANTIC_DESCRIPTOR_PATTERN = "step_*_feature_semantic_descriptors.npz"

SELF_REPLAY_THRESHOLDS: dict[str, tuple[str, str, float]] = {
    "feature_support_jaccard": ("phase3_support_jaccard", "eq", 1.0),
    "phase3_seed_influence_pearson": (
        "phase3_seed_influence_pearson",
        "ge",
        0.9999,
    ),
    "phase3_top1024_overlap": ("phase3_seed_top1024_overlap", "ge", 0.999),
    "compact_weighted_edge_jaccard": (
        "compact_weighted_edge_jaccard",
        "ge",
        0.999,
    ),
}

MOVEMENT_METRICS = (
    "compact_feature_jaccard",
    "compact_weighted_edge_jaccard",
    "phase3_support_jaccard",
    "phase3_seed_influence_pearson",
    "phase3_frontier_post_jaccard",
)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mean_or_none(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return float(mean(usable))


def _completion_key(artifacts_root: Path, completion_dir: Path) -> str:
    return str(completion_dir.relative_to(artifacts_root))


def _completion_map(artifacts_root: Path) -> dict[str, Path]:
    return {
        _completion_key(artifacts_root, completion_dir): completion_dir
        for completion_dir in sorted(artifacts_root.glob("prompt_*/completion_*"))
    }


def _compare_phase3_artifact_dirs(
    left_artifacts: Path, right_artifacts: Path
) -> dict[str, Any]:
    left_completions = _completion_map(left_artifacts)
    right_completions = _completion_map(right_artifacts)
    shared_completion_keys = sorted(set(left_completions) & set(right_completions))

    completion_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for completion_key in shared_completion_keys:
        left_completion = left_completions[completion_key]
        right_completion = right_completions[completion_key]
        left_bundles = {
            path.name: path
            for path in sorted(left_completion.glob(PHASE3_BUNDLE_PATTERN))
        }
        right_bundles = {
            path.name: path
            for path in sorted(right_completion.glob(PHASE3_BUNDLE_PATTERN))
        }
        shared_bundle_names = sorted(set(left_bundles) & set(right_bundles))
        completion_step_rows: list[dict[str, Any]] = []

        for bundle_name in shared_bundle_names:
            try:
                row = compare_phase3_seed_bundles(
                    left_bundles[bundle_name],
                    right_bundles[bundle_name],
                )
            except Exception as exc:  # pragma: no cover - defensive path
                warnings.append(
                    f"phase3 compare failed for {completion_key}/{bundle_name}: {exc}"
                )
                continue

            seed_stability = row.get("shared_support_score_stability", {}).get(
                "seed_feature_influences", {}
            )
            topk_1024 = (
                seed_stability.get("topk_overlap", {})
                .get("1024", {})
                .get("overlap_fraction_of_k")
            )
            metrics = {
                "completion_key": completion_key,
                "step_bundle": bundle_name,
                "support_jaccard": _as_float(
                    row.get("support", {}).get("feature_jaccard")
                ),
                "seed_influence_pearson": _as_float(seed_stability.get("pearson")),
                "seed_top1024_overlap": _as_float(topk_1024),
                "frontier_post_jaccard": _as_float(
                    row.get("frontier_post_locality", {}).get("jaccard")
                ),
            }
            completion_step_rows.append(metrics)
            step_rows.append(metrics)

        if completion_step_rows:
            completion_rows.append(
                {
                    "completion_key": completion_key,
                    "n_phase3_bundles_aligned": len(completion_step_rows),
                    "mean_support_jaccard": _mean_or_none(
                        [row["support_jaccard"] for row in completion_step_rows]
                    ),
                    "mean_seed_influence_pearson": _mean_or_none(
                        [row["seed_influence_pearson"] for row in completion_step_rows]
                    ),
                    "mean_seed_top1024_overlap": _mean_or_none(
                        [row["seed_top1024_overlap"] for row in completion_step_rows]
                    ),
                    "mean_frontier_post_jaccard": _mean_or_none(
                        [row["frontier_post_jaccard"] for row in completion_step_rows]
                    ),
                }
            )

    summary: dict[str, Any] = {
        "left_artifacts": str(left_artifacts),
        "right_artifacts": str(right_artifacts),
        "shared_completion_count": len(shared_completion_keys),
        "left_only_completion_count": len(
            set(left_completions) - set(right_completions)
        ),
        "right_only_completion_count": len(
            set(right_completions) - set(left_completions)
        ),
        "completion_comparisons": completion_rows,
        "step_comparisons": step_rows,
        "warnings": warnings,
    }
    if step_rows:
        summary["overall_mean_support_jaccard"] = _mean_or_none(
            [row["support_jaccard"] for row in step_rows]
        )
        summary["overall_mean_seed_influence_pearson"] = _mean_or_none(
            [row["seed_influence_pearson"] for row in step_rows]
        )
        summary["overall_mean_seed_top1024_overlap"] = _mean_or_none(
            [row["seed_top1024_overlap"] for row in step_rows]
        )
        summary["overall_mean_frontier_post_jaccard"] = _mean_or_none(
            [row["frontier_post_jaccard"] for row in step_rows]
        )
    return summary


def _compare_semantic_artifact_dirs(
    left_artifacts: Path, right_artifacts: Path
) -> dict[str, Any]:
    left_completions = _completion_map(left_artifacts)
    right_completions = _completion_map(right_artifacts)
    shared_completion_keys = sorted(set(left_completions) & set(right_completions))

    comparison_rows: list[dict[str, Any]] = []
    interpretation_counts: dict[str, int] = {}
    warnings: list[str] = []

    for completion_key in shared_completion_keys:
        left_descriptors = {
            path.name: path
            for path in sorted(
                left_completions[completion_key].glob(SEMANTIC_DESCRIPTOR_PATTERN)
            )
        }
        right_descriptors = {
            path.name: path
            for path in sorted(
                right_completions[completion_key].glob(SEMANTIC_DESCRIPTOR_PATTERN)
            )
        }
        shared_descriptor_names = sorted(set(left_descriptors) & set(right_descriptors))

        for descriptor_name in shared_descriptor_names:
            try:
                summary = compare_semantic_feature_descriptors(
                    left_descriptors[descriptor_name],
                    right_descriptors[descriptor_name],
                )
            except Exception as exc:  # pragma: no cover - defensive path
                warnings.append(
                    "semantic compare failed "
                    f"for {completion_key}/{descriptor_name}: {exc}"
                )
                continue

            interpretation = str(summary.get("interpretation") or "unknown")
            interpretation_counts[interpretation] = (
                interpretation_counts.get(interpretation, 0) + 1
            )
            comparison_rows.append(
                {
                    "completion_key": completion_key,
                    "step_descriptor": descriptor_name,
                    "candidate_feature_jaccard": _as_float(
                        summary.get("candidate_support", {}).get("feature_jaccard")
                    ),
                    "seed_influence_pearson": _as_float(
                        summary.get("shared_candidate_scores", {}).get(
                            "seed_influence_pearson"
                        )
                    ),
                    "interpretation": interpretation,
                }
            )

    if not comparison_rows:
        return {
            "status": "not_available",
            "shared_completion_count": len(shared_completion_keys),
            "shared_descriptor_count": 0,
            "warnings": warnings,
        }

    return {
        "status": "ok",
        "shared_completion_count": len(shared_completion_keys),
        "shared_descriptor_count": len(comparison_rows),
        "overall_mean_candidate_feature_jaccard": _mean_or_none(
            [row["candidate_feature_jaccard"] for row in comparison_rows]
        ),
        "overall_mean_seed_influence_pearson": _mean_or_none(
            [row["seed_influence_pearson"] for row in comparison_rows]
        ),
        "interpretation_counts": dict(sorted(interpretation_counts.items())),
        "comparisons": comparison_rows,
        "warnings": warnings,
    }


def _extract_similarity_metrics(
    compact_summary: dict[str, Any], phase3_summary: dict[str, Any]
) -> dict[str, float | None]:
    return {
        "compact_feature_jaccard": _as_float(
            compact_summary.get("overall_mean_feature_jaccard")
        ),
        "compact_weighted_edge_jaccard": _as_float(
            compact_summary.get("overall_mean_weighted_edge_jaccard")
        ),
        "phase3_support_jaccard": _as_float(
            phase3_summary.get("overall_mean_support_jaccard")
        ),
        "phase3_seed_influence_pearson": _as_float(
            phase3_summary.get("overall_mean_seed_influence_pearson")
        ),
        "phase3_seed_top1024_overlap": _as_float(
            phase3_summary.get("overall_mean_seed_top1024_overlap")
        ),
        "phase3_frontier_post_jaccard": _as_float(
            phase3_summary.get("overall_mean_frontier_post_jaccard")
        ),
    }


def _scan_dtype_roundtrip_loss(artifacts_root: Path) -> dict[str, Any]:
    manifest_paths = sorted(
        artifacts_root.glob("prompt_*/completion_*/completion.json")
    )
    completion_keys_with_loss: list[str] = []
    warnings: list[str] = []

    for manifest_path in manifest_paths:
        completion_key = _completion_key(artifacts_root, manifest_path.parent)
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:  # pragma: no cover - defensive path
            warnings.append(f"failed to read manifest {manifest_path}: {exc}")
            continue

        detected = bool(manifest.get("phase0_replay_dtype_roundtrip_loss")) or bool(
            manifest.get("phase0_replay_any_dtype_roundtrip_loss")
        )
        if not detected:
            for step in manifest.get("steps", []):
                if isinstance(step, dict) and bool(
                    step.get("phase0_replay_dtype_roundtrip_loss")
                ):
                    detected = True
                    break

        if detected:
            completion_keys_with_loss.append(completion_key)

    return {
        "artifact_root": str(artifacts_root),
        "manifest_count": len(manifest_paths),
        "completion_count_with_loss": len(completion_keys_with_loss),
        "completion_keys_with_loss": completion_keys_with_loss,
        "detected": bool(completion_keys_with_loss),
        "warnings": warnings,
    }


def _evaluate_check(value: float | None, op: str, threshold: float) -> bool:
    if value is None:
        return False
    if op == "eq":
        return value == threshold
    if op == "ge":
        return value >= threshold
    raise ValueError(f"Unsupported check operator: {op}")


def _self_replay_gate(
    similarities: dict[str, float | None],
    *,
    dtype_roundtrip_report: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    for check_name, (metric_key, op, threshold) in SELF_REPLAY_THRESHOLDS.items():
        value = similarities.get(metric_key)
        passed = _evaluate_check(value, op, threshold)
        checks[check_name] = {
            "metric": metric_key,
            "operator": op,
            "threshold": threshold,
            "value": value,
            "available": value is not None,
            "pass": passed,
        }

    warnings = list(dtype_roundtrip_report.get("warnings", []))
    if bool(dtype_roundtrip_report.get("detected")):
        warnings.append(
            "dtype_roundtrip_loss detected from completion manifest; strict self-replay "
            "thresholds remain enforced (looser thresholds are not yet auto-applied)."
        )

    return {
        "pass": all(check["pass"] for check in checks.values()),
        "checks": checks,
        "dtype_roundtrip_loss_detected": bool(dtype_roundtrip_report.get("detected")),
        "dtype_roundtrip_loss": dtype_roundtrip_report,
        "warnings": warnings,
    }


def _movement_summary(
    host_similarity: dict[str, float | None],
    donor_similarity: dict[str, float | None],
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for metric in MOVEMENT_METRICS:
        host_value = host_similarity.get(metric)
        donor_value = donor_similarity.get(metric)
        movement = (
            donor_value - host_value
            if host_value is not None and donor_value is not None
            else None
        )
        result[metric] = {
            "host_similarity": host_value,
            "donor_similarity": donor_value,
            "movement_score": movement,
        }
    return result


def _compare_pair(
    left_artifacts: Path,
    right_artifacts: Path,
    *,
    include_semantic: bool,
) -> dict[str, Any]:
    compact = compare_artifact_dirs(left_artifacts, right_artifacts)
    phase3 = _compare_phase3_artifact_dirs(left_artifacts, right_artifacts)
    semantic = (
        _compare_semantic_artifact_dirs(left_artifacts, right_artifacts)
        if include_semantic
        else {"status": "skipped"}
    )
    return {
        "left_artifacts": str(left_artifacts),
        "right_artifacts": str(right_artifacts),
        "similarities": _extract_similarity_metrics(compact, phase3),
        "compact": compact,
        "phase3": phase3,
        "semantic": semantic,
    }


def compare_phase0_replay_matrix(
    ascend_baseline: Path,
    cardinal_baseline: Path,
    ascend_self_replay: Path,
    cardinal_self_replay: Path,
    ascend_with_cardinal: Path,
    cardinal_with_ascend: Path,
    *,
    include_semantic: bool = True,
) -> dict[str, Any]:
    roots = {
        "ascend_baseline": ascend_baseline.resolve(),
        "cardinal_baseline": cardinal_baseline.resolve(),
        "ascend_self_replay": ascend_self_replay.resolve(),
        "cardinal_self_replay": cardinal_self_replay.resolve(),
        "ascend_with_cardinal": ascend_with_cardinal.resolve(),
        "cardinal_with_ascend": cardinal_with_ascend.resolve(),
    }

    pair_specs = (
        (
            "baseline_cross_cluster",
            "ascend_baseline",
            "cardinal_baseline",
        ),
        (
            "ascend_self_replay",
            "ascend_baseline",
            "ascend_self_replay",
        ),
        (
            "cardinal_self_replay",
            "cardinal_baseline",
            "cardinal_self_replay",
        ),
        (
            "ascend_with_cardinal_vs_host",
            "ascend_with_cardinal",
            "ascend_baseline",
        ),
        (
            "ascend_with_cardinal_vs_donor",
            "ascend_with_cardinal",
            "cardinal_baseline",
        ),
        (
            "cardinal_with_ascend_vs_host",
            "cardinal_with_ascend",
            "cardinal_baseline",
        ),
        (
            "cardinal_with_ascend_vs_donor",
            "cardinal_with_ascend",
            "ascend_baseline",
        ),
    )

    pairwise: dict[str, dict[str, Any]] = {}
    for pair_name, left_name, right_name in pair_specs:
        pairwise[pair_name] = _compare_pair(
            roots[left_name],
            roots[right_name],
            include_semantic=include_semantic,
        )

    ascend_dtype_report = _scan_dtype_roundtrip_loss(roots["ascend_self_replay"])
    cardinal_dtype_report = _scan_dtype_roundtrip_loss(roots["cardinal_self_replay"])

    self_replay_gate = {
        "ascend_self_replay": _self_replay_gate(
            pairwise["ascend_self_replay"]["similarities"],
            dtype_roundtrip_report=ascend_dtype_report,
        ),
        "cardinal_self_replay": _self_replay_gate(
            pairwise["cardinal_self_replay"]["similarities"],
            dtype_roundtrip_report=cardinal_dtype_report,
        ),
    }

    movement_scores = {
        "ascend_with_cardinal": _movement_summary(
            pairwise["ascend_with_cardinal_vs_host"]["similarities"],
            pairwise["ascend_with_cardinal_vs_donor"]["similarities"],
        ),
        "cardinal_with_ascend": _movement_summary(
            pairwise["cardinal_with_ascend_vs_host"]["similarities"],
            pairwise["cardinal_with_ascend_vs_donor"]["similarities"],
        ),
    }

    return {
        "artifact_roots": {name: str(path) for name, path in roots.items()},
        "pairwise": pairwise,
        "movement_scores": movement_scores,
        "self_replay_gate": self_replay_gate,
    }


def compare_phase0_replay_matrix_to_json(
    ascend_baseline: Path,
    cardinal_baseline: Path,
    ascend_self_replay: Path,
    cardinal_self_replay: Path,
    ascend_with_cardinal: Path,
    cardinal_with_ascend: Path,
    *,
    output_json: Path | None = None,
    include_semantic: bool = True,
) -> dict[str, Any]:
    result = compare_phase0_replay_matrix(
        ascend_baseline,
        cardinal_baseline,
        ascend_self_replay,
        cardinal_self_replay,
        ascend_with_cardinal,
        cardinal_with_ascend,
        include_semantic=include_semantic,
    )
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
