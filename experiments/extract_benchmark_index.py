from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

from extract_utils import ensure_dir, flatten_dict, read_json, write_csv, write_jsonl


DEFAULT_INPUT_ROOT = Path("/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked")
DEFAULT_OUTPUT_DIR = Path("experiments/extracted/weekend_exact_chunked")


def _infer_cluster(scenario_root: Path, scenario: dict[str, Any]) -> str | None:
    if scenario.get("cluster"):
        return str(scenario["cluster"])
    for part in scenario_root.parts:
        if part in {"ascend", "cardinal"}:
            return part
    return None


def _special_case_label(scenario_root: Path, scenario: dict[str, Any]) -> str | None:
    stage = str(scenario.get("stage") or "")
    joined = "/".join(scenario_root.parts)
    if "prompt94_compare" in stage or "prompt94_compare" in joined:
        return "prompt94_compare"
    return None


def _load_prompt_meta(artifact_dir: Path) -> dict[str, Any]:
    prompt_meta_files = sorted(artifact_dir.glob("prompt_*/prompt_meta.json"))
    if not prompt_meta_files:
        return {}
    return read_json(prompt_meta_files[0])


def _load_completion_manifests(artifact_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(artifact_dir.glob("prompt_*/completion_*/completion.json")):
        manifests.append(read_json(path))
    return manifests


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _to_int(value: Any) -> int | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _sanitize_column_fragment(value: str) -> str:
    fragment = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return fragment or "unknown"


def _relative_to_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _summarize_artifacts(artifact_dir: Path) -> dict[str, Any]:
    prompt_meta = _load_prompt_meta(artifact_dir)
    manifest_paths = sorted(artifact_dir.glob("prompt_*/completion_*/completion.json"))
    manifests = [read_json(path) for path in manifest_paths]

    steps = [
        step
        for manifest in manifests
        for step in manifest.get("steps", [])
        if isinstance(step, dict)
    ]
    diagnostics = [
        step.get("transcoder_diagnostics", {})
        for step in steps
        if isinstance(step.get("transcoder_diagnostics"), dict)
    ]

    tracked_step_fields = (
        "step_end_to_end_seconds",
        "attribution_seconds",
        "token_generation_seconds",
        "artifact_save_seconds",
    )

    telemetry_declared_paths: set[str] = set()
    telemetry_existing_paths: set[str] = set()
    telemetry_missing_paths: set[str] = set()
    telemetry_manifest_event_counts: list[int] = []
    completion_timing_summary_count = 0
    completion_timing_completion_end_to_end_values: list[float] = []
    completion_timing_step_counts: list[int] = []
    completion_timing_totals: dict[str, float] = {
        field_name: 0.0 for field_name in tracked_step_fields
    }
    completion_timing_totals_counts: dict[str, int] = {
        field_name: 0 for field_name in tracked_step_fields
    }
    completion_timing_averages_per_step: dict[str, list[float]] = {
        field_name: [] for field_name in tracked_step_fields
    }
    completion_phase_elapsed_totals: dict[str, float] = {}

    for manifest_path, manifest in zip(manifest_paths, manifests, strict=False):
        completion_dir = manifest_path.parent

        telemetry_ref = manifest.get("telemetry_events_path")
        resolved_telemetry_path: Path | None = None
        declared_telemetry_path: Path | None = None
        if isinstance(telemetry_ref, str) and telemetry_ref.strip():
            declared_telemetry_path = completion_dir / telemetry_ref.strip()
            telemetry_declared_paths.add(
                _relative_to_or_str(declared_telemetry_path, artifact_dir)
            )
            if declared_telemetry_path.exists():
                resolved_telemetry_path = declared_telemetry_path
            else:
                telemetry_missing_paths.add(
                    _relative_to_or_str(declared_telemetry_path, artifact_dir)
                )
        default_telemetry_path = completion_dir / "telemetry.jsonl"
        if resolved_telemetry_path is None and default_telemetry_path.exists():
            resolved_telemetry_path = default_telemetry_path

        if resolved_telemetry_path is not None:
            relative_path = _relative_to_or_str(resolved_telemetry_path, artifact_dir)
            telemetry_existing_paths.add(relative_path)

        telemetry_event_count = _to_int(manifest.get("telemetry_event_count"))
        if telemetry_event_count is not None:
            telemetry_manifest_event_counts.append(telemetry_event_count)

        timing_summary = manifest.get("timing_summary")
        if not isinstance(timing_summary, dict):
            continue

        completion_timing_summary_count += 1

        completion_end_to_end_seconds = _to_float(
            timing_summary.get("completion_end_to_end_seconds")
        )
        if completion_end_to_end_seconds is not None:
            completion_timing_completion_end_to_end_values.append(
                completion_end_to_end_seconds
            )

        timing_step_count = _to_int(timing_summary.get("step_count"))
        if timing_step_count is not None:
            completion_timing_step_counts.append(timing_step_count)

        totals = timing_summary.get("totals")
        if isinstance(totals, dict):
            for field_name in tracked_step_fields:
                total_seconds = _to_float(totals.get(field_name))
                if total_seconds is None:
                    continue
                completion_timing_totals[field_name] += total_seconds
                completion_timing_totals_counts[field_name] += 1

        averages_per_step = timing_summary.get("averages_per_step")
        if isinstance(averages_per_step, dict):
            for field_name in tracked_step_fields:
                avg_seconds = _to_float(averages_per_step.get(field_name))
                if avg_seconds is None:
                    continue
                completion_timing_averages_per_step[field_name].append(avg_seconds)

        completion_phase_elapsed = timing_summary.get(
            "attribution_phase_elapsed_seconds_total"
        )
        if isinstance(completion_phase_elapsed, dict):
            for phase_name, phase_elapsed_seconds in completion_phase_elapsed.items():
                if not isinstance(phase_name, str):
                    continue
                phase_elapsed_seconds_float = _to_float(phase_elapsed_seconds)
                if phase_elapsed_seconds_float is None:
                    continue
                completion_phase_elapsed_totals[phase_name] = (
                    completion_phase_elapsed_totals.get(phase_name, 0.0)
                    + phase_elapsed_seconds_float
                )

    step_phase_elapsed_totals: dict[str, float] = {}
    if not completion_phase_elapsed_totals:
        for step in steps:
            step_phase_elapsed = step.get("attribution_phase_elapsed_seconds")
            if not isinstance(step_phase_elapsed, dict):
                continue
            for phase_name, phase_elapsed_seconds in step_phase_elapsed.items():
                if not isinstance(phase_name, str):
                    continue
                phase_elapsed_seconds_float = _to_float(phase_elapsed_seconds)
                if phase_elapsed_seconds_float is None:
                    continue
                step_phase_elapsed_totals[phase_name] = (
                    step_phase_elapsed_totals.get(phase_name, 0.0)
                    + phase_elapsed_seconds_float
                )

    phase_elapsed_totals = (
        completion_phase_elapsed_totals
        if completion_phase_elapsed_totals
        else step_phase_elapsed_totals
    )
    phase_elapsed_source = (
        "completion_timing_summary"
        if completion_phase_elapsed_totals
        else "step_records"
        if step_phase_elapsed_totals
        else None
    )

    phase_elapsed_columns = {
        (
            "attribution_phase_elapsed_seconds_total_"
            f"{_sanitize_column_fragment(phase_name)}"
        ): round(total_seconds, 6)
        for phase_name, total_seconds in sorted(phase_elapsed_totals.items())
    }

    step_timing_values: dict[str, list[float]] = {
        field_name: [] for field_name in tracked_step_fields
    }
    for step in steps:
        for field_name in tracked_step_fields:
            field_seconds = _to_float(step.get(field_name))
            if field_seconds is not None:
                step_timing_values[field_name].append(field_seconds)

    step_timing_columns: dict[str, Any] = {
        "step_timing_sample_count": len(steps),
    }
    for field_name, values in step_timing_values.items():
        step_timing_columns[f"step_timing_{field_name}_total"] = (
            round(sum(values), 6) if values else None
        )
        step_timing_columns[f"step_timing_{field_name}_mean"] = (
            round(mean(values), 6) if values else None
        )
        step_timing_columns[f"step_timing_{field_name}_max"] = (
            round(max(values), 6) if values else None
        )

    telemetry_event_count_steps = [
        telemetry_event_count
        for telemetry_event_count in (
            _to_int(step.get("telemetry_event_count")) for step in steps
        )
        if telemetry_event_count is not None
    ]

    completion_timing_step_count_total = sum(completion_timing_step_counts)
    completion_timing_columns: dict[str, Any] = {
        "completion_timing_summary_count": completion_timing_summary_count,
        "completion_timing_step_count_total": completion_timing_step_count_total,
        "completion_timing_step_count_mean": (
            round(mean([float(value) for value in completion_timing_step_counts]), 6)
            if completion_timing_step_counts
            else None
        ),
        "completion_timing_completion_end_to_end_seconds_total": (
            round(sum(completion_timing_completion_end_to_end_values), 6)
            if completion_timing_completion_end_to_end_values
            else None
        ),
        "completion_timing_completion_end_to_end_seconds_mean": (
            round(mean(completion_timing_completion_end_to_end_values), 6)
            if completion_timing_completion_end_to_end_values
            else None
        ),
        "completion_timing_completion_end_to_end_seconds_max": (
            round(max(completion_timing_completion_end_to_end_values), 6)
            if completion_timing_completion_end_to_end_values
            else None
        ),
    }
    for field_name in tracked_step_fields:
        total_seconds = completion_timing_totals[field_name]
        value_count = completion_timing_totals_counts[field_name]
        completion_timing_columns[f"completion_timing_totals_{field_name}_total"] = (
            round(total_seconds, 6) if value_count else None
        )
        completion_timing_columns[
            f"completion_timing_totals_{field_name}_avg_per_completion"
        ] = (
            round(total_seconds / completion_timing_summary_count, 6)
            if completion_timing_summary_count and value_count
            else None
        )
        completion_timing_columns[
            f"completion_timing_totals_{field_name}_avg_per_step"
        ] = (
            round(total_seconds / completion_timing_step_count_total, 6)
            if completion_timing_step_count_total and value_count
            else None
        )
        completion_timing_columns[
            f"completion_timing_averages_per_step_{field_name}_mean"
        ] = (
            round(mean(completion_timing_averages_per_step[field_name]), 6)
            if completion_timing_averages_per_step[field_name]
            else None
        )

    resource_snapshot = manifests[0].get("resource_snapshot") if manifests else None
    first_manifest = manifests[0] if manifests else {}
    telemetry_existing_paths_sorted = sorted(telemetry_existing_paths)
    telemetry_declared_paths_sorted = sorted(telemetry_declared_paths)
    telemetry_missing_paths_sorted = sorted(telemetry_missing_paths)
    max_active_features = max(
        (
            step.get("n_active_features")
            for step in steps
            if step.get("n_active_features") is not None
        ),
        default=None,
    )
    max_edges_retained = max(
        (
            step.get("n_edges_retained")
            for step in steps
            if step.get("n_edges_retained") is not None
        ),
        default=None,
    )
    decoder_cache_hit_count = max(
        (
            diag.get("decoder_cache_hit_count")
            for diag in diagnostics
            if diag.get("decoder_cache_hit_count") is not None
        ),
        default=None,
    )
    decoder_cache_miss_count = max(
        (
            diag.get("decoder_cache_miss_count")
            for diag in diagnostics
            if diag.get("decoder_cache_miss_count") is not None
        ),
        default=None,
    )
    decoder_cache_eviction_count = max(
        (
            diag.get("decoder_cache_eviction_count")
            for diag in diagnostics
            if diag.get("decoder_cache_eviction_count") is not None
        ),
        default=None,
    )
    decoder_load_count = max(
        (
            diag.get("decoder_load_count")
            for diag in diagnostics
            if diag.get("decoder_load_count") is not None
        ),
        default=None,
    )
    decoder_load_seconds = max(
        (
            diag.get("decoder_load_seconds")
            for diag in diagnostics
            if diag.get("decoder_load_seconds") is not None
        ),
        default=None,
    )
    reconstruction_chunk_count = max(
        (
            diag.get("reconstruction_chunk_count")
            for diag in diagnostics
            if diag.get("reconstruction_chunk_count") is not None
        ),
        default=None,
    )
    reconstruction_seconds = max(
        (
            diag.get("reconstruction_seconds")
            for diag in diagnostics
            if diag.get("reconstruction_seconds") is not None
        ),
        default=None,
    )
    encode_sparse_seconds = max(
        (
            diag.get("encode_sparse_seconds")
            for diag in diagnostics
            if diag.get("encode_sparse_seconds") is not None
        ),
        default=None,
    )
    step_phase4_feature_batch_sizes = [
        int(step["phase4_feature_batch_size"])
        for step in steps
        if step.get("phase4_feature_batch_size") is not None
    ]
    step_phase4_planner_statuses = [
        str(step["phase4_feature_batch_planner_status"])
        for step in steps
        if step.get("phase4_feature_batch_planner_status") is not None
    ]
    manifest_phase4_feature_batch_sizes: list[int] = []
    manifest_phase4_effective_sizes: list[int] = []
    manifest_phase4_planner_statuses: list[str] = []
    manifest_phase4_planner_skip_reasons: list[str] = []
    for manifest in manifests:
        manifest_observed = manifest.get("phase4_feature_batch_sizes_observed")
        if isinstance(manifest_observed, list):
            manifest_phase4_feature_batch_sizes.extend(
                int(value)
                for value in manifest_observed
                if isinstance(value, (int, float))
            )
        manifest_effective = manifest.get("phase4_feature_batch_size_effective")
        if isinstance(manifest_effective, (int, float)):
            manifest_phase4_effective_sizes.append(int(manifest_effective))
        manifest_status = manifest.get("phase4_feature_batch_planner_status")
        if isinstance(manifest_status, str):
            manifest_phase4_planner_statuses.append(manifest_status)
        manifest_skip_reason = manifest.get("phase4_feature_batch_planner_skip_reason")
        if isinstance(manifest_skip_reason, str):
            manifest_phase4_planner_skip_reasons.append(manifest_skip_reason)
    all_phase4_feature_batch_sizes = sorted(
        set(
            step_phase4_feature_batch_sizes
            + manifest_phase4_feature_batch_sizes
            + manifest_phase4_effective_sizes
        )
    )
    all_phase4_planner_statuses = sorted(
        set(step_phase4_planner_statuses + manifest_phase4_planner_statuses)
    )

    return {
        "prompt_count": len(list(artifact_dir.glob("prompt_*"))),
        "completion_count": len(manifests),
        "gsm8k_index": prompt_meta.get("gsm8k_index"),
        "prompt_source": first_manifest.get(
            "prompt_source", prompt_meta.get("prompt_source")
        ),
        "fixture_name": first_manifest.get(
            "fixture_name", prompt_meta.get("fixture_name")
        ),
        "fixture_kind": first_manifest.get(
            "fixture_kind", prompt_meta.get("fixture_kind")
        ),
        "prepared_prompt_file": prompt_meta.get("prepared_prompt_file"),
        "prepared_prompt_meta_file": prompt_meta.get("prepared_prompt_meta_file"),
        "prompt_token_count": first_manifest.get(
            "prompt_token_count", prompt_meta.get("prompt_token_count")
        ),
        "initial_input_token_count": first_manifest.get(
            "initial_input_token_count", prompt_meta.get("initial_input_token_count")
        ),
        "generated_token_count": first_manifest.get("generated_token_count"),
        "completion_duration_seconds": first_manifest.get("duration_seconds"),
        "n_steps_traced": first_manifest.get("n_steps_traced"),
        "max_active_features": max_active_features,
        "max_edges_retained": max_edges_retained,
        "decoder_cache_hit_count": decoder_cache_hit_count,
        "decoder_cache_miss_count": decoder_cache_miss_count,
        "decoder_cache_eviction_count": decoder_cache_eviction_count,
        "decoder_load_count": decoder_load_count,
        "decoder_load_seconds": decoder_load_seconds,
        "reconstruction_chunk_count": reconstruction_chunk_count,
        "reconstruction_seconds": reconstruction_seconds,
        "encode_sparse_seconds": encode_sparse_seconds,
        "phase4_feature_batch_size_effective": (
            max(manifest_phase4_effective_sizes)
            if manifest_phase4_effective_sizes
            else max(all_phase4_feature_batch_sizes)
            if all_phase4_feature_batch_sizes
            else None
        ),
        "phase4_feature_batch_size_observed_min": (
            min(all_phase4_feature_batch_sizes)
            if all_phase4_feature_batch_sizes
            else None
        ),
        "phase4_feature_batch_size_observed_max": (
            max(all_phase4_feature_batch_sizes)
            if all_phase4_feature_batch_sizes
            else None
        ),
        "phase4_feature_batch_sizes_observed": all_phase4_feature_batch_sizes,
        "phase4_feature_batch_planner_status": (
            manifest_phase4_planner_statuses[-1]
            if manifest_phase4_planner_statuses
            else step_phase4_planner_statuses[-1]
            if step_phase4_planner_statuses
            else None
        ),
        "phase4_feature_batch_planner_statuses_observed": all_phase4_planner_statuses,
        "phase4_feature_batch_planner_skip_reason": (
            manifest_phase4_planner_skip_reasons[-1]
            if manifest_phase4_planner_skip_reasons
            else None
        ),
        "telemetry_present": bool(telemetry_existing_paths_sorted),
        "telemetry_manifest_declared_count": len(telemetry_declared_paths_sorted),
        "telemetry_file_count": len(telemetry_existing_paths_sorted),
        "telemetry_missing_file_count": len(telemetry_missing_paths_sorted),
        "telemetry_available_for_all_completions": bool(manifests)
        and len(telemetry_existing_paths_sorted) >= len(manifests),
        "telemetry_events_path_example": (
            telemetry_existing_paths_sorted[0]
            if telemetry_existing_paths_sorted
            else telemetry_declared_paths_sorted[0]
            if telemetry_declared_paths_sorted
            else None
        ),
        "telemetry_events_paths_found_sample": "|".join(
            telemetry_existing_paths_sorted[:3]
        )
        if telemetry_existing_paths_sorted
        else None,
        "telemetry_events_paths_missing_sample": "|".join(
            telemetry_missing_paths_sorted[:3]
        )
        if telemetry_missing_paths_sorted
        else None,
        "telemetry_event_count_manifest_total": (
            sum(telemetry_manifest_event_counts)
            if telemetry_manifest_event_counts
            else None
        ),
        "telemetry_event_count_manifest_mean": (
            round(mean([float(value) for value in telemetry_manifest_event_counts]), 6)
            if telemetry_manifest_event_counts
            else None
        ),
        "telemetry_event_count_steps_total": (
            sum(telemetry_event_count_steps) if telemetry_event_count_steps else None
        ),
        "attribution_phase_elapsed_seconds_total_source": phase_elapsed_source,
        "attribution_phase_elapsed_seconds_total_all_phases": (
            round(sum(phase_elapsed_totals.values()), 6)
            if phase_elapsed_totals
            else None
        ),
        **completion_timing_columns,
        **step_timing_columns,
        **phase_elapsed_columns,
        **flatten_dict(resource_snapshot, prefix="resource_snapshot_"),
    }


def build_row(result_path: Path) -> dict[str, Any]:
    scenario_root = result_path.parent
    result = read_json(result_path)
    scenario_path = scenario_root / "scenario.json"
    scenario = read_json(scenario_path) if scenario_path.exists() else {}
    artifact_dir = Path(result.get("output_dir") or scenario_root / "artifacts")
    profiling = result.get("profiling_summary", {})
    artifact_summary = _summarize_artifacts(artifact_dir)
    special_case = _special_case_label(scenario_root, scenario)
    cache_bytes = scenario.get("cross_batch_decoder_cache_bytes")

    row = {
        "scenario_root": str(scenario_root),
        "scenario_name": result.get("name")
        or scenario.get("name")
        or scenario_root.name,
        "stage": result.get("stage") or scenario.get("stage"),
        "cluster": _infer_cluster(scenario_root, scenario),
        "method": result.get("method") or scenario.get("method"),
        "status": result.get("status"),
        "returncode": result.get("returncode"),
        "duration_seconds": result.get("duration_seconds"),
        "timeout_minutes": result.get("timeout_minutes"),
        "scenario_file": str(scenario_path) if scenario_path.exists() else None,
        "result_file": str(result_path),
        "run_log_path": result.get("log_path") or str(scenario_root / "run.log"),
        "artifacts_dir": str(artifact_dir),
        "attribution_batch_size": scenario.get("attribution_batch_size"),
        "feature_batch_size": scenario.get("feature_batch_size"),
        "logit_batch_size": scenario.get("logit_batch_size"),
        "decoder_chunk_size": scenario.get("decoder_chunk_size"),
        "decoder_cache_bytes": cache_bytes,
        "decoder_cache_gib": None if cache_bytes is None else cache_bytes / (1024**3),
        "max_feature_nodes": scenario.get("max_feature_nodes"),
        "max_edges": scenario.get("max_edges"),
        "max_steps": scenario.get("max_steps"),
        "temperature": scenario.get("temperature"),
        "max_n_logits": scenario.get("max_n_logits"),
        "desired_logit_prob": scenario.get("desired_logit_prob"),
        "attribution_update_interval": scenario.get("attribution_update_interval"),
        "lazy_encoder": None
        if scenario.get("no_lazy_encoder") is None
        else not scenario.get("no_lazy_encoder"),
        "lazy_decoder": None
        if scenario.get("no_lazy_decoder") is None
        else not scenario.get("no_lazy_decoder"),
        "offload_enabled": None
        if scenario.get("no_offload") is None
        else not scenario.get("no_offload"),
        "verbose_attribution": scenario.get("verbose_attribution"),
        "profile_attribution": scenario.get("profile_attribution"),
        "profile_log_interval": scenario.get("profile_log_interval"),
        "save_raw": scenario.get("save_raw"),
        "is_special_case": special_case is not None,
        "special_case_label": special_case,
        **artifact_summary,
        **profiling,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a flat scenario-level index from weekend exact chunked benchmark results"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root benchmark directory containing scenario subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where extracted CSV/JSONL files will be written",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    rows = [build_row(path) for path in sorted(args.input_root.glob("**/result.json"))]

    preferred_headers = [
        "scenario_root",
        "scenario_name",
        "stage",
        "cluster",
        "method",
        "status",
        "returncode",
        "duration_seconds",
        "gsm8k_index",
        "prompt_source",
        "fixture_name",
        "fixture_kind",
        "attribution_batch_size",
        "feature_batch_size",
        "logit_batch_size",
        "decoder_chunk_size",
        "decoder_cache_gib",
        "max_feature_nodes",
        "max_edges",
        "max_steps",
        "prompt_token_count",
        "initial_input_token_count",
        "generated_token_count",
        "n_steps_traced",
        "max_active_features",
        "phase4_feature_batch_size_effective",
        "phase4_feature_batch_planner_status",
        "phase4_feature_batch_planner_skip_reason",
        "telemetry_present",
        "telemetry_file_count",
        "telemetry_event_count_manifest_total",
        "completion_timing_summary_count",
        "completion_timing_completion_end_to_end_seconds_mean",
        "completion_timing_totals_attribution_seconds_total",
        "completion_timing_totals_attribution_seconds_avg_per_step",
        "step_timing_attribution_seconds_mean",
        "attribution_phase_elapsed_seconds_total_all_phases",
        "phase3_duration_seconds",
        "phase4_duration_seconds",
        "phase4_avg_batch_seconds",
        "peak_rss_gib",
        "peak_cuda_reserved_gib",
        "run_log_path",
        "artifacts_dir",
        "is_special_case",
        "special_case_label",
    ]
    write_csv(
        args.output_dir / "benchmark_index.csv",
        rows,
        preferred_headers=preferred_headers,
    )
    write_jsonl(args.output_dir / "benchmark_index.jsonl", rows)
    print(f"Wrote {len(rows)} scenario rows to {args.output_dir}")


if __name__ == "__main__":
    main()
