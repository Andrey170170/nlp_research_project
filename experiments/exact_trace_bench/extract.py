from __future__ import annotations

import re
from collections import defaultdict
import math
from pathlib import Path
from statistics import mean
from typing import Any

from .config import DEFAULT_EXTRACTED_DIR, DEFAULT_LOGS_DIR, DEFAULT_SCRATCH_ROOT
from .io_utils import (
    ensure_dir,
    flatten_dict,
    parse_memory_value_to_gib,
    read_json,
    safe_stem,
    write_csv,
    write_jsonl,
)


MEMORY_RE = re.compile(
    r"rss=(?P<rss>n/a|\d+(?:\.\d+)?) GiB, "
    r"cuda_alloc=(?P<cuda_alloc>n/a|\d+(?:\.\d+)?) GiB, "
    r"cuda_reserved=(?P<cuda_reserved>n/a|\d+(?:\.\d+)?) GiB, "
    r"cuda_peak_alloc=(?P<cuda_peak_alloc>n/a|\d+(?:\.\d+)?) GiB, "
    r"cuda_peak_reserved=(?P<cuda_peak_reserved>n/a|\d+(?:\.\d+)?) GiB"
)
PHASE0_ENCODE_DONE_RE = re.compile(
    r"TRACE phase0\.encode_sparse\.done \| "
    r"total_active_features=(?P<active_features>\d+), "
    r"elapsed_s=(?P<seconds>\d+(?:\.\d+)?)"
)
PHASE0_RECON_DONE_RE = re.compile(
    r"TRACE phase0\.reconstruction\.done \| total_chunks=(?P<chunks>\d+), "
    r"elapsed_s=(?P<seconds>\d+(?:\.\d+)?)"
)
PHASE3_DONE_RE = re.compile(
    r"(?P<count>\d+) logit attribution\(s\) completed in (?P<seconds>\d+(?:\.\d+)?)s"
)
PHASE4_DONE_RE = re.compile(
    r"Feature attributions completed in (?P<seconds>\d+(?:\.\d+)?)s"
)
ATTRIBUTION_DONE_RE = re.compile(
    r"Attribution completed in (?P<seconds>\d+(?:\.\d+)?)s"
)
PHASE4_BATCH_RE = re.compile(
    r"Phase 4 batch (?P<batch_idx>\d+)/(?P<total_batches>\d+) in "
    r"(?P<seconds>\d+(?:\.\d+)?)s"
)
CACHE_EVENT_RE = re.compile(
    r"TRACE decoder\.cache\.(?P<event>hit|miss|eviction) \| .*?"
    r"(?P<counter_name>hit_count|miss_count|eviction_count)=(?P<count>\d+)"
    r"(?:, resident_bytes=(?P<resident_bytes>\d+))?"
)
CUDA_OOM_RE = re.compile(
    r"torch\.OutOfMemoryError: CUDA out of memory\. Tried to allocate "
    r"(?P<requested>[0-9.]+\s(?:GiB|MiB|KiB))\. "
    r"GPU 0 has a total capacity of (?P<total>[0-9.]+\s(?:GiB|MiB|KiB)) "
    r"of which (?P<free>[0-9.]+\s(?:GiB|MiB|KiB)) is free\. "
    r"Including non-PyTorch memory, this process has "
    r"(?P<in_use>[0-9.]+\s(?:GiB|MiB|KiB)) memory in use\."
)

JOB_ID_RE = re.compile(r"Job ID: (?P<job_id>\d+)")
ARRAY_TASK_RE = re.compile(r"Array task: (?P<array_task>\S+)")
NODE_RE = re.compile(r"Node: (?P<node>.+)")
CLUSTER_RE = re.compile(r"Cluster: (?P<cluster>.+)")
SCENARIOS_FILE_RE = re.compile(r"Scenarios file: (?P<scenarios_file>.+)")
OUTPUT_ROOT_RE = re.compile(r"Output root: (?P<output_root>.+)")
RUN_ID_RE = re.compile(r"Run ID: (?P<run_id>.+)")
RUN_NAME_RE = re.compile(r"Run name: (?P<run_name>.+)")
RUN_DESCRIPTION_RE = re.compile(r"Run description: (?P<run_description>.+)")
RUN_GOAL_RE = re.compile(r"Run goal: (?P<run_goal>.+)")
WRITING_DIR_RE = re.compile(r"Writing experiment results to (?P<scenario_root>.+)")
RUNNING_SCENARIO_RE = re.compile(r"Running scenario: (?P<scenario_name>.+)")
OOM_KILL_RE = re.compile(r"Detected (?P<count>\d+) oom_kill events")
TIMEOUT_RE = re.compile(r"time limit|timed out", re.IGNORECASE)


def _max_or_none(
    current: float | int | None,
    candidate: float | int | None,
) -> float | int | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


def _infer_cluster(scenario_root: Path, scenario: dict[str, Any]) -> str | None:
    if scenario.get("cluster"):
        return str(scenario["cluster"])
    for part in scenario_root.parts:
        if part in {"ascend", "cardinal"}:
            return part
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


def _load_run_config(artifact_dir: Path) -> dict[str, Any]:
    run_config_path = artifact_dir / "run_config.json"
    if not run_config_path.exists():
        return {}
    return read_json(run_config_path)


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_optional_metadata_value(value: Any) -> Any:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.lower() == "unset":
            return None
        return normalized
    return value


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


def _load_summary_run_metadata(scenario_root: Path) -> dict[str, Any]:
    candidates = [scenario_root / "summary.json", scenario_root.parent / "summary.json"]
    for summary_path in candidates:
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        run_metadata = summary.get("run_metadata")
        if isinstance(run_metadata, dict):
            return run_metadata
        if any(
            key in summary
            for key in (
                "run_id",
                "launch_id",
                "run_name",
                "run_description",
                "run_goal",
            )
        ):
            return {
                "run_id": summary.get("run_id") or summary.get("launch_id"),
                "run_name": summary.get("run_name"),
                "run_description": summary.get("run_description"),
                "run_goal": summary.get("run_goal"),
            }
    return {}


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

    first_manifest = manifests[0] if manifests else {}
    resource_snapshot = first_manifest.get("resource_snapshot")
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
    anomaly_debug_declared_paths: set[str] = set()
    anomaly_debug_existing_paths: set[str] = set()
    anomaly_debug_missing_paths: set[str] = set()
    anomaly_debug_refresh_counts: list[int] = []
    anomaly_debug_record_counts: list[int] = []
    anomaly_debug_statuses: list[str] = []
    anomaly_debug_cutoff_margin_mins: list[float] = []
    anomaly_debug_cutoff_margin_means: list[float] = []
    anomaly_debug_previous_overlap_means: list[float] = []
    anomaly_debug_first_overlap_means: list[float] = []
    anomaly_debug_deterministic_overlap_means: list[float] = []
    anomaly_debug_float64_overlap_means: list[float] = []
    anomaly_debug_rank_all_zero_refresh_counts: list[int] = []
    anomaly_debug_rank_effectively_all_zero_refresh_counts: list[int] = []
    anomaly_debug_rank_nonzero_count_mins: list[float] = []
    anomaly_debug_rank_nonzero_count_means: list[float] = []
    anomaly_debug_rank_effective_nonzero_count_means: list[float] = []
    anomaly_debug_rank_abs_sum_means: list[float] = []
    anomaly_debug_rank_max_maxes: list[float] = []
    anomaly_debug_refresh_elapsed_ms_means: list[float] = []
    anomaly_debug_first_refresh_float32_effectively_all_zero: list[int] = []
    anomaly_debug_first_refresh_float64_effectively_all_zero: list[int] = []
    anomaly_debug_phase3_logit_row_batch0_abs_sums: list[float] = []
    anomaly_debug_phase3_logit_row_batch0_max_abs: list[float] = []
    anomaly_debug_phase3_logit_row_batch0_nonfinite_counts: list[int] = []
    anomaly_debug_phase3_logit_row_batch0_row_l1_max: list[float] = []
    anomaly_debug_phase3_logit_row_batch0_row_l1_nonfinite_counts: list[int] = []
    anomaly_debug_feature_row_store_read_calls_per_refresh_means: list[float] = []
    anomaly_debug_feature_row_store_read_rows_per_refresh_means: list[float] = []
    cross_cluster_debug_declared_paths: set[str] = set()
    cross_cluster_debug_existing_paths: set[str] = set()
    cross_cluster_debug_missing_paths: set[str] = set()
    cross_cluster_debug_statuses: list[str] = []
    cross_cluster_debug_summary_scopes: list[str] = []
    cross_cluster_debug_phase0_checkpoint_present: list[int] = []
    cross_cluster_debug_phase3_checkpoint_present: list[int] = []
    cross_cluster_debug_checkpoints_declared_paths: set[str] = set()
    cross_cluster_debug_checkpoints_existing_paths: set[str] = set()
    cross_cluster_debug_checkpoints_missing_paths: set[str] = set()
    cross_cluster_debug_checkpoints_statuses: list[str] = []
    cross_cluster_debug_checkpoints_manifest_counts: list[int] = []
    cross_cluster_debug_batches_declared_paths: set[str] = set()
    cross_cluster_debug_batches_existing_paths: set[str] = set()
    cross_cluster_debug_batches_missing_paths: set[str] = set()
    cross_cluster_debug_batches_statuses: list[str] = []
    cross_cluster_debug_batches_manifest_counts: list[int] = []
    exact_trace_internal_dtype_requested_values: list[str] = []
    resolved_dtype_maps: list[dict[str, Any]] = []
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
    completion_phase_elapsed_totals_aggregate: dict[str, float] = {}
    completion_phase_wall_clock_elapsed_totals: dict[str, float] = {}

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

        anomaly_debug_ref = manifest.get("phase4_anomaly_debug_path")
        resolved_anomaly_debug_path: Path | None = None
        declared_anomaly_debug_path: Path | None = None
        if isinstance(anomaly_debug_ref, str) and anomaly_debug_ref.strip():
            declared_anomaly_debug_path = completion_dir / anomaly_debug_ref.strip()
            anomaly_debug_declared_paths.add(
                _relative_to_or_str(declared_anomaly_debug_path, artifact_dir)
            )
            if declared_anomaly_debug_path.exists():
                resolved_anomaly_debug_path = declared_anomaly_debug_path
            else:
                anomaly_debug_missing_paths.add(
                    _relative_to_or_str(declared_anomaly_debug_path, artifact_dir)
                )
        default_anomaly_debug_path = completion_dir / "phase4_anomaly_debug.json"
        if resolved_anomaly_debug_path is None and default_anomaly_debug_path.exists():
            resolved_anomaly_debug_path = default_anomaly_debug_path
        if resolved_anomaly_debug_path is not None:
            relative_path = _relative_to_or_str(
                resolved_anomaly_debug_path, artifact_dir
            )
            anomaly_debug_existing_paths.add(relative_path)
            anomaly_debug_payload = read_json(resolved_anomaly_debug_path)
            refresh_count = _to_int(anomaly_debug_payload.get("refresh_count"))
            if refresh_count is not None:
                anomaly_debug_refresh_counts.append(refresh_count)
            records = anomaly_debug_payload.get("records")
            if isinstance(records, list):
                anomaly_debug_record_counts.append(len(records))
            status = anomaly_debug_payload.get("status")
            if isinstance(status, str):
                anomaly_debug_statuses.append(status)
            summary = anomaly_debug_payload.get("summary")
            if isinstance(summary, dict):
                value = _to_float(summary.get("cutoff_margin_min"))
                if value is not None:
                    anomaly_debug_cutoff_margin_mins.append(value)
                value = _to_float(summary.get("cutoff_margin_mean"))
                if value is not None:
                    anomaly_debug_cutoff_margin_means.append(value)
                value = _to_float(summary.get("overlap_with_previous_mean"))
                if value is not None:
                    anomaly_debug_previous_overlap_means.append(value)
                value = _to_float(summary.get("overlap_with_first_mean"))
                if value is not None:
                    anomaly_debug_first_overlap_means.append(value)
                value = _to_float(summary.get("deterministic_shadow_overlap_mean"))
                if value is not None:
                    anomaly_debug_deterministic_overlap_means.append(value)
                value = _to_float(summary.get("float64_shadow_overlap_mean"))
                if value is not None:
                    anomaly_debug_float64_overlap_means.append(value)
                value = _to_int(summary.get("rank_signal_all_zero_refresh_count"))
                if value is not None:
                    anomaly_debug_rank_all_zero_refresh_counts.append(value)
                value = _to_int(
                    summary.get("rank_signal_effectively_all_zero_refresh_count")
                )
                if value is not None:
                    anomaly_debug_rank_effectively_all_zero_refresh_counts.append(value)
                value = _to_float(summary.get("rank_signal_nonzero_count_min"))
                if value is not None:
                    anomaly_debug_rank_nonzero_count_mins.append(value)
                value = _to_float(summary.get("rank_signal_nonzero_count_mean"))
                if value is not None:
                    anomaly_debug_rank_nonzero_count_means.append(value)
                value = _to_float(
                    summary.get("rank_signal_effective_nonzero_count_mean")
                )
                if value is not None:
                    anomaly_debug_rank_effective_nonzero_count_means.append(value)
                value = _to_float(summary.get("rank_signal_abs_sum_mean"))
                if value is not None:
                    anomaly_debug_rank_abs_sum_means.append(value)
                value = _to_float(summary.get("rank_signal_max_max"))
                if value is not None:
                    anomaly_debug_rank_max_maxes.append(value)
                value = _to_float(summary.get("refresh_elapsed_ms_mean"))
                if value is not None:
                    anomaly_debug_refresh_elapsed_ms_means.append(value)
                value = summary.get("first_refresh_float32_effectively_all_zero")
                if isinstance(value, bool):
                    anomaly_debug_first_refresh_float32_effectively_all_zero.append(
                        int(value)
                    )
                value = summary.get("first_refresh_float64_effectively_all_zero")
                if isinstance(value, bool):
                    anomaly_debug_first_refresh_float64_effectively_all_zero.append(
                        int(value)
                    )
                value = _to_float(summary.get("phase3_logit_row_batch_0_abs_sum"))
                if value is not None:
                    anomaly_debug_phase3_logit_row_batch0_abs_sums.append(value)
                value = _to_float(summary.get("phase3_logit_row_batch_0_max_abs"))
                if value is not None:
                    anomaly_debug_phase3_logit_row_batch0_max_abs.append(value)
                value = _to_int(summary.get("phase3_logit_row_batch_0_nonfinite_count"))
                if value is not None:
                    anomaly_debug_phase3_logit_row_batch0_nonfinite_counts.append(value)
                value = _to_float(summary.get("phase3_logit_row_batch_0_row_l1_max"))
                if value is not None:
                    anomaly_debug_phase3_logit_row_batch0_row_l1_max.append(value)
                value = _to_int(
                    summary.get("phase3_logit_row_batch_0_row_l1_nonfinite_count")
                )
                if value is not None:
                    anomaly_debug_phase3_logit_row_batch0_row_l1_nonfinite_counts.append(
                        value
                    )
                value = _to_float(
                    summary.get("feature_row_store_read_calls_per_refresh_mean")
                )
                if value is not None:
                    anomaly_debug_feature_row_store_read_calls_per_refresh_means.append(
                        value
                    )
                value = _to_float(
                    summary.get("feature_row_store_read_rows_per_refresh_mean")
                )
                if value is not None:
                    anomaly_debug_feature_row_store_read_rows_per_refresh_means.append(
                        value
                    )

        exact_trace_internal_dtype_requested = manifest.get(
            "exact_trace_internal_dtype_requested"
        )
        if isinstance(exact_trace_internal_dtype_requested, str):
            exact_trace_internal_dtype_requested_values.append(
                exact_trace_internal_dtype_requested
            )

        resolved_dtype_map = manifest.get("resolved_dtype_map")
        if isinstance(resolved_dtype_map, dict):
            resolved_dtype_maps.append(resolved_dtype_map)

        cross_cluster_debug_ref = manifest.get("cross_cluster_debug_path")
        resolved_cross_cluster_debug_path: Path | None = None
        declared_cross_cluster_debug_path: Path | None = None
        if isinstance(cross_cluster_debug_ref, str) and cross_cluster_debug_ref.strip():
            declared_cross_cluster_debug_path = (
                completion_dir / cross_cluster_debug_ref.strip()
            )
            cross_cluster_debug_declared_paths.add(
                _relative_to_or_str(declared_cross_cluster_debug_path, artifact_dir)
            )
            if declared_cross_cluster_debug_path.exists():
                resolved_cross_cluster_debug_path = declared_cross_cluster_debug_path
            else:
                cross_cluster_debug_missing_paths.add(
                    _relative_to_or_str(declared_cross_cluster_debug_path, artifact_dir)
                )
        default_cross_cluster_debug_path = (
            completion_dir / "cross_cluster_debug_summary.json"
        )
        if (
            resolved_cross_cluster_debug_path is None
            and default_cross_cluster_debug_path.exists()
        ):
            resolved_cross_cluster_debug_path = default_cross_cluster_debug_path
        if resolved_cross_cluster_debug_path is not None:
            relative_path = _relative_to_or_str(
                resolved_cross_cluster_debug_path,
                artifact_dir,
            )
            cross_cluster_debug_existing_paths.add(relative_path)
            cross_cluster_payload = read_json(resolved_cross_cluster_debug_path)
            status = cross_cluster_payload.get("status")
            if isinstance(status, str):
                cross_cluster_debug_statuses.append(status)
            summary_scope = manifest.get("cross_cluster_debug_summary_scope")
            if isinstance(summary_scope, str):
                cross_cluster_debug_summary_scopes.append(summary_scope)
            checkpoints = cross_cluster_payload.get("checkpoints")
            if isinstance(checkpoints, dict):
                cross_cluster_debug_phase0_checkpoint_present.append(
                    int("phase0_sparse_setup" in checkpoints)
                )
                cross_cluster_debug_phase3_checkpoint_present.append(
                    int("phase3_seed_ranking_pre_phase4" in checkpoints)
                )

        cross_cluster_checkpoints_ref = manifest.get(
            "cross_cluster_debug_checkpoints_path"
        )
        resolved_cross_cluster_checkpoints_path: Path | None = None
        declared_cross_cluster_checkpoints_path: Path | None = None
        if (
            isinstance(cross_cluster_checkpoints_ref, str)
            and cross_cluster_checkpoints_ref.strip()
        ):
            declared_cross_cluster_checkpoints_path = (
                completion_dir / cross_cluster_checkpoints_ref.strip()
            )
            cross_cluster_debug_checkpoints_declared_paths.add(
                _relative_to_or_str(
                    declared_cross_cluster_checkpoints_path, artifact_dir
                )
            )
            if declared_cross_cluster_checkpoints_path.exists():
                resolved_cross_cluster_checkpoints_path = (
                    declared_cross_cluster_checkpoints_path
                )
            else:
                cross_cluster_debug_checkpoints_missing_paths.add(
                    _relative_to_or_str(
                        declared_cross_cluster_checkpoints_path, artifact_dir
                    )
                )
        default_cross_cluster_checkpoints_path = (
            completion_dir / "cross_cluster_debug_checkpoints.jsonl"
        )
        if (
            resolved_cross_cluster_checkpoints_path is None
            and default_cross_cluster_checkpoints_path.exists()
        ):
            resolved_cross_cluster_checkpoints_path = (
                default_cross_cluster_checkpoints_path
            )
        if resolved_cross_cluster_checkpoints_path is not None:
            cross_cluster_debug_checkpoints_existing_paths.add(
                _relative_to_or_str(
                    resolved_cross_cluster_checkpoints_path, artifact_dir
                )
            )

        cross_cluster_batches_ref = manifest.get("cross_cluster_debug_batches_path")
        resolved_cross_cluster_batches_path: Path | None = None
        declared_cross_cluster_batches_path: Path | None = None
        if (
            isinstance(cross_cluster_batches_ref, str)
            and cross_cluster_batches_ref.strip()
        ):
            declared_cross_cluster_batches_path = (
                completion_dir / cross_cluster_batches_ref.strip()
            )
            cross_cluster_debug_batches_declared_paths.add(
                _relative_to_or_str(declared_cross_cluster_batches_path, artifact_dir)
            )
            if declared_cross_cluster_batches_path.exists():
                resolved_cross_cluster_batches_path = (
                    declared_cross_cluster_batches_path
                )
            else:
                cross_cluster_debug_batches_missing_paths.add(
                    _relative_to_or_str(
                        declared_cross_cluster_batches_path, artifact_dir
                    )
                )
        default_cross_cluster_batches_path = (
            completion_dir / "cross_cluster_debug_batches.jsonl"
        )
        if (
            resolved_cross_cluster_batches_path is None
            and default_cross_cluster_batches_path.exists()
        ):
            resolved_cross_cluster_batches_path = default_cross_cluster_batches_path
        if resolved_cross_cluster_batches_path is not None:
            cross_cluster_debug_batches_existing_paths.add(
                _relative_to_or_str(resolved_cross_cluster_batches_path, artifact_dir)
            )

        checkpoints_status = manifest.get("cross_cluster_debug_checkpoints_status")
        if isinstance(checkpoints_status, str):
            cross_cluster_debug_checkpoints_statuses.append(checkpoints_status)
        checkpoints_count = _to_int(
            manifest.get("cross_cluster_debug_checkpoints_count")
        )
        if checkpoints_count is not None:
            cross_cluster_debug_checkpoints_manifest_counts.append(checkpoints_count)

        batches_status = manifest.get("cross_cluster_debug_batches_status")
        if isinstance(batches_status, str):
            cross_cluster_debug_batches_statuses.append(batches_status)
        batches_count = _to_int(manifest.get("cross_cluster_debug_batches_count"))
        if batches_count is not None:
            cross_cluster_debug_batches_manifest_counts.append(batches_count)

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

        completion_phase_elapsed_aggregate = timing_summary.get(
            "attribution_phase_elapsed_seconds_total_aggregate",
            timing_summary.get("attribution_phase_elapsed_seconds_total"),
        )
        if isinstance(completion_phase_elapsed_aggregate, dict):
            for (
                phase_name,
                phase_elapsed_seconds,
            ) in completion_phase_elapsed_aggregate.items():
                if not isinstance(phase_name, str):
                    continue
                phase_elapsed_seconds_float = _to_float(phase_elapsed_seconds)
                if phase_elapsed_seconds_float is None:
                    continue
                completion_phase_elapsed_totals_aggregate[phase_name] = (
                    completion_phase_elapsed_totals_aggregate.get(phase_name, 0.0)
                    + phase_elapsed_seconds_float
                )

        completion_phase_elapsed_wall_clock = timing_summary.get(
            "attribution_phase_wall_clock_elapsed_seconds_total"
        )
        if isinstance(completion_phase_elapsed_wall_clock, dict):
            for (
                phase_name,
                phase_elapsed_seconds,
            ) in completion_phase_elapsed_wall_clock.items():
                if not isinstance(phase_name, str):
                    continue
                phase_elapsed_seconds_float = _to_float(phase_elapsed_seconds)
                if phase_elapsed_seconds_float is None:
                    continue
                completion_phase_wall_clock_elapsed_totals[phase_name] = (
                    completion_phase_wall_clock_elapsed_totals.get(phase_name, 0.0)
                    + phase_elapsed_seconds_float
                )

    step_phase_elapsed_totals_aggregate: dict[str, float] = {}
    step_phase_wall_clock_elapsed_totals: dict[str, float] = {}
    if (
        not completion_phase_elapsed_totals_aggregate
        or not completion_phase_wall_clock_elapsed_totals
    ):
        for step in steps:
            if not completion_phase_elapsed_totals_aggregate:
                step_phase_elapsed_aggregate = step.get(
                    "attribution_phase_elapsed_seconds_aggregate",
                    step.get("attribution_phase_elapsed_seconds"),
                )
                if isinstance(step_phase_elapsed_aggregate, dict):
                    for (
                        phase_name,
                        phase_elapsed_seconds,
                    ) in step_phase_elapsed_aggregate.items():
                        if not isinstance(phase_name, str):
                            continue
                        phase_elapsed_seconds_float = _to_float(phase_elapsed_seconds)
                        if phase_elapsed_seconds_float is None:
                            continue
                        step_phase_elapsed_totals_aggregate[phase_name] = (
                            step_phase_elapsed_totals_aggregate.get(phase_name, 0.0)
                            + phase_elapsed_seconds_float
                        )

            if not completion_phase_wall_clock_elapsed_totals:
                step_phase_elapsed_wall_clock = step.get(
                    "attribution_phase_wall_clock_elapsed_seconds"
                )
                if isinstance(step_phase_elapsed_wall_clock, dict):
                    for (
                        phase_name,
                        phase_elapsed_seconds,
                    ) in step_phase_elapsed_wall_clock.items():
                        if not isinstance(phase_name, str):
                            continue
                        phase_elapsed_seconds_float = _to_float(phase_elapsed_seconds)
                        if phase_elapsed_seconds_float is None:
                            continue
                        step_phase_wall_clock_elapsed_totals[phase_name] = (
                            step_phase_wall_clock_elapsed_totals.get(phase_name, 0.0)
                            + phase_elapsed_seconds_float
                        )

    phase_elapsed_totals_aggregate = (
        completion_phase_elapsed_totals_aggregate
        if completion_phase_elapsed_totals_aggregate
        else step_phase_elapsed_totals_aggregate
    )
    phase_elapsed_source_aggregate = (
        "completion_timing_summary"
        if completion_phase_elapsed_totals_aggregate
        else "step_records"
        if step_phase_elapsed_totals_aggregate
        else None
    )
    phase_elapsed_totals_wall_clock = (
        completion_phase_wall_clock_elapsed_totals
        if completion_phase_wall_clock_elapsed_totals
        else step_phase_wall_clock_elapsed_totals
    )
    phase_elapsed_source_wall_clock = (
        "completion_timing_summary"
        if completion_phase_wall_clock_elapsed_totals
        else "step_records"
        if step_phase_wall_clock_elapsed_totals
        else None
    )

    phase_elapsed_columns_aggregate = {
        (
            "attribution_phase_elapsed_seconds_total_"
            f"{_sanitize_column_fragment(phase_name)}"
        ): round(total_seconds, 6)
        for phase_name, total_seconds in sorted(phase_elapsed_totals_aggregate.items())
    }
    phase_elapsed_columns_wall_clock = {
        (
            "attribution_phase_wall_clock_elapsed_seconds_total_"
            f"{_sanitize_column_fragment(phase_name)}"
        ): round(total_seconds, 6)
        for phase_name, total_seconds in sorted(phase_elapsed_totals_wall_clock.items())
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

    telemetry_existing_paths_sorted = sorted(telemetry_existing_paths)
    telemetry_declared_paths_sorted = sorted(telemetry_declared_paths)
    telemetry_missing_paths_sorted = sorted(telemetry_missing_paths)
    anomaly_debug_existing_paths_sorted = sorted(anomaly_debug_existing_paths)
    anomaly_debug_declared_paths_sorted = sorted(anomaly_debug_declared_paths)
    anomaly_debug_missing_paths_sorted = sorted(anomaly_debug_missing_paths)
    cross_cluster_debug_existing_paths_sorted = sorted(
        cross_cluster_debug_existing_paths
    )
    cross_cluster_debug_declared_paths_sorted = sorted(
        cross_cluster_debug_declared_paths
    )
    cross_cluster_debug_missing_paths_sorted = sorted(cross_cluster_debug_missing_paths)
    cross_cluster_debug_checkpoints_existing_paths_sorted = sorted(
        cross_cluster_debug_checkpoints_existing_paths
    )
    cross_cluster_debug_checkpoints_declared_paths_sorted = sorted(
        cross_cluster_debug_checkpoints_declared_paths
    )
    cross_cluster_debug_checkpoints_missing_paths_sorted = sorted(
        cross_cluster_debug_checkpoints_missing_paths
    )
    cross_cluster_debug_batches_existing_paths_sorted = sorted(
        cross_cluster_debug_batches_existing_paths
    )
    cross_cluster_debug_batches_declared_paths_sorted = sorted(
        cross_cluster_debug_batches_declared_paths
    )
    cross_cluster_debug_batches_missing_paths_sorted = sorted(
        cross_cluster_debug_batches_missing_paths
    )
    resolved_dtype_map_latest = resolved_dtype_maps[-1] if resolved_dtype_maps else None
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
            "initial_input_token_count",
            prompt_meta.get("initial_input_token_count"),
        ),
        "generated_token_count": first_manifest.get("generated_token_count"),
        "completion_duration_seconds": first_manifest.get("duration_seconds"),
        "n_steps_traced": first_manifest.get("n_steps_traced"),
        "max_active_features": max(
            (
                step.get("n_active_features")
                for step in steps
                if step.get("n_active_features") is not None
            ),
            default=None,
        ),
        "max_edges_retained": max(
            (
                step.get("n_edges_retained")
                for step in steps
                if step.get("n_edges_retained") is not None
            ),
            default=None,
        ),
        "decoder_cache_hit_count": max(
            (
                diag.get("decoder_cache_hit_count")
                for diag in diagnostics
                if diag.get("decoder_cache_hit_count") is not None
            ),
            default=None,
        ),
        "decoder_cache_miss_count": max(
            (
                diag.get("decoder_cache_miss_count")
                for diag in diagnostics
                if diag.get("decoder_cache_miss_count") is not None
            ),
            default=None,
        ),
        "decoder_cache_eviction_count": max(
            (
                diag.get("decoder_cache_eviction_count")
                for diag in diagnostics
                if diag.get("decoder_cache_eviction_count") is not None
            ),
            default=None,
        ),
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
        "exact_trace_internal_dtype_requested": (
            exact_trace_internal_dtype_requested_values[-1]
            if exact_trace_internal_dtype_requested_values
            else None
        ),
        "exact_trace_internal_dtype_requested_values_observed": sorted(
            set(exact_trace_internal_dtype_requested_values)
        ),
        "resolved_dtype_map_present": bool(resolved_dtype_map_latest),
        "resolved_dtype_map_feature_row_storage_dtype": (
            resolved_dtype_map_latest.get("feature_row_storage_dtype")
            if isinstance(resolved_dtype_map_latest, dict)
            else None
        ),
        "resolved_dtype_map_row_abs_sum_dtype": (
            resolved_dtype_map_latest.get("row_abs_sum_dtype")
            if isinstance(resolved_dtype_map_latest, dict)
            else None
        ),
        "resolved_dtype_map_influence_compute_dtype": (
            resolved_dtype_map_latest.get("influence_compute_dtype")
            if isinstance(resolved_dtype_map_latest, dict)
            else None
        ),
        "resolved_dtype_map_planner_compute_dtype": (
            resolved_dtype_map_latest.get("planner_compute_dtype")
            if isinstance(resolved_dtype_map_latest, dict)
            else None
        ),
        "resolved_dtype_map_shadow_debug_compute_dtype": (
            resolved_dtype_map_latest.get("shadow_debug_compute_dtype")
            if isinstance(resolved_dtype_map_latest, dict)
            else None
        ),
        "cross_cluster_debug_present": bool(cross_cluster_debug_existing_paths_sorted),
        "cross_cluster_debug_manifest_declared_count": len(
            cross_cluster_debug_declared_paths_sorted
        ),
        "cross_cluster_debug_file_count": len(
            cross_cluster_debug_existing_paths_sorted
        ),
        "cross_cluster_debug_missing_file_count": len(
            cross_cluster_debug_missing_paths_sorted
        ),
        "cross_cluster_debug_path_example": (
            cross_cluster_debug_existing_paths_sorted[0]
            if cross_cluster_debug_existing_paths_sorted
            else cross_cluster_debug_declared_paths_sorted[0]
            if cross_cluster_debug_declared_paths_sorted
            else None
        ),
        "cross_cluster_debug_status": (
            cross_cluster_debug_statuses[-1] if cross_cluster_debug_statuses else None
        ),
        "cross_cluster_debug_summary_scope": (
            cross_cluster_debug_summary_scopes[-1]
            if cross_cluster_debug_summary_scopes
            else None
        ),
        "cross_cluster_debug_summary_scopes_observed": sorted(
            set(cross_cluster_debug_summary_scopes)
        ),
        "cross_cluster_debug_phase0_checkpoint_present_fraction": (
            round(
                mean(
                    [
                        float(value)
                        for value in cross_cluster_debug_phase0_checkpoint_present
                    ]
                ),
                6,
            )
            if cross_cluster_debug_phase0_checkpoint_present
            else None
        ),
        "cross_cluster_debug_phase3_seed_ranking_checkpoint_present_fraction": (
            round(
                mean(
                    [
                        float(value)
                        for value in cross_cluster_debug_phase3_checkpoint_present
                    ]
                ),
                6,
            )
            if cross_cluster_debug_phase3_checkpoint_present
            else None
        ),
        "cross_cluster_debug_checkpoints_present": bool(
            cross_cluster_debug_checkpoints_existing_paths_sorted
        ),
        "cross_cluster_debug_checkpoints_manifest_declared_count": len(
            cross_cluster_debug_checkpoints_declared_paths_sorted
        ),
        "cross_cluster_debug_checkpoints_file_count": len(
            cross_cluster_debug_checkpoints_existing_paths_sorted
        ),
        "cross_cluster_debug_checkpoints_missing_file_count": len(
            cross_cluster_debug_checkpoints_missing_paths_sorted
        ),
        "cross_cluster_debug_checkpoints_path_example": (
            cross_cluster_debug_checkpoints_existing_paths_sorted[0]
            if cross_cluster_debug_checkpoints_existing_paths_sorted
            else cross_cluster_debug_checkpoints_declared_paths_sorted[0]
            if cross_cluster_debug_checkpoints_declared_paths_sorted
            else None
        ),
        "cross_cluster_debug_checkpoints_status": (
            cross_cluster_debug_checkpoints_statuses[-1]
            if cross_cluster_debug_checkpoints_statuses
            else None
        ),
        "cross_cluster_debug_checkpoints_statuses_observed": sorted(
            set(cross_cluster_debug_checkpoints_statuses)
        ),
        "cross_cluster_debug_checkpoints_count": (
            cross_cluster_debug_checkpoints_manifest_counts[-1]
            if cross_cluster_debug_checkpoints_manifest_counts
            else None
        ),
        "cross_cluster_debug_batches_present": bool(
            cross_cluster_debug_batches_existing_paths_sorted
        ),
        "cross_cluster_debug_batches_manifest_declared_count": len(
            cross_cluster_debug_batches_declared_paths_sorted
        ),
        "cross_cluster_debug_batches_file_count": len(
            cross_cluster_debug_batches_existing_paths_sorted
        ),
        "cross_cluster_debug_batches_missing_file_count": len(
            cross_cluster_debug_batches_missing_paths_sorted
        ),
        "cross_cluster_debug_batches_path_example": (
            cross_cluster_debug_batches_existing_paths_sorted[0]
            if cross_cluster_debug_batches_existing_paths_sorted
            else cross_cluster_debug_batches_declared_paths_sorted[0]
            if cross_cluster_debug_batches_declared_paths_sorted
            else None
        ),
        "cross_cluster_debug_batches_status": (
            cross_cluster_debug_batches_statuses[-1]
            if cross_cluster_debug_batches_statuses
            else None
        ),
        "cross_cluster_debug_batches_statuses_observed": sorted(
            set(cross_cluster_debug_batches_statuses)
        ),
        "cross_cluster_debug_batches_count": (
            cross_cluster_debug_batches_manifest_counts[-1]
            if cross_cluster_debug_batches_manifest_counts
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
        "phase4_anomaly_debug_present": bool(anomaly_debug_existing_paths_sorted),
        "phase4_anomaly_debug_manifest_declared_count": len(
            anomaly_debug_declared_paths_sorted
        ),
        "phase4_anomaly_debug_file_count": len(anomaly_debug_existing_paths_sorted),
        "phase4_anomaly_debug_missing_file_count": len(
            anomaly_debug_missing_paths_sorted
        ),
        "phase4_anomaly_debug_path_example": (
            anomaly_debug_existing_paths_sorted[0]
            if anomaly_debug_existing_paths_sorted
            else anomaly_debug_declared_paths_sorted[0]
            if anomaly_debug_declared_paths_sorted
            else None
        ),
        "phase4_anomaly_debug_status": (
            anomaly_debug_statuses[-1] if anomaly_debug_statuses else None
        ),
        "phase4_anomaly_debug_refresh_count_mean": (
            round(mean([float(value) for value in anomaly_debug_refresh_counts]), 6)
            if anomaly_debug_refresh_counts
            else None
        ),
        "phase4_anomaly_debug_refresh_count_max": (
            max(anomaly_debug_refresh_counts) if anomaly_debug_refresh_counts else None
        ),
        "phase4_anomaly_debug_record_count_total": (
            sum(anomaly_debug_record_counts) if anomaly_debug_record_counts else None
        ),
        "phase4_anomaly_debug_cutoff_margin_min_min": (
            min(anomaly_debug_cutoff_margin_mins)
            if anomaly_debug_cutoff_margin_mins
            else None
        ),
        "phase4_anomaly_debug_cutoff_margin_mean_mean": (
            round(mean(anomaly_debug_cutoff_margin_means), 6)
            if anomaly_debug_cutoff_margin_means
            else None
        ),
        "phase4_anomaly_debug_overlap_with_previous_mean": (
            round(mean(anomaly_debug_previous_overlap_means), 6)
            if anomaly_debug_previous_overlap_means
            else None
        ),
        "phase4_anomaly_debug_overlap_with_first_mean": (
            round(mean(anomaly_debug_first_overlap_means), 6)
            if anomaly_debug_first_overlap_means
            else None
        ),
        "phase4_anomaly_debug_deterministic_overlap_mean": (
            round(mean(anomaly_debug_deterministic_overlap_means), 6)
            if anomaly_debug_deterministic_overlap_means
            else None
        ),
        "phase4_anomaly_debug_float64_overlap_mean": (
            round(mean(anomaly_debug_float64_overlap_means), 6)
            if anomaly_debug_float64_overlap_means
            else None
        ),
        "phase4_anomaly_debug_rank_signal_all_zero_refresh_count_mean": (
            round(
                mean(
                    [
                        float(value)
                        for value in anomaly_debug_rank_all_zero_refresh_counts
                    ]
                ),
                6,
            )
            if anomaly_debug_rank_all_zero_refresh_counts
            else None
        ),
        "phase4_anomaly_debug_rank_signal_effectively_all_zero_refresh_count_mean": (
            round(
                mean(
                    [
                        float(value)
                        for value in anomaly_debug_rank_effectively_all_zero_refresh_counts
                    ]
                ),
                6,
            )
            if anomaly_debug_rank_effectively_all_zero_refresh_counts
            else None
        ),
        "phase4_anomaly_debug_rank_signal_nonzero_count_min_mean": (
            round(mean(anomaly_debug_rank_nonzero_count_mins), 6)
            if anomaly_debug_rank_nonzero_count_mins
            else None
        ),
        "phase4_anomaly_debug_rank_signal_nonzero_count_mean_mean": (
            round(mean(anomaly_debug_rank_nonzero_count_means), 6)
            if anomaly_debug_rank_nonzero_count_means
            else None
        ),
        "phase4_anomaly_debug_rank_signal_effective_nonzero_count_mean": (
            round(mean(anomaly_debug_rank_effective_nonzero_count_means), 6)
            if anomaly_debug_rank_effective_nonzero_count_means
            else None
        ),
        "phase4_anomaly_debug_rank_signal_abs_sum_mean_mean": (
            round(mean(anomaly_debug_rank_abs_sum_means), 6)
            if anomaly_debug_rank_abs_sum_means
            else None
        ),
        "phase4_anomaly_debug_rank_signal_max_max": (
            max(anomaly_debug_rank_max_maxes) if anomaly_debug_rank_max_maxes else None
        ),
        "phase4_anomaly_debug_refresh_elapsed_ms_mean": (
            round(mean(anomaly_debug_refresh_elapsed_ms_means), 6)
            if anomaly_debug_refresh_elapsed_ms_means
            else None
        ),
        "phase4_anomaly_debug_first_refresh_float32_effectively_all_zero_fraction": (
            round(
                mean(
                    [
                        float(value)
                        for value in anomaly_debug_first_refresh_float32_effectively_all_zero
                    ]
                ),
                6,
            )
            if anomaly_debug_first_refresh_float32_effectively_all_zero
            else None
        ),
        "phase4_anomaly_debug_first_refresh_float64_effectively_all_zero_fraction": (
            round(
                mean(
                    [
                        float(value)
                        for value in anomaly_debug_first_refresh_float64_effectively_all_zero
                    ]
                ),
                6,
            )
            if anomaly_debug_first_refresh_float64_effectively_all_zero
            else None
        ),
        "phase4_anomaly_debug_phase3_logit_row_batch0_abs_sum_mean": (
            round(mean(anomaly_debug_phase3_logit_row_batch0_abs_sums), 6)
            if anomaly_debug_phase3_logit_row_batch0_abs_sums
            else None
        ),
        "phase4_anomaly_debug_phase3_logit_row_batch0_max_abs_mean": (
            round(mean(anomaly_debug_phase3_logit_row_batch0_max_abs), 6)
            if anomaly_debug_phase3_logit_row_batch0_max_abs
            else None
        ),
        "phase4_anomaly_debug_phase3_logit_row_batch0_nonfinite_count_mean": (
            round(
                mean(
                    [
                        float(v)
                        for v in anomaly_debug_phase3_logit_row_batch0_nonfinite_counts
                    ]
                ),
                6,
            )
            if anomaly_debug_phase3_logit_row_batch0_nonfinite_counts
            else None
        ),
        "phase4_anomaly_debug_phase3_logit_row_batch0_row_l1_max_mean": (
            round(mean(anomaly_debug_phase3_logit_row_batch0_row_l1_max), 6)
            if anomaly_debug_phase3_logit_row_batch0_row_l1_max
            else None
        ),
        "phase4_anomaly_debug_phase3_logit_row_batch0_row_l1_nonfinite_count_mean": (
            round(
                mean(
                    [
                        float(v)
                        for v in anomaly_debug_phase3_logit_row_batch0_row_l1_nonfinite_counts
                    ]
                ),
                6,
            )
            if anomaly_debug_phase3_logit_row_batch0_row_l1_nonfinite_counts
            else None
        ),
        "phase4_anomaly_debug_feature_row_store_read_calls_per_refresh_mean": (
            round(mean(anomaly_debug_feature_row_store_read_calls_per_refresh_means), 6)
            if anomaly_debug_feature_row_store_read_calls_per_refresh_means
            else None
        ),
        "phase4_anomaly_debug_feature_row_store_read_rows_per_refresh_mean": (
            round(mean(anomaly_debug_feature_row_store_read_rows_per_refresh_means), 6)
            if anomaly_debug_feature_row_store_read_rows_per_refresh_means
            else None
        ),
        "attribution_phase_elapsed_seconds_total_source": phase_elapsed_source_aggregate,
        "attribution_phase_elapsed_seconds_total_all_phases": (
            round(sum(phase_elapsed_totals_aggregate.values()), 6)
            if phase_elapsed_totals_aggregate
            else None
        ),
        "attribution_phase_elapsed_seconds_total_aggregate_source": (
            phase_elapsed_source_aggregate
        ),
        "attribution_phase_elapsed_seconds_total_aggregate_all_phases": (
            round(sum(phase_elapsed_totals_aggregate.values()), 6)
            if phase_elapsed_totals_aggregate
            else None
        ),
        "attribution_phase_wall_clock_elapsed_seconds_total_source": (
            phase_elapsed_source_wall_clock
        ),
        "attribution_phase_wall_clock_elapsed_seconds_total_all_phases": (
            round(sum(phase_elapsed_totals_wall_clock.values()), 6)
            if phase_elapsed_totals_wall_clock
            else None
        ),
        **completion_timing_columns,
        **step_timing_columns,
        **phase_elapsed_columns_aggregate,
        **phase_elapsed_columns_wall_clock,
        **flatten_dict(resource_snapshot, prefix="resource_snapshot_"),
    }


def build_benchmark_index_row(result_path: Path) -> dict[str, Any]:
    scenario_root = result_path.parent
    result = read_json(result_path)
    scenario_path = scenario_root / "scenario.json"
    scenario = read_json(scenario_path) if scenario_path.exists() else {}
    artifact_dir = Path(result.get("output_dir") or scenario_root / "artifacts")
    run_config = _load_run_config(artifact_dir)
    summary_run_metadata = _load_summary_run_metadata(scenario_root)
    scenario_run_metadata_raw = scenario.get("run_metadata")
    scenario_run_metadata = (
        scenario_run_metadata_raw if isinstance(scenario_run_metadata_raw, dict) else {}
    )
    result_run_metadata_raw = result.get("run_metadata")
    result_run_metadata = (
        result_run_metadata_raw if isinstance(result_run_metadata_raw, dict) else {}
    )
    run_id = _first_non_null(
        result.get("run_id"),
        result.get("launch_id"),
        result_run_metadata.get("run_id"),
        result_run_metadata.get("launch_id"),
        scenario_run_metadata.get("run_id"),
        scenario_run_metadata.get("launch_id"),
        summary_run_metadata.get("run_id"),
        summary_run_metadata.get("launch_id"),
    )
    run_name = _first_non_null(
        result.get("run_name"),
        result_run_metadata.get("run_name"),
        scenario_run_metadata.get("run_name"),
        summary_run_metadata.get("run_name"),
    )
    run_description = _first_non_null(
        result.get("run_description"),
        result_run_metadata.get("run_description"),
        scenario_run_metadata.get("run_description"),
        summary_run_metadata.get("run_description"),
    )
    run_goal = _first_non_null(
        result.get("run_goal"),
        result_run_metadata.get("run_goal"),
        scenario_run_metadata.get("run_goal"),
        summary_run_metadata.get("run_goal"),
    )

    profiling = result.get("profiling_summary", {})
    cache_bytes = run_config.get(
        "cross_batch_decoder_cache_bytes",
        scenario.get("cross_batch_decoder_cache_bytes"),
    )

    return {
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
        "run_id": run_id,
        "launch_id": run_id,
        "run_name": run_name,
        "run_description": run_description,
        "run_goal": run_goal,
        "attribution_batch_size": run_config.get(
            "attribution_batch_size", scenario.get("attribution_batch_size")
        ),
        "feature_batch_size": run_config.get(
            "feature_batch_size", scenario.get("feature_batch_size")
        ),
        "logit_batch_size": run_config.get(
            "logit_batch_size", scenario.get("logit_batch_size")
        ),
        "decoder_chunk_size": run_config.get(
            "decoder_chunk_size", scenario.get("decoder_chunk_size")
        ),
        "chunked_feature_replay_window": run_config.get(
            "chunked_feature_replay_window",
            scenario.get("chunked_feature_replay_window"),
        ),
        "error_vector_prefetch_lookahead": run_config.get(
            "error_vector_prefetch_lookahead",
            scenario.get("error_vector_prefetch_lookahead"),
        ),
        "stage_encoder_vecs_on_cpu": run_config.get(
            "stage_encoder_vecs_on_cpu", scenario.get("stage_encoder_vecs_on_cpu")
        ),
        "stage_error_vectors_on_cpu": run_config.get(
            "stage_error_vectors_on_cpu",
            scenario.get("stage_error_vectors_on_cpu"),
        ),
        "row_subchunk_size": run_config.get(
            "row_subchunk_size", scenario.get("row_subchunk_size")
        ),
        "plan_feature_batch_size": run_config.get(
            "plan_feature_batch_size", scenario.get("plan_feature_batch_size")
        ),
        "auto_scale_feature_batch_size": run_config.get(
            "auto_scale_feature_batch_size",
            scenario.get("auto_scale_feature_batch_size"),
        ),
        "feature_batch_size_max": run_config.get(
            "feature_batch_size_max", scenario.get("feature_batch_size_max")
        ),
        "feature_batch_target_reserved_fraction": run_config.get(
            "feature_batch_target_reserved_fraction",
            scenario.get("feature_batch_target_reserved_fraction"),
        ),
        "feature_batch_min_free_fraction": run_config.get(
            "feature_batch_min_free_fraction",
            scenario.get("feature_batch_min_free_fraction"),
        ),
        "feature_batch_probe_batches": run_config.get(
            "feature_batch_probe_batches",
            scenario.get("feature_batch_probe_batches"),
        ),
        "phase4_anomaly_debug": run_config.get(
            "phase4_anomaly_debug", scenario.get("phase4_anomaly_debug")
        ),
        "phase4_scheduler_mode": run_config.get(
            "phase4_scheduler_mode", scenario.get("phase4_scheduler_mode")
        ),
        "phase4_scheduler_version": run_config.get("phase4_scheduler_version"),
        "phase4_scheduler_policy": run_config.get("phase4_scheduler_policy"),
        "phase4_scheduler_debug": run_config.get(
            "phase4_scheduler_debug", scenario.get("phase4_scheduler_debug")
        ),
        "phase4_scheduler_telemetry_detail": run_config.get(
            "phase4_scheduler_telemetry_detail",
            scenario.get("phase4_scheduler_telemetry_detail"),
        ),
        "cross_cluster_debug": run_config.get(
            "cross_cluster_debug", scenario.get("cross_cluster_debug")
        ),
        "telemetry_max_events": run_config.get(
            "telemetry_max_events", scenario.get("telemetry_max_events")
        ),
        "exact_trace_internal_dtype": run_config.get(
            "exact_trace_internal_dtype",
            run_config.get(
                "exact_trace_internal_dtype_requested",
                scenario.get("exact_trace_internal_dtype"),
            ),
        ),
        "exact_trace_internal_dtype_contract_supported": run_config.get(
            "exact_trace_internal_dtype_contract_supported",
            not bool(run_config.get("save_raw")),
        ),
        "decoder_cache_bytes": cache_bytes,
        "decoder_cache_gib": None if cache_bytes is None else cache_bytes / (1024**3),
        "max_feature_nodes": scenario.get("max_feature_nodes"),
        "max_edges": scenario.get("max_edges"),
        "max_steps": scenario.get("max_steps"),
        **_summarize_artifacts(artifact_dir),
        **profiling,
    }


def extract_benchmark_index(input_root: Path) -> list[dict[str, Any]]:
    return [
        build_benchmark_index_row(path)
        for path in sorted(input_root.glob("**/result.json"))
    ]


def _guess_failure_stage(
    summary: dict[str, Any], result_status: str | None
) -> str | None:
    if result_status == "success":
        return None
    if summary.get("cuda_oom_requested_gib") is not None:
        if summary.get("phase0_encode_total_active_features") is None:
            return "phase0_encode"
        if summary.get("phase0_reconstruction_seconds") is None:
            return "phase0_reconstruction"
        if summary.get("phase3_logit_attribution_seconds") is None:
            return "phase3"
        if summary.get("phase4_feature_attribution_seconds") is None:
            return "phase4"
    if (
        summary.get("phase4_batches_observed", 0) > 0
        and summary.get("phase4_feature_attribution_seconds") is None
    ):
        return "phase4"
    if (
        summary.get("phase3_logit_attribution_seconds") is None
        and summary.get("phase0_reconstruction_seconds") is not None
    ):
        return "phase3"
    if (
        summary.get("phase0_reconstruction_seconds") is None
        and summary.get("phase0_encode_total_active_features") is not None
    ):
        return "phase0_reconstruction"
    if summary.get("phase0_encode_total_active_features") is None:
        return "phase0_encode"
    return "unknown"


def parse_run_log_summary(
    *,
    scenario_root: Path,
    scenario_name: str,
    stage: str | None,
    cluster: str | None,
    result_status: str | None,
    log_path: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scenario_root": str(scenario_root),
        "scenario_name": scenario_name,
        "stage": stage,
        "cluster": cluster,
        "result_status": result_status,
        "log_path": str(log_path),
        "phase0_encode_seconds": None,
        "phase0_encode_total_active_features": None,
        "phase0_reconstruction_seconds": None,
        "phase0_reconstruction_total_chunks": None,
        "phase3_logit_count": None,
        "phase3_logit_attribution_seconds": None,
        "phase4_feature_attribution_seconds": None,
        "attribution_total_seconds": None,
        "phase4_batches_observed": 0,
        "phase4_batch_seconds_mean": None,
        "phase4_batch_seconds_max": None,
        "log_peak_rss_gib": None,
        "log_peak_cuda_allocated_gib": None,
        "log_peak_cuda_reserved_gib": None,
        "log_peak_cuda_peak_allocated_gib": None,
        "log_peak_cuda_peak_reserved_gib": None,
        "log_max_decoder_cache_hit_count": None,
        "log_max_decoder_cache_miss_count": None,
        "log_max_decoder_cache_eviction_count": None,
        "log_max_decoder_cache_resident_bytes": None,
        "cuda_oom_requested_gib": None,
        "cuda_oom_total_gib": None,
        "cuda_oom_free_gib": None,
        "cuda_oom_in_use_gib": None,
    }

    phase4_batch_seconds: list[float] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            match = PHASE0_ENCODE_DONE_RE.search(stripped)
            if match:
                summary["phase0_encode_total_active_features"] = int(
                    match.group("active_features")
                )
                summary["phase0_encode_seconds"] = float(match.group("seconds"))

            match = PHASE0_RECON_DONE_RE.search(stripped)
            if match:
                summary["phase0_reconstruction_total_chunks"] = int(
                    match.group("chunks")
                )
                summary["phase0_reconstruction_seconds"] = float(match.group("seconds"))

            match = PHASE3_DONE_RE.search(stripped)
            if match:
                summary["phase3_logit_count"] = int(match.group("count"))
                summary["phase3_logit_attribution_seconds"] = float(
                    match.group("seconds")
                )

            match = PHASE4_DONE_RE.search(stripped)
            if match:
                summary["phase4_feature_attribution_seconds"] = float(
                    match.group("seconds")
                )

            match = ATTRIBUTION_DONE_RE.search(stripped)
            if match:
                summary["attribution_total_seconds"] = float(match.group("seconds"))

            match = PHASE4_BATCH_RE.search(stripped)
            if match:
                phase4_batch_seconds.append(float(match.group("seconds")))

            match = CACHE_EVENT_RE.search(stripped)
            if match:
                event = match.group("event")
                count = int(match.group("count"))
                if event == "hit":
                    summary["log_max_decoder_cache_hit_count"] = _max_or_none(
                        summary["log_max_decoder_cache_hit_count"],
                        count,
                    )
                elif event == "miss":
                    summary["log_max_decoder_cache_miss_count"] = _max_or_none(
                        summary["log_max_decoder_cache_miss_count"],
                        count,
                    )
                elif event == "eviction":
                    summary["log_max_decoder_cache_eviction_count"] = _max_or_none(
                        summary["log_max_decoder_cache_eviction_count"],
                        count,
                    )

                resident_bytes = match.group("resident_bytes")
                if resident_bytes is not None:
                    summary["log_max_decoder_cache_resident_bytes"] = _max_or_none(
                        summary["log_max_decoder_cache_resident_bytes"],
                        int(resident_bytes),
                    )

            match = MEMORY_RE.search(stripped)
            if match:
                summary["log_peak_rss_gib"] = _max_or_none(
                    summary["log_peak_rss_gib"],
                    parse_memory_value_to_gib(match.group("rss")),
                )
                summary["log_peak_cuda_allocated_gib"] = _max_or_none(
                    summary["log_peak_cuda_allocated_gib"],
                    parse_memory_value_to_gib(match.group("cuda_alloc")),
                )
                summary["log_peak_cuda_reserved_gib"] = _max_or_none(
                    summary["log_peak_cuda_reserved_gib"],
                    parse_memory_value_to_gib(match.group("cuda_reserved")),
                )
                summary["log_peak_cuda_peak_allocated_gib"] = _max_or_none(
                    summary["log_peak_cuda_peak_allocated_gib"],
                    parse_memory_value_to_gib(match.group("cuda_peak_alloc")),
                )
                summary["log_peak_cuda_peak_reserved_gib"] = _max_or_none(
                    summary["log_peak_cuda_peak_reserved_gib"],
                    parse_memory_value_to_gib(match.group("cuda_peak_reserved")),
                )

            match = CUDA_OOM_RE.search(stripped)
            if match:
                summary["cuda_oom_requested_gib"] = parse_memory_value_to_gib(
                    match.group("requested")
                )
                summary["cuda_oom_total_gib"] = parse_memory_value_to_gib(
                    match.group("total")
                )
                summary["cuda_oom_free_gib"] = parse_memory_value_to_gib(
                    match.group("free")
                )
                summary["cuda_oom_in_use_gib"] = parse_memory_value_to_gib(
                    match.group("in_use")
                )

    if phase4_batch_seconds:
        summary["phase4_batches_observed"] = len(phase4_batch_seconds)
        summary["phase4_batch_seconds_mean"] = mean(phase4_batch_seconds)
        summary["phase4_batch_seconds_max"] = max(phase4_batch_seconds)

    summary["failure_stage_guess"] = _guess_failure_stage(summary, result_status)
    return summary


def extract_runlog_summaries(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(input_root.glob("**/result.json")):
        scenario_root = result_path.parent
        result = read_json(result_path)
        scenario_path = scenario_root / "scenario.json"
        scenario = read_json(scenario_path) if scenario_path.exists() else {}
        log_path = Path(result.get("log_path") or scenario_root / "run.log")
        if not log_path.exists():
            continue
        rows.append(
            parse_run_log_summary(
                scenario_root=scenario_root,
                scenario_name=result.get("name")
                or scenario.get("name")
                or scenario_root.name,
                stage=result.get("stage") or scenario.get("stage"),
                cluster=_infer_cluster(scenario_root, scenario),
                result_status=result.get("status"),
                log_path=log_path,
            )
        )
    return rows


def _classify_err_text(text: str) -> tuple[str | None, int | None]:
    oom_kill_match = OOM_KILL_RE.search(text)
    if oom_kill_match:
        return "ram_oom", int(oom_kill_match.group("count"))
    if "Exceeded step memory limit" in text:
        return "ram_oom", None
    if "OOM Killed" in text or "oom_kill" in text:
        return "ram_oom", None
    if "torch.OutOfMemoryError: CUDA out of memory" in text:
        return "cuda_oom", None
    if TIMEOUT_RE.search(text):
        return "timeout", None
    if text.strip():
        return "other_error", None
    return None, None


def _parse_out_metadata(out_path: Path) -> dict[str, Any]:
    if not out_path.exists():
        return {}
    metadata: dict[str, Any] = {"out_file": str(out_path)}
    line_map = [
        (JOB_ID_RE, "job_id"),
        (ARRAY_TASK_RE, "array_task"),
        (NODE_RE, "node"),
        (CLUSTER_RE, "cluster"),
        (SCENARIOS_FILE_RE, "scenarios_file"),
        (OUTPUT_ROOT_RE, "output_root"),
        (RUN_ID_RE, "run_id"),
        (RUN_NAME_RE, "run_name"),
        (RUN_DESCRIPTION_RE, "run_description"),
        (RUN_GOAL_RE, "run_goal"),
        (WRITING_DIR_RE, "scenario_root"),
        (RUNNING_SCENARIO_RE, "scenario_name"),
    ]
    for raw_line in out_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for regex, key in line_map:
            match = regex.search(line)
            if match:
                metadata[key] = _normalize_optional_metadata_value(match.group(key))
    return metadata


def build_slurm_row(err_path: Path, benchmark_root: Path) -> dict[str, Any]:
    err_text = err_path.read_text(encoding="utf-8", errors="replace")
    failure_family, oom_kill_count = _classify_err_text(err_text)
    out_path = err_path.with_suffix(".out")
    out_metadata = _parse_out_metadata(out_path)
    scenario_root = out_metadata.get("scenario_root")
    result_path = Path(scenario_root) / "result.json" if scenario_root else None
    result_status = None
    if result_path is not None and result_path.exists():
        result_status = read_json(result_path).get("status")

    return {
        "err_file": str(err_path),
        "out_file": out_metadata.get("out_file"),
        "slurm_stem": safe_stem(err_path),
        "job_id": out_metadata.get("job_id"),
        "array_task": out_metadata.get("array_task"),
        "node": out_metadata.get("node"),
        "cluster": out_metadata.get("cluster"),
        "scenarios_file": out_metadata.get("scenarios_file"),
        "output_root": out_metadata.get("output_root"),
        "run_id": out_metadata.get("run_id"),
        "launch_id": out_metadata.get("run_id"),
        "run_name": out_metadata.get("run_name"),
        "run_description": out_metadata.get("run_description"),
        "run_goal": out_metadata.get("run_goal"),
        "scenario_root": scenario_root,
        "scenario_name": out_metadata.get("scenario_name"),
        "failure_family": failure_family,
        "oom_kill_event_count": oom_kill_count,
        "result_json_exists": result_path.exists()
        if result_path is not None
        else False,
        "result_status": result_status,
        "matches_benchmark_root": bool(
            scenario_root and str(scenario_root).startswith(str(benchmark_root))
        ),
        "err_excerpt": err_text.strip()[:500] if err_text.strip() else None,
    }


def extract_slurm_err_summary(
    logs_dir: Path, benchmark_root: Path
) -> list[dict[str, Any]]:
    return [
        build_slurm_row(path, benchmark_root) for path in sorted(logs_dir.glob("*.err"))
    ]


def _to_float(value: Any) -> float | None:
    if value in {None, "", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in {None, "", "None"}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def build_slurm_lookup(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scenario_root = row.get("scenario_root") or ""
        if scenario_root and row.get("matches_benchmark_root"):
            by_root[scenario_root].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        families = sorted(
            {row.get("failure_family") for row in group if row.get("failure_family")}
        )
        return {
            "slurm_err_file_count": len(group),
            "slurm_failure_families": "|".join(families) if families else None,
            "slurm_any_ram_oom": any(
                row.get("failure_family") == "ram_oom" for row in group
            ),
            "slurm_any_cuda_oom": any(
                row.get("failure_family") == "cuda_oom" for row in group
            ),
            "slurm_any_timeout": any(
                row.get("failure_family") == "timeout" for row in group
            ),
            "slurm_oom_kill_event_count": sum(
                _to_int(row.get("oom_kill_event_count")) or 0 for row in group
            ),
            "slurm_err_excerpt": next(
                (row.get("err_excerpt") for row in group if row.get("err_excerpt")),
                None,
            ),
        }

    return {key: summarize(group) for key, group in by_root.items()}


def merge_benchmark_tables(
    benchmark_rows: list[dict[str, Any]],
    runlog_rows: list[dict[str, Any]],
    slurm_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runlog_lookup = {
        row["scenario_root"]: row for row in runlog_rows if row.get("scenario_root")
    }
    slurm_by_root = build_slurm_lookup(slurm_rows)

    merged_rows: list[dict[str, Any]] = []
    for row in benchmark_rows:
        merged = dict(row)
        scenario_root = row.get("scenario_root") or ""
        runlog = runlog_lookup.get(scenario_root, {})
        slurm = slurm_by_root.get(scenario_root, {})

        for key, value in runlog.items():
            if key in {
                "scenario_root",
                "scenario_name",
                "stage",
                "cluster",
                "result_status",
            }:
                continue
            merged[key] = value
        merged.update(slurm)

        max_active_features = _to_float(merged.get("max_active_features"))
        duration_seconds = _to_float(merged.get("duration_seconds"))
        merged["runtime_per_million_active_features"] = (
            None
            if duration_seconds is None or max_active_features in {None, 0.0}
            else duration_seconds / (max_active_features / 1_000_000.0)
        )

        status = merged.get("status")
        slurm_any_ram_oom = _to_bool(merged.get("slurm_any_ram_oom"))
        slurm_any_timeout = _to_bool(merged.get("slurm_any_timeout"))
        cuda_oom_seen = _to_float(merged.get("cuda_oom_requested_gib")) is not None
        if status == "success":
            failure_family = "success"
        elif slurm_any_ram_oom:
            failure_family = "ram_oom"
        elif status == "oom" or cuda_oom_seen:
            failure_family = "cuda_oom"
        elif status == "timeout" or slurm_any_timeout:
            failure_family = "timeout"
        else:
            failure_family = "other_fail"
        merged["failure_family_final"] = failure_family
        merged_rows.append(merged)

    return merged_rows


def run_full_extraction(
    *,
    input_root: Path = DEFAULT_SCRATCH_ROOT,
    output_dir: Path = DEFAULT_EXTRACTED_DIR,
    logs_dir: Path | None = DEFAULT_LOGS_DIR,
) -> dict[str, int]:
    ensure_dir(output_dir)

    benchmark_rows = extract_benchmark_index(input_root)
    runlog_rows = extract_runlog_summaries(input_root)
    slurm_rows = extract_slurm_err_summary(logs_dir, input_root) if logs_dir else []
    merged_rows = merge_benchmark_tables(benchmark_rows, runlog_rows, slurm_rows)

    write_csv(output_dir / "benchmark_index.csv", benchmark_rows)
    write_jsonl(output_dir / "benchmark_index.jsonl", benchmark_rows)
    write_csv(output_dir / "runlog_summary.csv", runlog_rows)
    if logs_dir is not None:
        write_csv(output_dir / "slurm_err_summary.csv", slurm_rows)
    write_csv(output_dir / "benchmark_enriched.csv", merged_rows)
    write_jsonl(output_dir / "benchmark_enriched.jsonl", merged_rows)

    return {
        "benchmark_rows": len(benchmark_rows),
        "runlog_rows": len(runlog_rows),
        "slurm_rows": len(slurm_rows),
        "merged_rows": len(merged_rows),
    }
