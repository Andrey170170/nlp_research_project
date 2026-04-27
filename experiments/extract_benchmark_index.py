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
    phase0_donor_bundle_declared_paths: set[str] = set()
    phase0_donor_bundle_existing_paths: set[str] = set()
    phase0_donor_bundle_missing_paths: set[str] = set()
    phase0_donor_bundle_statuses: list[str] = []
    phase0_donor_bundle_capture_enabled_values: list[int] = []
    phase0_replay_modes: list[str] = []
    phase0_replay_statuses: list[str] = []
    phase0_replay_donor_context_policies: list[str] = []
    phase0_replay_donor_bundle_paths: list[str] = []
    phase0_replay_validation_warning_counts_latest: list[int] = []
    phase0_replay_validation_warning_counts_max: list[int] = []
    phase0_replay_dtype_roundtrip_losses_latest: list[bool] = []
    phase0_replay_dtype_roundtrip_losses_any: list[bool] = []
    phase3_seed_bundle_declared_paths: set[str] = set()
    phase3_seed_bundle_existing_paths: set[str] = set()
    phase3_seed_bundle_missing_paths: set[str] = set()
    phase3_seed_bundle_statuses: list[str] = []
    phase3_seed_bundle_capture_enabled_values: list[int] = []
    feature_semantic_descriptor_declared_paths: set[str] = set()
    feature_semantic_descriptor_existing_paths: set[str] = set()
    feature_semantic_descriptor_missing_paths: set[str] = set()
    feature_semantic_descriptor_statuses: list[str] = []
    feature_semantic_descriptor_capture_enabled_values: list[int] = []
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

        phase0_donor_bundle_capture_enabled = manifest.get(
            "phase0_donor_bundle_capture_enabled"
        )
        if isinstance(phase0_donor_bundle_capture_enabled, bool):
            phase0_donor_bundle_capture_enabled_values.append(
                int(phase0_donor_bundle_capture_enabled)
            )
        manifest_phase0_donor_bundle_status = manifest.get("phase0_donor_bundle_status")
        if isinstance(manifest_phase0_donor_bundle_status, str):
            phase0_donor_bundle_statuses.append(manifest_phase0_donor_bundle_status)
        manifest_phase0_donor_bundle_statuses = manifest.get(
            "phase0_donor_bundle_statuses_observed"
        )
        if isinstance(manifest_phase0_donor_bundle_statuses, list):
            phase0_donor_bundle_statuses.extend(
                status
                for status in manifest_phase0_donor_bundle_statuses
                if isinstance(status, str)
            )

        manifest_phase0_replay_mode = manifest.get("phase0_replay_mode")
        if isinstance(manifest_phase0_replay_mode, str):
            phase0_replay_modes.append(manifest_phase0_replay_mode)
        manifest_phase0_replay_status = manifest.get("phase0_replay_status")
        if isinstance(manifest_phase0_replay_status, str):
            phase0_replay_statuses.append(manifest_phase0_replay_status)
        manifest_phase0_replay_statuses = manifest.get(
            "phase0_replay_statuses_observed"
        )
        if isinstance(manifest_phase0_replay_statuses, list):
            phase0_replay_statuses.extend(
                status
                for status in manifest_phase0_replay_statuses
                if isinstance(status, str)
            )
        manifest_phase0_replay_context_policy = manifest.get(
            "phase0_donor_context_policy"
        )
        if isinstance(manifest_phase0_replay_context_policy, str):
            phase0_replay_donor_context_policies.append(
                manifest_phase0_replay_context_policy
            )
        manifest_phase0_replay_donor_bundle_path = manifest.get("phase0_donor_bundle")
        if (
            isinstance(manifest_phase0_replay_donor_bundle_path, str)
            and manifest_phase0_replay_donor_bundle_path.strip()
        ):
            phase0_replay_donor_bundle_paths.append(
                manifest_phase0_replay_donor_bundle_path.strip()
            )
        replay_warning_count = _to_int(
            manifest.get("phase0_replay_validation_warning_count")
        )
        if replay_warning_count is not None:
            phase0_replay_validation_warning_counts_latest.append(replay_warning_count)
        replay_warning_count_max = _to_int(
            manifest.get("phase0_replay_validation_warning_count_max")
        )
        if replay_warning_count_max is not None:
            phase0_replay_validation_warning_counts_max.append(replay_warning_count_max)
        replay_dtype_roundtrip_loss = manifest.get("phase0_replay_dtype_roundtrip_loss")
        if isinstance(replay_dtype_roundtrip_loss, bool):
            phase0_replay_dtype_roundtrip_losses_latest.append(
                replay_dtype_roundtrip_loss
            )
        replay_any_dtype_roundtrip_loss = manifest.get(
            "phase0_replay_any_dtype_roundtrip_loss"
        )
        if isinstance(replay_any_dtype_roundtrip_loss, bool):
            phase0_replay_dtype_roundtrip_losses_any.append(
                replay_any_dtype_roundtrip_loss
            )

        phase3_seed_bundle_capture_enabled = manifest.get(
            "phase3_seed_bundle_capture_enabled"
        )
        if isinstance(phase3_seed_bundle_capture_enabled, bool):
            phase3_seed_bundle_capture_enabled_values.append(
                int(phase3_seed_bundle_capture_enabled)
            )
        manifest_phase3_seed_bundle_status = manifest.get("phase3_seed_bundle_status")
        if isinstance(manifest_phase3_seed_bundle_status, str):
            phase3_seed_bundle_statuses.append(manifest_phase3_seed_bundle_status)
        manifest_phase3_seed_bundle_statuses = manifest.get(
            "phase3_seed_bundle_statuses_observed"
        )
        if isinstance(manifest_phase3_seed_bundle_statuses, list):
            phase3_seed_bundle_statuses.extend(
                status
                for status in manifest_phase3_seed_bundle_statuses
                if isinstance(status, str)
            )
        feature_semantic_descriptor_capture_enabled = manifest.get(
            "feature_semantic_descriptor_capture_enabled"
        )
        if isinstance(feature_semantic_descriptor_capture_enabled, bool):
            feature_semantic_descriptor_capture_enabled_values.append(
                int(feature_semantic_descriptor_capture_enabled)
            )
        manifest_feature_semantic_descriptor_status = manifest.get(
            "feature_semantic_descriptor_status"
        )
        if isinstance(manifest_feature_semantic_descriptor_status, str):
            feature_semantic_descriptor_statuses.append(
                manifest_feature_semantic_descriptor_status
            )
        manifest_feature_semantic_descriptor_statuses = manifest.get(
            "feature_semantic_descriptor_statuses_observed"
        )
        if isinstance(manifest_feature_semantic_descriptor_statuses, list):
            feature_semantic_descriptor_statuses.extend(
                status
                for status in manifest_feature_semantic_descriptor_statuses
                if isinstance(status, str)
            )

        manifest_steps = manifest.get("steps")
        if isinstance(manifest_steps, list):
            for step in manifest_steps:
                if not isinstance(step, dict):
                    continue
                phase0_donor_bundle_ref = step.get("phase0_donor_bundle_path")
                if (
                    isinstance(phase0_donor_bundle_ref, str)
                    and phase0_donor_bundle_ref.strip()
                ):
                    declared_phase0_donor_bundle_path = (
                        completion_dir / phase0_donor_bundle_ref.strip()
                    )
                    relative_path = _relative_to_or_str(
                        declared_phase0_donor_bundle_path,
                        artifact_dir,
                    )
                    phase0_donor_bundle_declared_paths.add(relative_path)
                    if declared_phase0_donor_bundle_path.exists():
                        phase0_donor_bundle_existing_paths.add(relative_path)
                    else:
                        phase0_donor_bundle_missing_paths.add(relative_path)
                step_phase0_donor_bundle_status = step.get("phase0_donor_bundle_status")
                if isinstance(step_phase0_donor_bundle_status, str):
                    phase0_donor_bundle_statuses.append(step_phase0_donor_bundle_status)
                step_phase0_donor_bundle_enabled = step.get(
                    "phase0_donor_bundle_capture_enabled"
                )
                if isinstance(step_phase0_donor_bundle_enabled, bool):
                    phase0_donor_bundle_capture_enabled_values.append(
                        int(step_phase0_donor_bundle_enabled)
                    )
                step_phase0_replay_mode = step.get("phase0_replay_mode")
                if isinstance(step_phase0_replay_mode, str):
                    phase0_replay_modes.append(step_phase0_replay_mode)
                step_phase0_replay_status = step.get("phase0_replay_status")
                if isinstance(step_phase0_replay_status, str):
                    phase0_replay_statuses.append(step_phase0_replay_status)
                step_phase0_replay_context_policy = step.get(
                    "phase0_replay_donor_context_policy"
                )
                if isinstance(step_phase0_replay_context_policy, str):
                    phase0_replay_donor_context_policies.append(
                        step_phase0_replay_context_policy
                    )
                step_phase0_replay_donor_bundle_path = step.get(
                    "phase0_replay_donor_bundle_path"
                )
                if (
                    isinstance(step_phase0_replay_donor_bundle_path, str)
                    and step_phase0_replay_donor_bundle_path.strip()
                ):
                    phase0_replay_donor_bundle_paths.append(
                        step_phase0_replay_donor_bundle_path.strip()
                    )
                step_replay_warning_count = _to_int(
                    step.get("phase0_replay_validation_warning_count")
                )
                if step_replay_warning_count is not None:
                    phase0_replay_validation_warning_counts_latest.append(
                        step_replay_warning_count
                    )
                step_replay_dtype_roundtrip_loss = step.get(
                    "phase0_replay_dtype_roundtrip_loss"
                )
                if isinstance(step_replay_dtype_roundtrip_loss, bool):
                    phase0_replay_dtype_roundtrip_losses_latest.append(
                        step_replay_dtype_roundtrip_loss
                    )
                phase3_seed_bundle_ref = step.get("phase3_seed_bundle_path")
                if (
                    isinstance(phase3_seed_bundle_ref, str)
                    and phase3_seed_bundle_ref.strip()
                ):
                    declared_phase3_seed_bundle_path = (
                        completion_dir / phase3_seed_bundle_ref.strip()
                    )
                    relative_path = _relative_to_or_str(
                        declared_phase3_seed_bundle_path,
                        artifact_dir,
                    )
                    phase3_seed_bundle_declared_paths.add(relative_path)
                    if declared_phase3_seed_bundle_path.exists():
                        phase3_seed_bundle_existing_paths.add(relative_path)
                    else:
                        phase3_seed_bundle_missing_paths.add(relative_path)
                step_phase3_seed_bundle_status = step.get("phase3_seed_bundle_status")
                if isinstance(step_phase3_seed_bundle_status, str):
                    phase3_seed_bundle_statuses.append(step_phase3_seed_bundle_status)
                step_phase3_seed_bundle_enabled = step.get(
                    "phase3_seed_bundle_capture_enabled"
                )
                if isinstance(step_phase3_seed_bundle_enabled, bool):
                    phase3_seed_bundle_capture_enabled_values.append(
                        int(step_phase3_seed_bundle_enabled)
                    )
                feature_semantic_descriptor_ref = step.get(
                    "feature_semantic_descriptor_path"
                )
                if (
                    isinstance(feature_semantic_descriptor_ref, str)
                    and feature_semantic_descriptor_ref.strip()
                ):
                    declared_feature_semantic_descriptor_path = (
                        completion_dir / feature_semantic_descriptor_ref.strip()
                    )
                    relative_path = _relative_to_or_str(
                        declared_feature_semantic_descriptor_path,
                        artifact_dir,
                    )
                    feature_semantic_descriptor_declared_paths.add(relative_path)
                    if declared_feature_semantic_descriptor_path.exists():
                        feature_semantic_descriptor_existing_paths.add(relative_path)
                    else:
                        feature_semantic_descriptor_missing_paths.add(relative_path)
                step_feature_semantic_descriptor_status = step.get(
                    "feature_semantic_descriptor_status"
                )
                if isinstance(step_feature_semantic_descriptor_status, str):
                    feature_semantic_descriptor_statuses.append(
                        step_feature_semantic_descriptor_status
                    )
                step_feature_semantic_descriptor_enabled = step.get(
                    "feature_semantic_descriptor_capture_enabled"
                )
                if isinstance(step_feature_semantic_descriptor_enabled, bool):
                    feature_semantic_descriptor_capture_enabled_values.append(
                        int(step_feature_semantic_descriptor_enabled)
                    )

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

    resource_snapshot = manifests[0].get("resource_snapshot") if manifests else None
    first_manifest = manifests[0] if manifests else {}
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
    phase0_donor_bundle_existing_paths_sorted = sorted(
        phase0_donor_bundle_existing_paths
    )
    phase0_donor_bundle_declared_paths_sorted = sorted(
        phase0_donor_bundle_declared_paths
    )
    phase0_donor_bundle_missing_paths_sorted = sorted(phase0_donor_bundle_missing_paths)
    phase3_seed_bundle_existing_paths_sorted = sorted(phase3_seed_bundle_existing_paths)
    phase3_seed_bundle_declared_paths_sorted = sorted(phase3_seed_bundle_declared_paths)
    phase3_seed_bundle_missing_paths_sorted = sorted(phase3_seed_bundle_missing_paths)
    feature_semantic_descriptor_existing_paths_sorted = sorted(
        feature_semantic_descriptor_existing_paths
    )
    feature_semantic_descriptor_declared_paths_sorted = sorted(
        feature_semantic_descriptor_declared_paths
    )
    feature_semantic_descriptor_missing_paths_sorted = sorted(
        feature_semantic_descriptor_missing_paths
    )
    resolved_dtype_map_latest = resolved_dtype_maps[-1] if resolved_dtype_maps else None
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
    phase0_replay_validation_warning_count_max_source = (
        phase0_replay_validation_warning_counts_max
        if phase0_replay_validation_warning_counts_max
        else phase0_replay_validation_warning_counts_latest
    )
    phase0_replay_dtype_roundtrip_loss_any_source = (
        phase0_replay_dtype_roundtrip_losses_any
        if phase0_replay_dtype_roundtrip_losses_any
        else phase0_replay_dtype_roundtrip_losses_latest
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
        "phase0_donor_bundle_capture_enabled_fraction": (
            round(
                mean(
                    [
                        float(value)
                        for value in phase0_donor_bundle_capture_enabled_values
                    ]
                ),
                6,
            )
            if phase0_donor_bundle_capture_enabled_values
            else None
        ),
        "phase0_donor_bundle_present": bool(phase0_donor_bundle_existing_paths_sorted),
        "phase0_donor_bundle_manifest_declared_count": len(
            phase0_donor_bundle_declared_paths_sorted
        ),
        "phase0_donor_bundle_file_count": len(
            phase0_donor_bundle_existing_paths_sorted
        ),
        "phase0_donor_bundle_missing_file_count": len(
            phase0_donor_bundle_missing_paths_sorted
        ),
        "phase0_donor_bundle_path_example": (
            phase0_donor_bundle_existing_paths_sorted[0]
            if phase0_donor_bundle_existing_paths_sorted
            else phase0_donor_bundle_declared_paths_sorted[0]
            if phase0_donor_bundle_declared_paths_sorted
            else None
        ),
        "phase0_donor_bundle_status": (
            phase0_donor_bundle_statuses[-1] if phase0_donor_bundle_statuses else None
        ),
        "phase0_donor_bundle_statuses_observed": sorted(
            set(phase0_donor_bundle_statuses)
        ),
        "phase0_replay_mode": (
            phase0_replay_modes[-1] if phase0_replay_modes else None
        ),
        "phase0_replay_modes_observed": sorted(set(phase0_replay_modes)),
        "phase0_replay_status": (
            phase0_replay_statuses[-1] if phase0_replay_statuses else None
        ),
        "phase0_replay_statuses_observed": sorted(set(phase0_replay_statuses)),
        "phase0_replay_donor_context_policy": (
            phase0_replay_donor_context_policies[-1]
            if phase0_replay_donor_context_policies
            else None
        ),
        "phase0_replay_donor_context_policies_observed": sorted(
            set(phase0_replay_donor_context_policies)
        ),
        "phase0_replay_donor_bundle_path": (
            phase0_replay_donor_bundle_paths[-1]
            if phase0_replay_donor_bundle_paths
            else None
        ),
        "phase0_replay_validation_warning_count": (
            phase0_replay_validation_warning_counts_latest[-1]
            if phase0_replay_validation_warning_counts_latest
            else None
        ),
        "phase0_replay_validation_warning_count_max": (
            max(phase0_replay_validation_warning_count_max_source)
            if phase0_replay_validation_warning_count_max_source
            else None
        ),
        "phase0_replay_dtype_roundtrip_loss": (
            phase0_replay_dtype_roundtrip_losses_latest[-1]
            if phase0_replay_dtype_roundtrip_losses_latest
            else None
        ),
        "phase0_replay_any_dtype_roundtrip_loss": (
            bool(any(phase0_replay_dtype_roundtrip_loss_any_source))
            if phase0_replay_dtype_roundtrip_loss_any_source
            else None
        ),
        "phase3_seed_bundle_capture_enabled_fraction": (
            round(
                mean(
                    [
                        float(value)
                        for value in phase3_seed_bundle_capture_enabled_values
                    ]
                ),
                6,
            )
            if phase3_seed_bundle_capture_enabled_values
            else None
        ),
        "phase3_seed_bundle_present": bool(phase3_seed_bundle_existing_paths_sorted),
        "phase3_seed_bundle_manifest_declared_count": len(
            phase3_seed_bundle_declared_paths_sorted
        ),
        "phase3_seed_bundle_file_count": len(phase3_seed_bundle_existing_paths_sorted),
        "phase3_seed_bundle_missing_file_count": len(
            phase3_seed_bundle_missing_paths_sorted
        ),
        "phase3_seed_bundle_path_example": (
            phase3_seed_bundle_existing_paths_sorted[0]
            if phase3_seed_bundle_existing_paths_sorted
            else phase3_seed_bundle_declared_paths_sorted[0]
            if phase3_seed_bundle_declared_paths_sorted
            else None
        ),
        "phase3_seed_bundle_status": (
            phase3_seed_bundle_statuses[-1] if phase3_seed_bundle_statuses else None
        ),
        "phase3_seed_bundle_statuses_observed": sorted(
            set(phase3_seed_bundle_statuses)
        ),
        "feature_semantic_descriptor_capture_enabled_fraction": (
            round(
                mean(
                    [
                        float(value)
                        for value in feature_semantic_descriptor_capture_enabled_values
                    ]
                ),
                6,
            )
            if feature_semantic_descriptor_capture_enabled_values
            else None
        ),
        "feature_semantic_descriptor_present": bool(
            feature_semantic_descriptor_existing_paths_sorted
        ),
        "feature_semantic_descriptor_manifest_declared_count": len(
            feature_semantic_descriptor_declared_paths_sorted
        ),
        "feature_semantic_descriptor_file_count": len(
            feature_semantic_descriptor_existing_paths_sorted
        ),
        "feature_semantic_descriptor_missing_file_count": len(
            feature_semantic_descriptor_missing_paths_sorted
        ),
        "feature_semantic_descriptor_path_example": (
            feature_semantic_descriptor_existing_paths_sorted[0]
            if feature_semantic_descriptor_existing_paths_sorted
            else feature_semantic_descriptor_declared_paths_sorted[0]
            if feature_semantic_descriptor_declared_paths_sorted
            else None
        ),
        "feature_semantic_descriptor_status": (
            feature_semantic_descriptor_statuses[-1]
            if feature_semantic_descriptor_statuses
            else None
        ),
        "feature_semantic_descriptor_statuses_observed": sorted(
            set(feature_semantic_descriptor_statuses)
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


def build_row(result_path: Path) -> dict[str, Any]:
    scenario_root = result_path.parent
    result = read_json(result_path)
    scenario_path = scenario_root / "scenario.json"
    scenario = read_json(scenario_path) if scenario_path.exists() else {}
    artifact_dir = Path(result.get("output_dir") or scenario_root / "artifacts")
    run_config_path = artifact_dir / "run_config.json"
    run_config = read_json(run_config_path) if run_config_path.exists() else {}
    profiling = result.get("profiling_summary", {})
    artifact_summary = _summarize_artifacts(artifact_dir)
    special_case = _special_case_label(scenario_root, scenario)
    cache_bytes = scenario.get("cross_batch_decoder_cache_bytes")
    save_raw = scenario.get("save_raw")
    exact_trace_internal_dtype = run_config.get(
        "exact_trace_internal_dtype",
        run_config.get("exact_trace_internal_dtype_requested"),
    )
    if exact_trace_internal_dtype is None:
        exact_trace_internal_dtype = scenario.get("exact_trace_internal_dtype")
    if exact_trace_internal_dtype is None:
        exact_trace_internal_dtype = scenario.get(
            "exact_trace_internal_dtype_requested"
        )
    phase0_activation_threshold_compare_mode = run_config.get(
        "phase0_activation_threshold_compare_mode"
    )
    if phase0_activation_threshold_compare_mode is None:
        phase0_activation_threshold_compare_mode = scenario.get(
            "phase0_activation_threshold_compare_mode"
        )

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
        "exact_trace_internal_dtype": exact_trace_internal_dtype,
        "phase0_activation_threshold_compare_mode": (
            phase0_activation_threshold_compare_mode
        ),
        "exact_trace_internal_dtype_contract_supported": run_config.get(
            "exact_trace_internal_dtype_contract_supported",
            not bool(save_raw),
        ),
        "cross_cluster_debug": run_config.get(
            "cross_cluster_debug", scenario.get("cross_cluster_debug")
        ),
        "telemetry_max_events": run_config.get(
            "telemetry_max_events", scenario.get("telemetry_max_events")
        ),
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
        "attribution_phase_wall_clock_elapsed_seconds_total_all_phases",
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
