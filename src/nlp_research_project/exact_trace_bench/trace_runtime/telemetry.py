"""Construction of stable per-step telemetry and debug-stream records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TraceStepContext:
    """Identity attached to records emitted for one generated trace step."""

    prompt_id: str
    completion_id: str
    step_index: int


@dataclass(frozen=True)
class Phase4TelemetryContext:
    """Resolved Phase-4 settings copied onto every attribution event."""

    feature_batch_size: int
    feature_batch_planner_status: str
    scheduler_mode_requested: str
    scheduler_mode_effective: str
    scheduler_version_requested: str | None
    scheduler_version_effective: str | None
    scheduler_policy_requested: str | None
    scheduler_policy_effective: str | None
    scheduler_debug: bool
    scheduler_telemetry_detail: str
    refresh_optimization_requested: str
    refresh_optimization_effective: str
    refresh_optimization_version_requested: str | None
    refresh_optimization_version_effective: str | None
    row_executor_requested: str
    row_executor_effective: str
    row_executor_version_requested: str | None
    row_executor_version_effective: str | None
    row_reduction_requested: str
    row_reduction_effective: str
    row_reduction_version_requested: str | None
    row_reduction_version_effective: str | None

    def record_fields(self) -> dict[str, Any]:
        p = asdict(self)
        return {
            "phase4_feature_batch_size": p["feature_batch_size"],
            "phase4_feature_batch_planner_status": p["feature_batch_planner_status"],
            "phase4_scheduler_requested_mode": p["scheduler_mode_requested"],
            "phase4_scheduler_mode": p["scheduler_mode_effective"],
            "phase4_scheduler_mode_requested": p["scheduler_mode_requested"],
            "phase4_scheduler_mode_effective": p["scheduler_mode_effective"],
            "phase4_scheduler_effective_mode": p["scheduler_mode_effective"],
            "phase4_scheduler_version": p["scheduler_version_effective"],
            "phase4_scheduler_version_requested": p["scheduler_version_requested"],
            "phase4_scheduler_version_effective": p["scheduler_version_effective"],
            "phase4_scheduler_effective_version": p["scheduler_version_effective"],
            "phase4_scheduler_policy": p["scheduler_policy_effective"],
            "phase4_scheduler_policy_requested": p["scheduler_policy_requested"],
            "phase4_scheduler_policy_effective": p["scheduler_policy_effective"],
            "phase4_scheduler_effective_policy": p["scheduler_policy_effective"],
            "phase4_scheduler_debug": p["scheduler_debug"],
            "phase4_scheduler_telemetry_detail": p["scheduler_telemetry_detail"],
            "phase4_refresh_optimization": p["refresh_optimization_effective"],
            "phase4_refresh_optimization_requested": p["refresh_optimization_requested"],
            "phase4_refresh_optimization_mode_requested": p["refresh_optimization_requested"],
            "phase4_refresh_optimization_effective": p["refresh_optimization_effective"],
            "phase4_refresh_optimization_mode_effective": p["refresh_optimization_effective"],
            "phase4_refresh_optimization_version": p["refresh_optimization_version_effective"],
            "phase4_refresh_optimization_version_requested": p["refresh_optimization_version_requested"],
            "phase4_refresh_optimization_version_effective": p["refresh_optimization_version_effective"],
            "phase4_row_executor": p["row_executor_effective"],
            "phase4_row_executor_requested": p["row_executor_requested"],
            "phase4_row_executor_mode_requested": p["row_executor_requested"],
            "phase4_row_executor_effective": p["row_executor_effective"],
            "phase4_row_executor_mode_effective": p["row_executor_effective"],
            "phase4_row_executor_version": p["row_executor_version_effective"],
            "phase4_row_executor_version_requested": p["row_executor_version_requested"],
            "phase4_row_executor_version_effective": p["row_executor_version_effective"],
            "phase4_row_reduction": p["row_reduction_effective"],
            "phase4_row_reduction_requested": p["row_reduction_requested"],
            "phase4_row_reduction_mode_requested": p["row_reduction_requested"],
            "phase4_row_reduction_effective": p["row_reduction_effective"],
            "phase4_row_reduction_mode_effective": p["row_reduction_effective"],
            "phase4_row_reduction_version": p["row_reduction_version_effective"],
            "phase4_row_reduction_version_requested": p["row_reduction_version_requested"],
            "phase4_row_reduction_version_effective": p["row_reduction_version_effective"],
        }


def normalize_telemetry_events(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    return [event for event in payload if isinstance(event, dict)]


def build_step_telemetry_records(
    *,
    step: TraceStepContext,
    phase4: Phase4TelemetryContext,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    shared = phase4.record_fields()
    records: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        record: dict[str, Any] = {
            "prompt_id": step.prompt_id,
            "completion_id": step.completion_id,
            "trace_step_index": step.step_index,
            "event_index": event_index,
            **shared,
        }
        record.update(event)
        records.append(record)
    return records


def normalize_cross_cluster_debug_records(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    return [record for record in payload if isinstance(record, dict)]


def build_cross_cluster_debug_records(
    *,
    prompt_id: str,
    completion_id: str,
    step_index: int,
    stream_name: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_records: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        normalized: dict[str, Any] = {
            "prompt_id": prompt_id,
            "completion_id": completion_id,
            "trace_step_index": step_index,
            "stream_name": stream_name,
            "record_index": record_index,
        }
        normalized.update(record)
        normalized_records.append(normalized)
    return normalized_records
