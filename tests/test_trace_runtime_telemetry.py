from nlp_research_project.exact_trace_bench.trace_runtime import (
    Phase4TelemetryContext,
    TraceStepContext,
    build_step_telemetry_records,
    normalize_telemetry_events,
)


def test_step_telemetry_records_preserve_schema_and_event_override() -> None:
    events = normalize_telemetry_events(
        [None, {"kind": "phase", "phase4_scheduler_mode": "event-value"}]
    )
    phase4 = Phase4TelemetryContext(
        feature_batch_size=8,
        feature_batch_planner_status="planned",
        scheduler_mode_requested="planner_v2",
        scheduler_mode_effective="planner_v2",
        scheduler_version_requested="planner_v2",
        scheduler_version_effective="planner_v2",
        scheduler_policy_requested="bounded_membership_selection",
        scheduler_policy_effective="bounded_membership_selection",
        scheduler_debug=True,
        scheduler_telemetry_detail="debug",
        refresh_optimization_requested="v1",
        refresh_optimization_effective="v1",
        refresh_optimization_version_requested="v1",
        refresh_optimization_version_effective="v1",
        row_executor_requested="streaming_v1",
        row_executor_effective="batched",
        row_executor_version_requested="streaming_v1",
        row_executor_version_effective="streaming_v1",
        row_reduction_requested="gpu_v1",
        row_reduction_effective="gpu_v1",
        row_reduction_version_requested="gpu_v1_staged",
        row_reduction_version_effective="gpu_v1_staged",
    )

    records = build_step_telemetry_records(
        step=TraceStepContext("prompt", "completion", 4),
        phase4=phase4,
        events=events,
    )

    assert len(records) == 1
    record = records[0]
    assert record["prompt_id"] == "prompt"
    assert record["completion_id"] == "completion"
    assert record["trace_step_index"] == 4
    assert record["event_index"] == 0
    assert record["phase4_feature_batch_size"] == 8
    assert record["phase4_scheduler_requested_mode"] == "planner_v2"
    assert record["phase4_scheduler_mode"] == "event-value"
    assert record["phase4_row_executor"] == "batched"
    assert record["phase4_row_reduction_version"] == "gpu_v1_staged"
