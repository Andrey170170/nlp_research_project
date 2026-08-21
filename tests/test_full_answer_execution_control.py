from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from nlp_research_project.exact_trace_bench.full_answer.execution_control import (
    RuntimeResourcePolicy,
    execution_resolution,
    selected_execution_record,
    validate_resource_policy,
)
from nlp_research_project.exact_trace_bench.full_answer.launch_spec import (
    build_full_answer_launch_spec,
    validate_full_answer_launch_record,
)


def _spec(**knob_overrides: Any) -> Any:
    knobs = {
        "feature_row_influence_mode": "cuda_windowed",
        "feature_row_influence_requirement": "required",
        "backward_engine_mode": "single_forward_batched_vjp",
        "runtime_resource_policy": "measure_only",
        "resource_planning_envelope": {
            "host_memory_stop_gib": 400.0,
            "walltime_seconds": 7200.0,
        },
        **knob_overrides,
    }
    return cast(
        Any,
        {
            "trajectory_id": "trajectory",
            "trace_id": "trace-0",
            "prefix_token_count": 1024,
            "target_token_id": 7,
            "graph_knobs": knobs,
        },
    )


def test_selected_execution_records_selection_and_scheduler_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXACT_TRACE_REQUESTED_MEM", "400G")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "409600")
    spec = _spec()

    record = selected_execution_record(
        trajectory_path=Path("trajectory.json"),
        trace_specs_path=Path("trace_specs.jsonl"),
        shards_path=Path("shards.json"),
        specs=[spec],
        shard={"shard_id": 0, "spec_indices": [0]},
        metadata={"run_id": "run"},
    )

    assert len(record["selection_fingerprint"]) == 64
    assert record["selected_config"] == spec["graph_knobs"]
    assert record["mechanism_selection"]["feature_row_influence_mode"] == (
        "cuda_windowed"
    )
    assert record["mechanism_selection"]["backward_engine_mode"] == (
        "single_forward_batched_vjp"
    )
    assert record["resources"]["runtime_policy"] == "measure_only"
    assert record["resources"]["runtime_enforced_fields"] == []
    assert (
        record["resources"]["scheduler_request"]["EXACT_TRACE_REQUESTED_MEM"] == "400G"
    )
    assert record["resources"]["scheduler_allocation"]["SLURM_MEM_PER_NODE"] == "409600"

    monkeypatch.setenv("SLURM_MEM_PER_NODE", "614400")
    changed_allocation = selected_execution_record(
        trajectory_path=Path("trajectory.json"),
        trace_specs_path=Path("trace_specs.jsonl"),
        shards_path=Path("shards.json"),
        specs=[spec],
        shard={"shard_id": 0, "spec_indices": [0]},
        metadata={"run_id": "run"},
    )
    assert (
        changed_allocation["selection_fingerprint"] == record["selection_fingerprint"]
    )
    assert changed_allocation["record_fingerprint"] != record["record_fingerprint"]


def test_selected_execution_records_mixed_configs_per_trace() -> None:
    first = _spec()
    second = _spec(feature_batch_size=32)
    second["trace_id"] = "trace-1"
    record = selected_execution_record(
        trajectory_path=Path("trajectory.json"),
        trace_specs_path=Path("trace_specs.jsonl"),
        shards_path=Path("shards.json"),
        specs=[first, second],
        shard={"shard_id": 0, "spec_indices": [0, 1]},
        metadata={},
    )

    assert record["configuration_mode"] == "per_trace"
    assert record["selected_config"] is None
    assert len(record["selected_configs"]) == 2
    assert record["mechanism_selection"] is None


def test_execution_resolution_records_requested_to_effective_delta() -> None:
    resolved = execution_resolution(
        selected_config={
            "backward_engine_mode": "single_forward_batched_vjp",
            "feature_row_influence_mode": "cuda_windowed",
            "feature_row_influence_requirement": "required",
            "feature_batch_size": None,
        },
        effective_execution={
            "storage": {
                "feature_row_influence_mode_resolved": "cuda_windowed",
                "feature_row_influence_requirement": "required",
                "feature_row_influence_resolution_reason": ("admitted_cuda_windowed"),
            },
            "batches": {
                "feature_batch_size": 32,
                "backward_engine_mode": "single_forward_batched_vjp",
            },
        },
    )

    by_name = {row["field"]: row for row in resolved["fields"]}
    assert by_name["backward_engine_mode"]["changed"] is False
    assert by_name["feature_row_influence_mode"] == {
        "field": "feature_row_influence_mode",
        "requested": "cuda_windowed",
        "effective": "cuda_windowed",
        "changed": False,
        "resolution_stage": "phase2_row_store",
        "reason": "admitted_cuda_windowed",
        "classification": "physical",
    }
    assert by_name["feature_batch_size"]["requested"] is None
    assert by_name["feature_batch_size"]["effective"] == 32
    assert by_name["feature_batch_size"]["changed"] is True


def test_measure_only_resource_policy_never_stops() -> None:
    validate_resource_policy(
        policy=RuntimeResourcePolicy.MEASURE_ONLY,
        envelope={"host_rss_stop_gib": 1.0, "walltime_seconds": 1.0},
        sample={"process_max_rss_kib": 10 * 1024**2, "elapsed_seconds": 10.0},
    )


def test_enforced_resource_policy_stops_on_supported_limit() -> None:
    with pytest.raises(RuntimeError, match="host_rss_stop_gib"):
        validate_resource_policy(
            policy=RuntimeResourcePolicy.ENFORCE,
            envelope={"host_rss_stop_gib": 1.0},
            sample={"process_max_rss_kib": 2 * 1024**2, "elapsed_seconds": 0.0},
        )


def test_launch_spec_fingerprint_rejects_renderer_side_mutation() -> None:
    spec = _spec()
    launch = build_full_answer_launch_spec(
        trajectory_path=Path("trajectory.json"),
        trace_specs_path=Path("trace_specs.jsonl"),
        shards_path=Path("shards.json"),
        output_root=Path("output"),
        shard_selection="0",
        run={"run_id": "run"},
        specs=[spec],
        planning_envelope=spec["graph_knobs"]["resource_planning_envelope"],
        scheduler_request={"memory": "600G", "walltime": "08:00:00"},
        runtime_resource_policy="measure_only",
        runtime_resource_override_rationale="optimization measurement",
        preheat={"policy": "file_cache", "paths": ["weights"]},
        workspace={"policy": "immutable_snapshot"},
        monitoring={"gpu_sampler": True},
    )
    record = launch.to_record()
    validate_full_answer_launch_record(record)

    record["scheduler_request"]["memory"] = "400G"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_full_answer_launch_record(record)
