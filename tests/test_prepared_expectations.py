from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench.full_answer.launch_spec import (
    build_full_answer_launch_spec,
    render_local_full_answer_command,
)
from nlp_research_project.exact_trace_bench.full_answer.prepared_expectations import (
    PreparedExpectationError,
    resolve_prepared_mechanism_expectations,
)


def _write_prepared(tmp_path: Path, *, execution_mode: str = "full") -> Path:
    trajectory_path = tmp_path / "trajectory.json"
    trace_specs_path = tmp_path / "trace_specs.jsonl"
    shards_path = tmp_path / "shards.json"
    trajectory_path.write_text("{}", encoding="utf-8")
    trace_specs_path.write_text("{}\n", encoding="utf-8")
    shards_path.write_text("{}", encoding="utf-8")
    config = {
        "feature_row_influence_mode": "cuda_windowed",
        "feature_row_influence_requirement": "required",
        "backward_engine_mode": "single_forward_batched_vjp",
        "nnsight_session_capacity": 128,
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 128,
        "decoder_active_row_residency": True,
        "decoder_active_row_residency_requirement": "required",
        "decoder_active_row_max_bytes": 0,
        "decoder_active_row_safety_margin_bytes": 16 * 1024**3,
        "diagnostic_stop_mode": "none",
        "runtime_resource_policy": "measure_only",
        "resource_planning_envelope": {"walltime_seconds": 3600},
    }
    launch_spec = build_full_answer_launch_spec(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        output_root=tmp_path / "run",
        shard_selection="0",
        run={"run_id": "repeat-a"},
        specs=[{"trace_id": "trace-a", "graph_knobs": config}],  # type: ignore[list-item]
        planning_envelope={"walltime_seconds": 3600},
        scheduler_request={},
        runtime_resource_policy="measure_only",
        runtime_resource_override_rationale="qualification measurement",
        preheat={"policy": "file_cache", "paths": []},
        workspace={"policy": "immutable_snapshot_required"},
        monitoring={},
    )
    launch = launch_spec.to_record()
    launch_path = tmp_path / "launch_spec.json"
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    prepared_path = tmp_path / "prepared_workload.json"
    prepared_path.write_text(
        json.dumps(
            {
                "launcher": "existing_full_answer_shard_runner",
                "launch_spec_path": str(launch_path),
                "launch_selection_fingerprint": launch["selection_fingerprint"],
                "launch_command": render_local_full_answer_command(launch_spec),
                "launch_output_root": str(tmp_path / "run"),
                "execution_mode": execution_mode,
            }
        ),
        encoding="utf-8",
    )
    return prepared_path


def test_prepared_expectations_freeze_full_batched_capacity_contract(
    tmp_path: Path,
) -> None:
    result = resolve_prepared_mechanism_expectations(_write_prepared(tmp_path))

    assert result == {
        "feature_row_influence_mode": "cuda_windowed",
        "backward_engine_mode": "single_forward_batched_vjp",
        "forward_graph_mode": "single_lane",
        "vjp_kernel_mode": "autograd_batched",
        "forward_lane_count": 1,
        "session_capacity": 128,
        "backward_batch_capacity": 128,
        "phase1_trace_batch_size_max": 128,
        "actual_phase1_forward_trace_width": 1,
        "phase3_batch_size": 128,
        "phase4_batch_size": 128,
        "require_full_completion": True,
        "decoder_active_row_residency": True,
        "decoder_active_row_residency_requirement": "required",
        "decoder_active_row_max_bytes": 0,
        "decoder_active_row_safety_margin_bytes": 16 * 1024**3,
    }


def test_prepared_expectations_reject_execution_mode_tampering(tmp_path: Path) -> None:
    prepared = _write_prepared(tmp_path, execution_mode="phase3-probe")

    with pytest.raises(PreparedExpectationError, match="execution_mode disagrees"):
        resolve_prepared_mechanism_expectations(prepared)
