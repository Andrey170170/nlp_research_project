from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_research_project.exact_trace_bench import perf_cli
from nlp_research_project.exact_trace_bench.full_answer.launch_spec import (
    build_full_answer_launch_spec,
)
from nlp_research_project.exact_trace_bench.full_answer.prepared_sequence import (
    PreparedSequenceError,
    build_prepared_sequence,
    load_prepared_sequence,
    run_prepared_sequence,
    write_prepared_sequence,
)


def _bundle(
    tmp_path: Path,
    name: str,
    *,
    width: int,
    output: Path,
    full_completion: bool = False,
) -> Path:
    bundle = tmp_path / name
    bundle.mkdir()
    trajectory = bundle / "trajectory.json"
    specs_path = bundle / "trace_specs.jsonl"
    shards = bundle / "shards.json"
    trajectory.write_text("{}\n")
    shards.write_text("{}\n")
    knobs = {
        "feature_row_influence_mode": "cuda_windowed",
        "feature_row_influence_requirement": "required",
        "backward_engine_mode": "duplicated_lanes",
        "nnsight_session_capacity": width,
        "phase1_trace_batch_size_max": width,
        "phase3_compute_microbatch_max_rows": width,
        "phase4_execution_batch_max_rows": width,
        "diagnostic_stop_mode": "none" if full_completion else "phase3_probe",
        "runtime_resource_policy": "measure_only",
        "resource_planning_envelope": {"walltime_seconds": 3600},
    }
    spec = cast(
        Any,
        {
            "trajectory_id": "trajectory",
            "trace_id": f"trace-{name}",
            "prefix_token_count": 256,
            "target_token_id": 7,
            "graph_knobs": knobs,
        },
    )
    specs_path.write_text(json.dumps(spec) + "\n")
    preheat = {"policy": "file_cache", "paths": ["/cache/model", "/cache/provider"]}
    launch = build_full_answer_launch_spec(
        trajectory_path=trajectory,
        trace_specs_path=specs_path,
        shards_path=shards,
        output_root=output,
        shard_selection="0",
        run={"run_id": name},
        specs=[spec],
        planning_envelope=knobs["resource_planning_envelope"],
        scheduler_request={"partition": "rai-gpu-grn", "gpus_per_task": 1},
        runtime_resource_policy="measure_only",
        runtime_resource_override_rationale="diagnostic measurement",
        preheat=preheat,
        workspace={"policy": "immutable_snapshot_required"},
        monitoring={"postrun_mechanism_validation": True},
    )
    launch_path = bundle / "launch_spec.json"
    launch_path.write_text(json.dumps(launch.to_record()))
    prepared_path = bundle / "prepared_workload.json"
    command = [
        "uv",
        "run",
        "exact-trace-bench",
        "run-full-answer-shard",
        "--trajectory",
        str(trajectory),
        "--trace-specs",
        str(specs_path),
        "--shards",
        str(shards),
        "--shard-id",
        "0",
        "--output-root",
        str(output),
        "--run-id",
        name,
    ]
    prepared_path.write_text(
        json.dumps(
            {
                "launcher": "existing_full_answer_shard_runner",
                "launch_command": command,
                "launch_output_root": str(output),
                "launch_spec_path": str(launch_path),
                "launch_selection_fingerprint": launch.to_record()[
                    "selection_fingerprint"
                ],
                "execution_mode": "full" if full_completion else "phase3-probe",
            }
        )
    )
    return bundle


@pytest.mark.parametrize("single_args", [[], ["--validate-only"]])
def test_single_modes_and_sequence_reject_the_same_command_input_mismatch(
    tmp_path: Path, single_args: list[str]
) -> None:
    bundle = _bundle(tmp_path, "entry", width=1, output=tmp_path / "output")
    prepared_path = bundle / "prepared_workload.json"
    prepared = json.loads(prepared_path.read_text())
    command = prepared["launch_command"]
    command[command.index("--trajectory") + 1] = str(bundle / "shards.json")
    prepared_path.write_text(json.dumps(prepared))

    with pytest.raises(ValueError, match="disagree on trajectory_path"):
        perf_cli.main(
            [
                "run-prepared-campaign-workload",
                str(bundle),
                *single_args,
            ]
        )
    with pytest.raises(PreparedSequenceError, match="disagree on trajectory_path"):
        build_prepared_sequence([("entry", bundle)])


@pytest.mark.parametrize(
    ("flag", "field"),
    [
        ("--trajectory", "trajectory_path"),
        ("--trace-specs", "trace_specs_path"),
        ("--shards", "shards_path"),
    ],
)
def test_prepared_validation_rejects_every_command_input_mismatch(
    tmp_path: Path,
    flag: str,
    field: str,
) -> None:
    bundle = _bundle(tmp_path, "entry", width=1, output=tmp_path / "output")
    prepared_path = bundle / "prepared_workload.json"
    prepared = json.loads(prepared_path.read_text())
    command = prepared["launch_command"]
    command[command.index(flag) + 1] = str(bundle / "launch_spec.json")
    prepared_path.write_text(json.dumps(prepared))

    with pytest.raises(ValueError, match=rf"disagree on {field}"):
        perf_cli.main(
            [
                "run-prepared-campaign-workload",
                str(bundle),
                "--validate-only",
            ]
        )


def test_prepared_validation_rejects_each_output_root_disagreement(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, "entry", width=1, output=tmp_path / "output")
    prepared_path = bundle / "prepared_workload.json"
    prepared = json.loads(prepared_path.read_text())
    command = prepared["launch_command"]
    command[command.index("--output-root") + 1] = str(tmp_path / "other-output")
    prepared_path.write_text(json.dumps(prepared))

    with pytest.raises(ValueError, match="command and prepared output root disagree"):
        perf_cli.main(
            [
                "run-prepared-campaign-workload",
                str(bundle),
                "--validate-only",
            ]
        )

    prepared["launch_output_root"] = str(tmp_path / "other-output")
    prepared_path.write_text(json.dumps(prepared))
    with pytest.raises(
        ValueError, match="specification and prepared output root disagree"
    ):
        perf_cli.main(
            [
                "run-prepared-campaign-workload",
                str(bundle),
                "--validate-only",
            ]
        )


def test_prepared_validation_requires_valid_spec_and_readable_inputs(
    tmp_path: Path,
) -> None:
    missing_spec = _bundle(
        tmp_path, "missing-spec", width=1, output=tmp_path / "missing-spec-output"
    )
    prepared_path = missing_spec / "prepared_workload.json"
    prepared = json.loads(prepared_path.read_text())
    prepared.pop("launch_spec_path")
    prepared_path.write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match="launch specification must be a non-empty"):
        perf_cli.main(
            [
                "run-prepared-campaign-workload",
                str(missing_spec),
                "--validate-only",
            ]
        )


def test_prepared_validation_rejects_launch_fingerprint_disagreement(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, "entry", width=1, output=tmp_path / "output")
    prepared_path = bundle / "prepared_workload.json"
    prepared = json.loads(prepared_path.read_text())
    prepared["launch_selection_fingerprint"] = "0" * 64
    prepared_path.write_text(json.dumps(prepared))

    with pytest.raises(ValueError, match="fingerprints disagree"):
        perf_cli.main(
            [
                "run-prepared-campaign-workload",
                str(bundle),
                "--validate-only",
            ]
        )

    missing_input = _bundle(
        tmp_path,
        "missing-input",
        width=1,
        output=tmp_path / "missing-input-output",
    )
    (missing_input / "trajectory.json").unlink()
    with pytest.raises(ValueError, match="trajectory_path must be a readable regular"):
        perf_cli.main(
            [
                "run-prepared-campaign-workload",
                str(missing_input),
                "--validate-only",
            ]
        )


def test_prepared_validation_rejects_unrecorded_command_options(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, "entry", width=1, output=tmp_path / "output")
    prepared_path = bundle / "prepared_workload.json"
    prepared = json.loads(prepared_path.read_text())
    prepared["launch_command"].extend(["--dry-run", "true"])
    prepared_path.write_text(json.dumps(prepared))

    with pytest.raises(ValueError, match="invalid recorded launch command"):
        perf_cli.main(
            [
                "run-prepared-campaign-workload",
                str(bundle),
                "--validate-only",
            ]
        )


def _write_probe_output(
    output: Path,
    *,
    configured_width: int,
    session_capacity: int | None = None,
) -> None:
    """Write one valid probe result at the sequence post-entry validation seam."""

    effective_session_capacity = session_capacity or configured_width
    shard_root = output / "shards/shard_000"
    token_root = shard_root / "token_000001"
    token_root.mkdir(parents=True)
    (shard_root / "shard.json").write_text(
        json.dumps({"status": "probe_completed"}) + "\n"
    )
    (token_root / "trace.json").write_text(
        json.dumps(
            {
                "status": "probe_completed",
                "graph_knobs": {
                    "feature_row_influence_mode": "cuda_windowed",
                    "backward_engine_mode": "duplicated_lanes",
                    "nnsight_session_capacity": effective_session_capacity,
                    "phase1_trace_batch_size_max": configured_width,
                    "phase3_compute_microbatch_max_rows": configured_width,
                    "phase4_execution_batch_max_rows": configured_width,
                },
                "effective_execution": {
                    "batches": {
                        "backward_engine_mode": "duplicated_lanes",
                        "forward_graph_mode": "logical_capacity",
                        "vjp_kernel_mode": "nnsight_injected",
                        "forward_lane_count": configured_width,
                        "session_capacity": effective_session_capacity,
                        "backward_batch_capacity": configured_width,
                        "trace_batch_size": configured_width,
                        "phase3_microbatch_max_rows": configured_width,
                        "phase4_execution_batch_max_rows": configured_width,
                    }
                },
            }
        )
        + "\n"
    )
    (token_root / "telemetry_live.jsonl").write_text(
        json.dumps({"attrs": {"feature_row_influence_mode_resolved": "cuda_windowed"}})
        + "\n"
        + json.dumps(
            {
                "phase": "phase1",
                "name": "phase1.forward",
                "attrs": {"trace_batch_size": configured_width},
            }
        )
        + "\n"
    )


def test_sequence_freezes_order_dependencies_and_mechanisms(tmp_path: Path) -> None:
    wide = _bundle(tmp_path, "wide", width=128, output=tmp_path / "wide-output")
    narrow = _bundle(tmp_path, "narrow", width=1, output=tmp_path / "narrow-output")
    record = build_prepared_sequence([("wide", wide), ("narrow", narrow)])
    path = write_prepared_sequence(tmp_path / "sequence.json", record)

    sequence = load_prepared_sequence(path, require_outputs_absent=True)

    assert [entry.entry_id for entry in sequence.entries] == ["wide", "narrow"]
    assert [
        entry.mechanism_expectations["forward_lane_count"] for entry in sequence.entries
    ] == [128, 1]
    assert len(sequence.fingerprint) == 64
    assert all(
        len(dependency["sha256"]) == 64
        for entry in record["entries"]
        for dependency in entry["dependencies"].values()
    )


def test_sequence_rejects_duplicate_or_existing_outputs(tmp_path: Path) -> None:
    output = tmp_path / "same-output"
    first = _bundle(tmp_path, "first", width=128, output=output)
    second = _bundle(tmp_path, "second", width=1, output=output)
    with pytest.raises(PreparedSequenceError, match="output roots must be unique"):
        build_prepared_sequence([("first", first), ("second", second)])

    record = build_prepared_sequence([("first", first)])
    path = write_prepared_sequence(tmp_path / "sequence.json", record)
    output.mkdir()
    with pytest.raises(PreparedSequenceError, match="existing sequence output"):
        load_prepared_sequence(path, require_outputs_absent=True)


def test_sequence_rejects_changed_dependency(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "entry", width=1, output=tmp_path / "output")
    record = build_prepared_sequence([("entry", bundle)])
    path = write_prepared_sequence(tmp_path / "sequence.json", record)
    (bundle / "trace_specs.jsonl").write_text("changed\n")
    with pytest.raises(PreparedSequenceError, match="dependency changed"):
        load_prepared_sequence(path, require_outputs_absent=True)


def test_sequence_stops_and_persists_failure(tmp_path: Path) -> None:
    first = _bundle(tmp_path, "first", width=128, output=tmp_path / "first-output")
    second = _bundle(tmp_path, "second", width=1, output=tmp_path / "second-output")
    record = build_prepared_sequence([("first", first), ("second", second)])
    path = write_prepared_sequence(tmp_path / "sequence.json", record)
    sequence = load_prepared_sequence(path, require_outputs_absent=True)
    calls: list[str] = []

    def run(entry):
        calls.append(entry.entry_id)
        return 3

    with pytest.raises(RuntimeError, match="runner exited 3"):
        run_prepared_sequence(
            sequence,
            state_root=tmp_path / "state",
            preflight_entry=lambda entry: None,
            run_entry=run,
            validate_entry=lambda entry: {},
        )

    status = json.loads((tmp_path / "state/status.json").read_text())
    assert calls == ["first"]
    assert status["status"] == "failed"
    assert status["entries"][0]["status"] == "failed"
    assert status["entries"][1]["status"] == "pending"


def test_sequence_persists_each_completion_and_validation(tmp_path: Path) -> None:
    first = _bundle(tmp_path, "first", width=128, output=tmp_path / "first-output")
    second = _bundle(tmp_path, "second", width=1, output=tmp_path / "second-output")
    record = build_prepared_sequence([("first", first), ("second", second)])
    path = write_prepared_sequence(tmp_path / "sequence.json", record)
    sequence = load_prepared_sequence(path, require_outputs_absent=True)

    result = run_prepared_sequence(
        sequence,
        state_root=tmp_path / "state",
        preflight_entry=lambda entry: None,
        run_entry=lambda entry: 0,
        validate_entry=lambda entry: {"label": entry.entry_id, "status": "complete"},
    )

    status = json.loads((tmp_path / "state/status.json").read_text())
    assert result == 0
    assert status["status"] == "complete"
    assert status["completed_entry_count"] == 2
    assert [entry["status"] for entry in status["entries"]] == ["complete", "complete"]
    for index, entry_id in enumerate(("first", "second")):
        validation = json.loads(
            (tmp_path / f"state/{index:02d}-{entry_id}-validation.json").read_text()
        )
        assert validation == {"label": entry_id, "status": "complete"}


def test_post_entry_validation_rejects_frozen_capacity_mismatch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    bundle = _bundle(tmp_path, "entry", width=128, output=output)
    record = build_prepared_sequence([("entry", bundle)])
    path = write_prepared_sequence(tmp_path / "sequence.json", record)
    sequence = load_prepared_sequence(path, require_outputs_absent=True)
    _write_probe_output(output, configured_width=128, session_capacity=64)

    with pytest.raises(RuntimeError, match="session capacities"):
        perf_cli._validate_prepared_sequence_entry(sequence.entries[0])


def test_post_entry_validation_rejects_incomplete_frozen_full_run(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    bundle = _bundle(
        tmp_path,
        "entry",
        width=128,
        output=output,
        full_completion=True,
    )
    record = build_prepared_sequence([("entry", bundle)])
    path = write_prepared_sequence(tmp_path / "sequence.json", record)
    sequence = load_prepared_sequence(path, require_outputs_absent=True)
    _write_probe_output(output, configured_width=128)

    with pytest.raises(RuntimeError, match="shard status"):
        perf_cli._validate_prepared_sequence_entry(sequence.entries[0])


def test_sequence_refuses_state_trace_collision(tmp_path: Path) -> None:
    output = tmp_path / "output"
    bundle = _bundle(tmp_path, "entry", width=1, output=output)
    record = build_prepared_sequence([("entry", bundle)])
    path = write_prepared_sequence(tmp_path / "sequence.json", record)
    sequence = load_prepared_sequence(path, require_outputs_absent=True)
    with pytest.raises(PreparedSequenceError, match="state root collides"):
        run_prepared_sequence(
            sequence,
            state_root=output,
            preflight_entry=lambda entry: None,
            run_entry=lambda entry: 0,
            validate_entry=lambda entry: {},
        )


@pytest.mark.parametrize("state_name", ["output/state", "."])
def test_sequence_refuses_nested_state_trace_collision(
    tmp_path: Path, state_name: str
) -> None:
    output = tmp_path / "output"
    bundle = _bundle(tmp_path, "entry", width=1, output=output)
    record = build_prepared_sequence([("entry", bundle)])
    path = write_prepared_sequence(tmp_path / "sequence.json", record)
    sequence = load_prepared_sequence(path, require_outputs_absent=True)
    with pytest.raises(PreparedSequenceError, match="state root collides"):
        run_prepared_sequence(
            sequence,
            state_root=tmp_path / state_name,
            preflight_entry=lambda entry: None,
            run_entry=lambda entry: 0,
            validate_entry=lambda entry: {},
        )
