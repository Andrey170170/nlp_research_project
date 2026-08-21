from __future__ import annotations

from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench.cli import build_parser as build_cli_parser
from nlp_research_project.exact_trace_bench.full_answer.schemas import (
    TraceSpec,
    Trajectory,
    build_trace_specs,
    validate_trace_spec,
)
from nlp_research_project.exact_trace_bench.full_answer.selection import select_tokens
from nlp_research_project.exact_trace_bench.perf_cli import (
    build_parser as build_perf_parser,
)
from nlp_research_project.exact_trace_bench.trace_runtime.request import (
    trace_policy_from_scenario,
)


def _trajectory() -> Trajectory:
    return {
        "schema_version": 1,
        "trajectory_id": "phase3_probe",
        "prompt_token_count": 2,
        "prompt_token_ids": [1, 2],
        "generated_tokens": [
            {
                "generated_index": 0,
                "absolute_token_position": 2,
                "token_id": 3,
                "token_text": " x",
                "is_stop": False,
            }
        ],
    }


def _phase3_probe_spec() -> TraceSpec:
    trajectory = _trajectory()
    selection = select_tokens(trajectory, explicit_indices=[0])
    return build_trace_specs(
        trajectory,
        selection,
        graph_knob_overrides={"diagnostic_stop_mode": "phase3_probe"},
    )[0]


def test_phase3_probe_maps_to_typed_execution_without_phase4_count() -> None:
    policy = trace_policy_from_scenario(
        {"method": "exact", "diagnostic_stop_mode": "phase3_probe"}
    )

    assert policy.execution.diagnostic_stop.mode == "phase3_probe"
    assert policy.execution.diagnostic_stop.phase4_batches is None

    with pytest.raises(ValueError, match="only for transition_probe"):
        trace_policy_from_scenario(
            {
                "method": "exact",
                "diagnostic_stop_mode": "phase3_probe",
                "diagnostic_stop_phase4_batches": 1,
            }
        )


def test_full_answer_schema_accepts_phase3_probe_and_rejects_phase4_count() -> None:
    spec = _phase3_probe_spec()
    validate_trace_spec(spec)

    spec["graph_knobs"]["diagnostic_stop_phase4_batches"] = 1
    with pytest.raises(ValueError, match="only for transition_probe"):
        validate_trace_spec(spec)


def test_cli_surfaces_accept_phase3_probe() -> None:
    full_answer = build_cli_parser().parse_args(
        [
            "build-full-answer-trace-specs",
            "--trajectory",
            str(Path("trajectory.json")),
            "--output-dir",
            str(Path("output")),
            "--diagnostic-stop-mode",
            "phase3_probe",
        ]
    )
    assert full_answer.diagnostic_stop_mode == "phase3_probe"

    performance = build_perf_parser().parse_args(
        [
            "run",
            "plt-4b",
            "--diagnostic-stop-mode",
            "phase3_probe",
            "--dry-run",
        ]
    )
    assert performance.diagnostic_stop_mode == "phase3_probe"

    campaign = build_perf_parser().parse_args(
        [
            "prepare-campaign-workload",
            "campaign.json",
            "workload",
            "--profile-role",
            "candidate",
            "--execution-mode",
            "phase3-probe",
            "--output-dir",
            "prepared",
        ]
    )
    assert campaign.execution_mode == "phase3-probe"
