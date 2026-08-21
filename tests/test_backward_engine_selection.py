from __future__ import annotations

import pytest

from nlp_research_project.exact_trace_bench.backward_selection import (
    backward_mechanism_record,
    resolve_backward_execution_selection,
)
from nlp_research_project.exact_trace_bench.cli import build_parser
from nlp_research_project.exact_trace_bench.config import base_trace_defaults
from nlp_research_project.exact_trace_bench.full_answer.schemas import (
    TARGET_MODE,
    build_trace_specs,
    validate_trace_spec,
)
from nlp_research_project.exact_trace_bench.full_answer.selection import select_tokens
from nlp_research_project.exact_trace_bench.trace_runtime.request import (
    trace_policy_from_scenario,
)


def _spec(*, backward_engine_mode: str) -> dict[str, object]:
    knobs = base_trace_defaults()
    knobs["backward_engine_mode"] = backward_engine_mode
    return {
        "schema_version": 1,
        "trace_id": "batched-vjp-selection",
        "trajectory_id": "trajectory",
        "generated_index": 0,
        "target_position": 1,
        "prefix_token_count": 1,
        "target_token_id": 7,
        "target_token_text": "x",
        "target_mode": TARGET_MODE,
        "selection_reasons": ["test"],
        "graph_knobs": knobs,
        "estimated_cost": 1,
    }


def test_trace_policy_preserves_explicit_batched_vjp_selection() -> None:
    scenario = base_trace_defaults()
    scenario["backward_engine_mode"] = "single_forward_batched_vjp"

    policy = trace_policy_from_scenario(scenario)

    assert policy.execution.backward.mode == "single_forward_batched_vjp"


def test_trace_policy_builds_explicit_component_pair_without_legacy_default() -> None:
    scenario = base_trace_defaults()
    scenario.pop("backward_engine_mode")
    scenario.update(
        {
            "forward_graph_mode": "single_lane",
            "vjp_kernel_mode": "autograd_serial",
        }
    )

    policy = trace_policy_from_scenario(scenario)

    assert policy.execution.backward.mode == "single_forward_serial_vjp"
    assert policy.execution.backward.forward_graph_mode == "single_lane"
    assert policy.execution.backward.vjp_kernel_mode == "autograd_serial"


def test_serial_preset_resolves_decomposed_identity() -> None:
    selection = resolve_backward_execution_selection(
        {"backward_engine_mode": "single_forward_serial_vjp"}
    )

    assert selection.backward_engine_mode == "single_forward_serial_vjp"
    assert selection.forward_graph_mode == "single_lane"
    assert selection.vjp_kernel_mode == "autograd_serial"
    assert (
        backward_mechanism_record(
            {
                "backward_engine_mode": "single_forward_serial_vjp",
                "nnsight_session_capacity": 128,
            }
        )["forward_lane_count"]
        == 1
    )


def test_explicit_pair_is_accepted_only_without_preset() -> None:
    explicit = {
        "forward_graph_mode": "single_lane",
        "vjp_kernel_mode": "autograd_serial",
    }
    selection = resolve_backward_execution_selection(explicit)
    assert selection.backward_engine_mode == "single_forward_serial_vjp"
    assert selection.selection_source == "explicit_pair"

    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_backward_execution_selection(
            {"backward_engine_mode": "duplicated_lanes", **explicit}
        )
    with pytest.raises(ValueError, match="must be selected together"):
        resolve_backward_execution_selection({"forward_graph_mode": "single_lane"})
    with pytest.raises(ValueError, match="unsupported backward execution"):
        resolve_backward_execution_selection(
            {
                "forward_graph_mode": "logical_capacity",
                "vjp_kernel_mode": "autograd_serial",
            }
        )


def test_trace_spec_builder_does_not_fabricate_default_preset_pair_conflict() -> None:
    trajectory = {
        "schema_version": 1,
        "trajectory_id": "trajectory",
        "prompt_token_count": 1,
        "prompt_token_ids": [3],
        "generated_tokens": [
            {
                "generated_index": 0,
                "absolute_token_position": 1,
                "token_id": 7,
                "token_text": "x",
                "is_stop": False,
            }
        ],
    }
    specs = build_trace_specs(
        trajectory,  # type: ignore[arg-type]
        select_tokens(trajectory, explicit_indices=[0]),  # type: ignore[arg-type]
        graph_knob_overrides={
            "forward_graph_mode": "single_lane",
            "vjp_kernel_mode": "autograd_serial",
        },
    )

    knobs = specs[0]["graph_knobs"]
    assert "backward_engine_mode" not in knobs
    validate_trace_spec(specs[0])


def test_trace_spec_rejects_unknown_or_replay_backed_batched_vjp_mode() -> None:
    validate_trace_spec(_spec(backward_engine_mode="single_forward_batched_vjp"))
    validate_trace_spec(_spec(backward_engine_mode="single_forward_serial_vjp"))
    with pytest.raises(ValueError, match="backward_engine_mode"):
        validate_trace_spec(_spec(backward_engine_mode="fallback"))

    replay = _spec(backward_engine_mode="single_forward_batched_vjp")
    replay["graph_knobs"]["phase3_gradient_replay_mode"] = "donor"  # type: ignore[index]
    with pytest.raises(
        ValueError, match="does not support phase3_gradient_replay_mode"
    ):
        validate_trace_spec(replay)

    serial_replay = _spec(backward_engine_mode="single_forward_serial_vjp")
    serial_replay["graph_knobs"]["phase3_gradient_replay_mode"] = "donor"  # type: ignore[index]
    with pytest.raises(
        ValueError, match="does not support phase3_gradient_replay_mode"
    ):
        validate_trace_spec(serial_replay)


def test_full_answer_cli_exposes_fail_closed_backward_engine_choice() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "build-full-answer-trace-specs",
            "--trajectory",
            "trajectory.json",
            "--output-dir",
            "specs",
            "--backward-engine-mode",
            "single_forward_batched_vjp",
        ]
    )
    assert args.backward_engine_mode == "single_forward_batched_vjp"


def test_full_answer_cli_exposes_serial_preset_and_explicit_axes() -> None:
    parser = build_parser()
    serial = parser.parse_args(
        [
            "build-full-answer-trace-specs",
            "--trajectory",
            "trajectory.json",
            "--output-dir",
            "specs",
            "--backward-engine-mode",
            "single_forward_serial_vjp",
            "--capture-phase3-gradient-bundle",
            "--capture-phase3-row-bundle",
        ]
    )
    assert serial.backward_engine_mode == "single_forward_serial_vjp"
    assert serial.capture_phase3_gradient_bundle is True
    assert serial.capture_phase3_row_bundle is True

    explicit = parser.parse_args(
        [
            "build-full-answer-trace-specs",
            "--trajectory",
            "trajectory.json",
            "--output-dir",
            "specs",
            "--forward-graph-mode",
            "single_lane",
            "--vjp-kernel-mode",
            "autograd_serial",
        ]
    )
    assert explicit.backward_engine_mode is None
    assert explicit.forward_graph_mode == "single_lane"
    assert explicit.vjp_kernel_mode == "autograd_serial"
