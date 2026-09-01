from __future__ import annotations

from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench.full_answer.schemas import (
    Trajectory,
    build_trace_specs,
    load_trace_selection,
    load_trace_specs,
    load_trajectory,
    normalize_trace_spec,
    validate_trace_spec,
    write_trace_selection,
    write_trace_specs,
)
from nlp_research_project.exact_trace_bench.full_answer.selection import select_tokens


def tiny_trajectory() -> Trajectory:
    return {
        "schema_version": 1,
        "trajectory_id": "traj_test",
        "prompt_token_count": 10,
        "prompt_token_ids": list(range(10)),
        "generated_tokens": [
            {
                "generated_index": 0,
                "absolute_token_position": 10,
                "token_id": 101,
                "token_text": "The",
                "logprob": -0.1,
                "is_stop": False,
            },
            {
                "generated_index": 1,
                "absolute_token_position": 11,
                "token_id": 102,
                "token_text": " answer",
                "logprob": -2.0,
                "is_stop": False,
            },
            {
                "generated_index": 2,
                "absolute_token_position": 12,
                "token_id": 103,
                "token_text": " is",
                "logprob": -0.3,
                "is_stop": False,
            },
            {
                "generated_index": 3,
                "absolute_token_position": 13,
                "token_id": 104,
                "token_text": " 42",
                "logprob": -1.5,
                "is_stop": False,
            },
            {
                "generated_index": 4,
                "absolute_token_position": 14,
                "token_id": 105,
                "token_text": ".",
                "is_stop": False,
            },
            {
                "generated_index": 5,
                "absolute_token_position": 15,
                "token_id": 106,
                "token_text": "<eos>",
                "is_stop": True,
            },
        ],
    }


def test_selection_policies_merge_reasons() -> None:
    selection = select_tokens(
        tiny_trajectory(),
        explicit_indices=[3],
        uniform_every_k=3,
        include_numeric=True,
        include_final_answer=True,
        high_surprisal_top_k=2,
    )

    assert selection["selected_indices"] == [1, 3, 4]
    assert selection["selection_reasons"]["3"] == [
        "explicit",
        "uniform_every_k",
        "numeric",
        "high_surprisal",
    ]
    assert selection["selection_reasons"]["4"] == ["final_answer"]


def test_all_token_selection_is_explicit_policy() -> None:
    selection = select_tokens(tiny_trajectory(), include_all=True)

    assert selection["selected_indices"] == list(range(6))
    assert selection["selection_policy"]["include_all"] is True
    assert selection["selection_reasons"]["0"] == ["all_tokens"]


def test_bad_index_validation() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        select_tokens(tiny_trajectory(), explicit_indices=[99])


def test_invalid_trajectory_rejects_inconsistent_prefix_contract() -> None:
    trajectory = tiny_trajectory()
    trajectory["prompt_token_ids"] = [1]
    with pytest.raises(ValueError, match="prompt_token_count"):
        select_tokens(trajectory, explicit_indices=[0])

    trajectory = tiny_trajectory()
    trajectory["generated_tokens"][2]["absolute_token_position"] = 99
    with pytest.raises(ValueError, match="absolute_token_position"):
        select_tokens(trajectory, explicit_indices=[0])


def test_schema_and_trace_spec_round_trip(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    import json

    trajectory_path.write_text(json.dumps(tiny_trajectory()), encoding="utf-8")
    trajectory = load_trajectory(trajectory_path)
    selection = select_tokens(trajectory, explicit_indices=[3])
    selection_path = tmp_path / "trace_selection.json"
    write_trace_selection(selection_path, selection)
    assert load_trace_selection(selection_path)["selected_indices"] == [3]

    specs = build_trace_specs(
        trajectory,
        selection,
        graph_knob_overrides={
            "edge_retention_policy_id": "typed_top_p_v1",
            "row_store_cache_control": "fadvise_dontneed_after_append_and_read_v1",
        },
    )
    assert specs[0]["prefix_token_count"] == 13
    # target_position is the absolute token index of generated token y_k:
    # prompt_token_count + generated_index. The independent trace prefix remains
    # prompt + y_0..y_{k-1}, so prefix_token_count has the same numeric value.
    assert specs[0]["target_position"] == 13
    assert specs[0]["target_token_id"] == 104
    assert specs[0]["target_mode"] == "frozen_target_only"
    assert specs[0]["estimated_cost"] == 13
    assert specs[0]["graph_knobs"]["exact_trace_internal_dtype"] == "fp32"
    assert specs[0]["graph_knobs"]["edge_retention_policy_id"] == "typed_top_p_v1"
    assert specs[0]["graph_knobs"]["correctness_probe_mode"] == "off"
    assert (
        specs[0]["graph_knobs"]["correctness_policy_id"]
        == "behavioral_closure_v1"
    )
    assert (
        specs[0]["graph_knobs"]["row_store_cache_control"]
        == "fadvise_dontneed_after_append_and_read_v1"
    )

    specs_path = tmp_path / "trace_specs.jsonl"
    write_trace_specs(specs_path, specs)
    assert load_trace_specs(specs_path)[0]["trace_id"] == "traj_test_tok000003"


def test_trace_spec_generation_rejects_mismatched_selection() -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    selection["trajectory_id"] = "other"

    with pytest.raises(ValueError, match="trajectory_id"):
        build_trace_specs(tiny_trajectory(), selection)


def test_trace_spec_normalization_makes_legacy_off_selection_explicit() -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(tiny_trajectory(), selection)[0]
    del spec["graph_knobs"]["correctness_probe_mode"]
    del spec["graph_knobs"]["correctness_policy_id"]

    normalized = normalize_trace_spec(spec)

    assert normalized["graph_knobs"]["correctness_probe_mode"] == "off"
    assert (
        normalized["graph_knobs"]["correctness_policy_id"]
        == "behavioral_closure_v1"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"correctness_probe_mode": "unknown"}, "correctness_probe_mode"),
        ({"correctness_policy_id": "unknown_v1"}, "correctness_policy_id"),
        (
            {
                "correctness_probe_mode": "required",
                "correctness_policy_id": None,
            },
            "correctness_policy_id",
        ),
    ],
)
def test_trace_spec_rejects_unadmitted_correctness_selection(
    overrides: dict[str, object],
    message: str,
) -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(
        tiny_trajectory(),
        selection,
        graph_knob_overrides=overrides,
    )[0]

    with pytest.raises(ValueError, match=message):
        validate_trace_spec(spec)


@pytest.mark.parametrize("mode", ["smoke", "required"])
def test_trace_spec_accepts_admitted_correctness_policy(mode: str) -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(
        tiny_trajectory(),
        selection,
        graph_knob_overrides={"correctness_probe_mode": mode},
    )[0]

    validate_trace_spec(spec)


def test_trace_spec_requires_paired_numerical_manifest_declaration() -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(
        tiny_trajectory(),
        selection,
        graph_knob_overrides={
            "correctness_numerical_manifest_path": "/tmp/numerical.json",
            "correctness_numerical_manifest_sha256": None,
        },
    )[0]

    with pytest.raises(ValueError, match="declared together"):
        validate_trace_spec(spec)


def test_trace_spec_accepts_pinned_numerical_manifest_declaration() -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(
        tiny_trajectory(),
        selection,
        graph_knob_overrides={
            "correctness_numerical_manifest_path": "/tmp/numerical.json",
            "correctness_numerical_manifest_sha256": "sha256:" + "a" * 64,
        },
    )[0]

    validate_trace_spec(spec)


def test_trace_spec_requires_paired_correctness_calibration_declaration() -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(
        tiny_trajectory(),
        selection,
        graph_knob_overrides={
            "correctness_calibration_manifest_path": "/tmp/calibration.json",
            "correctness_calibration_manifest_sha256": None,
        },
    )[0]

    with pytest.raises(ValueError, match="declared together"):
        validate_trace_spec(spec)


def test_trace_spec_accepts_pinned_correctness_calibration_declaration() -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(
        tiny_trajectory(),
        selection,
        graph_knob_overrides={
            "correctness_calibration_manifest_path": "/tmp/calibration.json",
            "correctness_calibration_manifest_sha256": "sha256:" + "a" * 64,
        },
    )[0]

    validate_trace_spec(spec)


def test_trace_spec_requires_paired_ordering_qualification_declaration() -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(
        tiny_trajectory(),
        selection,
        graph_knob_overrides={
            "correctness_ordering_qualification_summary_path": "/tmp/summary.json",
            "correctness_ordering_qualification_summary_sha256": None,
        },
    )[0]

    with pytest.raises(ValueError, match="ordering qualification summary"):
        validate_trace_spec(spec)


def test_trace_spec_accepts_pinned_ordering_qualification_declaration() -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(
        tiny_trajectory(),
        selection,
        graph_knob_overrides={
            "correctness_ordering_qualification_summary_path": "/tmp/summary.json",
            "correctness_ordering_qualification_summary_sha256": "sha256:" + "a" * 64,
        },
    )[0]

    validate_trace_spec(spec)


def test_trace_spec_rejects_ambiguous_required_feature_row_selection() -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(
        tiny_trajectory(),
        selection,
        graph_knob_overrides={
            "feature_row_influence_mode": "auto",
            "feature_row_influence_requirement": "required",
            "feature_row_gpu_resident_max_bytes": 1024,
            "feature_row_gpu_window_max_bytes": 1024,
        },
    )[0]

    with pytest.raises(
        ValueError, match="auto feature-row influence cannot be required"
    ):
        validate_trace_spec(spec)


def test_trace_spec_enforced_resource_policy_requires_enforceable_limit() -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(
        tiny_trajectory(),
        selection,
        graph_knob_overrides={
            "runtime_resource_policy": "enforce",
            "resource_planning_envelope": {"hbm_peak_fraction_max": 0.9},
        },
    )[0]

    with pytest.raises(ValueError, match="host_rss_stop_gib or walltime_seconds"):
        validate_trace_spec(spec)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "phase0_decoder_row_ranges": True,
                "decoder_active_row_residency": True,
                "decoder_active_row_max_bytes": 1024,
            },
            "PLT-compatible provider",
        ),
        (
            {
                "transcoder_architecture": "plt",
                "transcoder_provider_family": "gemmascope2-plt-1b-big-affine",
                "phase0_decoder_row_ranges": True,
                "decoder_active_row_max_bytes": 1024,
            },
            "decoder_active_row_residency=true",
        ),
        (
            {
                "transcoder_architecture": "plt",
                "transcoder_provider_family": "gemmascope2-plt-1b-big-affine",
                "phase0_decoder_row_ranges": True,
                "decoder_active_row_residency": True,
            },
            "positive decoder_active_row_safety_margin_bytes",
        ),
        (
            {
                "transcoder_architecture": "plt",
                "transcoder_provider_family": "gemmascope2-plt-1b-big-affine",
                "phase0_decoder_row_ranges": True,
                "decoder_active_row_residency": True,
                "decoder_active_row_max_bytes": 1024,
                "reuse_phase0_window_state": True,
            },
            "incompatible with reuse_phase0_window_state",
        ),
    ],
)
def test_trace_spec_rejects_invalid_phase0_range_dependencies(
    overrides: dict[str, object],
    message: str,
) -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    spec = build_trace_specs(
        tiny_trajectory(),
        selection,
        graph_knob_overrides=overrides,
    )[0]

    with pytest.raises(ValueError, match=message):
        validate_trace_spec(spec)
