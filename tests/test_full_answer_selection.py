from __future__ import annotations

from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench.full_answer.schemas import (
    build_trace_specs,
    load_trace_selection,
    load_trace_specs,
    load_trajectory,
    write_trace_selection,
    write_trace_specs,
)
from nlp_research_project.exact_trace_bench.full_answer.selection import select_tokens


def tiny_trajectory() -> dict:
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
        graph_knob_overrides={"max_edges": 7},
    )
    assert specs[0]["prefix_token_count"] == 13
    assert specs[0]["target_token_id"] == 104
    assert specs[0]["target_mode"] == "frozen_target_only"
    assert specs[0]["estimated_cost"] == 13
    assert specs[0]["graph_knobs"]["exact_trace_internal_dtype"] == "fp32"
    assert specs[0]["graph_knobs"]["max_edges"] == 7

    specs_path = tmp_path / "trace_specs.jsonl"
    write_trace_specs(specs_path, specs)
    assert load_trace_specs(specs_path)[0]["trace_id"] == "traj_test_tok000003"


def test_trace_spec_generation_rejects_mismatched_selection() -> None:
    selection = select_tokens(tiny_trajectory(), explicit_indices=[3])
    selection["trajectory_id"] = "other"

    with pytest.raises(ValueError, match="trajectory_id"):
        build_trace_specs(tiny_trajectory(), selection)
