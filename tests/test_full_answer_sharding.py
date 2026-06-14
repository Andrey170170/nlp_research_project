from __future__ import annotations

from nlp_research_project.exact_trace_bench.full_answer.sharding import (
    build_contiguous_window_lpt_shards,
    build_lpt_shards,
)


def _spec(index: int, cost: int) -> dict:
    return {
        "schema_version": 1,
        "trace_id": f"trace_{index}",
        "trajectory_id": "traj",
        "generated_index": index,
        "target_position": cost,
        "prefix_token_count": cost,
        "target_token_id": 100 + index,
        "target_token_text": str(index),
        "target_mode": "frozen_target_only",
        "selection_reasons": ["explicit"],
        "graph_knobs": {},
        "estimated_cost": cost,
    }


def test_lpt_sharding_is_deterministic_and_balanced() -> None:
    specs = [
        _spec(0, 9),
        _spec(1, 8),
        _spec(2, 7),
        _spec(3, 6),
        _spec(4, 5),
    ]
    shards = build_lpt_shards(
        specs, shard_count=2, trace_specs_file="trace_specs.jsonl"
    )

    assert shards == build_lpt_shards(
        specs, shard_count=2, trace_specs_file="trace_specs.jsonl"
    )
    assert shards["cost_model"] == "prefix_token_count_lpt_v1"
    assert [shard["spec_indices"] for shard in shards["shards"]] == [[0, 3, 4], [1, 2]]
    assert [shard["estimated_cost_sum"] for shard in shards["shards"]] == [20, 15]


def test_lpt_sharding_rejects_malformed_specs() -> None:
    try:
        build_lpt_shards(
            [{"estimated_cost": -1}],
            shard_count=1,
            trace_specs_file="trace_specs.jsonl",
        )
    except ValueError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("malformed trace spec was accepted")


def test_contiguous_window_lpt_keeps_windows_together() -> None:
    specs = [_spec(0, 10), _spec(1, 9), _spec(3, 2), _spec(4, 1), _spec(9, 8)]

    shards = build_contiguous_window_lpt_shards(
        specs,
        shard_count=2,
        trace_specs_file="trace_specs.jsonl",
        max_window_estimated_cost=100,
    )

    assert shards["cost_model"] == "contiguous_window_lpt_v1"
    assigned = [shard["spec_indices"] for shard in shards["shards"]]
    assert [0, 1] in assigned
    assert [2, 3] in assigned or any(
        set([2, 3]).issubset(set(indices)) for indices in assigned
    )
    assert shards["shards"][0]["windows"]


def test_contiguous_window_lpt_splits_dense_all_token_runs() -> None:
    specs = [_spec(index, 10) for index in range(20)]

    shards = build_contiguous_window_lpt_shards(
        specs, shard_count=4, trace_specs_file="trace_specs.jsonl"
    )

    assigned = [shard["spec_indices"] for shard in shards["shards"]]
    assert all(indices for indices in assigned)
    assert sorted(index for indices in assigned for index in indices) == list(range(20))
    assert max(shard["estimated_cost_sum"] for shard in shards["shards"]) <= 50
    assert shards["max_window_estimated_cost"] == 50
