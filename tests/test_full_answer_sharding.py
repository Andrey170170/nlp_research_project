from __future__ import annotations

from nlp_research_project.exact_trace_bench.full_answer.sharding import build_lpt_shards


def _spec(index: int, cost: int) -> dict:
    return {
        "schema_version": 1,
        "trace_id": f"trace_{index}",
        "trajectory_id": "traj",
        "generated_index": index,
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
