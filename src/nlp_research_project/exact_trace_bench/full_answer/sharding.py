from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

from .schemas import SCHEMA_VERSION, validate_trace_spec

COST_MODEL = "prefix_token_count_lpt_v1"


class Shard(TypedDict):
    shard_id: int
    estimated_cost_sum: int
    spec_indices: list[int]


def build_lpt_shards(
    specs: Sequence[Mapping[str, Any]],
    *,
    shard_count: int,
    trace_specs_file: Path | str,
) -> dict[str, Any]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    shards: list[Shard] = [
        {"shard_id": shard_id, "estimated_cost_sum": 0, "spec_indices": []}
        for shard_id in range(shard_count)
    ]
    indexed_costs = []
    for index, spec in enumerate(specs):
        validate_trace_spec(spec)
        indexed_costs.append((index, int(spec["estimated_cost"])))
    for spec_index, cost in sorted(indexed_costs, key=lambda item: (-item[1], item[0])):
        target = min(
            shards, key=lambda shard: (shard["estimated_cost_sum"], shard["shard_id"])
        )
        target["spec_indices"].append(spec_index)
        target["estimated_cost_sum"] += cost
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_specs_file": str(trace_specs_file),
        "cost_model": COST_MODEL,
        "shards": shards,
    }
