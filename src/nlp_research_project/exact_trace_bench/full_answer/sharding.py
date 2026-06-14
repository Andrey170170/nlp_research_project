from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

from .schemas import SCHEMA_VERSION, validate_trace_spec

COST_MODEL = "prefix_token_count_lpt_v1"
WINDOW_COST_MODEL = "contiguous_window_lpt_v1"


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


def _contiguous_windows(
    specs: Sequence[Mapping[str, Any]], *, max_window_cost: int | None = None
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_generated_index: int | None = None
    for spec_index, spec in enumerate(specs):
        validate_trace_spec(spec)
        generated_index = int(spec["generated_index"])
        cost = int(spec["estimated_cost"])
        would_exceed_cost = (
            current is not None
            and max_window_cost is not None
            and current["spec_indices"]
            and int(current["estimated_cost_sum"]) + cost > max_window_cost
        )
        if (
            current is None
            or previous_generated_index is None
            or generated_index != previous_generated_index + 1
            or would_exceed_cost
        ):
            current = {
                "start_generated_index": generated_index,
                "end_generated_index": generated_index,
                "spec_indices": [spec_index],
                "estimated_cost_sum": cost,
            }
            windows.append(current)
        else:
            current["end_generated_index"] = generated_index
            current["spec_indices"].append(spec_index)
            current["estimated_cost_sum"] += cost
        previous_generated_index = generated_index
    return windows


def build_contiguous_window_lpt_shards(
    specs: Sequence[Mapping[str, Any]],
    *,
    shard_count: int,
    trace_specs_file: Path | str,
    max_window_estimated_cost: int | None = None,
) -> dict[str, Any]:
    """Pack bounded contiguous generated-token windows via LPT.

    A dense all-token selection is one long contiguous run.  To preserve SLURM
    parallelism, long runs are first split into bounded contiguous windows and
    then windows are assigned by LPT.
    """
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    total_cost = 0
    max_spec_cost = 0
    for spec in specs:
        validate_trace_spec(spec)
        cost = int(spec["estimated_cost"])
        total_cost += cost
        max_spec_cost = max(max_spec_cost, cost)
    if max_window_estimated_cost is None and total_cost > 0:
        max_window_estimated_cost = max(max_spec_cost, ceil(total_cost / shard_count))
    if max_window_estimated_cost is not None and max_window_estimated_cost <= 0:
        raise ValueError("max_window_estimated_cost must be positive")
    shards: list[dict[str, Any]] = [
        {
            "shard_id": shard_id,
            "estimated_cost_sum": 0,
            "spec_indices": [],
            "windows": [],
        }
        for shard_id in range(shard_count)
    ]
    for window in sorted(
        _contiguous_windows(specs, max_window_cost=max_window_estimated_cost),
        key=lambda item: (
            -int(item["estimated_cost_sum"]),
            int(item["start_generated_index"]),
        ),
    ):
        target = min(
            shards, key=lambda shard: (shard["estimated_cost_sum"], shard["shard_id"])
        )
        target["spec_indices"].extend(window["spec_indices"])
        target["estimated_cost_sum"] += int(window["estimated_cost_sum"])
        target["windows"].append(dict(window))
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_specs_file": str(trace_specs_file),
        "cost_model": WINDOW_COST_MODEL,
        "max_window_estimated_cost": max_window_estimated_cost,
        "shards": shards,
    }
