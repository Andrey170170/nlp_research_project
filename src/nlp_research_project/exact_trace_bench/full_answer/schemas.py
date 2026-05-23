from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TypedDict, cast

from ..config import base_trace_defaults
from ..io_utils import iter_jsonl, read_json, write_json, write_jsonl

SCHEMA_VERSION = 1
TARGET_MODE = "frozen_target_only"


class GeneratedToken(TypedDict, total=False):
    generated_index: int
    absolute_token_position: int
    token_id: int
    token_text: str
    logprob: float | None
    probability: float | None
    rank: int | None
    is_stop: bool


class Trajectory(TypedDict, total=False):
    schema_version: int
    trajectory_id: str
    prompt_token_count: int
    prompt_token_ids: list[int]
    generated_tokens: list[GeneratedToken]


class TraceSpec(TypedDict):
    schema_version: int
    trace_id: str
    trajectory_id: str
    generated_index: int
    prefix_token_count: int
    target_token_id: int
    target_token_text: str
    target_mode: str
    selection_reasons: list[str]
    graph_knobs: dict[str, Any]
    estimated_cost: int


def load_trajectory(path: Path) -> Trajectory:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"trajectory must be a JSON object: {path}")
    validate_trajectory(payload)
    return payload


def validate_trajectory(trajectory: Mapping[str, Any]) -> None:
    if trajectory.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("trajectory schema_version must be 1")
    if not trajectory.get("trajectory_id"):
        raise ValueError("trajectory_id is required")
    prompt_token_count = trajectory.get("prompt_token_count")
    if not isinstance(prompt_token_count, int) or prompt_token_count < 0:
        raise ValueError("prompt_token_count must be an int")
    prompt_token_ids = trajectory.get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token_id, int) for token_id in prompt_token_ids
    ):
        raise ValueError("prompt_token_ids must be a list of ints")
    if len(prompt_token_ids) != prompt_token_count:
        raise ValueError("prompt_token_count must match len(prompt_token_ids)")
    tokens = trajectory.get("generated_tokens")
    if not isinstance(tokens, list):
        raise ValueError("generated_tokens must be a list")
    for expected_index, token in enumerate(tokens):
        if not isinstance(token, dict):
            raise ValueError("generated token entries must be objects")
        token_payload = cast(Mapping[str, Any], token)
        if token_payload.get("generated_index") != expected_index:
            raise ValueError("generated token indices must be contiguous from 0")
        expected_position = prompt_token_count + expected_index
        if token_payload.get("absolute_token_position") != expected_position:
            raise ValueError(
                "generated token absolute_token_position must equal "
                "prompt_token_count + generated_index"
            )
        if not isinstance(token_payload.get("token_id"), int):
            raise ValueError(
                f"generated token {expected_index} token_id must be an int"
            )
        if not isinstance(token_payload.get("token_text"), str):
            raise ValueError(
                f"generated token {expected_index} token_text must be a string"
            )
        if not isinstance(token_payload.get("is_stop"), bool):
            raise ValueError(f"generated token {expected_index} is_stop must be a bool")


def validate_trace_selection(
    selection: Mapping[str, Any],
    *,
    trajectory: Trajectory | None = None,
) -> None:
    if selection.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("trace selection schema_version must be 1")
    trajectory_id = selection.get("trajectory_id")
    if not trajectory_id:
        raise ValueError("trace selection trajectory_id is required")
    if trajectory is not None:
        validate_trajectory(trajectory)
        if trajectory_id != trajectory["trajectory_id"]:
            raise ValueError("trace selection trajectory_id does not match trajectory")
        token_count = len(trajectory["generated_tokens"])
    else:
        token_count = None

    selected_indices = selection.get("selected_indices")
    if not isinstance(selected_indices, list) or not all(
        isinstance(index, int) for index in selected_indices
    ):
        raise ValueError("selected_indices must be a list of ints")
    if selected_indices != sorted(set(selected_indices)):
        raise ValueError("selected_indices must be sorted and unique")
    if token_count is not None:
        for index in selected_indices:
            if index < 0 or index >= token_count:
                raise ValueError(f"selected index out of bounds: {index}")

    reasons = selection.get("selection_reasons")
    if not isinstance(reasons, dict):
        raise ValueError("selection_reasons must be an object")
    for index in selected_indices:
        key = str(index)
        values = reasons.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"selection_reasons missing non-empty reasons for {key}")
        if not all(isinstance(reason, str) and reason for reason in values):
            raise ValueError(f"selection_reasons for {key} must be strings")


def validate_trace_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("trace spec schema_version must be 1")
    for key in ("trace_id", "trajectory_id", "target_token_text"):
        if not isinstance(spec.get(key), str):
            raise ValueError(f"trace spec {key} must be a string")
    for key in (
        "generated_index",
        "prefix_token_count",
        "target_token_id",
        "estimated_cost",
    ):
        value = spec.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"trace spec {key} must be a non-negative int")
    if spec.get("target_mode") != TARGET_MODE:
        raise ValueError(f"trace spec target_mode must be {TARGET_MODE!r}")
    if not isinstance(spec.get("selection_reasons"), list) or not all(
        isinstance(reason, str) for reason in spec.get("selection_reasons", [])
    ):
        raise ValueError("trace spec selection_reasons must be a list of strings")
    if not isinstance(spec.get("graph_knobs"), dict):
        raise ValueError("trace spec graph_knobs must be an object")


def write_trace_selection(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def load_trace_selection(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"trace selection must be a JSON object: {path}")
    validate_trace_selection(payload)
    return payload


def build_trace_specs(
    trajectory: Trajectory,
    selection: dict[str, Any],
    *,
    graph_knob_overrides: dict[str, Any] | None = None,
) -> list[TraceSpec]:
    validate_trajectory(trajectory)
    validate_trace_selection(selection, trajectory=trajectory)
    trajectory_id = str(trajectory["trajectory_id"])
    prompt_token_count = int(trajectory["prompt_token_count"])
    generated_tokens = trajectory["generated_tokens"]
    reasons_by_index = {
        int(index): list(reasons)
        for index, reasons in selection.get("selection_reasons", {}).items()
    }
    knobs = base_trace_defaults()
    knobs.update(graph_knob_overrides or {})
    specs: list[TraceSpec] = []
    for generated_index in selection.get("selected_indices", []):
        token = generated_tokens[generated_index]
        prefix_token_count = prompt_token_count + generated_index
        specs.append(
            {
                "schema_version": SCHEMA_VERSION,
                "trace_id": f"{trajectory_id}_tok{generated_index:06d}",
                "trajectory_id": trajectory_id,
                "generated_index": generated_index,
                "prefix_token_count": prefix_token_count,
                "target_token_id": int(token["token_id"]),
                "target_token_text": str(token.get("token_text", "")),
                "target_mode": TARGET_MODE,
                "selection_reasons": reasons_by_index.get(generated_index, []),
                "graph_knobs": dict(knobs),
                "estimated_cost": prefix_token_count,
            }
        )
    return specs


def write_trace_specs(path: Path, specs: list[TraceSpec]) -> None:
    for spec in specs:
        validate_trace_spec(spec)
    write_jsonl(path, cast(list[dict[str, Any]], specs))


def load_trace_specs(path: Path) -> list[TraceSpec]:
    specs: list[TraceSpec] = []
    for row in iter_jsonl(path):
        validate_trace_spec(row)
        specs.append(cast(TraceSpec, row))
    return specs


def write_shards(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def load_shards(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"shards must be a JSON object: {path}")
    validate_shards(payload)
    return payload


def validate_shards(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("shards schema_version must be 1")
    if not isinstance(payload.get("trace_specs_file"), str):
        raise ValueError("shards trace_specs_file must be a string")
    if not isinstance(payload.get("cost_model"), str):
        raise ValueError("shards cost_model must be a string")
    shards = payload.get("shards")
    if not isinstance(shards, list):
        raise ValueError("shards must contain a list of shard objects")
    for expected_id, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError("shard entries must be objects")
        shard_payload = cast(Mapping[str, Any], shard)
        if shard_payload.get("shard_id") != expected_id:
            raise ValueError("shard_id values must be contiguous from 0")
        cost = shard_payload.get("estimated_cost_sum")
        if not isinstance(cost, int) or cost < 0:
            raise ValueError("estimated_cost_sum must be a non-negative int")
        spec_indices = shard_payload.get("spec_indices")
        if not isinstance(spec_indices, list) or not all(
            isinstance(index, int) and index >= 0 for index in spec_indices
        ):
            raise ValueError("spec_indices must be a list of non-negative ints")
