"""Canonical user-selected launch contract for full-answer tracing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ..backward_selection import backward_mechanism_record
from .schemas import TraceSpec

RuntimeResourcePolicyName = Literal["off", "measure_only", "enforce"]


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_selections(specs: Sequence[TraceSpec]) -> tuple[dict[str, Any], ...]:
    if not specs:
        raise ValueError("launch specification requires at least one trace spec")
    grouped: dict[str, dict[str, Any]] = {}
    for spec in specs:
        config = dict(spec["graph_knobs"])
        config_fingerprint = _fingerprint(config)
        entry = grouped.setdefault(
            config_fingerprint,
            {
                "config_fingerprint": config_fingerprint,
                "selected_config": config,
                "trace_ids": [],
            },
        )
        entry["trace_ids"].append(spec["trace_id"])
    return tuple(grouped.values())


@dataclass(frozen=True)
class FullAnswerLaunchSpec:
    """One renderer-independent statement of the selected execution."""

    trajectory_path: str
    trace_specs_path: str
    shards_path: str
    output_root: str
    shard_selection: str
    run: Mapping[str, str | None]
    selected_configs: tuple[Mapping[str, Any], ...]
    mechanism_selections: tuple[Mapping[str, Any], ...]
    planning_envelope: Mapping[str, Any]
    scheduler_request: Mapping[str, Any]
    runtime_resource_policy: RuntimeResourcePolicyName
    runtime_resource_override_rationale: str | None
    preheat: Mapping[str, Any]
    workspace: Mapping[str, Any]
    monitoring: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.selected_configs:
            raise ValueError("launch specification requires selected configs")
        if self.runtime_resource_policy not in {"off", "measure_only", "enforce"}:
            raise ValueError("invalid runtime resource policy")
        if self.runtime_resource_policy == "enforce" and not any(
            key in self.planning_envelope
            for key in ("host_rss_stop_gib", "walltime_seconds")
        ):
            raise ValueError(
                "enforce runtime resource policy requires host_rss_stop_gib or "
                "walltime_seconds"
            )
        if self.runtime_resource_policy != "enforce" and (
            self.runtime_resource_override_rationale is not None
            and not self.runtime_resource_override_rationale.strip()
        ):
            raise ValueError("runtime resource override rationale cannot be blank")

    def to_record(self) -> dict[str, Any]:
        selection = {"schema_version": 1, **asdict(self)}
        return {**selection, "selection_fingerprint": _fingerprint(selection)}


def build_full_answer_launch_spec(
    *,
    trajectory_path: Path,
    trace_specs_path: Path,
    shards_path: Path,
    output_root: Path,
    shard_selection: str,
    run: Mapping[str, str | None],
    specs: Sequence[TraceSpec],
    planning_envelope: Mapping[str, Any],
    scheduler_request: Mapping[str, Any],
    runtime_resource_policy: RuntimeResourcePolicyName,
    runtime_resource_override_rationale: str | None,
    preheat: Mapping[str, Any],
    workspace: Mapping[str, Any],
    monitoring: Mapping[str, Any],
) -> FullAnswerLaunchSpec:
    config_selections = _config_selections(specs)
    mechanism_selections = tuple(
        {
            "config_fingerprint": entry["config_fingerprint"],
            "trace_ids": list(entry["trace_ids"]),
            "feature_row_influence_mode": entry["selected_config"].get(
                "feature_row_influence_mode", "cpu_exact"
            ),
            "feature_row_influence_requirement": entry["selected_config"].get(
                "feature_row_influence_requirement", "preferred"
            ),
            **backward_mechanism_record(entry["selected_config"]),
        }
        for entry in config_selections
    )
    resource_selections = {
        (
            entry["selected_config"].get("runtime_resource_policy", "off"),
            _fingerprint(
                entry["selected_config"].get("resource_planning_envelope", {})
            ),
        )
        for entry in config_selections
    }
    expected_resource_selection = (
        runtime_resource_policy,
        _fingerprint(dict(planning_envelope)),
    )
    if resource_selections != {expected_resource_selection}:
        raise ValueError(
            "trace specs and launch specification disagree on resource selection"
        )
    return FullAnswerLaunchSpec(
        trajectory_path=str(trajectory_path),
        trace_specs_path=str(trace_specs_path),
        shards_path=str(shards_path),
        output_root=str(output_root),
        shard_selection=shard_selection,
        run=dict(run),
        selected_configs=config_selections,
        mechanism_selections=mechanism_selections,
        planning_envelope=dict(planning_envelope),
        scheduler_request=dict(scheduler_request),
        runtime_resource_policy=runtime_resource_policy,
        runtime_resource_override_rationale=runtime_resource_override_rationale,
        preheat=dict(preheat),
        workspace=dict(workspace),
        monitoring=dict(monitoring),
    )


def render_local_full_answer_command(
    launch: FullAnswerLaunchSpec,
    *,
    executable: Sequence[str] = ("uv", "run", "exact-trace-bench"),
) -> list[str]:
    if launch.shard_selection != "0":
        raise ValueError("local full-answer rendering requires one shard id")
    command = [
        *executable,
        "run-full-answer-shard",
        "--trajectory",
        launch.trajectory_path,
        "--trace-specs",
        launch.trace_specs_path,
        "--shards",
        launch.shards_path,
        "--shard-id",
        launch.shard_selection,
        "--output-root",
        launch.output_root,
    ]
    for key, flag in (
        ("run_id", "--run-id"),
        ("run_name", "--run-name"),
        ("run_description", "--run-description"),
        ("run_goal", "--run-goal"),
    ):
        value = launch.run.get(key)
        if value:
            command.extend((flag, value))
    return command


def validate_full_answer_launch_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != 1:
        raise ValueError("launch specification schema_version must be 1")
    recorded_fingerprint = record.get("selection_fingerprint")
    if not isinstance(recorded_fingerprint, str):
        raise ValueError("launch specification lacks selection_fingerprint")
    selection = {
        key: value for key, value in record.items() if key != "selection_fingerprint"
    }
    if _fingerprint(selection) != recorded_fingerprint:
        raise ValueError("launch specification fingerprint mismatch")
    if record.get("runtime_resource_policy") not in {
        "off",
        "measure_only",
        "enforce",
    }:
        raise ValueError("launch specification has invalid runtime resource policy")
    if not isinstance(record.get("scheduler_request"), Mapping):
        raise ValueError("launch specification scheduler_request must be an object")
    if not isinstance(record.get("planning_envelope"), Mapping):
        raise ValueError("launch specification planning_envelope must be an object")
