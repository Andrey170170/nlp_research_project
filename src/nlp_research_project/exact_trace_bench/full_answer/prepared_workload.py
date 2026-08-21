"""Fail-closed validation for one prepared full-answer workload."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .launch_spec import validate_full_answer_launch_record

PREPARED_RUNTIME_INPUT_FIELDS = (
    "trajectory_path",
    "trace_specs_path",
    "shards_path",
)
_COMMAND_PREFIX = ("uv", "run", "exact-trace-bench", "run-full-answer-shard")
_PATH_FLAGS = {
    "trajectory_path": "--trajectory",
    "trace_specs_path": "--trace-specs",
    "shards_path": "--shards",
}
_REQUIRED_FLAGS = (*_PATH_FLAGS.values(), "--shard-id", "--output-root")
_RUN_FLAGS = {
    "run_id": "--run-id",
    "run_name": "--run-name",
    "run_description": "--run-description",
    "run_goal": "--run-goal",
}
_ALLOWED_FLAGS = frozenset((*_REQUIRED_FLAGS, *_RUN_FLAGS.values()))


class PreparedWorkloadError(ValueError):
    """Raised when a prepared workload cannot be executed as recorded."""


@dataclass(frozen=True)
class ValidatedPreparedWorkload:
    """Execution-facing prepared record after all cross-file checks pass."""

    path: Path
    record: Mapping[str, Any]
    launch_command: tuple[str, ...]
    launch_output_root: Path
    launch_spec_path: Path
    launch_spec: Mapping[str, Any]
    dependencies: Mapping[str, Mapping[str, str]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PreparedWorkloadError(
            f"cannot read prepared dependency {path}: {error}"
        ) from error
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparedWorkloadError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreparedWorkloadError(f"{label} must be a JSON object: {path}")
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise PreparedWorkloadError(
            f"{label} must be a readable regular file: {resolved}"
        )
    try:
        with resolved.open("rb"):
            pass
    except OSError as error:
        raise PreparedWorkloadError(
            f"{label} must be a readable regular file: {resolved}: {error}"
        ) from error
    return resolved


def _resolve_regular_file(raw: object, *, relative_to: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PreparedWorkloadError(f"{label} must be a non-empty path")
    path = Path(raw)
    if not path.is_absolute():
        path = relative_to / path
    return _regular_file(path, label=label)


def _resolve_output_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PreparedWorkloadError(f"{label} must be a non-empty path")
    return Path(raw).resolve()


def _parse_command(raw: object) -> tuple[tuple[str, ...], dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise PreparedWorkloadError(
            "prepared workload has an invalid recorded launch command"
        )
    command: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise PreparedWorkloadError(
                "prepared workload has an invalid recorded launch command"
            )
        command.append(item)
    if tuple(command[:4]) != _COMMAND_PREFIX:
        raise PreparedWorkloadError(
            "prepared workload has an invalid recorded launch command"
        )
    tail = command[4:]
    if len(tail) % 2:
        raise PreparedWorkloadError(
            "prepared workload has an invalid recorded launch command"
        )
    values: dict[str, str] = {}
    for index in range(0, len(tail), 2):
        flag, value = tail[index : index + 2]
        if (
            flag not in _ALLOWED_FLAGS
            or flag in values
            or not value
            or value.startswith("--")
        ):
            raise PreparedWorkloadError(
                "prepared workload has an invalid recorded launch command"
            )
        values[flag] = value
    if any(flag not in values for flag in _REQUIRED_FLAGS):
        raise PreparedWorkloadError(
            "prepared workload has an invalid recorded launch command"
        )
    return tuple(command), values


def _validate_command_identity(
    values: Mapping[str, str], launch_spec: Mapping[str, Any]
) -> None:
    shard_selection = launch_spec.get("shard_selection")
    if not isinstance(shard_selection, str) or not shard_selection:
        raise PreparedWorkloadError(
            "launch specification lacks a non-empty shard_selection"
        )
    if values["--shard-id"] != shard_selection:
        raise PreparedWorkloadError(
            "launch command and launch specification disagree on shard_selection"
        )

    run = launch_spec.get("run")
    if not isinstance(run, Mapping):
        raise PreparedWorkloadError("launch specification run must be an object")
    for field, flag in _RUN_FLAGS.items():
        expected = run.get(field)
        if expected is not None and not isinstance(expected, str):
            raise PreparedWorkloadError(
                f"launch specification run.{field} must be a string or null"
            )
        recorded = values.get(flag)
        if (expected or None) != recorded:
            raise PreparedWorkloadError(
                f"launch command and launch specification disagree on run.{field}"
            )


def validate_prepared_workload(path: Path) -> ValidatedPreparedWorkload:
    """Validate one prepared record, its launch spec, command, and inputs."""

    prepared_path = path.resolve()
    if prepared_path.is_dir():
        prepared_path = prepared_path / "prepared_workload.json"
    prepared_path = _regular_file(prepared_path, label="prepared workload")
    prepared = _read_object(prepared_path, label="prepared workload")
    if prepared.get("launcher") != "existing_full_answer_shard_runner":
        raise PreparedWorkloadError(
            "prepared workload must use the existing full-answer shard runner"
        )
    command, command_values = _parse_command(prepared.get("launch_command"))

    launch_spec_path = _resolve_regular_file(
        prepared.get("launch_spec_path"),
        relative_to=prepared_path.parent,
        label="launch specification",
    )
    launch_spec = _read_object(launch_spec_path, label="launch specification")
    try:
        validate_full_answer_launch_record(launch_spec)
    except ValueError as error:
        raise PreparedWorkloadError(str(error)) from error
    launch_fingerprint = launch_spec["selection_fingerprint"]
    if prepared.get("launch_selection_fingerprint") != launch_fingerprint:
        raise PreparedWorkloadError(
            "prepared workload and launch specification fingerprints disagree"
        )
    _validate_command_identity(command_values, launch_spec)

    output_root = _resolve_output_path(
        prepared.get("launch_output_root"),
        label="prepared workload launch_output_root",
    )
    command_output = _resolve_output_path(
        command_values["--output-root"], label="launch command --output-root"
    )
    if command_output != output_root:
        raise PreparedWorkloadError("launch command and prepared output root disagree")
    spec_output = _resolve_output_path(
        launch_spec.get("output_root"),
        label="launch specification output_root",
    )
    if spec_output != output_root:
        raise PreparedWorkloadError(
            "launch specification and prepared output root disagree"
        )

    dependencies: dict[str, dict[str, str]] = {
        "prepared_workload": {
            "path": str(prepared_path),
            "sha256": _sha256(prepared_path),
        },
        "launch_spec": {
            "path": str(launch_spec_path),
            "sha256": _sha256(launch_spec_path),
        },
    }
    for field, flag in _PATH_FLAGS.items():
        input_path = _resolve_regular_file(
            launch_spec.get(field),
            relative_to=launch_spec_path.parent,
            label=field,
        )
        command_path = Path(command_values[flag]).resolve()
        if command_path != input_path:
            raise PreparedWorkloadError(
                f"launch command and launch specification disagree on {field}"
            )
        dependencies[field] = {
            "path": str(input_path),
            "sha256": _sha256(input_path),
        }

    return ValidatedPreparedWorkload(
        path=prepared_path,
        record=prepared,
        launch_command=command,
        launch_output_root=output_root,
        launch_spec_path=launch_spec_path,
        launch_spec=launch_spec,
        dependencies=dependencies,
    )
