"""Frozen, fail-closed execution sequences for prepared full-answer workloads."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from .prepared_expectations import resolve_prepared_mechanism_expectations
from .prepared_workload import (
    PREPARED_RUNTIME_INPUT_FIELDS,
    PreparedWorkloadError,
    validate_prepared_workload,
)

SEQUENCE_SCHEMA_VERSION = 1
_ENTRY_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")


class PreparedSequenceError(ValueError):
    """Raised when a prepared sequence is ambiguous, stale, or unsafe to run."""


@dataclass(frozen=True)
class PreparedSequenceEntry:
    """One immutable prepared workload in an ordered sequence."""

    entry_id: str
    prepared_workload_path: Path
    launch_output_root: Path
    mechanism_expectations: Mapping[str, str | int | bool]


@dataclass(frozen=True)
class PreparedSequence:
    """Validated sequence record and its execution-facing entries."""

    path: Path
    fingerprint: str
    record: Mapping[str, Any]
    entries: tuple[PreparedSequenceEntry, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparedSequenceError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreparedSequenceError(f"{label} must be a JSON object: {path}")
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise PreparedSequenceError(
            f"{label} must be a readable regular file: {resolved}"
        )
    return resolved


def _prepared_entry(entry_id: str, source: Path) -> dict[str, Any]:
    if not _ENTRY_ID.fullmatch(entry_id):
        raise PreparedSequenceError(
            f"invalid entry id {entry_id!r}; use 1-64 letters, digits, '.', '_' or '-'"
        )
    try:
        workload = validate_prepared_workload(source)
    except PreparedWorkloadError as error:
        raise PreparedSequenceError(str(error)) from error
    launch_spec = workload.launch_spec
    expectations = resolve_prepared_mechanism_expectations(workload.path)
    return {
        "entry_id": entry_id,
        "prepared_workload_path": str(workload.path),
        "launch_output_root": str(workload.launch_output_root),
        "launch_selection_fingerprint": launch_spec["selection_fingerprint"],
        "mechanism_expectations": expectations,
        "dependencies": dict(workload.dependencies),
        "preheat": launch_spec.get("preheat"),
        "scheduler_request": launch_spec.get("scheduler_request"),
        "workspace": launch_spec.get("workspace"),
    }


def build_prepared_sequence(
    entries: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    """Build a content-addressed record after validating every prepared input."""

    if not entries:
        raise PreparedSequenceError("prepared sequence requires at least one entry")
    records = [_prepared_entry(entry_id, path) for entry_id, path in entries]
    ids = [str(entry["entry_id"]) for entry in records]
    if len(ids) != len(set(ids)):
        raise PreparedSequenceError("prepared sequence entry ids must be unique")
    outputs = [Path(str(entry["launch_output_root"])).resolve() for entry in records]
    if any(
        _paths_overlap(left, right)
        for index, left in enumerate(outputs)
        for right in outputs[index + 1 :]
    ):
        raise PreparedSequenceError(
            "prepared sequence output roots must be unique and non-overlapping"
        )

    for field in ("preheat", "scheduler_request", "workspace"):
        first = records[0][field]
        if any(entry[field] != first for entry in records[1:]):
            raise PreparedSequenceError(
                f"all entries in one allocation must share the same {field} contract"
            )
    preheat = records[0]["preheat"]
    if not isinstance(preheat, Mapping) or preheat.get("policy") != "file_cache":
        raise PreparedSequenceError(
            "same-allocation prepared sequences require file_cache preheat"
        )
    paths = preheat.get("paths")
    if (
        not isinstance(paths, list)
        or not paths
        or not all(isinstance(path, str) and path for path in paths)
    ):
        raise PreparedSequenceError("file_cache preheat requires recorded paths")

    contract = {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "execution_policy": {
            "allocation_reuse": "single_allocation",
            "preheat_count": 1,
            "failure_policy": "stop_on_first_failure",
            "output_policy": "refuse_existing",
        },
        "preheat": dict(preheat),
        "entries": [
            {key: value for key, value in entry.items() if key != "preheat"}
            for entry in records
        ],
    }
    return {**contract, "sequence_fingerprint": _fingerprint(contract)}


def write_prepared_sequence(path: Path, record: Mapping[str, Any]) -> Path:
    """Create one sequence record without replacing an existing path."""

    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
    except FileExistsError as error:
        raise PreparedSequenceError(
            f"refusing to replace sequence record: {output}"
        ) from error
    return output


def load_prepared_sequence(
    path: Path, *, require_outputs_absent: bool
) -> PreparedSequence:
    """Validate a sequence, all dependencies, and optionally output freshness."""

    sequence_path = _regular_file(path, label="prepared sequence")
    record = _read_object(sequence_path, label="prepared sequence")
    if record.get("schema_version") != SEQUENCE_SCHEMA_VERSION:
        raise PreparedSequenceError(
            f"prepared sequence schema_version must be {SEQUENCE_SCHEMA_VERSION}"
        )
    fingerprint = record.get("sequence_fingerprint")
    if not isinstance(fingerprint, str):
        raise PreparedSequenceError("prepared sequence lacks sequence_fingerprint")
    contract = {
        key: value for key, value in record.items() if key != "sequence_fingerprint"
    }
    if _fingerprint(contract) != fingerprint:
        raise PreparedSequenceError("prepared sequence fingerprint mismatch")
    execution_policy = record.get("execution_policy")
    expected_policy = {
        "allocation_reuse": "single_allocation",
        "preheat_count": 1,
        "failure_policy": "stop_on_first_failure",
        "output_policy": "refuse_existing",
    }
    if execution_policy != expected_policy:
        raise PreparedSequenceError(
            "prepared sequence has an unsupported execution policy"
        )
    raw_entries = record.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise PreparedSequenceError("prepared sequence requires at least one entry")

    entries: list[PreparedSequenceEntry] = []
    seen_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping):
            raise PreparedSequenceError(f"sequence entry {index} must be an object")
        entry_id = raw.get("entry_id")
        if not isinstance(entry_id, str) or not _ENTRY_ID.fullmatch(entry_id):
            raise PreparedSequenceError(f"sequence entry {index} has invalid entry_id")
        if entry_id in seen_ids:
            raise PreparedSequenceError(f"duplicate sequence entry id: {entry_id}")
        seen_ids.add(entry_id)
        dependencies = raw.get("dependencies")
        if not isinstance(dependencies, Mapping):
            raise PreparedSequenceError(f"sequence entry {entry_id} lacks dependencies")
        dependency_paths: dict[str, Path] = {}
        for name in (
            "prepared_workload",
            "launch_spec",
            *PREPARED_RUNTIME_INPUT_FIELDS,
        ):
            dependency = dependencies.get(name)
            if not isinstance(dependency, Mapping):
                raise PreparedSequenceError(
                    f"sequence entry {entry_id} lacks dependency {name}"
                )
            raw_dependency_path = dependency.get("path")
            if not isinstance(raw_dependency_path, str) or not raw_dependency_path:
                raise PreparedSequenceError(
                    f"sequence entry {entry_id} dependency {name} lacks a path"
                )
            dependency_path = _regular_file(
                Path(raw_dependency_path), label=f"{entry_id} {name}"
            )
            dependency_paths[name] = dependency_path
            expected_sha = dependency.get("sha256")
            if (
                not isinstance(expected_sha, str)
                or _sha256(dependency_path) != expected_sha
            ):
                raise PreparedSequenceError(
                    f"sequence entry {entry_id} dependency changed: {name}"
                )
        prepared_path = dependency_paths["prepared_workload"]
        current = _prepared_entry(entry_id, prepared_path)
        for field in (
            "prepared_workload_path",
            "launch_output_root",
            "launch_selection_fingerprint",
            "mechanism_expectations",
            "dependencies",
            "scheduler_request",
            "workspace",
        ):
            if current[field] != raw.get(field):
                raise PreparedSequenceError(
                    f"sequence entry {entry_id} no longer matches {field}"
                )
        if current["preheat"] != record.get("preheat"):
            raise PreparedSequenceError(
                f"sequence entry {entry_id} no longer matches preheat contract"
            )
        output_root = Path(str(raw.get("launch_output_root"))).resolve()
        if any(_paths_overlap(output_root, other) for other in seen_outputs):
            raise PreparedSequenceError(
                f"overlapping sequence output root: {output_root}"
            )
        seen_outputs.add(output_root)
        if require_outputs_absent and output_root.exists():
            raise PreparedSequenceError(
                f"refusing existing sequence output: {output_root}"
            )
        expectations = raw.get("mechanism_expectations")
        assert isinstance(expectations, Mapping)
        typed_expectations = cast(Mapping[str, str | int | bool], expectations)
        entries.append(
            PreparedSequenceEntry(
                entry_id=entry_id,
                prepared_workload_path=prepared_path,
                launch_output_root=output_root,
                mechanism_expectations=dict(typed_expectations),
            )
        )
    return PreparedSequence(
        path=sequence_path,
        fingerprint=fingerprint,
        record=record,
        entries=tuple(entries),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def run_prepared_sequence(
    sequence: PreparedSequence,
    *,
    state_root: Path,
    preflight_entry: Callable[[PreparedSequenceEntry], None],
    run_entry: Callable[[PreparedSequenceEntry], int],
    validate_entry: Callable[[PreparedSequenceEntry], Mapping[str, Any]],
) -> int:
    """Run an already-preheated sequence, recording durable per-entry state."""

    state = state_root.resolve()
    colliding_outputs = [
        entry.launch_output_root
        for entry in sequence.entries
        if _paths_overlap(state, entry.launch_output_root)
    ]
    if colliding_outputs:
        raise PreparedSequenceError(
            "sequence state root collides with a trace output root: "
            f"state={state}, trace={colliding_outputs[0]}"
        )
    # Every entry is preflighted before any state or trace output is created.
    for entry in sequence.entries:
        preflight_entry(entry)
    try:
        state.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise PreparedSequenceError(
            f"refusing existing sequence state: {state}"
        ) from error

    snapshot = state / "prepared_sequence.json"
    snapshot.write_text(json.dumps(sequence.record, indent=2) + "\n", encoding="utf-8")
    entry_states: list[dict[str, Any]] = [
        {
            "entry_id": entry.entry_id,
            "status": "pending",
            "prepared_workload_path": str(entry.prepared_workload_path),
            "launch_output_root": str(entry.launch_output_root),
            "mechanism_expectations": dict(entry.mechanism_expectations),
        }
        for entry in sequence.entries
    ]
    status: dict[str, Any] = {
        "schema_version": 1,
        "sequence_fingerprint": sequence.fingerprint,
        "sequence_path": str(sequence.path),
        "status": "running",
        "started_at": _utc_now(),
        "completed_entry_count": 0,
        "entries": entry_states,
    }
    status_path = state / "status.json"
    _atomic_json(status_path, status)

    for index, entry in enumerate(sequence.entries):
        entry_state = entry_states[index]
        entry_state["status"] = "running"
        entry_state["started_at"] = _utc_now()
        status["current_entry_id"] = entry.entry_id
        _atomic_json(status_path, status)
        try:
            returncode = run_entry(entry)
            entry_state["runner_returncode"] = returncode
            if returncode != 0:
                raise RuntimeError(
                    f"sequence entry {entry.entry_id} runner exited {returncode}"
                )
            validation = dict(validate_entry(entry))
            validation_path = state / f"{index:02d}-{entry.entry_id}-validation.json"
            validation_path.write_text(
                json.dumps(validation, indent=2) + "\n", encoding="utf-8"
            )
            entry_state.update(
                {
                    "status": "complete",
                    "completed_at": _utc_now(),
                    "validation_path": str(validation_path),
                }
            )
            status["completed_entry_count"] = index + 1
            _atomic_json(status_path, status)
        except Exception as error:
            entry_state.update(
                {
                    "status": "failed",
                    "failed_at": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            status.update(
                {
                    "status": "failed",
                    "failed_at": _utc_now(),
                    "current_entry_id": entry.entry_id,
                }
            )
            _atomic_json(status_path, status)
            raise

    status.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "current_entry_id": None,
        }
    )
    _atomic_json(status_path, status)
    return 0
