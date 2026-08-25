"""Strict, atomic persistence for correctness reports."""

from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
from typing import Any
import uuid

from .contracts import CorrectnessReport


def save_correctness_report(report: CorrectnessReport, path: Path) -> None:
    """Atomically write one deterministic correctness report."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(
        report.to_json(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, report_path)
    finally:
        temporary.unlink(missing_ok=True)


def load_correctness_report(path: Path) -> CorrectnessReport:
    """Strictly reopen a report, rejecting schema drift and stale identity."""

    report_path = Path(path)
    payload = json.loads(
        report_path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise TypeError("correctness report must be a JSON object")
    return CorrectnessReport.from_json(payload)


def _object_without_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")
