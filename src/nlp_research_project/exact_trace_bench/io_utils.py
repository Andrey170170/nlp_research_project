from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def flatten_dict(
    payload: dict[str, Any] | None,
    *,
    prefix: str,
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not payload:
        return {}
    source = payload if keys is None else {key: payload.get(key) for key in keys}
    return {f"{prefix}{key}": value for key, value in source.items()}


def infer_headers(
    rows: Sequence[dict[str, Any]],
    *,
    preferred: Sequence[str] | None = None,
) -> list[str]:
    seen: set[str] = set()
    headers: list[str] = []

    for key in preferred or ():
        if key not in seen:
            headers.append(key)
            seen.add(key)

    for row in rows:
        for key in row:
            if key not in seen:
                headers.append(key)
                seen.add(key)

    return headers


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    preferred_headers: Sequence[str] | None = None,
) -> None:
    ensure_dir(path.parent)
    headers = infer_headers(rows, preferred=preferred_headers)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def to_json_string(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def parse_memory_value_to_gib(text: str | None) -> float | None:
    if text is None:
        return None
    raw = text.strip()
    if not raw or raw == "n/a":
        return None
    parts = raw.split()
    if len(parts) == 1:
        return float(parts[0])

    value = float(parts[0])
    unit = parts[1].lower()
    if unit in {"gib", "gb"}:
        return value
    if unit in {"mib", "mb"}:
        return value / 1024.0
    if unit in {"kib", "kb"}:
        return value / (1024.0 * 1024.0)
    return value


def safe_stem(path: Path) -> str:
    name = path.name
    if "." not in name:
        return name
    return name.rsplit(".", 1)[0]
