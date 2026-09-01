"""Bounded runtime identity capture for reproducible GPU trace jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ..runtime_provenance import capture_runtime_environment


def write_runtime_environment(path: Path) -> dict[str, Any]:
    """Atomically persist the bounded runtime identity without overwriting."""

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"runtime environment output already exists: {destination}"
        )
    payload = capture_runtime_environment()
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"runtime environment temporary exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = write_runtime_environment(args.output)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
