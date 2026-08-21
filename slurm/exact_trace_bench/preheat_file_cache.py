#!/usr/bin/env python3
"""CLI adapter for deterministic, bounded filesystem page-cache warming."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from nlp_research_project.exact_trace_bench.file_cache_preheat import (
    MAX_WORKERS,
    preheat_file_cache,
)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read files with a bounded worker pool to warm the filesystem page "
            "cache. Directory inputs are traversed deterministically; hard links "
            "and symlink targets are deduplicated by device and inode."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        help=(
            "Concurrent file readers (1-"
            f"{MAX_WORKERS}); default is min(8, max(1, allocated CPUs // 4))"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and record the file set without reading file contents.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.chunk_mib <= 0:
        raise ValueError("--chunk-mib must be positive")

    chunk_bytes = args.chunk_mib * 1024 * 1024
    manifest = preheat_file_cache(
        args.paths,
        chunk_bytes=chunk_bytes,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    if args.manifest_output is not None:
        _write_manifest(args.manifest_output, manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
