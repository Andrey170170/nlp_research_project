"""Deterministic, bounded filesystem page-cache warming.

The public interface deliberately hides file discovery, inode deduplication,
worker admission, positional I/O, and manifest accounting from launch wrappers.
"""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
MAX_WORKERS = 8
ALLOCATION_CPUS_PER_WORKER = 4


def _allocation_cpu_count(environ: Mapping[str, str]) -> int:
    """Return the CPUs available to this process, preferring the Slurm grant."""
    slurm_value = environ.get("SLURM_CPUS_PER_TASK")
    if slurm_value is not None:
        try:
            cpus = int(slurm_value)
        except ValueError as error:
            raise ValueError(
                f"SLURM_CPUS_PER_TASK must be a positive integer; got {slurm_value!r}"
            ) from error
        if cpus <= 0:
            raise ValueError(
                f"SLURM_CPUS_PER_TASK must be a positive integer; got {slurm_value!r}"
            )
        return cpus

    try:
        affinity_count = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity_count = 0
    if affinity_count > 0:
        return affinity_count
    return max(1, os.cpu_count() or 1)


def _worker_policy(
    workers: int | None,
    *,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    allocation_cpus = _allocation_cpu_count(environ)
    if workers is None:
        admitted = min(
            MAX_WORKERS,
            max(1, allocation_cpus // ALLOCATION_CPUS_PER_WORKER),
        )
        source = "allocation_bounded_default"
        derivation = "min(8, max(1, allocation_cpu_count // 4))"
    else:
        if workers <= 0:
            raise ValueError("workers must be positive")
        if workers > MAX_WORKERS:
            raise ValueError(
                f"workers must not exceed the safety limit of {MAX_WORKERS}"
            )
        admitted = workers
        source = "explicit"
        derivation = "explicit worker count bounded to 1..8"
    return {
        "worker_count_requested": workers,
        "worker_count_admitted": admitted,
        "worker_source": source,
        "worker_derivation": derivation,
        "allocation_cpu_count": allocation_cpus,
        "allocation_cpus_per_default_worker": ALLOCATION_CPUS_PER_WORKER,
        "maximum_worker_count": MAX_WORKERS,
    }


def _regular_files(paths: Sequence[Path]) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    for requested in paths:
        if not requested.exists():
            raise FileNotFoundError(f"preheat path does not exist: {requested}")
        if requested.is_dir():
            for root, dirnames, filenames in os.walk(requested, followlinks=False):
                dirnames.sort()
                candidates.extend(Path(root) / name for name in sorted(filenames))
        else:
            candidates.append(requested)

    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        file_stat = candidate.stat()
        if not stat.S_ISREG(file_stat.st_mode):
            continue
        identity = (file_stat.st_dev, file_stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        records.append(
            {
                "path": str(candidate.absolute()),
                "resolved_path": str(candidate.resolve()),
                "size_bytes": file_stat.st_size,
            }
        )
    records.sort(key=lambda record: record["path"])
    return records


def _read_file(path: Path, chunk_bytes: int) -> tuple[int, float]:
    """Read one file with bounded memory and offset-stable positional I/O."""
    buffer = bytearray(chunk_bytes)
    view = memoryview(buffer)
    bytes_read = 0
    started = time.perf_counter()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        while True:
            try:
                count = os.preadv(descriptor, [view], bytes_read)
            except InterruptedError:
                continue
            if count == 0:
                break
            if count < 0 or count > chunk_bytes:
                raise RuntimeError(f"invalid positional read count for {path}: {count}")
            bytes_read += count
    except OSError as error:
        raise OSError(
            error.errno,
            f"preheat read failed at byte offset {bytes_read}: {error.strerror}",
            str(path),
        ) from error
    finally:
        os.close(descriptor)
    return bytes_read, time.perf_counter() - started


def _read_records(
    records: list[dict[str, Any]],
    *,
    chunk_bytes: int,
    workers: int,
) -> int:
    def read_one(index: int) -> tuple[int, int, float]:
        record = records[index]
        bytes_read, elapsed = _read_file(Path(record["path"]), chunk_bytes)
        planned = int(record["size_bytes"])
        if bytes_read != planned:
            raise RuntimeError(
                f"file size changed while preheating {record['path']}: "
                f"planned={planned} read={bytes_read}"
            )
        return index, bytes_read, elapsed

    if workers == 1:
        results = (read_one(index) for index in range(len(records)))
        total_read = 0
        for index, bytes_read, elapsed in results:
            records[index]["bytes_read"] = bytes_read
            records[index]["elapsed_seconds"] = round(elapsed, 6)
            total_read += bytes_read
        return total_read

    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="file-cache-preheat",
    )
    futures: dict[Future[tuple[int, int, float]], int] = {}
    try:
        futures = {
            executor.submit(read_one, index): index for index in range(len(records))
        }
        total_read = 0
        for future in as_completed(futures):
            index, bytes_read, elapsed = future.result()
            records[index]["bytes_read"] = bytes_read
            records[index]["elapsed_seconds"] = round(elapsed, 6)
            total_read += bytes_read
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return total_read


def preheat_file_cache(
    paths: Sequence[Path],
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    workers: int | None = None,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Warm ``paths`` and return the complete provenance manifest.

    ``workers=None`` chooses a conservative default from the current CPU
    allocation. Explicit worker counts are admitted exactly up to
    ``MAX_WORKERS`` and the effective pool is no larger than the file set. A
    worker owns one file at a time, so read-buffer memory is bounded by
    ``effective_workers * chunk_bytes``.
    """
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if not paths:
        raise ValueError("at least one preheat path is required")

    values = os.environ if environ is None else environ
    policy = _worker_policy(workers, environ=values)
    started_at = datetime.now(timezone.utc)
    records = _regular_files(paths)
    if not records:
        raise RuntimeError("preheat file set is empty")
    effective_workers = min(int(policy["worker_count_admitted"]), len(records))

    total_planned = sum(int(record["size_bytes"]) for record in records)
    read_started = time.perf_counter()
    if dry_run:
        for record in records:
            record["bytes_read"] = 0
            record["elapsed_seconds"] = 0.0
        total_read = 0
    else:
        total_read = _read_records(
            records,
            chunk_bytes=chunk_bytes,
            workers=effective_workers,
        )
    elapsed = time.perf_counter() - read_started

    return {
        "schema_version": 1,
        "operation": "filesystem_page_cache_preheat",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "requested_paths": [str(path.absolute()) for path in paths],
        "deduplication": "st_dev_st_ino",
        "chunk_bytes": chunk_bytes,
        **policy,
        "worker_count_effective": effective_workers,
        "read_buffer_bytes_per_worker": chunk_bytes,
        "maximum_read_buffer_bytes": effective_workers * chunk_bytes,
        "scheduling": "serial" if effective_workers == 1 else "bounded_file_pool",
        "dry_run": dry_run,
        "unique_file_count": len(records),
        "total_bytes_planned": total_planned,
        "total_bytes_read": total_read,
        "elapsed_seconds": round(elapsed, 6),
        "throughput_mib_per_second": (
            round(total_read / (1024 * 1024) / elapsed, 3)
            if elapsed > 0 and total_read > 0
            else None
        ),
        "files": records,
    }
