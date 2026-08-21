from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench import file_cache_preheat
from nlp_research_project.exact_trace_bench.file_cache_preheat import (
    MAX_WORKERS,
    preheat_file_cache,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "slurm" / "exact_trace_bench" / "preheat_file_cache.py"


def _manifest_files(manifest: dict[str, object]) -> list[dict[str, object]]:
    files = manifest["files"]
    assert isinstance(files, list)
    return files


def test_discovery_is_sorted_and_deduplicates_overlaps_and_links(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    first = data / "b.bin"
    second = data / "a.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    hardlink = tmp_path / "hardlink.bin"
    hardlink.hardlink_to(first)
    symlink = tmp_path / "symlink.bin"
    symlink.symlink_to(second)

    manifest = preheat_file_cache(
        [data, hardlink, symlink, first],
        workers=1,
        dry_run=True,
        environ={"SLURM_CPUS_PER_TASK": "32"},
    )

    files = _manifest_files(manifest)
    assert [record["path"] for record in files] == sorted(
        [str(first.absolute()), str(second.absolute())]
    )
    assert manifest["unique_file_count"] == 2
    assert manifest["total_bytes_planned"] == len(b"firstsecond")
    assert manifest["total_bytes_read"] == 0


def test_workers_one_reads_exact_bytes_with_serial_compatible_accounting(
    tmp_path: Path,
) -> None:
    payloads = [b"0123456789", b"abcdefghijk"]
    for index, payload in enumerate(payloads):
        (tmp_path / f"{index}.bin").write_bytes(payload)

    manifest = preheat_file_cache(
        [tmp_path],
        workers=1,
        chunk_bytes=4,
        environ={"SLURM_CPUS_PER_TASK": "32"},
    )

    assert manifest["worker_count_requested"] == 1
    assert manifest["worker_count_effective"] == 1
    assert manifest["worker_source"] == "explicit"
    assert manifest["scheduling"] == "serial"
    assert manifest["maximum_read_buffer_bytes"] == 4
    assert manifest["total_bytes_planned"] == sum(map(len, payloads))
    assert manifest["total_bytes_read"] == sum(map(len, payloads))
    assert [record["bytes_read"] for record in _manifest_files(manifest)] == [
        len(payload) for payload in payloads
    ]


def test_positional_reader_retries_interruptions_and_partial_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "partial.bin"
    target.write_bytes(b"0123456789")
    real_preadv = os.preadv
    calls = 0

    def partial_preadv(fd: int, buffers: list[memoryview], offset: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError
        shortened = buffers[0][:2]
        return real_preadv(fd, [shortened], offset)

    monkeypatch.setattr(file_cache_preheat.os, "preadv", partial_preadv)
    manifest = preheat_file_cache(
        [target],
        workers=1,
        chunk_bytes=8,
        environ={"SLURM_CPUS_PER_TASK": "1"},
    )

    assert manifest["total_bytes_read"] == 10
    assert calls >= 6


def test_read_error_fails_closed_without_returning_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "error.bin"
    target.write_bytes(b"payload")

    def failed_preadv(fd: int, buffers: list[memoryview], offset: int) -> int:
        del fd, buffers, offset
        raise OSError(5, "injected I/O error")

    monkeypatch.setattr(file_cache_preheat.os, "preadv", failed_preadv)
    with pytest.raises(OSError, match="preheat read failed at byte offset 0"):
        preheat_file_cache(
            [target],
            workers=1,
            environ={"SLURM_CPUS_PER_TASK": "1"},
        )


def test_parallel_workers_overlap_reads_but_preserve_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / f"{index}.bin" for index in range(4)]
    for path in paths:
        path.write_bytes(path.name.encode())

    real_read_file = file_cache_preheat._read_file
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def observed_read(path: Path, chunk_bytes: int) -> tuple[int, float]:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        try:
            return real_read_file(path, chunk_bytes)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(file_cache_preheat, "_read_file", observed_read)
    manifest = preheat_file_cache(
        list(reversed(paths)),
        workers=2,
        chunk_bytes=2,
        environ={"SLURM_CPUS_PER_TASK": "32"},
    )

    assert maximum_active == 2
    assert manifest["scheduling"] == "bounded_file_pool"
    assert [record["path"] for record in _manifest_files(manifest)] == sorted(
        str(path.absolute()) for path in paths
    )


@pytest.mark.parametrize(
    ("allocation_cpus", "expected_workers"),
    [("1", 1), ("4", 1), ("8", 2), ("32", 8), ("128", 8)],
)
def test_default_worker_count_is_derived_from_allocation_and_bounded(
    tmp_path: Path, allocation_cpus: str, expected_workers: int
) -> None:
    for index in range(MAX_WORKERS):
        (tmp_path / f"payload-{index}.bin").write_bytes(b"payload")

    manifest = preheat_file_cache(
        [tmp_path],
        dry_run=True,
        environ={"SLURM_CPUS_PER_TASK": allocation_cpus},
    )

    assert manifest["worker_count_requested"] is None
    assert manifest["worker_count_admitted"] == expected_workers
    assert manifest["worker_count_effective"] == expected_workers
    assert manifest["maximum_read_buffer_bytes"] == (
        expected_workers * file_cache_preheat.DEFAULT_CHUNK_BYTES
    )
    assert manifest["worker_source"] == "allocation_bounded_default"
    assert manifest["worker_derivation"] == (
        "min(8, max(1, allocation_cpu_count // 4))"
    )
    assert manifest["allocation_cpu_count"] == int(allocation_cpus)


def test_worker_limit_is_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"payload")

    with pytest.raises(ValueError, match="safety limit"):
        preheat_file_cache([target], workers=MAX_WORKERS + 1)


def test_cli_writes_manifest_and_reports_worker_policy(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"payload")
    manifest_path = tmp_path / "manifest.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workers",
            "2",
            "--chunk-mib",
            "1",
            "--manifest-output",
            str(manifest_path),
            str(target),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "files" not in summary
    assert manifest["schema_version"] == 1
    assert manifest["worker_count_requested"] == 2
    assert manifest["worker_count_admitted"] == 2
    assert manifest["worker_count_effective"] == 1
    assert manifest["worker_source"] == "explicit"
    assert manifest["chunk_bytes"] == 1024 * 1024
    assert manifest["total_bytes_planned"] == len(b"payload")
    assert manifest["total_bytes_read"] == len(b"payload")
    assert summary == {key: value for key, value in manifest.items() if key != "files"}


def test_cli_failure_does_not_publish_manifest(tmp_path: Path) -> None:
    missing = tmp_path / "missing.bin"
    manifest_path = tmp_path / "manifest.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest-output",
            str(manifest_path),
            str(missing),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "preheat path does not exist" in result.stderr
    assert not manifest_path.exists()
