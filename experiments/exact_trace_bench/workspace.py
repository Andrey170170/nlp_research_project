from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_SCRATCH_ROOT, REPO_ROOT
from .io_utils import ensure_dir


DEFAULT_SNAPSHOT_ROOT = DEFAULT_SCRATCH_ROOT / "workspace_snapshots"

IGNORED_NAMES = {
    ".git",
    ".env",
    ".envrc",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "logs",
}


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORED_NAMES}


def _safe_git_head(source_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _make_read_only(path: Path) -> None:
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            try:
                os.chmod(Path(root) / directory, 0o555)
            except OSError:
                continue
        for file_name in files:
            try:
                os.chmod(Path(root) / file_name, 0o444)
            except OSError:
                continue
    try:
        os.chmod(path, 0o555)
    except OSError:
        pass


def create_workspace_snapshot(
    *,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    source_root: Path = REPO_ROOT,
    label: str | None = None,
    read_only: bool = True,
) -> Path:
    ensure_dir(snapshot_root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "" if not label else f"_{label}"
    snapshot_path = snapshot_root / f"workspace_{timestamp}{suffix}"
    if snapshot_path.exists():
        raise FileExistsError(f"Snapshot path already exists: {snapshot_path}")

    shutil.copytree(source_root, snapshot_path, ignore=_copy_ignore)

    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_root": str(source_root),
        "snapshot_root": str(snapshot_path),
        "git_head": _safe_git_head(source_root),
        "ignored_names": sorted(IGNORED_NAMES),
        "read_only": read_only,
    }
    (snapshot_path / ".exact_trace_bench_snapshot.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if read_only:
        _make_read_only(snapshot_path)

    return snapshot_path


def resolve_launch_workspace(
    *,
    immutable: bool,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    source_root: Path = REPO_ROOT,
    label: str | None = None,
) -> Path:
    if not immutable:
        return source_root
    return create_workspace_snapshot(
        snapshot_root=snapshot_root,
        source_root=source_root,
        label=label,
    )
