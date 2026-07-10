from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on older OSC Python
    import toml as tomllib

from .config import DEFAULT_SCRATCH_ROOT, REPO_ROOT
from .io_utils import ensure_dir


DEFAULT_SNAPSHOT_ROOT = DEFAULT_SCRATCH_ROOT / "workspace_snapshots"
DEFAULT_UV_SOURCE_NAME = "circuit-tracer"

IGNORED_NAMES = {
    ".git",
    ".env",
    ".envrc",
    ".ruff_cache",
    ".uv-cache",
    ".ty_cache",
    ".mypy_cache",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "logs",
}

PROJECT_EXCLUDED_RELATIVE_DIRS = {
    Path("experiments/explore"),
    Path("experiments/traces"),
    Path("experiments/extracted"),
    Path("experiments/figures"),
}


def _make_copy_ignore(
    source_root: Path,
    *,
    excluded_relative_dirs: set[Path] | None = None,
):
    excluded_relative_dirs = excluded_relative_dirs or set()

    def _copy_ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in IGNORED_NAMES}
        current_dir = Path(directory)
        try:
            relative_dir = current_dir.resolve().relative_to(source_root.resolve())
        except ValueError:
            relative_dir = Path()

        for name in names:
            candidate_rel = (
                (relative_dir / name) if relative_dir != Path() else Path(name)
            )
            if candidate_rel in excluded_relative_dirs:
                ignored.add(name)

        return ignored

    return _copy_ignore


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


def _safe_git_branch(source_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "branch", "--show-current"],
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


def _safe_git_dirty_files(source_root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=source_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _repo_state(source_root: Path) -> dict[str, Any]:
    return {
        "branch": _safe_git_branch(source_root),
        "commit": _safe_git_head(source_root),
        "dirty_files": _safe_git_dirty_files(source_root),
    }


def repo_state(source_root: Path) -> dict[str, Any]:
    """Return lightweight git provenance for a live workspace."""
    return _repo_state(source_root.resolve())


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


def _load_uv_source_path(
    source_root: Path,
    *,
    package_name: str = DEFAULT_UV_SOURCE_NAME,
) -> tuple[str, Path] | None:
    pyproject_path = source_root / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    tool = payload.get("tool") or {}
    uv = tool.get("uv") or {}
    sources = uv.get("sources") or {}
    source_entry = sources.get(package_name)
    if not isinstance(source_entry, dict):
        return None
    relative_path = source_entry.get("path")
    if not isinstance(relative_path, str):
        return None
    resolved = (source_root / relative_path).resolve()
    if not resolved.exists():
        return None
    return relative_path, resolved


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    excluded_relative_dirs: set[Path] | None = None,
) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=_make_copy_ignore(
            source,
            excluded_relative_dirs=excluded_relative_dirs,
        ),
    )


def _manifest_path_for_workspace(workspace_root: Path) -> Path:
    return workspace_root.parent / ".exact_trace_bench_snapshot.json"


def make_snapshot_read_only(workspace_root: Path) -> None:
    manifest_path = _manifest_path_for_workspace(workspace_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("read_only") is not True:
        manifest["read_only"] = True
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    _make_read_only(workspace_root.parent)


def load_snapshot_manifest(workspace_root: Path) -> dict[str, Any]:
    return json.loads(
        _manifest_path_for_workspace(workspace_root).read_text(encoding="utf-8")
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_non_writable(path: Path) -> bool:
    writable_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    return path.stat().st_mode & writable_bits == 0


def validate_launch_snapshot(
    *,
    workspace_root: Path,
    library_root: Path,
    import_roots: list[Path] | tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Validate an immutable launch snapshot and return its provenance."""
    workspace_root = workspace_root.resolve()
    library_root = library_root.resolve()
    manifest_path = _manifest_path_for_workspace(workspace_root)
    if not manifest_path.is_file():
        raise ValueError(f"Snapshot manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Snapshot manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Snapshot manifest must contain a JSON object: {manifest_path}")

    recorded_workspace = Path(str(manifest.get("snapshot_root", ""))).resolve()
    if recorded_workspace != workspace_root:
        raise ValueError(
            "Workspace root does not match manifest snapshot_root: "
            f"supplied={workspace_root}, recorded={recorded_workspace}"
        )

    uv_snapshots = manifest.get("uv_source_snapshots")
    if not isinstance(uv_snapshots, list):
        raise ValueError("Snapshot manifest uv_source_snapshots must be a list")
    editable_snapshot = next(
        (
            item
            for item in uv_snapshots
            if isinstance(item, dict)
            and item.get("package_name") == DEFAULT_UV_SOURCE_NAME
            and item.get("snapshot_path")
        ),
        None,
    )
    if editable_snapshot is None:
        raise ValueError(
            f"Snapshot manifest does not record the {DEFAULT_UV_SOURCE_NAME!r} uv source"
        )
    recorded_library = Path(str(editable_snapshot["snapshot_path"])).resolve()
    if recorded_library != library_root:
        raise ValueError(
            "Library root does not match recorded uv editable snapshot: "
            f"supplied={library_root}, recorded={recorded_library}"
        )

    for label, root in (
        ("workspace root", workspace_root),
        ("library root", library_root),
    ):
        if not root.is_dir():
            raise ValueError(f"Snapshot {label} does not exist: {root}")
        if not _is_non_writable(root):
            raise ValueError(f"Snapshot {label} is writable: {root}")
    if manifest.get("read_only") is not True:
        raise ValueError("Snapshot manifest read_only must be true")

    resolved_import_roots = [path.resolve() for path in import_roots]
    for import_root in resolved_import_roots:
        if not (
            _is_within(import_root, workspace_root)
            or _is_within(import_root, library_root)
        ):
            raise ValueError(
                "Import root is outside supplied snapshot roots: " f"{import_root}"
            )

    return {
        "workspace_mode": "immutable",
        "manifest_path": str(manifest_path.resolve()),
        "snapshot_container_root": str(workspace_root.parent.resolve()),
        "workspace_root": str(workspace_root),
        "library_workspace_root": str(library_root),
        "import_roots": [str(path) for path in resolved_import_roots],
        "read_only": True,
        "project_repo_state": manifest.get("repo_state"),
        "library_repo_state": editable_snapshot.get("repo_state"),
    }


def sibling_library_root(workspace_root: Path) -> Path | None:
    manifest_path = _manifest_path_for_workspace(workspace_root)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshots = manifest.get("uv_source_snapshots") or []
        if snapshots:
            first = snapshots[0]
            snapshot_path = first.get("snapshot_path")
            if snapshot_path:
                return Path(str(snapshot_path))

    fallback = (workspace_root.parent / "circuit-tracer_chunked").resolve()
    return fallback if fallback.exists() else None


def verify_import_paths(
    *,
    workspace_root: Path,
    library_root: Path | None = None,
) -> dict[str, Any]:
    import importlib
    import sys

    workspace_root = workspace_root.resolve()
    workspace_src = (workspace_root / "src").resolve()
    library_root = (
        library_root.resolve()
        if library_root is not None
        else sibling_library_root(workspace_root)
    )

    inserted_paths: list[str] = []
    if library_root is not None:
        sys.path.insert(0, str(library_root))
        inserted_paths.append(str(library_root))
    sys.path.insert(0, str(workspace_root))
    inserted_paths.append(str(workspace_root))
    sys.path.insert(0, str(workspace_src))
    inserted_paths.append(str(workspace_src))

    module_prefixes = (
        "nlp_research_project",
        "circuit_tracer",
        "trace_pipeline_chunked",
    )
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in module_prefixes
        )
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    try:
        exact_trace_bench = importlib.import_module(
            "nlp_research_project.exact_trace_bench"
        )
        exact_trace_bench_file = Path(exact_trace_bench.__file__).resolve()
        try:
            exact_trace_bench_file.relative_to(workspace_src)
        except ValueError as exc:
            raise ImportError(
                "nlp_research_project.exact_trace_bench resolved outside workspace_root/src: "
                f"{exact_trace_bench_file}"
            ) from exc

        circuit_tracer = importlib.import_module("circuit_tracer")
        trace_pipeline_chunked = importlib.import_module("trace_pipeline_chunked")

        circuit_tracer_file = str(Path(circuit_tracer.__file__).resolve())
        trace_pipeline_chunked_file = str(
            Path(trace_pipeline_chunked.__file__).resolve()
        )
    finally:
        for name in list(sys.modules):
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in module_prefixes
            ):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)

    return {
        "workspace_root": str(workspace_root),
        "workspace_src": str(workspace_src),
        "library_root": None if library_root is None else str(library_root),
        "inserted_paths": inserted_paths,
        "exact_trace_bench_file": str(exact_trace_bench_file),
        "circuit_tracer_file": circuit_tracer_file,
        "trace_pipeline_chunked_file": trace_pipeline_chunked_file,
    }


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

    project_snapshot_path = snapshot_path / source_root.name
    _copy_tree(
        source_root,
        project_snapshot_path,
        excluded_relative_dirs=PROJECT_EXCLUDED_RELATIVE_DIRS,
    )

    uv_source_snapshots: list[dict[str, Any]] = []
    uv_source = _load_uv_source_path(source_root)
    if uv_source is not None:
        relative_path, source_path = uv_source
        destination_path = (project_snapshot_path / relative_path).resolve()
        _copy_tree(source_path, destination_path)
        uv_source_snapshots.append(
            {
                "package_name": DEFAULT_UV_SOURCE_NAME,
                "relative_path": relative_path,
                "source_path": str(source_path),
                "snapshot_path": str(destination_path),
                "git_head": _safe_git_head(source_path),
                "repo_state": _repo_state(source_path),
            }
        )

    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_root": str(source_root),
        "snapshot_root": str(project_snapshot_path),
        "snapshot_container_root": str(snapshot_path),
        "git_head": _safe_git_head(source_root),
        "repo_state": _repo_state(source_root),
        "ignored_names": sorted(IGNORED_NAMES),
        "excluded_relative_dirs": [
            str(path) for path in sorted(PROJECT_EXCLUDED_RELATIVE_DIRS)
        ],
        "read_only": read_only,
        "uv_source_snapshots": uv_source_snapshots,
    }
    (snapshot_path / ".exact_trace_bench_snapshot.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if read_only:
        _make_read_only(snapshot_path)

    return project_snapshot_path


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
