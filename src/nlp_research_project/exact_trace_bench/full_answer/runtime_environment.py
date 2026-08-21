"""Bounded runtime identity capture for reproducible GPU trace jobs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ..perf_cli import gpu_provenance

_PACKAGE_NAMES = (
    "torch",
    "nnsight",
    "transformers",
    "circuit-tracer",
    "nlp-research-project",
)
_GPU_ENVIRONMENT_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_PARTITION",
    "CUDA_VISIBLE_DEVICES",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def capture_runtime_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture only the execution identity needed for scientific comparison.

    Deliberately do not serialize the process environment, command line, or
    credentials. Workspace paths and the external environment root are already
    launch provenance and are safe to retain.
    """

    env = os.environ if environ is None else environ
    gpu_environment = {
        key: value
        for key in _GPU_ENVIRONMENT_KEYS
        if (value := env.get(key)) is not None
    }
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "packages": _package_versions(),
        "gpu": gpu_provenance(gpu_environment),
        "workspace": {
            "project_root": env.get("WORKSPACE_ROOT"),
            "library_root": env.get("LIB_WORKSPACE_ROOT"),
            "uv_project_environment": env.get("UV_PROJECT_ENVIRONMENT"),
        },
    }


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
