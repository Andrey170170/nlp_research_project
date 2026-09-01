"""Neutral runtime-provenance capture shared by launch and correctness paths."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import torch

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


def gpu_provenance(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Capture bounded scheduler and visible-GPU identity."""

    env = os.environ if environ is None else environ
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    gpu_rows = [
        {
            "name": fields[0],
            "uuid": fields[1],
            "driver_version": fields[2],
            "memory_total_mib": int(fields[3]),
        }
        for line in completed.stdout.splitlines()
        if line.strip()
        for fields in ([field.strip() for field in line.split(",", maxsplit=3)],)
        if len(fields) == 4
    ]
    return {
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "slurm_job_name": env.get("SLURM_JOB_NAME"),
        "slurm_cluster_name": env.get("SLURM_CLUSTER_NAME"),
        "slurm_job_partition": env.get("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_returncode": completed.returncode,
        "gpus": gpu_rows,
    }


def capture_runtime_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture the bounded execution identity used for scientific comparison."""

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


__all__ = ["capture_runtime_environment", "gpu_provenance"]
