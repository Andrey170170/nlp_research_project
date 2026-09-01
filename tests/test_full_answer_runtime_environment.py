from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench import runtime_provenance
from nlp_research_project.exact_trace_bench.full_answer import runtime_environment


def test_runtime_environment_is_bounded_and_records_execution_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_provenance,
        "gpu_provenance",
        lambda environ: {
            "slurm_job_id": environ.get("SLURM_JOB_ID"),
            "gpus": [{"name": "NVIDIA H200", "uuid": "GPU-test"}],
        },
    )
    monkeypatch.setattr(
        runtime_provenance,
        "_package_versions",
        lambda: {"torch": "test-torch", "nnsight": "test-nnsight"},
    )
    environment = {
        "SLURM_JOB_ID": "123",
        "WORKSPACE_ROOT": "/snapshot/project",
        "LIB_WORKSPACE_ROOT": "/snapshot/library",
        "UV_PROJECT_ENVIRONMENT": "/environment",
        "HF_TOKEN": "must-not-be-recorded",
        "UNRELATED_SECRET": "must-not-be-recorded",
    }

    payload = runtime_provenance.capture_runtime_environment(environment)

    assert payload["gpu"] == {
        "slurm_job_id": "123",
        "gpus": [{"name": "NVIDIA H200", "uuid": "GPU-test"}],
    }
    assert payload["workspace"] == {
        "project_root": "/snapshot/project",
        "library_root": "/snapshot/library",
        "uv_project_environment": "/environment",
    }
    assert payload["runtime"]["torch_version"]
    assert payload["python"]["executable"]
    serialized = json.dumps(payload)
    assert "HF_TOKEN" not in serialized
    assert "must-not-be-recorded" not in serialized


def test_runtime_environment_write_is_atomic_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"schema_version": 1, "gpu": {"gpus": []}}
    monkeypatch.setattr(
        runtime_environment, "capture_runtime_environment", lambda: payload
    )
    output = tmp_path / "provenance" / "runtime_environment.json"

    assert runtime_environment.write_runtime_environment(output) == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert list(output.parent.glob(".*.tmp-*")) == []

    with pytest.raises(FileExistsError, match="already exists"):
        runtime_environment.write_runtime_environment(output)
