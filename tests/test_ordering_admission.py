from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from test_ordering_qualification_runner import _repeat_receipt

from nlp_research_project.exact_trace_bench.correctness import (
    ordering_admission as ordering_admission_module,
)
from nlp_research_project.exact_trace_bench.correctness.ordering_admission import (
    OrderingAdmissionError,
    admitted_nnsight_ordering,
    prepare_ordering_qualification_declaration,
)
from nlp_research_project.exact_trace_bench.correctness.ordering_qualification_runner import (
    compare_gate_receipts,
)


_SIBLING_COMMIT = "742a32aa2d3c56f392067186e3dce534d28ba1c0"
_EXECUTION_FINGERPRINT = "current-behavioral-execution"


class _Model:
    verification_intervened_capture_ordering_qualified = False

    def __init__(self) -> None:
        self.dtype = torch.bfloat16
        self.transcoders = SimpleNamespace(
            _module=SimpleNamespace(dtype=torch.bfloat16)
        )


def _canonical_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _runtime(*, host: str, job_id: str, gpu_uuid: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "captured_at": f"captured-{job_id}",
        "hostname": host,
        "python": {
            "executable": "/env/bin/python",
            "version": "3.12.13",
            "implementation": "CPython",
        },
        "runtime": {
            "torch_version": "2.10.0+cu128",
            "torch_cuda_version": "12.8",
            "cudnn_version": 91002,
        },
        "packages": {
            "torch": "2.10.0",
            "nnsight": "0.6.1",
            "transformers": "4.57.3",
            "circuit-tracer": "0.4.1",
            "nlp-research-project": None,
        },
        "gpu": {
            "slurm_job_id": job_id,
            "slurm_job_name": "qualification" if job_id == "1877900" else "r9",
            "slurm_cluster_name": "granite",
            "slurm_job_partition": "rai-gpu-grn",
            "cuda_visible_devices": "0",
            "nvidia_smi_returncode": 0,
            "gpus": [
                {
                    "name": "NVIDIA H200",
                    "uuid": gpu_uuid,
                    "driver_version": "595.71.05",
                    "memory_total_mib": 143771,
                }
            ],
        },
        "workspace": {
            "project_root": "/snapshot/project",
            "library_root": "/snapshot/library",
            "uv_project_environment": "/env",
        },
    }


def _fixture(
    tmp_path: Path, monkeypatch: Any
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    requested = {
        "model_name": "google/gemma-3-12b-it",
        "transcoder_architecture": "plt",
        "transcoder_provider_family": "gemmascope2-plt-12b-small",
        "repo_id": "google/gemma-scope-2-12b-it",
        "revision": None,
        "clt_subfolder": None,
        "plt_subfolder_template": (
            "transcoder_all/layer_{layer}_width_262k_l0_small/params.safetensors"
        ),
        "layer_count": 48,
        "feature_input_hook": "mlp.hook_in",
        "feature_output_hook": "hook_mlp_out",
        "lazy_encoder": True,
        "lazy_decoder": True,
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "checkpoint_asset_scope": "shared",
        "checkpoint_prefault_budget_bytes": 0,
        "transcoder_cache_dir": "/cache",
    }
    metadata = {
        "requested": requested,
        "detected": {
            "exact_chunked_decoder": True,
            "exact_chunked_provider": True,
            "capabilities": {
                "architecture": "plt",
                "decoder_output_topology": "same_layer",
            },
            "provider_fingerprint": {
                "architecture": "plt",
                "dtype": "torch.bfloat16",
                "decoder_output_topology": "same_layer",
            },
        },
    }
    scope = {
        "model_family": "gemma3",
        "provider_architecture": "plt",
        "decoder_output_topology": "same_layer",
        "dtype": "bfloat16",
    }
    request_binding = {
        "model_name": requested["model_name"],
        "provider_family": requested["transcoder_provider_family"],
        "provider_fingerprint": _canonical_fingerprint(requested),
        "loaded_provider_fingerprint": _canonical_fingerprint(metadata),
        "qualification_scope": scope,
        "trace_id": "representative-trace",
        "graph_fingerprint": "sha256:representative-graph",
        "variants": ["no_op"],
    }
    qualification_workspace = {
        "workspace_mode": "immutable",
        "read_only": True,
        "manifest_sha256": "a" * 64,
        "project_repo_state": {"commit": "qualified-project", "dirty_files": []},
        "library_repo_state": {"commit": _SIBLING_COMMIT, "dirty_files": []},
    }
    repeats = []
    for repeat_index in (1, 2):
        repeat = _repeat_receipt(repeat_index, fingerprint="same")
        repeat["evidence"]["request_binding"] = request_binding
        repeat["evidence"]["workspace"] = qualification_workspace
        repeat["evidence"]["runtime_environment"] = _runtime(
            host="qualification-host",
            job_id="1877900",
            gpu_uuid="GPU-qualified",
        )
        repeat["evidence_fingerprint"] = hashlib.sha256(
            json.dumps(
                repeat["evidence"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        repeats.append(repeat)
    summary = compare_gate_receipts(repeats)
    path = tmp_path / "qualification-summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    knobs = {
        **requested,
        "correctness_ordering_qualification_summary_path": str(path.resolve()),
        "correctness_ordering_qualification_summary_sha256": (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    }
    current_runtime = _runtime(host="r9-host", job_id="r9-job", gpu_uuid="GPU-r9")
    current_runtime["python"]["executable"] = "/different-env/bin/python"
    current_manifest = tmp_path / "current-snapshot-manifest.json"
    current_manifest.write_text('{"snapshot":"current"}', encoding="utf-8")
    monkeypatch.setattr(
        "nlp_research_project.exact_trace_bench.correctness.ordering_admission.capture_runtime_environment",
        lambda: current_runtime,
    )
    monkeypatch.setattr(
        "nlp_research_project.exact_trace_bench.correctness.ordering_admission._current_workspace_provenance",
        lambda: {
            "workspace_mode": "immutable",
            "read_only": True,
            "project_repo_state": {"commit": "r9-project", "dirty_files": []},
            "library_repo_state": {"commit": _SIBLING_COMMIT, "dirty_files": []},
            "manifest_path": str(current_manifest.resolve()),
        },
    )
    return path, knobs, metadata


def test_admission_is_model_local_bounded_and_revoked(
    tmp_path: Path, monkeypatch: Any
) -> None:
    path, knobs, metadata = _fixture(tmp_path, monkeypatch)
    model = _Model()

    with admitted_nnsight_ordering(
        model=model,
        graph_knobs=knobs,
        transcoder_metadata=metadata,
        execution_fingerprint=_EXECUTION_FINGERPRINT,
    ) as evidence:
        assert model.verification_intervened_capture_ordering_qualified is True
        assert evidence["source"] == {
            "path": str(path.resolve()),
            "sha256": knobs["correctness_ordering_qualification_summary_sha256"],
        }
        assert evidence["scope"]["model_name"] == "google/gemma-3-12b-it"
        assert evidence["behavioral_execution_fingerprint"] == _EXECUTION_FINGERPRINT
        assert "repeats" not in evidence

    assert "verification_intervened_capture_ordering_qualified" not in vars(model)
    assert model.verification_intervened_capture_ordering_qualified is False


def test_admission_requires_a_paired_summary_declaration() -> None:
    model = _Model()

    with pytest.raises(OrderingAdmissionError, match="not declared"):
        with admitted_nnsight_ordering(
            model=model,
            graph_knobs={},
            transcoder_metadata={},
            execution_fingerprint=_EXECUTION_FINGERPRINT,
        ):
            raise AssertionError("unreachable")

    assert "verification_intervened_capture_ordering_qualified" not in vars(model)


def test_admission_rejects_tampered_summary_before_activation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    path, knobs, metadata = _fixture(tmp_path, monkeypatch)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    model = _Model()

    with pytest.raises(OrderingAdmissionError, match="sha256 mismatch"):
        with admitted_nnsight_ordering(
            model=model,
            graph_knobs=knobs,
            transcoder_metadata=metadata,
            execution_fingerprint=_EXECUTION_FINGERPRINT,
        ):
            raise AssertionError("unreachable")

    assert "verification_intervened_capture_ordering_qualified" not in vars(model)


def test_preparation_revalidates_and_hashes_the_complete_summary(
    tmp_path: Path, monkeypatch: Any
) -> None:
    path, knobs, _ = _fixture(tmp_path, monkeypatch)

    declaration = prepare_ordering_qualification_declaration(path)

    assert declaration == {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": knobs["correctness_ordering_qualification_summary_sha256"],
    }


def test_preparation_hashes_and_parses_one_summary_byte_buffer(
    tmp_path: Path, monkeypatch: Any
) -> None:
    path, _, _ = _fixture(tmp_path, monkeypatch)
    original_read_bytes = Path.read_bytes
    reads: list[Path] = []

    def tracked_read_bytes(candidate: Path) -> bytes:
        if candidate.resolve() == path.resolve():
            reads.append(candidate.resolve())
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)

    prepare_ordering_qualification_declaration(path)

    assert reads == [path.resolve()]


def test_admission_rejects_self_inconsistent_embedded_receipts_even_with_new_source_sha(
    tmp_path: Path, monkeypatch: Any
) -> None:
    path, knobs, metadata = _fixture(tmp_path, monkeypatch)
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["evidence"]["repeats"][0]["evidence"]["repeat_index"] = 7
    summary["evidence_fingerprint"] = hashlib.sha256(
        json.dumps(
            summary["evidence"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(summary), encoding="utf-8")
    knobs["correctness_ordering_qualification_summary_sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    model = _Model()

    with pytest.raises(OrderingAdmissionError, match="embedded receipts"):
        with admitted_nnsight_ordering(
            model=model,
            graph_knobs=knobs,
            transcoder_metadata=metadata,
            execution_fingerprint=_EXECUTION_FINGERPRINT,
        ):
            raise AssertionError("unreachable")

    assert "verification_intervened_capture_ordering_qualified" not in vars(model)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model", "selected graph knobs"),
        ("provider", "selected graph knobs"),
        ("requested_provider", "requested provider fingerprint"),
        ("loaded_provider", "loaded full provider metadata"),
    ],
)
def test_admission_rejects_wrong_model_or_provider_identity(
    tmp_path: Path,
    monkeypatch: Any,
    mutation: str,
    message: str,
) -> None:
    _, knobs, metadata = _fixture(tmp_path, monkeypatch)
    knobs = dict(knobs)
    metadata = copy.deepcopy(metadata)
    if mutation == "model":
        knobs["model_name"] = "google/gemma-3-4b-it"
    elif mutation == "provider":
        knobs["transcoder_provider_family"] = "gemmascope2-plt-12b-big"
    elif mutation == "requested_provider":
        knobs["transcoder_cache_dir"] = "/other-cache"
        metadata["requested"]["transcoder_cache_dir"] = "/other-cache"
    else:
        metadata["detected"]["exact_chunked_decoder"] = False
    model = _Model()

    with pytest.raises(OrderingAdmissionError, match=message):
        with admitted_nnsight_ordering(
            model=model,
            graph_knobs=knobs,
            transcoder_metadata=metadata,
            execution_fingerprint=_EXECUTION_FINGERPRINT,
        ):
            raise AssertionError("unreachable")

    assert "verification_intervened_capture_ordering_qualified" not in vars(model)


def test_admission_rejects_runtime_stack_drift(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _, knobs, metadata = _fixture(tmp_path, monkeypatch)
    drifted = _runtime(host="r9-host", job_id="r9-job", gpu_uuid="GPU-r9")
    drifted["packages"]["nnsight"] = "0.7.0"
    monkeypatch.setattr(
        "nlp_research_project.exact_trace_bench.correctness.ordering_admission.capture_runtime_environment",
        lambda: drifted,
    )
    model = _Model()

    with pytest.raises(OrderingAdmissionError, match="runtime stack"):
        with admitted_nnsight_ordering(
            model=model,
            graph_knobs=knobs,
            transcoder_metadata=metadata,
            execution_fingerprint=_EXECUTION_FINGERPRINT,
        ):
            raise AssertionError("unreachable")

    assert "verification_intervened_capture_ordering_qualified" not in vars(model)


def test_admission_rejects_current_snapshot_sibling_drift(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _, knobs, metadata = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nlp_research_project.exact_trace_bench.correctness.ordering_admission._current_workspace_provenance",
        lambda: {
            "workspace_mode": "immutable",
            "read_only": True,
            "project_repo_state": {"commit": "r9-project", "dirty_files": []},
            "library_repo_state": {"commit": "wrong-sibling", "dirty_files": []},
            "manifest_path": str(
                (tmp_path / "current-snapshot-manifest.json").resolve()
            ),
        },
    )
    model = _Model()

    with pytest.raises(OrderingAdmissionError, match="sibling commit"):
        with admitted_nnsight_ordering(
            model=model,
            graph_knobs=knobs,
            transcoder_metadata=metadata,
            execution_fingerprint=_EXECUTION_FINGERPRINT,
        ):
            raise AssertionError("unreachable")

    assert "verification_intervened_capture_ordering_qualified" not in vars(model)


def test_admission_validates_real_immutable_snapshot_provenance(
    tmp_path: Path, monkeypatch: Any
) -> None:
    real_current_workspace_provenance = (
        ordering_admission_module._current_workspace_provenance
    )
    _, knobs, metadata = _fixture(tmp_path, monkeypatch)
    snapshot_container = tmp_path / "snapshot"
    workspace_root = snapshot_container / "project"
    library_root = snapshot_container / "circuit-tracer_chunked"
    workspace_root.mkdir(parents=True)
    library_root.mkdir()
    manifest_path = snapshot_container / ".exact_trace_bench_snapshot.json"
    manifest_path.write_text(
        json.dumps(
            {
                "snapshot_root": str(workspace_root.resolve()),
                "read_only": True,
                "repo_state": {"commit": "r9-project", "dirty_files": []},
                "uv_source_snapshots": [
                    {
                        "package_name": "circuit-tracer",
                        "snapshot_path": str(library_root.resolve()),
                        "repo_state": {
                            "commit": _SIBLING_COMMIT,
                            "dirty_files": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    workspace_root.chmod(0o555)
    library_root.chmod(0o555)
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("LIB_WORKSPACE_ROOT", str(library_root))
    monkeypatch.setattr(
        ordering_admission_module,
        "_current_workspace_provenance",
        real_current_workspace_provenance,
    )

    with admitted_nnsight_ordering(
        model=_Model(),
        graph_knobs=knobs,
        transcoder_metadata=metadata,
        execution_fingerprint=_EXECUTION_FINGERPRINT,
    ) as evidence:
        assert evidence["workspace"]["current_project_commit"] == "r9-project"
        assert evidence["workspace"]["current_sibling_commit"] == _SIBLING_COMMIT
        assert evidence["workspace"]["current_manifest_path"] == str(
            manifest_path.resolve()
        )
        assert evidence["workspace"]["current_manifest_sha256"] == (
            "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        )


def test_admission_is_revoked_when_verification_raises(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _, knobs, metadata = _fixture(tmp_path, monkeypatch)
    model = _Model()

    with pytest.raises(RuntimeError, match="verification failed"):
        with admitted_nnsight_ordering(
            model=model,
            graph_knobs=knobs,
            transcoder_metadata=metadata,
            execution_fingerprint=_EXECUTION_FINGERPRINT,
        ):
            assert model.verification_intervened_capture_ordering_qualified is True
            raise RuntimeError("verification failed")

    assert "verification_intervened_capture_ordering_qualified" not in vars(model)
    assert model.verification_intervened_capture_ordering_qualified is False


@pytest.mark.parametrize("pre_admitted", [False, True])
def test_admission_rejects_any_pre_admitted_model_instance(
    tmp_path: Path, monkeypatch: Any, pre_admitted: bool
) -> None:
    _, knobs, metadata = _fixture(tmp_path, monkeypatch)
    model = _Model()
    model.verification_intervened_capture_ordering_qualified = pre_admitted

    with pytest.raises(OrderingAdmissionError, match="already ordering-admitted"):
        with admitted_nnsight_ordering(
            model=model,
            graph_knobs=knobs,
            transcoder_metadata=metadata,
            execution_fingerprint=_EXECUTION_FINGERPRINT,
        ):
            raise AssertionError("unreachable")

    assert model.verification_intervened_capture_ordering_qualified is pre_admitted


def test_admission_rejects_a_true_class_default(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _, knobs, metadata = _fixture(tmp_path, monkeypatch)

    class UnsafeModel(_Model):
        verification_intervened_capture_ordering_qualified = True

    model = UnsafeModel()

    with pytest.raises(
        OrderingAdmissionError, match="class ordering admission default"
    ):
        with admitted_nnsight_ordering(
            model=model,
            graph_knobs=knobs,
            transcoder_metadata=metadata,
            execution_fingerprint=_EXECUTION_FINGERPRINT,
        ):
            raise AssertionError("unreachable")

    assert "verification_intervened_capture_ordering_qualified" not in vars(model)
