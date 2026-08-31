from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from nlp_research_project.exact_trace_bench.correctness.ordering_qualification_runner import (
    QualificationGateError,
    build_gate_receipt,
    collect_input_artifacts,
    compare_gate_receipts,
    invoke_sibling_qualification,
    write_gate_receipt,
)


class _FakeSiblingReceipt:
    def __init__(
        self,
        *,
        status: str = "qualified",
        mutation: str | None = None,
        science_marker: str = "same",
    ) -> None:
        self._status = status
        self._mutation = mutation
        self._science_marker = science_marker

    def to_dict(self) -> dict[str, object]:
        scope = {
            "model_family": "gemma3",
            "transcoder_architecture": "plt",
            "decoder_output_topology": "same_layer",
            "dtype": "bfloat16",
        }
        request_evidence = {
            "scope": scope,
            "scope_bindings": {},
            "tolerance": {},
            "identity": {"trace_id": "trace-1"},
            "target": {},
            "variants": [],
            "observed_downstream_nodes": [],
            "execution_policy": {
                "deadline_seconds": 120.0,
                "cleanup_reserve_seconds": 1.0,
                "predicted_baseline_seconds": 1.0,
                "predicted_variant_seconds": 1.0,
            },
        }
        request_fingerprint = _sibling_fingerprint(request_evidence)
        science_request_evidence = {
            key: request_evidence[key]
            for key in (
                "scope",
                "tolerance",
                "identity",
                "target",
                "variants",
                "observed_downstream_nodes",
            )
        }
        result = {
            "verdict": self._status,
            "comparisons": [],
            "reasons": [],
            "science_marker": self._science_marker,
        }
        qualification_evidence = {
            "schema": "nnsight_intervened_forward_ordering_qualification",
            "schema_version": 1,
            "qualification_claim": "intervened_forward_capture_ordering_only",
            "science_request_fingerprint": _sibling_fingerprint(
                science_request_evidence
            ),
            "scope": scope,
            "result": result,
        }
        qualification_fingerprint = _sibling_fingerprint(qualification_evidence)
        provenance = {"source": "unit-test"}
        bound_evidence = {
            **qualification_evidence,
            "request_fingerprint": request_fingerprint,
            "qualification_fingerprint": qualification_fingerprint,
            "provenance": provenance,
        }
        payload: dict[str, object] = {
            **qualification_evidence,
            "request_fingerprint": request_fingerprint,
            "schema_version": 1,
            "status": self._status,
            "evidence_fingerprint": _sibling_fingerprint(bound_evidence),
            "qualification_fingerprint": qualification_fingerprint,
            "provenance": provenance,
            "scope_bindings": {},
            "request_evidence": request_evidence,
        }
        if self._mutation == "schema":
            payload["schema"] = "forged"
        elif self._mutation == "status":
            payload["status"] = "qualified" if self._status != "qualified" else "rejected"
        elif self._mutation == "request":
            request_evidence["identity"] = {"trace_id": "forged"}
        elif self._mutation == "qualification_fingerprint":
            payload["qualification_fingerprint"] = "sha256:forged"
        elif self._mutation == "evidence_fingerprint":
            payload["evidence_fingerprint"] = "sha256:forged"
        return payload


def _sibling_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class _FakeRequest:
    captured: dict[str, object] | None = None

    @classmethod
    def from_execution_requests(cls, **kwargs: object) -> object:
        cls.captured = kwargs
        return SimpleNamespace(kind="qualification-request")


def test_invoke_sibling_qualification_uses_public_request_constructor() -> None:
    execution_request = SimpleNamespace(identity=SimpleNamespace(trace_id="trace-1"))
    observed: dict[str, object] = {}

    def qualify(model: object, request: object) -> _FakeSiblingReceipt:
        observed["model"] = model
        observed["request"] = request
        return _FakeSiblingReceipt()

    def validate(request: object, receipt: object) -> None:
        observed["validated_request"] = request
        observed["validated_receipt"] = receipt

    api = SimpleNamespace(
        OrderingQualificationRequest=_FakeRequest,
        qualify_nnsight_ordering=qualify,
        validate_ordering_qualification_receipt=validate,
    )
    model = object()

    receipt = invoke_sibling_qualification(
        api,
        model=model,
        execution_requests=(execution_request,),
        scope={"model_family": "gemma3", "provider_architecture": "plt"},
        provenance={"trace_id": "trace-1"},
    )

    assert receipt.to_dict()["status"] == "qualified"
    assert observed == {
        "model": model,
        "request": SimpleNamespace(kind="qualification-request"),
        "validated_request": SimpleNamespace(kind="qualification-request"),
        "validated_receipt": receipt,
    }
    assert _FakeRequest.captured == {
        "execution_requests": (execution_request,),
        "scope": {"model_family": "gemma3", "provider_architecture": "plt"},
        "provenance": {"trace_id": "trace-1"},
    }


def test_invoke_sibling_qualification_requires_public_integrity_validator() -> None:
    api = SimpleNamespace(
        OrderingQualificationRequest=_FakeRequest,
        qualify_nnsight_ordering=lambda model, request: _FakeSiblingReceipt(),
    )

    with pytest.raises(QualificationGateError, match="validation API"):
        invoke_sibling_qualification(
            api,
            model=object(),
            execution_requests=(object(),),
            scope={"model_family": "gemma3"},
            provenance={},
        )


def test_invoke_sibling_qualification_propagates_integrity_rejection() -> None:
    def reject(request: object, receipt: object) -> None:
        raise ValueError("qualification fingerprint mismatch")

    api = SimpleNamespace(
        OrderingQualificationRequest=_FakeRequest,
        qualify_nnsight_ordering=lambda model, request: _FakeSiblingReceipt(),
        validate_ordering_qualification_receipt=reject,
    )

    with pytest.raises(ValueError, match="qualification fingerprint mismatch"):
        invoke_sibling_qualification(
            api,
            model=object(),
            execution_requests=(object(),),
            scope={"model_family": "gemma3"},
            provenance={},
        )


def test_collect_input_artifacts_hashes_content_not_only_paths(tmp_path: Path) -> None:
    report = tmp_path / "behavioral.json"
    report.write_bytes(b"behavioral")
    graph = tmp_path / "graph.npz"
    graph.write_bytes(b"graph")

    observed = collect_input_artifacts(
        {"behavioral_report": report, "graph": graph}
    )

    assert observed == {
        "behavioral_report": {
            "path": str(report.resolve()),
            "sha256": hashlib.sha256(b"behavioral").hexdigest(),
            "size_bytes": len(b"behavioral"),
        },
        "graph": {
            "path": str(graph.resolve()),
            "sha256": hashlib.sha256(b"graph").hexdigest(),
            "size_bytes": len(b"graph"),
        },
    }


def test_build_gate_receipt_binds_sibling_evidence_inputs_and_snapshot() -> None:
    inputs = {
        "behavioral_report": {
            "path": "/scratch/behavioral.json",
            "sha256": "a" * 64,
            "size_bytes": 100,
        }
    }
    workspace = {
        "workspace_mode": "immutable",
        "read_only": True,
        "project_repo_state": {"commit": "project-commit", "dirty": False},
        "library_repo_state": {"commit": "library-commit", "dirty": False},
        "manifest_sha256": "b" * 64,
    }
    runtime = {
        "packages": {"torch": "2.10.0", "nnsight": "0.6.1"},
        "gpu": {"device_name": "NVIDIA H200"},
    }

    receipt = build_gate_receipt(
        sibling_receipt=_FakeSiblingReceipt(),
        request_binding={"trace_id": "trace-1", "provider_fingerprint": "provider"},
        input_artifacts=inputs,
        workspace_provenance=workspace,
        runtime_environment=runtime,
        repeat_index=1,
    )

    assert receipt["schema"] == "nnsight_ordering_qualification_gate"
    assert receipt["schema_version"] == 1
    assert receipt["status"] == "qualified"
    assert receipt["qualification_only_no_runtime_promotion"] is True
    assert receipt["evidence"]["sibling_receipt"][
        "qualification_fingerprint"
    ].startswith("sha256:")
    assert receipt["evidence"]["input_artifacts"] == inputs
    assert receipt["evidence"]["workspace"] == workspace
    assert receipt["evidence"]["runtime_environment"] == runtime
    assert receipt["evidence"]["repeat_index"] == 1
    expected = hashlib.sha256(
        json.dumps(
            receipt["evidence"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert receipt["evidence_fingerprint"] == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema mismatch"),
        ("status", "status disagrees with verdict"),
        ("request", "request fingerprint mismatch"),
        ("qualification_fingerprint", "qualification fingerprint mismatch"),
        ("evidence_fingerprint", "evidence fingerprint mismatch"),
    ],
)
def test_build_gate_receipt_refuses_tampered_sibling_receipt(
    mutation: str,
    message: str,
) -> None:
    with pytest.raises(QualificationGateError, match=message):
        build_gate_receipt(
            sibling_receipt=_FakeSiblingReceipt(mutation=mutation),
            request_binding={"trace_id": "trace-1"},
            input_artifacts={},
            workspace_provenance={"workspace_mode": "immutable", "read_only": True},
            runtime_environment={},
            repeat_index=1,
        )


@pytest.mark.parametrize("status", ["rejected", "refused"])
def test_nonqualifying_sibling_result_is_durably_bound_without_promotion(
    tmp_path: Path,
    status: str,
) -> None:
    receipt = build_gate_receipt(
        sibling_receipt=_FakeSiblingReceipt(status=status),
        request_binding={"trace_id": "trace-1"},
        input_artifacts={},
        workspace_provenance={"workspace_mode": "immutable", "read_only": True},
        runtime_environment={},
        repeat_index=1,
    )
    destination = tmp_path / f"{status}.json"

    write_gate_receipt(destination, receipt)

    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert persisted["status"] == status
    assert persisted["qualification_only_no_runtime_promotion"] is True
    with pytest.raises(QualificationGateError, match="not a qualified gate receipt"):
        compare_gate_receipts((persisted, persisted))


def test_build_gate_receipt_requires_immutable_read_only_snapshot() -> None:
    with pytest.raises(QualificationGateError, match="immutable read-only"):
        build_gate_receipt(
            sibling_receipt=_FakeSiblingReceipt(),
            request_binding={"trace_id": "trace-1"},
            input_artifacts={},
            workspace_provenance={"workspace_mode": "live", "read_only": False},
            runtime_environment={},
            repeat_index=1,
        )


@pytest.mark.parametrize("repeat_index", [True, 0, -1, 1.5, "1"])
def test_build_gate_receipt_requires_positive_integer_repeat_index(
    repeat_index: object,
) -> None:
    with pytest.raises(QualificationGateError, match="positive integer"):
        build_gate_receipt(
            sibling_receipt=_FakeSiblingReceipt(),
            request_binding={"trace_id": "trace-1"},
            input_artifacts={},
            workspace_provenance={"workspace_mode": "immutable", "read_only": True},
            runtime_environment={},
            repeat_index=cast(int, repeat_index),
        )


def _repeat_receipt(
    repeat_index: int,
    *,
    fingerprint: str,
    torch_version: str = "2.10.0",
) -> dict[str, Any]:
    receipt = build_gate_receipt(
        sibling_receipt=_FakeSiblingReceipt(science_marker=fingerprint),
        request_binding={"trace_id": "trace-1"},
        input_artifacts={"graph": {"sha256": "a" * 64}},
        workspace_provenance={
            "workspace_mode": "immutable",
            "read_only": True,
            "manifest_sha256": "b" * 64,
        },
        runtime_environment={
            "captured_at": f"repeat-{repeat_index}",
            "hostname": f"host-{repeat_index}",
            "python": {
                "executable": "/env/bin/python",
                "version": "3.12.13",
                "implementation": "CPython",
            },
            "runtime": {
                "torch_version": torch_version,
                "torch_cuda_version": "12.8",
                "cudnn_version": 90100,
            },
            "packages": {
                "torch": torch_version,
                "nnsight": "0.6.1",
                "transformers": "4.57.3",
            },
            "gpu": {
                "slurm_job_id": str(100 + repeat_index),
                "slurm_job_name": f"repeat-{repeat_index}",
                "slurm_cluster_name": "granite",
                "slurm_job_partition": "rai-gpu-grn",
                "cuda_visible_devices": "0",
                "nvidia_smi_returncode": 0,
                "gpus": [
                    {
                        "name": "NVIDIA H200",
                        "uuid": "GPU-stable",
                        "driver_version": "580.65",
                        "memory_total_mib": 143771,
                    }
                ],
            },
        },
        repeat_index=repeat_index,
    )
    return receipt


def test_compare_gate_receipts_requires_matching_fresh_process_science() -> None:
    summary = compare_gate_receipts(
        (_repeat_receipt(1, fingerprint="same"), _repeat_receipt(2, fingerprint="same"))
    )

    assert summary["schema"] == "nnsight_ordering_qualification_repetition_gate"
    assert summary["status"] == "qualified"
    assert summary["qualification_only_no_runtime_promotion"] is True
    assert summary["evidence"]["repeat_indices"] == [1, 2]
    assert summary["evidence"]["qualification_fingerprint"].startswith("sha256:")
    assert len(summary["evidence"]["repeat_gate_fingerprints"]) == 2


def test_compare_gate_receipts_refuses_scientific_fingerprint_drift() -> None:
    with pytest.raises(QualificationGateError, match="scientific fingerprint"):
        compare_gate_receipts(
            (
                _repeat_receipt(1, fingerprint="first"),
                _repeat_receipt(2, fingerprint="second"),
            )
        )


def test_compare_gate_receipts_refuses_stable_runtime_contract_drift() -> None:
    with pytest.raises(QualificationGateError, match="runtime contract mismatch"):
        compare_gate_receipts(
            (
                _repeat_receipt(1, fingerprint="same", torch_version="2.10.0"),
                _repeat_receipt(2, fingerprint="same", torch_version="2.11.0"),
            )
        )


def test_compare_gate_receipts_refuses_tampered_repeat_envelope() -> None:
    first = _repeat_receipt(1, fingerprint="same")
    second = _repeat_receipt(2, fingerprint="same")
    second["evidence"]["request_binding"]["trace_id"] = "changed"

    with pytest.raises(QualificationGateError, match="evidence_fingerprint"):
        compare_gate_receipts((first, second))


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "status",
        "request",
        "qualification_fingerprint",
        "evidence_fingerprint",
    ],
)
def test_compare_gate_receipts_refuses_two_identically_forged_sibling_receipts(
    mutation: str,
) -> None:
    receipts = [
        copy.deepcopy(_repeat_receipt(1, fingerprint="same")),
        copy.deepcopy(_repeat_receipt(2, fingerprint="same")),
    ]
    for receipt in receipts:
        sibling = receipt["evidence"]["sibling_receipt"]
        if mutation == "schema":
            sibling["schema"] = "forged"
        elif mutation == "status":
            sibling["status"] = "rejected"
        elif mutation == "request":
            sibling["request_evidence"]["identity"] = {"trace_id": "forged"}
        elif mutation == "qualification_fingerprint":
            sibling["qualification_fingerprint"] = "sha256:forged"
        elif mutation == "evidence_fingerprint":
            sibling["evidence_fingerprint"] = "sha256:forged"
        receipt["evidence_fingerprint"] = hashlib.sha256(
            json.dumps(
                receipt["evidence"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    with pytest.raises(QualificationGateError, match="integrity failure"):
        compare_gate_receipts(tuple(receipts))


def test_write_gate_receipt_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "receipts" / "qualification.json"
    payload = {"schema": "gate", "status": "qualified"}

    write_gate_receipt(destination, payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="already exists"):
        write_gate_receipt(destination, payload)
    assert not list(destination.parent.glob(".*.tmp-*"))


def test_granite_launcher_is_targeted_snapshot_bound_h200_gate() -> None:
    path = (
        Path(__file__).parents[1]
        / "slurm/exact_trace_bench/qualify_nnsight_ordering.granite.sbatch"
    )
    text = path.read_text(encoding="utf-8")

    assert "#SBATCH --partition=rai-gpu-grn" in text
    assert "#SBATCH --qos=rai-gpu-grn-short" in text
    assert "#SBATCH --gres=gpu:h200:1" in text
    assert "#SBATCH --mem=200G" in text
    assert "#SBATCH --time=02:00:00" in text
    assert (
        'source "$WORKSPACE_ROOT/slurm/exact_trace_bench/require_snapshot_workspace.sh"'
        in text
    )
    assert '"$UV_PROJECT_ENVIRONMENT/bin/python"' in text
    assert "qualify_nnsight_ordering.py" in text
    assert "qualification-summary.json" in text
    assert "--compare" in text
    assert "run-prepared-campaign-workload" not in text
    assert "run-full-answer" not in text
