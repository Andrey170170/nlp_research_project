from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from circuit_tracer.verification import (
    FeatureNode,
    FeatureValue,
    InterventionExecutionRequest,
    InterventionSemantics,
    InterventionVariant,
    OrderingQualificationRequest,
    PreactivationIntervention,
    TargetState,
    TraceIdentity,
    VariantKind,
)
from nlp_research_project.exact_trace_bench.correctness.propagated_ordering_diagnostic_runner import (
    PropagatedOrderingDiagnosticError,
    build_diagnostic_receipt,
    diagnostic_scope_from_graph_knobs,
    invoke_sibling_diagnostic,
    write_diagnostic_receipt,
)


class _FakeRequest:
    captured: dict[str, object] | None = None

    @classmethod
    def from_execution_requests(cls, **kwargs: object) -> object:
        cls.captured = kwargs
        return SimpleNamespace(kind="ordering-qualification-request")


class _FakeDiagnosticReceipt:
    def __init__(self, *, status: str = "complete") -> None:
        self._status = status

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "propagated_ordering_diagnostic",
            "schema_version": 1,
            "status": self._status,
            "diagnostic_only_no_runtime_promotion": True,
            "diagnostic_fingerprint": "sha256:" + "a" * 64,
            "evidence_fingerprint": "sha256:" + "b" * 64,
            "evidence": {"earliest_divergence": "layer_35.feature_input"},
        }


def test_project_diagnostic_scope_is_accepted_by_real_sibling_request() -> None:
    source = FeatureNode(0, 1, 0)
    downstream = FeatureNode(1, 1, 1)
    predicted = (FeatureValue(downstream, 0.0),)
    intervention = (PreactivationIntervention(source, 4.0, 2.0, 2.0),)
    execution = InterventionExecutionRequest(
        TraceIdentity(
            "diagnostic-scope",
            "graph",
            "provider",
            "semantic",
            "execution",
            (1, 2, 3),
            3,
            4,
        ),
        TargetState(),
        (
            InterventionVariant(
                "no_op",
                VariantKind.NO_OP,
                InterventionSemantics.DIRECT_FROZEN,
                (),
                0.0,
            ),
            InterventionVariant(
                "direct",
                VariantKind.DIRECT_DOUBLE,
                InterventionSemantics.DIRECT_FROZEN,
                intervention,
                None,
                predicted,
            ),
            InterventionVariant(
                "propagated",
                VariantKind.NECESSITY_HIGH,
                InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
                intervention,
                None,
                predicted,
            ),
        ),
        (downstream,),
        120.0,
        1.0,
        1.0,
        1.0,
    )
    scope = diagnostic_scope_from_graph_knobs(
        {
            "model_name": "google/gemma-3-12b-it",
            "transcoder_architecture": "plt",
        }
    )

    request = OrderingQualificationRequest.from_execution_requests(
        execution_requests=(execution,),
        scope=scope,
        provenance={"diagnostic_kind": "propagated_ordering_boundary_localization"},
    )

    assert scope == {
        "model_family": "gemma3",
        "provider_architecture": "plt",
        "decoder_output_topology": "same_layer",
        "dtype": "bfloat16",
    }
    assert request.scope.transcoder_architecture == "plt"


def test_invoke_sibling_diagnostic_uses_public_api_and_validates_serialized_receipt() -> None:
    execution_request = SimpleNamespace(identity=SimpleNamespace(trace_id="trace-1"))
    observed: dict[str, object] = {}

    def diagnose(model: object, request: object) -> _FakeDiagnosticReceipt:
        observed["model"] = model
        observed["request"] = request
        return _FakeDiagnosticReceipt()

    def validate(request: object, receipt: object) -> None:
        observed["validated_request"] = request
        observed["validated_receipt"] = receipt

    def validate_serialized(payload: object) -> None:
        observed["validated_serialized"] = payload

    api = SimpleNamespace(
        OrderingQualificationRequest=_FakeRequest,
        diagnose_propagated_ordering=diagnose,
        validate_propagated_ordering_diagnostic_receipt=validate,
        validate_serialized_propagated_ordering_diagnostic_receipt=validate_serialized,
    )
    model = object()

    payload = invoke_sibling_diagnostic(
        api,
        model=model,
        execution_requests=(execution_request,),
        scope={"model_family": "gemma3", "provider_architecture": "plt"},
        provenance={"trace_id": "trace-1"},
    )

    assert payload["status"] == "complete"
    receipt = observed["validated_receipt"]
    assert observed == {
        "model": model,
        "request": SimpleNamespace(kind="ordering-qualification-request"),
        "validated_request": SimpleNamespace(kind="ordering-qualification-request"),
        "validated_receipt": receipt,
        "validated_serialized": payload,
    }
    assert _FakeRequest.captured == {
        "execution_requests": (execution_request,),
        "scope": {"model_family": "gemma3", "provider_architecture": "plt"},
        "provenance": {"trace_id": "trace-1"},
    }


def test_build_diagnostic_receipt_is_non_promoting_and_provenance_bound() -> None:
    sibling = _FakeDiagnosticReceipt().to_dict()
    inputs = {
        "behavioral_report": {
            "path": "/scratch/behavioral.json",
            "sha256": "c" * 64,
            "size_bytes": 100,
        }
    }
    workspace = {
        "workspace_mode": "immutable",
        "read_only": True,
        "manifest_sha256": "d" * 64,
    }
    runtime = {"gpu": {"device_name": "NVIDIA H200"}}

    receipt = build_diagnostic_receipt(
        sibling_receipt=sibling,
        request_binding={"trace_id": "trace-1", "provider_fingerprint": "provider"},
        input_artifacts=inputs,
        workspace_provenance=workspace,
        runtime_environment=runtime,
    )

    assert receipt["schema"] == "propagated_ordering_diagnostic_envelope"
    assert receipt["schema_version"] == 1
    assert receipt["status"] == "complete"
    assert receipt["diagnostic_only_no_runtime_promotion"] is True
    assert receipt["diagnostic_only_no_qualification_claim"] is True
    assert "qualification_only_no_runtime_promotion" not in receipt
    assert receipt["evidence"]["sibling_receipt"] == sibling
    assert receipt["evidence"]["input_artifacts"] == inputs
    assert receipt["evidence"]["workspace"] == workspace
    assert receipt["evidence"]["runtime_environment"] == runtime
    expected = hashlib.sha256(
        json.dumps(
            receipt["evidence"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert receipt["evidence_fingerprint"] == expected


def test_build_diagnostic_receipt_requires_immutable_read_only_snapshot() -> None:
    with pytest.raises(PropagatedOrderingDiagnosticError, match="immutable read-only"):
        build_diagnostic_receipt(
            sibling_receipt=_FakeDiagnosticReceipt().to_dict(),
            request_binding={"trace_id": "trace-1"},
            input_artifacts={},
            workspace_provenance={"workspace_mode": "live", "read_only": False},
            runtime_environment={},
        )


@pytest.mark.parametrize(
    "status", ["qualified", "rejected", "refused", "failed", "unknown"]
)
def test_invoke_sibling_diagnostic_refuses_non_diagnostic_status(status: str) -> None:
    api = SimpleNamespace(
        OrderingQualificationRequest=_FakeRequest,
        diagnose_propagated_ordering=lambda model, request: _FakeDiagnosticReceipt(
            status=status
        ),
        validate_propagated_ordering_diagnostic_receipt=lambda request, receipt: None,
        validate_serialized_propagated_ordering_diagnostic_receipt=lambda payload: None,
    )

    with pytest.raises(PropagatedOrderingDiagnosticError, match="invalid status"):
        invoke_sibling_diagnostic(
            api,
            model=object(),
            execution_requests=(object(),),
            scope={"model_family": "gemma3"},
            provenance={},
        )


def test_write_diagnostic_receipt_is_atomic_and_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import circuit_tracer.verification as sibling_api

    monkeypatch.setattr(
        sibling_api,
        "validate_serialized_propagated_ordering_diagnostic_receipt",
        lambda payload: None,
    )
    destination = tmp_path / "receipts" / "diagnostic.json"
    payload = build_diagnostic_receipt(
        sibling_receipt=_FakeDiagnosticReceipt().to_dict(),
        request_binding={"trace_id": "trace-1"},
        input_artifacts={},
        workspace_provenance={"workspace_mode": "immutable", "read_only": True},
        runtime_environment={},
    )

    write_diagnostic_receipt(destination, payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="already exists"):
        write_diagnostic_receipt(destination, payload)
    assert not list(destination.parent.glob(".*.tmp-*"))


def test_write_diagnostic_receipt_refuses_malformed_or_tampered_envelopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import circuit_tracer.verification as sibling_api

    monkeypatch.setattr(
        sibling_api,
        "validate_serialized_propagated_ordering_diagnostic_receipt",
        lambda payload: None,
    )
    destination = tmp_path / "diagnostic.json"
    with pytest.raises(PropagatedOrderingDiagnosticError, match="schema version"):
        write_diagnostic_receipt(
            destination,
            {
                "schema": "propagated_ordering_diagnostic_envelope",
                "status": "complete",
            },
        )
    assert not destination.exists()

    valid = build_diagnostic_receipt(
        sibling_receipt=_FakeDiagnosticReceipt().to_dict(),
        request_binding={"trace_id": "trace-1"},
        input_artifacts={},
        workspace_provenance={"workspace_mode": "immutable", "read_only": True},
        runtime_environment={},
    )
    tampered = copy.deepcopy(valid)
    tampered["evidence"]["request_binding"]["trace_id"] = "forged"
    with pytest.raises(PropagatedOrderingDiagnosticError, match="evidence_fingerprint"):
        write_diagnostic_receipt(destination, tampered)
    assert not destination.exists()

    missing_marker = copy.deepcopy(valid)
    del missing_marker["diagnostic_only_no_qualification_claim"]
    with pytest.raises(PropagatedOrderingDiagnosticError, match="qualification marker"):
        write_diagnostic_receipt(destination, missing_marker)
    assert not destination.exists()


def test_write_diagnostic_receipt_requires_embedded_sibling_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import circuit_tracer.verification as sibling_api

    def reject(payload: object) -> None:
        raise ValueError("diagnostic fingerprint mismatch")

    monkeypatch.setattr(
        sibling_api,
        "validate_serialized_propagated_ordering_diagnostic_receipt",
        reject,
    )
    payload = build_diagnostic_receipt(
        sibling_receipt=_FakeDiagnosticReceipt().to_dict(),
        request_binding={"trace_id": "trace-1"},
        input_artifacts={},
        workspace_provenance={"workspace_mode": "immutable", "read_only": True},
        runtime_environment={},
    )
    destination = tmp_path / "diagnostic.json"

    with pytest.raises(PropagatedOrderingDiagnosticError, match="integrity failure"):
        write_diagnostic_receipt(destination, payload)
    assert not destination.exists()


def test_granite_launcher_is_single_run_snapshot_bound_h200_diagnostic() -> None:
    path = (
        Path(__file__).parents[1]
        / "slurm/exact_trace_bench/diagnose_nnsight_propagation.granite.sbatch"
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
    assert "diagnose_nnsight_propagation.py" in text
    assert "diagnostic-receipt.json" in text
    assert "for repeat_index" not in text
    assert "--repeat-index" not in text
    assert "qualification-summary.json" not in text
    assert "--compare" not in text
    assert "run-prepared-campaign-workload" not in text
    assert "run-full-answer" not in text
