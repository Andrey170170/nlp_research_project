"""Project-owned provenance envelope for propagated-ordering diagnostics.

The sibling library owns the model-backed diagnostic and numerical evidence.
This module calls only that public diagnostic API and binds its receipt to the
immutable paired workspace, production inputs, and GPU runtime.  Diagnostic
receipts are intentionally separate from ordering-qualification receipts and
cannot qualify or promote a runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

DIAGNOSTIC_ENVELOPE_SCHEMA = "propagated_ordering_diagnostic_envelope"
DIAGNOSTIC_ENVELOPE_SCHEMA_VERSION = 1
_DIAGNOSTIC_STATUSES = frozenset({"complete"})


class PropagatedOrderingDiagnosticError(ValueError):
    """Diagnostic evidence is incomplete, invalid, or promotion-shaped."""


def diagnostic_scope_from_graph_knobs(
    graph_knobs: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the exact scope accepted by the sibling diagnostic request."""

    architecture = graph_knobs.get("transcoder_architecture")
    if architecture != "plt":
        raise PropagatedOrderingDiagnosticError(
            "this diagnostic lane is scoped to same-layer PLT providers"
        )
    model_name = graph_knobs.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise PropagatedOrderingDiagnosticError(
            "graph_knobs.model_name must be a non-empty string"
        )
    if "gemma-3" not in model_name.lower():
        raise PropagatedOrderingDiagnosticError(
            "this diagnostic lane is scoped to Gemma 3"
        )
    return {
        "model_family": "gemma3",
        "provider_architecture": "plt",
        "decoder_output_topology": "same_layer",
        "dtype": "bfloat16",
    }


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def invoke_sibling_diagnostic(
    api: Any,
    *,
    model: Any,
    execution_requests: Sequence[Any],
    scope: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Run and validate only the sibling's public propagation diagnostic API."""

    try:
        request_type = api.OrderingQualificationRequest
        request = request_type.from_execution_requests(
            execution_requests=tuple(execution_requests),
            scope=dict(scope),
            provenance=dict(provenance),
        )
        receipt = api.diagnose_propagated_ordering(model, request)
        validate_receipt = api.validate_propagated_ordering_diagnostic_receipt
        validate_serialized = (
            api.validate_serialized_propagated_ordering_diagnostic_receipt
        )
    except AttributeError as exc:
        raise PropagatedOrderingDiagnosticError(
            "sibling propagated-ordering diagnostic or validation API is unavailable"
        ) from exc
    if not callable(getattr(receipt, "to_dict", None)):
        raise PropagatedOrderingDiagnosticError(
            "sibling diagnostic receipt lacks to_dict()"
        )
    if not callable(validate_receipt) or not callable(validate_serialized):
        raise PropagatedOrderingDiagnosticError(
            "sibling propagated-ordering diagnostic validation API is unavailable"
        )

    validate_receipt(request, receipt)
    try:
        payload = dict(receipt.to_dict())
    except (TypeError, ValueError) as exc:
        raise PropagatedOrderingDiagnosticError(
            "sibling diagnostic receipt is not a JSON object"
        ) from exc
    status = payload.get("status")
    if hasattr(status, "value"):
        status = status.value
        payload["status"] = status
    if status not in _DIAGNOSTIC_STATUSES:
        raise PropagatedOrderingDiagnosticError(
            f"sibling propagated-ordering diagnostic has an invalid status: {status!r}"
        )
    if payload.get("diagnostic_only_no_runtime_promotion") is not True:
        raise PropagatedOrderingDiagnosticError(
            "sibling receipt lacks the diagnostic-only no-promotion marker"
        )
    try:
        _canonical(payload)
    except (TypeError, ValueError) as exc:
        raise PropagatedOrderingDiagnosticError(
            "sibling diagnostic receipt is not canonical JSON"
        ) from exc
    try:
        validate_serialized(payload)
    except (TypeError, ValueError) as exc:
        raise PropagatedOrderingDiagnosticError(
            f"sibling diagnostic receipt integrity failure: {exc}"
        ) from exc
    return payload


def build_diagnostic_receipt(
    *,
    sibling_receipt: Mapping[str, Any],
    request_binding: Mapping[str, Any],
    input_artifacts: Mapping[str, Mapping[str, object]],
    workspace_provenance: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a fail-closed, provenance-bound, non-promoting diagnostic envelope."""

    if (
        workspace_provenance.get("workspace_mode") != "immutable"
        or workspace_provenance.get("read_only") is not True
    ):
        raise PropagatedOrderingDiagnosticError(
            "propagated-ordering diagnostic requires an immutable read-only snapshot"
        )
    sibling_payload = dict(sibling_receipt)
    status = sibling_payload.get("status")
    if status not in _DIAGNOSTIC_STATUSES:
        raise PropagatedOrderingDiagnosticError(
            f"sibling propagated-ordering diagnostic has an invalid status: {status!r}"
        )
    if sibling_payload.get("diagnostic_only_no_runtime_promotion") is not True:
        raise PropagatedOrderingDiagnosticError(
            "sibling receipt lacks the diagnostic-only no-promotion marker"
        )
    evidence = {
        "request_binding": dict(request_binding),
        "input_artifacts": {
            label: dict(value) for label, value in input_artifacts.items()
        },
        "workspace": dict(workspace_provenance),
        "runtime_environment": dict(runtime_environment),
        "sibling_receipt": sibling_payload,
    }
    try:
        evidence_fingerprint = hashlib.sha256(
            _canonical(evidence).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise PropagatedOrderingDiagnosticError(
            "propagated-ordering diagnostic evidence is not canonical JSON"
        ) from exc
    return {
        "schema": DIAGNOSTIC_ENVELOPE_SCHEMA,
        "schema_version": DIAGNOSTIC_ENVELOPE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "diagnostic_only_no_runtime_promotion": True,
        "diagnostic_only_no_qualification_claim": True,
        "evidence_fingerprint": evidence_fingerprint,
        "evidence": evidence,
    }


def validate_serialized_diagnostic_receipt(payload: Mapping[str, Any]) -> None:
    """Validate a persisted project envelope and its embedded sibling receipt."""

    serialized = dict(payload)
    if serialized.get("schema") != DIAGNOSTIC_ENVELOPE_SCHEMA:
        raise PropagatedOrderingDiagnosticError(
            "propagated-ordering diagnostic envelope schema mismatch"
        )
    if serialized.get("schema_version") != DIAGNOSTIC_ENVELOPE_SCHEMA_VERSION:
        raise PropagatedOrderingDiagnosticError(
            "propagated-ordering diagnostic envelope schema version mismatch"
        )
    status = serialized.get("status")
    if status not in _DIAGNOSTIC_STATUSES:
        raise PropagatedOrderingDiagnosticError(
            f"propagated-ordering diagnostic envelope has an invalid status: {status!r}"
        )
    if serialized.get("diagnostic_only_no_runtime_promotion") is not True:
        raise PropagatedOrderingDiagnosticError(
            "propagated-ordering diagnostic envelope promotion marker mismatch"
        )
    if serialized.get("diagnostic_only_no_qualification_claim") is not True:
        raise PropagatedOrderingDiagnosticError(
            "propagated-ordering diagnostic envelope qualification marker mismatch"
        )
    evidence = serialized.get("evidence")
    if not isinstance(evidence, Mapping):
        raise PropagatedOrderingDiagnosticError(
            "propagated-ordering diagnostic envelope evidence must be an object"
        )
    for field in (
        "request_binding",
        "input_artifacts",
        "workspace",
        "runtime_environment",
        "sibling_receipt",
    ):
        if not isinstance(evidence.get(field), Mapping):
            raise PropagatedOrderingDiagnosticError(
                f"propagated-ordering diagnostic evidence.{field} must be an object"
            )
    workspace = cast(Mapping[str, Any], evidence["workspace"])
    if (
        workspace.get("workspace_mode") != "immutable"
        or workspace.get("read_only") is not True
    ):
        raise PropagatedOrderingDiagnosticError(
            "propagated-ordering diagnostic envelope lacks an immutable read-only snapshot"
        )
    try:
        observed_fingerprint = hashlib.sha256(
            _canonical(evidence).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise PropagatedOrderingDiagnosticError(
            "propagated-ordering diagnostic envelope evidence is not canonical JSON"
        ) from exc
    if serialized.get("evidence_fingerprint") != observed_fingerprint:
        raise PropagatedOrderingDiagnosticError(
            "propagated-ordering diagnostic evidence_fingerprint mismatch"
        )
    sibling = cast(Mapping[str, Any], evidence["sibling_receipt"])
    if sibling.get("status") != status:
        raise PropagatedOrderingDiagnosticError(
            "propagated-ordering diagnostic sibling status mismatch"
        )
    try:
        from circuit_tracer.verification import (
            validate_serialized_propagated_ordering_diagnostic_receipt,
        )
    except ImportError as exc:
        raise PropagatedOrderingDiagnosticError(
            "sibling serialized propagated-ordering diagnostic validation API is unavailable"
        ) from exc
    try:
        validate_serialized_propagated_ordering_diagnostic_receipt(sibling)
    except (TypeError, ValueError) as exc:
        raise PropagatedOrderingDiagnosticError(
            f"sibling diagnostic receipt integrity failure: {exc}"
        ) from exc


def write_diagnostic_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist one diagnostic receipt and refuse overwrites."""

    validate_serialized_diagnostic_receipt(payload)
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"diagnostic receipt already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"diagnostic temporary already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        try:
            persisted = json.loads(temporary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PropagatedOrderingDiagnosticError(
                "cannot reload temporary propagated-ordering diagnostic receipt"
            ) from exc
        if not isinstance(persisted, Mapping):
            raise PropagatedOrderingDiagnosticError(
                "temporary propagated-ordering diagnostic receipt is not an object"
            )
        validate_serialized_diagnostic_receipt(persisted)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DIAGNOSTIC_ENVELOPE_SCHEMA",
    "DIAGNOSTIC_ENVELOPE_SCHEMA_VERSION",
    "PropagatedOrderingDiagnosticError",
    "build_diagnostic_receipt",
    "diagnostic_scope_from_graph_knobs",
    "invoke_sibling_diagnostic",
    "validate_serialized_diagnostic_receipt",
    "write_diagnostic_receipt",
]
