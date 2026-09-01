"""Project-owned provenance envelope for the NNSight ordering qualification.

The sibling library owns the model-backed qualification and its numerical
receipt.  This module binds that receipt to the immutable paired workspace,
the production trace inputs that selected the cases, and the GPU runtime that
executed them.  It deliberately does not promote the qualified capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

from ..workspace import validate_launch_snapshot

GATE_SCHEMA = "nnsight_ordering_qualification_gate"
GATE_SCHEMA_VERSION = 1
REQUIRED_SIBLING_SCHEMA_VERSION = 3
REQUIRED_QUALIFICATION_CLAIM = (
    "intervened_forward_capture_ordering_with_joint_canonical_feature_projection"
)
REQUIRED_PROPAGATED_MUTATION_POLICY = "graph_pinned_preactivation_v1"
REQUIRED_FEATURE_INPUT_CAPTURE_POLICY = "selected_feature_input_vectors_v1"
REQUIRED_FEATURE_PROJECTION_POLICY = "joint_canonical_provider_preactivation_v1"
REQUIRED_NATIVE_FEATURE_VALUES_POLICY = "diagnostic_only_v1"


class QualificationGateError(ValueError):
    """Qualification evidence is incomplete or cannot support promotion."""


class _SerializableReceipt(Protocol):
    def to_dict(self) -> Mapping[str, Any]: ...


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise QualificationGateError(f"cannot hash qualification input: {path}") from exc
    return digest.hexdigest()


def collect_input_artifacts(paths: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    """Hash every production input that selected or bound qualification cases."""

    artifacts: dict[str, dict[str, object]] = {}
    for label, raw_path in paths.items():
        path = raw_path.resolve()
        if not path.is_file():
            raise QualificationGateError(
                f"qualification input must be a regular file: {label}={path}"
            )
        artifacts[label] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return artifacts


def collect_workspace_provenance(
    *, workspace_root: Path, library_root: Path
) -> dict[str, Any]:
    """Validate the paired immutable snapshot and bind its manifest bytes."""

    provenance = validate_launch_snapshot(
        workspace_root=workspace_root,
        library_root=library_root,
        import_roots=(workspace_root / "src", workspace_root, library_root),
    )
    manifest_path = Path(str(provenance["manifest_path"]))
    return {**provenance, "manifest_sha256": _sha256_file(manifest_path)}


def invoke_sibling_qualification(
    api: Any,
    *,
    model: Any,
    execution_requests: Sequence[Any],
    scope: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> _SerializableReceipt:
    """Call only the sibling's public qualification request and runner API."""

    try:
        request_type = api.OrderingQualificationRequest
        request = request_type.from_execution_requests(
            execution_requests=tuple(execution_requests),
            scope=dict(scope),
            provenance=dict(provenance),
        )
        receipt = api.qualify_nnsight_ordering(model, request)
        validator = api.validate_ordering_qualification_receipt
    except AttributeError as exc:
        raise QualificationGateError(
            "sibling ordering qualification or validation API is unavailable"
        ) from exc
    if not callable(getattr(receipt, "to_dict", None)):
        raise QualificationGateError("sibling qualification receipt lacks to_dict()")
    if not callable(validator):
        raise QualificationGateError(
            "sibling ordering qualification validation API is unavailable"
        )
    validator(request, receipt)
    return cast(_SerializableReceipt, receipt)


def _receipt_payload(receipt: _SerializableReceipt) -> dict[str, Any]:
    try:
        payload = dict(receipt.to_dict())
    except (TypeError, ValueError) as exc:
        raise QualificationGateError(
            "sibling qualification receipt is not a JSON object"
        ) from exc
    status = payload.get("status")
    if hasattr(status, "value"):
        status = status.value
        payload["status"] = status
    if status not in {"qualified", "rejected", "refused"}:
        raise QualificationGateError(
            f"sibling ordering qualification has an invalid status: {status!r}"
        )
    try:
        _canonical(payload)
    except (TypeError, ValueError) as exc:
        raise QualificationGateError(
            "sibling qualification receipt is not canonical JSON"
        ) from exc
    _validate_serialized_sibling_receipt(payload)
    return payload


def _validate_serialized_sibling_receipt(payload: Mapping[str, Any]) -> None:
    try:
        from circuit_tracer.verification import (
            validate_serialized_ordering_qualification_receipt,
        )
    except ImportError as exc:
        raise QualificationGateError(
            "sibling serialized qualification validation API is unavailable"
        ) from exc
    try:
        validate_serialized_ordering_qualification_receipt(payload)
    except (TypeError, ValueError) as exc:
        raise QualificationGateError(
            f"sibling qualification receipt integrity failure: {exc}"
        ) from exc


def build_gate_receipt(
    *,
    sibling_receipt: _SerializableReceipt,
    request_binding: Mapping[str, Any],
    input_artifacts: Mapping[str, Mapping[str, object]],
    workspace_provenance: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    repeat_index: int,
) -> dict[str, Any]:
    """Build a fail-closed, provenance-bound project qualification envelope."""

    if (
        workspace_provenance.get("workspace_mode") != "immutable"
        or workspace_provenance.get("read_only") is not True
    ):
        raise QualificationGateError(
            "ordering qualification requires an immutable read-only snapshot"
        )
    if (
        isinstance(repeat_index, bool)
        or not isinstance(repeat_index, int)
        or repeat_index < 1
    ):
        raise QualificationGateError("repeat_index must be a positive integer")
    sibling_payload = _receipt_payload(sibling_receipt)
    sibling_status = str(sibling_payload["status"])
    evidence = {
        "request_binding": dict(request_binding),
        "input_artifacts": {
            label: dict(value) for label, value in input_artifacts.items()
        },
        "workspace": dict(workspace_provenance),
        "runtime_environment": dict(runtime_environment),
        "repeat_index": repeat_index,
        "sibling_receipt": sibling_payload,
    }
    try:
        evidence_fingerprint = hashlib.sha256(
            _canonical(evidence).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise QualificationGateError(
            "qualification evidence is not canonical JSON"
        ) from exc
    return {
        "schema": GATE_SCHEMA,
        "schema_version": GATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": sibling_status,
        "qualification_only_no_runtime_promotion": True,
        "evidence_fingerprint": evidence_fingerprint,
        "evidence": evidence,
    }


def _validated_repeat_receipt(
    raw_receipt: Mapping[str, Any],
) -> tuple[int, str, Mapping[str, Any]]:
    if (
        raw_receipt.get("schema") != GATE_SCHEMA
        or raw_receipt.get("schema_version") != GATE_SCHEMA_VERSION
        or raw_receipt.get("status") != "qualified"
        or raw_receipt.get("qualification_only_no_runtime_promotion") is not True
    ):
        raise QualificationGateError("repeat is not a qualified gate receipt")
    evidence = raw_receipt.get("evidence")
    if not isinstance(evidence, Mapping):
        raise QualificationGateError("repeat receipt evidence must be an object")
    expected_fingerprint = raw_receipt.get("evidence_fingerprint")
    observed_fingerprint = hashlib.sha256(
        _canonical(evidence).encode("utf-8")
    ).hexdigest()
    if expected_fingerprint != observed_fingerprint:
        raise QualificationGateError("repeat receipt evidence_fingerprint mismatch")
    repeat_index = evidence.get("repeat_index")
    if (
        isinstance(repeat_index, bool)
        or not isinstance(repeat_index, int)
        or repeat_index < 1
    ):
        raise QualificationGateError("repeat receipt has an invalid repeat_index")
    sibling = evidence.get("sibling_receipt")
    if not isinstance(sibling, Mapping):
        raise QualificationGateError("repeat sibling receipt must be an object")
    _validate_serialized_sibling_receipt(sibling)
    request_evidence = sibling.get("request_evidence")
    qualification_policy = (
        request_evidence.get("qualification_policy")
        if isinstance(request_evidence, Mapping)
        else None
    )
    if (
        sibling.get("schema_version") != REQUIRED_SIBLING_SCHEMA_VERSION
        or sibling.get("qualification_claim") != REQUIRED_QUALIFICATION_CLAIM
        or not isinstance(qualification_policy, Mapping)
        or qualification_policy.get("propagated_mutation")
        != REQUIRED_PROPAGATED_MUTATION_POLICY
        or qualification_policy.get("feature_input_capture")
        != REQUIRED_FEATURE_INPUT_CAPTURE_POLICY
        or qualification_policy.get("feature_projection")
        != REQUIRED_FEATURE_PROJECTION_POLICY
        or qualification_policy.get("native_feature_values")
        != REQUIRED_NATIVE_FEATURE_VALUES_POLICY
    ):
        raise QualificationGateError(
            "repeat sibling receipt lacks the required canonical projection policy"
        )
    qualification_fingerprint = sibling.get("qualification_fingerprint")
    if not isinstance(qualification_fingerprint, str) or not qualification_fingerprint:
        raise QualificationGateError(
            "repeat sibling receipt lacks a scientific qualification_fingerprint"
        )
    return repeat_index, qualification_fingerprint, evidence


def _stable_runtime_contract(evidence: Mapping[str, Any]) -> dict[str, Any]:
    runtime_environment = evidence.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping):
        raise QualificationGateError("repeat runtime_environment must be an object")
    stable: dict[str, Any] = {}
    for key in ("python", "runtime", "packages", "gpu"):
        value = runtime_environment.get(key)
        if not isinstance(value, Mapping):
            raise QualificationGateError(
                f"repeat runtime_environment.{key} must be an object"
            )
        stable[key] = dict(value)
    gpu = cast(dict[str, Any], stable["gpu"])
    # These identify the launch instance, not the runtime contract. Device,
    # driver, CUDA visibility, cluster, and partition remain compared.
    gpu.pop("slurm_job_id", None)
    gpu.pop("slurm_job_name", None)
    if "schema_version" in runtime_environment:
        stable["schema_version"] = runtime_environment["schema_version"]
    return stable


def compare_gate_receipts(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require two fresh-process repeats with identical scientific evidence."""

    if len(receipts) != 2:
        raise QualificationGateError(
            "fresh-process qualification requires exactly two repeat receipts"
        )
    validated = [_validated_repeat_receipt(receipt) for receipt in receipts]
    repeat_indices = [item[0] for item in validated]
    if repeat_indices != [1, 2]:
        raise QualificationGateError(
            "fresh-process qualification requires ordered repeat indices [1, 2]"
        )
    scientific_fingerprints = [item[1] for item in validated]
    if len(set(scientific_fingerprints)) != 1:
        raise QualificationGateError(
            "fresh-process scientific fingerprint mismatch"
        )

    binding_payloads = [
        {
            "request_binding": evidence.get("request_binding"),
            "input_artifacts": evidence.get("input_artifacts"),
            "workspace": evidence.get("workspace"),
        }
        for _, _, evidence in validated
    ]
    binding_fingerprints = [
        hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        for payload in binding_payloads
    ]
    if len(set(binding_fingerprints)) != 1:
        raise QualificationGateError(
            "fresh-process qualification input or workspace binding mismatch"
        )

    runtime_contract_fingerprints = [
        hashlib.sha256(
            _canonical(_stable_runtime_contract(evidence)).encode("utf-8")
        ).hexdigest()
        for _, _, evidence in validated
    ]
    if len(set(runtime_contract_fingerprints)) != 1:
        raise QualificationGateError(
            "fresh-process stable runtime contract mismatch"
        )

    evidence = {
        "qualification_fingerprint": scientific_fingerprints[0],
        "binding_fingerprint": binding_fingerprints[0],
        "runtime_contract_fingerprint": runtime_contract_fingerprints[0],
        "repeat_indices": repeat_indices,
        "repeat_gate_fingerprints": [
            str(receipt["evidence_fingerprint"]) for receipt in receipts
        ],
        # Embed the complete children so the summary remains independently
        # inspectable even if paths are moved after the job.
        "repeats": [dict(receipt) for receipt in receipts],
    }
    return {
        "schema": "nnsight_ordering_qualification_repetition_gate",
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "qualified",
        "qualification_only_no_runtime_promotion": True,
        "evidence_fingerprint": hashlib.sha256(
            _canonical(evidence).encode("utf-8")
        ).hexdigest(),
        "evidence": evidence,
    }


def write_gate_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist one receipt and refuse all overwrite attempts."""

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"qualification receipt already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"qualification temporary already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationGateError(f"cannot load repeat receipt: {path}") from exc
    if not isinstance(value, Mapping):
        raise QualificationGateError(f"repeat receipt must be an object: {path}")
    return cast(Mapping[str, Any], value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize fresh-process ordering qualification")
    parser.add_argument("--compare", nargs=2, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = compare_gate_receipts(
        tuple(_load_json_object(path) for path in args.compare)
    )
    write_gate_receipt(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


__all__ = [
    "GATE_SCHEMA",
    "GATE_SCHEMA_VERSION",
    "QualificationGateError",
    "build_gate_receipt",
    "collect_input_artifacts",
    "collect_workspace_provenance",
    "compare_gate_receipts",
    "invoke_sibling_qualification",
    "write_gate_receipt",
]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationGateError as exc:
        raise SystemExit(f"ordering qualification comparison refused: {exc}") from exc
