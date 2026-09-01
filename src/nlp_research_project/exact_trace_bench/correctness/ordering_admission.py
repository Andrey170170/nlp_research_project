"""Fail-closed, exact-scope admission of qualified NNSight ordering.

The sibling runtime deliberately defaults this capability to disabled.  This
module is the sole project-owned seam that verifies an immutable qualification
summary and enables the capability on one loaded model for one behavioral
verification call.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch

from ..runtime_provenance import capture_runtime_environment
from ..workspace import validate_launch_snapshot
from .ordering_qualification_runner import (
    QualificationGateError,
    compare_gate_receipts,
)

SUMMARY_PATH_KNOB = "correctness_ordering_qualification_summary_path"
SUMMARY_SHA256_KNOB = "correctness_ordering_qualification_summary_sha256"
_ADMISSION_ATTRIBUTE = "verification_intervened_capture_ordering_qualified"
_SUMMARY_SCHEMA = "nnsight_ordering_qualification_repetition_gate"
_SUMMARY_SCHEMA_VERSION = 1
_EXPECTED_SCOPE = {
    "model_family": "gemma3",
    "provider_architecture": "plt",
    "decoder_output_topology": "same_layer",
    "dtype": "bfloat16",
}
_REQUESTED_PROVIDER_FIELDS = (
    "model_name",
    "transcoder_architecture",
    "transcoder_provider_family",
    "repo_id",
    "revision",
    "clt_subfolder",
    "plt_subfolder_template",
    "layer_count",
    "feature_input_hook",
    "feature_output_hook",
    "lazy_encoder",
    "lazy_decoder",
    "decoder_chunk_size",
    "cross_batch_decoder_cache_bytes",
    "checkpoint_asset_scope",
    "checkpoint_prefault_budget_bytes",
    "transcoder_cache_dir",
)


class OrderingAdmissionError(ValueError):
    """Qualification evidence cannot admit the selected runtime."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise OrderingAdmissionError(
            f"cannot read ordering qualification summary: {path}"
        ) from error
    return "sha256:" + digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OrderingAdmissionError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OrderingAdmissionError(f"{label} must be a non-empty string")
    return value


def ordering_qualification_declaration_from_graph_knobs(
    graph_knobs: Mapping[str, Any],
) -> dict[str, str] | None:
    """Return a paired immutable declaration or reject an ambiguous selection."""

    raw_path = graph_knobs.get(SUMMARY_PATH_KNOB)
    raw_sha256 = graph_knobs.get(SUMMARY_SHA256_KNOB)
    if raw_path is None and raw_sha256 is None:
        return None
    if raw_path is None or raw_sha256 is None:
        raise OrderingAdmissionError(
            "ordering qualification summary path and sha256 must be declared together"
        )
    path = Path(_string(raw_path, SUMMARY_PATH_KNOB))
    if not path.is_absolute():
        raise OrderingAdmissionError(
            "ordering qualification summary path must be absolute"
        )
    sha256 = _string(raw_sha256, SUMMARY_SHA256_KNOB)
    if (
        not sha256.startswith("sha256:")
        or len(sha256) != 71
        or any(character not in "0123456789abcdef" for character in sha256[7:])
    ):
        raise OrderingAdmissionError(
            "ordering qualification summary sha256 must be a sha256 fingerprint"
        )
    return {"path": str(path.resolve()), "sha256": sha256}


def prepare_ordering_qualification_declaration(path: Path) -> dict[str, str]:
    """Hash and structurally validate a qualification summary for preparation."""

    resolved = path.resolve()
    _, _, sha256 = _load_validated_summary(resolved, expected_sha256=None)
    return {"manifest_path": str(resolved), "manifest_sha256": sha256}


def _read_summary_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise OrderingAdmissionError(
            f"cannot load ordering qualification summary: {path}"
        ) from error


def _load_validated_summary(
    path: Path, *, expected_sha256: str | None
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    source = _read_summary_bytes(path)
    observed_sha256 = "sha256:" + hashlib.sha256(source).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise OrderingAdmissionError("ordering qualification summary sha256 mismatch")
    try:
        summary = _mapping(json.loads(source), "ordering qualification summary")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OrderingAdmissionError(
            f"cannot load ordering qualification summary: {path}"
        ) from error
    if (
        summary.get("schema") != _SUMMARY_SCHEMA
        or summary.get("schema_version") != _SUMMARY_SCHEMA_VERSION
        or summary.get("status") != "qualified"
        or summary.get("qualification_only_no_runtime_promotion") is not True
    ):
        raise OrderingAdmissionError(
            "ordering qualification summary is not a qualified non-promoting v1 gate"
        )
    evidence = _mapping(summary.get("evidence"), "summary.evidence")
    if (
        summary.get("evidence_fingerprint")
        != hashlib.sha256(_canonical(evidence).encode("utf-8")).hexdigest()
    ):
        raise OrderingAdmissionError(
            "ordering qualification summary evidence_fingerprint mismatch"
        )
    repeats = evidence.get("repeats")
    if not isinstance(repeats, Sequence) or isinstance(repeats, (str, bytes)):
        raise OrderingAdmissionError("summary.evidence.repeats must be an array")
    try:
        recomputed = compare_gate_receipts(
            tuple(_mapping(item, "summary repeat") for item in repeats)
        )
    except (QualificationGateError, TypeError, ValueError) as error:
        raise OrderingAdmissionError(
            f"ordering qualification embedded receipts failed validation: {error}"
        ) from error
    recomputed_evidence = _mapping(
        recomputed.get("evidence"), "recomputed summary.evidence"
    )
    if _canonical(recomputed_evidence) != _canonical(evidence):
        raise OrderingAdmissionError(
            "ordering qualification summary evidence does not exactly recompute"
        )
    return summary, evidence, observed_sha256


def _normalized_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    python = _mapping(runtime.get("python"), "runtime.python")
    gpu = _mapping(runtime.get("gpu"), "runtime.gpu")
    raw_gpus = gpu.get("gpus")
    if not isinstance(raw_gpus, Sequence) or isinstance(raw_gpus, (str, bytes)):
        raise OrderingAdmissionError("runtime.gpu.gpus must be an array")
    gpus = []
    for index, raw_gpu in enumerate(raw_gpus):
        item = _mapping(raw_gpu, f"runtime.gpu.gpus[{index}]")
        gpus.append(
            {
                key: item.get(key)
                for key in ("name", "driver_version", "memory_total_mib")
            }
        )
    return {
        "python": {
            "version": python.get("version"),
            "implementation": python.get("implementation"),
        },
        "runtime": dict(_mapping(runtime.get("runtime"), "runtime.runtime")),
        "packages": dict(_mapping(runtime.get("packages"), "runtime.packages")),
        "gpu": {
            "count": len(gpus),
            "gpus": sorted(gpus, key=_canonical),
            "slurm_cluster_name": gpu.get("slurm_cluster_name"),
            "slurm_job_partition": gpu.get("slurm_job_partition"),
            "nvidia_smi_returncode": gpu.get("nvidia_smi_returncode"),
        },
    }


def _current_workspace_provenance() -> Mapping[str, Any]:
    workspace_raw = os.environ.get("WORKSPACE_ROOT")
    library_raw = os.environ.get("LIB_WORKSPACE_ROOT")
    if not workspace_raw or not library_raw:
        raise OrderingAdmissionError(
            "WORKSPACE_ROOT and LIB_WORKSPACE_ROOT are required for ordering admission"
        )
    workspace_root = Path(workspace_raw).resolve()
    library_root = Path(library_raw).resolve()
    try:
        return validate_launch_snapshot(
            workspace_root=workspace_root,
            library_root=library_root,
            import_roots=(workspace_root / "src", workspace_root, library_root),
        )
    except (OSError, TypeError, ValueError) as error:
        raise OrderingAdmissionError(
            f"current ordering-admission workspace is not immutable: {error}"
        ) from error


def _provider_identity(
    *,
    graph_knobs: Mapping[str, Any],
    transcoder_metadata: Mapping[str, Any],
    request_binding: Mapping[str, Any],
    model: Any,
) -> dict[str, Any]:
    requested = _mapping(transcoder_metadata.get("requested"), "transcoder.requested")
    selected_requested = {
        key: graph_knobs.get(key) for key in _REQUESTED_PROVIDER_FIELDS
    }
    if selected_requested != dict(requested):
        raise OrderingAdmissionError(
            "selected graph knobs do not match loaded requested provider metadata"
        )
    requested_fingerprint = _fingerprint(requested)
    if request_binding.get("provider_fingerprint") != requested_fingerprint:
        raise OrderingAdmissionError("requested provider fingerprint mismatch")
    loaded_fingerprint = _fingerprint(transcoder_metadata)
    if request_binding.get("loaded_provider_fingerprint") != loaded_fingerprint:
        raise OrderingAdmissionError(
            "loaded full provider metadata fingerprint mismatch"
        )
    expected_model = _string(request_binding.get("model_name"), "binding.model_name")
    expected_family = _string(
        request_binding.get("provider_family"), "binding.provider_family"
    )
    if (
        graph_knobs.get("model_name") != expected_model
        or graph_knobs.get("transcoder_provider_family") != expected_family
        or graph_knobs.get("transcoder_architecture") != "plt"
        or "gemma-3" not in expected_model.lower()
    ):
        raise OrderingAdmissionError(
            "model or provider family is outside qualified scope"
        )
    scope = _mapping(
        request_binding.get("qualification_scope"), "binding.qualification_scope"
    )
    if dict(scope) != _EXPECTED_SCOPE:
        raise OrderingAdmissionError("qualification scope is not exact Gemma3 PLT BF16")
    detected = _mapping(transcoder_metadata.get("detected"), "transcoder.detected")
    capabilities = _mapping(
        detected.get("capabilities"), "transcoder.detected.capabilities"
    )
    provider_fingerprint = _mapping(
        detected.get("provider_fingerprint"),
        "transcoder.detected.provider_fingerprint",
    )
    if (
        capabilities.get("architecture") != "plt"
        or capabilities.get("decoder_output_topology") != "same_layer"
        or provider_fingerprint.get("architecture") != "plt"
        or provider_fingerprint.get("decoder_output_topology") != "same_layer"
        or provider_fingerprint.get("dtype") != "torch.bfloat16"
        or getattr(model, "dtype", None) is not torch.bfloat16
    ):
        raise OrderingAdmissionError(
            "loaded model/provider is not exact same-layer PLT BF16"
        )
    transcoders = getattr(model, "transcoders", None)
    provider = getattr(transcoders, "_module", transcoders)
    if getattr(provider, "dtype", None) is not torch.bfloat16:
        raise OrderingAdmissionError("loaded provider dtype is not bfloat16")
    return {
        "model_name": expected_model,
        "provider_family": expected_family,
        "requested_provider_fingerprint": requested_fingerprint,
        "loaded_provider_fingerprint": loaded_fingerprint,
        **_EXPECTED_SCOPE,
    }


def _workspace_identity(
    *, qualified: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    qualified_library = _mapping(
        qualified.get("library_repo_state"), "qualified workspace library_repo_state"
    )
    current_library = _mapping(
        current.get("library_repo_state"), "current workspace library_repo_state"
    )
    current_project = _mapping(
        current.get("project_repo_state"), "current workspace project_repo_state"
    )
    if (
        qualified.get("workspace_mode") != "immutable"
        or qualified.get("read_only") is not True
    ):
        raise OrderingAdmissionError(
            "qualified workspace is not immutable and read-only"
        )
    if (
        current.get("workspace_mode") != "immutable"
        or current.get("read_only") is not True
    ):
        raise OrderingAdmissionError("current workspace is not immutable and read-only")
    qualified_commit = _string(
        qualified_library.get("commit"), "qualified sibling commit"
    )
    current_commit = _string(current_library.get("commit"), "current sibling commit")
    if current_commit != qualified_commit:
        raise OrderingAdmissionError(
            "current sibling commit does not match qualification"
        )
    if qualified_library.get("dirty_files") not in (None, []):
        raise OrderingAdmissionError("qualified sibling snapshot records dirty files")
    if current_library.get("dirty_files") not in (None, []):
        raise OrderingAdmissionError(
            "current immutable sibling snapshot records dirty files"
        )
    current_project_commit = _string(
        current_project.get("commit"), "current project commit"
    )
    if current_project.get("dirty_files") not in (None, []):
        raise OrderingAdmissionError(
            "current immutable project snapshot records dirty files"
        )
    qualified_manifest_sha256 = _string(
        qualified.get("manifest_sha256"), "qualified workspace manifest_sha256"
    )
    current_manifest_path = Path(
        _string(current.get("manifest_path"), "current workspace manifest_path")
    )
    return {
        "qualified_sibling_commit": qualified_commit,
        "current_sibling_commit": current_commit,
        "current_project_commit": current_project_commit,
        "qualified_manifest_sha256": qualified_manifest_sha256,
        "current_manifest_path": str(current_manifest_path.resolve()),
        "current_manifest_sha256": _sha256_file(current_manifest_path),
        "current_workspace_mode": "immutable",
        "current_read_only": True,
    }


@contextmanager
def admitted_nnsight_ordering(
    *,
    model: Any,
    graph_knobs: Mapping[str, Any],
    transcoder_metadata: Mapping[str, Any],
    execution_fingerprint: str,
) -> Iterator[dict[str, Any]]:
    """Temporarily admit one model after exact qualification revalidation."""

    current_execution_fingerprint = _string(
        execution_fingerprint, "behavioral execution_fingerprint"
    )
    declaration = ordering_qualification_declaration_from_graph_knobs(graph_knobs)
    if declaration is None:
        raise OrderingAdmissionError(
            "required ordering qualification summary is not declared"
        )
    path = Path(declaration["path"])
    summary, evidence, _ = _load_validated_summary(
        path, expected_sha256=declaration["sha256"]
    )
    repeats = cast(Sequence[Mapping[str, Any]], evidence["repeats"])
    first_repeat_evidence = _mapping(
        repeats[0].get("evidence"), "summary first repeat evidence"
    )
    request_binding = _mapping(
        first_repeat_evidence.get("request_binding"), "qualified request_binding"
    )
    scope = _provider_identity(
        graph_knobs=graph_knobs,
        transcoder_metadata=transcoder_metadata,
        request_binding=request_binding,
        model=model,
    )
    qualified_runtime = _mapping(
        first_repeat_evidence.get("runtime_environment"), "qualified runtime"
    )
    current_runtime = capture_runtime_environment()
    qualified_runtime_contract = _normalized_runtime(qualified_runtime)
    current_runtime_contract = _normalized_runtime(current_runtime)
    if current_runtime_contract != qualified_runtime_contract:
        raise OrderingAdmissionError(
            "current runtime stack does not match qualification"
        )
    qualified_workspace = _mapping(
        first_repeat_evidence.get("workspace"), "qualified workspace"
    )
    workspace = _workspace_identity(
        qualified=qualified_workspace,
        current=_current_workspace_provenance(),
    )

    model_vars = getattr(model, "__dict__", {})
    if _ADMISSION_ATTRIBUTE in model_vars:
        raise OrderingAdmissionError("model instance is already ordering-admitted")
    if getattr(type(model), _ADMISSION_ATTRIBUTE, None) is not False:
        raise OrderingAdmissionError(
            "model class ordering admission default is not false"
        )
    bounded_evidence = {
        "source": dict(declaration),
        "summary_evidence_fingerprint": summary["evidence_fingerprint"],
        "binding_fingerprint": evidence["binding_fingerprint"],
        "qualification_fingerprint": evidence["qualification_fingerprint"],
        "behavioral_execution_fingerprint": current_execution_fingerprint,
        "scope": scope,
        "runtime": {
            "qualified_summary_runtime_contract_fingerprint": evidence[
                "runtime_contract_fingerprint"
            ],
            "normalized_runtime_fingerprint": _fingerprint(current_runtime_contract),
        },
        "workspace": workspace,
    }
    setattr(model, _ADMISSION_ATTRIBUTE, True)
    try:
        yield bounded_evidence
    finally:
        try:
            delattr(model, _ADMISSION_ATTRIBUTE)
        except AttributeError:
            pass


__all__ = [
    "OrderingAdmissionError",
    "SUMMARY_PATH_KNOB",
    "SUMMARY_SHA256_KNOB",
    "admitted_nnsight_ordering",
    "ordering_qualification_declaration_from_graph_knobs",
    "prepare_ordering_qualification_declaration",
]
