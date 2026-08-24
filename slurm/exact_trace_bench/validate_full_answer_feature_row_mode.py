#!/usr/bin/env python3
"""Validate required execution mechanisms of a completed full-answer shard."""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

from circuit_tracer.observability.device_timing import (
    PHASE4_CUDA_ACCOUNTING_SCOPE_V2,
    PHASE4_CUDA_ESTIMATOR_V2,
    PHASE4_CUDA_INTERVAL_SEMANTICS,
    PHASE4_CUDA_SAMPLING_HASH,
    PHASE4_CUDA_SAMPLING_SCHEME_V2,
    PHASE4_CUDA_SAMPLING_SEED,
    PHASE4_DEVICE_TIMING_CONTRACT_V1,
    PHASE4_DEVICE_TIMING_CONTRACT_V2,
    PHASE4_OPTIONAL_TIMING_SUBSTAGES,
    PHASE4_REQUIRED_TIMING_SUBSTAGES,
    PHASE4_TIMING_SAMPLING_POLICIES,
)
from nlp_research_project.exact_trace_bench.backward_selection import (
    backward_mechanism_record,
)
from nlp_research_project.exact_trace_bench.compact_io import (
    CANONICAL_TYPED_BUCKET_NAMES,
    load_compact_graph,
    summarize_feature_positions,
)
from nlp_research_project.exact_trace_bench.trace_runtime.artifacts import (
    load_and_validate_capture_artifact,
)


class FeatureRowModeValidationError(RuntimeError):
    """Raised when a completed shard did not use a required mechanism."""


_CAPTURE_SIDECAR_KNOBS = {
    "phase0_donor_bundle": "capture_phase0_donor_bundle",
    "phase3_seed_bundle": "capture_phase3_seed_bundle",
    "phase3_gradient_bundle": "capture_phase3_gradient_bundle",
    "phase3_row_bundle": "capture_phase3_row_bundle",
    "feature_semantic_descriptors": "capture_feature_semantic_descriptors",
}

_PHASE4_CORE_DEVICE_TIMING_SUBSTAGES = tuple(
    substage.value for substage in PHASE4_REQUIRED_TIMING_SUBSTAGES
)
_PHASE4_OPTIONAL_DEVICE_TIMING_SUBSTAGES = tuple(
    substage.value for substage in PHASE4_OPTIONAL_TIMING_SUBSTAGES
)
_PHASE4_TIMING_POLICY_BY_NAME = {
    substage.value: policy
    for substage, policy in PHASE4_TIMING_SAMPLING_POLICIES.items()
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FeatureRowModeValidationError(f"expected a JSON object: {path}")
    return payload


def _required_nonempty_string(
    attrs: dict[str, Any], field: str, *, label: str, source: Path
) -> str:
    value = attrs.get(field)
    if not isinstance(value, str) or not value:
        raise FeatureRowModeValidationError(
            f"{label} has invalid {field}={value!r}: {source}"
        )
    return value


def _required_finite_number(
    attrs: dict[str, Any],
    field: str,
    *,
    label: str,
    source: Path,
    nonnegative: bool = False,
) -> float:
    value = attrs.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise FeatureRowModeValidationError(
            f"{label} has invalid numeric {field}={value!r}: {source}"
        )
    result = float(value)
    if nonnegative and result < 0:
        raise FeatureRowModeValidationError(
            f"{label} has negative {field}={value!r}: {source}"
        )
    return result


def _canonical_gpu_uuid(value: str) -> str:
    """Normalize Torch and nvidia-smi spellings of one physical GPU UUID."""

    normalized = value.strip().casefold()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    if not normalized:
        raise ValueError("GPU UUID must contain an identity after normalization")
    return normalized


def _validate_structured_vjp_evidence(
    *,
    compute_batch_events: list[tuple[Path, dict[str, Any]]],
    expected_vjp_kernel_mode: str | None,
    label: str,
) -> dict[str, Any]:
    if not compute_batch_events:
        raise FeatureRowModeValidationError(
            f"{label} has no context.compute_batch telemetry for structured VJP evidence"
        )
    requested_paths: set[str] = set()
    effective_invocations: set[str] = set()
    is_grads_batched_values: set[bool] = set()
    fallback_states: set[str] = set()
    observation_methods: set[str] = set()
    for source, attrs in compute_batch_events:
        requested_path = _required_nonempty_string(
            attrs, "vjp_requested_path", label=label, source=source
        )
        effective_invocation = _required_nonempty_string(
            attrs, "vjp_effective_invocation", label=label, source=source
        )
        is_grads_batched = attrs.get("vjp_is_grads_batched")
        if not isinstance(is_grads_batched, bool):
            raise FeatureRowModeValidationError(
                f"{label} has invalid vjp_is_grads_batched={is_grads_batched!r}: "
                f"{source}"
            )
        fallback_state = _required_nonempty_string(
            attrs, "vjp_fallback_state", label=label, source=source
        )
        observation_method = _required_nonempty_string(
            attrs, "vjp_fallback_observation_method", label=label, source=source
        )
        if effective_invocation != "torch.autograd.grad":
            raise FeatureRowModeValidationError(
                f"{label} effective VJP invocation is {effective_invocation!r}; "
                f"expected 'torch.autograd.grad': {source}"
            )
        expected_is_batched = requested_path == "autograd_batched"
        if is_grads_batched is not expected_is_batched:
            raise FeatureRowModeValidationError(
                f"{label} VJP evidence is inconsistent: requested_path="
                f"{requested_path!r}, is_grads_batched={is_grads_batched!r}: {source}"
            )
        if expected_is_batched:
            if fallback_state != "unknown":
                raise FeatureRowModeValidationError(
                    f"{label} batched VJP fallback state is {fallback_state!r}; "
                    f"expected the honest observable state 'unknown': {source}"
                )
            if observation_method != "direct_call_contract_and_success":
                raise FeatureRowModeValidationError(
                    f"{label} batched VJP fallback observation method is "
                    f"{observation_method!r}; expected "
                    f"'direct_call_contract_and_success': {source}"
                )
            fallback_reason = _required_nonempty_string(
                attrs, "vjp_fallback_state_reason", label=label, source=source
            )
            if fallback_reason != (
                "pytorch_has_no_programmatic_per_invocation_vmap_fallback_signal"
            ):
                raise FeatureRowModeValidationError(
                    f"{label} batched VJP unknown fallback reason is "
                    f"{fallback_reason!r}: {source}"
                )
        else:
            if fallback_state != "not_applicable":
                raise FeatureRowModeValidationError(
                    f"{label} serial VJP fallback state is {fallback_state!r}; "
                    f"expected 'not_applicable': {source}"
                )
            if observation_method != "direct_call_contract_and_success":
                raise FeatureRowModeValidationError(
                    f"{label} serial VJP observation method is "
                    f"{observation_method!r}; expected "
                    f"'direct_call_contract_and_success': {source}"
                )
        requested_paths.add(requested_path)
        effective_invocations.add(effective_invocation)
        is_grads_batched_values.add(is_grads_batched)
        fallback_states.add(fallback_state)
        observation_methods.add(observation_method)

    if expected_vjp_kernel_mode is not None and requested_paths != {
        expected_vjp_kernel_mode
    }:
        raise FeatureRowModeValidationError(
            f"{label} structured VJP requested paths {sorted(requested_paths)!r}; "
            f"expected only {expected_vjp_kernel_mode!r}"
        )
    for field, values in (
        ("effective invocations", effective_invocations),
        ("is_grads_batched values", is_grads_batched_values),
        ("fallback states", fallback_states),
        ("fallback observation methods", observation_methods),
    ):
        if len(values) != 1:
            raise FeatureRowModeValidationError(
                f"{label} has inconsistent structured VJP {field}: {sorted(values)!r}"
            )
    return {
        "event_count": len(compute_batch_events),
        "requested_paths": sorted(requested_paths),
        "effective_invocations": sorted(effective_invocations),
        "is_grads_batched_values": sorted(is_grads_batched_values),
        "fallback_states": sorted(fallback_states),
        "fallback_observation_methods": sorted(observation_methods),
    }


def _validate_phase4_device_timing(
    *,
    phase4_events: list[tuple[Path, dict[str, Any]]],
    expected_backend: str | None,
    label: str,
) -> dict[str, Any]:
    if not phase4_events:
        raise FeatureRowModeValidationError(
            f"{label} has no phase4.feature_attribution device-timing telemetry"
        )
    backends: set[str] = set()
    device_ordinals: set[int] = set()
    device_uuids: set[str] = set()
    observed_substages: set[str] = set()
    optional_substage_presence = {
        substage: False for substage in _PHASE4_OPTIONAL_DEVICE_TIMING_SUBSTAGES
    }
    for source, attrs in phase4_events:
        contract_version = attrs.get(
            "phase4_timing_contract_version", PHASE4_DEVICE_TIMING_CONTRACT_V1
        )
        if contract_version not in {
            PHASE4_DEVICE_TIMING_CONTRACT_V1,
            PHASE4_DEVICE_TIMING_CONTRACT_V2,
        }:
            raise FeatureRowModeValidationError(
                f"{label} Phase-4 timing contract is unsupported: "
                f"{contract_version!r}: {source}"
            )
        backend = _required_nonempty_string(
            attrs, "phase4_timing_backend", label=label, source=source
        )
        if expected_backend is not None and backend != expected_backend:
            raise FeatureRowModeValidationError(
                f"{label} Phase-4 timing backend is {backend!r}; expected "
                f"{expected_backend!r}: {source}"
            )
        device = _required_nonempty_string(
            attrs, "phase4_timing_device", label=label, source=source
        )
        if not device.startswith("cuda"):
            raise FeatureRowModeValidationError(
                f"{label} Phase-4 timing device is not CUDA: {device!r}: {source}"
            )
        ordinal = attrs.get("phase4_timing_cuda_visible_device_ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
            raise FeatureRowModeValidationError(
                f"{label} has invalid Phase-4 CUDA device ordinal {ordinal!r}: {source}"
            )
        device_uuid = _required_nonempty_string(
            attrs, "phase4_timing_cuda_device_uuid", label=label, source=source
        )
        if attrs.get("phase4_timing_unavailable_reason") is not None:
            raise FeatureRowModeValidationError(
                f"{label} Phase-4 CUDA timing is marked unavailable: {source}"
            )
        if (
            attrs.get("phase4_timing_cuda_device_identity_unavailable_reason")
            is not None
        ):
            raise FeatureRowModeValidationError(
                f"{label} Phase-4 CUDA device identity is marked unavailable: {source}"
            )
        _required_nonempty_string(
            attrs, "phase4_timing_synchronization_scope", label=label, source=source
        )
        accounting_scope = _required_nonempty_string(
            attrs, "phase4_timing_accounting_scope", label=label, source=source
        )
        expected_accounting_scope = (
            "non_overlapping_ranges_stratified_cuda_estimate_v1"
            if contract_version == PHASE4_DEVICE_TIMING_CONTRACT_V1
            else PHASE4_CUDA_ACCOUNTING_SCOPE_V2
        )
        if accounting_scope != expected_accounting_scope:
            raise FeatureRowModeValidationError(
                f"{label} Phase-4 timing accounting scope is "
                f"{accounting_scope!r}: {source}"
            )
        sampling_scheme = _required_nonempty_string(
            attrs, "phase4_timing_cuda_sampling_scheme", label=label, source=source
        )
        sampling_hash = _required_nonempty_string(
            attrs, "phase4_timing_cuda_sampling_hash", label=label, source=source
        )
        estimator = _required_nonempty_string(
            attrs, "phase4_timing_cuda_estimator", label=label, source=source
        )
        interval_semantics = _required_nonempty_string(
            attrs,
            "phase4_timing_cuda_interval_semantics",
            label=label,
            source=source,
        )
        expected_sampling_scheme = (
            "deterministic_blake2b_jittered_offset_per_stride_block_v1"
            if contract_version == PHASE4_DEVICE_TIMING_CONTRACT_V1
            else PHASE4_CUDA_SAMPLING_SCHEME_V2
        )
        expected_estimator = (
            "horvitz_thompson_stride_weight_v1"
            if contract_version == PHASE4_DEVICE_TIMING_CONTRACT_V1
            else PHASE4_CUDA_ESTIMATOR_V2
        )
        if (
            sampling_scheme != expected_sampling_scheme
            or sampling_hash != PHASE4_CUDA_SAMPLING_HASH
            or estimator != expected_estimator
            or interval_semantics != PHASE4_CUDA_INTERVAL_SEMANTICS
        ):
            raise FeatureRowModeValidationError(
                f"{label} Phase-4 CUDA sampling contract is unsupported: "
                f"scheme={sampling_scheme!r}, hash={sampling_hash!r}, "
                f"estimator={estimator!r}, interval_semantics="
                f"{interval_semantics!r}: {source}"
            )
        sampling_seed = attrs.get("phase4_timing_cuda_sampling_seed")
        if (
            not isinstance(sampling_seed, int)
            or isinstance(sampling_seed, bool)
            or sampling_seed != PHASE4_CUDA_SAMPLING_SEED
        ):
            raise FeatureRowModeValidationError(
                f"{label} has invalid Phase-4 CUDA sampling seed "
                f"{sampling_seed!r}; expected {PHASE4_CUDA_SAMPLING_SEED}: "
                f"{source}"
            )
        if contract_version == PHASE4_DEVICE_TIMING_CONTRACT_V2:
            for substage in _PHASE4_CORE_DEVICE_TIMING_SUBSTAGES:
                status = attrs.get(f"phase4_timing_{substage}_cuda_estimate_status")
                if status != "complete":
                    raise FeatureRowModeValidationError(
                        f"{label} Phase-4 timing estimate for {substage!r} "
                        f"refuses an incomplete final stratum: status={status!r}: "
                        f"{source}"
                    )
        wall_lifecycle = _required_finite_number(
            attrs,
            "phase4_timing_lifecycle_wall_elapsed_ms",
            label=label,
            source=source,
            nonnegative=True,
        )
        cuda_lifecycle = _required_finite_number(
            attrs,
            "phase4_timing_lifecycle_cuda_event_elapsed_ms",
            label=label,
            source=source,
            nonnegative=True,
        )
        wall_accounted = _required_finite_number(
            attrs,
            "phase4_timing_wall_accounted_elapsed_ms",
            label=label,
            source=source,
            nonnegative=True,
        )
        wall_residual = _required_finite_number(
            attrs,
            "phase4_timing_wall_residual_elapsed_ms",
            label=label,
            source=source,
            nonnegative=True,
        )
        cuda_accounted = _required_finite_number(
            attrs,
            "phase4_timing_cuda_estimated_accounted_elapsed_ms",
            label=label,
            source=source,
            nonnegative=True,
        )
        cuda_residual = _required_finite_number(
            attrs,
            "phase4_timing_cuda_estimated_residual_elapsed_ms",
            label=label,
            source=source,
        )
        for clock, lifecycle, accounted, residual in (
            ("wall", wall_lifecycle, wall_accounted, wall_residual),
            ("CUDA-event", cuda_lifecycle, cuda_accounted, cuda_residual),
        ):
            if not math.isclose(
                lifecycle, accounted + residual, rel_tol=1e-6, abs_tol=1e-3
            ):
                raise FeatureRowModeValidationError(
                    f"{label} Phase-4 {clock} timing does not reconcile: "
                    f"lifecycle={lifecycle}, accounted={accounted}, "
                    f"residual={residual}: {source}"
                )
        for substage in _PHASE4_CORE_DEVICE_TIMING_SUBSTAGES:
            _validate_phase4_sampled_substage(
                attrs=attrs,
                substage=substage,
                expected_stride=_PHASE4_TIMING_POLICY_BY_NAME[
                    substage
                ].cuda_sample_stride,
                contract_version=contract_version,
                label=label,
                source=source,
            )
            observed_substages.add(substage)
        for optional_substage in _PHASE4_OPTIONAL_DEVICE_TIMING_SUBSTAGES:
            if f"phase4_timing_{optional_substage}_population_count" in attrs:
                _validate_phase4_sampled_substage(
                    attrs=attrs,
                    substage=optional_substage,
                    expected_stride=_PHASE4_TIMING_POLICY_BY_NAME[
                        optional_substage
                    ].cuda_sample_stride,
                    contract_version=contract_version,
                    label=label,
                    source=source,
                )
                observed_substages.add(optional_substage)
                optional_substage_presence[optional_substage] = True
        _required_nonempty_string(
            attrs,
            "phase4_timing_instrumentation_overhead_scope",
            label=label,
            source=source,
        )
        for field in (
            "phase4_timing_instrumentation_recording_host_overhead_ms",
            "phase4_timing_instrumentation_total_host_overhead_ms",
            "phase4_timing_resolution_host_elapsed_ms",
        ):
            _required_finite_number(
                attrs, field, label=label, source=source, nonnegative=True
            )
        for field in (
            "phase4_timing_cuda_event_object_count",
            "phase4_timing_cuda_event_record_count",
        ):
            count = attrs.get(field)
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise FeatureRowModeValidationError(
                    f"{label} has invalid {field}={count!r}: {source}"
                )
        if (
            attrs["phase4_timing_cuda_event_object_count"]
            != attrs["phase4_timing_cuda_event_record_count"]
        ):
            raise FeatureRowModeValidationError(
                f"{label} Phase-4 CUDA event object/record counts differ: {source}"
            )
        backends.add(backend)
        device_ordinals.add(ordinal)
        device_uuids.add(device_uuid)
    if len(backends) != 1 or len(device_ordinals) != 1 or len(device_uuids) != 1:
        raise FeatureRowModeValidationError(
            f"{label} has inconsistent Phase-4 timing identity: "
            f"backends={sorted(backends)!r}, ordinals={sorted(device_ordinals)!r}, "
            f"uuids={sorted(device_uuids)!r}"
        )
    return {
        "event_count": len(phase4_events),
        "backends": sorted(backends),
        "cuda_visible_device_ordinals": sorted(device_ordinals),
        "cuda_device_uuids": sorted(device_uuids),
        "observed_substages": sorted(observed_substages),
        "optional_substage_presence": optional_substage_presence,
    }


def _validate_phase4_sampled_substage(
    *,
    attrs: dict[str, Any],
    substage: str,
    expected_stride: int,
    contract_version: str,
    label: str,
    source: Path,
) -> None:
    prefix = f"phase4_timing_{substage}"
    population = attrs.get(f"{prefix}_population_count")
    sample_count = attrs.get(f"{prefix}_cuda_sample_count")
    stride = attrs.get(f"{prefix}_cuda_sample_stride")
    for field, value in (
        ("population_count", population),
        ("cuda_sample_count", sample_count),
        ("cuda_sample_stride", stride),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise FeatureRowModeValidationError(
                f"{label} has invalid {prefix}_{field}={value!r}: {source}"
            )
    assert isinstance(population, int)
    assert isinstance(sample_count, int)
    assert isinstance(stride, int)
    if sample_count > population or stride != expected_stride:
        raise FeatureRowModeValidationError(
            f"{label} has inconsistent sampled timing counts for {substage!r}: "
            f"population={population}, samples={sample_count}, stride={stride}, "
            f"expected_stride={expected_stride}: {source}"
        )
    complete_blocks, partial_block_size = divmod(population, stride)
    allowed_sample_counts = {complete_blocks}
    if partial_block_size:
        allowed_sample_counts.add(complete_blocks + 1)
    if sample_count not in allowed_sample_counts:
        raise FeatureRowModeValidationError(
            f"{label} stratified sample count for {substage!r} is {sample_count}; "
            f"expected one of {sorted(allowed_sample_counts)!r}: {source}"
        )
    if contract_version == PHASE4_DEVICE_TIMING_CONTRACT_V2:
        expected_used_sample_count = complete_blocks + int(partial_block_size > 0)
        if sample_count != expected_used_sample_count:
            raise FeatureRowModeValidationError(
                f"{label} v2 timing did not use one sample per stratum for "
                f"{substage!r}: samples={sample_count}, "
                f"expected={expected_used_sample_count}: {source}"
            )
        recorded_sample_count = attrs.get(f"{prefix}_cuda_recorded_sample_count")
        if (
            not isinstance(recorded_sample_count, int)
            or isinstance(recorded_sample_count, bool)
            or recorded_sample_count < sample_count
            or recorded_sample_count > sample_count * 2
        ):
            raise FeatureRowModeValidationError(
                f"{label} has invalid recorded CUDA sample count for "
                f"{substage!r}: recorded={recorded_sample_count!r}, "
                f"used={sample_count}: {source}"
            )
        estimate_status = attrs.get(f"{prefix}_cuda_estimate_status")
        if estimate_status != "complete":
            raise FeatureRowModeValidationError(
                f"{label} Phase-4 timing estimate for {substage!r} refuses an "
                f"incomplete final stratum: status={estimate_status!r}: {source}"
            )
    _required_finite_number(
        attrs,
        f"{prefix}_wall_elapsed_ms",
        label=label,
        source=source,
        nonnegative=True,
    )
    sampled_elapsed = _required_finite_number(
        attrs,
        f"{prefix}_cuda_sampled_elapsed_ms",
        label=label,
        source=source,
        nonnegative=True,
    )
    estimated_elapsed = _required_finite_number(
        attrs,
        f"{prefix}_cuda_estimated_total_elapsed_ms",
        label=label,
        source=source,
        nonnegative=True,
    )
    if contract_version == PHASE4_DEVICE_TIMING_CONTRACT_V1:
        expected_estimate = sampled_elapsed * stride
    else:
        tail_population = attrs.get(f"{prefix}_cuda_tail_population_count")
        if (
            not isinstance(tail_population, int)
            or isinstance(tail_population, bool)
            or tail_population < 0
            or tail_population >= stride
            or tail_population != partial_block_size
        ):
            raise FeatureRowModeValidationError(
                f"{label} has invalid final-stratum population for {substage!r}: "
                f"tail={tail_population!r}, expected={partial_block_size}: {source}"
            )
        tail_status = attrs.get(f"{prefix}_cuda_tail_status")
        expected_tail_status = "sampled" if partial_block_size else "not_applicable"
        if tail_status != expected_tail_status:
            raise FeatureRowModeValidationError(
                f"{label} has incomplete final timing stratum for {substage!r}: "
                f"tail_status={tail_status!r}, expected={expected_tail_status!r}: "
                f"{source}"
            )
        tail_sample_source = attrs.get(f"{prefix}_cuda_tail_sample_source")
        allowed_tail_sources = (
            {"hashed_primary", "block_start_fallback"}
            if partial_block_size
            else {"not_applicable"}
        )
        if tail_sample_source not in allowed_tail_sources:
            raise FeatureRowModeValidationError(
                f"{label} has invalid final-stratum sample source for "
                f"{substage!r}: sample_source={tail_sample_source!r}, "
                f"expected one of {sorted(allowed_tail_sources)!r}: {source}"
            )
        complete_sampled_elapsed = _required_finite_number(
            attrs,
            f"{prefix}_cuda_complete_block_sampled_elapsed_ms",
            label=label,
            source=source,
            nonnegative=True,
        )
        tail_sampled_elapsed = _required_finite_number(
            attrs,
            f"{prefix}_cuda_incomplete_tail_sampled_elapsed_ms",
            label=label,
            source=source,
            nonnegative=True,
        )
        if not math.isclose(
            sampled_elapsed,
            complete_sampled_elapsed + tail_sampled_elapsed,
            rel_tol=1e-6,
            abs_tol=1e-3,
        ):
            raise FeatureRowModeValidationError(
                f"{label} sampled CUDA timing components do not reconcile for "
                f"{substage!r}: {source}"
            )
        expected_estimate = (
            complete_sampled_elapsed * stride
            + tail_sampled_elapsed * partial_block_size
        )
    if not math.isclose(
        estimated_elapsed,
        expected_estimate,
        rel_tol=1e-6,
        abs_tol=1e-3,
    ):
        raise FeatureRowModeValidationError(
            f"{label} sampled CUDA estimate for {substage!r} is inconsistent: {source}"
        )


def _validate_per_device_gpu_resource_evidence(
    *, output_root: Path, label: str
) -> dict[str, Any]:
    path = output_root / "resource_summary.json"
    summary = _load_json(path)
    if (
        summary.get("gpu_sampling_status") != "ok"
        or summary.get("resource_validation_passed") is not True
    ):
        raise FeatureRowModeValidationError(
            f"{label} GPU sampling did not complete successfully: {path}"
        )
    sample_count = summary.get("gpu_sample_count")
    device_count = summary.get("gpu_device_count")
    devices = summary.get("gpu_devices")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count <= 0
        or not isinstance(device_count, int)
        or isinstance(device_count, bool)
        or device_count <= 0
        or not isinstance(devices, list)
        or len(devices) != device_count
    ):
        raise FeatureRowModeValidationError(
            f"{label} has invalid per-device GPU sample counts: {path}"
        )
    identities: list[tuple[int, str]] = []
    per_device_sample_count = 0
    for device in devices:
        if not isinstance(device, dict):
            raise FeatureRowModeValidationError(
                f"{label} has a non-object GPU device summary: {path}"
            )
        index = device.get("gpu_index")
        uuid = device.get("gpu_uuid")
        name = device.get("gpu_name")
        device_samples = device.get("sample_count")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or not isinstance(uuid, str)
            or not uuid
            or not isinstance(name, str)
            or not name
            or not isinstance(device_samples, int)
            or isinstance(device_samples, bool)
            or device_samples <= 0
        ):
            raise FeatureRowModeValidationError(
                f"{label} has ambiguous GPU device identity index={index!r}, "
                f"uuid={uuid!r}, name={name!r}, sample_count={device_samples!r}: "
                f"{path}"
            )
        _validate_gpu_device_metrics(device=device, label=label, source=path)
        identities.append((index, uuid))
        per_device_sample_count += device_samples
    if per_device_sample_count != sample_count:
        raise FeatureRowModeValidationError(
            f"{label} per-device GPU sample counts total {per_device_sample_count}; "
            f"expected {sample_count}: {path}"
        )
    indexes = [identity[0] for identity in identities]
    uuids = [identity[1] for identity in identities]
    if len(set(indexes)) != len(indexes) or len(set(uuids)) != len(uuids):
        raise FeatureRowModeValidationError(
            f"{label} has duplicate GPU device indexes or UUIDs: {path}"
        )
    primary_identity = (
        summary.get("gpu_summary_device_index"),
        summary.get("gpu_summary_device_uuid"),
    )
    if sum(identity == primary_identity for identity in identities) != 1:
        raise FeatureRowModeValidationError(
            f"{label} busiest GPU summary does not map to exactly one sampled "
            f"device: {path}"
        )
    primary = next(
        device
        for device in devices
        if (device["gpu_index"], device["gpu_uuid"]) == primary_identity
    )
    for field in (
        "gpu_sm_utilization_mean_percent",
        "gpu_sm_utilization_p95_percent",
        "gpu_sm_utilization_max_percent",
        "gpu_memory_utilization_mean_percent",
        "gpu_memory_utilization_p95_percent",
        "gpu_memory_utilization_max_percent",
        "gpu_power_mean_watts",
        "gpu_power_max_watts",
        "gpu_framebuffer_peak_mib",
        "gpu_framebuffer_total_mib",
        "gpu_framebuffer_peak_fraction",
    ):
        if summary.get(field) != primary.get(field):
            raise FeatureRowModeValidationError(
                f"{label} top-level {field} does not match the busiest sampled "
                f"device: {path}"
            )
    if summary.get("gpu_summary_device_name") != primary.get("gpu_name"):
        raise FeatureRowModeValidationError(
            f"{label} busiest GPU name does not match its sampled device: {path}"
        )
    expected_scope = "single_device" if device_count == 1 else "busiest_device"
    if summary.get("gpu_summary_scope") != expected_scope:
        raise FeatureRowModeValidationError(
            f"{label} GPU summary scope is {summary.get('gpu_summary_scope')!r}; "
            f"expected {expected_scope!r}: {path}"
        )
    return {
        "sample_count": sample_count,
        "device_count": device_count,
        "device_identities": [
            {"gpu_index": index, "gpu_uuid": uuid} for index, uuid in identities
        ],
        "summary_device_index": primary_identity[0],
        "summary_device_uuid": primary_identity[1],
    }


def _validate_gpu_device_metrics(
    *, device: dict[str, Any], label: str, source: Path
) -> None:
    utilization_fields = (
        "gpu_sm_utilization_mean_percent",
        "gpu_sm_utilization_p95_percent",
        "gpu_sm_utilization_max_percent",
        "gpu_memory_utilization_mean_percent",
        "gpu_memory_utilization_p95_percent",
        "gpu_memory_utilization_max_percent",
    )
    utilization = {
        field: _required_finite_number(
            device, field, label=label, source=source, nonnegative=True
        )
        for field in utilization_fields
    }
    if any(value > 100.0 for value in utilization.values()):
        raise FeatureRowModeValidationError(
            f"{label} GPU utilization is outside [0, 100]: {source}"
        )
    for prefix in ("gpu_sm_utilization", "gpu_memory_utilization"):
        maximum = utilization[f"{prefix}_max_percent"]
        if (
            utilization[f"{prefix}_mean_percent"] > maximum
            or utilization[f"{prefix}_p95_percent"] > maximum
        ):
            raise FeatureRowModeValidationError(
                f"{label} GPU utilization summary exceeds its maximum: {source}"
            )
    power_mean = _required_finite_number(
        device,
        "gpu_power_mean_watts",
        label=label,
        source=source,
        nonnegative=True,
    )
    power_max = _required_finite_number(
        device,
        "gpu_power_max_watts",
        label=label,
        source=source,
        nonnegative=True,
    )
    if power_mean > power_max:
        raise FeatureRowModeValidationError(
            f"{label} GPU mean power exceeds maximum power: {source}"
        )
    framebuffer_peak = _required_finite_number(
        device,
        "gpu_framebuffer_peak_mib",
        label=label,
        source=source,
        nonnegative=True,
    )
    framebuffer_total = _required_finite_number(
        device,
        "gpu_framebuffer_total_mib",
        label=label,
        source=source,
        nonnegative=True,
    )
    framebuffer_fraction = _required_finite_number(
        device,
        "gpu_framebuffer_peak_fraction",
        label=label,
        source=source,
        nonnegative=True,
    )
    if (
        framebuffer_total <= 0
        or framebuffer_peak > framebuffer_total
        or framebuffer_fraction > 1.0
        or not math.isclose(
            framebuffer_fraction,
            framebuffer_peak / framebuffer_total,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        raise FeatureRowModeValidationError(
            f"{label} GPU framebuffer summary is inconsistent: {source}"
        )


def validate_feature_row_mode(
    *,
    output_root: Path,
    expected_mode: str,
    label: str,
    expected_backward_engine_mode: str | None = None,
    expected_forward_graph_mode: str | None = None,
    expected_vjp_kernel_mode: str | None = None,
    expected_forward_lane_count: int | None = None,
    expected_session_capacity: int | None = None,
    expected_backward_batch_capacity: int | None = None,
    expected_phase1_trace_batch_size_max: int | None = None,
    expected_phase1_effective_trace_batch_size: int | None = None,
    expected_phase3_batch_size: int | None = None,
    expected_phase4_batch_size: int | None = None,
    expected_decoder_active_row_residency: bool | None = None,
    expected_decoder_active_row_residency_requirement: str | None = None,
    expected_decoder_active_row_max_bytes: int | None = None,
    expected_decoder_active_row_safety_margin_bytes: int | None = None,
    require_structured_vjp_evidence: bool = False,
    require_phase4_device_timing: bool = False,
    expected_phase4_device_timing_backend: str | None = None,
    require_unambiguous_per_device_gpu_evidence: bool = False,
    expected_graph_feature_count: int | None = None,
    expected_graph_edge_count: int | None = None,
    require_canonical_typed_buckets: bool = False,
    require_full_completion: bool = False,
) -> dict[str, Any]:
    shard_root = output_root / "shards" / "shard_000"
    shard = _load_json(shard_root / "shard.json")
    shard_status = shard.get("status")
    allowed_shard_statuses = (
        {"complete"}
        if require_full_completion
        else {
            "complete",
            "probe_completed",
        }
    )
    if shard_status not in allowed_shard_statuses:
        raise FeatureRowModeValidationError(
            f"{label} shard status is {shard_status!r}, expected one of "
            f"{sorted(allowed_shard_statuses)!r}"
        )

    trace_paths = sorted(shard_root.glob("token_*/trace.json"))
    if not trace_paths:
        raise FeatureRowModeValidationError(f"{label} has no trace artifacts")
    requested_modes: set[str] = set()
    requested_backward_engines: set[str] = set()
    requested_forward_graph_modes: set[str] = set()
    requested_vjp_kernel_modes: set[str] = set()
    effective_backward_engines: set[str] = set()
    effective_forward_graph_modes: set[str] = set()
    effective_vjp_kernel_modes: set[str] = set()
    effective_forward_lane_counts: set[int] = set()
    requested_session_capacities: set[int] = set()
    requested_phase1_trace_batch_sizes: set[int] = set()
    requested_phase3_batch_sizes: set[int] = set()
    requested_phase4_batch_sizes: set[int] = set()
    effective_session_capacities: set[int] = set()
    effective_backward_batch_capacities: set[int] = set()
    prepared_trace_batch_sizes: set[int] = set()
    actual_phase1_trace_batch_sizes: set[int] = set()
    effective_phase3_batch_sizes: set[int] = set()
    effective_phase4_batch_sizes: set[int] = set()
    requested_capture_names: set[str] = set()
    written_capture_count = 0
    graph_count = 0
    graph_validation_reports: list[dict[str, Any]] = []
    active_row_evidence: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        trace = _load_json(trace_path)
        allowed_trace_statuses = (
            {"ok"}
            if require_full_completion
            else {
                "ok",
                "probe_completed",
            }
        )
        if trace.get("status") not in allowed_trace_statuses:
            raise FeatureRowModeValidationError(
                f"{label} trace status is {trace.get('status')!r}: {trace_path}"
            )
        knobs = trace.get("graph_knobs")
        if isinstance(knobs, dict) and knobs.get("feature_row_influence_mode"):
            requested_modes.add(str(knobs["feature_row_influence_mode"]))
        if isinstance(knobs, dict):
            if expected_decoder_active_row_residency is not None:
                _validate_requested_active_row_contract(
                    knobs=knobs,
                    label=label,
                    trace_path=trace_path,
                    expected_residency=expected_decoder_active_row_residency,
                    expected_requirement=(
                        expected_decoder_active_row_residency_requirement
                    ),
                    expected_max_bytes=expected_decoder_active_row_max_bytes,
                    expected_safety_margin_bytes=(
                        expected_decoder_active_row_safety_margin_bytes
                    ),
                )
            for field, destination in (
                ("nnsight_session_capacity", requested_session_capacities),
                (
                    "phase1_trace_batch_size_max",
                    requested_phase1_trace_batch_sizes,
                ),
                (
                    "phase3_compute_microbatch_max_rows",
                    requested_phase3_batch_sizes,
                ),
                (
                    "phase4_execution_batch_max_rows",
                    requested_phase4_batch_sizes,
                ),
            ):
                value = knobs.get(field)
                if isinstance(value, int) and not isinstance(value, bool):
                    destination.add(value)
            requested_captures = {
                name
                for name, knob in _CAPTURE_SIDECAR_KNOBS.items()
                if bool(knobs.get(knob, False))
            }
            requested_capture_names.update(requested_captures)
            if requested_captures:
                written_capture_count += _validate_capture_artifacts(
                    trace_path=trace_path,
                    trace=trace,
                    requested=requested_captures,
                    label=label,
                )
            try:
                backward_identity = backward_mechanism_record(knobs)
            except ValueError as error:
                raise FeatureRowModeValidationError(
                    f"{label} has invalid requested backward selection: {error}"
                ) from error
            requested_backward_engines.add(
                str(backward_identity["backward_engine_mode"])
            )
            requested_forward_graph_modes.add(
                str(backward_identity["forward_graph_mode"])
            )
            requested_vjp_kernel_modes.add(str(backward_identity["vjp_kernel_mode"]))
        effective_execution = trace.get("effective_execution")
        batches = (
            effective_execution.get("batches")
            if isinstance(effective_execution, dict)
            else None
        )
        if isinstance(batches, dict):
            backward_engine = batches.get("backward_engine_mode")
            if backward_engine is not None:
                effective_backward_engines.add(str(backward_engine))
            forward_graph_mode = batches.get("forward_graph_mode")
            if forward_graph_mode is not None:
                effective_forward_graph_modes.add(str(forward_graph_mode))
            vjp_kernel_mode = batches.get("vjp_kernel_mode")
            if vjp_kernel_mode is not None:
                effective_vjp_kernel_modes.add(str(vjp_kernel_mode))
            forward_lane_count = batches.get("forward_lane_count")
            if isinstance(forward_lane_count, int) and not isinstance(
                forward_lane_count, bool
            ):
                effective_forward_lane_counts.add(forward_lane_count)
            for field, destination in (
                ("session_capacity", effective_session_capacities),
                ("backward_batch_capacity", effective_backward_batch_capacities),
                ("trace_batch_size", prepared_trace_batch_sizes),
                ("phase3_microbatch_max_rows", effective_phase3_batch_sizes),
                (
                    "phase4_execution_batch_max_rows",
                    effective_phase4_batch_sizes,
                ),
            ):
                value = batches.get(field)
                if isinstance(value, int) and not isinstance(value, bool):
                    destination.add(value)
        if require_full_completion:
            graph_validation_reports.append(
                _validate_compact_graph(
                    trace_path=trace_path,
                    trace=trace,
                    label=label,
                    expected_feature_count=expected_graph_feature_count,
                    expected_edge_count=expected_graph_edge_count,
                    require_canonical_typed_buckets=(require_canonical_typed_buckets),
                )
            )
            graph_count += 1
        if expected_decoder_active_row_residency:
            evidence = trace.get("decoder_active_row_residency")
            if not isinstance(evidence, dict):
                raise FeatureRowModeValidationError(
                    f"{label} has no persisted decoder active-row evidence: "
                    f"{trace_path}"
                )
            active_row_evidence.append(
                _validate_active_row_evidence(
                    evidence=evidence,
                    label=label,
                    trace_path=trace_path,
                    expected_requirement=(
                        expected_decoder_active_row_residency_requirement
                    ),
                    expected_max_bytes=expected_decoder_active_row_max_bytes,
                    expected_safety_margin_bytes=(
                        expected_decoder_active_row_safety_margin_bytes
                    ),
                )
            )
    if requested_modes != {expected_mode}:
        raise FeatureRowModeValidationError(
            f"{label} requested feature-row modes {sorted(requested_modes)!r}; "
            f"expected only {expected_mode!r}"
        )

    resolved_modes: set[str] = set()
    compute_batch_events: list[tuple[Path, dict[str, Any]]] = []
    phase4_events: list[tuple[Path, dict[str, Any]]] = []
    for telemetry_path in sorted(shard_root.glob("token_*/telemetry_live.jsonl")):
        with telemetry_path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                attrs = event.get("attrs")
                if isinstance(attrs, dict) and attrs.get(
                    "feature_row_influence_mode_resolved"
                ):
                    resolved_modes.add(
                        str(attrs["feature_row_influence_mode_resolved"])
                    )
                if not isinstance(attrs, dict):
                    continue
                phase = event.get("phase")
                name = event.get("name")
                if name == "context.compute_batch":
                    compute_batch_events.append((telemetry_path, attrs))
                elif name == "phase4.feature_attribution":
                    phase4_events.append((telemetry_path, attrs))
                if phase == "phase1" or (
                    isinstance(name, str) and name.startswith("phase1.")
                ):
                    for field in (
                        "trace_batch_size",
                        "forward_lane_count",
                        "batch_capacity",
                        "backward_batch_capacity",
                    ):
                        value = attrs.get(field)
                        if isinstance(value, int) and not isinstance(value, bool):
                            actual_phase1_trace_batch_sizes.add(value)
                            break

    resolution_source = "telemetry"
    if expected_mode == "cpu_exact" and not resolved_modes:
        # Native cpu_exact uses the file store directly, without constructing the
        # accelerated-store wrapper that emits requested/resolved mode telemetry.
        effective_modes = requested_modes
        resolution_source = "native_request"
    else:
        effective_modes = resolved_modes
    if effective_modes != {expected_mode}:
        raise FeatureRowModeValidationError(
            f"{label} resolved feature-row modes {sorted(resolved_modes)!r}; "
            f"expected only {expected_mode!r}"
        )

    if expected_backward_engine_mode is not None:
        if requested_backward_engines != {expected_backward_engine_mode}:
            raise FeatureRowModeValidationError(
                f"{label} requested backward engines "
                f"{sorted(requested_backward_engines)!r}; expected only "
                f"{expected_backward_engine_mode!r}"
            )
        if effective_backward_engines != {expected_backward_engine_mode}:
            raise FeatureRowModeValidationError(
                f"{label} effective backward engines "
                f"{sorted(effective_backward_engines)!r}; expected only "
                f"{expected_backward_engine_mode!r}"
            )
    for name, expected, requested, effective in (
        (
            "forward graph modes",
            expected_forward_graph_mode,
            requested_forward_graph_modes,
            effective_forward_graph_modes,
        ),
        (
            "VJP kernel modes",
            expected_vjp_kernel_mode,
            requested_vjp_kernel_modes,
            effective_vjp_kernel_modes,
        ),
    ):
        if expected is None:
            continue
        if requested != {expected}:
            raise FeatureRowModeValidationError(
                f"{label} requested {name} {sorted(requested)!r}; expected only "
                f"{expected!r}"
            )
        if effective != {expected}:
            raise FeatureRowModeValidationError(
                f"{label} effective {name} {sorted(effective)!r}; expected only "
                f"{expected!r}"
            )
    if expected_forward_lane_count is not None and effective_forward_lane_counts != {
        expected_forward_lane_count
    }:
        raise FeatureRowModeValidationError(
            f"{label} effective forward-lane counts "
            f"{sorted(effective_forward_lane_counts)!r}; expected only "
            f"{expected_forward_lane_count!r}"
        )

    capacity_checks = (
        (
            "session capacities",
            expected_session_capacity,
            requested_session_capacities,
            effective_session_capacities,
        ),
        (
            "Phase-3 batch sizes",
            expected_phase3_batch_size,
            requested_phase3_batch_sizes,
            effective_phase3_batch_sizes,
        ),
        (
            "Phase-4 batch sizes",
            expected_phase4_batch_size,
            requested_phase4_batch_sizes,
            effective_phase4_batch_sizes,
        ),
    )
    for name, expected, requested, effective in capacity_checks:
        if expected is None:
            continue
        if requested != {expected}:
            raise FeatureRowModeValidationError(
                f"{label} requested {name} {sorted(requested)!r}; expected only "
                f"{expected!r}"
            )
        if effective != {expected}:
            raise FeatureRowModeValidationError(
                f"{label} effective {name} {sorted(effective)!r}; expected only "
                f"{expected!r}"
            )
    if (
        expected_phase1_trace_batch_size_max is not None
        and requested_phase1_trace_batch_sizes != {expected_phase1_trace_batch_size_max}
    ):
        raise FeatureRowModeValidationError(
            f"{label} requested Phase-1 trace batch size maxima "
            f"{sorted(requested_phase1_trace_batch_sizes)!r}; expected only "
            f"{expected_phase1_trace_batch_size_max!r}"
        )
    if (
        expected_phase1_effective_trace_batch_size is not None
        and actual_phase1_trace_batch_sizes
        != {expected_phase1_effective_trace_batch_size}
    ):
        raise FeatureRowModeValidationError(
            f"{label} effective Phase-1 trace batch sizes "
            f"{sorted(actual_phase1_trace_batch_sizes)!r}; expected only "
            f"{expected_phase1_effective_trace_batch_size!r}"
        )
    if (
        expected_backward_batch_capacity is not None
        and effective_backward_batch_capacities != {expected_backward_batch_capacity}
    ):
        raise FeatureRowModeValidationError(
            f"{label} effective backward batch capacities "
            f"{sorted(effective_backward_batch_capacities)!r}; expected only "
            f"{expected_backward_batch_capacity!r}"
        )

    structured_vjp_evidence = None
    if require_structured_vjp_evidence:
        structured_vjp_evidence = _validate_structured_vjp_evidence(
            compute_batch_events=compute_batch_events,
            expected_vjp_kernel_mode=expected_vjp_kernel_mode,
            label=label,
        )
    phase4_device_timing = None
    if require_phase4_device_timing:
        phase4_device_timing = _validate_phase4_device_timing(
            phase4_events=phase4_events,
            expected_backend=expected_phase4_device_timing_backend,
            label=label,
        )
    per_device_gpu_resource_evidence = None
    if require_unambiguous_per_device_gpu_evidence:
        per_device_gpu_resource_evidence = _validate_per_device_gpu_resource_evidence(
            output_root=output_root,
            label=label,
        )
    if (
        phase4_device_timing is not None
        and per_device_gpu_resource_evidence is not None
    ):
        timing_uuids = set(phase4_device_timing["cuda_device_uuids"])
        summary_uuid = per_device_gpu_resource_evidence["summary_device_uuid"]
        canonical_timing_uuids = {_canonical_gpu_uuid(value) for value in timing_uuids}
        canonical_summary_uuid = _canonical_gpu_uuid(summary_uuid)
        if canonical_timing_uuids != {canonical_summary_uuid}:
            raise FeatureRowModeValidationError(
                f"{label} Phase-4 CUDA timing UUIDs {sorted(timing_uuids)!r} do "
                f"not match the busiest sampled GPU UUID {summary_uuid!r}"
            )

    return {
        "schema_version": 1,
        "label": label,
        "status": shard_status,
        "expected_mode": expected_mode,
        "requested_modes": sorted(requested_modes),
        "resolved_modes": sorted(resolved_modes),
        "effective_modes": sorted(effective_modes),
        "resolution_source": resolution_source,
        "expected_backward_engine_mode": expected_backward_engine_mode,
        "requested_backward_engine_modes": sorted(requested_backward_engines),
        "effective_backward_engine_modes": sorted(effective_backward_engines),
        "expected_forward_graph_mode": expected_forward_graph_mode,
        "requested_forward_graph_modes": sorted(requested_forward_graph_modes),
        "effective_forward_graph_modes": sorted(effective_forward_graph_modes),
        "expected_vjp_kernel_mode": expected_vjp_kernel_mode,
        "requested_vjp_kernel_modes": sorted(requested_vjp_kernel_modes),
        "effective_vjp_kernel_modes": sorted(effective_vjp_kernel_modes),
        "expected_forward_lane_count": expected_forward_lane_count,
        "effective_forward_lane_counts": sorted(effective_forward_lane_counts),
        "expected_session_capacity": expected_session_capacity,
        "requested_session_capacities": sorted(requested_session_capacities),
        "effective_session_capacities": sorted(effective_session_capacities),
        "expected_backward_batch_capacity": expected_backward_batch_capacity,
        "effective_backward_batch_capacities": sorted(
            effective_backward_batch_capacities
        ),
        "expected_phase1_trace_batch_size_max": (expected_phase1_trace_batch_size_max),
        "expected_phase1_effective_trace_batch_size": (
            expected_phase1_effective_trace_batch_size
        ),
        "requested_phase1_trace_batch_sizes": sorted(
            requested_phase1_trace_batch_sizes
        ),
        "prepared_trace_batch_sizes": sorted(prepared_trace_batch_sizes),
        "actual_phase1_trace_batch_sizes": sorted(actual_phase1_trace_batch_sizes),
        "expected_phase3_batch_size": expected_phase3_batch_size,
        "requested_phase3_batch_sizes": sorted(requested_phase3_batch_sizes),
        "effective_phase3_batch_sizes": sorted(effective_phase3_batch_sizes),
        "expected_phase4_batch_size": expected_phase4_batch_size,
        "requested_phase4_batch_sizes": sorted(requested_phase4_batch_sizes),
        "effective_phase4_batch_sizes": sorted(effective_phase4_batch_sizes),
        "expected_decoder_active_row_residency": (
            expected_decoder_active_row_residency
        ),
        "expected_decoder_active_row_residency_requirement": (
            expected_decoder_active_row_residency_requirement
        ),
        "expected_decoder_active_row_max_bytes": (
            expected_decoder_active_row_max_bytes
        ),
        "expected_decoder_active_row_safety_margin_bytes": (
            expected_decoder_active_row_safety_margin_bytes
        ),
        "decoder_active_row_evidence": active_row_evidence,
        "require_structured_vjp_evidence": require_structured_vjp_evidence,
        "structured_vjp_evidence": structured_vjp_evidence,
        "require_phase4_device_timing": require_phase4_device_timing,
        "expected_phase4_device_timing_backend": (
            expected_phase4_device_timing_backend
        ),
        "phase4_device_timing": phase4_device_timing,
        "require_unambiguous_per_device_gpu_evidence": (
            require_unambiguous_per_device_gpu_evidence
        ),
        "per_device_gpu_resource_evidence": per_device_gpu_resource_evidence,
        "expected_graph_feature_count": expected_graph_feature_count,
        "expected_graph_edge_count": expected_graph_edge_count,
        "require_canonical_typed_buckets": require_canonical_typed_buckets,
        "require_full_completion": require_full_completion,
        "validated_graph_count": graph_count,
        "graph_validation_reports": graph_validation_reports,
        "requested_capture_names": sorted(requested_capture_names),
        "written_capture_count": written_capture_count,
        "trace_count": len(trace_paths),
    }


def _validate_requested_active_row_contract(
    *,
    knobs: dict[str, Any],
    label: str,
    trace_path: Path,
    expected_residency: bool,
    expected_requirement: str | None,
    expected_max_bytes: int | None,
    expected_safety_margin_bytes: int | None,
) -> None:
    checks = (
        ("decoder_active_row_residency", expected_residency),
        (
            "decoder_active_row_residency_requirement",
            expected_requirement,
        ),
        ("decoder_active_row_max_bytes", expected_max_bytes),
        (
            "decoder_active_row_safety_margin_bytes",
            expected_safety_margin_bytes,
        ),
    )
    defaults: dict[str, object] = {
        "decoder_active_row_residency": False,
        "decoder_active_row_residency_requirement": "preferred",
        "decoder_active_row_max_bytes": 0,
        "decoder_active_row_safety_margin_bytes": 0,
    }
    for field, expected in checks:
        if expected is None:
            continue
        actual = knobs.get(field, defaults[field])
        if actual != expected or type(actual) is not type(expected):
            raise FeatureRowModeValidationError(
                f"{label} requested {field}={actual!r}; expected {expected!r}: "
                f"{trace_path}"
            )


def _required_nonnegative_int(
    evidence: dict[str, Any], field: str, *, label: str, trace_path: Path
) -> int:
    value = evidence.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FeatureRowModeValidationError(
            f"{label} active-row evidence has invalid {field}={value!r}: {trace_path}"
        )
    return value


def _validate_active_row_evidence(
    *,
    evidence: dict[str, Any],
    label: str,
    trace_path: Path,
    expected_requirement: str | None,
    expected_max_bytes: int | None,
    expected_safety_margin_bytes: int | None,
) -> dict[str, Any]:
    if evidence.get("requested") is not True:
        raise FeatureRowModeValidationError(
            f"{label} active-row evidence did not record requested=true: {trace_path}"
        )
    if (
        expected_requirement is not None
        and evidence.get("requirement") != expected_requirement
    ):
        raise FeatureRowModeValidationError(
            f"{label} active-row requirement is {evidence.get('requirement')!r}; "
            f"expected {expected_requirement!r}: {trace_path}"
        )
    max_bytes = _required_nonnegative_int(
        evidence, "max_bytes_requested", label=label, trace_path=trace_path
    )
    if expected_max_bytes is not None and max_bytes != expected_max_bytes:
        raise FeatureRowModeValidationError(
            f"{label} active-row max_bytes_requested={max_bytes}; expected "
            f"{expected_max_bytes}: {trace_path}"
        )
    if expected_safety_margin_bytes is not None:
        safety_margin = _required_nonnegative_int(
            evidence, "safety_margin_bytes", label=label, trace_path=trace_path
        )
        if safety_margin != expected_safety_margin_bytes:
            raise FeatureRowModeValidationError(
                f"{label} active-row safety_margin_bytes={safety_margin}; expected "
                f"{expected_safety_margin_bytes}: {trace_path}"
            )

    effective = evidence.get("effective")
    if not isinstance(effective, bool):
        raise FeatureRowModeValidationError(
            f"{label} active-row evidence has invalid effective={effective!r}: "
            f"{trace_path}"
        )
    if expected_requirement == "required":
        if not effective or evidence.get("fallback_reason") is not None:
            raise FeatureRowModeValidationError(
                f"{label} required active-row residency was not effective without "
                f"fallback: {trace_path}"
            )
        if evidence.get("admission_reason") != "admitted":
            raise FeatureRowModeValidationError(
                f"{label} required active-row admission reason is "
                f"{evidence.get('admission_reason')!r}: {trace_path}"
            )

    normalized = dict(evidence)
    if expected_max_bytes == 0:
        if evidence.get("admission_policy") != "live_hbm_headroom":
            raise FeatureRowModeValidationError(
                f"{label} dynamic active-row admission policy is "
                f"{evidence.get('admission_policy')!r}; expected "
                f"'live_hbm_headroom': {trace_path}"
            )
        dynamic_budget = _required_nonnegative_int(
            evidence, "dynamic_budget_bytes", label=label, trace_path=trace_path
        )
        effective_budget = _required_nonnegative_int(
            evidence, "effective_budget_bytes", label=label, trace_path=trace_path
        )
        resident = evidence.get("resident")
        hbm = evidence.get("hbm")
        if not isinstance(resident, dict) or not isinstance(hbm, dict):
            raise FeatureRowModeValidationError(
                f"{label} dynamic active-row evidence lacks resident/HBM records: "
                f"{trace_path}"
            )
        estimated_bytes = _required_nonnegative_int(
            resident, "estimated_bytes", label=label, trace_path=trace_path
        )
        free_bytes = _required_nonnegative_int(
            hbm, "free_bytes", label=label, trace_path=trace_path
        )
        total_bytes = _required_nonnegative_int(
            hbm, "total_bytes", label=label, trace_path=trace_path
        )
        if (
            effective_budget != dynamic_budget
            or estimated_bytes <= 0
            or free_bytes <= 0
            or total_bytes <= 0
            or free_bytes > total_bytes
        ):
            raise FeatureRowModeValidationError(
                f"{label} dynamic active-row HBM budget evidence is inconsistent: "
                f"{trace_path}"
            )
        if effective and (dynamic_budget <= 0 or estimated_bytes > effective_budget):
            raise FeatureRowModeValidationError(
                f"{label} admitted dynamic active rows exceed the effective HBM "
                f"budget: {trace_path}"
            )
        if not effective and (
            evidence.get("fallback_reason") is None
            or evidence.get("admission_reason") == "admitted"
            or estimated_bytes <= effective_budget
        ):
            raise FeatureRowModeValidationError(
                f"{label} preferred dynamic active-row refusal is inconsistent: "
                f"{trace_path}"
            )
        if expected_safety_margin_bytes is not None and dynamic_budget != max(
            0, free_bytes - expected_safety_margin_bytes
        ):
            raise FeatureRowModeValidationError(
                f"{label} dynamic active-row budget does not equal free HBM minus "
                f"the frozen safety margin: {trace_path}"
            )
    return normalized


def _validate_compact_graph(
    *,
    trace_path: Path,
    trace: dict[str, Any],
    label: str,
    expected_feature_count: int | None,
    expected_edge_count: int | None,
    require_canonical_typed_buckets: bool,
) -> dict[str, Any]:
    raw_path = trace.get("graph_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise FeatureRowModeValidationError(
            f"{label} full trace has no graph_path: {trace_path}"
        )
    path = Path(raw_path)
    if not path.is_absolute():
        path = trace_path.parent / path
    if not path.is_file() or path.stat().st_size == 0:
        raise FeatureRowModeValidationError(
            f"{label} full trace graph is missing or empty: {path}"
        )
    try:
        generated_index = trace.get("generated_index")
        graph = load_compact_graph(
            path,
            expected_step_idx=(
                generated_index
                if isinstance(generated_index, int)
                and not isinstance(generated_index, bool)
                else None
            ),
        )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise FeatureRowModeValidationError(
            f"{label} full trace graph failed strict compact loading: {path}: {error}"
        ) from error
    target_position = trace.get("target_position")
    if (
        not isinstance(target_position, int)
        or isinstance(target_position, bool)
        or target_position < 0
    ):
        raise FeatureRowModeValidationError(
            f"{label} full trace has invalid target_position={target_position!r}: "
            f"{trace_path}"
        )
    positions = summarize_feature_positions(
        graph, max_position_exclusive=target_position
    )
    if positions.has_future_positions:
        raise FeatureRowModeValidationError(
            f"{label} full trace graph has {positions.future_position_count} "
            f"feature endpoints at or after target position {target_position}: {path}"
        )
    feature_count = graph.step.n_features
    edge_count = len(graph.step.weights)
    if expected_feature_count is not None and feature_count != expected_feature_count:
        raise FeatureRowModeValidationError(
            f"{label} full trace graph has {feature_count} selected features; "
            f"expected {expected_feature_count}: {path}"
        )
    if expected_edge_count is not None and edge_count != expected_edge_count:
        raise FeatureRowModeValidationError(
            f"{label} full trace graph has {edge_count} compact edges; expected "
            f"{expected_edge_count}: {path}"
        )
    if require_canonical_typed_buckets and (
        graph.compact_save_format != "typed_bucketed"
        or graph.bucket_names != CANONICAL_TYPED_BUCKET_NAMES
    ):
        raise FeatureRowModeValidationError(
            f"{label} full trace graph is not in canonical typed-bucket format: "
            f"format={graph.compact_save_format!r}, buckets={graph.bucket_names!r}: "
            f"{path}"
        )
    return {
        "path": str(path),
        "step_idx": graph.step.step_idx,
        "target_position": target_position,
        "feature_count": feature_count,
        "compact_edge_count": edge_count,
        "compact_save_format": graph.compact_save_format,
        "bucket_names": list(graph.bucket_names),
        "typed_edge_count": (
            0 if graph.bucket_weights is None else len(graph.bucket_weights)
        ),
        "typed_feature_endpoint_count": positions.typed_feature_endpoint_count,
        "max_feature_position": positions.max_position,
        "future_position_count": positions.future_position_count,
    }


def _validate_capture_artifacts(
    *,
    trace_path: Path,
    trace: dict[str, Any],
    requested: set[str],
    label: str,
) -> int:
    report = trace.get("capture_artifact_status")
    if not isinstance(report, dict):
        raise FeatureRowModeValidationError(
            f"{label} requested captures but has no structured capture report: "
            f"{trace_path}"
        )
    reported_requested = {str(name) for name in report.get("requested", [])}
    written = {str(name) for name in report.get("written", [])}
    missing = list(report.get("missing", []))
    failed = list(report.get("failed", []))
    if reported_requested != requested:
        raise FeatureRowModeValidationError(
            f"{label} capture report requested {sorted(reported_requested)!r}; "
            f"trace config requested {sorted(requested)!r}: {trace_path}"
        )
    if report.get("complete") is not True or missing or failed:
        raise FeatureRowModeValidationError(
            f"{label} capture report is incomplete: missing={missing!r}, "
            f"failed={failed!r}: {trace_path}"
        )
    if written != requested:
        raise FeatureRowModeValidationError(
            f"{label} capture report wrote {sorted(written)!r}; expected "
            f"{sorted(requested)!r}: {trace_path}"
        )
    paths = report.get("paths")
    if not isinstance(paths, dict):
        raise FeatureRowModeValidationError(
            f"{label} capture report has no paths mapping: {trace_path}"
        )
    for name in sorted(requested):
        raw_path = paths.get(name)
        if not isinstance(raw_path, str) or not raw_path:
            raise FeatureRowModeValidationError(
                f"{label} capture {name!r} has no artifact path: {trace_path}"
            )
        path = Path(raw_path)
        if not path.is_absolute():
            path = trace_path.parent / path
        if not path.is_file() or path.stat().st_size == 0:
            raise FeatureRowModeValidationError(
                f"{label} capture {name!r} is missing or empty: {path}"
            )
        try:
            load_and_validate_capture_artifact(name, path)
        except (OSError, ValueError) as error:
            raise FeatureRowModeValidationError(
                f"{label} capture {name!r} failed strict schema validation: "
                f"{path}: {error}"
            ) from error
    return len(written)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-mode", required=True)
    parser.add_argument("--expected-backward-engine-mode")
    parser.add_argument("--expected-forward-graph-mode")
    parser.add_argument("--expected-vjp-kernel-mode")
    parser.add_argument("--expected-forward-lane-count", type=int)
    parser.add_argument("--expected-session-capacity", type=int)
    parser.add_argument("--expected-backward-batch-capacity", type=int)
    parser.add_argument("--expected-phase1-trace-batch-size-max", type=int)
    parser.add_argument("--expected-phase1-effective-trace-batch-size", type=int)
    parser.add_argument("--expected-phase3-batch-size", type=int)
    parser.add_argument("--expected-phase4-batch-size", type=int)
    parser.add_argument(
        "--expected-decoder-active-row-residency",
        type=int,
        choices=(0, 1),
    )
    parser.add_argument("--expected-decoder-active-row-residency-requirement")
    parser.add_argument("--expected-decoder-active-row-max-bytes", type=int)
    parser.add_argument("--expected-decoder-active-row-safety-margin-bytes", type=int)
    parser.add_argument("--require-structured-vjp-evidence", action="store_true")
    parser.add_argument("--require-phase4-device-timing", action="store_true")
    parser.add_argument("--expected-phase4-device-timing-backend")
    parser.add_argument(
        "--require-unambiguous-per-device-gpu-evidence", action="store_true"
    )
    parser.add_argument("--expected-graph-feature-count", type=int)
    parser.add_argument("--expected-graph-edge-count", type=int)
    parser.add_argument("--require-canonical-typed-buckets", action="store_true")
    parser.add_argument("--require-full-completion", action="store_true")
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = validate_feature_row_mode(
            output_root=args.output_root,
            expected_mode=args.expected_mode,
            label=args.label,
            expected_backward_engine_mode=args.expected_backward_engine_mode,
            expected_forward_graph_mode=args.expected_forward_graph_mode,
            expected_vjp_kernel_mode=args.expected_vjp_kernel_mode,
            expected_forward_lane_count=args.expected_forward_lane_count,
            expected_session_capacity=args.expected_session_capacity,
            expected_backward_batch_capacity=args.expected_backward_batch_capacity,
            expected_phase1_trace_batch_size_max=(
                args.expected_phase1_trace_batch_size_max
            ),
            expected_phase1_effective_trace_batch_size=(
                args.expected_phase1_effective_trace_batch_size
            ),
            expected_phase3_batch_size=args.expected_phase3_batch_size,
            expected_phase4_batch_size=args.expected_phase4_batch_size,
            expected_decoder_active_row_residency=(
                None
                if args.expected_decoder_active_row_residency is None
                else bool(args.expected_decoder_active_row_residency)
            ),
            expected_decoder_active_row_residency_requirement=(
                args.expected_decoder_active_row_residency_requirement
            ),
            expected_decoder_active_row_max_bytes=(
                args.expected_decoder_active_row_max_bytes
            ),
            expected_decoder_active_row_safety_margin_bytes=(
                args.expected_decoder_active_row_safety_margin_bytes
            ),
            require_structured_vjp_evidence=args.require_structured_vjp_evidence,
            require_phase4_device_timing=args.require_phase4_device_timing,
            expected_phase4_device_timing_backend=(
                args.expected_phase4_device_timing_backend
            ),
            require_unambiguous_per_device_gpu_evidence=(
                args.require_unambiguous_per_device_gpu_evidence
            ),
            expected_graph_feature_count=args.expected_graph_feature_count,
            expected_graph_edge_count=args.expected_graph_edge_count,
            require_canonical_typed_buckets=args.require_canonical_typed_buckets,
            require_full_completion=args.require_full_completion,
        )
    except (FeatureRowModeValidationError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
