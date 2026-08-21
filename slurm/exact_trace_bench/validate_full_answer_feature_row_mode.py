#!/usr/bin/env python3
"""Validate required execution mechanisms of a completed full-answer shard."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from nlp_research_project.exact_trace_bench.backward_selection import (
    backward_mechanism_record,
)
from nlp_research_project.exact_trace_bench.compact_io import load_compact
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FeatureRowModeValidationError(f"expected a JSON object: {path}")
    return payload


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
            _validate_compact_graph(trace_path=trace_path, trace=trace, label=label)
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
        "require_full_completion": require_full_completion,
        "validated_graph_count": graph_count,
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
    *, trace_path: Path, trace: dict[str, Any], label: str
) -> None:
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
        load_compact(path)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise FeatureRowModeValidationError(
            f"{label} full trace graph failed strict compact loading: {path}: {error}"
        ) from error


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
            require_full_completion=args.require_full_completion,
        )
    except (FeatureRowModeValidationError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
