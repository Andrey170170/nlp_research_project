from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typed_graph_fixtures import write_typed_graph
from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    COMPACT_SAVE_FORMAT,
    DEFAULT_RETENTION_POLICY_ID,
    SCHEMA_VERSION,
    get_retention_policy,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "slurm"
    / "exact_trace_bench"
    / "validate_full_answer_feature_row_mode.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "validate_full_answer_feature_row_mode",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
FeatureRowModeValidationError = _MODULE.FeatureRowModeValidationError
validate_feature_row_mode = _MODULE.validate_feature_row_mode
_validate_phase4_sampled_substage = _MODULE._validate_phase4_sampled_substage


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _hash(value: object, *, dtype: np.dtype | None = None) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype=dtype)
    return hashlib.blake2s(array.view(np.uint8).tobytes(), digest_size=8).hexdigest()


def _valid_capture_payload(name: str) -> dict[str, object]:
    target_ids = np.asarray([7], dtype=np.int64)
    probabilities = np.asarray([0.5], dtype=np.float32)
    common: dict[str, object] = {
        "schema_version": np.asarray(2, dtype=np.int64),
        "status": np.asarray("captured"),
        "target_token_ids": target_ids,
        "target_probabilities": probabilities,
        "target_token_ids_hash": np.asarray(
            _hash(target_ids, dtype=np.dtype(np.int64))
        ),
        "target_probability_hash": np.asarray(
            _hash(probabilities, dtype=np.dtype(np.float64))
        ),
        "active_feature_count": np.asarray(1, dtype=np.int64),
        "active_features_hash": np.asarray("external-feature-hash"),
        "activation_values_hash": np.asarray("external-activation-hash"),
    }
    if name == "phase3_gradient_bundle":
        gradients = np.asarray([[[[0.25]]]], dtype=np.float32)
        return {
            **common,
            "capture_kind": np.asarray("phase3_gradient_bundle_v2"),
            "gradient_batch_representation": np.asarray("canonical_target_width_v1"),
            "canonical_target_width": np.asarray(1, dtype=np.int64),
            "gradients": gradients,
            "layer_mask": np.asarray([True]),
            "batch_call_indices": np.asarray([0], dtype=np.int64),
            "per_layer_abs_sum": np.asarray([0.25], dtype=np.float64),
            "per_layer_max_abs": np.asarray([0.25], dtype=np.float64),
            "per_layer_nonfinite_count": np.asarray([0], dtype=np.int64),
            "per_layer_hashes": np.asarray(
                [_hash(gradients[0], dtype=np.dtype(np.float32))]
            ),
            "gradient_hash": np.asarray(_hash(gradients, dtype=np.dtype(np.float32))),
        }
    if name == "phase3_row_bundle":
        feature_rows = np.asarray([[0.25]], dtype=np.float32)
        error_rows = np.asarray([[[0.125]]], dtype=np.float32)
        token_rows = np.asarray([[-0.125]], dtype=np.float32)
        row_abs_sums = np.asarray([0.5], dtype=np.float64)
        return {
            **common,
            "capture_kind": np.asarray("phase3_row_bundle_v2"),
            "phase3_feature_rows": feature_rows,
            "row_abs_sums": row_abs_sums,
            "feature_abs_sums": np.asarray([0.25], dtype=np.float64),
            "error_abs_sums": np.asarray([0.125], dtype=np.float64),
            "token_abs_sums": np.asarray([0.125], dtype=np.float64),
            "nonfeature_row_layout": np.asarray("target_layer_position_v1"),
            "phase3_error_rows_by_layer": error_rows,
            "phase3_token_rows": token_rows,
            "total_active_features": np.asarray(1, dtype=np.int64),
            "error_column_count": np.asarray(1, dtype=np.int64),
            "token_column_count": np.asarray(1, dtype=np.int64),
            "row_hash": np.asarray(_hash(feature_rows, dtype=np.dtype(np.float32))),
            "row_abs_sum_hash": np.asarray(
                _hash(row_abs_sums, dtype=np.dtype(np.float64))
            ),
            "error_rows_hash": np.asarray(_hash(error_rows)),
            "token_rows_hash": np.asarray(_hash(token_rows)),
        }
    raise AssertionError(f"test fixture does not support capture {name!r}")


def _rewrite_npz(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {field: np.asarray(archive[field]) for field in archive.files}
    mutate(payload)
    np.savez_compressed(path, **payload)


def _upgrade_graph_to_canonical_typed(path: Path) -> None:
    """Historical helper name retained for old test call sites; graph is already v2."""
    return
    with np.load(path, allow_pickle=False) as archive:
        payload = {field: np.asarray(archive[field]) for field in archive.files}
    bucket_names = (
        "feature<-feature",
        "feature<-error",
        "feature<-token",
        "logit<-feature",
        "logit<-error",
        "logit<-token",
    )
    payload.update(
        {
            "compact_save_format": np.asarray("typed_bucketed"),
            "bucket_row_idx": np.asarray([], dtype=np.int64),
            "bucket_col_idx": np.asarray([], dtype=np.int64),
            "bucket_weights": np.asarray([], dtype=np.float32),
            "bucket_ids": np.asarray([], dtype=np.int16),
            "bucket_names": np.asarray(bucket_names),
            "bucket_metadata_json": np.asarray(
                json.dumps(
                    [
                        {
                            "bucket": bucket,
                            "raw_total_abs_mass": 0.0,
                            "retained_abs_mass": 0.0,
                            "retained_fraction": None,
                            "raw_nnz": 0,
                            "retained_nnz": 0,
                            "policy": {"top_p": 1.0, "cap": None},
                            "weights_signed": True,
                        }
                        for bucket in bucket_names
                    ]
                )
            ),
            "error_node_shape": np.asarray([1, 2], dtype=np.int32),
            "token_ids": np.asarray([100, 101], dtype=np.int64),
            "logit_token_ids": np.asarray([42], dtype=np.int64),
        }
    )
    np.savez_compressed(path, **payload)


def _write_shard(
    tmp_path: Path,
    *,
    requested_mode: str,
    resolved_mode: str | None,
    status: str = "complete",
    backward_engine_mode: str = "duplicated_lanes",
    effective_backward_engine_mode: str | None = None,
    effective_forward_graph_mode: str | None = None,
    effective_vjp_kernel_mode: str | None = None,
    effective_forward_lane_count: int | None = None,
    session_capacity: int | None = None,
    backward_batch_capacity: int | None = None,
    phase1_trace_batch_size: int | None = None,
    actual_phase1_trace_batch_size: int | None = None,
    phase3_batch_size: int | None = None,
    phase4_batch_size: int | None = None,
    write_graph: bool = False,
    captures: tuple[str, ...] = (),
    active_row_knobs: dict[str, object] | None = None,
    active_row_evidence: dict[str, object] | None = None,
) -> Path:
    output_root = tmp_path / "run"
    shard_root = output_root / "shards" / "shard_000"
    token_root = shard_root / "token_000001"
    _write_json(shard_root / "shard.json", {"status": status})
    token_root.mkdir(parents=True, exist_ok=True)
    knobs = {
        "feature_row_influence_mode": requested_mode,
        "backward_engine_mode": backward_engine_mode,
    }
    if active_row_knobs is not None:
        knobs.update(active_row_knobs)
    for field, value in (
        ("nnsight_session_capacity", session_capacity),
        ("phase1_trace_batch_size_max", phase1_trace_batch_size),
        ("phase3_compute_microbatch_max_rows", phase3_batch_size),
        ("phase4_execution_batch_max_rows", phase4_batch_size),
    ):
        if value is not None:
            knobs[field] = value
    capture_paths = {}
    for capture in captures:
        knobs[f"capture_{capture}"] = True
        capture_path = token_root / f"{capture}.npz"
        np.savez_compressed(capture_path, **_valid_capture_payload(capture))
        capture_paths[capture] = str(capture_path)
    trace = {
        "status": "probe_completed" if status == "probe_completed" else "ok",
        "generated_index": 1,
        "target_position": 2,
        "graph_knobs": knobs,
        "effective_execution": {
            "batches": {
                "backward_engine_mode": effective_backward_engine_mode,
                "forward_graph_mode": effective_forward_graph_mode,
                "vjp_kernel_mode": effective_vjp_kernel_mode,
                "forward_lane_count": effective_forward_lane_count,
                "session_capacity": session_capacity,
                "backward_batch_capacity": backward_batch_capacity,
                "trace_batch_size": phase1_trace_batch_size,
                "phase3_microbatch_max_rows": phase3_batch_size,
                "phase4_execution_batch_max_rows": phase4_batch_size,
            }
        },
    }
    if write_graph:
        graph_path = token_root / "graph.npz"
        write_typed_graph(
            graph_path,
            step_idx=1,
            feature_ids=[(0, 0, 1)],
            token_text="token",
        )
        trace["graph_path"] = str(graph_path)
    if captures:
        trace["capture_artifact_status"] = {
            "schema_version": 1,
            "requested": list(captures),
            "written": list(captures),
            "missing": [],
            "failed": [],
            "paths": capture_paths,
            "complete": True,
        }
    if active_row_evidence is not None:
        trace["decoder_active_row_residency"] = active_row_evidence
    _write_json(token_root / "trace.json", trace)
    events = []
    if resolved_mode is not None:
        events.append(
            {
                "attrs": {
                    "feature_row_influence_mode_resolved": resolved_mode,
                }
            }
        )
    if actual_phase1_trace_batch_size is None:
        actual_phase1_trace_batch_size = phase1_trace_batch_size
    if actual_phase1_trace_batch_size is not None:
        events.append(
            {
                "phase": "phase1",
                "name": "phase1.forward",
                "attrs": {"trace_batch_size": actual_phase1_trace_batch_size},
            }
        )
    (token_root / "telemetry_live.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return output_root


def _dynamic_active_row_evidence() -> dict[str, object]:
    gib = 1024**3
    free_bytes = 100 * gib
    safety_margin = 16 * gib
    budget = free_bytes - safety_margin
    return {
        "requested": True,
        "effective": True,
        "fallback_reason": None,
        "requirement": "required",
        "admission_reason": "admitted",
        "admission_policy": "live_hbm_headroom",
        "max_bytes_requested": 0,
        "max_bytes_effective": budget,
        "safety_margin_bytes": safety_margin,
        "dynamic_budget_bytes": budget,
        "effective_budget_bytes": budget,
        "resident": {
            "estimated_bytes": 9 * gib,
            "bytes": 9 * gib,
            "row_count": 1,
        },
        "hbm": {
            "free_bytes": free_bytes,
            "total_bytes": 140 * gib,
            "allocated_bytes": 20 * gib,
            "reserved_bytes": 21 * gib,
            "device": "cuda:0",
        },
    }


def _append_telemetry_event(output_root: Path, event: dict[str, Any]) -> None:
    telemetry_path = output_root / "shards/shard_000/token_000001/telemetry_live.jsonl"
    with telemetry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _valid_vjp_event() -> dict[str, Any]:
    return {
        "phase": "phase4",
        "name": "context.compute_batch",
        "attrs": {
            "vjp_requested_path": "autograd_batched",
            "vjp_effective_invocation": "torch.autograd.grad",
            "vjp_is_grads_batched": True,
            "vjp_fallback_state": "unknown",
            "vjp_fallback_observation_method": "direct_call_contract_and_success",
            "vjp_fallback_state_reason": (
                "pytorch_has_no_programmatic_per_invocation_vmap_fallback_signal"
            ),
        },
    }


def _valid_phase4_timing_event() -> dict[str, Any]:
    attrs: dict[str, object] = {
        "phase4_timing_backend": "cuda_events_stratified_jitter_deferred_v1",
        "phase4_timing_device": "cuda:0",
        "phase4_timing_cuda_visible_device_ordinal": 0,
        "phase4_timing_cuda_device_uuid": "GPU-bbb",
        "phase4_timing_unavailable_reason": None,
        "phase4_timing_cuda_device_identity_unavailable_reason": None,
        "phase4_timing_synchronization_scope": ("lifecycle_completion_single_boundary"),
        "phase4_timing_accounting_scope": (
            "non_overlapping_ranges_stratified_cuda_estimate_v1"
        ),
        "phase4_timing_cuda_sampling_scheme": (
            "deterministic_blake2b_jittered_offset_per_stride_block_v1"
        ),
        "phase4_timing_cuda_sampling_seed": 0x0C17C017,
        "phase4_timing_cuda_sampling_hash": "blake2b_64",
        "phase4_timing_cuda_estimator": "horvitz_thompson_stride_weight_v1",
        "phase4_timing_cuda_interval_semantics": (
            "captured_current_stream_interval_latency_includes_host_enqueue_gaps_"
            "and_stream_waits_v1"
        ),
        "phase4_timing_lifecycle_wall_elapsed_ms": 120.0,
        "phase4_timing_lifecycle_cuda_event_elapsed_ms": 100.0,
        "phase4_timing_wall_accounted_elapsed_ms": 80.0,
        "phase4_timing_wall_residual_elapsed_ms": 40.0,
        "phase4_timing_cuda_estimated_accounted_elapsed_ms": 424.0,
        "phase4_timing_cuda_estimated_residual_elapsed_ms": -324.0,
        "phase4_timing_cuda_event_object_count": 18,
        "phase4_timing_cuda_event_record_count": 18,
        "phase4_timing_instrumentation_overhead_scope": (
            "event_construction_and_record_host_calls"
        ),
        "phase4_timing_instrumentation_recording_host_overhead_ms": 0.5,
        "phase4_timing_instrumentation_total_host_overhead_ms": 0.5,
        "phase4_timing_resolution_host_elapsed_ms": 0.25,
    }
    substages = (
        "refresh_row_store_read",
        "refresh_influence_normalization",
        "refresh_direct_accumulation",
        "executor_encoder_materialize",
        "executor_compute_batch",
        "executor_cpu_staging",
        "executor_denominator",
        "executor_row_store_write",
    )
    for substage in substages:
        attrs[f"phase4_timing_{substage}_population_count"] = 1
        attrs[f"phase4_timing_{substage}_cuda_sample_count"] = 1
        attrs[f"phase4_timing_{substage}_cuda_sample_stride"] = (
            16 if substage.startswith("refresh_") else 1
        )
        attrs[f"phase4_timing_{substage}_wall_elapsed_ms"] = 10.0
        attrs[f"phase4_timing_{substage}_cuda_sampled_elapsed_ms"] = 8.0
        attrs[f"phase4_timing_{substage}_cuda_estimated_total_elapsed_ms"] = (
            128.0 if substage.startswith("refresh_") else 8.0
        )
    return {
        "phase": "phase4",
        "name": "phase4.feature_attribution",
        "attrs": attrs,
    }


def _write_valid_resource_summary(output_root: Path) -> None:
    devices = [
        {
            "gpu_index": 0,
            "gpu_uuid": "GPU-aaa",
            "gpu_name": "NVIDIA H200",
            "sample_count": 3,
            "gpu_sm_utilization_mean_percent": 10.0,
            "gpu_sm_utilization_p95_percent": 15.0,
            "gpu_sm_utilization_max_percent": 20.0,
            "gpu_memory_utilization_mean_percent": 5.0,
            "gpu_memory_utilization_p95_percent": 7.0,
            "gpu_memory_utilization_max_percent": 10.0,
            "gpu_power_mean_watts": 100.0,
            "gpu_power_max_watts": 150.0,
            "gpu_framebuffer_peak_mib": 1000.0,
            "gpu_framebuffer_total_mib": 143771.0,
            "gpu_framebuffer_peak_fraction": 1000.0 / 143771.0,
        },
        {
            "gpu_index": 1,
            "gpu_uuid": "GPU-bbb",
            "gpu_name": "NVIDIA H200",
            "sample_count": 3,
            "gpu_sm_utilization_mean_percent": 80.0,
            "gpu_sm_utilization_p95_percent": 90.0,
            "gpu_sm_utilization_max_percent": 100.0,
            "gpu_memory_utilization_mean_percent": 20.0,
            "gpu_memory_utilization_p95_percent": 25.0,
            "gpu_memory_utilization_max_percent": 30.0,
            "gpu_power_mean_watts": 500.0,
            "gpu_power_max_watts": 600.0,
            "gpu_framebuffer_peak_mib": 100000.0,
            "gpu_framebuffer_total_mib": 143771.0,
            "gpu_framebuffer_peak_fraction": 100000.0 / 143771.0,
        },
    ]
    primary = devices[1]
    _write_json(
        output_root / "resource_summary.json",
        {
            "gpu_sampling_status": "ok",
            "resource_validation_passed": True,
            "gpu_sample_count": 6,
            "gpu_device_count": 2,
            "gpu_devices": devices,
            "gpu_summary_scope": "busiest_device",
            "gpu_summary_device_index": 1,
            "gpu_summary_device_uuid": "GPU-bbb",
            "gpu_summary_device_name": "NVIDIA H200",
            **{
                field: primary[field]
                for field in primary
                if field.startswith("gpu_")
                and field not in {"gpu_index", "gpu_uuid", "gpu_name"}
            },
        },
    )


def test_native_cpu_exact_does_not_require_accelerator_resolution_event(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cpu_exact",
        resolved_mode=None,
    )

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cpu_exact",
        label="control",
    )

    assert result["effective_modes"] == ["cpu_exact"]
    assert result["resolution_source"] == "native_request"


def test_accelerated_mode_requires_matching_resolution_event(tmp_path: Path) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
    )

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cuda_windowed",
        label="candidate",
    )

    assert result["effective_modes"] == ["cuda_windowed"]
    assert result["resolution_source"] == "telemetry"


def test_dynamic_active_row_contract_is_validated_postrun(tmp_path: Path) -> None:
    gib = 1024**3
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        active_row_knobs={
            "decoder_active_row_residency": True,
            "decoder_active_row_residency_requirement": "required",
            "decoder_active_row_max_bytes": 0,
            "decoder_active_row_safety_margin_bytes": 16 * gib,
        },
        active_row_evidence=_dynamic_active_row_evidence(),
    )

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cuda_windowed",
        label="dynamic-active-rows",
        expected_decoder_active_row_residency=True,
        expected_decoder_active_row_residency_requirement="required",
        expected_decoder_active_row_max_bytes=0,
        expected_decoder_active_row_safety_margin_bytes=16 * gib,
    )

    assert result["decoder_active_row_evidence"][0]["effective"] is True
    assert (
        result["decoder_active_row_evidence"][0]["admission_policy"]
        == "live_hbm_headroom"
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda evidence: evidence.update(
                {"effective": False, "fallback_reason": "insufficient_hbm"}
            ),
            "was not effective",
        ),
        (lambda evidence: evidence.pop("hbm"), "lacks resident/HBM"),
    ],
)
def test_dynamic_active_row_contract_rejects_incomplete_evidence(
    tmp_path: Path, mutation, match: str
) -> None:
    gib = 1024**3
    evidence = _dynamic_active_row_evidence()
    mutation(evidence)
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        active_row_knobs={
            "decoder_active_row_residency": True,
            "decoder_active_row_residency_requirement": "required",
            "decoder_active_row_max_bytes": 0,
            "decoder_active_row_safety_margin_bytes": 16 * gib,
        },
        active_row_evidence=evidence,
    )

    with pytest.raises(FeatureRowModeValidationError, match=match):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            label="dynamic-active-rows",
            expected_decoder_active_row_residency=True,
            expected_decoder_active_row_residency_requirement="required",
            expected_decoder_active_row_max_bytes=0,
            expected_decoder_active_row_safety_margin_bytes=16 * gib,
        )


def test_preferred_dynamic_active_row_refusal_is_valid_evidence(
    tmp_path: Path,
) -> None:
    gib = 1024**3
    evidence = _dynamic_active_row_evidence()
    evidence.update(
        {
            "effective": False,
            "requirement": "preferred",
            "fallback_reason": "estimated_bytes_exceed_dynamic_budget",
            "admission_reason": "estimated_bytes_exceed_dynamic_budget",
            "dynamic_budget_bytes": 4 * gib,
            "effective_budget_bytes": 4 * gib,
            "hbm": {
                "free_bytes": 20 * gib,
                "total_bytes": 140 * gib,
                "allocated_bytes": 20 * gib,
                "reserved_bytes": 21 * gib,
                "device": "cuda:0",
            },
        }
    )
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        active_row_knobs={
            "decoder_active_row_residency": True,
            "decoder_active_row_residency_requirement": "preferred",
            "decoder_active_row_max_bytes": 0,
            "decoder_active_row_safety_margin_bytes": 16 * gib,
        },
        active_row_evidence=evidence,
    )

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cuda_windowed",
        label="preferred-dynamic-active-rows",
        expected_decoder_active_row_residency=True,
        expected_decoder_active_row_residency_requirement="preferred",
        expected_decoder_active_row_max_bytes=0,
        expected_decoder_active_row_safety_margin_bytes=16 * gib,
    )

    assert result["decoder_active_row_evidence"][0]["effective"] is False


def test_probe_completion_is_a_valid_terminal_mode_result(tmp_path: Path) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        status="probe_completed",
    )

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cuda_windowed",
        label="candidate-probe",
    )

    assert result["status"] == "probe_completed"


def test_requested_capture_contract_is_validated_postrun(tmp_path: Path) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        status="probe_completed",
        captures=("phase3_gradient_bundle", "phase3_row_bundle"),
    )

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cuda_windowed",
        label="capture-probe",
    )

    assert result["requested_capture_names"] == [
        "phase3_gradient_bundle",
        "phase3_row_bundle",
    ]
    assert result["written_capture_count"] == 2


def test_requested_capture_contract_rejects_missing_artifact(tmp_path: Path) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        status="probe_completed",
        captures=("phase3_gradient_bundle",),
    )
    trace_path = output_root / "shards/shard_000/token_000001/trace.json"
    trace = json.loads(trace_path.read_text())
    Path(trace["capture_artifact_status"]["paths"]["phase3_gradient_bundle"]).unlink()

    with pytest.raises(FeatureRowModeValidationError, match="missing or empty"):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            label="capture-probe",
        )


def test_requested_capture_contract_rejects_truncated_npz(tmp_path: Path) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        status="probe_completed",
        captures=("phase3_gradient_bundle",),
    )
    artifact = output_root / "shards/shard_000/token_000001/phase3_gradient_bundle.npz"
    artifact.write_bytes(artifact.read_bytes()[:32])

    with pytest.raises(
        FeatureRowModeValidationError,
        match="failed strict schema validation",
    ):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            label="capture-probe",
        )


def test_requested_capture_contract_rejects_tampered_gradient_hash(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        status="probe_completed",
        captures=("phase3_gradient_bundle",),
    )
    artifact = output_root / "shards/shard_000/token_000001/phase3_gradient_bundle.npz"

    def mutate(payload: dict[str, np.ndarray]) -> None:
        payload["gradients"] = payload["gradients"].copy()
        payload["gradients"][0, 0, 0, 0] = 0.5
        payload["per_layer_abs_sum"] = np.asarray([0.5], dtype=np.float64)
        payload["per_layer_max_abs"] = np.asarray([0.5], dtype=np.float64)

    _rewrite_npz(artifact, mutate)

    with pytest.raises(FeatureRowModeValidationError, match="per_layer_hashes"):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            label="capture-probe",
        )


def test_requested_capture_contract_rejects_missing_schema_field(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        status="probe_completed",
        captures=("phase3_gradient_bundle",),
    )
    artifact = output_root / "shards/shard_000/token_000001/phase3_gradient_bundle.npz"

    def mutate(payload: dict[str, np.ndarray]) -> None:
        del payload["gradient_hash"]

    _rewrite_npz(artifact, mutate)

    with pytest.raises(FeatureRowModeValidationError, match="missing required fields"):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            label="capture-probe",
        )


def test_requested_capture_contract_rejects_row_aggregate_mismatch(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        status="probe_completed",
        captures=("phase3_row_bundle",),
    )
    artifact = output_root / "shards/shard_000/token_000001/phase3_row_bundle.npz"

    def mutate(payload: dict[str, np.ndarray]) -> None:
        payload["error_abs_sums"] = np.asarray([0.25], dtype=np.float64)

    _rewrite_npz(artifact, mutate)

    with pytest.raises(
        FeatureRowModeValidationError,
        match="row_abs_sums do not match feature/error/token split sums",
    ):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            label="capture-probe",
        )


def test_full_run_without_requested_captures_remains_valid(tmp_path: Path) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cpu_exact",
        resolved_mode=None,
        status="complete",
    )

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cpu_exact",
        label="full-run",
    )

    assert result["requested_capture_names"] == []
    assert result["written_capture_count"] == 0


def test_strict_full_completion_reopens_graph_and_freezes_capacities(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        backward_engine_mode="single_forward_batched_vjp",
        effective_backward_engine_mode="single_forward_batched_vjp",
        effective_forward_graph_mode="single_lane",
        effective_vjp_kernel_mode="autograd_batched",
        effective_forward_lane_count=1,
        session_capacity=128,
        backward_batch_capacity=128,
        phase1_trace_batch_size=128,
        actual_phase1_trace_batch_size=1,
        phase3_batch_size=128,
        phase4_batch_size=128,
        write_graph=True,
    )

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cuda_windowed",
        expected_backward_engine_mode="single_forward_batched_vjp",
        expected_forward_graph_mode="single_lane",
        expected_vjp_kernel_mode="autograd_batched",
        expected_forward_lane_count=1,
        expected_session_capacity=128,
        expected_backward_batch_capacity=128,
        expected_phase1_trace_batch_size_max=128,
        expected_phase1_effective_trace_batch_size=1,
        expected_phase3_batch_size=128,
        expected_phase4_batch_size=128,
        require_full_completion=True,
        label="full-self-consistency",
    )

    assert result["validated_graph_count"] == 1
    assert result["effective_backward_batch_capacities"] == [128]
    assert result["effective_forward_lane_counts"] == [1]
    assert result["prepared_trace_batch_sizes"] == [128]
    assert result["actual_phase1_trace_batch_sizes"] == [1]


def test_strict_full_completion_rejects_probe_result(tmp_path: Path) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        status="probe_completed",
    )

    with pytest.raises(FeatureRowModeValidationError, match="shard status"):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            require_full_completion=True,
            label="full-self-consistency",
        )


def test_strict_full_completion_rejects_unreadable_graph(tmp_path: Path) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        write_graph=True,
    )
    graph = output_root / "shards/shard_000/token_000001/graph.npz"
    graph.write_bytes(graph.read_bytes()[:32])

    with pytest.raises(
        FeatureRowModeValidationError,
        match="failed strict compact loading",
    ):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            require_full_completion=True,
            label="full-self-consistency",
        )


def test_strict_full_completion_enforces_exact_canonical_graph_contract(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        write_graph=True,
    )
    graph = output_root / "shards/shard_000/token_000001/graph.npz"
    _upgrade_graph_to_canonical_typed(graph)

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cuda_windowed",
        expected_graph_feature_count=1,
        require_typed_only_graph=True,
        expected_graph_schema_version=SCHEMA_VERSION,
        expected_graph_save_format=COMPACT_SAVE_FORMAT,
        expected_graph_retention_policy_id=DEFAULT_RETENTION_POLICY_ID,
        expected_graph_retention_policy_fingerprint=get_retention_policy().fingerprint,
        require_full_completion=True,
        label="strict-graph-contract",
    )

    report = result["graph_validation_reports"][0]
    assert report["path"] == str(graph)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["compact_save_format"] == COMPACT_SAVE_FORMAT
    assert report["retention_policy_id"] == DEFAULT_RETENTION_POLICY_ID
    assert report["retention_policy_fingerprint"] == get_retention_policy().fingerprint
    assert set(report["typed_edge_count_by_bucket"]) == set(report["bucket_names"])
    assert report["typed_edge_count"] == sum(
        report["typed_edge_count_by_bucket"].values()
    )
    assert report["future_position_count"] == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"expected_graph_feature_count": 2}, "selected features"),
        ({"expected_graph_edge_count": 2}, "removed legacy global-projection"),
        ({"expected_graph_schema_version": 999}, "schema_version"),
    ],
)
def test_strict_full_completion_rejects_graph_acceptance_mismatch(
    tmp_path: Path, kwargs: dict[str, object], match: str
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        write_graph=True,
    )

    with pytest.raises(FeatureRowModeValidationError, match=match):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            require_full_completion=True,
            label="strict-graph-contract",
            **kwargs,
        )


def test_strict_full_completion_rejects_target_or_future_feature_position(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        write_graph=True,
    )
    trace_path = output_root / "shards/shard_000/token_000001/trace.json"
    trace = json.loads(trace_path.read_text())
    trace["target_position"] = 0
    _write_json(trace_path, trace)

    with pytest.raises(FeatureRowModeValidationError, match="at or after target"):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            require_full_completion=True,
            label="strict-position-contract",
        )


def test_strict_capacity_gate_rejects_effective_mismatch(tmp_path: Path) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        session_capacity=128,
    )
    trace_path = output_root / "shards/shard_000/token_000001/trace.json"
    trace = json.loads(trace_path.read_text())
    trace["effective_execution"]["batches"]["session_capacity"] = 64
    _write_json(trace_path, trace)

    with pytest.raises(
        FeatureRowModeValidationError, match="effective session capacities"
    ):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            expected_session_capacity=128,
            label="full-self-consistency",
        )


def test_accelerated_mode_rejects_cpu_fallback(tmp_path: Path) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cpu_exact",
    )

    with pytest.raises(FeatureRowModeValidationError, match="resolved feature-row"):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            label="candidate",
        )


def test_required_backward_engine_and_forward_lane_topology_match(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        backward_engine_mode="single_forward_batched_vjp",
        effective_backward_engine_mode="single_forward_batched_vjp",
        effective_forward_graph_mode="single_lane",
        effective_vjp_kernel_mode="autograd_batched",
        effective_forward_lane_count=1,
        status="probe_completed",
    )

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cuda_windowed",
        expected_backward_engine_mode="single_forward_batched_vjp",
        expected_forward_graph_mode="single_lane",
        expected_vjp_kernel_mode="autograd_batched",
        expected_forward_lane_count=1,
        label="batched-vjp-probe",
    )

    assert result["effective_backward_engine_modes"] == ["single_forward_batched_vjp"]
    assert result["effective_forward_lane_counts"] == [1]
    assert result["effective_forward_graph_modes"] == ["single_lane"]
    assert result["effective_vjp_kernel_modes"] == ["autograd_batched"]


def test_required_decomposed_identity_rejects_kernel_mismatch(tmp_path: Path) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        backward_engine_mode="single_forward_serial_vjp",
        effective_backward_engine_mode="single_forward_serial_vjp",
        effective_forward_graph_mode="single_lane",
        effective_vjp_kernel_mode="autograd_batched",
        effective_forward_lane_count=1,
    )

    with pytest.raises(FeatureRowModeValidationError, match="effective VJP kernel"):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            expected_backward_engine_mode="single_forward_serial_vjp",
            expected_forward_graph_mode="single_lane",
            expected_vjp_kernel_mode="autograd_serial",
            expected_forward_lane_count=1,
            label="serial-vjp-candidate",
        )


@pytest.mark.parametrize(
    ("effective_backward_engine_mode", "effective_forward_lane_count", "match"),
    [
        ("duplicated_lanes", 1, "effective backward engines"),
        ("single_forward_batched_vjp", 8, "effective forward-lane counts"),
    ],
)
def test_required_backward_topology_rejects_effective_mismatch(
    tmp_path: Path,
    effective_backward_engine_mode: str,
    effective_forward_lane_count: int,
    match: str,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        backward_engine_mode="single_forward_batched_vjp",
        effective_backward_engine_mode=effective_backward_engine_mode,
        effective_forward_lane_count=effective_forward_lane_count,
    )

    with pytest.raises(FeatureRowModeValidationError, match=match):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            expected_backward_engine_mode="single_forward_batched_vjp",
            expected_forward_lane_count=1,
            label="batched-vjp-candidate",
        )


def test_evidence_hardening_accepts_honest_unknown_vjp_and_device_identity(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        backward_engine_mode="single_forward_batched_vjp",
        effective_backward_engine_mode="single_forward_batched_vjp",
        effective_forward_graph_mode="single_lane",
        effective_vjp_kernel_mode="autograd_batched",
        effective_forward_lane_count=1,
    )
    _append_telemetry_event(output_root, _valid_vjp_event())
    _append_telemetry_event(output_root, _valid_phase4_timing_event())
    _write_valid_resource_summary(output_root)

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cuda_windowed",
        expected_vjp_kernel_mode="autograd_batched",
        require_structured_vjp_evidence=True,
        require_phase4_device_timing=True,
        expected_phase4_device_timing_backend=(
            "cuda_events_stratified_jitter_deferred_v1"
        ),
        require_unambiguous_per_device_gpu_evidence=True,
        label="evidence-hardened-run",
    )

    assert result["structured_vjp_evidence"]["fallback_states"] == ["unknown"]
    assert result["phase4_device_timing"]["cuda_visible_device_ordinals"] == [0]
    assert result["phase4_device_timing"]["optional_substage_presence"] == {
        "refresh_influence_matmul": False,
        "refresh_transfer_cast_abs": False,
    }
    assert result["per_device_gpu_resource_evidence"]["device_count"] == 2


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("vjp_fallback_observation_method", None, "observation_method"),
        (
            "vjp_fallback_observation_method",
            "pytorch_runtime_warning_only",
            "observation method",
        ),
        ("vjp_is_grads_batched", False, "inconsistent"),
        ("vjp_fallback_state", "observed", "honest observable state"),
        ("vjp_fallback_state_reason", None, "fallback_state_reason"),
        ("vjp_fallback_state_reason", "no warnings seen", "unknown fallback reason"),
    ],
)
def test_structured_vjp_gate_rejects_incomplete_or_inconsistent_evidence(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
        backward_engine_mode="single_forward_batched_vjp",
        effective_backward_engine_mode="single_forward_batched_vjp",
        effective_forward_graph_mode="single_lane",
        effective_vjp_kernel_mode="autograd_batched",
        effective_forward_lane_count=1,
    )
    event = _valid_vjp_event()
    event["attrs"][field] = value
    _append_telemetry_event(output_root, event)

    with pytest.raises(FeatureRowModeValidationError, match=match):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            expected_vjp_kernel_mode="autograd_batched",
            require_structured_vjp_evidence=True,
            label="vjp-evidence",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda attrs: attrs.__setitem__("phase4_timing_backend", "wall_clock_v1"),
            "timing backend",
        ),
        (
            lambda attrs: attrs.pop(
                "phase4_timing_executor_compute_batch_population_count"
            ),
            "executor_compute_batch_population_count",
        ),
        (
            lambda attrs: attrs.__setitem__(
                "phase4_timing_cuda_estimated_residual_elapsed_ms", None
            ),
            "cuda_estimated_residual_elapsed_ms",
        ),
        (
            lambda attrs: attrs.__setitem__(
                "phase4_timing_cuda_estimated_residual_elapsed_ms", 21.0
            ),
            "does not reconcile",
        ),
        (
            lambda attrs: attrs.__setitem__(
                "phase4_timing_refresh_row_store_read_cuda_sample_stride", 1
            ),
            "expected_stride=16",
        ),
        (
            lambda attrs: attrs.__setitem__(
                "phase4_timing_executor_compute_batch_cuda_sample_count", 2
            ),
            "inconsistent sampled timing counts",
        ),
        (
            lambda attrs: attrs.__setitem__(
                "phase4_timing_refresh_row_store_read_cuda_estimated_total_elapsed_ms",
                8.0,
            ),
            "sampled CUDA estimate",
        ),
        (
            lambda attrs: attrs.__setitem__(
                "phase4_timing_cuda_sampling_scheme", "fixed_offset_v0"
            ),
            "sampling contract is unsupported",
        ),
        (
            lambda attrs: attrs.__setitem__(
                "phase4_timing_wall_residual_elapsed_ms", -1.0
            ),
            "negative",
        ),
        (
            lambda attrs: attrs.__setitem__(
                "phase4_timing_unavailable_reason", "event_resolution_failed"
            ),
            "marked unavailable",
        ),
    ],
)
def test_phase4_device_timing_gate_rejects_incomplete_evidence(
    tmp_path: Path, mutation, match: str
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
    )
    event = _valid_phase4_timing_event()
    mutation(event["attrs"])
    _append_telemetry_event(output_root, event)

    with pytest.raises(FeatureRowModeValidationError, match=match):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            require_phase4_device_timing=True,
            expected_phase4_device_timing_backend=(
                "cuda_events_stratified_jitter_deferred_v1"
            ),
            label="phase4-timing",
        )


def test_stratified_timing_accepts_unsampled_partial_stride_block(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
    )
    event = _valid_phase4_timing_event()
    event["attrs"]["phase4_timing_refresh_row_store_read_population_count"] = 17
    event["attrs"]["phase4_timing_refresh_row_store_read_cuda_sample_count"] = 1
    _append_telemetry_event(output_root, event)

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cuda_windowed",
        require_phase4_device_timing=True,
        expected_phase4_device_timing_backend=(
            "cuda_events_stratified_jitter_deferred_v1"
        ),
        label="stratified-partial-block",
    )

    assert result["phase4_device_timing"]["event_count"] == 1


@pytest.mark.parametrize(
    "mutation", ["duplicate_uuid", "unmapped_primary", "sample_count_mismatch"]
)
def test_per_device_gpu_gate_rejects_ambiguous_identity(
    tmp_path: Path, mutation: str
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
    )
    _write_valid_resource_summary(output_root)
    path = output_root / "resource_summary.json"
    summary = json.loads(path.read_text())
    if mutation == "duplicate_uuid":
        summary["gpu_devices"][1]["gpu_uuid"] = "GPU-aaa"
    elif mutation == "unmapped_primary":
        summary["gpu_summary_device_uuid"] = "GPU-missing"
    else:
        summary["gpu_devices"][1]["sample_count"] = 2
    _write_json(path, summary)

    with pytest.raises(FeatureRowModeValidationError, match="GPU|busiest"):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            require_unambiguous_per_device_gpu_evidence=True,
            label="gpu-evidence",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_name", "ambiguous GPU device identity"),
        ("utilization_over_100", "outside.*100"),
        ("negative_power", "negative gpu_power_mean_watts"),
        ("framebuffer_peak_over_total", "framebuffer summary is inconsistent"),
        ("framebuffer_fraction_mismatch", "framebuffer summary is inconsistent"),
        ("top_level_metric_mismatch", "top-level gpu_power_max_watts"),
    ],
)
def test_per_device_gpu_gate_rejects_invalid_metrics(
    tmp_path: Path, mutation: str, match: str
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
    )
    _write_valid_resource_summary(output_root)
    path = output_root / "resource_summary.json"
    summary = json.loads(path.read_text())
    primary = summary["gpu_devices"][1]
    if mutation == "missing_name":
        primary["gpu_name"] = ""
    elif mutation == "utilization_over_100":
        primary["gpu_sm_utilization_max_percent"] = 101.0
    elif mutation == "negative_power":
        primary["gpu_power_mean_watts"] = -1.0
    elif mutation == "framebuffer_peak_over_total":
        primary["gpu_framebuffer_peak_mib"] = 200000.0
    elif mutation == "framebuffer_fraction_mismatch":
        primary["gpu_framebuffer_peak_fraction"] = 0.5
    else:
        summary["gpu_power_max_watts"] = 599.0
    _write_json(path, summary)

    with pytest.raises(FeatureRowModeValidationError, match=match):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            require_unambiguous_per_device_gpu_evidence=True,
            label="gpu-metrics",
        )


def test_phase4_timing_uuid_must_match_busiest_sampled_device(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
    )
    event = _valid_phase4_timing_event()
    event["attrs"]["phase4_timing_cuda_device_uuid"] = "GPU-aaa"
    _append_telemetry_event(output_root, event)
    _write_valid_resource_summary(output_root)

    with pytest.raises(FeatureRowModeValidationError, match="busiest sampled GPU UUID"):
        validate_feature_row_mode(
            output_root=output_root,
            expected_mode="cuda_windowed",
            require_phase4_device_timing=True,
            expected_phase4_device_timing_backend=(
                "cuda_events_stratified_jitter_deferred_v1"
            ),
            require_unambiguous_per_device_gpu_evidence=True,
            label="cross-source-device-identity",
        )


def test_phase4_timing_uuid_accepts_torch_uuid_without_nvidia_smi_prefix(
    tmp_path: Path,
) -> None:
    output_root = _write_shard(
        tmp_path,
        requested_mode="cuda_windowed",
        resolved_mode="cuda_windowed",
    )
    event = _valid_phase4_timing_event()
    event["attrs"]["phase4_timing_cuda_device_uuid"] = "bbb"
    _append_telemetry_event(output_root, event)
    _write_valid_resource_summary(output_root)

    result = validate_feature_row_mode(
        output_root=output_root,
        expected_mode="cuda_windowed",
        require_phase4_device_timing=True,
        expected_phase4_device_timing_backend=(
            "cuda_events_stratified_jitter_deferred_v1"
        ),
        require_unambiguous_per_device_gpu_evidence=True,
        label="cross-source-device-identity",
    )

    assert result["phase4_device_timing"]["cuda_device_uuids"] == ["bbb"]
    assert (
        result["per_device_gpu_resource_evidence"]["summary_device_uuid"] == "GPU-bbb"
    )


def test_v2_device_timing_uses_actual_sampled_tail_size() -> None:
    attrs = {
        "phase4_timing_refresh_influence_matmul_population_count": 34,
        "phase4_timing_refresh_influence_matmul_cuda_sample_count": 3,
        "phase4_timing_refresh_influence_matmul_cuda_recorded_sample_count": 6,
        "phase4_timing_refresh_influence_matmul_cuda_sample_stride": 16,
        "phase4_timing_refresh_influence_matmul_cuda_tail_population_count": 2,
        "phase4_timing_refresh_influence_matmul_cuda_tail_status": "sampled",
        "phase4_timing_refresh_influence_matmul_cuda_tail_sample_source": (
            "hashed_primary"
        ),
        "phase4_timing_refresh_influence_matmul_cuda_estimate_status": "complete",
        "phase4_timing_refresh_influence_matmul_wall_elapsed_ms": 20.0,
        "phase4_timing_refresh_influence_matmul_cuda_sampled_elapsed_ms": 7.5,
        "phase4_timing_refresh_influence_matmul_cuda_complete_block_sampled_elapsed_ms": 5.0,
        "phase4_timing_refresh_influence_matmul_cuda_incomplete_tail_sampled_elapsed_ms": 2.5,
        "phase4_timing_refresh_influence_matmul_cuda_estimated_total_elapsed_ms": 85.0,
    }

    _validate_phase4_sampled_substage(
        attrs=attrs,
        substage="refresh_influence_matmul",
        expected_stride=16,
        contract_version="phase4_device_timing_v2",
        label="v2-tail",
        source=Path("telemetry.jsonl"),
    )


def test_v2_device_timing_refuses_unsampled_final_stratum() -> None:
    attrs = {
        "phase4_timing_refresh_influence_matmul_population_count": 33,
        "phase4_timing_refresh_influence_matmul_cuda_sample_count": 2,
        "phase4_timing_refresh_influence_matmul_cuda_recorded_sample_count": 2,
        "phase4_timing_refresh_influence_matmul_cuda_sample_stride": 16,
        "phase4_timing_refresh_influence_matmul_cuda_tail_population_count": 1,
        "phase4_timing_refresh_influence_matmul_cuda_tail_status": "unsampled",
        "phase4_timing_refresh_influence_matmul_cuda_tail_sample_source": "missing",
        "phase4_timing_refresh_influence_matmul_cuda_estimate_status": (
            "refused_incomplete_tail"
        ),
        "phase4_timing_refresh_influence_matmul_wall_elapsed_ms": 20.0,
        "phase4_timing_refresh_influence_matmul_cuda_sampled_elapsed_ms": 5.0,
        "phase4_timing_refresh_influence_matmul_cuda_complete_block_sampled_elapsed_ms": 5.0,
        "phase4_timing_refresh_influence_matmul_cuda_incomplete_tail_sampled_elapsed_ms": 0.0,
        "phase4_timing_refresh_influence_matmul_cuda_estimated_total_elapsed_ms": None,
    }

    with pytest.raises(FeatureRowModeValidationError, match="one sample per stratum"):
        _validate_phase4_sampled_substage(
            attrs=attrs,
            substage="refresh_influence_matmul",
            expected_stride=16,
            contract_version="phase4_device_timing_v2",
            label="v2-tail",
            source=Path("telemetry.jsonl"),
        )


@pytest.mark.parametrize(
    "substage",
    ["refresh_influence_normalization", "refresh_direct_accumulation"],
)
def test_v2_device_timing_accepts_24101_fallback_tail(substage: str) -> None:
    prefix = f"phase4_timing_{substage}"
    attrs = {
        f"{prefix}_population_count": 24_101,
        f"{prefix}_cuda_sample_count": 1_507,
        f"{prefix}_cuda_recorded_sample_count": 3_000,
        f"{prefix}_cuda_sample_stride": 16,
        f"{prefix}_cuda_tail_population_count": 5,
        f"{prefix}_cuda_tail_status": "sampled",
        f"{prefix}_cuda_tail_sample_source": "block_start_fallback",
        f"{prefix}_cuda_estimate_status": "complete",
        f"{prefix}_wall_elapsed_ms": 20.0,
        f"{prefix}_cuda_sampled_elapsed_ms": 3_767.5,
        f"{prefix}_cuda_complete_block_sampled_elapsed_ms": 3_765.0,
        f"{prefix}_cuda_incomplete_tail_sampled_elapsed_ms": 2.5,
        f"{prefix}_cuda_estimated_total_elapsed_ms": 60_252.5,
    }

    _validate_phase4_sampled_substage(
        attrs=attrs,
        substage=substage,
        expected_stride=16,
        contract_version="phase4_device_timing_v2",
        label="v2-24101-tail",
        source=Path("telemetry.jsonl"),
    )
