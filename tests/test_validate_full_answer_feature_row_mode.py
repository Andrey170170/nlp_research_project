from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

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
        np.savez_compressed(
            graph_path,
            row_idx=np.asarray([0], dtype=np.int32),
            col_idx=np.asarray([0], dtype=np.int32),
            weights=np.asarray([1.0], dtype=np.float32),
            feature_ids=np.asarray([[0, 0, 1]], dtype=np.int64),
            token_text=np.asarray("token"),
            logprob=np.asarray(-0.5),
            n_features=np.asarray(1, dtype=np.int32),
            step_idx=np.asarray(0, dtype=np.int32),
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
