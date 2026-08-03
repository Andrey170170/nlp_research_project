from __future__ import annotations

import json
import builtins
import argparse
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_research_project.exact_trace_bench.full_answer.aggregate import (
    aggregate_shards,
)
from nlp_research_project.exact_trace_bench import cli as full_answer_cli
from nlp_research_project.exact_trace_bench import compact_io
from nlp_research_project.exact_trace_bench.full_answer import runner as runner_module
from nlp_research_project.exact_trace_bench.full_answer.runner import (
    _trace_request,
    dry_run_shard,
    forced_target_payload,
    list_shard_specs,
    load_shard_inputs,
    prefix_view_metadata,
    prepare_full_sequence_cache,
    reconstruct_full_sequence_token_ids,
    reconstruct_prefix_token_ids,
    run_real_shard,
    session_reuse_metadata,
)
from nlp_research_project.exact_trace_bench.trace_runtime import provider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _stub_compact_packager(monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "_compact_result_to_bucketed_compact",
        lambda *_args, **_kwargs: types.SimpleNamespace(step={}),
    )
    monkeypatch.setattr(
        compact_io,
        "save_bucketed_compact",
        lambda _bundle, path: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_text("graph"),
        ),
    )


def _write_tiny_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    trajectory = {
        "schema_version": 1,
        "trajectory_id": "traj_runner",
        "prompt_token_count": 2,
        "prompt_token_ids": [101, 102],
        "generated_tokens": [
            {
                "generated_index": 0,
                "absolute_token_position": 2,
                "token_id": 201,
                "token_text": "A",
                "is_stop": False,
            },
            {
                "generated_index": 1,
                "absolute_token_position": 3,
                "token_id": 202,
                "token_text": "7",
                "is_stop": False,
            },
        ],
    }
    specs = [
        {
            "schema_version": 1,
            "trace_id": "traj_runner_tok000000",
            "trajectory_id": "traj_runner",
            "generated_index": 0,
            "target_position": 2,
            "prefix_token_count": 2,
            "target_token_id": 201,
            "target_token_text": "A",
            "target_mode": "frozen_target_only",
            "selection_reasons": ["explicit"],
            "graph_knobs": {"max_edges": 10},
            "estimated_cost": 2,
        },
        {
            "schema_version": 1,
            "trace_id": "traj_runner_tok000001",
            "trajectory_id": "traj_runner",
            "generated_index": 1,
            "target_position": 3,
            "prefix_token_count": 3,
            "target_token_id": 202,
            "target_token_text": "7",
            "target_mode": "frozen_target_only",
            "selection_reasons": ["numeric"],
            "graph_knobs": {"max_edges": 10},
            "estimated_cost": 3,
        },
    ]
    shards = {
        "schema_version": 1,
        "trace_specs_file": "trace_specs.jsonl",
        "cost_model": "prefix_token_count_lpt_v1",
        "shards": [
            {"shard_id": 0, "estimated_cost_sum": 3, "spec_indices": [1]},
            {"shard_id": 1, "estimated_cost_sum": 2, "spec_indices": [0]},
        ],
    }
    trajectory_path = tmp_path / "trajectory.json"
    specs_path = tmp_path / "trace_specs.jsonl"
    shards_path = tmp_path / "shards.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    specs_path.write_text(
        "\n".join(json.dumps(spec) for spec in specs) + "\n", encoding="utf-8"
    )
    shards_path.write_text(json.dumps(shards), encoding="utf-8")
    return trajectory_path, specs_path, shards_path


def test_dry_run_shard_writes_expected_files_and_metadata(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    result = dry_run_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )
    assert result["status"] == "dry_run"
    shard_dir = tmp_path / "run" / "shards" / "shard_000"
    assert (shard_dir / "shard.json").exists()
    trace_path = shard_dir / "token_000001" / "trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["status"] == "dry_run"
    assert trace["graph_path"] is None
    assert trace["forced_target"] == {
        "token_id": 202,
        "token_text": "7",
        "target_mode": "frozen_target_only",
        "attribution_targets": [202],
    }
    assert trace["selection_reasons"] == ["numeric"]
    assert trace["target_position"] == 3
    assert trace["prefix_token_count"] == 3
    shard = json.loads((shard_dir / "shard.json").read_text(encoding="utf-8"))
    assert shard["target_positions"] == [3]
    assert (shard_dir / "trace_results.jsonl").exists()


def test_list_mode_returns_specs_without_writing_token_dirs(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    rows = list_shard_specs(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=1,
    )
    assert [row["generated_index"] for row in rows] == [0]
    assert not (tmp_path / "shards").exists()


def test_trace_request_builds_canonical_domain_policies(tmp_path: Path) -> None:
    from circuit_tracer import TraceRequest

    spec = cast(
        Any,
        {
            "target_token_id": 7,
            "target_position": 3,
            "graph_knobs": {
            "row_subchunk_size": 128,
            "plan_feature_batch_size": True,
            "feature_batch_size_max": 512,
            "feature_vjp_tape_batch_window": 2,
            "feature_vjp_tape_max_bytes": 4096,
            "decoder_page_prefetch_depth": 1,
            "decoder_active_row_residency": True,
            "decoder_active_row_max_bytes": 8192,
            "phase0_decoder_row_ranges": True,
            "transcoder_architecture": "plt",
            "transcoder_provider_family": "gemmascope2-plt-1b-big-affine",
            "phase4_scheduler_mode": "planner_v1",
            "phase4_scheduler_telemetry_detail": "debug",
            "phase4_refresh_optimization": "v1",
            "phase4_refresh_prepared_chunk_cache_bytes": 0,
            "phase4_refresh_active_row_accumulation": "direct_v1",
            "phase4_row_executor": "streaming_v1",
            "phase4_row_reduction": "gpu_v1",
            "phase3_frontier_buffer_relative_epsilon": 0.01,
            "phase3_frontier_buffer_max_extra": 256,
            "phase4_frontier_buffer_relative_epsilon": 0.02,
            "phase4_frontier_buffer_max_extra_per_refresh": 16,
            "phase4_frontier_buffer_max_extra_total": 128,
            "row_store_cache_control": "fadvise_dontneed_after_append_v1",
            "row_store_preallocate": True,
            "cross_cluster_debug": True,
            "capture_phase0_donor_bundle": True,
            "capture_phase3_seed_bundle": True,
            "capture_feature_semantic_descriptors": True,
            "semantic_descriptor_top_k": 1024,
            "semantic_descriptor_dim": 32,
            "decoder_chunk_size": 256,
            "cross_batch_decoder_cache_bytes": 8589934592,
            "feature_batch_size": None,
            "chunked_feature_replay_window": 16,
            "error_vector_prefetch_lookahead": 8,
            "stage_encoder_vecs_on_cpu": False,
            "stage_error_vectors_on_cpu": False,
            "exact_encoder_residency": "active_cpu",
            "phase1_trace_batch_policy": "cap_effective_batches",
            "phase1_trace_batch_size_max": 16,
            "feature_row_retention": "none_recompute",
            "full_retention_backend": "column_tiled_v1",
            "feature_row_influence_mode": "cuda_windowed",
            "feature_row_gpu_resident_max_bytes": 1024,
            "feature_row_gpu_window_max_bytes": 2048,
            "feature_row_gpu_resident_safety_margin_bytes": 4096,
            "nnsight_session_capacity": 64,
            "telemetry_max_events": 500,
            "diagnostic_stop_mode": "transition_probe",
            "diagnostic_stop_phase4_batches": 2,
            },
        },
    )
    model = types.SimpleNamespace(backend="nnsight")
    request = _trace_request(
        model=model,
        prompt_token_ids=[101, 102, 201],
        spec=spec,
        prefix_metadata={"mode": "independent_prefix"},
        full_sequence_mode=False,
        telemetry_jsonl_path=tmp_path / "telemetry_live.jsonl",
    )

    assert isinstance(request, TraceRequest)
    assert request.problem.model is model
    assert request.problem.prompt.tolist() == [101, 102, 201]
    assert request.problem.targets.tolist() == [7]
    assert request.problem.prefix_view.mode == "independent_prefix"
    assert request.problem.prefix_view.target_position == 3
    assert request.semantics.source_batch_size == 256
    assert request.execution.session.capacity == 64
    assert request.execution.session.phase1_trace_batch_size_max == 16
    assert request.execution.storage.retention == "none_recompute"
    assert request.execution.storage.full_retention_backend == "column_tiled_v1"
    assert request.execution.storage.feature_row_influence_mode == "cuda_windowed"
    assert request.execution.storage.gpu_resident_max_bytes == 1024
    assert request.execution.storage.gpu_window_max_bytes == 2048
    assert request.execution.storage.gpu_resident_safety_margin_bytes == 4096
    assert request.execution.replay.decoder_contraction_tile == 128
    assert request.execution.frontier.feature_vjp_tape_batch_window == 2
    assert request.execution.frontier.feature_vjp_tape_max_bytes == 4096
    assert request.execution.frontier.decoder_page_prefetch_depth == 1
    assert request.execution.frontier.decoder_active_row_residency is True
    assert request.execution.frontier.decoder_active_row_max_bytes == 8192
    assert request.execution.frontier.phase0_decoder_row_ranges is True
    assert request.semantics.frontier.scheduler == "planner_v1"
    assert request.execution.session.decoder_cache.enabled is True
    assert request.execution.session.decoder_cache.max_bytes == 8589934592
    assert request.execution.observability.telemetry_max_events == 500
    assert request.execution.diagnostic_stop.mode == "transition_probe"
    assert request.execution.diagnostic_stop.phase4_batches == 2
    assert request.execution.observability.telemetry_jsonl_path == (
        tmp_path / "telemetry_live.jsonl"
    )
    assert request.evidence.metadata["prefix_view_metadata"] == {}


@pytest.mark.parametrize(
    ("knob", "value", "message"),
    [
        (
            "decoder_active_row_residency",
            1,
            "decoder_active_row_residency must be a bool",
        ),
        (
            "decoder_active_row_max_bytes",
            -1,
            "decoder_active_row_max_bytes must be a non-negative int",
        ),
        (
            "decoder_active_row_max_bytes",
            True,
            "decoder_active_row_max_bytes must be a non-negative int",
        ),
        (
            "phase0_decoder_row_ranges",
            1,
            "phase0_decoder_row_ranges must be a bool",
        ),
    ],
)
def test_model_load_knobs_rejects_invalid_active_row_controls(
    knob: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        runner_module._model_load_knobs(
            [cast(Any, {"graph_knobs": {knob: value}})]
        )


def test_trace_request_uses_legacy_phase4_rows_when_canonical_default_is_none() -> None:
    spec = cast(
        Any,
        {
            "target_token_id": 7,
            "target_position": 3,
            "graph_knobs": {
                "phase4_execution_batch_max_rows": None,
                "phase4_compute_microbatch_max_rows": 23,
            },
        },
    )

    request = _trace_request(
        model=types.SimpleNamespace(backend="nnsight"),
        prompt_token_ids=[101, 102, 201],
        spec=spec,
        prefix_metadata={"mode": "independent_prefix"},
        full_sequence_mode=False,
    )

    assert request.execution.session.phase4_execution_batch_max_rows == 23


def test_prefix_view_metadata_matches_reconstructed_prefix(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    trajectory, specs, _shard = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
    )
    prefix = reconstruct_prefix_token_ids(trajectory, specs[0])

    metadata = prefix_view_metadata(trajectory, specs[0], prefix)

    assert metadata["mode"] == "independent_prefix"
    assert metadata["trajectory_id"] == "traj_runner"
    assert metadata["trace_id"] == "traj_runner_tok000001"
    assert metadata["target_position"] == 3
    assert metadata["prefix_token_count"] == len(prefix) == 3
    assert metadata["target_token_ids"] == [202]


def test_full_sequence_cache_is_reused_for_prefix_metadata(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    trajectory, specs, _shard = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
    )
    specs[0]["graph_knobs"]["input_context_mode"] = "full_sequence"
    prefix = reconstruct_prefix_token_ids(trajectory, specs[0])
    cache = prepare_full_sequence_cache(trajectory, specs)
    assert cache is not None

    metadata = prefix_view_metadata(
        trajectory,
        specs[0],
        prefix,
        full_sequence_token_ids=cache["token_ids"],
        full_sequence_token_ids_sha256=cache["token_ids_sha256"],
    )

    assert metadata["mode"] == "full_sequence_target_position"
    assert metadata["full_sequence_token_count"] == 4
    assert metadata["input_token_ids_sha256"] == cache["token_ids_sha256"]
    assert len(metadata["prefix_token_ids_sha256"]) == 64


def test_session_reuse_request_falls_back_to_per_token_metadata() -> None:
    spec = {
        "graph_knobs": {"trajectory_session_mode": "experimental_reuse"},
    }

    metadata = session_reuse_metadata(cast(Any, spec))
    expected = {
        "trajectory_session_mode": "experimental_reuse",
        "session_reuse_requested": True,
        "session_reuse_effective": False,
        "session_reuse_fallback": "per_token_path",
    }
    for key, value in expected.items():
        assert metadata[key] == value
    assert metadata["reuse_phase0_window_state_requested"] is False
    assert metadata["reuse_target_logits_requested"] is False


def test_prefix_view_metadata_full_sequence_mode(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    trajectory, specs, _shard = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
    )
    spec = specs[0]
    spec["graph_knobs"]["input_context_mode"] = "full_sequence"
    spec["graph_knobs"]["phase1_trace_batch_policy"] = "cap_effective_batches"
    spec["graph_knobs"]["phase1_trace_batch_size_max"] = 16
    prefix = reconstruct_prefix_token_ids(trajectory, spec)

    metadata = prefix_view_metadata(trajectory, spec, prefix)

    assert metadata["mode"] == "full_sequence_target_position"
    assert metadata["output_position"] == spec["target_position"] - 1
    assert metadata["input_token_count"] == 4
    assert metadata["full_sequence_token_count"] == 4
    assert len(metadata["input_token_ids_sha256"]) == 64
    assert reconstruct_full_sequence_token_ids(trajectory) == [101, 102, 201, 202]


def test_real_shard_forwards_prefix_view_metadata_without_model_load(
    tmp_path: Path, monkeypatch
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    specs_rows = [
        json.loads(line)
        for line in specs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for spec_row in specs_rows:
        spec_row["graph_knobs"]["phase1_trace_batch_policy"] = "cap_effective_batches"
        spec_row["graph_knobs"]["phase1_trace_batch_size_max"] = 16
    specs_path.write_text(
        "\n".join(json.dumps(row) for row in specs_rows) + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_trace_one(request):
        captured["request"] = request
        output = {
            "telemetry_events": [
                {
                    "event_type": "ranker_frontier",
                    "attrs": {
                        "ranker_frontier_cutoff_gap": 0.125,
                        "ranker_frontier_relative_cutoff_gap": 0.25,
                        "ranker_frontier_near_cutoff_count": 7,
                        "ranker_frontier_max_feature_nodes_cap_bound": True,
                    },
                }
            ],
            "phase3_frontier_buffer_metadata": {
                "status": "expanded",
                "extra_feature_count": 2,
            },
            "phase4_frontier_buffer_metadata": {
                "extra_feature_count_total": 3,
            },
            "feature_semantic_descriptors": {
                "status": "captured",
                "candidate_features": [[0, 0, 1]],
            },
        }
        return types.SimpleNamespace(output=output, telemetry_summary={})

    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            tensor=lambda data, dtype=None: data,
            long=object(),
            Tensor=type("FakeTensor", (), {}),
        ),
    )
    monkeypatch.setattr(provider, "load_model", lambda **_kwargs: object())
    _stub_compact_packager(monkeypatch)
    import circuit_tracer

    monkeypatch.setattr(circuit_tracer, "trace_one", fake_trace_one)

    result = run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    assert result["status"] == "complete"
    request = captured["request"]
    evidence = cast(dict[str, Any], request.evidence.metadata["prefix_view_metadata"])
    assert request.execution.session.phase1_trace_batch_policy == "cap_effective_batches"
    assert request.execution.session.phase1_trace_batch_size_max == 16
    assert evidence["trace_id"] == "traj_runner_tok000001"
    assert "target_position" not in evidence
    assert evidence["prefix_token_count"] == 3
    assert evidence["target_token_ids"] == [202]
    trace = json.loads(
        (
            tmp_path / "run" / "shards" / "shard_000" / "token_000001" / "trace.json"
        ).read_text(encoding="utf-8")
    )
    assert trace["prefix_view_metadata"]["mode"] == "independent_prefix"
    assert trace["prefix_view_metadata"]["target_position"] == 3
    assert {
        key: value
        for key, value in trace["prefix_view_metadata"].items()
        if key not in {"mode", "target_position"}
    } == evidence
    assert trace["phase3_frontier_buffer_metadata"] == {
        "status": "expanded",
        "extra_feature_count": 2,
    }
    assert trace["phase4_frontier_buffer_metadata"] == {
        "extra_feature_count_total": 3,
    }
    assert trace["telemetry_event_count"] == 1
    telemetry_path = Path(trace["telemetry_events_path"])
    assert telemetry_path == (
        tmp_path / "run" / "shards" / "shard_000" / "token_000001" / "telemetry.jsonl"
    )
    telemetry_rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    assert telemetry_rows[0]["trace_id"] == "traj_runner_tok000001"
    assert telemetry_rows[0]["generated_index"] == 1
    assert telemetry_rows[0]["target_token_id"] == 202
    assert telemetry_rows[0]["event"]["attrs"] == {
        "ranker_frontier_cutoff_gap": 0.125,
        "ranker_frontier_relative_cutoff_gap": 0.25,
        "ranker_frontier_near_cutoff_count": 7,
        "ranker_frontier_max_feature_nodes_cap_bound": True,
    }
    assert trace["transcoder"]["requested"]["transcoder_architecture"] == "clt"
    assert (
        trace["transcoder"]["requested"]["transcoder_provider_family"]
        == "gemmascope2-clt-1b-medium-affine"
    )
    assert (
        tmp_path
        / "run"
        / "shards"
        / "shard_000"
        / "token_000001"
        / "feature_semantic_descriptors.npz"
    ).exists()


def test_real_shard_persists_exception_attached_telemetry(
    tmp_path: Path, monkeypatch
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)

    def fake_trace_one(_request):
        exc = RuntimeError("synthetic attribution failure")
        setattr(
            exc,
            "circuit_tracer_telemetry_summary",
            {"event_count": 1, "stored_event_count": 1, "dropped_event_count": 0},
        )
        setattr(
            exc,
            "circuit_tracer_telemetry_events",
            [
                {
                    "scope": "phase",
                    "name": "phase1.forward",
                    "phase": "phase1",
                    "attrs": {"active_features": 227051},
                }
            ],
        )
        raise exc

    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            tensor=lambda data, dtype=None: data,
            long=object(),
            Tensor=type("FakeTensor", (), {}),
        ),
    )
    monkeypatch.setattr(provider, "load_model", lambda **_kwargs: object())
    _stub_compact_packager(monkeypatch)
    import circuit_tracer

    monkeypatch.setattr(circuit_tracer, "trace_one", fake_trace_one)

    result = run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    assert result["status"] == "error"
    trace_path = (
        tmp_path / "run" / "shards" / "shard_000" / "token_000001" / "trace.json"
    )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["status"] == "error"
    assert trace["telemetry_summary"]["event_count"] == 1
    assert trace["telemetry_event_count"] == 1
    telemetry_path = Path(trace["telemetry_events_path"])
    telemetry_rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    assert telemetry_rows[0]["trace_id"] == "traj_runner_tok000001"
    assert telemetry_rows[0]["event"]["name"] == "phase1.forward"
    assert telemetry_rows[0]["event"]["attrs"] == {"active_features": 227051}


def test_real_shard_persists_probe_without_graph_packaging(
    tmp_path: Path, monkeypatch
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    rows = [json.loads(line) for line in specs_path.read_text().splitlines()]
    rows[1]["graph_knobs"].update(
        {
            "diagnostic_stop_mode": "transition_probe",
            "diagnostic_stop_phase4_batches": 2,
        }
    )
    specs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def fake_trace_one(_request):
        return types.SimpleNamespace(
            output=None,
            status="probe_completed",
            semantic_fingerprint="semantic-probe",
            execution_fingerprint="execution-probe",
            telemetry_summary={
                "diagnostic_stop_mode": "transition_probe",
                "phase4_batches_completed": 2,
            },
            telemetry_events=(
                {
                    "scope": "run",
                    "name": "attribute.probe_completed",
                    "attrs": {"status": "probe_completed"},
                },
            ),
        )

    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            tensor=lambda data, dtype=None: data,
            long=object(),
            Tensor=type("FakeTensor", (), {}),
        ),
    )
    monkeypatch.setattr(provider, "load_model", lambda **_kwargs: object())
    import circuit_tracer

    monkeypatch.setattr(circuit_tracer, "trace_one", fake_trace_one)

    result = run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    assert result["status"] == "probe_completed"
    token_dir = tmp_path / "run" / "shards" / "shard_000" / "token_000001"
    trace = json.loads((token_dir / "trace.json").read_text())
    assert trace["status"] == "probe_completed"
    assert trace["graph_path"] is None
    assert trace["diagnostic_stop_mode"] == "transition_probe"
    assert trace["phase4_batches_completed"] == 2
    assert not (token_dir / "graph.npz").exists()
    shard = json.loads(
        (tmp_path / "run" / "shards" / "shard_000" / "shard.json").read_text()
    )
    assert shard["status"] == "probe_completed"
    assert shard["shard_health"]["failed_token_count"] == 0
    assert shard["shard_health"]["diagnostic_token_count"] == 1
    assert shard["shard_health"]["retry_recommended"] is False


def test_real_shard_forwards_full_sequence_prompt_and_output_position(
    tmp_path: Path, monkeypatch
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    rows = [json.loads(line) for line in specs_path.read_text().splitlines()]
    rows[1]["graph_knobs"]["input_context_mode"] = "full_sequence"
    specs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    captured: dict[str, object] = {}

    def fake_trace_one(request):
        captured["request"] = request
        return types.SimpleNamespace(output={}, telemetry_summary={})

    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            tensor=lambda data, dtype=None: data,
            long=object(),
            Tensor=type("FakeTensor", (), {}),
        ),
    )
    monkeypatch.setattr(provider, "load_model", lambda **_kwargs: object())
    _stub_compact_packager(monkeypatch)
    import circuit_tracer

    monkeypatch.setattr(circuit_tracer, "trace_one", fake_trace_one)

    run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    request = captured["request"]
    assert request.problem.prompt == [101, 102, 201, 202]
    assert request.problem.output_position == 2
    assert request.problem.prefix_view.mode == "full_sequence_target_position"
    assert request.problem.prefix_view.target_position == 3
    evidence = cast(dict[str, Any], request.evidence.metadata["prefix_view_metadata"])
    assert "mode" not in evidence
    assert "target_position" not in evidence


def test_real_shard_experimental_reuse_falls_back_to_canonical_per_token_trace(
    tmp_path: Path, monkeypatch
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    rows = [json.loads(line) for line in specs_path.read_text().splitlines()]
    rows[1]["graph_knobs"]["input_context_mode"] = "full_sequence"
    rows[1]["graph_knobs"]["trajectory_session_mode"] = "experimental_reuse"
    specs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    captured: dict[str, object] = {}

    class Cache:
        fingerprint: object | None = None

    class Transcoders:
        def __init__(self) -> None:
            self.created = 0
            self.cleared = 0
            self.cache = Cache()

        def create_decoder_block_cache(self, *, fingerprint=None):
            self.created += 1
            self.cache.fingerprint = fingerprint
            return self.cache

        def clear_decoder_block_cache(self, cache) -> None:
            assert cache is self.cache
            self.cleared += 1

    transcoders = Transcoders()
    model = types.SimpleNamespace(transcoders=transcoders, device="cpu")

    def fake_trace_one(request):
        captured["request"] = request
        return types.SimpleNamespace(output={}, telemetry_summary={})

    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            tensor=lambda data, dtype=None: data,
            long=object(),
            Tensor=type("FakeTensor", (), {}),
        ),
    )
    monkeypatch.setattr(provider, "load_model", lambda **_kwargs: model)
    _stub_compact_packager(monkeypatch)
    import circuit_tracer

    monkeypatch.setattr(circuit_tracer, "trace_one", fake_trace_one)

    run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    request = captured["request"]
    assert request.problem.prompt == [101, 102, 201, 202]
    assert transcoders.created == 0
    assert transcoders.cleared == 0
    trace = json.loads(
        (
            tmp_path / "run" / "shards" / "shard_000" / "token_000001" / "trace.json"
        ).read_text(encoding="utf-8")
    )
    assert trace["trajectory_session"]["session_reuse_effective"] is False
    assert trace["trajectory_session"]["session_reuse_fallback"] == "per_token_path"
    assert trace["trajectory_session"]["decoder_cache_reuse_effective"] is False


def test_real_shard_canonical_per_token_requests_preserve_each_spec(
    tmp_path: Path, monkeypatch
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    rows = [json.loads(line) for line in specs_path.read_text().splitlines()]
    rows[0]["graph_knobs"]["input_context_mode"] = "full_sequence"
    rows[1]["graph_knobs"]["input_context_mode"] = "full_sequence"
    rows[1]["graph_knobs"]["trajectory_session_mode"] = "experimental_reuse"
    specs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    shards = json.loads(shards_path.read_text(encoding="utf-8"))
    shards["shards"] = [
        {"shard_id": 0, "estimated_cost_sum": 5, "spec_indices": [0, 1]}
    ]
    shards_path.write_text(json.dumps(shards), encoding="utf-8")
    calls: list[Any] = []

    class Cache:
        fingerprint: object | None = None

    class Transcoders:
        def __init__(self) -> None:
            self.cache = Cache()

        def create_decoder_block_cache(self, *, fingerprint=None):
            self.cache.fingerprint = fingerprint
            return self.cache

        def clear_decoder_block_cache(self, cache) -> None:
            assert cache is self.cache

    transcoders = Transcoders()
    model = types.SimpleNamespace(transcoders=transcoders, device="cpu")

    def fake_trace_one(request):
        calls.append(request)
        return types.SimpleNamespace(output={}, telemetry_summary={})

    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(tensor=lambda data, dtype=None: data, long=object()),
    )
    monkeypatch.setattr(provider, "load_model", lambda **_kwargs: model)
    _stub_compact_packager(monkeypatch)
    import circuit_tracer

    monkeypatch.setattr(circuit_tracer, "trace_one", fake_trace_one)

    run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    assert len(calls) == 2
    assert calls[0].problem.targets == [201]
    assert calls[1].problem.targets == [202]
    assert calls[0].problem.output_position == 1
    assert calls[1].problem.output_position == 2


def test_real_shard_window_reuse_uses_window_session(
    tmp_path: Path, monkeypatch
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    rows = [json.loads(line) for line in specs_path.read_text().splitlines()]
    for row in rows:
        row["graph_knobs"].update(
            {
                "input_context_mode": "full_sequence",
                "trajectory_session_mode": "window_reuse_v1",
                "reuse_phase0_window_state": True,
                "reuse_target_logits": True,
                "phase0_window_scope": "shard_window",
                "phase0_window_max_prefix_policy": "max_target_position",
                "phase0_window_reference_checks": "off",
            }
        )
    specs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    shards_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trace_specs_file": "trace_specs.jsonl",
                "cost_model": "contiguous_window_lpt_v1",
                "shards": [
                    {
                        "shard_id": 0,
                        "estimated_cost_sum": 5,
                        "spec_indices": [0, 1],
                        "windows": [
                            {
                                "window_start_generated_index": 0,
                                "window_end_generated_index": 1,
                                "window_max_target_position": 3,
                                "target_positions": [2, 3],
                                "spec_indices": [0, 1],
                                "estimated_cost_sum": 5,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[int] = []
    session_inits: list[dict[str, object]] = []

    class FakeSession:
        def trace_window(self, target_position, *, reuse, request):
            calls.append(int(target_position))
            assert reuse is True
            assert request.problem.output_position == int(target_position) - 1
            output = {
                "phase0_window_state_reuse_effective": True,
                "target_logit_source": "full_sequence_window_logits",
            }
            return types.SimpleNamespace(output=output, telemetry_summary={})

        def close(self) -> None:
            calls.append(-1)

    def fake_open_session(request, *, window):
        session_inits.append({"request": request, "window": window})
        return FakeSession()

    class Transcoders:
        def create_decoder_block_cache(self, *, fingerprint=None):
            return types.SimpleNamespace(fingerprint=fingerprint)

        def clear_decoder_block_cache(self, _cache) -> None:
            pass

    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    fake_torch = types.SimpleNamespace(
        tensor=lambda data, dtype=None: data,
        long=object(),
        Tensor=type("FakeTensor", (), {}),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    monkeypatch.setattr(
        provider,
        "load_model",
        lambda **_kwargs: types.SimpleNamespace(
            transcoders=Transcoders(), device="cpu"
        ),
    )
    _stub_compact_packager(monkeypatch)
    import circuit_tracer

    monkeypatch.setattr(circuit_tracer, "open_session", fake_open_session)

    result = run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    assert result["status"] == "complete"
    assert calls == [2, 3, -1]
    assert session_inits[0]["window"].max_prefix_len == 3
    assert session_inits[0]["request"].problem.prompt == [101, 102, 201, 202]
    trace = json.loads(
        (
            tmp_path / "run" / "shards" / "shard_000" / "token_000001" / "trace.json"
        ).read_text(encoding="utf-8")
    )
    assert trace["trajectory_session"]["trajectory_session_mode"] == "window_reuse_v1"
    assert trace["trajectory_session"]["reuse_phase0_window_state_effective"] is True
    assert trace["trajectory_session"]["reuse_target_logits_effective"] is True


def test_aggregate_handles_dry_run_rows(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    run_root = tmp_path / "run"
    dry_run_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=run_root,
    )
    aggregate = aggregate_shards(run_root)
    assert aggregate["status_counts"] == {"dry_run": 1}
    assert (run_root / "aggregate.json").exists()
    assert (run_root / "per_token_metrics.jsonl").exists()
    assert "target_token_id" in (run_root / "per_token_metrics.csv").read_text(
        encoding="utf-8"
    )
    assert "target_position" in (run_root / "per_token_metrics.csv").read_text(
        encoding="utf-8"
    )


def test_shard_input_validation_rejects_target_position_mismatch(
    tmp_path: Path,
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    rows = [
        json.loads(line) for line in specs_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["target_position"] = 99
    rows[1]["prefix_token_count"] = 99
    specs_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    try:
        load_shard_inputs(
            trajectory_path=trajectory_path,
            trace_specs_path=specs_path,
            shards_path=shards_path,
            shard_id=0,
        )
    except ValueError as exc:
        assert "target_position" in str(exc)
    else:
        raise AssertionError("expected target_position mismatch to fail")


def test_shard_inputs_normalize_old_specs_without_target_position(
    tmp_path: Path,
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    rows = [
        json.loads(line) for line in specs_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[1].pop("target_position")
    specs_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    _trajectory, specs, _shard = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
    )
    assert specs[0]["target_position"] == 3


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        "MPLCONFIGDIR": "/tmp/nlp-research-project-matplotlib",
        "PYTHONPATH": str(SRC_ROOT),
    }
    return subprocess.run(
        [sys.executable, "-m", "nlp_research_project.exact_trace_bench", *args],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_cli_dry_run_and_aggregate(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    run_root = tmp_path / "run"
    proc = _run_cli(
        "run-full-answer-shard",
        "--trajectory",
        str(trajectory_path),
        "--trace-specs",
        str(specs_path),
        "--shards",
        str(shards_path),
        "--shard-id",
        "0",
        "--output-root",
        str(run_root),
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr
    proc = _run_cli("aggregate-full-answer-shards", "--run-root", str(run_root))
    assert proc.returncode == 0, proc.stderr
    assert (
        json.loads((run_root / "aggregate.json").read_text(encoding="utf-8"))[
            "token_count"
        ]
        == 1
    )


def test_cli_list_does_not_require_output_root(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    proc = _run_cli(
        "run-full-answer-shard",
        "--trajectory",
        str(trajectory_path),
        "--trace-specs",
        str(specs_path),
        "--shards",
        str(shards_path),
        "--shard-id",
        "1",
        "--list",
    )
    assert proc.returncode == 0, proc.stderr
    assert "generated_index=0" in proc.stdout
    assert "target_position=2" in proc.stdout


def test_cli_dry_run_requires_output_root(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    proc = _run_cli(
        "run-full-answer-shard",
        "--trajectory",
        str(trajectory_path),
        "--trace-specs",
        str(specs_path),
        "--shards",
        str(shards_path),
        "--shard-id",
        "0",
        "--dry-run",
    )
    assert proc.returncode != 0
    assert "--output-root" in proc.stderr


def test_cli_real_shard_exits_nonzero_on_error_status(
    tmp_path: Path, monkeypatch
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    monkeypatch.setattr(
        "nlp_research_project.exact_trace_bench.full_answer.runner.run_real_shard",
        lambda **kwargs: {"status": "error", "shard_dir": "x"},
    )
    args = argparse.Namespace(
        trajectory=trajectory_path,
        trace_specs=specs_path,
        shards=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
        list=False,
        dry_run=False,
        run_id=None,
        run_name=None,
        run_description=None,
        run_goal=None,
    )
    try:
        full_answer_cli._cmd_run_full_answer_shard(args)
    except RuntimeError as exc:
        assert "full-answer shard failed" in str(exc)
    else:
        raise AssertionError("expected non-ok shard status to fail the CLI")


def test_cli_real_shard_accepts_terminal_probe_status(
    tmp_path: Path, monkeypatch
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    monkeypatch.setattr(
        "nlp_research_project.exact_trace_bench.full_answer.runner.run_real_shard",
        lambda **kwargs: {"status": "probe_completed", "shard_dir": "x"},
    )
    args = argparse.Namespace(
        trajectory=trajectory_path,
        trace_specs=specs_path,
        shards=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
        list=False,
        dry_run=False,
        run_id=None,
        run_name=None,
        run_description=None,
        run_goal=None,
    )
    full_answer_cli._cmd_run_full_answer_shard(args)


def test_real_shard_requires_slurm_before_heavy_imports(
    tmp_path: Path, monkeypatch
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("EXACT_TRACE_ALLOW_LOCAL_GPU", raising=False)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {
            "torch",
            "nlp_research_project.exact_trace_bench.trace_runtime.provider",
            "nlp_research_project.exact_trace_bench.compact_io",
            "circuit_tracer",
        }:
            raise AssertionError(f"heavy import attempted: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    try:
        from nlp_research_project.exact_trace_bench.full_answer.runner import (
            run_real_shard,
        )

        run_real_shard(
            trajectory_path=trajectory_path,
            trace_specs_path=specs_path,
            shards_path=shards_path,
            shard_id=0,
            output_root=tmp_path / "run",
        )
    except RuntimeError as exc:
        assert "SLURM_JOB_ID" in str(exc)
    else:
        raise AssertionError("expected SLURM guard to fail before heavy imports")


def test_prefix_reconstruction_and_forced_target_payload(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    trajectory, specs, _ = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
    )
    assert reconstruct_prefix_token_ids(trajectory, specs[0]) == [101, 102, 201]
    assert forced_target_payload(specs[0]) == {
        "token_id": 202,
        "token_text": "7",
        "target_mode": "frozen_target_only",
        "attribution_targets": [202],
    }
