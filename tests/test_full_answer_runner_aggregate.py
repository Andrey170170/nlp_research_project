from __future__ import annotations

import json
import builtins
import argparse
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, cast

from nlp_research_project.exact_trace_bench.full_answer.aggregate import (
    aggregate_shards,
)
from nlp_research_project.exact_trace_bench import cli as full_answer_cli
from nlp_research_project.exact_trace_bench.full_answer.runner import (
    _attribute_performance_kwargs,
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


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


def test_attribute_performance_kwargs_forward_safe_knobs() -> None:
    kwargs = _attribute_performance_kwargs(
        {
            "row_subchunk_size": 128,
            "plan_feature_batch_size": True,
            "feature_batch_size_max": 512,
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
            "exact_encoder_residency": "active_pinned_cpu",
        }
    )
    assert kwargs == {
        "row_subchunk_size": 128,
        "plan_feature_batch_size": True,
        "feature_batch_size_max": 512,
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
        "chunked_feature_replay_window": 16,
        "error_vector_prefetch_lookahead": 8,
        "stage_encoder_vecs_on_cpu": False,
        "stage_error_vectors_on_cpu": False,
        "exact_encoder_residency": "active_pinned_cpu",
        "cross_cluster_debug": True,
        "capture_phase0_donor_bundle": True,
        "capture_phase3_seed_bundle": True,
        "capture_feature_semantic_descriptors": True,
        "semantic_descriptor_top_k": 1024,
        "semantic_descriptor_dim": 32,
    }


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
    captured: dict[str, object] = {}

    def fake_attribute(**kwargs):
        captured.update(kwargs)
        return {
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

    def fake_save_compact(step, graph_path):
        Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
        Path(graph_path).write_text("graph")

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
    monkeypatch.setitem(
        sys.modules,
        "circuit_utils",
        types.SimpleNamespace(
            save_bucketed_compact=lambda _bundle, graph_path: fake_save_compact(
                {}, graph_path
            ),
            save_compact=fake_save_compact,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trace_pipeline",
        types.SimpleNamespace(load_model=lambda **_kwargs: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "trace_pipeline_chunked",
        types.SimpleNamespace(
            compact_result_to_bucketed_compact=lambda *_args, **_kwargs: (
                types.SimpleNamespace(step={})
            ),
            resolve_internal_precision=lambda _dtype: "float32",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "circuit_tracer.attribution.attribute_nnsight",
        types.SimpleNamespace(attribute=fake_attribute),
    )

    result = run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    assert result["status"] == "complete"
    metadata = cast(dict[str, Any], captured["prefix_view_metadata"])
    assert metadata["trace_id"] == "traj_runner_tok000001"
    assert metadata["target_position"] == 3
    assert metadata["prefix_token_count"] == 3
    assert metadata["target_token_ids"] == [202]
    trace = json.loads(
        (
            tmp_path / "run" / "shards" / "shard_000" / "token_000001" / "trace.json"
        ).read_text(encoding="utf-8")
    )
    assert trace["prefix_view_metadata"] == metadata
    assert trace["phase3_frontier_buffer_metadata"] == {
        "status": "expanded",
        "extra_feature_count": 2,
    }
    assert trace["phase4_frontier_buffer_metadata"] == {
        "extra_feature_count_total": 3,
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


def test_real_shard_forwards_full_sequence_prompt_and_output_position(
    tmp_path: Path, monkeypatch
) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    rows = [json.loads(line) for line in specs_path.read_text().splitlines()]
    rows[1]["graph_knobs"]["input_context_mode"] = "full_sequence"
    specs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    captured: dict[str, object] = {}

    def fake_attribute(**kwargs):
        captured.update(kwargs)
        return {}

    def fake_save_compact(step, graph_path):
        Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
        Path(graph_path).write_text("graph")

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
    monkeypatch.setitem(
        sys.modules,
        "circuit_utils",
        types.SimpleNamespace(
            save_bucketed_compact=lambda _bundle, graph_path: fake_save_compact(
                {}, graph_path
            ),
            save_compact=fake_save_compact,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trace_pipeline",
        types.SimpleNamespace(load_model=lambda **_kwargs: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "trace_pipeline_chunked",
        types.SimpleNamespace(
            compact_result_to_bucketed_compact=lambda *_args, **_kwargs: (
                types.SimpleNamespace(step={})
            ),
            resolve_internal_precision=lambda _dtype: "float32",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "circuit_tracer.attribution.attribute_nnsight",
        types.SimpleNamespace(attribute=fake_attribute),
    )

    run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    assert captured["prompt"] == [101, 102, 201, 202]
    assert captured["output_position"] == 2
    metadata = cast(dict[str, Any], captured["prefix_view_metadata"])
    assert metadata["mode"] == "full_sequence_target_position"


def test_real_shard_experimental_reuse_passes_shared_decoder_cache(
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

    def fake_attribute(**kwargs):
        captured.update(kwargs)
        return {}

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
    monkeypatch.setitem(
        sys.modules,
        "circuit_utils",
        types.SimpleNamespace(
            save_bucketed_compact=lambda _bundle, graph_path: (
                Path(graph_path).parent.mkdir(parents=True, exist_ok=True),
                Path(graph_path).write_text("graph"),
            ),
            save_compact=lambda _step, graph_path: Path(graph_path).write_text("graph"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trace_pipeline",
        types.SimpleNamespace(load_model=lambda **_kwargs: model),
    )
    monkeypatch.setitem(
        sys.modules,
        "trace_pipeline_chunked",
        types.SimpleNamespace(
            compact_result_to_bucketed_compact=lambda *_args, **_kwargs: (
                types.SimpleNamespace(step={})
            ),
            resolve_internal_precision=lambda _dtype: "float32",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "circuit_tracer.attribution.attribute_nnsight",
        types.SimpleNamespace(attribute=fake_attribute),
    )

    run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    assert captured["decoder_chunk_cache"] is transcoders.cache
    assert captured["decoder_cache_fingerprint"] == transcoders.cache.fingerprint
    assert transcoders.created == 1
    assert transcoders.cleared == 1
    trace = json.loads(
        (
            tmp_path / "run" / "shards" / "shard_000" / "token_000001" / "trace.json"
        ).read_text(encoding="utf-8")
    )
    assert trace["trajectory_session"]["session_reuse_effective"] is True
    assert trace["trajectory_session"]["decoder_cache_reuse_effective"] is True


def test_real_shard_decoder_cache_reuse_is_per_spec_opt_in(
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
    calls: list[dict[str, object]] = []

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

    def fake_attribute(**kwargs):
        calls.append(dict(kwargs))
        return {}

    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(tensor=lambda data, dtype=None: data, long=object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "circuit_utils",
        types.SimpleNamespace(
            save_bucketed_compact=lambda _bundle, graph_path: (
                Path(graph_path).parent.mkdir(parents=True, exist_ok=True),
                Path(graph_path).write_text("graph"),
            ),
            save_compact=lambda _step, graph_path: Path(graph_path).write_text("graph"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trace_pipeline",
        types.SimpleNamespace(load_model=lambda **_kwargs: model),
    )
    monkeypatch.setitem(
        sys.modules,
        "trace_pipeline_chunked",
        types.SimpleNamespace(
            compact_result_to_bucketed_compact=lambda *_args, **_kwargs: (
                types.SimpleNamespace(step={})
            ),
            resolve_internal_precision=lambda _dtype: "float32",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "circuit_tracer.attribution.attribute_nnsight",
        types.SimpleNamespace(attribute=fake_attribute),
    )

    run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    assert len(calls) == 2
    assert calls[0]["decoder_chunk_cache"] is None
    assert calls[1]["decoder_chunk_cache"] is transcoders.cache


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
        def __init__(self, **kwargs) -> None:
            session_inits.append(dict(kwargs))

        def attribute_target_position(self, target_position, **kwargs):
            calls.append(int(target_position))
            return {
                "phase0_window_state_reuse_effective": True,
                "target_logit_source": "full_sequence_window_logits",
            }

        def cleanup(self) -> None:
            calls.append(-1)

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

    def fake_save_bucketed(_bundle, graph_path) -> None:
        Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
        Path(graph_path).write_text("graph")

    monkeypatch.setitem(
        sys.modules,
        "circuit_utils",
        types.SimpleNamespace(
            save_bucketed_compact=fake_save_bucketed,
            save_compact=lambda _step, graph_path: Path(graph_path).write_text("graph"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trace_pipeline",
        types.SimpleNamespace(
            load_model=lambda **_kwargs: types.SimpleNamespace(
                transcoders=Transcoders(), device="cpu"
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trace_pipeline_chunked",
        types.SimpleNamespace(
            compact_result_to_bucketed_compact=lambda *_args, **_kwargs: (
                types.SimpleNamespace(step={})
            ),
            resolve_internal_precision=lambda _dtype: "float32",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "circuit_tracer.attribution.attribute_nnsight",
        types.SimpleNamespace(
            attribute=lambda **_kwargs: {},
            FullSequenceWindowAttributionSession=FakeSession,
        ),
    )

    result = run_real_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )

    assert result["status"] == "complete"
    assert calls == [2, 3, -1]
    assert session_inits[0]["window_max_prefix_len"] == 3
    assert session_inits[0]["reuse_phase0_window_state"] is True
    assert session_inits[0]["reuse_target_logits"] is True
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
    env = {"PYTHONPATH": str(SRC_ROOT)}
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
            "trace_pipeline",
            "circuit_utils",
            "trace_pipeline_chunked",
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
