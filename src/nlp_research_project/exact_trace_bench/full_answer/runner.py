from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, cast

from ..io_utils import ensure_dir, write_json, write_jsonl
from ..transcoder_config import (
    PUBLIC_TRANSCODER_KNOB_KEYS,
    resolve_transcoder_load_config,
    transcoder_config_to_json,
)
from .schemas import TraceSpec, load_shards, load_trace_specs, load_trajectory


def _shard_dir(output_root: Path, shard_id: int) -> Path:
    return output_root / "shards" / f"shard_{shard_id:03d}"


def select_shard(shards: Mapping[str, Any], shard_id: int) -> dict[str, Any]:
    shard_rows = shards.get("shards")
    if not isinstance(shard_rows, list):
        raise ValueError("shards payload missing shards list")
    for shard in shard_rows:
        if isinstance(shard, dict) and shard.get("shard_id") == shard_id:
            return cast(dict[str, Any], shard)
    raise ValueError(f"shard_id not found: {shard_id}")


def assigned_specs(specs: list[TraceSpec], shard: Mapping[str, Any]) -> list[TraceSpec]:
    selected: list[TraceSpec] = []
    for index in shard.get("spec_indices", []):
        if not isinstance(index, int) or index < 0 or index >= len(specs):
            raise ValueError(f"spec index out of bounds for shard: {index}")
        selected.append(specs[index])
    return selected


def load_shard_inputs(
    *, trajectory_path: Path, trace_specs_path: Path, shards_path: Path, shard_id: int
) -> tuple[dict[str, Any], list[TraceSpec], dict[str, Any]]:
    trajectory = load_trajectory(trajectory_path)
    specs = load_trace_specs(trace_specs_path)
    shards = load_shards(shards_path)
    shard = select_shard(shards, shard_id)
    selected = assigned_specs(specs, shard)
    trajectory_id = trajectory["trajectory_id"]
    for spec in selected:
        if spec["trajectory_id"] != trajectory_id:
            raise ValueError("trace spec trajectory_id does not match trajectory")
        validate_trace_spec_against_trajectory(trajectory, spec)
    return cast(dict[str, Any], trajectory), selected, shard


def validate_trace_spec_against_trajectory(
    trajectory: Mapping[str, Any], spec: TraceSpec
) -> None:
    """Fail before attribution if position metadata diverges from the prefix."""
    prompt_token_count = trajectory.get("prompt_token_count")
    generated_tokens = trajectory.get("generated_tokens")
    if not isinstance(prompt_token_count, int) or prompt_token_count < 0:
        raise ValueError("trajectory prompt_token_count must be a non-negative int")
    if not isinstance(generated_tokens, list):
        raise ValueError("trajectory generated_tokens must be a list")
    generated_index = spec["generated_index"]
    if generated_index < 0 or generated_index >= len(generated_tokens):
        raise ValueError(f"generated_index out of bounds: {generated_index}")
    expected_position = prompt_token_count + generated_index
    if spec["target_position"] != expected_position:
        raise ValueError(
            "trace spec target_position does not match "
            "prompt_token_count + generated_index "
            f"({spec['target_position']} != {expected_position})"
        )
    token = generated_tokens[generated_index]
    if not isinstance(token, Mapping):
        raise ValueError("trajectory generated token row must be an object")
    if token.get("absolute_token_position") != expected_position:
        raise ValueError(
            "trajectory generated token absolute_token_position does not match "
            "prompt_token_count + generated_index"
        )


def list_shard_specs(
    *, trajectory_path: Path, trace_specs_path: Path, shards_path: Path, shard_id: int
) -> list[dict[str, Any]]:
    _, specs, _ = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        shard_id=shard_id,
    )
    return [_summary_row(spec, shard_id=shard_id) for spec in specs]


def print_shard_specs(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"shard={row['shard_id']} generated_index={row['generated_index']} "
            f"target_position={row['target_position']} trace_id={row['trace_id']} "
            f"target={row['target_token_id']} "
            f"text={row['target_token_text']!r} cost={row['estimated_cost']}"
        )


def dry_run_shard(
    *,
    trajectory_path: Path,
    trace_specs_path: Path,
    shards_path: Path,
    shard_id: int,
    output_root: Path,
) -> dict[str, Any]:
    trajectory, specs, shard = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        shard_id=shard_id,
    )
    root = _shard_dir(output_root, shard_id)
    ensure_dir(root)
    shard_payload = {
        "schema_version": 1,
        "status": "dry_run",
        "shard": shard,
        "trajectory_id": trajectory["trajectory_id"],
        "trace_specs_file": str(trace_specs_path),
        "shards_file": str(shards_path),
        "token_count": len(specs),
        "target_positions": [spec["target_position"] for spec in specs],
    }
    write_json(root / "shard.json", shard_payload)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        trace = _trace_payload(spec, shard_id=shard_id)
        token_dir = root / f"token_{spec['generated_index']:06d}"
        write_json(token_dir / "trace.json", trace)
        rows.append(trace)
    write_jsonl(root / "trace_results.jsonl", rows)
    return {"shard_dir": str(root), "token_count": len(rows), "status": "dry_run"}


def reconstruct_prefix_token_ids(
    trajectory: Mapping[str, Any], spec: TraceSpec
) -> list[int]:
    """Return frozen prompt + generated prefix immediately before target token."""
    generated_index = spec["generated_index"]
    prompt_token_ids = trajectory.get("prompt_token_ids")
    generated_tokens = trajectory.get("generated_tokens")
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token_id, int) for token_id in prompt_token_ids
    ):
        raise ValueError("trajectory prompt_token_ids must be a list of ints")
    if not isinstance(generated_tokens, list):
        raise ValueError("trajectory generated_tokens must be a list")
    if generated_index < 0 or generated_index >= len(generated_tokens):
        raise ValueError(f"generated_index out of bounds: {generated_index}")
    prefix_generated = [
        int(token["token_id"])
        for token in generated_tokens[:generated_index]
        if isinstance(token, Mapping) and isinstance(token.get("token_id"), int)
    ]
    if len(prefix_generated) != generated_index:
        raise ValueError("generated prefix contains malformed token rows")
    prefix = [int(token_id) for token_id in prompt_token_ids] + prefix_generated
    if spec["prefix_token_count"] != len(prefix):
        raise ValueError(
            "trace spec prefix_token_count does not match reconstructed prefix "
            f"({spec['prefix_token_count']} != {len(prefix)})"
        )
    if spec["target_position"] != len(prefix):
        raise ValueError(
            "trace spec target_position does not match reconstructed target "
            f"position ({spec['target_position']} != {len(prefix)})"
        )
    return prefix


def reconstruct_full_sequence_token_ids(trajectory: Mapping[str, Any]) -> list[int]:
    prompt_token_ids = trajectory.get("prompt_token_ids")
    generated_tokens = trajectory.get("generated_tokens")
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token_id, int) for token_id in prompt_token_ids
    ):
        raise ValueError("trajectory prompt_token_ids must be a list of ints")
    if not isinstance(generated_tokens, list):
        raise ValueError("trajectory generated_tokens must be a list")
    generated = [
        int(token["token_id"])
        for token in generated_tokens
        if isinstance(token, Mapping) and isinstance(token.get("token_id"), int)
    ]
    if len(generated) != len(generated_tokens):
        raise ValueError("generated sequence contains malformed token rows")
    return [int(token_id) for token_id in prompt_token_ids] + generated


def _hash_token_ids(token_ids: list[int]) -> str:
    payload = json.dumps(
        [int(token_id) for token_id in token_ids], separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prefix_view_metadata(
    trajectory: Mapping[str, Any],
    spec: TraceSpec,
    prefix_token_ids: list[int],
    *,
    full_sequence_token_ids: list[int] | None = None,
    full_sequence_token_ids_sha256: str | None = None,
) -> dict[str, Any]:
    knobs = spec.get("graph_knobs", {})
    input_context_mode = (
        knobs.get("input_context_mode") if isinstance(knobs, Mapping) else None
    )
    metadata = {
        "schema_version": 1,
        "mode": "full_sequence_target_position"
        if input_context_mode == "full_sequence"
        else "independent_prefix",
        "trajectory_id": trajectory["trajectory_id"],
        "trace_id": spec["trace_id"],
        "target_position": spec["target_position"],
        "prefix_token_count": spec["prefix_token_count"],
        "target_token_ids": [spec["target_token_id"]],
        "prefix_token_ids_sha256": _hash_token_ids(prefix_token_ids),
    }
    if input_context_mode == "full_sequence":
        full_sequence = full_sequence_token_ids or reconstruct_full_sequence_token_ids(
            trajectory
        )
        full_sequence_hash = full_sequence_token_ids_sha256 or _hash_token_ids(
            full_sequence
        )
        metadata["output_position"] = spec["target_position"] - 1
        metadata["input_token_count"] = len(full_sequence)
        metadata["full_sequence_token_count"] = len(full_sequence)
        metadata["input_token_ids_sha256"] = full_sequence_hash
    return metadata


def prepare_full_sequence_cache(
    trajectory: Mapping[str, Any], specs: list[TraceSpec]
) -> dict[str, Any] | None:
    """Reconstruct full-sequence ids/hash once when any shard spec requests it."""
    if not any(
        spec.get("graph_knobs", {}).get("input_context_mode") == "full_sequence"
        for spec in specs
    ):
        return None
    token_ids = reconstruct_full_sequence_token_ids(trajectory)
    return {"token_ids": token_ids, "token_ids_sha256": _hash_token_ids(token_ids)}


def session_reuse_metadata(spec: TraceSpec) -> dict[str, Any]:
    knobs = spec.get("graph_knobs", {})
    mode = knobs.get("trajectory_session_mode", "per_token")
    requested = mode not in (None, "per_token")
    return {
        "trajectory_session_mode": mode,
        "session_reuse_requested": bool(requested),
        "session_reuse_effective": False,
        "session_reuse_fallback": "per_token_path" if requested else None,
        "reuse_phase0_window_state_requested": bool(
            knobs.get("reuse_phase0_window_state", False)
        ),
        "reuse_phase0_window_state_effective": False,
        "reuse_target_logits_requested": bool(knobs.get("reuse_target_logits", False)),
        "reuse_target_logits_effective": False,
        "target_logit_source": "per_target_trace",
    }


def _shard_windows(
    specs: list[TraceSpec], shard: Mapping[str, Any]
) -> list[dict[str, Any]]:
    original_indices = shard.get("spec_indices", [])
    original_to_spec: dict[int, TraceSpec] = {}
    if isinstance(original_indices, list) and len(original_indices) == len(specs):
        for original_index, spec in zip(original_indices, specs):
            if isinstance(original_index, int):
                original_to_spec[original_index] = spec
    windows = shard.get("windows")
    result: list[dict[str, Any]] = []
    if isinstance(windows, list) and original_to_spec:
        for window_id, window in enumerate(windows):
            if not isinstance(window, Mapping):
                continue
            window_mapping = cast(Mapping[str, Any], window)
            window_spec_indices = window_mapping.get("spec_indices")
            if not isinstance(window_spec_indices, list):
                continue
            window_specs = [
                original_to_spec[index]
                for index in window_spec_indices
                if isinstance(index, int) and index in original_to_spec
            ]
            if window_specs:
                payload = dict(window_mapping)
                payload["window_id"] = window_id
                payload["specs"] = window_specs
                result.append(payload)
    if result:
        return result
    if not specs:
        return []
    return [
        {
            "window_id": 0,
            "window_start_generated_index": min(
                int(spec["generated_index"]) for spec in specs
            ),
            "window_end_generated_index": max(
                int(spec["generated_index"]) for spec in specs
            ),
            "window_max_target_position": max(
                int(spec["target_position"]) for spec in specs
            ),
            "target_positions": [int(spec["target_position"]) for spec in specs],
            "specs": specs,
        }
    ]


def _validate_window_session_specs(window_specs: list[TraceSpec]) -> None:
    window_modes = {
        spec.get("graph_knobs", {}).get("trajectory_session_mode", "per_token")
        for spec in window_specs
    }
    if "window_reuse_v1" not in window_modes:
        return
    for spec in window_specs:
        knobs = spec.get("graph_knobs", {})
        if knobs.get("trajectory_session_mode") != "window_reuse_v1":
            continue
        if knobs.get("input_context_mode") != "full_sequence":
            raise ValueError(
                "window_reuse_v1 requires input_context_mode='full_sequence'"
            )
    shared_keys = (
        "reuse_phase0_window_state",
        "reuse_target_logits",
        "phase0_window_scope",
        "phase0_window_max_prefix_policy",
        "phase0_window_reference_checks",
    )
    first_knobs = next(
        spec["graph_knobs"]
        for spec in window_specs
        if spec.get("graph_knobs", {}).get("trajectory_session_mode")
        == "window_reuse_v1"
    )
    for spec in window_specs:
        knobs = spec.get("graph_knobs", {})
        if knobs.get("trajectory_session_mode") != "window_reuse_v1":
            continue
        for key in shared_keys:
            if knobs.get(key) != first_knobs.get(key):
                raise ValueError(f"window_reuse_v1 specs in a window must share {key}")


def _normalize_target_logit_source(source: Any) -> str:
    if source in (None, "context", "override"):
        return "per_target_trace"
    return str(source)


def forced_target_payload(spec: TraceSpec) -> dict[str, Any]:
    return {
        "token_id": spec["target_token_id"],
        "token_text": spec["target_token_text"],
        "target_mode": spec["target_mode"],
        "attribution_targets": [spec["target_token_id"]],
    }


def _runtime_metadata(
    *,
    run_id: str | None,
    run_name: str | None,
    run_description: str | None,
    run_goal: str | None,
) -> dict[str, Any]:
    workspace_root = os.environ.get("WORKSPACE_ROOT")
    library_workspace_root = os.environ.get("LIB_WORKSPACE_ROOT")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    return {
        "run_id": run_id or os.environ.get("RUN_ID") or None,
        "run_name": run_name or os.environ.get("RUN_NAME") or None,
        "run_description": run_description or os.environ.get("RUN_DESCRIPTION") or None,
        "run_goal": run_goal or os.environ.get("RUN_GOAL") or None,
        "workspace_root": workspace_root,
        "library_workspace_root": library_workspace_root,
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
    }


def _shard_record(
    *,
    status: str,
    shard: Mapping[str, Any],
    trajectory_id: str,
    trace_specs_path: Path,
    shards_path: Path,
    token_count: int,
    metadata: Mapping[str, Any],
    target_positions: list[int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "shard": dict(shard),
        "trajectory_id": trajectory_id,
        "trace_specs_file": str(trace_specs_path),
        "shards_file": str(shards_path),
        "token_count": token_count,
        "target_positions": target_positions or [],
    }
    payload.update(metadata)
    return payload


def _trace_runtime_payload(
    *,
    spec: TraceSpec,
    shard_id: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _trace_payload(spec, shard_id=shard_id)
    payload.update(metadata)
    return payload


def _target_token_metadata(
    trajectory: Mapping[str, Any], spec: TraceSpec
) -> dict[str, Any]:
    generated_tokens = trajectory.get("generated_tokens")
    if not isinstance(generated_tokens, list):
        raise ValueError("trajectory generated_tokens must be a list")
    generated_index = spec["generated_index"]
    token = generated_tokens[generated_index]
    if not isinstance(token, Mapping):
        raise ValueError("trajectory generated token row must be an object")
    return {
        "target_logprob": token.get("logprob"),
        "target_probability": token.get("probability"),
        "target_rank": token.get("rank"),
    }


def _graph_summary(
    step: Mapping[str, Any] | Any, graph_path: Path | None
) -> dict[str, Any]:
    summary: dict[str, Any] = {"saved": graph_path is not None}
    if graph_path is not None:
        summary["graph_path"] = str(graph_path)
    if isinstance(step, Mapping):
        for key in (
            "node_count",
            "edge_count",
            "feature_node_count",
            "max_edges",
            "max_feature_nodes",
            "active_feature_count",
        ):
            if key in step:
                summary[key] = step[key]
        nested = step.get("summary")
        if isinstance(nested, Mapping):
            for key in ("node_count", "edge_count", "feature_node_count"):
                if key in nested and key not in summary:
                    summary[key] = nested[key]
    return summary


def _model_load_knobs(specs: list[TraceSpec]) -> dict[str, Any]:
    resolved: dict[str, Any] | None = None
    for spec in specs:
        knobs = spec["graph_knobs"]
        if not isinstance(knobs, Mapping):
            raise ValueError("trace spec graph_knobs must be an object")
        config = transcoder_config_to_json(
            resolve_transcoder_load_config(knobs, preserve_default_values=True)
        )
        if config["decoder_chunk_size"] <= 0:
            raise ValueError("decoder_chunk_size must be a positive int")
        if config["cross_batch_decoder_cache_bytes"] < 0:
            raise ValueError(
                "cross_batch_decoder_cache_bytes must be a non-negative int"
            )
        active_row_residency = knobs.get("decoder_active_row_residency", False)
        if not isinstance(active_row_residency, bool):
            raise ValueError("decoder_active_row_residency must be a bool")
        active_row_max_bytes = knobs.get("decoder_active_row_max_bytes", 0)
        if (
            isinstance(active_row_max_bytes, bool)
            or not isinstance(active_row_max_bytes, int)
            or active_row_max_bytes < 0
        ):
            raise ValueError("decoder_active_row_max_bytes must be a non-negative int")
        if resolved is not None and any(
            config[k] != resolved[k] for k in PUBLIC_TRANSCODER_KNOB_KEYS
        ):
            raise ValueError(
                "all specs in a shard must share transcoder provider/load config"
            )
        resolved = config
    return resolved or transcoder_config_to_json(resolve_transcoder_load_config())


def _compact_result_to_bucketed_compact(*args: Any, **kwargs: Any) -> Any:
    """Load the canonical compact packager only on the SLURM execution path."""
    from nlp_research_project.exact_trace_bench.trace_runtime.compact_graph import (
        compact_result_to_bucketed_compact,
    )

    return compact_result_to_bucketed_compact(*args, **kwargs)


def _trace_request(
    *,
    model: Any,
    prompt_token_ids: list[int],
    spec: TraceSpec,
    prefix_metadata: Mapping[str, Any],
    full_sequence_mode: bool,
) -> Any:
    """Translate one harness spec into canonical, subsystem-owned policies."""
    from circuit_tracer import (
        AttributionProblem,
        DecoderCachePolicy,
        ExecutionConstraints,
        FrontierExpansionPlan,
        FrontierSemantics,
        ObservabilityPolicy,
        PrefixViewTarget,
        ReplayPlan,
        RowStoragePlan,
        SessionPlan,
        TraceEvidence,
        TraceRequest,
        TraceSemantics,
    )

    import torch

    knobs = spec["graph_knobs"]
    semantics = TraceSemantics(
        source_batch_size=int(knobs.get("attribution_batch_size", 256)),
        feature_batch_size=knobs.get("feature_batch_size"),
        logit_batch_size=knobs.get("logit_batch_size"),
        max_feature_nodes=int(knobs.get("max_feature_nodes", 8192)),
        diagnostic_feature_cap=knobs.get("diagnostic_feature_cap"),
        update_interval=int(knobs.get("attribution_update_interval", 4)),
        exact_trace_internal_dtype=str(
            knobs.get("exact_trace_internal_dtype", "fp32")
        ),
        phase0_activation_threshold_compare_mode=str(
            knobs.get("phase0_activation_threshold_compare_mode", "baseline")
        ),
        frontier=FrontierSemantics(
            scheduler=str(knobs.get("phase4_scheduler_mode", "locality")),
            refresh_policy=str(knobs.get("phase4_refresh_policy", "standard")),
            refresh_interval_multiplier=int(
                knobs.get("phase4_refresh_interval_multiplier", 1)
            ),
            ranker=str(knobs.get("phase4_ranker", "argsort")),
            phase3_buffer_relative_epsilon=knobs.get(
                "phase3_frontier_buffer_relative_epsilon"
            ),
            phase3_buffer_max_extra=int(
                knobs.get("phase3_frontier_buffer_max_extra", 0)
            ),
            phase4_buffer_relative_epsilon=knobs.get(
                "phase4_frontier_buffer_relative_epsilon"
            ),
            phase4_buffer_max_extra_per_refresh=int(
                knobs.get("phase4_frontier_buffer_max_extra_per_refresh", 0)
            ),
            phase4_buffer_max_extra_total=int(
                knobs.get("phase4_frontier_buffer_max_extra_total", 0)
            ),
        ),
    )
    decoder_cache_bytes = int(knobs.get("cross_batch_decoder_cache_bytes", 0))
    execution = ExecutionConstraints(
        session=SessionPlan(
            capacity=knobs.get("nnsight_session_capacity"),
            phase3_microbatch_max_rows=knobs.get(
                "phase3_compute_microbatch_max_rows"
            ),
            phase4_execution_batch_max_rows=(
                knobs["phase4_execution_batch_max_rows"]
                if knobs.get("phase4_execution_batch_max_rows") is not None
                else knobs.get("phase4_compute_microbatch_max_rows")
            ),
            phase1_trace_batch_policy=str(
                knobs.get("phase1_trace_batch_policy", "legacy")
            ),
            phase1_trace_batch_size_max=knobs.get("phase1_trace_batch_size_max"),
            decoder_cache=DecoderCachePolicy(
                enabled=decoder_cache_bytes > 0,
                max_bytes=decoder_cache_bytes or None,
            ),
        ),
        storage=RowStoragePlan(
            retention=str(knobs.get("feature_row_retention", "full_file")),
            full_retention_backend=str(
                knobs.get("full_retention_backend", "full_file")
            ),
            feature_column_tile_size=int(
                knobs.get("feature_row_column_tile_size", 2048)
            ),
            influence_row_tile_size=int(knobs.get("influence_row_tile_size", 4096)),
            influence_column_tile_size=int(
                knobs.get("influence_column_tile_size", 2048)
            ),
            cache_control=str(knobs.get("row_store_cache_control", "off")),
            temp_root_policy=str(knobs.get("row_store_temp_root_policy", "default")),
            temp_root=knobs.get("row_store_temp_root"),
            preallocate=bool(knobs.get("row_store_preallocate", True)),
            replay_tile_cache_bytes=knobs.get("replay_tile_cache_bytes"),
            exact_encoder_residency=str(
                knobs.get("exact_encoder_residency", "lazy")
            ),
        ),
        replay=ReplayPlan(
            feature_window=int(knobs.get("chunked_feature_replay_window") or 4),
            error_vector_prefetch_lookahead=int(
                knobs.get("error_vector_prefetch_lookahead") or 2
            ),
            stage_encoder_vecs_on_cpu=knobs.get("stage_encoder_vecs_on_cpu"),
            stage_error_vectors_on_cpu=knobs.get("stage_error_vectors_on_cpu"),
            decoder_contraction_tile=knobs.get("row_subchunk_size"),
            phase0_donor_bundle=knobs.get("phase0_donor_bundle"),
            phase0_mode=str(knobs.get("phase0_replay_mode", "disabled")),
            phase0_donor_context_policy=str(
                knobs.get("phase0_donor_context_policy", "strict")
            ),
            phase3_gradient_donor_bundle=knobs.get("phase3_gradient_donor_bundle"),
            phase3_gradient_mode=str(
                knobs.get("phase3_gradient_replay_mode", "disabled")
            ),
            phase3_row_donor_bundle=knobs.get("phase3_row_donor_bundle"),
            phase3_row_mode=str(knobs.get("phase3_row_replay_mode", "disabled")),
            phase3_validation_policy=str(
                knobs.get("phase3_replay_validation_policy", "strict")
            ),
        ),
        frontier=FrontierExpansionPlan(
            scheduler_debug=bool(knobs.get("phase4_scheduler_debug", False)),
            scheduler_telemetry_detail=str(
                knobs.get("phase4_scheduler_telemetry_detail", "normal")
            ),
            refresh_optimization=str(
                knobs.get("phase4_refresh_optimization", "v1")
            ),
            refresh_prepared_chunk_cache_bytes=int(
                knobs.get("phase4_refresh_prepared_chunk_cache_bytes", 0)
            ),
            refresh_active_row_accumulation=str(
                knobs.get("phase4_refresh_active_row_accumulation", "direct_v1")
            ),
            row_executor=str(knobs.get("phase4_row_executor", "batched")),
            row_reduction=str(knobs.get("phase4_row_reduction", "gpu_v1")),
            feature_batch_planning=bool(knobs.get("plan_feature_batch_size", False)),
            feature_batch_size_max=knobs.get("feature_batch_size_max"),
            feature_batch_target_reserved_fraction=float(
                knobs.get("feature_batch_target_reserved_fraction", 0.9)
            ),
            feature_batch_min_free_fraction=float(
                knobs.get("feature_batch_min_free_fraction", 0.05)
            ),
            feature_batch_probe_batches=int(
                knobs.get("feature_batch_probe_batches", 1)
            ),
            feature_vjp_tape_batch_window=int(
                knobs.get("feature_vjp_tape_batch_window", 1)
            ),
            feature_vjp_tape_max_bytes=int(
                knobs.get("feature_vjp_tape_max_bytes", 0)
            ),
            decoder_page_prefetch_depth=int(
                knobs.get("decoder_page_prefetch_depth", 0)
            ),
            decoder_active_row_residency=bool(
                knobs.get("decoder_active_row_residency", False)
            ),
            decoder_active_row_max_bytes=int(
                knobs.get("decoder_active_row_max_bytes", 0)
            ),
        ),
        observability=ObservabilityPolicy(
            verbose=bool(knobs.get("verbose_attribution", True)),
            profile=bool(knobs.get("profile_attribution", True)),
            profile_log_interval=int(knobs.get("profile_log_interval", 1)),
            telemetry_max_events=knobs.get("telemetry_max_events"),
            phase4_anomaly_debug=bool(knobs.get("phase4_anomaly_debug", False)),
            cross_cluster_debug=bool(knobs.get("cross_cluster_debug", False)),
            capture_phase0_donor_bundle=bool(
                knobs.get("capture_phase0_donor_bundle", False)
            ),
            capture_phase3_seed_bundle=bool(
                knobs.get("capture_phase3_seed_bundle", False)
            ),
            capture_phase3_gradient_bundle=bool(
                knobs.get("capture_phase3_gradient_bundle", False)
            ),
            capture_phase3_row_bundle=bool(
                knobs.get("capture_phase3_row_bundle", False)
            ),
            capture_feature_semantic_descriptors=bool(
                knobs.get("capture_feature_semantic_descriptors", False)
            ),
            semantic_descriptor_top_k=int(
                knobs.get("semantic_descriptor_top_k", 2048)
            ),
            semantic_descriptor_dim=int(knobs.get("semantic_descriptor_dim", 64)),
        ),
        offload="cpu",
        compact_output=not bool(knobs.get("save_raw_graph", False)),
    )
    prefix_evidence = {
        key: value
        for key, value in prefix_metadata.items()
        if key not in {"mode", "target_position"}
    }
    return TraceRequest(
        problem=AttributionProblem(
            model=model,
            prompt=torch.tensor(prompt_token_ids, dtype=torch.long),
            targets=torch.tensor([spec["target_token_id"]], dtype=torch.long),
            max_n_logits=1,
            desired_logit_prob=1.0,
            output_position=spec["target_position"] - 1
            if full_sequence_mode
            else None,
            prefix_view=PrefixViewTarget(
                mode=(
                    "full_sequence_target_position"
                    if full_sequence_mode
                    else "independent_prefix"
                ),
                target_position=int(spec["target_position"]),
            ),
        ),
        semantics=semantics,
        execution=execution,
        evidence=TraceEvidence(
            name="full_answer_runner",
            version="1",
            metadata={"prefix_view_metadata": prefix_evidence},
        ),
    )


def _npz_ready(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.numpy()
    if value is None or isinstance(value, (str, int, float, bool)):
        return np.asarray(value)
    if isinstance(value, (list, tuple)) and all(
        item is None or isinstance(item, (str, int, float, bool)) for item in value
    ):
        return np.asarray(value)
    return np.asarray(json.dumps(_json_ready(value), sort_keys=True))


def _json_ready(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "tensor": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _save_compact_debug_sidecars(
    token_dir: Path, compact_result: Mapping[str, Any]
) -> dict[str, str]:
    import numpy as np

    sidecar_keys = (
        "phase0_donor_bundle",
        "phase3_seed_bundle",
        "phase3_gradient_bundle",
        "phase3_row_bundle",
        "feature_semantic_descriptors",
        "cross_cluster_debug_summary",
        "cross_cluster_debug_checkpoints",
        "cross_cluster_debug_batches",
    )
    sidecars: dict[str, str] = {}
    for key in sidecar_keys:
        payload = compact_result.get(key)
        if payload is None:
            continue
        ensure_dir(token_dir)
        if key.startswith("cross_cluster_debug"):
            path = token_dir / f"{key}.json"
            write_json(path, _json_ready(payload))
        else:
            path = token_dir / f"{key}.npz"
            if not isinstance(payload, Mapping):
                raise TypeError(f"compact debug sidecar {key} must be a mapping")
            np.savez_compressed(
                path,
                **{
                    str(item_key): _npz_ready(item)
                    for item_key, item in payload.items()
                },
            )
        sidecars[key] = str(path)
    return sidecars


def _persist_compact_telemetry_events(
    *,
    token_dir: Path,
    compact_result: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    events = compact_result.get("telemetry_events")
    if not events:
        return {"telemetry_event_count": 0, "telemetry_events_path": None}
    if not isinstance(events, list):
        events = [events]

    path = token_dir / "telemetry.jsonl"
    context = {
        "run_id": trace.get("run_id"),
        "shard_id": trace.get("shard_id"),
        "trace_id": trace.get("trace_id"),
        "trajectory_id": trace.get("trajectory_id"),
        "generated_index": trace.get("generated_index"),
        "target_position": trace.get("target_position"),
        "target_token_id": trace.get("target_token_id"),
        "target_token_text": trace.get("target_token_text"),
    }
    rows = []
    for index, event in enumerate(events):
        rows.append(
            {
                **context,
                "telemetry_event_index": index,
                "event": _json_ready(event),
            }
        )
    write_jsonl(path, rows)
    return {"telemetry_event_count": len(rows), "telemetry_events_path": str(path)}


_TELEMETRY_EXCEPTION_SUMMARY_ATTR = "circuit_tracer_telemetry_summary"
_TELEMETRY_EXCEPTION_EVENTS_ATTR = "circuit_tracer_telemetry_events"


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    pending: list[BaseException | None] = [exc]
    seen: set[int] = set()
    chain: list[BaseException] = []
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        pending.extend(
            [
                getattr(current, "original", None),
                current.__cause__,
                current.__context__,
            ]
        )
    return chain


def _exception_telemetry_payload(exc: BaseException) -> dict[str, Any] | None:
    for candidate in _iter_exception_chain(exc):
        summary = getattr(candidate, _TELEMETRY_EXCEPTION_SUMMARY_ATTR, None)
        events = getattr(candidate, _TELEMETRY_EXCEPTION_EVENTS_ATTR, None)
        if summary is None and events is None:
            continue
        if events is None:
            events = []
        elif not isinstance(events, list):
            events = [events]
        return {
            "telemetry_summary": _json_ready(summary),
            "telemetry_events": events,
            "telemetry_exception_type": type(candidate).__name__,
        }
    return None


def run_real_shard(
    *,
    trajectory_path: Path,
    trace_specs_path: Path,
    shards_path: Path,
    shard_id: int,
    output_root: Path,
    run_id: str | None = None,
    run_name: str | None = None,
    run_description: str | None = None,
    run_goal: str | None = None,
) -> dict[str, Any]:
    """SLURM-only real tracing path; imports model stack only inside this function."""
    trajectory, specs, shard = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        shard_id=shard_id,
    )
    metadata = _runtime_metadata(
        run_id=run_id,
        run_name=run_name,
        run_description=run_description,
        run_goal=run_goal,
    )
    if (
        os.environ.get("SLURM_JOB_ID") is None
        and os.environ.get("EXACT_TRACE_ALLOW_LOCAL_GPU") != "1"
    ):
        raise RuntimeError(
            "run-full-answer-shard requires SLURM_JOB_ID; set EXACT_TRACE_ALLOW_LOCAL_GPU=1 "
            "to override for local GPU debugging before importing model code"
        )
    root = _shard_dir(output_root, shard_id)
    ensure_dir(root)
    trace_results_path = root / "trace_results.jsonl"
    trace_results_path.write_text("", encoding="utf-8")
    write_json(
        root / "shard.json",
        _shard_record(
            status="running",
            shard=shard,
            trajectory_id=trajectory["trajectory_id"],
            trace_specs_path=trace_specs_path,
            shards_path=shards_path,
            token_count=len(specs),
            metadata=metadata,
            target_positions=[spec["target_position"] for spec in specs],
        ),
    )

    from nlp_research_project.exact_trace_bench import compact_io as circuit_utils
    from nlp_research_project.exact_trace_bench.trace_runtime.provider import (
        get_model_transcoder_metadata,
        load_model,
    )
    from circuit_tracer import SessionWindow, open_session, trace_one
    model_load_knobs = _model_load_knobs(specs)
    model = load_model(
        exact_chunked_decoder=True,
        **model_load_knobs,
    )
    transcoder_metadata = get_model_transcoder_metadata(model) or {
        "requested": model_load_knobs
    }
    metadata = {**metadata, "transcoder": transcoder_metadata}
    rows: list[dict[str, Any]] = []
    full_sequence_cache = prepare_full_sequence_cache(trajectory, specs)
    for window in _shard_windows(specs, shard):
        window_specs = cast(list[TraceSpec], window["specs"])
        _validate_window_session_specs(window_specs)
        window_session = None
        if any(
            spec.get("graph_knobs", {}).get("trajectory_session_mode")
            == "window_reuse_v1"
            for spec in window_specs
        ):
            if full_sequence_cache is None:
                raise ValueError("window_reuse_v1 requires full_sequence cache")
            window_session_knobs = next(
                spec["graph_knobs"]
                for spec in window_specs
                if spec.get("graph_knobs", {}).get("trajectory_session_mode")
                == "window_reuse_v1"
            )
            if bool(window_session_knobs.get("reuse_phase0_window_state", False)) != bool(
                window_session_knobs.get("reuse_target_logits", False)
            ):
                raise ValueError(
                    "canonical window sessions require phase-0 and target-logit reuse "
                    "to be enabled or disabled together"
                )
        try:
            for spec in window_specs:
                token_dir = root / f"token_{spec['generated_index']:06d}"
                graph_path = token_dir / "graph.npz"
                trace = _trace_runtime_payload(
                    spec=spec, shard_id=shard_id, metadata=metadata
                )
                trace["status"] = "running"
                trace["forced_target"] = forced_target_payload(spec)
                trace.update(_target_token_metadata(trajectory, spec))
                trace["window_session"] = {
                    key: value for key, value in window.items() if key != "specs"
                }
                started = time.perf_counter()
                try:
                    prefix_token_ids = reconstruct_prefix_token_ids(trajectory, spec)
                    prefix_metadata = prefix_view_metadata(
                        trajectory,
                        spec,
                        prefix_token_ids,
                        full_sequence_token_ids=full_sequence_cache["token_ids"]
                        if full_sequence_cache
                        else None,
                        full_sequence_token_ids_sha256=full_sequence_cache[
                            "token_ids_sha256"
                        ]
                        if full_sequence_cache
                        else None,
                    )
                    trace["prefix_view_metadata"] = prefix_metadata
                    mode = spec.get("graph_knobs", {}).get(
                        "trajectory_session_mode", "per_token"
                    )
                    trace["trajectory_session"] = {
                        **session_reuse_metadata(spec),
                        "session_reuse_fallback": None
                        if mode == "window_reuse_v1"
                        else session_reuse_metadata(spec)["session_reuse_fallback"],
                        "decoder_cache_reuse_effective": False,
                    }
                    knobs = spec["graph_knobs"]
                    save_raw_graph = bool(knobs.get("save_raw_graph", False))
                    raw_graph_path = token_dir / "graph.pt"
                    trace["graph_path"] = str(
                        raw_graph_path if save_raw_graph else graph_path
                    )
                    debug_sidecars: dict[str, str] = {}
                    full_sequence_mode = (
                        knobs.get("input_context_mode") == "full_sequence"
                    )
                    if full_sequence_mode and full_sequence_cache is None:
                        raise ValueError(
                            "full_sequence cache missing for full_sequence trace spec"
                        )
                    if full_sequence_mode:
                        assert full_sequence_cache is not None
                        prompt_token_ids = full_sequence_cache["token_ids"]
                    else:
                        prompt_token_ids = prefix_token_ids
                    request = _trace_request(
                        model=model,
                        prompt_token_ids=prompt_token_ids,
                        spec=spec,
                        prefix_metadata=prefix_metadata,
                        full_sequence_mode=full_sequence_mode,
                    )
                    if mode == "window_reuse_v1":
                        if window_session is None:
                            window_session = open_session(
                                request,
                                window=SessionWindow(
                                    max_prefix_len=int(
                                        window["window_max_target_position"]
                                    )
                                ),
                            )
                        trace_result = window_session.trace_window(
                            int(spec["target_position"]),
                            reuse=bool(
                                knobs.get("reuse_phase0_window_state", False)
                            ),
                            request=request,
                        )
                    elif request.execution.session.decoder_cache.enabled:
                        if window_session is None:
                            window_session = open_session(request)
                        trace_result = window_session.trace(request)
                    else:
                        trace_result = trace_one(request)
                    trace["trajectory_session"][
                        "decoder_cache_reuse_effective"
                    ] = bool(
                        request.execution.session.decoder_cache.enabled
                        and window_session is not None
                    )
                    graph_result = trace_result.output
                    if trace_result.telemetry_summary:
                        trace["telemetry_summary"] = _json_ready(
                            trace_result.telemetry_summary
                        )
                    if save_raw_graph:
                        raw_graph_path.parent.mkdir(parents=True, exist_ok=True)
                        graph_result.to_pt(str(raw_graph_path))
                        graph_summary: dict[str, Any] = {
                            "saved": True,
                            "graph_path": str(raw_graph_path),
                            "format": "raw_graph_pt",
                            "file_bytes": raw_graph_path.stat().st_size,
                        }
                        adjacency = getattr(graph_result, "adjacency_matrix", None)
                        if adjacency is not None:
                            graph_summary["adjacency_shape"] = list(adjacency.shape)
                            graph_summary["adjacency_dtype"] = str(adjacency.dtype)
                        active_features = getattr(graph_result, "active_features", None)
                        if active_features is not None:
                            graph_summary["active_feature_count"] = int(
                                active_features.shape[0]
                            )
                        selected_features = getattr(
                            graph_result, "selected_features", None
                        )
                        if selected_features is not None:
                            graph_summary["selected_feature_count"] = int(
                                selected_features.numel()
                            )
                    else:
                        compact_result = graph_result
                        debug_sidecars = _save_compact_debug_sidecars(
                            token_dir, compact_result
                        )
                        bucketed = _compact_result_to_bucketed_compact(
                            compact_result,
                            spec["generated_index"],
                            token_text=spec["target_token_text"],
                            max_edges=int(knobs.get("max_edges", 20000)),
                        )
                        circuit_utils.save_bucketed_compact(bucketed, graph_path)
                        step = bucketed.step
                        graph_summary = _graph_summary(step, graph_path)
                        graph_summary["format"] = "typed_bucketed"
                        trace.update(
                            _persist_compact_telemetry_events(
                                token_dir=token_dir,
                                compact_result=compact_result,
                                trace=trace,
                            )
                        )
                    trace.update(
                        {
                            "status": "ok",
                            "error": None,
                            "timings": {
                                "trace_seconds": time.perf_counter() - started,
                            },
                            "graph_summary": graph_summary,
                        }
                    )
                    if debug_sidecars:
                        trace["debug_sidecars"] = debug_sidecars
                    if not save_raw_graph:
                        for key in (
                            "phase0_window_state_reuse_requested",
                            "phase0_window_state_reuse_effective",
                            "target_logit_source",
                        ):
                            if key in compact_result:
                                trace["trajectory_session"][key] = _json_ready(
                                    compact_result.get(key)
                                )
                        trace["trajectory_session"][
                            "reuse_phase0_window_state_effective"
                        ] = bool(
                            compact_result.get(
                                "phase0_window_state_reuse_effective", False
                            )
                        )
                        trace["trajectory_session"]["reuse_target_logits_effective"] = (
                            compact_result.get("target_logit_source")
                            == "full_sequence_window_logits"
                        )
                        trace["trajectory_session"]["target_logit_source"] = (
                            _normalize_target_logit_source(
                                compact_result.get("target_logit_source")
                            )
                        )
                        trace["trajectory_session"]["session_reuse_effective"] = bool(
                            trace["trajectory_session"].get(
                                "decoder_cache_reuse_effective", False
                            )
                            or trace["trajectory_session"].get(
                                "reuse_phase0_window_state_effective", False
                            )
                            or trace["trajectory_session"].get(
                                "reuse_target_logits_effective", False
                            )
                        )
                    if (
                        not save_raw_graph
                    ) and "phase3_frontier_buffer_metadata" in compact_result:
                        trace["phase3_frontier_buffer_metadata"] = _json_ready(
                            compact_result.get("phase3_frontier_buffer_metadata")
                        )
                    if "phase4_frontier_buffer_metadata" in compact_result:
                        trace["phase4_frontier_buffer_metadata"] = _json_ready(
                            compact_result.get("phase4_frontier_buffer_metadata")
                        )
                except (
                    Exception
                ) as exc:  # pragma: no cover - exercised only in SLURM real mode
                    trace.update(_exception_payload(exc))
                    exception_telemetry = _exception_telemetry_payload(exc)
                    if exception_telemetry is not None:
                        trace["telemetry_summary"] = exception_telemetry[
                            "telemetry_summary"
                        ]
                        trace["telemetry_exception_type"] = exception_telemetry[
                            "telemetry_exception_type"
                        ]
                        trace.update(
                            _persist_compact_telemetry_events(
                                token_dir=token_dir,
                                compact_result=exception_telemetry,
                                trace=trace,
                            )
                        )
                    trace.update(
                        {
                            "status": "error",
                            "error": repr(exc),
                            "error_traceback": traceback.format_exc(),
                            "timings": {
                                "trace_seconds": time.perf_counter() - started,
                            },
                        }
                    )
                write_json(token_dir / "trace.json", trace)
                rows.append(trace)
                with trace_results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(trace, sort_keys=True) + "\n")
        finally:
            if window_session is not None:
                window_session.close()
    final_status = "complete" if all(r["status"] == "ok" for r in rows) else "error"
    token_seconds = [
        float(r.get("timings", {}).get("trace_seconds", 0.0)) for r in rows
    ]
    health = {
        "actual_total_seconds": sum(token_seconds),
        "max_token_seconds": max(token_seconds) if token_seconds else 0.0,
        "failed_token_count": sum(1 for r in rows if r.get("status") != "ok"),
        "predicted_cost_sum": int(
            shard.get("estimated_cost_sum", sum(s["estimated_cost"] for s in specs))
        ),
        "retry_recommended": any(r.get("status") != "ok" for r in rows),
    }
    write_json(
        root / "shard.json",
        _shard_record(
            status=final_status,
            shard=shard,
            trajectory_id=trajectory["trajectory_id"],
            trace_specs_path=trace_specs_path,
            shards_path=shards_path,
            token_count=len(specs),
            metadata={**metadata, "shard_health": health},
            target_positions=[spec["target_position"] for spec in specs],
        ),
    )
    return {"shard_dir": str(root), "token_count": len(rows), "status": final_status}


def _summary_row(spec: TraceSpec, *, shard_id: int) -> dict[str, Any]:
    return {
        "shard_id": shard_id,
        "trace_id": spec["trace_id"],
        "trajectory_id": spec["trajectory_id"],
        "generated_index": spec["generated_index"],
        "target_position": spec["target_position"],
        "prefix_token_count": spec["prefix_token_count"],
        "target_token_id": spec["target_token_id"],
        "target_token_text": spec["target_token_text"],
        "target_mode": spec["target_mode"],
        "estimated_cost": spec["estimated_cost"],
        "selection_reasons": spec["selection_reasons"],
    }


def _exception_payload(exc: Exception) -> dict[str, Any]:
    """Return diagnostic-safe exception metadata for per-token trace failures.

    nnsight wraps inner exceptions in dynamically generated ``NNsightException``
    classes whose ``repr`` is often just ``NNsightException()``.  ``str(exc)``
    and ``print_exception`` carry the reconstructed trace, so preserve those in
    the token trace JSON instead of only storing ``repr(exc)``.
    """

    payload: dict[str, Any] = {
        "error": repr(exc),
        "error_type": type(exc).__name__,
        "error_module": type(exc).__module__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    original = getattr(exc, "original", None)
    if original is not None:
        payload.update(
            {
                "original_error_type": type(original).__name__,
                "original_error_module": type(original).__module__,
                "original_error": repr(original),
                "original_error_message": str(original),
            }
        )
    cause = exc.__cause__
    if cause is not None:
        payload["cause_error"] = repr(cause)
        payload["cause_error_type"] = type(cause).__name__
    context = exc.__context__
    if context is not None:
        payload["context_error"] = repr(context)
        payload["context_error_type"] = type(context).__name__
    return payload


def _trace_payload(spec: TraceSpec, *, shard_id: int) -> dict[str, Any]:
    forced_target = forced_target_payload(spec)
    payload = _summary_row(spec, shard_id=shard_id)
    payload.update(
        {
            "schema_version": 1,
            "status": "dry_run",
            "graph_path": None,
            "graph_knobs": spec["graph_knobs"],
            "forced_target": forced_target,
            "error": None,
        }
    )
    return payload
